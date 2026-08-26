"""Output comparison helpers for standard-I/O and call-based code tasks."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any

__all__ = ["compare_call_output", "compare_standard_output", "normalize_standard_output"]

def normalize_standard_output(output: str) -> str:
    """Normalize output whitespace for standard programming-contest judging.

    Token-based judges treat runs of spaces, tabs, and newlines as separators.
    Do the same here so harmless formatting differences (for example multiple
    spaces between ``YES``/``NO`` tokens) do not turn a correct program into a
    zero-reward sample.
    """
    return " ".join(output.split())


def _decimal_tokens(output: str) -> list[Decimal] | None:
    tokens = output.split()
    if not tokens:
        return None
    try:
        values = [Decimal(token) for token in tokens]
    except InvalidOperation:
        return None
    if not all(value.is_finite() for value in values):
        return None
    return values


def _decimal_tolerance(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        tolerance = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return tolerance if tolerance.is_finite() and tolerance >= 0 else None


def _numbers_close(
    actual: int | float | Decimal,
    expected: int | float | Decimal,
    *,
    abs_tol: Any = None,
    rel_tol: Any = None,
) -> bool:
    try:
        actual_decimal = Decimal(str(actual))
        expected_decimal = Decimal(str(expected))
    except InvalidOperation:
        return False
    if not actual_decimal.is_finite() or not expected_decimal.is_finite():
        return False
    absolute_tolerance = _decimal_tolerance(abs_tol) or Decimal(0)
    relative_tolerance = _decimal_tolerance(rel_tol) or Decimal(0)
    tolerance = max(absolute_tolerance, relative_tolerance * max(abs(actual_decimal), abs(expected_decimal)))
    return abs(actual_decimal - expected_decimal) <= tolerance


def compare_standard_output(actual: str, expected: str, *, abs_tol: Any = None, rel_tol: Any = None) -> bool:
    """Compare standard-I/O output using tolerance only when explicitly supplied.

    Without a tolerance, this is a strict token comparison: whitespace is
    normalized, but ``1`` and ``1.0`` (or any different integer/string token)
    are not interchangeable.
    """
    actual = normalize_standard_output(actual)
    expected = normalize_standard_output(expected)
    if actual == expected:
        return True

    if abs_tol is None and rel_tol is None:
        return False

    actual_numbers = _decimal_tokens(actual)
    expected_numbers = _decimal_tokens(expected)
    if actual_numbers is None or expected_numbers is None or len(actual_numbers) != len(expected_numbers):
        return False
    return all(
        _numbers_close(a, b, abs_tol=abs_tol, rel_tol=rel_tol)
        for a, b in zip(actual_numbers, expected_numbers, strict=True)
    )


def _parse_expected(expected: Any) -> Any:
    if not isinstance(expected, str):
        return expected
    try:
        return json.loads(expected)
    except (json.JSONDecodeError, TypeError):
        # Eurus also contains plain string ground truths such as ``A`` rather
        # than JSON string literals such as ``"A"``.
        return expected


def _compare_objects(actual: Any, expected: Any, *, abs_tol: Any = None, rel_tol: Any = None) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual is expected

    if isinstance(actual, (int, float, Decimal)) and isinstance(expected, (int, float, Decimal)):
        if abs_tol is None and rel_tol is None:
            return type(actual) is type(expected) and actual == expected
        return _numbers_close(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol)

    if isinstance(actual, str) or isinstance(expected, str):
        return isinstance(actual, str) and isinstance(expected, str) and actual == expected

    if actual is None or expected is None:
        return actual is None and expected is None

    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(
            _compare_objects(a, e, abs_tol=abs_tol, rel_tol=rel_tol)
            for a, e in zip(actual, expected, strict=True)
        )

    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return False
        return all(_compare_objects(actual[key], expected[key], abs_tol=abs_tol, rel_tol=rel_tol) for key in actual)

    return False


def compare_call_output(stdout: str, expected: Any, *, abs_tol: Any = None, rel_tol: Any = None) -> bool:
    """Parse sandbox JSON stdout and compare it, tolerating floats only if requested."""
    try:
        actual = json.loads(stdout.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False

    expected = _parse_expected(expected)

    # Normal comparison.
    if _compare_objects(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol):
        return True

    # PRIME / Eurus / TACO compatibility: some call-based ground truths wrap
    # the actual return value in one extra singleton list.  Do this only as a
    # fallback: a genuine list return such as [1, 2, 3] must still be compared
    # against [1, 2, 3] without unwrapping it.
    if isinstance(expected, (list, tuple)) and len(expected) == 1:
        if _compare_objects(actual, expected[0], abs_tol=abs_tol, rel_tol=rel_tol):
            return True

    return False
