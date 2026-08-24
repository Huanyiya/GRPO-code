"""Output comparison helpers for standard-I/O and call-based code tasks."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any

__all__ = ["compare_call_output", "compare_standard_output", "normalize_standard_output"]

_REL_TOL = Decimal("1e-6")
_ABS_TOL = Decimal("1e-8")


def normalize_standard_output(output: str) -> str:
    output = output.replace("\r\n", "\n")
    output = "\n".join(line.rstrip(" \t") for line in output.split("\n"))
    return output.rstrip("\n")


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


def _numbers_close(actual: int | float | Decimal, expected: int | float | Decimal) -> bool:
    try:
        actual_decimal = Decimal(str(actual))
        expected_decimal = Decimal(str(expected))
    except InvalidOperation:
        return False
    if not actual_decimal.is_finite() or not expected_decimal.is_finite():
        return False
    tolerance = max(_ABS_TOL, _REL_TOL * max(abs(actual_decimal), abs(expected_decimal)))
    return abs(actual_decimal - expected_decimal) <= tolerance


def compare_standard_output(actual: str, expected: str) -> bool:
    actual = normalize_standard_output(actual)
    expected = normalize_standard_output(expected)
    if actual == expected:
        return True

    actual_numbers = _decimal_tokens(actual)
    expected_numbers = _decimal_tokens(expected)
    if actual_numbers is None or expected_numbers is None or len(actual_numbers) != len(expected_numbers):
        return False
    return all(_numbers_close(a, b) for a, b in zip(actual_numbers, expected_numbers, strict=True))


def _parse_expected(expected: Any) -> Any:
    if not isinstance(expected, str):
        return expected
    try:
        return json.loads(expected)
    except (json.JSONDecodeError, TypeError):
        # Eurus also contains plain string ground truths such as ``A`` rather
        # than JSON string literals such as ``"A"``.
        return expected


def _compare_objects(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual is expected

    if isinstance(actual, (int, float, Decimal)) and isinstance(expected, (int, float, Decimal)):
        return _numbers_close(actual, expected)

    if isinstance(actual, str) or isinstance(expected, str):
        return isinstance(actual, str) and isinstance(expected, str) and actual == expected

    if actual is None or expected is None:
        return actual is None and expected is None

    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(_compare_objects(a, e) for a, e in zip(actual, expected, strict=True))

    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return False
        return all(_compare_objects(actual[key], expected[key]) for key in actual)

    return False


def compare_call_output(stdout: str, expected: Any) -> bool:
    """Parse sandbox JSON stdout and recursively compare it with ground truth."""
    try:
        actual = json.loads(stdout.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False
    return _compare_objects(actual, _parse_expected(expected))
