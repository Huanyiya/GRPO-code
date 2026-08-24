"""Fail-fast binary code reward for Eurus standard-input/output tasks."""

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from slime_plugins.rewards import code_compare, code_sandbox

__all__ = ["reward_func"]

_MAX_TEST_CASES = 15


_PYTHON_FENCE_RE = re.compile(
    r"```[ \t]*(?:python|py)[ \t]*\r?\n(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_GENERIC_FENCE_RE = re.compile(
    r"```[^\r\n]*\r?\n(?P<code>.*?)```",
    flags=re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class CodeTestCase:
    input_value: Any
    expected_output: Any
    input_length: int
    original_index: int


@dataclass(frozen=True, slots=True)
class ParsedTestCases:
    """Validated Eurus testcases."""

    test_cases: list[CodeTestCase]
    fn_name: str | None


def _last_fenced_code(pattern: re.Pattern[str], response: str) -> str | None:
    matches = list(pattern.finditer(response))
    if not matches:
        return None
    code = matches[-1].group("code").strip()
    return code or None


def _extract_python_code(response: Any) -> str | None:
    """Extract Python using the last Python fence, generic fence, or full text."""
    if not isinstance(response, str) or not response.strip():
        return None

    code = _last_fenced_code(_PYTHON_FENCE_RE, response)
    if code is not None:
        return code

    code = _last_fenced_code(_GENERIC_FENCE_RE, response)
    if code is not None:
        return code

    code = response.strip()
    return code or None


def _has_valid_python_syntax(code: str) -> bool:
    """Compile only to an AST/code object; never execute generated code."""
    try:
        compile(code, "<model-response>", "exec")
    except (SyntaxError, ValueError, TypeError, OverflowError):
        return False
    return True


def _string_or_lines(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(line, str) for line in value):
        return "\n".join(value)
    return None


def _parse_test_cases(label: Any) -> ParsedTestCases | None:
    """Parse Eurus standard-I/O or call-based testcases."""
    if not isinstance(label, Mapping):
        return None

    ground_truth = label.get("ground_truth")
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        return None

    try:
        payload = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    if not isinstance(payload, Mapping):
        return None

    fn_name = payload.get("fn_name")
    if fn_name is not None and (not isinstance(fn_name, str) or not fn_name.strip()):
        return None
    fn_name = fn_name.strip() if isinstance(fn_name, str) else None

    inputs = payload.get("inputs")
    outputs = payload.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None
    if not inputs or len(inputs) != len(outputs):
        return None

    test_cases = []
    for index, (input_value, output_value) in enumerate(zip(inputs, outputs, strict=True)):
        if fn_name is None:
            input_value = _string_or_lines(input_value)
            output_value = _string_or_lines(output_value)
            if input_value is None or output_value is None:
                return None
            input_for_length = input_value
        else:
            if isinstance(input_value, list):
                input_for_length = json.dumps(input_value, ensure_ascii=False, sort_keys=True)
            elif isinstance(input_value, str):
                input_for_length = input_value
            else:
                return None
        test_cases.append(
            CodeTestCase(
                input_value=input_value,
                expected_output=output_value,
                input_length=len(input_for_length),
                original_index=index,
            )
        )

    return ParsedTestCases(test_cases=test_cases, fn_name=fn_name)


def _select_test_cases(test_cases: list[CodeTestCase]) -> list[CodeTestCase]:
    """Deterministically select the 15 longest stdin strings."""
    if len(test_cases) <= _MAX_TEST_CASES:
        return list(test_cases)
    return sorted(test_cases, key=lambda case: (-case.input_length, case.original_index))[:_MAX_TEST_CASES]


def _call_args(input_value: Any) -> list[Any] | None:
    if isinstance(input_value, list):
        return input_value
    if not isinstance(input_value, str):
        return None
    if not input_value.strip():
        return []
    try:
        return [json.loads(line) for line in input_value.splitlines()]
    except json.JSONDecodeError:
        return None


def _build_call_wrapper(model_code: str, fn_name: str, args: list[Any]) -> str:
    """Build the executable used for one call-based testcase."""
    args_json_literal = repr(json.dumps(args, ensure_ascii=False))
    fn_name_literal = repr(fn_name)
    return f"""{model_code}

import json as _slime_json

_SLIME_FN_NAME = {fn_name_literal}
_slime_args = _slime_json.loads({args_json_literal})

if _SLIME_FN_NAME in globals() and callable(globals()[_SLIME_FN_NAME]):
    _slime_target = globals()[_SLIME_FN_NAME]
elif "Solution" in globals():
    _slime_instance = globals()["Solution"]()
    _slime_target = getattr(_slime_instance, _SLIME_FN_NAME)
else:
    raise AttributeError(f"Function or Solution method {{_SLIME_FN_NAME!r}} not found")

_slime_result = _slime_target(*_slime_args)
print(_slime_json.dumps(_slime_result, ensure_ascii=False))
"""


async def reward_func(args, sample, **kwargs) -> float:
    """Return one only when every selected standard-I/O testcase passes."""
    del args, kwargs

    try:
        code = _extract_python_code(getattr(sample, "response", None))
        if code is None or not _has_valid_python_syntax(code):
            return 0.0

        parsed = _parse_test_cases(getattr(sample, "label", None))
        if parsed is None:
            return 0.0

        for test_case in _select_test_cases(parsed.test_cases):
            if parsed.fn_name is None:
                executable = code
                stdin = test_case.input_value
            else:
                args = _call_args(test_case.input_value)
                if args is None:
                    return 0.0
                executable = _build_call_wrapper(code, parsed.fn_name, args)
                stdin = ""

            result = await asyncio.to_thread(code_sandbox.run_code, executable, stdin)
            if not result.success or result.status != "ok":
                return 0.0
            if parsed.fn_name is None:
                output_matches = code_compare.compare_standard_output(result.stdout, test_case.expected_output)
            else:
                output_matches = code_compare.compare_call_output(result.stdout, test_case.expected_output)
            if not output_matches:
                return 0.0
    except Exception:
        # Bad data, malformed service responses, and client failures must not
        # escape into the rollout loop.
        return 0.0

    return 1.0
