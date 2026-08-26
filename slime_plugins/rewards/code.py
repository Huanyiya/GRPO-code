"""Fail-fast binary code reward for Eurus standard-input/output tasks."""

import asyncio
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
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

_TOLERANCE_VALUE_RE = (
    r"(?:\$?\s*10\s*(?:\^|\*\*)?\s*\{?\s*-\s*\d+\s*\}?\s*\$?"
    r"|\$?\s*\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*\$?)"
)
_TOLERANCE_LIMIT_RE = (
    r"(?:at\s+most|no\s+more\s+than|not\s+more\s+than|"
    r"less\s+than(?:\s+or\s+equal\s+to)?|"
    r"(?:does|do|will|should|must)?\s*not\s+exceed|"
    r"not\s+(?:be\s+)?(?:higher|greater|larger)\s+than|within)"
)
_BOTH_TOLERANCES_RE = re.compile(
    rf"\b(?:absolute\s+(?:or|and)\s+relative|relative\s+(?:or|and)\s+absolute)\s+error\b"
    rf".{{0,100}}?\b{_TOLERANCE_LIMIT_RE}\s*"
    rf"(?P<value>{_TOLERANCE_VALUE_RE})",
    flags=re.IGNORECASE,
)
_ONE_TOLERANCE_RE = re.compile(
    rf"\b(?P<kind>absolute|relative)\s+error\b"
    rf".{{0,100}}?\b{_TOLERANCE_LIMIT_RE}\s*"
    rf"(?P<value>{_TOLERANCE_VALUE_RE})",
    flags=re.IGNORECASE,
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


@dataclass(frozen=True, slots=True)
class FloatTolerance:
    """Tolerance explicitly stated by a problem statement, if any."""

    abs_tol: Decimal | None = None
    rel_tol: Decimal | None = None


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


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(item.get("content", "") for item in prompt if isinstance(item, Mapping))
    return ""


def _parse_tolerance_value(value: str) -> Decimal | None:
    normalized = value.replace("$", "").replace("−", "-").replace("{", "").replace("}", "").replace("^", "**")
    normalized = re.sub(r"\s+", "", normalized)
    power_match = re.fullmatch(r"10(?:\*\*)?-(\d+)", normalized)
    try:
        parsed = Decimal(10) ** -int(power_match.group(1)) if power_match else Decimal(normalized)
    except (InvalidOperation, ValueError, OverflowError):
        return None
    return parsed if parsed.is_finite() and parsed >= 0 else None


def _extract_float_tolerance(prompt: Any) -> FloatTolerance:
    """Read only explicitly stated absolute/relative error limits from the prompt."""
    text = _prompt_to_text(prompt)
    both_match = _BOTH_TOLERANCES_RE.search(text)
    if both_match is not None:
        value = _parse_tolerance_value(both_match.group("value"))
        if value is not None:
            return FloatTolerance(abs_tol=value, rel_tol=value)

    absolute_tolerance = None
    relative_tolerance = None
    for match in _ONE_TOLERANCE_RE.finditer(text):
        value = _parse_tolerance_value(match.group("value"))
        if value is None:
            continue
        if match.group("kind").lower() == "absolute":
            absolute_tolerance = value
        else:
            relative_tolerance = value
    return FloatTolerance(abs_tol=absolute_tolerance, rel_tol=relative_tolerance)


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


def _save_outputs_enabled() -> bool:
    """Whether to attach a serializable reward trace to the sample."""
    return os.environ.get("SAVE_OUTPUTS", "").strip().lower() == "true"


def _finish_reward(sample: Any, trace: dict[str, Any] | None, reward: float, failure_reason: str | None) -> float:
    """Store optional diagnostics without changing the reward contract."""
    if trace is not None:
        trace["final_reward"] = reward
        trace["failure_reason"] = failure_reason
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(sample, "metadata", metadata)
        metadata["code_reward_trace"] = trace
    return reward


def _sandbox_result_to_dict(result: code_sandbox.ExecutionResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "status": result.status,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.return_code,
    }


async def reward_func(args, sample, **kwargs) -> float:
    """Return one only when every selected standard-I/O testcase passes."""
    del args, kwargs

    trace: dict[str, Any] | None = {"generated_code": None, "test_cases": []} if _save_outputs_enabled() else None

    try:
        code = _extract_python_code(getattr(sample, "response", None))
        if trace is not None:
            trace["generated_code"] = code
        if code is None:
            return _finish_reward(sample, trace, 0.0, "no_python_code")
        if not _has_valid_python_syntax(code):
            return _finish_reward(sample, trace, 0.0, "invalid_python_syntax")

        parsed = _parse_test_cases(getattr(sample, "label", None))
        if parsed is None:
            return _finish_reward(sample, trace, 0.0, "invalid_test_cases")
        float_tolerance = _extract_float_tolerance(getattr(sample, "prompt", None))

        for test_case in _select_test_cases(parsed.test_cases):
            if parsed.fn_name is None:
                executable = code
                stdin = test_case.input_value
            else:
                call_args = _call_args(test_case.input_value)
                if call_args is None:
                    return _finish_reward(sample, trace, 0.0, "invalid_call_arguments")
                executable = _build_call_wrapper(code, parsed.fn_name, call_args)
                stdin = ""

            result = await asyncio.to_thread(code_sandbox.run_code, executable, stdin)
            testcase_trace: dict[str, Any] | None = None
            if trace is not None:
                testcase_trace = {
                    "original_index": test_case.original_index,
                    "input": test_case.input_value,
                    "expected_output": test_case.expected_output,
                    "sandbox_result": _sandbox_result_to_dict(result),
                    "output_matches": None,
                }
                trace["test_cases"].append(testcase_trace)
            if not result.success or result.status != "ok":
                return _finish_reward(sample, trace, 0.0, f"sandbox_{result.status}")
            if parsed.fn_name is None:
                output_matches = code_compare.compare_standard_output(
                    result.stdout,
                    test_case.expected_output,
                    abs_tol=float_tolerance.abs_tol,
                    rel_tol=float_tolerance.rel_tol,
                )
            else:
                output_matches = code_compare.compare_call_output(
                    result.stdout,
                    test_case.expected_output,
                    abs_tol=float_tolerance.abs_tol,
                    rel_tol=float_tolerance.rel_tol,
                )
            if testcase_trace is not None:
                testcase_trace["output_matches"] = output_matches
            if not output_matches:
                return _finish_reward(sample, trace, 0.0, "output_mismatch")
    except Exception as exc:
        # Bad data, malformed service responses, and client failures must not
        # escape into the rollout loop.
        if trace is not None:
            trace["exception"] = f"{type(exc).__name__}: {exc}"
        return _finish_reward(sample, trace, 0.0, "reward_exception")

    return _finish_reward(sample, trace, 1.0, None)
