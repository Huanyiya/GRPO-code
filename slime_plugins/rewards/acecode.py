"""Sandboxed binary reward for AceCode function-implementation tasks."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any

from slime_plugins.rewards import code as eurus_code
from slime_plugins.rewards import code_sandbox

__all__ = ["reward_func"]

def _parse_test_cases(label: Any) -> list[str] | None:
    if not isinstance(label, Mapping):
        return None
    test_cases = label.get("test_cases")
    if not (
        isinstance(test_cases, list)
        and test_cases
        and all(isinstance(test_case, str) and test_case.strip() for test_case in test_cases)
    ):
        return None
    return test_cases


def _finish_reward(sample: Any, trace: dict[str, Any] | None, reward: float, failure_reason: str | None) -> float:
    if trace is not None:
        trace["final_reward"] = reward
        trace["failure_reason"] = failure_reason
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(sample, "metadata", metadata)
        # Reuse the generic code-reward key so the script's existing rollout
        # output writer records AceCode traces as well as Eurus traces.
        metadata["code_reward_trace"] = trace
    return reward


async def reward_func(args: Any, sample: Any, **kwargs: Any) -> float:
    """Run AceCode assertions against generated code in SandboxFusion.

    Eurus validation rows retain their ``ground_truth`` labels, so delegate
    those rows to the existing Eurus reward implementation.
    """
    label = getattr(sample, "label", None)
    if isinstance(label, Mapping) and "ground_truth" in label:
        return await eurus_code.reward_func(args, sample, **kwargs)

    del args, kwargs
    trace: dict[str, Any] | None = (
        {"generated_code": None, "test_cases": []}
        if os.environ.get("SAVE_OUTPUTS", "").strip().lower() == "true"
        else None
    )
    try:
        generated_code = eurus_code._extract_python_code(getattr(sample, "response", None))
        if trace is not None:
            trace["generated_code"] = generated_code
        if generated_code is None:
            return _finish_reward(sample, trace, 0.0, "no_python_code")
        if not eurus_code._has_valid_python_syntax(generated_code):
            return _finish_reward(sample, trace, 0.0, "invalid_python_syntax")

        test_cases = _parse_test_cases(label)
        if test_cases is None:
            return _finish_reward(sample, trace, 0.0, "invalid_test_cases")

        for original_index, test_case in enumerate(test_cases):
            # Every test gets a fresh interpreter: setup statements in one
            # AceCode test cannot leak mutable state into another.
            executable = f"{generated_code}\n\n# AceCode test case\n{test_case}\n"
            try:
                compile(executable, "<acecode-reward>", "exec")
            except SyntaxError:
                return _finish_reward(sample, trace, 0.0, "invalid_test_case_syntax")

            result = await asyncio.to_thread(code_sandbox.run_code, executable, "")
            if trace is not None:
                trace["test_cases"].append(
                    {
                        "original_index": original_index,
                        "test_case": test_case,
                        "sandbox_result": eurus_code._sandbox_result_to_dict(result),
                    }
                )
            if not result.success or result.status != "ok":
                return _finish_reward(sample, trace, 0.0, f"sandbox_{result.status}")
    except Exception as exc:
        if trace is not None:
            trace["exception"] = f"{type(exc).__name__}: {exc}"
        return _finish_reward(sample, trace, 0.0, "reward_exception")

    return _finish_reward(sample, trace, 1.0, None)
