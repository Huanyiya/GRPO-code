"""Math reward adapter for the processed DAPO parquet dataset."""

import re
from collections.abc import Mapping
from typing import Any

from slime.rollout.rm_hub.math_utils import grade_answer_verl


def _get_ground_truth(label: Any) -> str:
    """Extract ``reward_model.ground_truth`` from a parquet sample label."""
    if isinstance(label, Mapping):
        ground_truth = label.get("ground_truth")
    else:
        # Also allow a scalar label so this reward remains easy to test/reuse.
        ground_truth = label

    if ground_truth is None:
        raise ValueError(f"Missing ground_truth in reward label: {label!r}")
    return str(ground_truth)


async def reward_func(args, sample, **kwargs) -> float:
    """Return 1 for a correct boxed or final ``Answer:`` answer.

    Unlike slime's ``deepscaler`` reward, this function does not require a
    ``</think>`` separator, so it works with Qwen3.5 non-thinking responses.
    """
    del args, kwargs
    ground_truth = _get_ground_truth(sample.label)
    response = sample.response

    # DAPO training prompts request \boxed{}, while the AIME25 evaluation set
    # requests a final line such as "Answer: 70". Feed both forms through the
    # same established mathematical equivalence checker.
    if "\\boxed" not in response:
        answer_lines = re.findall(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$", response)
        if answer_lines:
            response = rf"\boxed{{{answer_lines[-1].strip()}}}"

    return float(grade_answer_verl(response, ground_truth))
