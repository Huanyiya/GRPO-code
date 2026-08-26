#!/usr/bin/env python3
"""Convert AceCode's raw JSONL into the format used by SLIME code GRPO."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/mnt/cpfs/users/wxh/GRPO/dataset/AceCoder/train/train_rl/OpenRLHF/data/"
    "acecode_87K/acecode_87K_hard.json_hard"
)
DEFAULT_OUTPUT = DEFAULT_SOURCE.with_name("acecode_87K_hard.slime.jsonl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _convert_row(row: object, line_number: int) -> dict[str, object]:
    if not isinstance(row, dict):
        raise ValueError(f"line {line_number}: expected a JSON object")

    messages = row.get("context_messages")
    test_cases = row.get("test_cases")
    if not (
        isinstance(messages, list)
        and messages
        and all(
            isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and isinstance(message.get("content"), str)
            for message in messages
        )
    ):
        raise ValueError(f"line {line_number}: invalid context_messages")
    if not (
        isinstance(test_cases, list)
        and test_cases
        and all(isinstance(test_case, str) and test_case.strip() for test_case in test_cases)
    ):
        raise ValueError(f"line {line_number}: invalid test_cases")

    # Syntax checking catches corrupt labels before a costly distributed run.
    for test_case_index, test_case in enumerate(test_cases):
        try:
            # Some valid labels contain regular-expression strings such as
            # "\\d". They are harmless SyntaxWarnings on current Python,
            # but should not make a normal conversion excessively noisy.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                compile(test_case, f"<acecode-test-{line_number}-{test_case_index}>", "exec")
        except SyntaxError as exc:
            raise ValueError(f"line {line_number}: invalid test case {test_case_index}: {exc}") from exc

    metadata = {key: row[key] for key in ("id", "source") if key in row}
    return {
        "prompt": messages,
        "reward_model": {"test_cases": test_cases},
        "metadata": metadata,
    }


def main() -> None:
    args = _parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"AceCode source does not exist: {args.source}")
    if args.output.resolve() == args.source.resolve():
        raise ValueError("--output must not overwrite --source")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with args.source.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                raw_row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            output.write(json.dumps(_convert_row(raw_row, line_number), ensure_ascii=False) + "\n")
            rows += 1

    print(f"Wrote {rows} rows to {args.output}")


if __name__ == "__main__":
    main()
