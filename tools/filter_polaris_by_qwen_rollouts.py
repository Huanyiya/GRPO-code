#!/usr/bin/env python3
"""Keep Polaris problems whose Qwen rollout accuracy is strictly between 0 and 1.

The script talks to an already-running OpenAI-compatible SGLang server.  It
submits eight independent samples for every problem, scores each sample with
the same boxed-answer equivalence checker used by the GRPO reward, and writes
only problems with 1..7 correct samples as JSONL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import aiohttp
from tqdm import tqdm

# Allow direct execution as `python tools/filter_polaris_by_qwen_rollouts.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.rollout.rm_hub.math_utils import grade_answer_verl


DEFAULT_INPUT = "/mnt/cpfs/users/zhy/opd/OPD/datasets/Polaris-53K/polaris-data-53K.jsonl"
DEFAULT_OUTPUT = "/mnt/cpfs/users/zhy/opd/OPD/datasets/Polaris-53K/polaris-data-53K-qwen3.5-9b-1to7of8.jsonl"

SYSTEM_PROMPT = (
    "You are a mathematical problem solver. Solve the problem carefully. "
    "Do not produce a hidden reasoning block. End the response with the final answer "
    r"in exactly this form: \boxed{...}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000/v1",
        help="SGLang OpenAI API base URL; /v1 is added automatically if omitted.",
    )
    parser.add_argument("--model", default="default", help="The model name sent to the OpenAI-compatible API.")
    parser.add_argument("--rollouts-per-problem", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-response-tokens", type=int, default=16384)
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=32,
        help="Global concurrent HTTP generations, across all problems.",
    )
    parser.add_argument(
        "--max-inflight-problems",
        type=int,
        default=16,
        help="Bounded number of problems being evaluated simultaneously.",
    )
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=None, help="Only process this many input rows; useful for a smoke test.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def response_is_correct(response: str, answer: Any) -> bool:
    """Match the non-thinking GRPO reward's boxed/`Answer:` behavior."""
    if "\\boxed" not in response:
        answer_lines = re.findall(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$", response)
        if answer_lines:
            response = rf"\boxed{{{answer_lines[-1].strip()}}}"
    return bool(grade_answer_verl(response, str(answer)))


async def generate_one(
    session: aiohttp.ClientSession,
    url: str,
    record: dict[str, Any],
    args: argparse.Namespace,
    request_semaphore: asyncio.Semaphore,
    seed: int,
) -> str:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["problem"]},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_response_tokens,
        "seed": seed,
        "stream": False,
        # Qwen3.5's template accepts this and keeps the request non-thinking.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            async with request_semaphore:
                async with session.post(url, json=payload) as response:
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")
                    result = await response.json()
            choices = result.get("choices", [])
            if not choices:
                raise RuntimeError(f"SGLang response has no choices: {result}")
            content = choices[0].get("message", {}).get("content")
            if not isinstance(content, str):
                raise RuntimeError(f"SGLang response has no message content: {result}")
            return content
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as error:
            last_error = error
            if attempt < args.retries:
                await asyncio.sleep(min(2**attempt, 10))

    assert last_error is not None
    raise last_error


async def evaluate_problem(
    source_index: int,
    record: dict[str, Any],
    session: aiohttp.ClientSession,
    completions_url: str,
    args: argparse.Namespace,
    request_semaphore: asyncio.Semaphore,
    rollout_progress: tqdm,
) -> tuple[int, dict[str, Any], int]:
    rollout_tasks = [
        asyncio.create_task(
            generate_one(
                session,
                completions_url,
                record,
                args,
                request_semaphore,
                args.seed + source_index * args.rollouts_per_problem + rollout_index,
            )
        )
        for rollout_index in range(args.rollouts_per_problem)
    ]
    # The callback runs in this event loop, so tqdm is updated for every
    # completed sample rather than only after all eight samples of a problem.
    for rollout_task in rollout_tasks:
        rollout_task.add_done_callback(lambda _task: rollout_progress.update(1))
    results = await asyncio.gather(*rollout_tasks, return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise RuntimeError(f"{len(errors)}/{args.rollouts_per_problem} rollouts failed; first error: {errors[0]}")
    responses = [result for result in results if isinstance(result, str)]
    correct_count = sum(response_is_correct(response, record["answer"]) for response in responses)
    return source_index, record, correct_count


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


async def run(args: argparse.Namespace) -> None:
    if args.rollouts_per_problem != 8:
        raise ValueError("This filter is intentionally defined for exactly 8 rollouts per problem.")
    if args.max_concurrent_requests < 1 or args.max_inflight_problems < 1:
        raise ValueError("Concurrency values must be positive.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {args.input}")
    if args.output.resolve() == args.input.resolve():
        raise ValueError("--output must not overwrite --input.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}; use --overwrite to replace it.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completions_url = f"{normalize_base_url(args.base_url)}/chat/completions"
    input_total = count_nonempty_lines(args.input)
    target_total = min(input_total, args.limit) if args.limit is not None else input_total
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=args.max_concurrent_requests, enable_cleanup_closed=True)
    request_semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    completed = kept = invalid = failed = 0
    pending: set[asyncio.Task] = set()

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"SGLang endpoint: {completions_url}")
    print(
        f"Evaluating {target_total} problems x {args.rollouts_per_problem} rollouts "
        f"(request concurrency={args.max_concurrent_requests})"
    )

    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
        with args.input.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
            progress = tqdm(
                total=target_total * args.rollouts_per_problem,
                desc="Polaris rollouts",
                unit="rollout",
                dynamic_ncols=True,
            )

            def submit(source_index: int, raw_line: str) -> None:
                nonlocal invalid
                try:
                    record = json.loads(raw_line)
                    if not isinstance(record.get("problem"), str) or "answer" not in record:
                        raise ValueError("expected string field 'problem' and field 'answer'")
                except (json.JSONDecodeError, ValueError) as error:
                    invalid += 1
                    progress.update(args.rollouts_per_problem)
                    progress.set_postfix(kept=kept, invalid=invalid, failed=failed)
                    tqdm.write(f"Skipping invalid row {source_index}: {error}")
                    return
                pending.add(
                    asyncio.create_task(
                        evaluate_problem(
                            source_index,
                            record,
                            session,
                            completions_url,
                            args,
                            request_semaphore,
                            progress,
                        )
                    )
                )

            submitted = 0
            for source_index, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                if args.limit is not None and submitted >= args.limit:
                    break
                submitted += 1
                submit(source_index, raw_line)

                if len(pending) < args.max_inflight_problems:
                    continue
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    completed, kept, failed = write_result(task, output, completed, kept, failed, progress)

            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    completed, kept, failed = write_result(task, output, completed, kept, failed, progress)
            progress.close()

    print(
        f"Finished: evaluated={completed}, kept={kept}, invalid={invalid}, failed={failed}.\n"
        f"New training dataset total: {kept}\n"
        f"Saved to: {args.output}"
    )


def write_result(
    task: asyncio.Task,
    output,
    completed: int,
    kept: int,
    failed: int,
    progress: tqdm,
) -> tuple[int, int, int]:
    try:
        source_index, record, correct_count = task.result()
        completed += 1
        if 1 <= correct_count <= 7:
            # Extra metadata is harmless to Slime's problem/answer data loader and
            # lets us audit the filter without storing huge generated responses.
            record["rollout_filter"] = {
                "model": "qwen3.5-9b",
                "rollouts": 8,
                "correct": correct_count,
                "source_line": source_index,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            kept += 1
    except Exception as error:  # Keep a long filtering job alive despite isolated failures.
        completed += 1
        failed += 1
        tqdm.write(f"Problem rollout failed: {type(error).__name__}: {error}")

    progress.set_postfix(kept=kept, failed=failed)
    return completed, kept, failed


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
