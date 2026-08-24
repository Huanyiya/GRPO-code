#!/usr/bin/env python3
"""Sample a reproducible Polaris JSONL subset by difficulty labels."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--difficulties", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    wanted = set(args.difficulties)
    candidates = []
    with args.input.open() as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip() and json.loads(line).get("difficulty") in wanted:
                candidates.append((line_number, line))

    if len(candidates) < args.count:
        raise ValueError(f"Only {len(candidates)} matching samples; cannot sample {args.count}.")

    selected = random.Random(args.seed).sample(candidates, args.count)
    selected.sort(key=lambda pair: pair[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.writelines(line for _, line in selected)

    counts = Counter(json.loads(line)["difficulty"] for _, line in selected)
    print(f"Saved {len(selected)} samples to {args.output}")
    print("Difficulty counts:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
