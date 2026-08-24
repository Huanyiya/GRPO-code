#!/usr/bin/env python3
"""Create a reproducible difficulty-balanced subset of Polaris-53K."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-boundary", type=int, default=5000)
    parser.add_argument(
        "--append-difficulties-to",
        type=Path,
        help="Append every sample with these difficulty labels to an existing JSONL file.",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["1/8", "2/8"],
        help="Difficulty labels used with --append-difficulties-to (default: 1/8 2/8).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.append_difficulties_to:
        selected = []
        selected_labels = set(args.difficulties)
        with args.input.open() as f:
            for line in f:
                if line.strip() and json.loads(line).get("difficulty") in selected_labels:
                    selected.append(line)
        with args.append_difficulties_to.open("a") as f:
            f.writelines(selected)
        counts = Counter(json.loads(line)["difficulty"] for line in selected)
        print(f"Appended {len(selected)} samples to {args.append_difficulties_to}")
        print("Appended difficulty counts:", dict(sorted(counts.items())))
        return

    by_difficulty = {"0/8": [], "1/8": [], "2/8": [], "3/8": []}

    with args.input.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            difficulty = item.get("difficulty")
            if difficulty in by_difficulty:
                by_difficulty[difficulty].append((line_number, line))

    for difficulty in ("0/8", "3/8"):
        if len(by_difficulty[difficulty]) < args.sample_per_boundary:
            raise ValueError(
                f"Only {len(by_difficulty[difficulty])} samples have difficulty {difficulty}; "
                f"need {args.sample_per_boundary}."
            )

    rng = random.Random(args.seed)
    selected = by_difficulty["1/8"] + by_difficulty["2/8"]
    selected += rng.sample(by_difficulty["0/8"], args.sample_per_boundary)
    selected += rng.sample(by_difficulty["3/8"], args.sample_per_boundary)

    # Keep the original dataset ordering after sampling for deterministic streaming.
    selected.sort(key=lambda pair: pair[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.writelines(line for _, line in selected)

    counts = Counter(json.loads(line)["difficulty"] for _, line in selected)
    print(f"Saved {len(selected)} samples to {args.output}")
    print("Selected difficulty counts:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
