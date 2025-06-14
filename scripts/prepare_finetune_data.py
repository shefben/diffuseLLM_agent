#!/usr/bin/env python3
from pathlib import Path
import argparse

from src.learning.ft_dataset import build_finetune_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fine-tuning dataset from success memory")
    parser.add_argument("data_dir", type=Path, help="Directory containing success_memory.jsonl")
    parser.add_argument("output", type=Path, help="Output JSONL dataset path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    success = build_finetune_dataset(args.data_dir, args.output, verbose=args.verbose)
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
