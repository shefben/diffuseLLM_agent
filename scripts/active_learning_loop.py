#!/usr/bin/env python3
"""Run periodic fine-tuning and predictor training using success memory."""

import argparse
from pathlib import Path
from time import sleep

from src.utils.memory_logger import load_success_memory
from src.learning.ft_dataset import build_finetune_dataset
from src.learning.core_predictor import CorePredictor



def main() -> None:
    parser = argparse.ArgumentParser(description="Active learning loop")
    parser.add_argument(
        "data_dir", type=Path, help="Directory with success_memory.jsonl"
    )
    parser.add_argument("model_name", type=str, help="Base model for LoRA fine-tune")
    parser.add_argument("output_dir", type=Path, help="Dir to save adapters")
    parser.add_argument(
        "--interval", type=int, default=3600, help="Seconds between checks"
    )
    parser.add_argument(
        "--predictor-model", type=Path, default=Path("./models/core_predictor.joblib")
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    while True:
        entries = load_success_memory(args.data_dir, verbose=args.verbose)
        if args.verbose:
            print(f"Active learning: loaded {len(entries)} success entries")
        if len(entries) >= 5:
            dataset_path = args.data_dir / "finetune_dataset.jsonl"
            build_finetune_dataset(args.data_dir, dataset_path, verbose=args.verbose)
            cmd = [
                "python3",
                "scripts/finetune_lora.py",
                str(args.data_dir),
                args.model_name,
                str(args.output_dir),
                "--dataset-path",
                str(dataset_path),
            ]
            if args.verbose:
                cmd.append("--verbose")
            try:
                import subprocess

                subprocess.run(cmd, check=False)
            except Exception as e:
                if args.verbose:
                    print(
                        f"Active learning warning: failed to run LoRA fine-tuning: {e}"
                    )

            predictor = CorePredictor(
                model_path=args.predictor_model, verbose=args.verbose
            )
            predictor.train(
                training_data_path=args.data_dir, model_output_path=args.predictor_model
            )
        else:
            if args.verbose:
                print("Active learning: not enough data for training")
        sleep(args.interval)


if __name__ == "__main__":
    main()
