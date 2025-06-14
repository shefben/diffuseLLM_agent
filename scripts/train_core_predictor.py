#!/usr/bin/env python3
from pathlib import Path
import argparse

from src.learning.core_predictor import CorePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CorePredictor from success memory")
    parser.add_argument("data_dir", type=Path, help="Directory containing success_memory.jsonl")
    parser.add_argument("--model-path", type=Path, default=Path("./models/core_predictor.joblib"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    predictor = CorePredictor(model_path=args.model_path, verbose=args.verbose)
    ok = predictor.train(training_data_path=args.data_dir, model_output_path=args.model_path)
    if not ok:
        exit(1)


if __name__ == "__main__":
    main()
