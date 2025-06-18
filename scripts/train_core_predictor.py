#!/usr/bin/env python3
from pathlib import Path
import argparse

from src.utils.config_loader import load_app_config
from src.learning.core_predictor import CorePredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the CorePredictor from success memory"
    )
    parser.add_argument("--config", type=Path, default=None, help="Config YAML")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_app_config(args.config)
    project_root = Path(cfg.get("general", {}).get("project_root", "."))
    if not project_root.is_absolute():
        project_root = Path.cwd() / project_root

    data_dir = args.data_dir or Path(
        cfg.get("general", {}).get("data_dir", ".agent_data")
    )
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    model_path = args.model_path or Path(
        cfg.get("active_learning", {}).get(
            "predictor_model", "./models/core_predictor.joblib"
        )
    )
    verbose = args.verbose or cfg.get("general", {}).get("verbose", False)

    predictor = CorePredictor(model_path=model_path, verbose=verbose)
    ok = predictor.train(training_data_path=data_dir, model_output_path=model_path)
    if not ok:
        exit(1)


if __name__ == "__main__":
    main()
