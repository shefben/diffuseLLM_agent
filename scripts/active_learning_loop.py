#!/usr/bin/env python3
"""Run periodic fine-tuning and predictor training using success memory."""

import argparse
from pathlib import Path
from time import sleep

from src.utils.config_loader import load_app_config
from src.utils.memory_logger import load_success_memory
from src.learning.ft_dataset import build_finetune_dataset
from src.learning.core_predictor import CorePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Active learning loop")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--predictor-model", type=Path, default=None)
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

    al_cfg = cfg.get("active_learning", {})
    model_name = args.model_name or al_cfg.get("model_name", "gpt2")
    output_dir = args.output_dir or Path(al_cfg.get("output_dir", "./lora_adapters"))
    interval = args.interval or al_cfg.get("interval", 3600)
    predictor_model = args.predictor_model or Path(
        al_cfg.get("predictor_model", "./models/core_predictor.joblib")
    )

    verbose = args.verbose or cfg.get("general", {}).get("verbose", False)

    while True:
        entries = load_success_memory(data_dir, verbose=verbose)
        if verbose:
            print(f"Active learning: loaded {len(entries)} success entries")
        if len(entries) >= 5:
            dataset_path = data_dir / "finetune_dataset.jsonl"
            build_finetune_dataset(data_dir, dataset_path, verbose=verbose)
            cmd = ["python3", "scripts/finetune_lora.py"]
            if args.config:
                cmd += ["--config", str(args.config)]
            cmd += [
                "--data-dir",
                str(data_dir),
                "--model-name",
                model_name,
                "--output-dir",
                str(output_dir),
                "--dataset-path",
                str(dataset_path),
            ]
            if verbose:
                cmd.append("--verbose")
            try:
                import subprocess

                subprocess.run(cmd, check=False)
            except Exception as e:
                if verbose:
                    print(
                        f"Active learning warning: failed to run LoRA fine-tuning: {e}"
                    )

            predictor = CorePredictor(model_path=predictor_model, verbose=verbose)
            predictor.train(
                training_data_path=data_dir, model_output_path=predictor_model
            )
        else:
            if verbose:
                print("Active learning: not enough data for training")
        sleep(interval)


if __name__ == "__main__":
    main()
