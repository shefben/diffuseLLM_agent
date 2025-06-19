#!/usr/bin/env python3
from pathlib import Path
import argparse

from src.utils.config_loader import load_app_config
from src.learning.ft_dataset import build_finetune_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create fine-tuning dataset from success memory"
    )
    parser.add_argument("--config", type=Path, default=None, help="Config YAML")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
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

    output_path = args.output or data_dir / "finetune_dataset.jsonl"
    verbose = args.verbose or cfg.get("general", {}).get("verbose", False)
    success = build_finetune_dataset(data_dir, output_path, verbose=verbose)
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
