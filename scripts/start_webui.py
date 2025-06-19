#!/usr/bin/env python3
"""Convenience wrapper to launch the Flask web interface after initialization."""

import argparse
from pathlib import Path
from webapp.app import initialize_components, app, app_config_global
from src.utils.config_loader import load_app_config
import subprocess


def start_active_learning_loop(
    model_name: str,
    output_dir: Path,
    interval: int,
    predictor_model: Path,
    config_path: Path | None,
    verbose: bool,
) -> None:
    """Spawn the active learning loop in a background subprocess."""
    data_dir = Path(app_config_global.get("general", {}).get("data_dir", ".agent_data"))
    if not data_dir.is_absolute():
        project_root = Path(
            app_config_global.get("general", {}).get("project_root", ".")
        )
        data_dir = project_root / data_dir
    cmd = ["python3", "scripts/active_learning_loop.py"]
    if config_path:
        cmd += ["--config", str(config_path)]
    cmd += [
        "--data-dir",
        str(data_dir),
        "--model-name",
        model_name,
        "--output-dir",
        str(output_dir),
        "--interval",
        str(interval),
        "--predictor-model",
        str(predictor_model),
    ]
    if verbose:
        cmd.append("--verbose")
    try:
        subprocess.Popen(cmd)
    except Exception as exc:  # pragma: no cover - subprocess errors
        print(f"Failed to start active learning loop: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start diffuseLLM web UI")
    parser.add_argument(
        "project_root", type=Path, nargs="?", help="Path to the codebase"
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional config YAML"
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--active-learning", action="store_true")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--predictor-model", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_app_config(args.config)
    port = args.port or cfg.get("webui", {}).get("port", 5001)

    al_cfg = cfg.get("active_learning", {})
    model_name = args.model_name or al_cfg.get("model_name", "gpt2")
    output_dir = args.output_dir or Path(al_cfg.get("output_dir", "./lora_adapters"))
    interval = args.interval or al_cfg.get("interval", 3600)
    predictor_model = args.predictor_model or Path(
        al_cfg.get("predictor_model", "./models/core_predictor.joblib")
    )

    if args.project_root:
        initialize_components(args.project_root, args.config, args.verbose)

    if args.active_learning or al_cfg.get("enabled", False):
        start_active_learning_loop(
            model_name=model_name,
            output_dir=output_dir,
            interval=interval,
            predictor_model=predictor_model,
            config_path=args.config,
            verbose=args.verbose,
        )

    app.run(host="0.0.0.0", port=port, debug=args.verbose)


if __name__ == "__main__":
    main()
