#!/usr/bin/env python3
"""Initialize all phases and launch the web UI."""

import argparse
from pathlib import Path
import subprocess
from webapp.app import initialize_components, app, app_config_global


def start_active_learning_loop(
    model_name: str,
    output_dir: Path,
    interval: int,
    predictor_model: Path,
    verbose: bool,
) -> None:
    """Spawn the active learning loop in a background subprocess."""
    data_dir = Path(app_config_global.get("general", {}).get("data_dir", ".agent_data"))
    if not data_dir.is_absolute():
        project_root = Path(
            app_config_global.get("general", {}).get("project_root", ".")
        )
        data_dir = project_root / data_dir
    cmd = [
        "python3",
        "scripts/active_learning_loop.py",
        str(data_dir),
        model_name,
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
    parser = argparse.ArgumentParser(
        description="Launch diffuseLLM assistant with web UI"
    )
    parser.add_argument("project_root", type=Path, help="Path to the codebase")
    parser.add_argument("--config", type=Path, default=None, help="Configuration YAML")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument(
        "--active-learning",
        action="store_true",
        help="Run active learning loop in background",
    )
    parser.add_argument(
        "--model-name", type=str, default="gpt2", help="Base model for LoRA fine-tuning"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./lora_adapters"))
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument(
        "--predictor-model", type=Path, default=Path("./models/core_predictor.joblib")
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    initialize_components(args.project_root, args.config, args.verbose)

    if args.active_learning:
        start_active_learning_loop(
            model_name=args.model_name,
            output_dir=args.output_dir,
            interval=args.interval,
            predictor_model=args.predictor_model,
            verbose=args.verbose,
        )

    app.run(host="0.0.0.0", port=args.port, debug=args.verbose)


if __name__ == "__main__":
    main()
