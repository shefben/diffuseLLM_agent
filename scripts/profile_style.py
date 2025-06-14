#!/usr/bin/env python3
"""Run the style profiling orchestrator on a repository."""

import argparse
from pathlib import Path

from src.profiler.orchestrator import run_phase1_style_profiling_pipeline
from src.utils.config_loader import load_app_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate style fingerprint and configs"
    )
    parser.add_argument("project_root", type=Path, help="Root of project to profile")
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional config YAML"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    app_config = load_app_config(args.config)
    app_config["general"]["project_root"] = str(args.project_root.resolve())
    if args.verbose:
        app_config["general"]["verbose"] = True

    run_phase1_style_profiling_pipeline(args.project_root, app_config)


if __name__ == "__main__":
    main()
