#!/usr/bin/env python3
"""Minimal CLI wrapper for running the assistant on a spec."""

import argparse
from pathlib import Path
import yaml

from src.utils.config_loader import load_app_config
from src.planner.phase_planner import PhasePlanner
from src.planner.spec_model import Spec
from src.digester.repository_digester import RepositoryDigester


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diffuseLLM assistant on a spec YAML"
    )
    parser.add_argument("spec", type=Path, help="Path to YAML spec")
    parser.add_argument("--project-root", type=Path, default=None, help="Project root")
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional config YAML"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    app_config = load_app_config(args.config)
    project_root = args.project_root or Path(
        app_config.get("general", {}).get("project_root", Path.cwd())
    )
    project_root = project_root.resolve()
    app_config["general"]["project_root"] = str(project_root)
    if args.verbose:
        app_config["general"]["verbose"] = True

    with open(args.spec, "r", encoding="utf-8") as f:
        spec_data = yaml.safe_load(f)
    spec = Spec(**spec_data)

    digester = RepositoryDigester(repo_path=project_root, app_config=app_config)
    planner = PhasePlanner(
        project_root_path=project_root, app_config=app_config, digester=digester
    )

    planner.plan_phases(
        spec, workflow_type=app_config.get("planner", {}).get("workflow_type")
    )


if __name__ == "__main__":
    main()
