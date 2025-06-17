#!/usr/bin/env python3
"""Convenience wrapper to launch the Flask web interface after initialization."""

import argparse
from pathlib import Path
from webapp.app import initialize_components, app
from src.utils.config_loader import load_app_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Start diffuseLLM web UI")
    parser.add_argument("project_root", type=Path, help="Path to the codebase")
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional config YAML"
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_app_config(args.config)
    port = args.port or cfg.get("webui", {}).get("port", 5001)

    initialize_components(args.project_root, args.config, args.verbose)

    app.run(host="0.0.0.0", port=port, debug=args.verbose)


if __name__ == "__main__":
    main()
