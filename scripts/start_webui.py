#!/usr/bin/env python3
"""Convenience wrapper to launch the Flask web interface after initialization."""

import argparse
from pathlib import Path
from webapp.app import initialize_components, app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start diffuseLLM web UI")
    parser.add_argument("project_root", type=Path, help="Path to the codebase")
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional config YAML"
    )
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    initialize_components(args.project_root, args.config, args.verbose)

    app.run(host="0.0.0.0", port=args.port, debug=args.verbose)


if __name__ == "__main__":
    main()
