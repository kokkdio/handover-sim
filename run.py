#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_MAP = {
    "record_trajectory": Path("examples/record_trajectory.py"),
    "predict_human_trajectory": Path("examples/predict_human_trajectory.py"),
}


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    return config


def build_cli_args(params: dict) -> list[str]:
    args: list[str] = []
    for key, value in params.items():
        flag = f"--{key.replace('_', '-') }"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif value is None:
            continue
        else:
            args.extend([flag, str(value)])
    return args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an example with a config YAML file.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config file under configs/ablations.",
    )
    parser.add_argument(
        "--script",
        choices=list(SCRIPT_MAP.keys()),
        help="Override the script name from the config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional seed to override the config file.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    script_name = args.script or config.get("script")
    if not script_name:
        raise ValueError("Config file must define a 'script' key.")
    if script_name not in SCRIPT_MAP:
        raise ValueError(
            f"Unknown script '{script_name}'. Supported scripts: {', '.join(SCRIPT_MAP)}"
        )

    params = config.get("params", {}) or {}
    if not isinstance(params, dict):
        raise ValueError("Config file 'params' section must be a mapping.")
    if args.seed is not None:
        params["seed"] = args.seed

    script_path = SCRIPT_MAP[script_name]
    if not script_path.exists():
        raise FileNotFoundError(f"Example script not found: {script_path}")

    cmd = [sys.executable, str(script_path)] + build_cli_args(params)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
