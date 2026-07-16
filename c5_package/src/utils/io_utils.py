from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(project_path(".env"))


def _resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

        def repl(match: re.Match[str]) -> str:
            return os.getenv(match.group(1), "")

        return pattern.sub(repl, value)
    return value


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    load_dotenv_if_available()
    with project_path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _resolve_env_placeholders(raw)


def ensure_parent(path: str | Path) -> None:
    project_path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> Path:
    full = project_path(path)
    full.mkdir(parents=True, exist_ok=True)
    return full


def read_json(path: str | Path, default: Any | None = None) -> Any:
    full = project_path(path)
    if not full.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON 文件不存在: {full}")
    with full.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    ensure_parent(path)
    with project_path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_text(path: str | Path, default: str = "") -> str:
    full = project_path(path)
    if not full.exists():
        return default
    return full.read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    ensure_parent(path)
    project_path(path).write_text(text, encoding="utf-8")

