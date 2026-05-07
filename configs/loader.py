"""configs/loader.py — Utility to load YAML config and prompts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_prompts() -> dict:
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)
