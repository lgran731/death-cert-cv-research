from pathlib import Path
from typing import Any, Dict

import yaml


def load_states_config(path: str = "configs/states.yaml") -> Dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_state_eras(state_code: str, path: str = "configs/states.yaml"):
    cfg = load_states_config(path)
    defaults = cfg.get("defaults", {})
    state_cfg = cfg.get("states", {}).get(state_code.upper(), {})
    min_year = state_cfg.get("min_year", defaults.get("min_year", 1900))
    max_year = state_cfg.get("max_year", defaults.get("max_year", 1948))
    buckets = defaults.get("era_buckets", [])
    return {"min_year": min_year, "max_year": max_year, "era_buckets": buckets}

