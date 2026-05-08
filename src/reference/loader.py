import json
from pathlib import Path
from typing import Any, Dict, List


REFERENCE_DIR = Path("data/reference")


def _load_json(name: str) -> Dict[str, Any]:
    path = REFERENCE_DIR / name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_names(era_bucket: str, region: str = "US") -> List[str]:
    data = _load_json("names_by_era.json")
    return data.get(era_bucket, {}).get(region, [])


def get_counties(state_code: str, year: int) -> List[str]:
    data = _load_json("counties_by_state_year.json")
    year_map = data.get(state_code.upper(), {})
    return year_map.get(str(year), [])


def get_causes(era_bucket: str) -> List[str]:
    data = _load_json("icd_causes_by_era.json")
    return data.get(era_bucket, [])


def get_occupations(era_bucket: str) -> List[str]:
    data = _load_json("occupations_by_era.json")
    return data.get(era_bucket, [])

