import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from PIL import Image


def _normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _find_image_key(sample: Dict) -> Optional[str]:
    for key, value in sample.items():
        if isinstance(value, Image.Image):
            return key
    for key in sample.keys():
        if "image" in key.lower():
            return key
    return None


def _pick_first(sample: Dict, candidates: Iterable[str]) -> Optional[str]:
    normalized = {_normalize_name(k): k for k in sample.keys()}
    for candidate in candidates:
        key = normalized.get(_normalize_name(candidate))
        if key:
            value = sample.get(key)
            if value is not None and str(value).strip() != "":
                return str(value).strip()
    return None


def _infer_year(text: str) -> Optional[int]:
    if not text:
        return None
    matches = re.findall(r"\b(19[0-4]\d)\b", text)
    if not matches:
        return None
    return int(matches[0])


def _extract_metadata(sample: Dict) -> Tuple[str, str, int]:
    state = (
        _pick_first(sample, ["state", "state_name", "state_code"])
        or "MD"
    )
    county = (
        _pick_first(sample, ["county", "city", "county_name", "countyofdeath"])
        or "Baltimore City"
    )
    raw_year = _pick_first(
        sample,
        ["year", "death_year", "date_of_death", "dod", "death_date", "date"],
    )
    year = _infer_year(raw_year or "") or 1950
    return state, county, year


def fetch_pilot(
    dataset_id: str,
    output_dir: Path,
    ground_truth_csv: Path,
    split: str = "train",
    limit: Optional[int] = None,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_csv.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_id, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    first = ds[0]
    image_key = _find_image_key(first)
    if image_key is None:
        raise RuntimeError(
            f"Could not find an image column in dataset sample keys: {list(first.keys())}"
        )

    rows = []
    for i, sample in enumerate(ds):
        image = sample[image_key]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        file_name = f"pilot_{i:05d}.jpg"
        file_path = output_dir / file_name
        image.save(file_path, format="JPEG", quality=95)

        state, county, year = _extract_metadata(sample)
        row = {
            "filename": str(Path("data/raw") / file_name).replace("\\", "/"),
            "state": state,
            "county": county,
            "year": year,
            "source": dataset_id,
            "split": split,
        }
        rows.append(row)

    with open(ground_truth_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "state", "county", "year", "source", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} pilot images to: {output_dir}")
    print(f"Wrote ground truth CSV to: {ground_truth_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch tier-1 pilot death certificate data.")
    parser.add_argument(
        "--dataset",
        default="Rasi1610/Deathce502_series1_new",
        help="Hugging Face dataset id",
    )
    parser.add_argument("--split", default="train", help="Dataset split to fetch")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows")
    parser.add_argument("--out-dir", default="data/raw", help="Image output directory")
    parser.add_argument(
        "--ground-truth-csv",
        default="data/index_pilot_groundtruth.csv",
        help="Ground-truth CSV output path",
    )
    args = parser.parse_args()

    fetch_pilot(
        dataset_id=args.dataset,
        output_dir=Path(args.out_dir),
        ground_truth_csv=Path(args.ground_truth_csv),
        split=args.split,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

