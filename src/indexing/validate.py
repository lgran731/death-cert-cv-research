import argparse
import csv
from pathlib import Path
from typing import Dict


def _load_by_filename(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row for row in reader if row.get("filename")}


def _safe_int(text: str) -> int:
    try:
        return int(text)
    except Exception:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate predicted index against pilot ground truth."
    )
    parser.add_argument("--pred", default="data/index.csv", help="Predicted index CSV")
    parser.add_argument(
        "--truth",
        default="data/index_pilot_groundtruth.csv",
        help="Ground truth pilot CSV",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred)
    truth_path = Path(args.truth)
    if not pred_path.exists() or not truth_path.exists():
        raise FileNotFoundError("Both --pred and --truth CSV files must exist.")

    pred = _load_by_filename(pred_path)
    truth = _load_by_filename(truth_path)
    common = sorted(set(pred.keys()) & set(truth.keys()))
    if not common:
        raise RuntimeError("No overlapping filenames between prediction and truth.")

    state_ok = 0
    county_ok = 0
    year_ok = 0
    for key in common:
        p = pred[key]
        t = truth[key]

        if (p.get("state") or "").strip().upper() == (t.get("state") or "").strip().upper():
            state_ok += 1
        if (p.get("county") or "").strip().lower() == (t.get("county") or "").strip().lower():
            county_ok += 1
        if _safe_int(p.get("year", "")) == _safe_int(t.get("year", "")):
            year_ok += 1

    n = len(common)
    print(f"Rows compared: {n}")
    print(f"State accuracy:  {state_ok / n:.3%}")
    print(f"County accuracy: {county_ok / n:.3%}")
    print(f"Year accuracy:   {year_ok / n:.3%}")
    print("Targets: state >= 95%, county >= 80%, year >= 90%")


if __name__ == "__main__":
    main()

