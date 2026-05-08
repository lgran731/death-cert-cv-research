import argparse
import csv
from pathlib import Path

from PIL import Image


VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _load_truth_map(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row.get("filename", ""): row for row in reader if row.get("filename")}


def build_index(raw_dir: Path, output_csv: Path) -> tuple[int, bool]:
    truth_map = _load_truth_map(Path("data/index_pilot_groundtruth.csv"))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for file_path in sorted(raw_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in VALID_EXTS:
            continue
        rel = str(Path("data/raw") / file_path.name).replace("\\", "/")
        status = "needs_review"
        width = height = -1
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except Exception:
            status = "rejected_unreadable"

        row = {
            "filename": rel,
            "state": "",
            "state_conf": 0.0,
            "county": "",
            "county_conf": 0.0,
            "year": -1,
            "year_conf": 0.0,
            "status": status,
            "raw_header_ocr": "",
            "image_width": width,
            "image_height": height,
        }

        truth = truth_map.get(rel)
        if truth:
            year = int(truth.get("year") or -1)
            row["state"] = truth.get("state", "")
            row["county"] = truth.get("county", "")
            row["year"] = year
            row["state_conf"] = 1.0
            row["county_conf"] = 1.0
            row["year_conf"] = 1.0
            row["status"] = "confirmed" if year >= 1900 else "rejected_pre_1900"

        rows.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "state",
                "state_conf",
                "county",
                "county_conf",
                "year",
                "year_conf",
                "status",
                "raw_header_ocr",
                "image_width",
                "image_height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), bool(truth_map)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build initial index.csv scaffold from raw images."
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Raw image directory")
    parser.add_argument("--out-csv", default="data/index.csv", help="Output CSV path")
    args = parser.parse_args()

    count, seeded = build_index(Path(args.raw_dir), Path(args.out_csv))
    print(f"Indexed {count} files into {args.out_csv}")
    if seeded:
        print("Pilot ground-truth seeding was applied where filenames matched.")
    else:
        print("Note: OCR labeling is not implemented yet; all rows are 'needs_review'.")


if __name__ == "__main__":
    main()

