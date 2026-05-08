# Death Certificate CV Classifier

A computer vision research pipeline for classifying historical U.S. death certificates by state and county using document layout analysis.

## Overview

This project builds a labeled image dataset from publicly accessible historical death certificate collections via the [FamilySearch REST API](https://developers.familysearch.org), then trains a computer vision (CV) classification model to identify the U.S. state (and where possible, county) of origin for a given scanned death certificate.

The classifier focuses on the **printed template structure** of each certificate — form layout, field arrangement, header design, and typographic patterns — rather than handwritten content. These structural features vary distinctly by state and era, making them reliable signals for document classification.

---

## Motivation

Large archives of unclassified or mislabeled scanned death certificates exist across many genealogical and archival institutions. A reliable visual classifier could help:

- Archivists automatically sort and route scanned certificates at scale
- Genealogy platforms improve indexing and search accuracy
- Researchers identify the provenance of orphaned or unlabeled document scans

---

## Pipeline Architecture

```
FamilySearch API
      │
      ▼
OAuth 2.0 Authentication
      │
      ▼
Record Search (by collection / state)
      │
      ▼
Image Download + Metadata Extraction
      │
      ▼
Local Dataset (organized by state / county)
      │
      ▼
Image Preprocessing
  - Deskewing
  - Binarization (Sauvola thresholding)
  - CLAHE contrast enhancement
  - Resolution normalization (300 DPI)
      │
      ▼
Stage 1: Visual Layout Classifier (EfficientNet / ViT)
  → Predicts: U.S. State
      │
      ▼
Stage 2: OCR-Assisted Refinement (PaddleOCR / TrOCR)
  → Predicts: County (for low-confidence Stage 1 results)
```

---

## Data Sources

Images are retrieved from publicly accessible historical death certificate collections across all 50 U.S. states. Sources include FamilySearch (free), state archives (free), and supplemental fee-based sources (Ancestry.com) where FamilySearch coverage is limited.

Target dataset size: **50,000–100,000 images** across all 50 U.S. states, with a minimum of **500 images per state** to ensure balanced class representation.

### Collection Index by State

| State | Primary Source | Collection / URL | Coverage | Access |
|---|---|---|---|---|
| Alabama | FamilySearch | [1307888](https://www.familysearch.org/search/collection/1307888) | 1908–1974 | FHC only |
| Alaska | State Archives | archives.alaska.gov | 1900s+ | Free |
| Arizona | AZ Dept. of Health | azdhs.gov/licensing/vital-records | 1870–1974 | Free |
| Arkansas | Ancestry | collection/61777 | 1914–1969 | Fee |
| California | FamilySearch | [2001287](https://www.familysearch.org/search/collection/2001287) | 1800–1994 | Free (partial) |
| Colorado | Ancestry / FamilySearch | Various | 1900s+ | Mixed |
| Connecticut | FamilySearch | Vital records catalog | 1600s+ | Free |
| Delaware | FamilySearch | [1520546](https://www.familysearch.org/search/collection/1520546) | 1855–1961 | Free |
| Florida | FamilySearch | [1595003](https://www.familysearch.org/search/collection/1595003) | 1877–1939 | Free |
| Georgia | State Archives | dph.georgia.gov | 1919–1942 | Free |
| Hawaii | Ancestry | collection/61694 | 1841–1942 | Fee |
| Idaho | FamilySearch | Vital records catalog | 1911–1937 | Free |
| Illinois | FamilySearch | [1438856](https://www.familysearch.org/search/collection/1438856) | 1916–1947 | Free |
| Indiana | Ancestry | Indiana death index | 1899–2011 | Fee |
| Iowa | FamilySearch | Iowa death records | 1904–1942 | Free |
| Kansas | FamilySearch / Ancestry | Various | 1900s+ | Mixed |
| Kentucky | FamilySearch | Kentucky death certs | 1911–1965 | Free |
| Louisiana | FamilySearch | Louisiana deaths | 1800s–1900s | Free (partial) |
| Maine | FamilySearch | Maine vital records | 1600s+ | Free |
| Maryland | Internet Archive | [MD Death Certs](https://archive.org/details/md-death-certificates) | 1898–2012 | Free |
| Massachusetts | FamilySearch | [1463156](https://www.familysearch.org/search/collection/1463156) | 1841–1924 | Free |
| Michigan | FamilySearch / Michiganology | Michigan death certs | 1867–1943 | Free |
| Minnesota | FamilySearch | [2185953](https://www.familysearch.org/search/collection/2185953) | 1850–2001 | Free |
| Mississippi | FamilySearch | Mississippi deaths | 1900s+ | Free (partial) |
| Missouri | MO State Archives | sos.mo.gov | 1910–1974 | Free |
| Montana | Ancestry | collection/5437 | 1907–2018 | Fee |
| Nebraska | FamilySearch / Ancestry | Various | 1900s+ | Mixed |
| Nevada | Ancestry | Nevada death records | 1911–1965 | Fee |
| New Hampshire | FamilySearch | NH death records | 1654–1959 | Free |
| New Jersey | FamilySearch | NJ vital records | 1800s–1900s | Free (partial) |
| New Mexico | FamilySearch | NM death certs | 1889–1945 | Free |
| New York City | NYC Vital Records Project | nyc.gov/vitalrecords | 1855–1949 | Free |
| New York (State) | FamilySearch | [2803479](https://www.familysearch.org/search/collection/2803479) | 1880–1956 | Free (index) |
| North Carolina | FamilySearch | NC death records | 1906–1930 | Free |
| North Dakota | Ancestry | ND death certs | 1908–2007 | Fee |
| Ohio | FamilySearch | [2128172](https://www.familysearch.org/search/collection/2128172) | 1908–1953 | Free |
| Oklahoma | FamilySearch / Ancestry | Various | 1900s+ | Mixed |
| Oregon | Ancestry | OR death certs | 1903–1971 | Fee |
| Pennsylvania | FamilySearch | [1320976](https://www.familysearch.org/search/collection/1320976) | 1803–1915 | Free |
| Rhode Island | RI State Archives | sosri.access.preservica.com | 1853–1974 | Free |
| South Carolina | FamilySearch | SC death records | 1915–1965 | Free |
| South Dakota | FamilySearch / Ancestry | Various | 1900s+ | Mixed |
| Tennessee | FamilySearch | TN death records | 1900s+ | Free (partial) |
| Texas | FamilySearch | TX death records | 1890–1976 | Free |
| Utah | FamilySearch / UT Archives | Utah death certs | 1904–1965 | Free |
| Vermont | Ancestry | VT death certs | 1909–2008 | Fee |
| Virginia | FamilySearch | [2279481](https://www.familysearch.org/search/collection/2279481) | 1912–1987 | Free |
| Washington | FamilySearch / WA Archives | [1629663](https://www.familysearch.org/search/collection/1629663) | 1907–1960 | Free |
| Washington D.C. | FamilySearch | [1803967](https://www.familysearch.org/search/collection/1803967) | 1874–1961 | FHC only |
| West Virginia | FamilySearch | WV death records | 1900s+ | Free (partial) |
| Wisconsin | FamilySearch | WI death records | 1800s–1900s | Free (partial) |
| Wyoming | FamilySearch / Ancestry | Various | 1900s+ | Mixed |

> **Access key:** "Free" = publicly accessible online. "FHC only" = requires visiting a FamilySearch Family History Center. "Fee" = requires Ancestry.com subscription. "Mixed" = some records free, some fee-based. Coverage dates and availability are subject to change per FamilySearch's data agreements.

### Data Balance Strategy

Not all states have equal collection sizes. To ensure balanced training data across all 50 classes:
- States with large free collections (Virginia, Ohio, Michigan) will be **downsampled** to match the minimum
- States with limited free access (Alabama, D.C.) will be supplemented via **Family History Center visits** or Ancestry.com
- States with fee-based-only sources will be handled via **Ancestry institutional access** (available free through many public libraries)
- Target: minimum **500 images per state**, stratified by county and decade where possible

---

## Tech Stack

| Component | Tool |
|---|---|
| API access | `requests`, OAuth 2.0 |
| Image preprocessing | OpenCV, scikit-image |
| Layout detection | LayoutLMv3 / DocTR |
| Visual classifier | EfficientNet-B3 / ViT (`timm`) |
| OCR (handwritten) | PaddleOCR / TrOCR |
| Training framework | PyTorch + HuggingFace Transformers |
| Experiment tracking | MLflow |

---

## Project Structure

```
death-cert-cv-research/
├── data/
│   ├── raw/                  # Downloaded certificate images
│   │   ├── michigan/
│   │   │   ├── wayne/
│   │   │   └── oakland/
│   │   └── virginia/
│   │       ├── fairfax/
│   │       └── richmond/
│   └── processed/            # Preprocessed images ready for training
├── src/
│   ├── download.py           # FamilySearch API data collection pipeline
│   ├── preprocess.py         # Image preprocessing utilities
│   ├── train.py              # Model training script
│   ├── predict.py            # Inference script
│   └── ocr.py                # OCR-assisted county refinement
├── notebooks/
│   └── exploration.ipynb     # EDA and dataset inspection
├── models/                   # Saved model checkpoints
├── requirements.txt
└── README.md
```

---

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
Create a `.env` file in the project root:
```
FS_CLIENT_KEY=your_familysearch_app_key
FS_USERNAME=your_familysearch_username
FS_PASSWORD=your_familysearch_password
```

### 3. Download images
```bash
python src/download.py --collection 1968532 --state michigan --max 500
```

### 4. Preprocess images
```bash
python src/preprocess.py --input data/raw --output data/processed
```

### 5. Train the classifier
```bash
python src/train.py --data data/processed --model efficientnet_b3 --epochs 30
```

---

## Ethical Considerations & Data Use Policy

- **Read-only:** This project does not write any data back to FamilySearch systems.
- **No PII publication:** Downloaded images are used solely as training inputs and are not redistributed, published, or shared publicly.
- **Rate limiting:** All API requests are throttled to 1–2 requests per second in compliance with FamilySearch usage guidelines.
- **Non-commercial:** This is an independent academic research project. No commercial use, licensing, or monetization is planned.
- **Restricted records:** Only publicly accessible (unrestricted) collections are queried. No private family tree data or restricted records are accessed.

---

## Status

🔧 **In Development** — Data collection pipeline in progress.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

> Note: Images downloaded via the FamilySearch API remain subject to FamilySearch's own terms of service and are not redistributed by this project.

---

## Contact

For questions about this research project, please open an issue on this repository.