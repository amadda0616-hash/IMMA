"""
src/sort_by_titleblock.py

Multi-lingual (KO / EN / RU / JP) engineering drawing sorter.
Classifies JPGs into Stage 1 / Stage 2 datasets based on TitleBlock presence.

Detection logic
---------------
1. Crop the bottom ~35% region of the drawing (typical TitleBlock area).
2. Run Tesseract OCR with combined ``lang='kor+eng+rus+jpn'``.
3. Match the OCR text against a multilingual TitleBlock keyword dictionary
   (case / whitespace insensitive substring match).
4. Compute auxiliary line density (Canny + HoughLinesP) in the same region.
5. Apply the rule:

   ============================================  ==========================
   condition                                     decision
   ============================================  ==========================
   keyword_hits >= keyword_threshold (default 2) stage1_titleblock
   keyword_hits == 0 AND line_density < thr      stage2_no_titleblock
   else                                          manual_review
   ============================================  ==========================

Output
------
- Move classified files to::

      data/stage1_titleblock/
      data/stage2_no_titleblock/
      data/manual_review/

- Write a manifest CSV to ``outputs/sort_titleblock_manifest.csv``.

CLI
---
::

    python src/sort_by_titleblock.py
    python src/sort_by_titleblock.py --input <path> --dryrun
    python src/sort_by_titleblock.py --keyword-threshold 2
    python src/sort_by_titleblock.py --tesseract "C:/Program Files/Tesseract-OCR/tesseract.exe"
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DST_STAGE1 = DATA_DIR / "stage1_titleblock"
DST_STAGE2 = DATA_DIR / "stage2_no_titleblock"
DST_MANUAL = DATA_DIR / "manual_review"
MANIFEST_PATH = OUTPUT_DIR / "sort_titleblock_manifest.csv"

OCR_LANGS = "kor+eng+rus+jpn"
BOTTOM_CROP_RATIO = 0.35
LINE_DENSITY_THRESHOLD = 0.0008  # heuristic; tune via dryrun + manifest review
SUPPORTED_EXTS = {".jpg", ".jpeg"}

# Tesseract default install path on Windows (UB Mannheim build).
WIN_TESSERACT_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------------------------------
# Multilingual TitleBlock keyword dictionary
# ---------------------------------------------------------------------------
KEYWORDS = {
    "en": [
        "TITLE", "DRAWING", "DWG", "DWG NO", "DRAWING NO",
        "SCALE", "MATERIAL", "REV", "REVISION", "DATE",
        "DRAWN", "DRAWN BY", "CHECKED", "CHECKED BY",
        "APPROVED", "SHEET", "PART NO", "PART NUMBER", "PROJECT",
    ],
    "ko": [
        "도번", "도면", "도면번호", "척도", "재질", "개정",
        "날짜", "일자", "작성", "작성자", "검도", "검도자",
        "승인", "시트", "부품번호", "품번", "제목", "도명",
    ],
    "ru": [
        "ЧЕРТЕЖ", "МАСШТАБ", "МАТЕРИАЛ", "ИЗМ", "ДАТА",
        "ЛИСТ", "РАЗРАБ", "ПРОВ", "УТВ",
        "НАИМЕНОВАНИЕ", "ОБОЗНАЧЕНИЕ", "ИЗМЕНЕНИЕ",
    ],
    "jp": [
        "図面", "図番", "縮尺", "材質", "改訂", "日付",
        "作成", "検図", "承認", "シート",
        "部品番号", "題目", "タイトル", "品番",
    ],
}
ALL_KEYWORDS: List[Tuple[str, str]] = [
    (lang, kw) for lang, kws in KEYWORDS.items() for kw in kws
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sort_by_titleblock")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_tesseract(custom_path: str | None) -> None:
    """Configure ``pytesseract.tesseract_cmd`` for Windows defaults."""
    if custom_path:
        pytesseract.pytesseract.tesseract_cmd = custom_path
        return
    if sys.platform.startswith("win"):
        default = Path(WIN_TESSERACT_DEFAULT)
        if default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default)


def imread_unicode(path: Path) -> np.ndarray | None:
    """``cv2.imread`` replacement that tolerates non-ASCII (KO/JP/RU) paths
    on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:  # noqa: BLE001
        log.error("imread failed for %s: %s", path, e)
        return None


def crop_bottom(img: np.ndarray, ratio: float = BOTTOM_CROP_RATIO) -> np.ndarray:
    """Return the bottom ``ratio`` fraction of the image."""
    h = img.shape[0]
    cut = int(h * (1.0 - ratio))
    return img[cut:, :]


def normalize_text(s: str) -> str:
    """Uppercase + remove all whitespace for robust keyword matching."""
    return "".join(s.upper().split())


def count_keywords(ocr_text: str) -> Tuple[int, List[str]]:
    """Count multilingual TitleBlock keyword hits.

    Returns ``(unique_hits, ['en:TITLE', 'ko:도번', ...])``.
    """
    norm = normalize_text(ocr_text)
    hits: List[str] = []
    seen = set()
    for lang, kw in ALL_KEYWORDS:
        nkw = normalize_text(kw)
        if not nkw:
            continue
        key = (lang, nkw)
        if nkw in norm and key not in seen:
            hits.append(f"{lang}:{kw}")
            seen.add(key)
    return len(hits), hits


def compute_line_density(gray_bottom: np.ndarray) -> float:
    """Approximate horizontal+vertical line density in the bottom crop.

    A dense TitleBlock grid typically yields a noticeably higher value than a
    clean View region. Returned as ``total_line_pixels / area``.
    """
    if gray_bottom.size == 0:
        return 0.0
    blurred = cv2.GaussianBlur(gray_bottom, (3, 3), 0)
    edges = cv2.Canny(blurred, 60, 180)
    h, w = edges.shape
    min_len = max(40, int(0.10 * w))
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=min_len, maxLineGap=5,
    )
    if lines is None:
        return 0.0
    total_len = 0.0
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
        # near-horizontal or near-vertical only (TitleBlock grid)
        if dy <= 3 or dx <= 3:
            total_len += float(np.hypot(dx, dy))
    area = float(h * w)
    return total_len / area if area > 0 else 0.0


def ocr_bottom_region(bgr: np.ndarray) -> str:
    """OCR the bottom crop with combined CJK + Cyrillic + Latin languages."""
    bottom = crop_bottom(bgr, BOTTOM_CROP_RATIO)
    if bottom.size == 0:
        return ""
    rgb = cv2.cvtColor(bottom, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    try:
        # PSM 6 == "Assume a single uniform block of text" — works well for
        # the rectangular TitleBlock area.
        return pytesseract.image_to_string(pil, lang=OCR_LANGS, config="--psm 6")
    except pytesseract.TesseractError as e:
        log.warning("Tesseract error: %s", e)
        return ""


def decide(keyword_hits: int,
           line_density: float,
           keyword_threshold: int,
           density_threshold: float) -> str:
    if keyword_hits >= keyword_threshold:
        return "stage1_titleblock"
    if keyword_hits == 0 and line_density < density_threshold:
        return "stage2_no_titleblock"
    return "manual_review"


def safe_move(src: Path, dst_dir: Path) -> Path:
    """Move ``src`` into ``dst_dir`` while avoiding name collisions."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    n = 1
    while target.exists():
        target = dst_dir / f"{src.stem}_{n}{src.suffix}"
        n += 1
    shutil.move(str(src), str(target))
    return target


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------
def process_one(img_path: Path,
                keyword_threshold: int,
                density_threshold: float,
                dryrun: bool) -> dict:
    """Classify and (optionally) move one image. Returns a manifest record."""
    record = {
        "filename": img_path.name,
        "keyword_hits": 0,
        "matched_keywords": "",
        "line_density": 0.0,
        "decision": "error",
        "src_path": str(img_path),
        "dst_path": "",
        "note": "",
    }

    bgr = imread_unicode(img_path)
    if bgr is None:
        record["note"] = "imread_failed"
        return record

    bottom = crop_bottom(bgr)
    gray_bottom = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    line_density = compute_line_density(gray_bottom)

    ocr_text = ocr_bottom_region(bgr)
    hits, matched = count_keywords(ocr_text)

    decision = decide(hits, line_density, keyword_threshold, density_threshold)

    record.update({
        "keyword_hits": hits,
        "matched_keywords": ";".join(matched),
        "line_density": round(line_density, 6),
        "decision": decision,
    })

    if dryrun:
        return record

    dst_dir = {
        "stage1_titleblock": DST_STAGE1,
        "stage2_no_titleblock": DST_STAGE2,
        "manual_review": DST_MANUAL,
    }.get(decision, DST_MANUAL)
    moved = safe_move(img_path, dst_dir)
    record["dst_path"] = str(moved)
    return record


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def write_manifest(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename", "keyword_hits", "matched_keywords",
        "line_density", "decision", "src_path", "dst_path", "note",
    ]
    # utf-8-sig so Excel on Windows displays CJK / Cyrillic correctly.
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(r)


def collect_images(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify drawings into Stage1/Stage2 datasets via "
                    "TitleBlock detection (multilingual: KO/EN/RU/JP).",
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR,
                   help="Source folder of JPGs (default: ./dataset).")
    p.add_argument("--keyword-threshold", type=int, default=2,
                   help="Min keyword hits to classify as stage1 (default: 2).")
    p.add_argument("--density-threshold", type=float,
                   default=LINE_DENSITY_THRESHOLD,
                   help="Max line density to classify as stage2 when "
                        "keyword_hits == 0 (default: %(default)s).")
    p.add_argument("--dryrun", action="store_true",
                   help="Do not move files; only write manifest.")
    p.add_argument("--tesseract", type=str, default=None,
                   help="Path to tesseract.exe (optional override).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_tesseract(args.tesseract)

    input_dir: Path = args.input
    if not input_dir.exists():
        log.error("Input directory not found: %s", input_dir)
        return 2

    images = collect_images(input_dir)
    if not images:
        log.warning("No JPG/JPEG images found under %s", input_dir)
        return 0

    log.info("Found %d images. dryrun=%s", len(images), args.dryrun)
    log.info("Tesseract cmd: %s",
             pytesseract.pytesseract.tesseract_cmd or "(PATH lookup)")
    log.info("OCR languages: %s", OCR_LANGS)

    if not args.dryrun:
        for d in (DST_STAGE1, DST_STAGE2, DST_MANUAL):
            d.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    counts = {"stage1_titleblock": 0,
              "stage2_no_titleblock": 0,
              "manual_review": 0,
              "error": 0}

    for i, img in enumerate(images, 1):
        rec = process_one(
            img,
            keyword_threshold=args.keyword_threshold,
            density_threshold=args.density_threshold,
            dryrun=args.dryrun,
        )
        counts[rec["decision"]] = counts.get(rec["decision"], 0) + 1
        records.append(rec)
        if i % 25 == 0 or i == len(images):
            log.info(
                "[%d/%d] %s -> %s (hits=%d, ld=%.5f)",
                i, len(images), img.name,
                rec["decision"], rec["keyword_hits"], rec["line_density"],
            )

    write_manifest(records, MANIFEST_PATH)

    log.info("Manifest written: %s", MANIFEST_PATH)
    log.info(
        "Summary:  stage1=%d  stage2=%d  manual=%d  error=%d",
        counts.get("stage1_titleblock", 0),
        counts.get("stage2_no_titleblock", 0),
        counts.get("manual_review", 0),
        counts.get("error", 0),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
