"""
src/sort_by_drawing_type.py

D-026 — 가공도면 / 조립도면 자동 분류 (휴리스틱).

배경
----
Roboflow 데이터셋 4,587 장에 **가공도면 + 조립도면 혼재**.
- 가공도면 (manufacturing): 치수·표면거칠기·GD&T 다수 → Stage 1·2·3 모두 학습 가능
- 조립도면 (assembly):     부품번호 풍선·BOM 표 → Stage 1 만, Stage 2 무용

본 모듈은 학습 없이 OpenCV + Pytesseract 휴리스틱으로 분류.

분류 시그널
-----------
**Manufacturing positive (가공도면 시그널)**
- 치수 표기 빈도 — Ø, R, M, ±, mm 단위, 숫자+단위 패턴
- 표면거칠기 심볼 — Ra/Rz 키워드
- GD&T 프레임 — 사각 + 컴파트먼트 (옵션)

**Assembly positive (조립도면 시그널)**
- 부품번호 풍선 — Hough Circles (작은 원, 보통 30~80 px)
- BOM 표 — 우상단 / 우하단의 큰 격자 표
- 다수 풍선 (≥ 10개)

판정 룰 (기본)
--------------
- BOM 표 검출 OR (풍선 ≥ 10 AND 치수 < 5) → ``assembly``
- 치수 ≥ 5 AND 풍선 < 5                    → ``manufacturing``
- 그 외                                     → ``manual_review_type``

CLI
---
::

    # dryrun (이동 없음, manifest 만 생성)
    python src/sort_by_drawing_type.py --dryrun

    # 실제 분류 + 이동
    python src/sort_by_drawing_type.py

    # 임계값 조정
    python src/sort_by_drawing_type.py \\
        --dim-min 5 --balloon-asm 10
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / constants (D-024, D-025, D-026)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DST_MFG    = DATA_DIR / "manufacturing"
DST_ASM    = DATA_DIR / "assembly"
DST_REVIEW = DATA_DIR / "manual_review_type"
MANIFEST_PATH = OUTPUT_DIR / "sort_drawing_type_manifest.csv"

# 5-language OCR (D-025)
OCR_LANGS = "kor+eng+rus+jpn+chi_sim+chi_tra"

# Windows fallback (Linux/WSL2 uses PATH)
WIN_TESSERACT_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SUPPORTED_EXTS = {".jpg", ".jpeg"}

# Default thresholds (CLI override 가능)
DEFAULT_DIM_MIN_FOR_MFG    = 5      # 가공도면 판정용 최소 치수 표기 수
DEFAULT_BALLOON_MIN_FOR_ASM = 10    # 조립도면 판정용 최소 풍선 수
DEFAULT_BALLOON_MAX_FOR_MFG = 5     # 가공도면 판정 시 풍선 상한
DEFAULT_DIM_MAX_FOR_ASM    = 5      # 조립도면 판정 시 치수 상한 (BOM 없을 때)

# ---------------------------------------------------------------------------
# Multilingual BOM / quantity keywords (D-025 5 languages)
# ---------------------------------------------------------------------------
BOM_KEYWORDS = {
    # English
    "PART NO", "PART NUMBER", "ITEM", "QTY", "QUANTITY", "B.O.M",
    "BILL OF MATERIAL", "BILL OF MATERIALS",
    # Korean
    "품번", "품명", "수량", "부품번호", "자재명",
    # Japanese
    "部品番号", "品番", "数量", "員数", "名称",
    # Russian
    "ПОЗ", "КОЛ", "НАИМЕНОВАНИЕ", "ОБОЗН",
    # Chinese (簡/繁)
    "序号", "序號", "数量", "數量", "名称", "名稱", "零件号", "零件號",
    "件数", "件數", "材料", "規格",
}

# ---------------------------------------------------------------------------
# Dimension regex patterns (universal symbols, language-agnostic)
# ---------------------------------------------------------------------------
DIM_PATTERNS = [
    re.compile(r"[Ø⌀φ]\s*\d+(?:\.\d+)?"),       # Ø25.4, ⌀12, φ8
    re.compile(r"\bR\s*\d+(?:\.\d+)?"),          # R0.5, R2
    re.compile(r"\bM\s*\d+(?:\.\d+)?"),          # M8, M12x1.25
    re.compile(r"±\s*\d+(?:\.\d+)?"),            # ±0.05
    re.compile(r"\d+(?:\.\d+)?\s*[°]"),          # 90°
    re.compile(r"\d+(?:\.\d+)?\s*(?:mm|MM|μm|um)"),  # 25mm
    re.compile(r"\d+(?:\.\d+)?\s*[+\-]\s*0?\.\d+"),  # 25 +0.05 -0.05
    re.compile(r"\d+\s*[xX×]\s*[Ø⌀]"),            # 4xØ6
]

ROUGHNESS_PATTERNS = [
    re.compile(r"\bRa\s*\d+(?:\.\d+)?"),         # Ra1.6
    re.compile(r"\bRz\s*\d+(?:\.\d+)?"),         # Rz6.3
    re.compile(r"\bRmax\s*\d+(?:\.\d+)?"),       # Rmax
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sort_by_drawing_type")


# ---------------------------------------------------------------------------
# I/O helpers (Unicode-safe; multilingual filenames per D-013/D-025)
# ---------------------------------------------------------------------------
def setup_tesseract(custom_path: Optional[str] = None) -> None:
    try:
        import pytesseract  # noqa: PLC0415
    except ImportError:
        return
    if custom_path:
        pytesseract.pytesseract.tesseract_cmd = custom_path
        return
    if sys.platform.startswith("win"):
        default = Path(WIN_TESSERACT_DEFAULT)
        if default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default)


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:  # noqa: BLE001
        log.error("imread failed for %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def ocr_full_image(bgr: np.ndarray, langs: str = OCR_LANGS) -> str:
    """Run multilingual OCR on the whole drawing. Returns concatenated text."""
    try:
        import pytesseract  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return ""
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return pytesseract.image_to_string(pil, lang=langs, config="--psm 6")
    except Exception as e:  # noqa: BLE001
        log.debug("OCR failed: %s", e)
        return ""


def count_dimensions(text: str) -> int:
    """Count dimension-pattern matches in OCR text."""
    if not text:
        return 0
    total = 0
    for pat in DIM_PATTERNS:
        total += len(pat.findall(text))
    return total


def detect_roughness(text: str) -> int:
    """Count surface-roughness pattern matches."""
    if not text:
        return 0
    total = 0
    for pat in ROUGHNESS_PATTERNS:
        total += len(pat.findall(text))
    return total


def detect_bom_keywords(text: str) -> int:
    """Count BOM-related keyword hits across 5 languages."""
    if not text:
        return 0
    norm = " ".join(text.upper().split())
    hits = 0
    for kw in BOM_KEYWORDS:
        if kw.upper() in norm:
            hits += 1
    return hits


def detect_balloons(bgr: np.ndarray) -> int:
    """Count small numbered circles (Hough Circles) — assembly part-number balloons.

    Tuning notes
    ------------
    Engineering drawing balloons are typically:
      - 30~80 px diameter (depends on resolution)
      - Solid circular line (high contrast)
      - Contain 1~3 digit numbers inside
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # Adaptive radius based on image size (1500 px image → ~20-50 px radius)
    short_side = min(h, w)
    min_r = max(8, int(short_side * 0.008))
    max_r = max(min_r + 10, int(short_side * 0.030))

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(short_side * 0.020),
        param1=80, param2=35,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return 0
    return int(circles.shape[1])


def detect_bom_table(bgr: np.ndarray, ocr_text: str = "") -> bool:
    """Heuristic BOM table detection.

    Strategy
    --------
    1. Strong signal: BOM keyword hits ≥ 2 (multilingual)
    2. Geometric signal: large rectangle in upper-right OR upper-left
       with ≥ 5 horizontal divider lines (table rows)
    """
    # 1) Keyword signal
    if detect_bom_keywords(ocr_text) >= 2:
        return True

    # 2) Geometric signal
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)

    # Detect long horizontal lines
    min_len = int(w * 0.10)   # at least 10% of width
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=min_len, maxLineGap=8,
    )
    if lines is None:
        return False

    # Filter: horizontal lines (|dy| ≤ 3) in upper half (y < h * 0.55)
    horizontal_top = [
        (x1, y1, x2, y2) for x1, y1, x2, y2 in lines[:, 0, :]
        if abs(int(y2) - int(y1)) <= 3
        and y1 < h * 0.55
        and abs(int(x2) - int(x1)) >= min_len
    ]
    # Group by approximate y to avoid double-counting same line
    rows_seen: List[int] = []
    for _, y1, _, _ in horizontal_top:
        if not any(abs(y1 - y0) < 8 for y0 in rows_seen):
            rows_seen.append(int(y1))

    return len(rows_seen) >= 5


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify(dim_count: int,
             roughness_count: int,
             balloon_count: int,
             bom_detected: bool,
             dim_min_mfg: int,
             balloon_min_asm: int,
             balloon_max_mfg: int,
             dim_max_asm: int) -> Tuple[str, str]:
    """Return (decision, reason)."""
    # Strong assembly: BOM detected
    if bom_detected:
        return "assembly", "BOM table detected"
    # Strong assembly: many balloons + few dimensions
    if balloon_count >= balloon_min_asm and dim_count < dim_max_asm:
        return "assembly", (f"balloons={balloon_count}≥{balloon_min_asm} "
                            f"AND dims={dim_count}<{dim_max_asm}")
    # Manufacturing: many dimensions + few balloons
    if dim_count >= dim_min_mfg and balloon_count < balloon_max_mfg:
        return "manufacturing", (f"dims={dim_count}≥{dim_min_mfg} "
                                 f"AND balloons={balloon_count}<{balloon_max_mfg}")
    # Otherwise: ambiguous
    return "manual_review_type", (f"dims={dim_count}, balloons={balloon_count}, "
                                  f"BOM={bom_detected}")


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------
def process_one(img_path: Path,
                args: argparse.Namespace) -> Dict[str, str]:
    record = {
        "filename": img_path.name,
        "dim_count": 0,
        "roughness_count": 0,
        "balloon_count": 0,
        "bom_detected": "False",
        "decision": "error",
        "reason": "",
        "src_path": str(img_path),
        "dst_path": "",
    }
    bgr = imread_unicode(img_path)
    if bgr is None:
        record["reason"] = "imread_failed"
        return record

    text = ocr_full_image(bgr) if not args.no_ocr else ""

    dim_count       = count_dimensions(text)
    rough_count     = detect_roughness(text)
    balloon_count   = detect_balloons(bgr)
    bom_detected    = detect_bom_table(bgr, ocr_text=text)

    decision, reason = classify(
        dim_count=dim_count,
        roughness_count=rough_count,
        balloon_count=balloon_count,
        bom_detected=bom_detected,
        dim_min_mfg=args.dim_min,
        balloon_min_asm=args.balloon_asm,
        balloon_max_mfg=args.balloon_mfg_max,
        dim_max_asm=args.dim_asm_max,
    )

    record.update({
        "dim_count": dim_count,
        "roughness_count": rough_count,
        "balloon_count": balloon_count,
        "bom_detected": str(bom_detected),
        "decision": decision,
        "reason": reason,
    })

    if args.dryrun:
        return record

    dst_dir = {
        "manufacturing": DST_MFG,
        "assembly": DST_ASM,
        "manual_review_type": DST_REVIEW,
    }.get(decision, DST_REVIEW)
    dst_dir.mkdir(parents=True, exist_ok=True)

    target = dst_dir / img_path.name
    n = 1
    while target.exists():
        target = dst_dir / f"{img_path.stem}_{n}{img_path.suffix}"
        n += 1

    if args.copy:
        shutil.copy2(img_path, target)
    else:
        shutil.move(str(img_path), str(target))
    record["dst_path"] = str(target)
    return record


# ---------------------------------------------------------------------------
# Main I/O
# ---------------------------------------------------------------------------
def write_manifest(records: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename", "dim_count", "roughness_count", "balloon_count",
        "bom_detected", "decision", "reason", "src_path", "dst_path",
    ]
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
        description="D-026 — 가공/조립 도면 자동 분류 (휴리스틱).",
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR,
                   help="입력 폴더 (default: ./dataset)")
    p.add_argument("--dryrun", action="store_true",
                   help="이동 X, manifest 만 생성")
    p.add_argument("--copy", action="store_true",
                   help="이동 대신 복사")
    p.add_argument("--no-ocr", action="store_true",
                   help="OCR 비활성화 (속도 ↑, 정확도 ↓)")
    p.add_argument("--dim-min", type=int, default=DEFAULT_DIM_MIN_FOR_MFG,
                   help=f"가공도면 판정 최소 치수 수 (default: %(default)d)")
    p.add_argument("--dim-asm-max", type=int, default=DEFAULT_DIM_MAX_FOR_ASM,
                   help=f"조립도면 판정 시 치수 상한 (default: %(default)d)")
    p.add_argument("--balloon-asm", type=int, default=DEFAULT_BALLOON_MIN_FOR_ASM,
                   help=f"조립도면 판정 최소 풍선 수 (default: %(default)d)")
    p.add_argument("--balloon-mfg-max", type=int, default=DEFAULT_BALLOON_MAX_FOR_MFG,
                   help=f"가공도면 판정 시 풍선 상한 (default: %(default)d)")
    p.add_argument("--tesseract", type=str, default=None,
                   help="(Windows) tesseract.exe 경로")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_tesseract(args.tesseract)

    if not args.input.exists():
        log.error("Input dir not found: %s", args.input)
        return 2

    images = collect_images(args.input)
    if not images:
        log.warning("No JPG/JPEG under %s", args.input)
        return 0

    log.info("Found %d images. dryrun=%s copy=%s no-ocr=%s",
             len(images), args.dryrun, args.copy, args.no_ocr)
    log.info("Thresholds: dim_min=%d, dim_asm_max=%d, "
             "balloon_asm=%d, balloon_mfg_max=%d",
             args.dim_min, args.dim_asm_max,
             args.balloon_asm, args.balloon_mfg_max)
    log.info("OCR languages: %s", OCR_LANGS if not args.no_ocr else "(disabled)")

    if not args.dryrun:
        for d in (DST_MFG, DST_ASM, DST_REVIEW):
            d.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, str]] = []
    counts = {"manufacturing": 0, "assembly": 0,
              "manual_review_type": 0, "error": 0}

    # tqdm progress bar with live classification stats (set_postfix).
    # 25장마다 INFO 로그도 보존 (CI/log 친화).
    pbar = tqdm(images, desc="Sorting drawings", unit="img",
                dynamic_ncols=True, leave=True)
    for i, img in enumerate(pbar, 1):
        rec = process_one(img, args)
        counts[rec["decision"]] = counts.get(rec["decision"], 0) + 1
        records.append(rec)

        # 진행바 우측에 실시간 분류 통계 표시
        pbar.set_postfix(
            mfg=counts["manufacturing"],
            asm=counts["assembly"],
            review=counts["manual_review_type"],
            err=counts.get("error", 0),
        )

        # 25장마다 또는 마지막에 상세 로그 (스크롤되어 history 남음)
        if i % 25 == 0 or i == len(images):
            log.info("[%d/%d] %s → %s (dims=%s balloons=%s bom=%s)",
                     i, len(images), img.name,
                     rec["decision"], rec["dim_count"],
                     rec["balloon_count"], rec["bom_detected"])

    write_manifest(records, MANIFEST_PATH)
    log.info("Manifest: %s", MANIFEST_PATH)
    log.info("Summary: mfg=%d  asm=%d  review=%d  error=%d",
             counts.get("manufacturing", 0),
             counts.get("assembly", 0),
             counts.get("manual_review_type", 0),
             counts.get("error", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
