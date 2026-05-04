"""
src/visualize_labels.py

YOLO det 라벨 시각화 도구 (2026-04-29 신규).

Stage 1 Version A 자동 라벨 (`outputs/auto_labels/labels/*.txt`) 을
원본 이미지 (`dataset/*.jpg`) 위에 5 클래스별 색상으로 bbox 를 그려
사람이 시각 검수할 수 있는 사본을 만든다.

다양한 필터 모드 지원:
- ``--random N``     : random N 장만 시각화 (빠른 정확도 평가)
- ``--limit N``      : 처음 N 장
- ``--priority X``   : auto_labels manifest 의 priority 컬럼 필터 (low_conf 등)
- ``--classes A B``  : 특정 클래스만 (예: PMI Table)
- ``--all``          : 전체 (5,793 장 — 디스크 ~3GB 부담)

5 클래스 색상 (D-028, BGR):
::

    Isometric  : 빨강
    PMI        : 초록  ★ Stage 2 입력 영역
    Table      : 파랑
    Text       : 주황
    View       : 보라

CLI
---
::

    # 빠른 정확도 평가 — random 100 장
    python src/visualize_labels.py --random 100

    # Low conf 만 (모델이 헷갈려 하는 케이스)
    python src/visualize_labels.py --priority low_conf

    # PMI 만 시각화 (작은 박스 정확도 점검)
    python src/visualize_labels.py --classes PMI --random 50

    # 처음 5장 (테스트)
    python src/visualize_labels.py --limit 5

    # 전체 (디스크 부담)
    python src/visualize_labels.py --all

검수 후
-------
::

    explorer.exe outputs\visualized
    # Windows Explorer 큰 아이콘으로 검토
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs" / "auto_labels" / "labels"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "auto_labels" / "manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualized"

# 5 클래스 (D-028, Roboflow data.yaml 순서)
CLASS_NAMES: List[str] = ["Isometric", "PMI", "Table", "Text", "View"]

# BGR 색상 (cv2 기본)
CLASS_COLORS: List[Tuple[int, int, int]] = [
    (0, 0, 255),       # 0: Isometric — 빨강
    (0, 200, 0),       # 1: PMI       — 초록 ★ Stage 2 입력
    (255, 0, 0),       # 2: Table     — 파랑
    (0, 165, 255),     # 3: Text      — 주황
    (200, 0, 200),     # 4: View      — 보라
]

SUPPORTED_EXTS = {".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("visualize_labels")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """Unicode 경로 안전 read (Windows 한글 파일명 대응)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:  # noqa: BLE001
        log.warning("imread failed for %s: %s", path.name, e)
        return None


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """Unicode 경로 안전 write."""
    try:
        ext = path.suffix if path.suffix else ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("imwrite failed for %s: %s", path.name, e)
        return False


def parse_yolo_labels(path: Path,
                      class_filter: Optional[Set[int]] = None
                      ) -> List[Tuple[int, float, float, float, float]]:
    """YOLO det txt → list of (cls, cx, cy, w, h) (정규화).

    class_filter 가 주어지면 해당 클래스 ID 만 반환.
    """
    rows: List[Tuple[int, float, float, float, float]] = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = (float(x) for x in parts[1:])
            except ValueError:
                continue
            if class_filter is not None and cls not in class_filter:
                continue
            rows.append((cls, cx, cy, w, h))
    except Exception as e:  # noqa: BLE001
        log.warning("parse failed for %s: %s", path.name, e)
    return rows


def draw_bboxes(img: np.ndarray,
                rows: List[Tuple[int, float, float, float, float]],
                line_thickness: int = 2,
                font_scale: float = 0.5,
                show_label: bool = True,
                ) -> np.ndarray:
    """이미지 위에 bbox 그리기. 반환: 새 이미지 (원본 복사본)."""
    out = img.copy()
    h_img, w_img = img.shape[:2]

    for cls, cx, cy, w, h in rows:
        # 정규화 → 픽셀
        x1 = int((cx - w / 2) * w_img)
        y1 = int((cy - h / 2) * h_img)
        x2 = int((cx + w / 2) * w_img)
        y2 = int((cy + h / 2) * h_img)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

        color = CLASS_COLORS[cls] if 0 <= cls < len(CLASS_COLORS) else (128, 128, 128)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)

        if show_label and 0 <= cls < len(CLASS_NAMES):
            label = CLASS_NAMES[cls]
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # 라벨 배경 (text 위에 살짝)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1),
                          color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> Dict[str, Dict]:
    """auto_labels manifest.csv → {filename: {priority, n_boxes, avg_conf, ...}}"""
    out: Dict[str, Dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("filename")
            if fn:
                out[fn] = row
    return out


def filter_by_priority(label_files: List[Path],
                       manifest: Dict[str, Dict],
                       priorities: Set[str]) -> List[Path]:
    """priority 컬럼이 일치하는 라벨만 반환."""
    filtered = []
    for txt in label_files:
        # label stem → image filename 추정
        img_name = f"{txt.stem}.jpg"
        prio = manifest.get(img_name, {}).get("priority", "")
        if prio in priorities:
            filtered.append(txt)
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="YOLO det 라벨 시각화 도구 — bbox 그려진 사본 생성",
    )
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR,
                   help=f"라벨 폴더 (default: {DEFAULT_LABELS_DIR})")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                   help=f"원본 이미지 폴더 (default: {DEFAULT_DATASET_DIR})")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                   help=f"auto_labels manifest (priority filter 시) (default: {DEFAULT_MANIFEST})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"출력 폴더 (default: {DEFAULT_OUTPUT_DIR})")

    # Filter options (mutually exclusive in spirit)
    p.add_argument("--all", action="store_true",
                   help="전체 시각화 (5,793장 — 디스크 ~3GB 부담)")
    p.add_argument("--limit", type=int, default=0,
                   help="처음 N장만 (테스트용)")
    p.add_argument("--random", type=int, default=0, dest="random_n",
                   help="random N장만 (정확도 일반 평가)")
    p.add_argument("--priority", nargs="+", default=None,
                   choices=["empty", "low_conf", "review", "auto_pass", "high"],
                   help="auto_labels manifest priority 필터 (예: low_conf empty)")
    p.add_argument("--classes", nargs="+", default=None,
                   choices=CLASS_NAMES,
                   help=f"특정 클래스만 그림 (default: 전체)")
    p.add_argument("--seed", type=int, default=42,
                   help="random 시 시드 (default: 42)")

    # Drawing options
    p.add_argument("--line", type=int, default=2,
                   help="bbox 선 두께 (default: 2)")
    p.add_argument("--font", type=float, default=0.5,
                   help="라벨 텍스트 크기 (default: 0.5)")
    p.add_argument("--no-label", action="store_true",
                   help="클래스 이름 텍스트 숨김 (bbox 만)")
    args = p.parse_args()

    # --- Validate ----
    if not args.labels_dir.exists():
        log.error("Labels dir not found: %s", args.labels_dir)
        log.error("auto_label_stage1.py 를 먼저 실행 필요")
        return 2
    if not args.dataset_dir.exists():
        log.error("Dataset dir not found: %s", args.dataset_dir)
        return 2

    # --- Class filter (id set) ----
    class_filter: Optional[Set[int]] = None
    if args.classes:
        class_filter = set(CLASS_NAMES.index(c) for c in args.classes)
        log.info("Class filter: %s (ids %s)", args.classes, sorted(class_filter))

    # --- 라벨 파일 스캔 ----
    label_files = sorted(args.labels_dir.glob("*.txt"))
    log.info("Found %d label files", len(label_files))

    # --- Filter 적용 ----
    if args.priority:
        manifest = load_manifest(args.manifest)
        if not manifest:
            log.error("Manifest not loaded (필요 시): %s", args.manifest)
            return 2
        before = len(label_files)
        label_files = filter_by_priority(label_files, manifest,
                                          set(args.priority))
        log.info("Priority filter %s: %d → %d", args.priority, before, len(label_files))

    # --- 필터 검증 (적어도 하나 필수) ----
    no_filter_applied = (
        args.random_n <= 0
        and args.limit <= 0
        and not args.all
        and not args.priority
        and not args.classes
    )
    if no_filter_applied:
        log.error("필터 옵션 필요: --random N / --limit N / --priority X / --classes A / --all")
        log.error("예: python src/visualize_labels.py --random 100")
        return 1

    # --- Sampling (priority/classes 필터 후 추가 sampling 가능) ----
    if args.random_n > 0:
        random.seed(args.seed)
        if args.random_n < len(label_files):
            label_files = random.sample(label_files, args.random_n)
        log.info("Random sample: %d (seed=%d)", len(label_files), args.seed)
    elif args.limit > 0:
        label_files = label_files[: args.limit]
        log.info("Limit: %d", len(label_files))
    # else: priority/classes 필터만 (또는 --all) → 그대로 진행

    if not label_files:
        log.warning("No labels match filter. Exit.")
        return 0

    log.info("Visualizing %d images → %s", len(label_files), args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    # --- 처리 ----
    n_ok = 0
    n_missing_img = 0
    n_no_bbox = 0
    n_err = 0

    pbar = tqdm(label_files, desc="Drawing bboxes", unit="img",
                dynamic_ncols=True, leave=True)
    for txt in pbar:
        # 라벨 파싱
        rows = parse_yolo_labels(txt, class_filter=class_filter)
        if not rows and class_filter:
            # 필터 적용 후 박스 없음 → skip
            n_no_bbox += 1
            continue

        # 이미지 매칭
        img_path = args.dataset_dir / f"{txt.stem}.jpg"
        if not img_path.exists():
            img_path = args.dataset_dir / f"{txt.stem}.jpeg"
        if not img_path.exists():
            n_missing_img += 1
            continue

        img = imread_unicode(img_path)
        if img is None:
            n_err += 1
            continue

        # bbox 그림
        out = draw_bboxes(img, rows,
                          line_thickness=args.line,
                          font_scale=args.font,
                          show_label=not args.no_label)

        # 저장
        out_path = args.output / img_path.name
        if imwrite_unicode(out_path, out):
            n_ok += 1
        else:
            n_err += 1

        pbar.set_postfix(ok=n_ok, missing=n_missing_img, err=n_err)

    # --- Summary ----
    log.info("=" * 60)
    log.info("Visualize complete: %d images", n_ok)
    log.info("  Missing image: %d", n_missing_img)
    log.info("  No bbox after filter: %d", n_no_bbox)
    log.info("  Errors: %d", n_err)
    log.info("=" * 60)
    log.info("Output: %s", args.output)
    log.info("")
    log.info("[다음 단계] Windows Explorer 큰 아이콘 보기:")
    log.info("  explorer.exe %s", args.output)
    log.info("")
    log.info("[색상 범례]")
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        bgr = f"BGR{color}"
        log.info("  %d  %-10s  %s", i, name, bgr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
