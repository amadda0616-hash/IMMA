"""
src/extract_pmi_crops_v3.py

Stage 1 Version A → PMI crop 추출 (v3: aspect-aware adaptive padding + optional upscale).

배경
----
- v1 (extract_pmi_crops.py, fixed 10px): 화살표 잘림
- v2 (extract_pmi_crops.py, per-axis adaptive): 90% 비회전 / 80% 회전
  → 회전 잘림 문제의 80% = 45° 회전 텍스트 (정사각형 bbox)
  → axis-aligned per-axis 계산은 대각선 화살표 캡처 한계
- v3 (이 파일, aspect-aware): 정사각형 bbox 에 uniform 큰 padding 적용
  → 45° 회전 텍스트의 대각선 화살표/리더선 보강
  → 가로/세로형은 v2 의 per-axis 유지 (인접 치수 침입 최소화)

- v3 확장 (upscale 옵션): crop 저장 직전 균등 업스케일
  → 원본 도면 해상도 부족으로 PMI 텍스트 가독성 낮은 문제 보완
  → --upscale 3.0 권장 (텍스트 3배 확대, Lanczos 보간 부드러움)
  → 정보량 추가 X (보간) but CVAT 라벨링 효율 ↑
  → 균등 스케일 → 좌표 마이그레이션 가능 (migrate_cvat_labels.py)
  → optional --sharpen (unsharp mask 로 텍스트 윤곽 선명화)

핵심 로직
--------
::

    aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1.0)

    if aspect < aspect_threshold (default 1.5):
        # 정사각형 추정 (45° 회전 가능성)
        long_side = max(bbox_w, bbox_h)
        pad = clamp(long_side × ratio_square (default 0.6), [min, max])
        pad_x = pad_y = pad
        strategy = "square_diagonal"
    else:
        # 가로/세로형 → v2 per-axis 동일
        pad_x = clamp(bbox_w × ratio (default 0.4), [min, max])
        pad_y = clamp(bbox_h × ratio (default 0.4), [min, max])
        strategy = "per_axis"

산출물
------
::

    outputs/cvat_stage2_input_v3/      ← v3 (aspect-aware, D-037 확장)
    ├── DwgFoo__PMI_000.jpg
    ├── ...
    └── manifest.csv                    ← + aspect_ratio, padding_strategy 컬럼

CLI
---
::

    # ★ 권장 (default — v3 default 폴더)
    python src/extract_pmi_crops_v3.py

    # ★ Upscale 적용 — 텍스트 가독성 부족 시 권장 (3배 확대)
    python src/extract_pmi_crops_v3.py \
        --output outputs/cvat_stage2_input_v3_upscaled \
        --upscale 3.0

    # Upscale + Sharpen — 텍스트 윤곽 선명화 추가
    python src/extract_pmi_crops_v3.py \
        --output outputs/cvat_stage2_input_v3_upscaled \
        --upscale 3.0 \
        --sharpen

    # 정사각형 padding 비율 튜닝
    python src/extract_pmi_crops_v3.py --padding-ratio-square 0.5

    # aspect threshold 튜닝 (1.5 → 1.3 으로 더 엄격하게)
    python src/extract_pmi_crops_v3.py --aspect-threshold 1.3

    # v2 와 비교용 다른 폴더로 출력
    python src/extract_pmi_crops_v3.py --output outputs/cvat_stage2_input_v3_test

비교 실험
--------
v2 (extract_pmi_crops.py) — per-axis only::

    python src/extract_pmi_crops.py    # → outputs/cvat_stage2_input_v2/

v3 (이 파일) — aspect-aware::

    python src/extract_pmi_crops_v3.py # → outputs/cvat_stage2_input_v3/

관련 의사결정
-------------
- D-024 group_key 보존 (manifest 기록)
- D-028 5 클래스 (PMI = Roboflow cls 1)
- D-029 매핑 (PMI 는 매핑 X)
- D-034 hierarchical (PMI = Stage 2 입력)
- D-036 옵션 B 정책 (auto_pass + review priority 만)
- D-037 adaptive padding (v2 per-axis, v3 aspect-aware 확장)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "checkpoints" / "yolo_det.pt"
DEFAULT_DRAWINGS_LIST = PROJECT_ROOT / "outputs" / "stage2_input_drawings.txt"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cvat_stage2_input_v3"  # D-037 v3

# Stage 1 5 클래스 (D-028, Roboflow data.yaml 순서)
CLS_ISOMETRIC = 0
CLS_PMI = 1         # ★ Stage 2 입력 (D-034)
CLS_TABLE = 2
CLS_TEXT = 3
CLS_VIEW = 4

SUPPORTED_EXTS = {".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_pmi_crops_v3")


# ---------------------------------------------------------------------------
# Unicode-safe IO (Windows 한글 파일명 대응)
# ---------------------------------------------------------------------------
def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:  # noqa: BLE001
        log.warning("imread failed for %s: %s", path.name, e)
        return None


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
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


# ---------------------------------------------------------------------------
# Drawing list loader
# ---------------------------------------------------------------------------
def load_drawings(args) -> List[Path]:
    """입력 옵션에 따라 도면 list 반환."""
    drawings: List[Path] = []

    if args.input is not None:
        if not args.input.exists():
            log.error("Input dir not found: %s", args.input)
            return []
        for p in args.input.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                drawings.append(p)
        log.info("Direct folder %s: %d images", args.input, len(drawings))
    elif args.drawings.exists():
        for line in args.drawings.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            full = args.dataset_dir / name
            if full.exists():
                drawings.append(full)
            else:
                log.warning("File not found: %s", full)
        log.info("Drawings list %s: %d images", args.drawings, len(drawings))
    else:
        log.error("입력 옵션 필요: --input <폴더> 또는 --drawings <list.txt>")
        return []

    return sorted(drawings)


# ---------------------------------------------------------------------------
# Padding 계산 (v3: aspect-aware)
# ---------------------------------------------------------------------------
def calc_padding_v3(bbox_w: float, bbox_h: float, args):
    """
    v3 aspect-aware padding 계산.

    Returns
    -------
    pad_x, pad_y, aspect_ratio, strategy
        - strategy: "square_diagonal" | "per_axis" | "fixed"
    """
    if args.padding_mode == "fixed":
        return args.padding, args.padding, 1.0, "fixed"

    short = max(min(bbox_w, bbox_h), 1.0)
    long_ = max(bbox_w, bbox_h)
    aspect = long_ / short

    if aspect < args.aspect_threshold:
        # 정사각형 (45° 회전 가능성) → uniform padding (긴 변 × ratio_square)
        pad = int(long_ * args.padding_ratio_square)
        pad = max(args.padding_min, min(pad, args.padding_max))
        return pad, pad, aspect, "square_diagonal"
    else:
        # 가로/세로형 → per-axis (v2 동일 로직)
        pad_x = int(bbox_w * args.padding_ratio)
        pad_x = max(args.padding_min, min(pad_x, args.padding_max))
        pad_y = int(bbox_h * args.padding_ratio)
        pad_y = max(args.padding_min, min(pad_y, args.padding_max))
        return pad_x, pad_y, aspect, "per_axis"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage 1 Version A → PMI crop 추출 v3 (aspect-aware adaptive padding)",
    )
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help=f"Stage 1 가중치 (default: {DEFAULT_WEIGHTS})")
    p.add_argument("--drawings", type=Path, default=DEFAULT_DRAWINGS_LIST,
                   help=f"도면 filename list (default: {DEFAULT_DRAWINGS_LIST})")
    p.add_argument("--input", type=Path, default=None,
                   help="직접 입력 폴더 (--drawings 무시)")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                   help=f"원본 도면 폴더 (default: {DEFAULT_DATASET_DIR})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"PMI crop 출력 폴더 (default: {DEFAULT_OUTPUT_DIR})")
    # --- Padding (v3 aspect-aware) ----
    p.add_argument("--padding-mode", choices=["adaptive", "fixed"], default="adaptive",
                   help="padding 방식 (default: adaptive — aspect-aware)")
    p.add_argument("--padding", type=int, default=30,
                   help="fixed 모드 padding px (default: 30)")
    p.add_argument("--padding-ratio", type=float, default=0.4,
                   help="가로/세로형 (per-axis) ratio (default: 0.4)")
    p.add_argument("--padding-ratio-square", type=float, default=0.6,
                   help="정사각형 (square_diagonal) ratio — 45° 회전 보강 (default: 0.6)")
    p.add_argument("--aspect-threshold", type=float, default=1.5,
                   help="정사각형 판정 threshold (long/short < threshold) (default: 1.5)")
    p.add_argument("--padding-min", type=int, default=30,
                   help="adaptive 모드 최소 padding px (default: 30)")
    p.add_argument("--padding-max", type=int, default=80,
                   help="adaptive 모드 최대 padding px (default: 80)")
    # --- Upscale (해상도 보완 — 라벨링 가독성) ---
    p.add_argument("--upscale", type=float, default=1.0,
                   help="crop 업스케일 배율 (default: 1.0, 권장 3.0)")
    p.add_argument("--upscale-method", choices=["lanczos", "bicubic", "nearest"],
                   default="lanczos",
                   help="업스케일 보간 방식 (default: lanczos — 가장 부드러움)")
    p.add_argument("--sharpen", action="store_true",
                   help="업스케일 후 unsharp mask 적용 (텍스트 선명화)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="confidence 임계값 (default: 0.25)")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="inference 해상도 (default: 1280)")
    p.add_argument("--device", default=None,
                   help="GPU id (e.g. 0) 또는 cpu")
    p.add_argument("--limit", type=int, default=0,
                   help="처리 도면 수 제한 (0 = 전체)")
    p.add_argument("--min-pmi-area", type=int, default=100,
                   help="최소 PMI 박스 면적 px² (default: 100)")
    args = p.parse_args()

    # --- Validate ----
    if not args.weights.exists():
        log.error("Weights not found: %s", args.weights)
        log.error("Stage 1 학습 완료 필요. python src/stage1_layout.py train ...")
        return 2

    # --- Load drawings ----
    drawings = load_drawings(args)
    if not drawings:
        return 1

    if args.limit > 0:
        drawings = drawings[: args.limit]
        log.info("Limited to first %d", len(drawings))

    log.info("Total drawings to process: %d", len(drawings))

    # --- Lazy import ----
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as e:
        log.error("ultralytics not installed: %s", e)
        return 3

    log.info("Loading model: %s", args.weights)
    model = YOLO(str(args.weights))

    # --- Output dir ----
    args.output.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", args.output)
    if args.padding_mode == "adaptive":
        log.info("Padding mode: aspect-aware [v3, D-037]")
        log.info("  - per-axis ratio       : %.2f (가로/세로형)", args.padding_ratio)
        log.info("  - square ratio         : %.2f (정사각형, 45° 회전 보강)",
                 args.padding_ratio_square)
        log.info("  - aspect threshold     : %.2f (long/short)", args.aspect_threshold)
        log.info("  - min/max padding      : %d / %d px",
                 args.padding_min, args.padding_max)
    else:
        log.info("Padding mode: fixed (%d px)", args.padding)
    # Upscale 설정 출력
    if args.upscale > 1.0:
        log.info("Upscale: %.1fx (%s%s) — 텍스트 가독성 보완",
                 args.upscale, args.upscale_method,
                 " + sharpen" if args.sharpen else "")
        log.info("         ★ migrate_cvat_labels.py --scale %.1f 로 좌표 변환 가능",
                 args.upscale)
    else:
        log.info("Upscale: 1.0x (원본 해상도 유지)")
    log.info("Conf: %.2f / imgsz: %d / min_pmi_area: %d",
             args.conf, args.imgsz, args.min_pmi_area)

    # --- Process ----
    n_pmi_total = 0
    n_pmi_skipped_small = 0
    n_no_pmi = 0
    n_err = 0
    records: List[dict] = []

    pbar = tqdm(drawings, desc="Extracting PMI v3", unit="img",
                dynamic_ncols=True, leave=True)
    for img_path in pbar:
        try:
            results = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001
            log.error("Inference failed for %s: %s", img_path.name, e)
            n_err += 1
            continue

        if not results:
            n_no_pmi += 1
            continue

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            n_no_pmi += 1
            continue

        img = imread_unicode(img_path)
        if img is None:
            n_err += 1
            continue
        h_img, w_img = img.shape[:2]

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)

        pmi_indices = [i for i, c in enumerate(cls_arr) if int(c) == CLS_PMI]
        if not pmi_indices:
            n_no_pmi += 1
            continue

        group_key = img_path.stem.split(".rf.")[0]
        pmi_idx = 0

        for i in pmi_indices:
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            conf_val = float(conf_arr[i])

            bbox_w = x2 - x1
            bbox_h = y2 - y1
            area = bbox_w * bbox_h
            if area < args.min_pmi_area:
                n_pmi_skipped_small += 1
                continue

            # v3: aspect-aware padding
            pad_x, pad_y, aspect, strategy = calc_padding_v3(bbox_w, bbox_h, args)

            # 이미지 경계 clamp
            px1 = max(0, int(x1 - pad_x))
            py1 = max(0, int(y1 - pad_y))
            px2 = min(w_img, int(x2 + pad_x))
            py2 = min(h_img, int(y2 + pad_y))

            crop = img[py1:py2, px1:px2]
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                n_pmi_skipped_small += 1
                continue

            # v3 확장: upscale (해상도 보완 — 라벨링 가독성)
            crop_h_pre = crop.shape[0]
            crop_w_pre = crop.shape[1]
            if args.upscale > 1.0:
                interp_map = {
                    "lanczos": cv2.INTER_LANCZOS4,
                    "bicubic": cv2.INTER_CUBIC,
                    "nearest": cv2.INTER_NEAREST,
                }
                new_w = int(crop_w_pre * args.upscale)
                new_h = int(crop_h_pre * args.upscale)
                crop = cv2.resize(
                    crop, (new_w, new_h),
                    interpolation=interp_map[args.upscale_method],
                )
                if args.sharpen:
                    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0)
                    crop = cv2.addWeighted(crop, 1.5, blurred, -0.5, 0)

            crop_name = f"{img_path.stem}__PMI_{pmi_idx:03d}.jpg"
            crop_path = args.output / crop_name
            if not imwrite_unicode(crop_path, crop):
                n_err += 1
                continue

            # Manifest (v3: aspect_ratio + padding_strategy 추가)
            records.append({
                "crop_filename": crop_name,
                "source_drawing": img_path.name,
                "source_group_key": group_key,
                "pmi_idx": pmi_idx,
                "bbox_x1": int(x1),
                "bbox_y1": int(y1),
                "bbox_x2": int(x2),
                "bbox_y2": int(y2),
                "bbox_w": int(bbox_w),
                "bbox_h": int(bbox_h),
                "aspect_ratio": f"{aspect:.2f}",
                "padding_strategy": strategy,
                "pad_x": pad_x,
                "pad_y": pad_y,
                "crop_x1": px1,
                "crop_y1": py1,
                "crop_x2": px2,
                "crop_y2": py2,
                "crop_w": px2 - px1,
                "crop_h": py2 - py1,
                "upscale_factor": args.upscale,
                "upscale_method": args.upscale_method if args.upscale > 1.0 else "none",
                "sharpen": "on" if args.sharpen and args.upscale > 1.0 else "off",
                "saved_w": crop.shape[1],
                "saved_h": crop.shape[0],
                "padding_mode": args.padding_mode,
                "conf": f"{conf_val:.4f}",
            })

            pmi_idx += 1
            n_pmi_total += 1

        pbar.set_postfix(crops=n_pmi_total, small=n_pmi_skipped_small,
                         no_pmi=n_no_pmi, err=n_err)

    # --- Manifest CSV ----
    manifest_path = args.output / "manifest.csv"
    if records:
        with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        log.info("Manifest: %s", manifest_path)

    # --- Summary ----
    n_drawings = len(drawings)
    log.info("=" * 60)
    log.info("PMI crop 추출 완료 (v3 aspect-aware)")
    log.info("  Drawings processed   : %d", n_drawings)
    log.info("  No PMI (skipped)     : %d", n_no_pmi)
    log.info("  Errors               : %d", n_err)
    log.info("  PMI crops saved      : %d", n_pmi_total)
    log.info("  Small PMI skipped    : %d (area < %d px²)",
             n_pmi_skipped_small, args.min_pmi_area)
    if n_drawings > 0:
        log.info("  Avg PMI/drawing      : %.1f", n_pmi_total / n_drawings)
    log.info("=" * 60)
    log.info("Output: %s", args.output)
    log.info("")
    # padding 통계 (v3: strategy 별 분포 + aspect)
    if records:
        pads_x = [r["pad_x"] for r in records]
        pads_y = [r["pad_y"] for r in records]
        aspects = [float(r["aspect_ratio"]) for r in records]
        strategies = [r["padding_strategy"] for r in records]

        log.info("  pad_x (가로 padding) : min=%d / max=%d / mean=%.1f px",
                 min(pads_x), max(pads_x), sum(pads_x) / len(pads_x))
        log.info("  pad_y (세로 padding) : min=%d / max=%d / mean=%.1f px",
                 min(pads_y), max(pads_y), sum(pads_y) / len(pads_y))
        log.info("  aspect ratio        : min=%.2f / max=%.2f / mean=%.2f",
                 min(aspects), max(aspects), sum(aspects) / len(aspects))

        # Strategy 분포
        n_square = strategies.count("square_diagonal")
        n_peraxis = strategies.count("per_axis")
        n_fixed = strategies.count("fixed")
        log.info("  Strategy 분포        : square_diagonal=%d / per_axis=%d / fixed=%d",
                 n_square, n_peraxis, n_fixed)

        # square_diagonal 통계 (회전 텍스트 보강 검증용)
        if n_square > 0:
            sq_pads = [r["pad_x"] for r in records if r["padding_strategy"] == "square_diagonal"]
            log.info("  square_diagonal pad : min=%d / max=%d / mean=%.1f px",
                     min(sq_pads), max(sq_pads), sum(sq_pads) / len(sq_pads))

        # Upscale 통계 (적용 시)
        if args.upscale > 1.0:
            saved_w = [r["saved_w"] for r in records]
            saved_h = [r["saved_h"] for r in records]
            log.info("  Upscale 적용         : %.1fx (%s%s)",
                     args.upscale, args.upscale_method,
                     " + sharpen" if args.sharpen else "")
            log.info("  saved crop size      : w_mean=%.0f / h_mean=%.0f px",
                     sum(saved_w) / len(saved_w), sum(saved_h) / len(saved_h))

    log.info("=" * 60)
    log.info("[다음 단계 — Stage 2 OBB 라벨링]")
    log.info("  1. v2 와 시각 비교:")
    log.info("     - v2: outputs/cvat_stage2_input_v2/")
    log.info("     - v3: %s", args.output)
    log.info("  2. v3 가 더 나으면 → ZIP + CVAT 업로드")
    log.info("     cd %s && zip ../stage2_v3.zip *.jpg", args.output)
    log.info("  3. CVAT Task: Stage2_PMI_v3_844")
    log.info("  4. Labels: Measure / GDT / Roughness")
    log.info("  5. OBB 라벨링 → export YOLO format → Stage 2 학습")

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
