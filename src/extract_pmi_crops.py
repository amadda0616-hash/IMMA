"""
src/extract_pmi_crops.py

Stage 1 Version A → PMI crop 추출 (Stage 2 OBB 라벨링 입력 준비).

배경
----
- D-034: PMI 영역 = Stage 2 OBB 의 입력 (계층 구조 — Stage 1 axis-aligned PMI bbox →
  Stage 2 OBB 가 PMI 안의 Measure/GDT/Roughness 검출)
- D-036: 옵션 B — Stage 2 입력은 auto_pass + review priority 만 사용 (회전 변형 제외)
- D-037: adaptive padding 도입 — v1 fixed 10px 의 문제점:
  (a) 화살표/리더선이 잘려 학습 신호 부족
  (b) 큰 padding (50~100px) 은 인접 치수 침입 (overlap) 야기
  → **per-axis** adaptive padding (x/y 독립 계산):
     pad_x = clamp(bbox_w × ratio, [min, max])
     pad_y = clamp(bbox_h × ratio, [min, max])
  → 회전/세로 텍스트의 긴 축 방향 화살표 캡처 향상
  → 인접 치수 침입 최소화 (긴 축만 padding 증가)
  → default: ratio 0.4, min 30, max 80

워크플로
-------
1. ``outputs/stage2_input_drawings.txt`` 의 도면 list 읽기 (또는 ``--input`` 폴더)
2. Stage 1 Version A 모델 inference
3. PMI 클래스 (cls=1, Roboflow data.yaml) bbox 만 필터
4. 각 PMI 영역 crop (padding-mode: adaptive | fixed)
5. ``outputs/cvat_stage2_input_v2/`` 저장 (v2 — adaptive padding)
6. ``manifest.csv`` — crop ↔ source drawing + bbox 좌표 + 적용 padding 기록

산출물
------
::

    outputs/cvat_stage2_input_v2/      ← v2 (adaptive padding, D-037)
    ├── DwgFoo__PMI_000.jpg            ← CVAT 업로드용
    ├── DwgFoo__PMI_001.jpg
    ├── ...
    └── manifest.csv                    ← 원본 좌표 + padding_applied 기록

CLI
---
::

    # ★ 권장 (per-axis adaptive padding, default)
    python src/extract_pmi_crops.py

    # adaptive 파라미터 튜닝
    python src/extract_pmi_crops.py \
        --padding-mode adaptive \
        --padding-ratio 0.4 \
        --padding-min 30 \
        --padding-max 80

    # 호환성 — fixed padding (구 v1 방식, 모든 방향 동일)
    python src/extract_pmi_crops.py --padding-mode fixed --padding 30

    # 직접 폴더 + 처음 N장
    python src/extract_pmi_crops.py --input dataset/ --limit 20

    # conf 임계값 (작은 PMI 더 잡기)
    python src/extract_pmi_crops.py --conf 0.15

    # 최소 PMI 면적 (너무 작은 PMI 제외)
    python src/extract_pmi_crops.py --min-pmi-area 200

다음 단계
--------
1. CVAT 로컬 docker (~10분 설치)
2. Project: ``Stage2_Annotation_OBB``
3. Task: ``Stage2_PMI_v2_844`` (v1 task 는 백업으로 보존)
4. Labels: ``Measure`` / ``GDT`` / ``Roughness`` (rotated rectangle)
5. ``outputs/cvat_stage2_input_v2/`` 의 .jpg 만 ZIP 으로 업로드 (manifest.csv 제외)
6. OBB 라벨링 → export YOLO format

관련 의사결정
-------------
- D-024 group_key 보존 (manifest 기록)
- D-028 5 클래스 (PMI = Roboflow cls 1)
- D-029 매핑 (PMI 는 매핑 X — Roboflow 이름 그대로)
- D-034 hierarchical (PMI = Stage 2 입력)
- D-036 옵션 B 정책 (auto_pass + review priority 만)
- D-037 adaptive padding (v2, 인접 치수 침입 최소화 + 화살표 보존)
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cvat_stage2_input_v2"  # D-037: v2 (adaptive padding)

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
log = logging.getLogger("extract_pmi_crops")


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
        # Direct folder
        if not args.input.exists():
            log.error("Input dir not found: %s", args.input)
            return []
        for p in args.input.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                drawings.append(p)
        log.info("Direct folder %s: %d images", args.input, len(drawings))
    elif args.drawings.exists():
        # Filename list
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
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage 1 Version A → PMI crop 추출 (Stage 2 OBB 라벨링 입력)",
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
    # --- Padding (D-037: per-axis adaptive) ----
    p.add_argument("--padding-mode", choices=["adaptive", "fixed"], default="adaptive",
                   help="padding 방식 (default: adaptive — 축별 독립 비례)")
    p.add_argument("--padding", type=int, default=30,
                   help="fixed 모드 padding px (default: 30, 구 v1=10)")
    p.add_argument("--padding-ratio", type=float, default=0.4,
                   help="adaptive 모드: bbox_w × ratio (x), bbox_h × ratio (y) (default: 0.4)")
    p.add_argument("--padding-min", type=int, default=30,
                   help="adaptive 모드 최소 padding px (default: 30)")
    p.add_argument("--padding-max", type=int, default=80,
                   help="adaptive 모드 최대 padding px (default: 80)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="confidence 임계값 (default: 0.25)")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="inference 해상도 (default: 1280, Stage 1 학습과 동일)")
    p.add_argument("--device", default=None,
                   help="GPU id (e.g. 0) 또는 cpu")
    p.add_argument("--limit", type=int, default=0,
                   help="처리 도면 수 제한 (0 = 전체, default: 0)")
    p.add_argument("--min-pmi-area", type=int, default=100,
                   help="최소 PMI 박스 면적 px² (default: 100, 너무 작은 PMI 제외)")
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
        log.info("Padding mode: adaptive per-axis (ratio=%.2f, min=%d, max=%d) [D-037]",
                 args.padding_ratio, args.padding_min, args.padding_max)
    else:
        log.info("Padding mode: fixed (%d px, 모든 방향 동일)", args.padding)
    log.info("Conf: %.2f / imgsz: %d / min_pmi_area: %d",
             args.conf, args.imgsz, args.min_pmi_area)

    # --- Process ----
    n_pmi_total = 0
    n_pmi_skipped_small = 0
    n_no_pmi = 0
    n_err = 0
    records: List[dict] = []

    pbar = tqdm(drawings, desc="Extracting PMI", unit="img",
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

        # 원본 이미지 (실제 좌표용)
        img = imread_unicode(img_path)
        if img is None:
            n_err += 1
            continue
        h_img, w_img = img.shape[:2]

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)

        # PMI 만 필터
        pmi_indices = [i for i, c in enumerate(cls_arr) if int(c) == CLS_PMI]
        if not pmi_indices:
            n_no_pmi += 1
            continue

        group_key = img_path.stem.split(".rf.")[0]
        pmi_idx = 0

        for i in pmi_indices:
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            conf_val = float(conf_arr[i])

            # 면적 체크
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            area = bbox_w * bbox_h
            if area < args.min_pmi_area:
                n_pmi_skipped_small += 1
                continue

            # D-037: Padding 계산 (per-axis adaptive | fixed)
            if args.padding_mode == "adaptive":
                pad_x = int(bbox_w * args.padding_ratio)
                pad_x = max(args.padding_min, min(pad_x, args.padding_max))
                pad_y = int(bbox_h * args.padding_ratio)
                pad_y = max(args.padding_min, min(pad_y, args.padding_max))
            else:  # fixed
                pad_x = args.padding
                pad_y = args.padding

            # 이미지 경계 clamp
            px1 = max(0, int(x1 - pad_x))
            py1 = max(0, int(y1 - pad_y))
            px2 = min(w_img, int(x2 + pad_x))
            py2 = min(h_img, int(y2 + pad_y))

            crop = img[py1:py2, px1:px2]
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                n_pmi_skipped_small += 1
                continue

            # 저장
            crop_name = f"{img_path.stem}__PMI_{pmi_idx:03d}.jpg"
            crop_path = args.output / crop_name
            if not imwrite_unicode(crop_path, crop):
                n_err += 1
                continue

            # Manifest (D-037: per-axis padding 기록)
            records.append({
                "crop_filename": crop_name,
                "source_drawing": img_path.name,
                "source_group_key": group_key,
                "pmi_idx": pmi_idx,
                "bbox_x1": int(x1),
                "bbox_y1": int(y1),
                "bbox_x2": int(x2),
                "bbox_y2": int(y2),
                "crop_x1": px1,
                "crop_y1": py1,
                "crop_x2": px2,
                "crop_y2": py2,
                "crop_w": px2 - px1,
                "crop_h": py2 - py1,
                "padding_mode": args.padding_mode,
                "pad_x": pad_x,
                "pad_y": pad_y,
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
    log.info("PMI crop 추출 완료")
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
    # padding 통계 (per-axis 통계 — D-037)
    if records:
        pads_x = [r["pad_x"] for r in records]
        pads_y = [r["pad_y"] for r in records]
        log.info("  pad_x (가로 padding) : min=%d / max=%d / mean=%.1f px",
                 min(pads_x), max(pads_x), sum(pads_x) / len(pads_x))
        log.info("  pad_y (세로 padding) : min=%d / max=%d / mean=%.1f px",
                 min(pads_y), max(pads_y), sum(pads_y) / len(pads_y))
        # 회전/세로 텍스트 비율 — pad_y > pad_x 인 crop 비율
        n_vertical = sum(1 for x, y in zip(pads_x, pads_y) if y > x)
        n_horizontal = sum(1 for x, y in zip(pads_x, pads_y) if x > y)
        n_square = sum(1 for x, y in zip(pads_x, pads_y) if x == y)
        log.info("  Crop 형태 분포        : 가로형=%d / 세로형=%d / 정사각형=%d",
                 n_horizontal, n_vertical, n_square)

    log.info("=" * 60)
    log.info("[다음 단계 — Stage 2 OBB 라벨링]")
    log.info("  1. CVAT docker 설치 (필요 시):")
    log.info("     git clone https://github.com/cvat-ai/cvat && cd cvat")
    log.info("     docker compose up -d  → http://localhost:8080")
    log.info("  2. Project 생성: Stage2_Annotation_OBB")
    log.info("  3. Task 생성: Stage2_PMI_v2_844 (v1 task 는 백업으로 보존)")
    log.info("  4. Labels: Measure / GDT / Roughness (rectangle, rotation 가능)")
    log.info("  5. ZIP 업로드 (이미지만, manifest.csv 제외):")
    log.info("     cd %s && zip ../stage2_v2.zip *.jpg", args.output)
    log.info("  6. OBB 라벨링 → export YOLO format → Stage 2 학습")

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
