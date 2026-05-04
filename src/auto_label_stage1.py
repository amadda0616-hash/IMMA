"""
src/auto_label_stage1.py

Active Learning Step 6 — Stage 1 자동 라벨링.

Stage 1 모델 (`checkpoints/yolo_det.pt`, Version A) 으로 ``dataset/`` 의
미라벨 도면을 자동 라벨링. 결과를 **Roboflow Pre-annotation Import 호환**
형식 (YOLO det txt) 으로 저장하고, confidence 분포 기반 **Active Learning
우선순위 매니페스트** 생성 (낮은 confidence → 우선 검수).

워크플로
-------
1. ``dataset/`` 의 모든 JPG 스캔
2. ``IMMA.v1i.yolov11/`` 의 seed (이미 라벨링된 100장) 제외
3. 나머지 ~5,739장에 대해 ``predict_one()`` 호출
4. YOLO txt 라벨 + 통계 manifest 저장 (avg_conf 오름차순)

출력 구조 (``--output`` default = ``outputs/auto_labels/``):
::

    outputs/auto_labels/
    ├── labels/                       ← YOLO txt (5클래스 D-028, Roboflow 이름 그대로)
    │   ├── CAD_Drawing01.txt
    │   └── ...
    ├── images/                       ← 원본 이미지 symlink (Linux) 또는 복사 (Windows)
    │   ├── CAD_Drawing01.jpg
    │   └── ...
    └── manifest.csv                  ← UTF-8-SIG, avg_conf 오름차순

CLI
---
::

    python src/auto_label_stage1.py \\
        --weights checkpoints/yolo_det.pt \\
        --input dataset/ \\
        --output outputs/auto_labels/ \\
        --conf 0.25 --imgsz 1280 --device 0

    # 디버깅 — 처음 50장만
    python src/auto_label_stage1.py --limit 50

    # seed (라벨링된 100장) 도 포함
    python src/auto_label_stage1.py --include-seed

관련 의사결정
-------------
- D-024  Group-aware split (.rf.<hash> group key 보존)
- D-026  가공/조립 분류 (sort_by_drawing_type 와 병행 가능)
- D-028  Stage 1 5 클래스 (Isometric/PMI/Table/Text/View)
- D-029  Roboflow→내부 매핑 (출력은 Roboflow 이름 그대로 — Roboflow Import 호환성)
- D-030  cu128 (RTX 5080 Blackwell)
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "checkpoints" / "yolo_det.pt"
DEFAULT_INPUT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "auto_labels"
DEFAULT_SEED_DIR = PROJECT_ROOT / "IMMA.v1i.yolov11"

# Stage 1 클래스 (D-028, Roboflow data.yaml 순서)
# 출력 txt 의 class_id 는 이 순서 — Roboflow Pre-annotation Import 호환성
CLASS_NAMES_RF: List[str] = ["Isometric", "PMI", "Table", "Text", "View"]

SUPPORTED_EXTS = {".jpg", ".jpeg"}

# Active Learning 우선순위 임계값
LOW_CONF_THRESHOLD = 0.5      # avg_conf < 0.5 → 우선 검수
HIGH_CONF_THRESHOLD = 0.65    # avg_conf >= 0.65 → 자동 검수 패스 가능
                              # (2026-04-29 갱신: 0.85 → 0.65)
                              # 5,839장 실측 결과 max avg_conf = 0.845 / mean = 0.539
                              # 도면당 평균 50개 박스 (PMI 작은 박스 다수) → 평균 conf 끌어내림
                              # 0.85 임계값에서는 auto_pass = 0 (너무 엄격)
                              # 0.65 임계값 → 약 127장 (2.2%) auto_pass 분류 가능

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto_label_stage1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_seed_stems(seed_dir: Path) -> Set[str]:
    """Return set of file stems (no extension) for seed images already labeled."""
    stems: Set[str] = set()
    if not seed_dir.exists():
        log.warning("Seed dir not found (skip filtering): %s", seed_dir)
        return stems
    for split in ("train", "valid", "test"):
        img_dir = seed_dir / split / "images"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() in SUPPORTED_EXTS:
                stems.add(img.stem)
    return stems


def predict_to_yolo_txt(
    image_path: Path,
    model,
    conf_thr: float = 0.25,
    imgsz: int = 1280,
    device: Optional[str] = None,
) -> tuple[List[str], List[float]]:
    """
    Run YOLO predict on one image.

    Returns
    -------
    (txt_lines, confs) — txt_lines: List of "class_id cx cy w h" (normalized).
    """
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf_thr,
        device=device,
        verbose=False,
    )

    txt_lines: List[str] = []
    confs: List[float] = []

    if not results:
        return txt_lines, confs

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return txt_lines, confs

    img_h, img_w = r.orig_shape  # (h, w)
    boxes = r.boxes
    xywh = boxes.xywh.cpu().numpy() if hasattr(boxes.xywh, "cpu") else np.asarray(boxes.xywh)
    cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)

    for (cx, cy, w, h), c, p in zip(xywh, cls, conf):
        if p < conf_thr:
            continue
        # Normalize to [0, 1]
        cx_n = float(cx) / img_w
        cy_n = float(cy) / img_h
        w_n = float(w) / img_w
        h_n = float(h) / img_h
        # Clamp
        cx_n = max(0.0, min(1.0, cx_n))
        cy_n = max(0.0, min(1.0, cy_n))
        w_n = max(0.0, min(1.0, w_n))
        h_n = max(0.0, min(1.0, h_n))

        cls_id = int(c)
        line = f"{cls_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"
        txt_lines.append(line)
        confs.append(float(p))

    return txt_lines, confs


def link_or_copy(src: Path, dst: Path) -> str:
    """Symlink (Linux) or copy (Windows fallback). Returns 'link' / 'copy' / 'skip'."""
    if dst.exists():
        return "skip"
    try:
        dst.symlink_to(src.resolve())
        return "link"
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        return "copy"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage 1 자동 라벨링 (Active Learning Step 6)",
    )
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help=f"Stage 1 가중치 (default: {DEFAULT_WEIGHTS})")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help=f"입력 이미지 폴더 (default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"출력 폴더 (default: {DEFAULT_OUTPUT})")
    p.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR,
                   help="이미 라벨링된 seed 폴더 (이 안의 이미지는 제외)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="confidence 임계값 (default: 0.25, ultralytics 기본)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8,
                   help="(현재 미사용 — predict 1장씩 처리)")
    p.add_argument("--device", default=None,
                   help="GPU id (e.g. 0) 또는 'cpu'")
    p.add_argument("--limit", type=int, default=None,
                   help="처리 제한 (디버깅용 — 처음 N장만)")
    p.add_argument("--include-seed", action="store_true",
                   help="seed (이미 라벨링됨) 도 포함 — 보통은 제외")
    p.add_argument("--copy-images", action="store_true",
                   help="symlink 대신 이미지 복사 (Windows 또는 Roboflow zip 업로드 시)")
    args = p.parse_args()

    # --- Validate ----
    if not args.weights.exists():
        log.error("Weights not found: %s", args.weights)
        log.error("Stage 1 학습이 완료되었는지 확인 필요. 'src/stage1_layout.py train' 참조.")
        return 2
    if not args.input.exists():
        log.error("Input dir not found: %s", args.input)
        return 2

    # --- Lazy import ----
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as e:
        log.error("ultralytics not installed: %s", e)
        return 3

    # --- Find images ----
    all_images = sorted(
        [p for p in args.input.iterdir()
         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    )
    log.info("Total images in %s: %d", args.input, len(all_images))

    seed_stems: Set[str] = set() if args.include_seed else get_seed_stems(args.seed_dir)
    if seed_stems:
        log.info("Seed images to skip: %d (%s)", len(seed_stems), args.seed_dir)

    images_to_process = [img for img in all_images if img.stem not in seed_stems]
    log.info("Images to auto-label: %d", len(images_to_process))

    if args.limit and args.limit > 0:
        images_to_process = images_to_process[:args.limit]
        log.info("Limited to first %d (debug mode)", len(images_to_process))

    if not images_to_process:
        log.warning("No images to process. Exit.")
        return 0

    # --- Setup output ----
    labels_dir = args.output / "labels"
    images_dir = args.output / "images"
    manifest_path = args.output / "manifest.csv"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", args.output)
    log.info("  - labels:   %s", labels_dir)
    log.info("  - images:   %s (%s)", images_dir,
             "copy" if args.copy_images else "symlink")
    log.info("  - manifest: %s", manifest_path)

    # --- Load model ----
    log.info("Loading model: %s", args.weights)
    model = YOLO(str(args.weights))
    log.info("Model classes: %s", model.names)
    log.info("Conf threshold: %.2f / imgsz: %d / device: %s",
             args.conf, args.imgsz, args.device or "auto")

    # --- Process ----
    records: List[Dict] = []
    n_classes_count = {i: 0 for i in range(len(CLASS_NAMES_RF))}
    n_empty = 0
    n_low_conf = 0
    n_high_conf = 0
    n_errors = 0

    pbar = tqdm(images_to_process, desc="Auto-labeling", unit="img",
                dynamic_ncols=True, leave=True)

    for img_path in pbar:
        try:
            txt_lines, confs = predict_to_yolo_txt(
                img_path, model,
                conf_thr=args.conf,
                imgsz=args.imgsz,
                device=args.device,
            )
        except Exception as e:  # noqa: BLE001
            log.error("Failed %s: %s", img_path.name, e)
            n_errors += 1
            continue

        # Save txt (always — even if empty, Roboflow 호환)
        txt_path = labels_dir / f"{img_path.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))
            if txt_lines:
                f.write("\n")

        # Image symlink/copy
        img_dst = images_dir / img_path.name
        if not img_dst.exists():
            if args.copy_images:
                shutil.copy2(img_path, img_dst)
            else:
                link_or_copy(img_path, img_dst)

        # Stats
        if not txt_lines:
            n_empty += 1
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        max_conf = max(confs) if confs else 0.0
        min_conf = min(confs) if confs else 0.0

        priority = "high"  # default — 검수 우선순위 낮음 (모델이 확신)
        if not confs:
            priority = "empty"   # 검출 0건 — 사람이 보고 결정
        elif avg_conf < LOW_CONF_THRESHOLD:
            priority = "low_conf"
            n_low_conf += 1
        elif avg_conf >= HIGH_CONF_THRESHOLD:
            priority = "auto_pass"
            n_high_conf += 1
        else:
            priority = "review"

        # Per-class count
        for line in txt_lines:
            try:
                cls_id = int(line.split()[0])
                if 0 <= cls_id < len(CLASS_NAMES_RF):
                    n_classes_count[cls_id] += 1
            except (ValueError, IndexError):
                pass

        records.append({
            "filename": img_path.name,
            "n_boxes": len(txt_lines),
            "avg_conf": f"{avg_conf:.4f}",
            "min_conf": f"{min_conf:.4f}",
            "max_conf": f"{max_conf:.4f}",
            "priority": priority,
            "label_path": str(txt_path.relative_to(args.output).as_posix()),
        })

        pbar.set_postfix(empty=n_empty, low=n_low_conf, high=n_high_conf, err=n_errors)

    # --- Write manifest (sorted by priority + avg_conf) ----
    # 우선순위: empty (사람 직접 라벨) → low_conf (우선 검수) → review → auto_pass
    priority_order = {"empty": 0, "low_conf": 1, "review": 2, "auto_pass": 3, "high": 2}
    records_sorted = sorted(
        records,
        key=lambda r: (priority_order.get(r["priority"], 99), float(r["avg_conf"])),
    )

    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "n_boxes", "avg_conf", "min_conf", "max_conf",
            "priority", "label_path",
        ])
        writer.writeheader()
        writer.writerows(records_sorted)

    # --- Summary ----
    n_total = len(records)
    log.info("=" * 60)
    log.info("Auto-label complete: %d images", n_total)
    log.info("  Empty (0 boxes):                 %4d  (%.1f%%) — 사람이 직접 라벨",
             n_empty, n_empty / n_total * 100)
    log.info("  Low conf (avg < %.2f):            %4d  (%.1f%%) — ★ 우선 검수",
             LOW_CONF_THRESHOLD, n_low_conf, n_low_conf / n_total * 100)
    log.info("  High conf (avg ≥ %.2f):           %4d  (%.1f%%) — 자동 패스 가능",
             HIGH_CONF_THRESHOLD, n_high_conf, n_high_conf / n_total * 100)
    log.info("  Errors:                          %4d", n_errors)
    log.info("Per-class total bboxes:")
    for cls_id, name in enumerate(CLASS_NAMES_RF):
        log.info("  %d  %-10s  %d", cls_id, name, n_classes_count[cls_id])
    log.info("=" * 60)
    log.info("Next: Roboflow Pre-annotation Import")
    log.info("  - Manifest: %s", manifest_path)
    log.info("  - Sort by 'priority' column → 'low_conf' / 'empty' 먼저 검수")

    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
