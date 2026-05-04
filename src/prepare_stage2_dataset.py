"""
src/prepare_stage2_dataset.py

CVAT YOLO OBB export → Stage 2 학습 데이터 준비 (Phase 8~11 통합).

처리 단계
---------
1. **Phase 8**: SKIP 클래스 (id=3) 박스 라인 제거 + SKIP-only frame 통째 제외 (Option B)
2. **Phase 9, 11**: ``data/annotation/{images,labels}/{train,valid}/`` 구조 생성 + 이미지 Copy
3. **Phase 10**: Group-aware 80/20 split (D-024) — group leak 0 검증
4. ``data.yaml`` 생성 (3 클래스: Measure / GDT / Roughness)

배경
----
- **D-024**: Group-aware split — 같은 원본 도면에서 잘린 PMI crop 들이 한쪽 split 에 모임 (data leakage 방지)
- **D-039**: Stage 3-A → PaddleOCR-VL-1.5 (별도 경로). Stage 2 학습은 본 스크립트로 준비
- **Phase 8 Option B**: SKIP-only frame Stage 2 학습 데이터에서만 제외
  - 이미지 원본 (``outputs/cvat_stage2_input_v3_upscaled/``) 은 보존 — Stage 3-A Rescue 용
  - ``outputs/skip_lists/stage1_fp_notes.txt`` (23개) 는 별도 경로로 Stage 3-A 입력
- **D-026**: WSL2 호환성 — symlink 대신 Copy 사용

CLI
---
::

    # 표준 실행 (D-039 정책)
    python src/prepare_stage2_dataset.py \\
        --labels-dir outputs/cvat_yolo_obb_raw/labels/train \\
        --images-dir outputs/cvat_stage2_input_v3_upscaled \\
        --output-dir data/annotation \\
        --split-ratio 0.8 \\
        --skip-class-id 3 \\
        --seed 42

    # 검증만 (저장 안 함)
    python src/prepare_stage2_dataset.py \\
        --labels-dir outputs/cvat_yolo_obb_raw/labels/train \\
        --images-dir outputs/cvat_stage2_input_v3_upscaled \\
        --output-dir data/annotation \\
        --dry-run

산출물
------
::

    data/annotation/
    ├── data.yaml                       ← 3 클래스 정의
    ├── images/
    │   ├── train/  (~595 jpg)
    │   └── valid/  (~149 jpg)
    └── labels/
        ├── train/  (~595 txt, YOLO OBB 8-point)
        └── valid/  (~149 txt)

관련 의사결정
-------------
- D-024 group-aware split (검증 통과)
- D-026 WSL2 호환성 (Copy)
- D-028 Stage 2 클래스 (Measure / GDT / Roughness)
- D-039 Stage 3-A PaddleOCR-VL-1.5 (별도 경로)
- Phase 8 Option B (SKIP-only Stage 2 학습 제외)
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants (D-028 Stage 2 클래스)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Measure", "GDT", "Roughness"]  # id 0, 1, 2
SKIP_CLASS_ID = 3  # CVAT export 시 SKIP 클래스 id

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_stage2_dataset")


# ---------------------------------------------------------------------------
# Group key extraction (D-024)
# ---------------------------------------------------------------------------
def extract_group_key(filename: str) -> str:
    """
    YOLO OBB 라벨/이미지 파일명에서 group key 추출.

    예: "0_QRV08-20-_000001_jpg.rf.39798ff2a76caa1f219cf3e3f0522061__PMI_000.txt"
        → "0_QRV08-20-_000001_jpg"
    """
    stem = Path(filename).stem
    if "__PMI_" in stem:
        stem = stem.split("__PMI_")[0]
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    return stem


# ---------------------------------------------------------------------------
# Step 1: SKIP 박스 제거 + SKIP-only frame 식별
# ---------------------------------------------------------------------------
def filter_skip_boxes(
    label_file: Path, skip_class_id: int
) -> Tuple[List[str], int, int, Counter]:
    """
    라벨 파일에서 SKIP 클래스 라인 제거.

    Returns
    -------
    (kept_lines, n_kept, n_removed, class_counter)
    """
    kept_lines: List[str] = []
    n_removed = 0
    n_kept = 0
    class_counter: Counter = Counter()

    if not label_file.exists():
        return kept_lines, n_kept, n_removed, class_counter

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                # 형식 오류 라인 → 보존 (안전)
                kept_lines.append(line)
                n_kept += 1
                continue

            if class_id == skip_class_id:
                n_removed += 1
            else:
                kept_lines.append(line)
                n_kept += 1
                class_counter[class_id] += 1

    return kept_lines, n_kept, n_removed, class_counter


# ---------------------------------------------------------------------------
# Step 2: Group-aware split (D-024)
# ---------------------------------------------------------------------------
def group_aware_split(
    frames: List[str], split_ratio: float, seed: int,
) -> Tuple[List[str], List[str], Dict]:
    """
    Group-aware 80/20 split.

    같은 group_key 의 frame 들은 같은 split 에 배치 (data leakage 방지).
    """
    # group_key → frames 매핑
    group_to_frames: Dict[str, List[str]] = defaultdict(list)
    for frame in frames:
        gk = extract_group_key(frame)
        group_to_frames[gk].append(frame)

    rng = random.Random(seed)
    groups = sorted(group_to_frames.keys())  # 결정론적 정렬
    rng.shuffle(groups)

    total_frames = len(frames)
    target_train = int(total_frames * split_ratio)

    train_groups: List[str] = []
    valid_groups: List[str] = []
    train_count = 0

    for gk in groups:
        n = len(group_to_frames[gk])
        # train 부족분 우선 채우기 (마지막 group 도 포함되도록)
        if train_count + n / 2 <= target_train or not train_groups:
            train_groups.append(gk)
            train_count += n
        else:
            valid_groups.append(gk)

    train_frames: List[str] = []
    valid_frames: List[str] = []
    for gk in train_groups:
        train_frames.extend(sorted(group_to_frames[gk]))
    for gk in valid_groups:
        valid_frames.extend(sorted(group_to_frames[gk]))

    stats = {
        "n_groups": len(groups),
        "n_train_groups": len(train_groups),
        "n_valid_groups": len(valid_groups),
        "n_train_frames": len(train_frames),
        "n_valid_frames": len(valid_frames),
        "split_ratio_actual": len(train_frames) / max(1, total_frames),
    }
    return train_frames, valid_frames, stats


def verify_group_leak(
    train_frames: List[str], valid_frames: List[str]
) -> int:
    """
    Group leak 검증 — train 과 valid 가 같은 group_key 를 공유하는지 확인.

    Returns
    -------
    int — leak 된 frame 수 (0 = 정상)
    """
    train_groups = {extract_group_key(f) for f in train_frames}
    valid_groups = {extract_group_key(f) for f in valid_frames}
    overlap = train_groups & valid_groups

    if not overlap:
        return 0

    leak_count = sum(1 for f in train_frames if extract_group_key(f) in overlap)
    leak_count += sum(1 for f in valid_frames if extract_group_key(f) in overlap)
    return leak_count


# ---------------------------------------------------------------------------
# Step 3: 출력 구조 생성 + 이미지 Copy
# ---------------------------------------------------------------------------
def write_dataset(
    output_dir: Path,
    train_frames: List[str],
    valid_frames: List[str],
    label_data: Dict[str, List[str]],
    images_dir: Path,
) -> Dict[str, Dict[str, int]]:
    """
    data/annotation/{images,labels}/{train,valid}/ 구조 생성.
    """
    splits: List[Tuple[str, List[str]]] = [
        ("train", train_frames),
        ("valid", valid_frames),
    ]
    counts: Dict[str, Dict[str, int]] = {}

    for split_name, frames in splits:
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        n_img = 0
        n_lbl = 0
        n_img_missing = 0

        for frame_stem in frames:
            # 라벨 저장
            lbl_file = lbl_out / f"{frame_stem}.txt"
            content = "\n".join(label_data[frame_stem])
            if content and not content.endswith("\n"):
                content += "\n"
            lbl_file.write_text(content, encoding="utf-8", newline="\n")
            n_lbl += 1

            # 이미지 Copy (D-026 회피: symlink 대신 copy)
            src_img = images_dir / f"{frame_stem}.jpg"
            if src_img.exists():
                dst_img = img_out / f"{frame_stem}.jpg"
                shutil.copy2(src_img, dst_img)
                n_img += 1
            else:
                n_img_missing += 1
                log.warning("Image missing: %s", src_img.name)

        counts[split_name] = {
            "n_images": n_img,
            "n_labels": n_lbl,
            "n_images_missing": n_img_missing,
        }
    return counts


def write_data_yaml(output_dir: Path) -> None:
    """data.yaml 생성 (3 클래스, Stage 2 OBB)."""
    yaml_content = (
        "# Stage 2 OBB 학습 데이터셋\n"
        "# 생성 시점: prepare_stage2_dataset.py\n"
        "# D-024 group-aware split / D-028 5 클래스 / D-039 Option B (SKIP 제외)\n"
        "\n"
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "\n"
        "names:\n"
        f"  0: {CLASS_NAMES[0]}\n"
        f"  1: {CLASS_NAMES[1]}\n"
        f"  2: {CLASS_NAMES[2]}\n"
        "\n"
        "nc: 3\n"
    )
    (output_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "CVAT YOLO OBB export → Stage 2 학습 데이터 준비 "
            "(Phase 8~11: SKIP 제거 + group-aware split + Copy + data.yaml)"
        ),
    )
    p.add_argument("--labels-dir", type=Path, required=True,
                   help="CVAT export 라벨 폴더 (e.g., outputs/cvat_yolo_obb_raw/labels/train)")
    p.add_argument("--images-dir", type=Path, required=True,
                   help="이미지 폴더 (e.g., outputs/cvat_stage2_input_v3_upscaled)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="출력 폴더 (e.g., data/annotation)")
    p.add_argument("--split-ratio", type=float, default=0.8,
                   help="Train 비율 (default: 0.8 = 80/20)")
    p.add_argument("--skip-class-id", type=int, default=SKIP_CLASS_ID,
                   help=f"SKIP 클래스 id (default: {SKIP_CLASS_ID})")
    p.add_argument("--seed", type=int, default=42,
                   help="랜덤 시드 (default: 42)")
    p.add_argument("--dry-run", action="store_true",
                   help="검증만 하고 파일 저장 안 함")
    args = p.parse_args()

    # --- 입력 검증 ---
    if not args.labels_dir.exists():
        log.error("Labels dir not found: %s", args.labels_dir)
        return 1
    if not args.images_dir.exists():
        log.error("Images dir not found: %s", args.images_dir)
        return 2

    log.info("=" * 60)
    log.info("Stage 2 학습 데이터 준비 (Phase 8~11)")
    log.info("=" * 60)
    log.info("Labels dir : %s", args.labels_dir)
    log.info("Images dir : %s", args.images_dir)
    log.info("Output dir : %s", args.output_dir)
    log.info("Split ratio: %.2f / SKIP class id: %d / Seed: %d",
             args.split_ratio, args.skip_class_id, args.seed)
    if args.dry_run:
        log.info("Mode       : DRY-RUN (저장 안 함)")
    log.info("=" * 60)

    # --- Step 1: 라벨 파일 처리 ---
    label_files = sorted(args.labels_dir.glob("*.txt"))
    if not label_files:
        log.error("No label files found in %s", args.labels_dir)
        return 3
    log.info("Found %d label files", len(label_files))

    label_data: Dict[str, List[str]] = {}
    n_total_lines = 0
    n_total_removed = 0
    n_skip_only_frames = 0
    n_kept_frames = 0
    class_counter: Counter = Counter()

    for lbl_file in label_files:
        kept, n_kept, n_removed, cnt = filter_skip_boxes(
            lbl_file, args.skip_class_id,
        )
        n_total_lines += (n_kept + n_removed)
        n_total_removed += n_removed
        class_counter.update(cnt)

        if not kept:
            # SKIP-only frame → Option B 통째 제외
            n_skip_only_frames += 1
        else:
            label_data[lbl_file.stem] = kept
            n_kept_frames += 1

    log.info("")
    log.info("=== Phase 8: SKIP 박스 제거 + SKIP-only frame 제외 ===")
    log.info("  Total label files       : %d", len(label_files))
    log.info("  Total boxes (입력)      : %d", n_total_lines)
    log.info("  SKIP boxes 제거         : %d", n_total_removed)
    log.info("  Remaining boxes         : %d", n_total_lines - n_total_removed)
    log.info("  SKIP-only frames 제외   : %d (★ Option B)", n_skip_only_frames)
    log.info("  Kept frames             : %d", n_kept_frames)
    log.info("")
    log.info("  클래스별 박스 분포 (kept):")
    for cls_id in sorted(class_counter):
        cls_name = (
            CLASS_NAMES[cls_id]
            if 0 <= cls_id < len(CLASS_NAMES)
            else f"unknown({cls_id})"
        )
        log.info("    %d (%s): %d", cls_id, cls_name, class_counter[cls_id])

    if not label_data:
        log.error("All frames are SKIP-only. No data to train. 종료.")
        return 4

    # --- Step 2: Group-aware split ---
    train_frames, valid_frames, stats = group_aware_split(
        list(label_data.keys()), args.split_ratio, args.seed,
    )

    log.info("")
    log.info("=== Phase 10: Group-aware split (D-024) ===")
    log.info("  Total frames            : %d", n_kept_frames)
    log.info("  Unique groups           : %d", stats["n_groups"])
    log.info("  Train groups / frames   : %d / %d (%.1f%%)",
             stats["n_train_groups"], stats["n_train_frames"],
             stats["split_ratio_actual"] * 100)
    log.info("  Valid groups / frames   : %d / %d (%.1f%%)",
             stats["n_valid_groups"], stats["n_valid_frames"],
             (1 - stats["split_ratio_actual"]) * 100)

    # Group leak 검증
    leak = verify_group_leak(train_frames, valid_frames)
    if leak == 0:
        log.info("  Group leak              : 0 ✅ (D-024 PASS)")
    else:
        log.error("  Group leak              : %d ❌ (D-024 FAIL)", leak)
        return 5

    if args.dry_run:
        log.info("")
        log.info("[DRY-RUN] 파일 저장 안 함. 종료.")
        return 0

    # --- Step 3: 출력 구조 생성 + 이미지 Copy ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = write_dataset(
        args.output_dir, train_frames, valid_frames,
        label_data, args.images_dir,
    )

    log.info("")
    log.info("=== Phase 9, 11: data/annotation/ 구조 생성 + 이미지 Copy ===")
    for split_name in ("train", "valid"):
        c = counts[split_name]
        marker = " ⚠" if c["n_images_missing"] > 0 else ""
        log.info("  %s: images=%d / labels=%d / images_missing=%d%s",
                 split_name, c["n_images"], c["n_labels"],
                 c["n_images_missing"], marker)

    # --- Step 4: data.yaml ---
    write_data_yaml(args.output_dir)
    log.info("")
    log.info("=== data.yaml 생성 (3 클래스: Measure / GDT / Roughness) ===")
    log.info("  %s", args.output_dir / "data.yaml")

    # --- 요약 ---
    log.info("")
    log.info("=" * 60)
    log.info("Stage 2 학습 데이터 준비 완료")
    log.info("=" * 60)
    log.info("Output: %s", args.output_dir.resolve())
    log.info("")
    log.info("[다음 단계]")
    log.info("  Phase 12. V3-A 라벨 검증:")
    log.info("    python -m src.validate.check_labels_obb \\")
    log.info("        --labels-dir %s/labels/train \\", args.output_dir)
    log.info("        --cfg %s/data.yaml", args.output_dir)
    log.info("")
    log.info("  Phase 12.5. configs/yolo_obb.yaml augmentation 강화 (옵션 C)")
    log.info("")
    log.info("  Phase 13. Stage 2 학습 시작 (~5h):")
    log.info("    python src/stage2_annotation.py train \\")
    log.info("        --data %s/data.yaml \\", args.output_dir)
    log.info("        --model yolo11m-obb.pt --epochs 150 --imgsz 1024 --batch 8 --device 0")

    return 0


if __name__ == "__main__":
    sys.exit(main())
