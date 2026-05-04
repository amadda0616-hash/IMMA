"""
src/prepare_kfold_dataset.py

Stage 2 OBB 학습용 K-fold Cross-Validation 데이터셋 준비 (D-024 group-aware).

배경
----
- Option F (5-fold CV) — Stage 2 학습의 robust 평가 + overfit 최대 방지
- 569 frames (Phase 8 정리 후) → 5 fold 분할
- Group-aware split (D-024 정합성 유지)
- 이미지/라벨 **복사 안 함** — 기존 ``data/annotation/`` 절대경로 참조 (디스크 절약)

처리 단계
---------
1. ``data/annotation/{images,labels}/{train,valid}/`` 의 모든 frame 풀링
2. group_key 추출 (``__PMI_`` 앞 + ``.rf.`` 앞)
3. K-fold group-aware 분배 (round-robin 균등 분배)
4. 각 fold 별:
   - ``train.txt`` — train 이미지 절대 경로 list
   - ``val.txt`` — val 이미지 절대 경로 list
   - ``data.yaml`` — ultralytics dataset config
5. group leak 검증 (모든 fold = 0 보장)

산출물
------
::

    data/annotation_kfold/
    ├── fold_0/
    │   ├── data.yaml
    │   ├── train.txt   ← 이미지 절대 경로 list
    │   └── val.txt
    ├── fold_1/
    ├── fold_2/
    ├── fold_3/
    └── fold_4/

CLI
---
::

    python src/prepare_kfold_dataset.py \\
        --data-dir data/annotation \\
        --output-dir data/annotation_kfold \\
        --k 5 \\
        --seed 42

관련 의사결정
-------------
- D-024 group-aware split (per fold 검증)
- D-028 5 클래스 (Stage 2 = Measure / GDT / Roughness)
- D-039 Stage 3-A PaddleOCR-VL-1.5
- Phase 8 Option B (SKIP-only frame 제외)
- Option F K-fold CV
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Measure", "GDT", "Roughness"]  # D-028 (Stage 2)
SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_kfold")


# ---------------------------------------------------------------------------
# Group key extraction (D-024)
# ---------------------------------------------------------------------------
def extract_group_key(filename: str) -> str:
    """
    YOLO OBB 라벨/이미지 파일명에서 group key 추출.

    예: "0_QRV08-20-_000001_jpg.rf.39798ff2a76caa1f219cf3e3f0522061__PMI_000.jpg"
        → "0_QRV08-20-_000001_jpg"
    """
    stem = Path(filename).stem
    if "__PMI_" in stem:
        stem = stem.split("__PMI_")[0]
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    return stem


# ---------------------------------------------------------------------------
# K-fold split (Group-aware)
# ---------------------------------------------------------------------------
def kfold_split(
    frames: List[Path], k: int, seed: int
) -> List[Tuple[List[Path], List[Path]]]:
    """
    Group-aware K-fold split.

    같은 group 의 frame 들은 같은 fold 의 같은 split (train OR val) 에 배치.

    Parameters
    ----------
    frames : list of Path
        모든 이미지 파일 경로
    k : int
        fold 수 (e.g., 5)
    seed : int
        랜덤 시드

    Returns
    -------
    list of (train_frames, valid_frames) tuples — 각 fold 별
    """
    # group_key → frames 매핑
    group_to_frames: Dict[str, List[Path]] = defaultdict(list)
    for f in frames:
        gk = extract_group_key(f.name)
        group_to_frames[gk].append(f)

    rng = random.Random(seed)
    groups = sorted(group_to_frames.keys())  # 결정론적 정렬
    rng.shuffle(groups)

    # Round-robin 분배 (group 단위, frame 수 균형 시도)
    # 더 균일한 분배를 위해 group 크기 기반 정렬 + 그리디 분배
    groups_with_size = [(g, len(group_to_frames[g])) for g in groups]
    # 큰 group 부터 (그리디 균형 분배)
    groups_with_size.sort(key=lambda x: -x[1])

    fold_groups: List[List[str]] = [[] for _ in range(k)]
    fold_sizes: List[int] = [0] * k

    for g, size in groups_with_size:
        # 가장 작은 fold 에 배치
        smallest_fold = min(range(k), key=lambda i: fold_sizes[i])
        fold_groups[smallest_fold].append(g)
        fold_sizes[smallest_fold] += size

    # K-fold 결과 생성
    folds: List[Tuple[List[Path], List[Path]]] = []
    for fold_idx in range(k):
        valid_groups_in_fold = set(fold_groups[fold_idx])
        train_groups_in_fold = set()
        for i, gs in enumerate(fold_groups):
            if i != fold_idx:
                train_groups_in_fold.update(gs)

        train_frames: List[Path] = []
        valid_frames: List[Path] = []
        for g, fs in group_to_frames.items():
            if g in valid_groups_in_fold:
                valid_frames.extend(sorted(fs))
            elif g in train_groups_in_fold:
                train_frames.extend(sorted(fs))

        folds.append((sorted(train_frames), sorted(valid_frames)))

    return folds


def verify_group_leak(
    train_frames: List[Path], valid_frames: List[Path]
) -> int:
    """
    Group leak 검증.

    Returns
    -------
    int — leak frame 수 (0 = 정상)
    """
    train_groups = {extract_group_key(f.name) for f in train_frames}
    valid_groups = {extract_group_key(f.name) for f in valid_frames}
    overlap = train_groups & valid_groups
    if not overlap:
        return 0
    leak_count = sum(1 for f in train_frames if extract_group_key(f.name) in overlap)
    leak_count += sum(1 for f in valid_frames if extract_group_key(f.name) in overlap)
    return leak_count


# ---------------------------------------------------------------------------
# Frame collection
# ---------------------------------------------------------------------------
def collect_frames(data_dir: Path) -> List[Path]:
    """
    data/annotation/{images,labels}/{train,valid}/ 에서 모든 (이미지, 라벨) 쌍 수집.

    라벨이 없는 이미지는 제외 (Phase 8 SKIP-only 제외 규칙 일관성).
    """
    images: List[Path] = []
    n_no_label = 0

    for split in ("train", "valid"):
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        if not img_dir.exists():
            log.warning("Missing dir: %s", img_dir)
            continue

        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in SUPPORTED_IMG_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists() and lbl.stat().st_size > 0:
                images.append(img.resolve())
            else:
                n_no_label += 1

    log.info("Frames collected: %d (skipped %d without labels)", len(images), n_no_label)
    return images


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_fold(
    fold_dir: Path,
    fold_idx: int,
    k: int,
    train_frames: List[Path],
    valid_frames: List[Path],
) -> None:
    """fold 폴더에 train.txt + val.txt + data.yaml 생성."""
    fold_dir.mkdir(parents=True, exist_ok=True)

    # train.txt
    train_txt = fold_dir / "train.txt"
    with open(train_txt, "w", encoding="utf-8", newline="\n") as f:
        for img in train_frames:
            f.write(f"{img}\n")

    # val.txt
    val_txt = fold_dir / "val.txt"
    with open(val_txt, "w", encoding="utf-8", newline="\n") as f:
        for img in valid_frames:
            f.write(f"{img}\n")

    # data.yaml
    yaml_lines = [
        f"# Stage 2 OBB K-fold {fold_idx}/{k - 1}",
        "# 생성 시점: prepare_kfold_dataset.py",
        "# D-024 group-aware split (group leak 0)",
        "# D-028 3 클래스 (Measure / GDT / Roughness)",
        "# D-039 Stage 3-A 별도 (PaddleOCR-VL-1.5)",
        "",
        f"path: {fold_dir.resolve()}",
        f"train: {train_txt.name}",
        f"val: {val_txt.name}",
        "",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {i}: {name}")
    yaml_lines.append("")
    yaml_lines.append(f"nc: {len(CLASS_NAMES)}")

    (fold_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Stage 2 OBB K-fold Cross-Validation 데이터셋 준비 "
            "(group-aware D-024, 이미지 복사 X)"
        ),
    )
    p.add_argument("--data-dir", type=Path, required=True,
                   help="기존 dataset 폴더 (e.g., data/annotation)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="K-fold 출력 폴더 (e.g., data/annotation_kfold)")
    p.add_argument("--k", type=int, default=5,
                   help="fold 수 (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="랜덤 시드 (default: 42)")
    p.add_argument("--dry-run", action="store_true",
                   help="검증만 (파일 저장 안 함)")
    args = p.parse_args()

    if not args.data_dir.exists():
        log.error("Data dir not found: %s", args.data_dir)
        return 1
    if args.k < 2:
        log.error("--k 는 2 이상이어야 함 (받은 값: %d)", args.k)
        return 2

    log.info("=" * 60)
    log.info("Stage 2 OBB K-fold Cross-Validation 데이터셋 준비")
    log.info("=" * 60)
    log.info("Data dir   : %s", args.data_dir)
    log.info("Output dir : %s", args.output_dir)
    log.info("K          : %d", args.k)
    log.info("Seed       : %d", args.seed)
    if args.dry_run:
        log.info("Mode       : DRY-RUN (저장 안 함)")
    log.info("=" * 60)

    # --- Step 1: frame 수집 ---
    frames = collect_frames(args.data_dir)
    if not frames:
        log.error("No frames collected. 종료.")
        return 3

    # --- Step 2: K-fold split ---
    log.info("")
    log.info("Performing %d-fold group-aware split ...", args.k)
    folds = kfold_split(frames, args.k, args.seed)

    # --- Step 3: fold 별 통계 + 검증 ---
    log.info("")
    log.info("=== Fold 분포 ===")
    log.info("%-8s %-10s %-10s %-12s %-12s %-10s",
             "Fold", "Train", "Valid", "Train grp", "Valid grp", "Leak")
    log.info("-" * 70)

    all_pass = True
    for fold_idx, (train, valid) in enumerate(folds):
        train_groups = {extract_group_key(f.name) for f in train}
        valid_groups = {extract_group_key(f.name) for f in valid}
        leak = verify_group_leak(train, valid)
        leak_marker = "0 ✓" if leak == 0 else f"{leak} ✗"
        if leak > 0:
            all_pass = False
        log.info("%-8d %-10d %-10d %-12d %-12d %-10s",
                 fold_idx, len(train), len(valid),
                 len(train_groups), len(valid_groups), leak_marker)

    log.info("-" * 70)
    if all_pass:
        log.info("Group leak: 0 (모든 fold) ✅ D-024 PASS")
    else:
        log.error("Group leak 발생! 종료.")
        return 4

    # --- Step 4: 출력 ---
    if args.dry_run:
        log.info("")
        log.info("[DRY-RUN] 파일 저장 안 함. 종료.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("=== Fold 파일 생성 ===")
    for fold_idx, (train, valid) in enumerate(folds):
        fold_dir = args.output_dir / f"fold_{fold_idx}"
        write_fold(fold_dir, fold_idx, args.k, train, valid)
        log.info("  %s/data.yaml + train.txt(%d) + val.txt(%d)",
                 fold_dir.relative_to(args.output_dir.parent),
                 len(train), len(valid))

    log.info("")
    log.info("=" * 60)
    log.info("K-fold 데이터셋 준비 완료")
    log.info("=" * 60)
    log.info("Output: %s", args.output_dir.resolve())
    log.info("")
    log.info("[다음 단계 — Phase 13: K-fold 순차 학습]")
    log.info("  python src/train_kfold.py \\")
    log.info("      --kfold-dir %s \\", args.output_dir)
    log.info("      --k %d \\", args.k)
    log.info("      --model yolo11l-obb.pt \\")
    log.info("      --imgsz 1280 --batch 4 \\")
    log.info("      --epochs 250 --patience 120 \\")
    log.info("      --device 0 --save-period 50")

    return 0


if __name__ == "__main__":
    sys.exit(main())
