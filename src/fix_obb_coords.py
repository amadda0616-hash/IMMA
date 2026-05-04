"""
src/fix_obb_coords.py

YOLO OBB 라벨 파일의 좌표 [0, 1] 범위 강제 클리핑.

배경
----
- V3-A 검증 시 ``obb_validity_rate`` FAIL — 좌표가 [0, 1] 범위 밖에 있는 OBB 박스 발견
- CVAT 라벨링 중 박스가 이미지 경계 밖으로 살짝 벗어나는 경우 발생
- 모든 invalid 케이스가 단순 좌표 범위 초과 (자기교차 / 누락 0건)
- 클리핑으로 안전하게 해결 (학습 영향 최소)

처리 정책 (★ 옵션 2: defensive)
-----------------------------
1. 전체 라벨 파일 검사 (recursive)
2. 각 OBB 라인의 좌표 클립 ([0, 1] 강제)
3. **변경된 파일만 write-back** (I/O 최소화 + byte-identical 보존)
4. 통계 출력 (파일 / OBB / 클립 정도)
5. ``--dry-run`` 옵션 (검증)
6. ``--backup-dir`` 옵션 (안전 백업)

CLI
---
::

    # 검증 (저장 안 함)
    python src/fix_obb_coords.py \\
        --labels-dir data/annotation/labels \\
        --dry-run

    # 실제 적용 (백업 안 함)
    python src/fix_obb_coords.py --labels-dir data/annotation/labels

    # 실제 적용 + 백업
    python src/fix_obb_coords.py \\
        --labels-dir data/annotation/labels \\
        --backup-dir data/annotation/labels_backup

YOLO OBB 라벨 형식 (8-point)
---------------------------
::

    class_id x1 y1 x2 y2 x3 y3 x4 y4

    예 (정상):     0 0.10 0.20 0.50 0.20 0.50 0.40 0.10 0.40
    예 (invalid):  0 0.10 -0.07 0.55 0.20 0.86 0.05 0.42 1.00 (-0.07 < 0, 1.00 OK but 1.00 < 1 strict)

관련 의사결정
-------------
- D-024 group-aware split (검증 통과)
- D-039 Stage 3-A PaddleOCR-VL-1.5
- V3-A obb_validity_rate critical 임계값 ≥ 1.00
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fix_obb_coords")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COORD_MIN = 0.0
COORD_MAX = 1.0
COORDS_PER_OBB = 8  # 4-point polygon, 8 coordinates


# ---------------------------------------------------------------------------
# Core: 단일 라인 클리핑
# ---------------------------------------------------------------------------
def clip_obb_line(line: str) -> Tuple[str, float, bool]:
    """
    한 OBB 라인의 좌표를 [0, 1] 로 클립.

    Returns
    -------
    (new_line, max_delta, was_clipped)
        - new_line: 클립된 라인 (변경 없으면 원본 그대로)
        - max_delta: 클립된 최대 delta (변경 없으면 0.0)
        - was_clipped: 클립이 발생했는지 여부
    """
    parts = line.strip().split()
    if len(parts) < 1 + COORDS_PER_OBB:
        # 형식 오류 → 원본 유지
        return line, 0.0, False

    cid = parts[0]
    try:
        coords = [float(x) for x in parts[1 : 1 + COORDS_PER_OBB]]
    except ValueError:
        return line, 0.0, False

    # 클립 + delta 계산
    max_delta = 0.0
    was_clipped = False
    new_coords: List[float] = []
    for c in coords:
        if c < COORD_MIN:
            delta = COORD_MIN - c
            new_coords.append(COORD_MIN)
            was_clipped = True
            max_delta = max(max_delta, delta)
        elif c > COORD_MAX:
            delta = c - COORD_MAX
            new_coords.append(COORD_MAX)
            was_clipped = True
            max_delta = max(max_delta, delta)
        else:
            new_coords.append(c)

    if not was_clipped:
        return line.rstrip("\n"), 0.0, False

    # 추가 인자 보존 (예: 9번째 이후 메타정보)
    extra = parts[1 + COORDS_PER_OBB:]
    new_parts = [cid] + [f"{c:.6f}" for c in new_coords] + extra
    new_line = " ".join(new_parts)
    return new_line, max_delta, True


# ---------------------------------------------------------------------------
# 단일 파일 처리
# ---------------------------------------------------------------------------
def process_file(
    txt_file: Path, dry_run: bool = False
) -> Tuple[int, int, float, Counter]:
    """
    한 라벨 파일 처리.

    Returns
    -------
    (n_lines, n_clipped, max_delta_in_file, class_counter)
    """
    n_lines = 0
    n_clipped = 0
    max_delta = 0.0
    class_counter: Counter = Counter()
    new_lines: List[str] = []
    file_modified = False

    with open(txt_file, "r", encoding="utf-8") as f:
        original = f.readlines()

    for line in original:
        if not line.strip():
            new_lines.append(line)
            continue
        n_lines += 1
        new_line, delta, was_clipped = clip_obb_line(line)
        if was_clipped:
            n_clipped += 1
            max_delta = max(max_delta, delta)
            file_modified = True
            # 클래스 카운트
            try:
                cid = int(new_line.split()[0])
                class_counter[cid] += 1
            except (ValueError, IndexError):
                pass
            new_lines.append(new_line + "\n")
        else:
            new_lines.append(line if line.endswith("\n") else line + "\n")

    # 변경된 파일만 write-back
    if file_modified and not dry_run:
        with open(txt_file, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(new_lines)

    return n_lines, n_clipped, max_delta, class_counter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "YOLO OBB 라벨 좌표 [0, 1] 클립 (V3-A obb_validity_rate FAIL 해결)"
        ),
    )
    p.add_argument("--labels-dir", type=Path, required=True,
                   help="라벨 폴더 (e.g., data/annotation/labels) — recursive 검색")
    p.add_argument("--dry-run", action="store_true",
                   help="검증만 (파일 저장 안 함)")
    p.add_argument("--backup-dir", type=Path, default=None,
                   help="원본 백업 폴더 (지정 시 변경 전 파일 복사)")
    args = p.parse_args()

    if not args.labels_dir.exists():
        log.error("Labels dir not found: %s", args.labels_dir)
        return 1

    # 라벨 파일 목록
    label_files = sorted(args.labels_dir.rglob("*.txt"))
    if not label_files:
        log.error("No label files in %s", args.labels_dir)
        return 2

    log.info("=" * 60)
    log.info("YOLO OBB 좌표 [0, 1] 클립")
    log.info("=" * 60)
    log.info("Labels dir : %s", args.labels_dir)
    log.info("Total files: %d", len(label_files))
    if args.dry_run:
        log.info("Mode       : DRY-RUN (저장 안 함)")
    if args.backup_dir:
        log.info("Backup dir : %s", args.backup_dir)
    log.info("=" * 60)

    # --- 백업 (요청 시) ---
    if args.backup_dir and not args.dry_run:
        if args.backup_dir.exists():
            log.error("Backup dir already exists: %s — 다른 경로 지정 또는 삭제", args.backup_dir)
            return 3
        shutil.copytree(args.labels_dir, args.backup_dir)
        log.info("Backup created: %s", args.backup_dir)

    # --- 처리 ---
    n_total_lines = 0
    n_total_clipped = 0
    files_modified = 0
    files_unchanged = 0
    overall_max_delta = 0.0
    overall_min_delta = float("inf")
    total_class_counter: Counter = Counter()
    modified_files: List[str] = []

    for txt_file in label_files:
        n_lines, n_clipped, max_delta, cnt = process_file(txt_file, args.dry_run)
        n_total_lines += n_lines
        n_total_clipped += n_clipped
        total_class_counter.update(cnt)
        if n_clipped > 0:
            files_modified += 1
            overall_max_delta = max(overall_max_delta, max_delta)
            overall_min_delta = min(overall_min_delta, max_delta)
            modified_files.append(txt_file.name)
        else:
            files_unchanged += 1

    if overall_min_delta == float("inf"):
        overall_min_delta = 0.0

    # --- 통계 ---
    log.info("")
    log.info("=== 처리 결과 ===")
    log.info("  Total files scanned   : %d", len(label_files))
    log.info("  Files modified        : %d (%.1f%%)",
             files_modified, files_modified / max(1, len(label_files)) * 100)
    log.info("  Files unchanged       : %d (%.1f%%)",
             files_unchanged, files_unchanged / max(1, len(label_files)) * 100)
    log.info("  Total OBBs scanned    : %d", n_total_lines)
    log.info("  OBBs clipped          : %d", n_total_clipped)
    if n_total_clipped > 0:
        log.info("  Max clip delta        : %.6f (%.2f%%)",
                 overall_max_delta, overall_max_delta * 100)
        log.info("  Min clip delta        : %.6f (%.2f%%)",
                 overall_min_delta, overall_min_delta * 100)
        log.info("")
        log.info("  클립된 OBB의 클래스 분포:")
        for cid in sorted(total_class_counter):
            log.info("    class %d: %d", cid, total_class_counter[cid])

    # --- 수정된 파일 목록 (처음 10개) ---
    if modified_files:
        log.info("")
        log.info("  수정된 파일 (처음 10개):")
        for fn in modified_files[:10]:
            log.info("    - %s", fn)
        if len(modified_files) > 10:
            log.info("    ... 총 %d 개", len(modified_files))

    # --- 최종 메시지 ---
    log.info("")
    if args.dry_run:
        log.info("[DRY-RUN] 파일 저장 안 함. 종료.")
    elif n_total_clipped == 0:
        log.info("✅ 모든 OBB 가 [0, 1] 범위 내 — 변경 없음.")
    else:
        log.info("✅ %d 박스 클립 완료. %d 파일 변경됨.",
                 n_total_clipped, files_modified)
        log.info("")
        log.info("[다음 단계 — V3-A 재검증]")
        log.info("  python -m src.validate.check_labels_obb \\")
        log.info("      --labels-dir %s/train \\", args.labels_dir)
        log.info("      --cfg %s/data.yaml", args.labels_dir.parent)

    return 0


if __name__ == "__main__":
    sys.exit(main())
