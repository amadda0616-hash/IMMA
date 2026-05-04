"""
src/exclude_groups.py

D-024 group-aware 제외 도구 (2026-04-29).

사용자 검수 결과 (조립도면 group_key 리스트) 를 받아 같은 원본의
모든 ``.rf.<hash>`` 변형을 일괄 ``dataset_excluded/`` 로 이동.

배경
----
``sort_by_yolo_pmi.py`` 의 분류 결과 + 사람 검수 → 조립도면 group_key 추출.
D-024 정합성 (같은 원본의 모든 증강 변형 한쪽에만) 을 보존하기 위해
group_key 단위로 일괄 처리.

작업
----
1. ``outputs/exclude_list.txt`` 의 group_key 파싱
2. ``dataset/*.jpg`` 스캔 → ``filename.split('.rf.')[0]`` 매칭
3. 매치된 모든 .rf.<hash> 변형을 ``dataset_excluded/`` 로 이동
4. 동기 처리: ``outputs/auto_labels/labels/<stem>.txt`` 도 이동
5. 통계 + manifest 출력

CLI
---
::

    # 미리보기 (이동 X)
    python src/exclude_groups.py --list outputs/exclude_list.txt --dryrun

    # 실제 이동
    python src/exclude_groups.py --list outputs/exclude_list.txt

    # 복사 (이동 대신 — 원본 dataset/ 보존)
    python src/exclude_groups.py --list outputs/exclude_list.txt --copy

exclude_list.txt 형식
---------------------
::

    # 주석 (#) 가능. 빈 줄 무시.
    # group_key 만 기록 (.rf.<hash> 부분 제거)
    0301040003_SHAFT-ARMATURE_REV-01_page_1_png
    0_700bar-1-_000001_jpg
    0_QFR10-39-_000001_jpg

관련 의사결정
-------------
- D-024 Group-aware split — 같은 group_key 의 모든 변형 한쪽에만
- D-026 가공/조립 분류 (sort_by_yolo_pmi.py 연계)
- D-035 사람 검수 정책
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Set, Tuple

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIST_PATH = PROJECT_ROOT / "outputs" / "exclude_list.txt"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_EXCLUDED_DIR = PROJECT_ROOT / "dataset_excluded"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs" / "auto_labels" / "labels"
DEFAULT_LABELS_EXCLUDED_DIR = PROJECT_ROOT / "outputs" / "auto_labels" / "labels_excluded"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "exclude_groups_manifest.csv"

SUPPORTED_EXTS = {".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("exclude_groups")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_exclude_list(path: Path) -> Set[str]:
    """exclude_list.txt → set of group_keys.

    형식:
    - 빈 줄 무시
    - # 시작 라인 = 주석 (무시)
    - 각 라인 = group_key (예: ``0301040003_SHAFT-ARMATURE_REV-01_page_1_png``)
    - 만약 .rf.<hash>.jpg 가 입력되면 자동으로 group_key 만 추출
    """
    if not path.exists():
        raise FileNotFoundError(f"Exclude list not found: {path}")

    keys: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # filename 형태로 들어와도 group_key 만 추출
            if ".rf." in line:
                line = line.split(".rf.")[0]
            # 확장자 제거 (jpg/jpeg)
            for ext in SUPPORTED_EXTS:
                if line.lower().endswith(ext):
                    line = line[: -len(ext)]
                    break
            keys.add(line)
    return keys


def find_files_by_group(dataset_dir: Path,
                       group_keys: Set[str]) -> List[Tuple[str, Path]]:
    """dataset_dir 의 모든 .jpg/.jpeg 중 group_key 매칭되는 파일 반환.

    Returns
    -------
    [(group_key, file_path), ...]
    """
    matches: List[Tuple[str, Path]] = []
    for img in dataset_dir.iterdir():
        if not img.is_file() or img.suffix.lower() not in SUPPORTED_EXTS:
            continue
        # group_key 추출
        gk = img.stem.split(".rf.")[0]
        if gk in group_keys:
            matches.append((gk, img))
    return matches


def move_or_copy(src: Path, dst: Path, copy: bool = False, dryrun: bool = False) -> str:
    """이동 또는 복사. 반환: 'move' / 'copy' / 'dryrun' / 'skip'."""
    if dst.exists():
        return "skip"
    if dryrun:
        return "dryrun"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
        return "copy"
    shutil.move(str(src), str(dst))
    return "move"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="D-024 group-aware 제외 도구 — 조립도면 group_key 단위 일괄 이동",
    )
    p.add_argument("--list", type=Path, default=DEFAULT_LIST_PATH,
                   help=f"제외할 group_key 리스트 (default: {DEFAULT_LIST_PATH})")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                   help=f"원본 데이터셋 (default: {DEFAULT_DATASET_DIR})")
    p.add_argument("--excluded-dir", type=Path, default=DEFAULT_EXCLUDED_DIR,
                   help=f"제외 폴더 (default: {DEFAULT_EXCLUDED_DIR})")
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR,
                   help=f"라벨 폴더 (default: {DEFAULT_LABELS_DIR})")
    p.add_argument("--labels-excluded-dir", type=Path,
                   default=DEFAULT_LABELS_EXCLUDED_DIR,
                   help=f"제외 라벨 폴더 (default: {DEFAULT_LABELS_EXCLUDED_DIR})")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH,
                   help=f"실행 manifest CSV (default: {DEFAULT_MANIFEST_PATH})")
    p.add_argument("--copy", action="store_true",
                   help="이동 대신 복사 (원본 dataset/ 보존)")
    p.add_argument("--dryrun", action="store_true",
                   help="미리보기 (실제 파일 이동 X)")
    p.add_argument("--no-labels", action="store_true",
                   help="라벨 동기 이동 안 함 (이미지만)")
    args = p.parse_args()

    # --- Validate ----
    if not args.list.exists():
        log.error("Exclude list not found: %s", args.list)
        log.error("먼저 sort_by_yolo_pmi.py 실행 + 검수 결과를 exclude_list.txt 로 작성")
        return 2
    if not args.dataset_dir.exists():
        log.error("Dataset dir not found: %s", args.dataset_dir)
        return 2

    # --- Load exclude list ----
    try:
        group_keys = load_exclude_list(args.list)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 2

    if not group_keys:
        log.warning("Empty exclude list. Exit.")
        return 0

    log.info("Loaded %d group_keys from %s", len(group_keys), args.list)

    # --- Find matching files ----
    matches = find_files_by_group(args.dataset_dir, group_keys)
    log.info("Matched %d files (.rf.<hash> variants) across %d unique group_keys",
             len(matches), len(set(gk for gk, _ in matches)))

    # 매칭 안 된 group_key 알림
    matched_keys = set(gk for gk, _ in matches)
    unmatched = group_keys - matched_keys
    if unmatched:
        log.warning("⚠️  Unmatched group_keys (%d) — typo 또는 이미 제거됨:",
                    len(unmatched))
        for gk in list(unmatched)[:10]:
            log.warning("    %s", gk)
        if len(unmatched) > 10:
            log.warning("    ... 외 %d", len(unmatched) - 10)

    if not matches:
        log.warning("No files to exclude. Exit.")
        return 0

    # --- Mode 표시 ----
    log.info("Mode: %s", "dryrun (preview)" if args.dryrun
             else ("copy" if args.copy else "move"))
    log.info("Dataset → Excluded: %s → %s", args.dataset_dir, args.excluded_dir)
    if not args.no_labels:
        log.info("Labels  → Excluded: %s → %s",
                 args.labels_dir, args.labels_excluded_dir)

    # --- 실행 ----
    records = []
    img_stats = {"move": 0, "copy": 0, "dryrun": 0, "skip": 0}
    lbl_stats = {"move": 0, "copy": 0, "dryrun": 0, "skip": 0, "missing": 0}

    for gk, img_path in matches:
        # 이미지 이동/복사
        img_dst = args.excluded_dir / img_path.name
        img_mode = move_or_copy(img_path, img_dst, copy=args.copy, dryrun=args.dryrun)
        img_stats[img_mode] += 1

        # 라벨 동기 이동/복사
        lbl_mode = "missing"
        if not args.no_labels:
            lbl_src = args.labels_dir / f"{img_path.stem}.txt"
            if lbl_src.exists():
                lbl_dst = args.labels_excluded_dir / f"{img_path.stem}.txt"
                lbl_mode = move_or_copy(lbl_src, lbl_dst, copy=args.copy,
                                         dryrun=args.dryrun)
            lbl_stats[lbl_mode] += 1

        records.append({
            "group_key": gk,
            "filename": img_path.name,
            "img_action": img_mode,
            "lbl_action": lbl_mode if not args.no_labels else "skipped",
        })

    # --- Manifest CSV ----
    if not args.dryrun or args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "group_key", "filename", "img_action", "lbl_action",
            ])
            writer.writeheader()
            writer.writerows(records)
        log.info("Manifest: %s", args.manifest)

    # --- Summary ----
    log.info("=" * 60)
    log.info("Excluded %d files across %d group_keys",
             len(matches), len(matched_keys))
    log.info("Image stats: %s", img_stats)
    if not args.no_labels:
        log.info("Label stats: %s", lbl_stats)
    if unmatched:
        log.info("Unmatched group_keys: %d (확인 필요)", len(unmatched))
    log.info("=" * 60)

    if args.dryrun:
        log.info("")
        log.info("[Dryrun 완료. 실제 이동하려면 --dryrun 빼고 재실행]")
    else:
        log.info("")
        log.info("[학습 데이터 정리 완료. 다음 단계]")
        log.info("  - dataset/ : 학습 잔여 (가공 + 부품도면)")
        log.info("  - dataset_excluded/ : 조립도면 보관 (삭제 X)")
        log.info("  - outputs/auto_labels/labels/ : 동기 정리됨")
        log.info("  - Stage 2 라벨링 진행 가능")

    return 0


if __name__ == "__main__":
    sys.exit(main())
