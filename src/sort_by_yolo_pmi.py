"""
src/sort_by_yolo_pmi.py

D-026 대체 (2026-04-29) — Stage 1 Version A 자동 라벨 기반
가공/조립/모호 도면 분류기.

배경
----
이전 휴리스틱 분류기 (`sort_by_drawing_type.py`, OCR + Hough) 가 5,839장
실측에서 mfg=0/asm=5,313/review=526 비현실적 결과로 폐기됨.
원인: OCR 치수 검출 실패 + BOM false positive.

본 모듈은 **Stage 1 Version A 모델 (mAP 0.9364)** 의 추론 결과
(`outputs/auto_labels/labels/*.txt`) 를 활용해 PMI 카운트 기반 분류:

    PMI ≥ 5                            → manufacturing
    PMI < 5 AND (Iso ≥ 1 OR Table ≥ 3) → assembly (★ 사람 검수 후보)
    PMI < 5 AND no signal              → manual_review_type (★ 사람 검수)

5,839장 분포 분석 (실측):
- manufacturing: 5,349장 (91.6%)
- assembly:        441장 (7.6%)   ← 검수 대상
- manual_review:    49장 (0.8%)   ← 검수 대상

산출물
------
::

    outputs/sort_by_yolo_pmi/
    ├── manifest.csv                    ← 전체 분류 + per-class counts (UTF-8-SIG)
    ├── README.md                       ← 검수 가이드
    ├── manufacturing/                  ← symlinks (~5,349장, 가공도면)
    ├── assembly/                       ← symlinks (~441장, ★ 검수 대상)
    └── manual_review/                  ← symlinks (~49장, ★ 검수 대상)

CLI
---
::

    # 기본 (전체 분류 + symlink)
    python src/sort_by_yolo_pmi.py

    # manifest 만 (폴더 안 만듦)
    python src/sort_by_yolo_pmi.py --no-folders

    # symlink 대신 이미지 복사 (Windows / 디스크 부담 시)
    python src/sort_by_yolo_pmi.py --copy-images

    # 디버그: 처음 N장만
    python src/sort_by_yolo_pmi.py --limit 100

관련 의사결정
-------------
- D-024 Group-aware split (group_key = filename.split('.rf.')[0])
- D-026 가공/조립 분류 (휴리스틱 폐기 → 본 모듈 대체)
- D-028 Stage 1 5 클래스 (Isometric/PMI/Table/Text/View)
- D-031 PMI dominant 분포 (실측 80.6%)
- D-035 Pre-annotation 스킵 — 본 모듈도 사람 검수 후 group_key 단위 제외
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs" / "auto_labels" / "labels"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sort_by_yolo_pmi"

# Stage 1 클래스 (D-028, Roboflow data.yaml 순서)
CLASS_NAMES: List[str] = ["Isometric", "PMI", "Table", "Text", "View"]
CLS_ISO, CLS_PMI, CLS_TABLE, CLS_TEXT, CLS_VIEW = 0, 1, 2, 3, 4

# 분류 임계값 (D-031 실측 기반)
PMI_MFG_MIN = 5         # PMI ≥ 5 → manufacturing
TABLE_ASM_MIN = 3       # Table ≥ 3 + PMI<5 → assembly 시그널 (BOM/Rev 다수)
ISO_ASM_MIN = 1         # Isometric ≥ 1 + PMI<5 → assembly 시그널 (3D 투영도)

DECISIONS = ["manufacturing", "assembly", "manual_review_type"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sort_by_yolo_pmi")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_yolo_label(path: Path) -> Dict[int, int]:
    """Parse YOLO det txt → class id 별 카운트 (5 클래스).

    Returns
    -------
    {0: iso_count, 1: pmi_count, 2: table_count, 3: text_count, 4: view_count}
    """
    counts = {i: 0 for i in range(5)}
    if not path.exists():
        return counts
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                continue
            cls = int(parts[0])
            if 0 <= cls < 5:
                counts[cls] += 1
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to parse %s: %s", path.name, e)
    return counts


def classify(counts: Dict[int, int]) -> Tuple[str, str]:
    """카운트 → (decision, reason)."""
    pmi = counts[CLS_PMI]
    iso = counts[CLS_ISO]
    table = counts[CLS_TABLE]

    if pmi >= PMI_MFG_MIN:
        return "manufacturing", f"PMI={pmi}≥{PMI_MFG_MIN}"
    # PMI < 5 영역
    if iso >= ISO_ASM_MIN and table >= TABLE_ASM_MIN:
        return "assembly", f"PMI={pmi}<{PMI_MFG_MIN}, Iso={iso}≥1, Table={table}≥{TABLE_ASM_MIN}"
    if iso >= ISO_ASM_MIN:
        return "assembly", f"PMI={pmi}<{PMI_MFG_MIN}, Iso={iso}≥1 (3D view)"
    if table >= TABLE_ASM_MIN:
        return "assembly", f"PMI={pmi}<{PMI_MFG_MIN}, Table={table}≥{TABLE_ASM_MIN} (BOM/Rev)"
    return "manual_review_type", f"PMI={pmi}<{PMI_MFG_MIN}, Iso={iso}, Table={table} (모호)"


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> str:
    """symlink 또는 copy. 반환: 'link' / 'copy' / 'skip'."""
    if dst.exists():
        return "skip"
    if copy:
        shutil.copy2(src, dst)
        return "copy"
    try:
        dst.symlink_to(src.resolve())
        return "link"
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        return "copy"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def write_readme(out_dir: Path, stats: Dict[str, int]) -> None:
    """검수 가이드 README.md 자동 생성."""
    txt = f"""# Sort by YOLO PMI — 검수 가이드

> Stage 1 Version A 자동 라벨 기반 분류 결과. **D-026 대체** (2026-04-29).

## 분류 결과

| Decision | 도면 수 | 비율 | 검수 필요 |
|---|---|---|---|
| manufacturing | {stats.get('manufacturing', 0):,} | {stats.get('manufacturing', 0)/sum(stats.values())*100:.1f}% | ✗ (학습 유지) |
| **assembly** | {stats.get('assembly', 0):,} | {stats.get('assembly', 0)/sum(stats.values())*100:.1f}% | **★ 검수 대상** |
| **manual_review_type** | {stats.get('manual_review_type', 0):,} | {stats.get('manual_review_type', 0)/sum(stats.values())*100:.1f}% | **★ 검수 대상** |

## 분류 규칙

```
PMI ≥ 5                            → manufacturing (가공도면)
PMI < 5 AND (Iso ≥ 1 OR Table ≥ 3) → assembly (조립도면 후보)
PMI < 5 AND signal 없음            → manual_review_type (모호)
```

## 검수 절차

### 1. assembly/ 폴더 시각 확인 (~40분)

Windows Explorer 큰 아이콘 보기 → 도면 한눈에 확인.
조립도면 명백한 것만 추려서 group_key 기록.

### 2. manual_review/ 폴더 확인 (~5분)

49장 (모호 케이스). 마찬가지로 조립도면이면 기록.

### 3. exclude_list.txt 작성

```
# outputs/exclude_list.txt
# group_key (filename.split('.rf.')[0]) 만 기록 — 같은 원본의 모든 변형 자동 제외
0301040003_SHAFT-ARMATURE_REV-01_page_1_png
0_700bar-1-_000001_jpg
...
```

### 4. 자동 group 단위 제외

```bash
python src/exclude_groups.py --list outputs/exclude_list.txt --dryrun
# 미리보기 OK 면:
python src/exclude_groups.py --list outputs/exclude_list.txt
```

dataset/ → dataset_excluded/ 로 이동 (삭제 X, 보존).

## 트리비아

- assembly 후보 ~441장 중 실제 조립도면 ~100장 추정 (사용자 직관)
- False positive ~340장 = 부품도면 / 간단한 가공도면 / 3D 위주 도면
- 사람 검수로 false positive 제거 → 정확한 조립도면 ~100 group 추출

관련 의사결정: D-024 / D-026 / D-031 / D-035 (PROJECT_HANDOFF.md §11)
"""
    (out_dir / "README.md").write_text(txt, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="D-026 대체 — Stage 1 Version A 자동 라벨 기반 가공/조립 분류",
    )
    p.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR,
                   help=f"YOLO det 라벨 폴더 (default: {DEFAULT_LABELS_DIR})")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                   help=f"원본 이미지 폴더 (default: {DEFAULT_DATASET_DIR})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"출력 폴더 (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--no-folders", action="store_true",
                   help="symlink/copy 안 하고 manifest CSV 만 생성")
    p.add_argument("--copy-images", action="store_true",
                   help="모든 폴더 copy (manufacturing 도 copy — 디스크 ~3GB 추가)")
    p.add_argument("--all-symlink", action="store_true",
                   help="모든 폴더 symlink (Windows Explorer 검은 화면 위험 — Linux 전용)")
    p.add_argument("--limit", type=int, default=None,
                   help="처리 제한 (디버깅용)")
    args = p.parse_args()

    # --- Validate ----
    if not args.labels_dir.exists():
        log.error("Labels dir not found: %s", args.labels_dir)
        log.error("auto_label_stage1.py 를 먼저 실행 필요.")
        return 2
    if not args.dataset_dir.exists() and not args.no_folders:
        log.error("Dataset dir not found: %s", args.dataset_dir)
        return 2

    # --- 라벨 파일 스캔 ----
    label_files = sorted(args.labels_dir.glob("*.txt"))
    log.info("Found %d label files in %s", len(label_files), args.labels_dir)

    if args.limit and args.limit > 0:
        label_files = label_files[:args.limit]
        log.info("Limited to first %d", len(label_files))

    if not label_files:
        log.error("No label files. Exit.")
        return 1

    # --- 출력 디렉터리 준비 ----
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.no_folders:
        for dec in DECISIONS:
            (args.output / dec).mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"

    log.info("Output: %s", args.output)
    log.info("  manifest: %s", manifest_path)
    if not args.no_folders:
        log.info("  folders:  %s", ", ".join(DECISIONS))
        log.info("  mode:     %s", "copy" if args.copy_images else "symlink")

    # --- 분류 ----
    records: List[Dict[str, str]] = []
    decision_counts: Dict[str, int] = {d: 0 for d in DECISIONS}
    link_stats = {"link": 0, "copy": 0, "skip": 0, "missing_image": 0}

    pbar = tqdm(label_files, desc="Classifying", unit="img",
                dynamic_ncols=True, leave=True)
    for txt_path in pbar:
        stem = txt_path.stem
        counts = parse_yolo_label(txt_path)
        decision, reason = classify(counts)
        decision_counts[decision] += 1

        # 원본 이미지 매칭
        img_path = args.dataset_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = args.dataset_dir / f"{stem}.jpeg"

        # symlink/copy
        # 기본 정책 (WSL2 호환):
        #   - manufacturing  → symlink (5,349장, 디스크 절약)
        #   - assembly       → copy   (검수용 — Windows Explorer 호환)
        #   - manual_review  → copy   (검수용)
        # --copy-images 시 모두 copy / --all-symlink 시 모두 symlink
        if not args.no_folders and img_path.exists():
            dst = args.output / decision / img_path.name
            if args.copy_images:
                use_copy = True
            elif args.all_symlink:
                use_copy = False
            else:
                # 기본: 검수 폴더만 copy (Windows 호환)
                use_copy = decision in ("assembly", "manual_review_type")
            mode = link_or_copy(img_path, dst, copy=use_copy)
            link_stats[mode] += 1
        elif not args.no_folders:
            link_stats["missing_image"] += 1

        # group key (D-024)
        group_key = stem.split(".rf.")[0]

        records.append({
            "filename": img_path.name,
            "group_key": group_key,
            "iso_count": counts[CLS_ISO],
            "pmi_count": counts[CLS_PMI],
            "table_count": counts[CLS_TABLE],
            "text_count": counts[CLS_TEXT],
            "view_count": counts[CLS_VIEW],
            "n_total": sum(counts.values()),
            "decision": decision,
            "reason": reason,
        })

        pbar.set_postfix(
            mfg=decision_counts["manufacturing"],
            asm=decision_counts["assembly"],
            review=decision_counts["manual_review_type"],
        )

    # --- Manifest CSV ----
    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "group_key",
            "iso_count", "pmi_count", "table_count", "text_count", "view_count",
            "n_total", "decision", "reason",
        ])
        writer.writeheader()
        writer.writerows(records)

    # --- README.md ----
    if not args.no_folders:
        write_readme(args.output, decision_counts)

    # --- Summary ----
    n_total = len(records)
    log.info("=" * 60)
    log.info("Classification complete: %d drawings", n_total)
    log.info("  manufacturing      : %5d (%.1f%%)",
             decision_counts["manufacturing"],
             decision_counts["manufacturing"] / n_total * 100)
    log.info("  assembly           : %5d (%.1f%%) ★ 검수 대상",
             decision_counts["assembly"],
             decision_counts["assembly"] / n_total * 100)
    log.info("  manual_review_type : %5d (%.1f%%) ★ 검수 대상",
             decision_counts["manual_review_type"],
             decision_counts["manual_review_type"] / n_total * 100)
    if not args.no_folders:
        log.info("Symlink stats: %s", link_stats)
    log.info("=" * 60)
    log.info("Manifest: %s", manifest_path)
    if not args.no_folders:
        log.info("Folders:")
        for dec in DECISIONS:
            n = decision_counts[dec]
            log.info("  %s  (%d images)", args.output / dec, n)
        log.info("README: %s", args.output / "README.md")
    log.info("")
    log.info("[다음 단계]")
    log.info("  1. assembly/ 폴더 시각 확인 (Windows Explorer 큰 아이콘)")
    log.info("  2. manual_review/ 폴더 확인")
    log.info("  3. 조립도면 group_key -> outputs/exclude_list.txt 작성")
    log.info("  4. python src/exclude_groups.py --list outputs/exclude_list.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
