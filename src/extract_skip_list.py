"""
src/extract_skip_list.py

CVAT XML 에서 SKIP 라벨 frame 추출 + reason 카테고리별 분류.

배경
----
Stage 2 OBB 라벨링 시 다음 케이스를 SKIP 라벨로 마킹:
- 가독성 ↓ (unreadable)
- Stage 1 false positive (단면도/상세도/제3각법/Table/Notes/Isometric/기타)

이 도구는 CVAT XML export 를 파싱하여 reason 별로 분리된 파일 list 를 생성.
이 list 들은 다음에 활용:
1. **stage1_fp_notes.txt** → ★ rescue_misclassified_notes.py 에 입력 (D-038)
   - Notes 영역 오검출 → Donut OCR 로 텍스트 추출 → JSON 메타데이터에 병합
2. **all_skip.txt** → Stage 3 fine-tune 데이터 자동 제외
3. **summary.csv** → Stage 1 Version B 학습 시 카테고리별 보강 우선순위

지원 reason 카테고리
--------------------
- ``unreadable`` — 가독성 한계 (Type A)
- ``stage1_fp_section`` — 단면도 기호
- ``stage1_fp_detail`` — 상세도 기호
- ``stage1_fp_projection`` — 제3각법 기호
- ``stage1_fp_table`` — 표제란/BOM/도장
- ``stage1_fp_notes`` — ★ 일반 주석 (rescue 대상)
- ``stage1_fp_isometric`` — 등각도
- ``stage1_fp_other`` — 기타 Stage 1 false positive
- ``other`` — 그 외

CLI
---
::

    # 기본 (모든 reason 별 분리 출력)
    python src/extract_skip_list.py \
        --xml outputs/cvat_stage2_v3_final.xml \
        --output-dir outputs/skip_lists/

    # CVAT XML 만 입력 (frame 번호 list 등 보조 입력 안 씀)
    python src/extract_skip_list.py \
        --xml outputs/annotations.xml \
        --output-dir outputs/skip_lists/

산출물
------
::

    outputs/skip_lists/
    ├── unreadable.txt
    ├── stage1_fp_section.txt
    ├── stage1_fp_detail.txt
    ├── stage1_fp_projection.txt
    ├── stage1_fp_table.txt
    ├── stage1_fp_notes.txt          ← ★ rescue 대상 (D-038)
    ├── stage1_fp_isometric.txt
    ├── stage1_fp_other.txt
    ├── other.txt
    ├── all_skip.txt                  ← 전체 통합 (Stage 3 제외 list)
    └── summary.csv                   ← 카테고리별 카운트

관련 의사결정
-------------
- D-037 adaptive padding (v3 base)
- D-038 Stage 1 fp Notes rescue (★ stage1_fp_notes.txt 생성 필수)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKIP_LABEL = "SKIP"
REASON_ATTR = "reason"
DEFAULT_REASON = "stage1_fp_other"  # CVAT 라벨 정의의 default_value 와 일치

# 알려진 reason 카테고리 (rescue 분기 결정에 사용)
KNOWN_REASONS = {
    "unreadable",
    "stage1_fp_section",
    "stage1_fp_detail",
    "stage1_fp_projection",
    "stage1_fp_table",
    "stage1_fp_notes",       # ★ D-038 rescue 대상
    "stage1_fp_isometric",
    "stage1_fp_other",
    "other",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_skip_list")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_basename(image_name: str) -> str:
    """
    CVAT image name 에서 파일명만 추출.

    예: "cvat_stage2_input_v3_upscaled/DwgFoo__PMI_005.jpg" → "DwgFoo__PMI_005.jpg"
    """
    return Path(image_name).name


def extract_reason(box_elem: ET.Element) -> str:
    """
    <box> element 안의 <attribute name="reason"> 값 추출.

    없으면 default_value (stage1_fp_other) 반환.
    """
    for attr in box_elem.findall("attribute"):
        if attr.get("name") == REASON_ATTR:
            text = (attr.text or "").strip()
            if text:
                return text
    return DEFAULT_REASON


# ---------------------------------------------------------------------------
# Main parsing
# ---------------------------------------------------------------------------
def parse_cvat_xml(xml_path: Path) -> Dict[str, Set[str]]:
    """
    CVAT XML 을 파싱하여 reason 별로 crop filename set 반환.

    Returns
    -------
    dict[str, set[str]]
        {
            "unreadable": {"DwgFoo__PMI_005.jpg", ...},
            "stage1_fp_notes": {...},
            ...
        }
    """
    log.info("Reading: %s", xml_path)
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        log.error("XML parse error: %s", e)
        return {}

    root = tree.getroot()
    if root.tag != "annotations":
        log.error("Expected root <annotations>, got <%s>", root.tag)
        return {}

    by_reason: Dict[str, Set[str]] = {}
    n_images = 0
    n_skip_boxes = 0
    n_unknown_reason = 0
    unknown_values: Counter = Counter()

    for img in root.findall("image"):
        n_images += 1
        img_name = img.get("name")
        if not img_name:
            continue
        basename = get_basename(img_name)

        # SKIP 라벨이 있는 박스만 추출
        for box in img.findall("box"):
            label = box.get("label")
            if label != SKIP_LABEL:
                continue
            n_skip_boxes += 1

            reason = extract_reason(box)
            if reason not in KNOWN_REASONS:
                n_unknown_reason += 1
                unknown_values[reason] += 1
                # 알 수 없는 값도 그대로 분류 (사용자 정의 가능)
            by_reason.setdefault(reason, set()).add(basename)

    log.info("Total images parsed     : %d", n_images)
    log.info("Total SKIP boxes        : %d", n_skip_boxes)
    if n_unknown_reason > 0:
        log.warning("Unknown reason values   : %d", n_unknown_reason)
        for r, c in unknown_values.most_common():
            log.warning("  - %s : %d", r, c)

    return by_reason


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_reason_files(by_reason: Dict[str, Set[str]],
                       output_dir: Path) -> Dict[str, int]:
    """
    reason 별 .txt 파일 생성. 각 파일은 alphabetic sort 된 filename list.

    Returns
    -------
    dict[str, int]
        파일명별 라인 수
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}

    for reason, files in by_reason.items():
        out_path = output_dir / f"{reason}.txt"
        sorted_files = sorted(files)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# SKIP reason: {reason}\n")
            f.write(f"# Count: {len(sorted_files)}\n")
            f.write(f"# Source: CVAT XML SKIP 라벨\n")
            for fn in sorted_files:
                f.write(f"{fn}\n")
        counts[reason] = len(sorted_files)
        log.info("  %-30s : %d files → %s", reason, len(sorted_files), out_path.name)

    return counts


def write_all_skip(by_reason: Dict[str, Set[str]], output_dir: Path) -> int:
    """
    모든 SKIP 파일을 통합한 all_skip.txt 생성.
    Stage 3 자동 제외 list 로 활용.
    """
    all_files: Set[str] = set()
    for files in by_reason.values():
        all_files.update(files)

    out_path = output_dir / "all_skip.txt"
    sorted_files = sorted(all_files)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# All SKIP crops (전체 SKIP 통합)\n")
        f.write(f"# Count: {len(sorted_files)}\n")
        f.write(f"# 사용처: Stage 3 fine-tune 데이터 자동 제외\n")
        for fn in sorted_files:
            f.write(f"{fn}\n")

    log.info("  %-30s : %d files → %s", "all_skip", len(sorted_files), out_path.name)
    return len(sorted_files)


def write_summary_csv(by_reason: Dict[str, Set[str]], output_dir: Path,
                      total_crops: int = 0) -> None:
    """
    summary.csv 생성. 카테고리별 카운트 + 비율.
    """
    out_path = output_dir / "summary.csv"
    counts = [(reason, len(files)) for reason, files in sorted(by_reason.items())]
    total_skip = sum(c for _, c in counts)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count", "ratio_of_skip", "ratio_of_total"])
        for reason, count in counts:
            ratio_skip = (count / total_skip * 100) if total_skip > 0 else 0
            ratio_total = (count / total_crops * 100) if total_crops > 0 else 0
            writer.writerow([
                reason,
                count,
                f"{ratio_skip:.2f}%",
                f"{ratio_total:.2f}%" if total_crops > 0 else "N/A",
            ])
        # 합계 행
        writer.writerow([
            "TOTAL_SKIP",
            total_skip,
            "100.00%",
            f"{total_skip / total_crops * 100:.2f}%" if total_crops > 0 else "N/A",
        ])
        if total_crops > 0:
            writer.writerow([
                "TOTAL_CROPS",
                total_crops,
                "—",
                "100.00%",
            ])

    log.info("  %-30s : %d categories → %s", "summary.csv", len(counts), out_path.name)


# ---------------------------------------------------------------------------
# Verbose stats
# ---------------------------------------------------------------------------
def print_summary(by_reason: Dict[str, Set[str]], total_crops: int = 0) -> None:
    """
    콘솔 요약 출력. rescue 대상 (stage1_fp_notes) 강조.
    """
    log.info("=" * 60)
    log.info("SKIP 카테고리 분포")
    log.info("=" * 60)

    total_skip = sum(len(files) for files in by_reason.values())

    # 카테고리별 정렬 (count 내림차순)
    sorted_reasons = sorted(by_reason.items(), key=lambda x: -len(x[1]))

    for reason, files in sorted_reasons:
        count = len(files)
        ratio = (count / total_skip * 100) if total_skip > 0 else 0
        marker = ""
        if reason == "stage1_fp_notes":
            marker = " ★ RESCUE 대상 (D-038)"
        elif reason == "unreadable":
            marker = " (Type A: 가독성 한계)"
        elif reason.startswith("stage1_fp_"):
            marker = " (Type B: Stage 1 FP)"

        log.info("  %-25s : %4d (%5.2f%%)%s", reason, count, ratio, marker)

    log.info("-" * 60)
    log.info("  %-25s : %4d", "TOTAL_SKIP", total_skip)
    if total_crops > 0:
        log.info("  %-25s : %4d", "TOTAL_CROPS", total_crops)
        log.info("  SKIP ratio of total     : %.2f%%",
                 total_skip / total_crops * 100)
    log.info("=" * 60)

    # ★ rescue 안내
    if "stage1_fp_notes" in by_reason and len(by_reason["stage1_fp_notes"]) > 0:
        n_rescue = len(by_reason["stage1_fp_notes"])
        log.info("")
        log.info("[★ 다음 단계 — D-038 Notes Rescue]")
        log.info("  %d 개의 stage1_fp_notes crop 을 Donut OCR 로 처리:", n_rescue)
        log.info("  python src/rescue_misclassified_notes.py \\")
        log.info("      --skip-list outputs/skip_lists/stage1_fp_notes.txt \\")
        log.info("      --crops-dir outputs/cvat_stage2_input_v3_upscaled/ \\")
        log.info("      --output outputs/rescued_notes.json")
        log.info("")
        log.info("  → 결과는 최종 JSON 의 'general_notes' 필드에 병합됨")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="CVAT XML 에서 SKIP 라벨 추출 + reason 카테고리별 분류 (D-038 rescue 지원)",
    )
    p.add_argument("--xml", type=Path, required=True,
                   help="CVAT export XML (annotations.xml)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/skip_lists"),
                   help="출력 폴더 (default: outputs/skip_lists)")
    p.add_argument("--total-crops", type=int, default=0,
                   help="전체 crop 수 (비율 계산용, default: XML에서 자동)")
    args = p.parse_args()

    if not args.xml.exists():
        log.error("Input XML not found: %s", args.xml)
        return 1

    log.info("=" * 60)
    log.info("SKIP 라벨 추출 + 분류")
    log.info("  Input      : %s", args.xml)
    log.info("  Output dir : %s", args.output_dir)
    log.info("=" * 60)

    # XML 파싱
    by_reason = parse_cvat_xml(args.xml)
    if not by_reason:
        log.warning("SKIP 라벨이 없거나 파싱 실패. 종료.")
        return 0

    # total_crops 자동 계산 (XML 안 모든 image)
    total_crops = args.total_crops
    if total_crops == 0:
        try:
            tree = ET.parse(args.xml)
            total_crops = len(tree.getroot().findall("image"))
        except ET.ParseError:
            total_crops = 0

    # 출력
    log.info("")
    log.info("Writing output files ...")
    counts = write_reason_files(by_reason, args.output_dir)
    write_all_skip(by_reason, args.output_dir)
    write_summary_csv(by_reason, args.output_dir, total_crops)

    # 요약
    log.info("")
    print_summary(by_reason, total_crops)

    return 0


if __name__ == "__main__":
    sys.exit(main())
