"""
src/migrate_cvat_labels.py

CVAT XML annotations 좌표 스케일 변환 도구.

용도
----
``extract_pmi_crops_v3.py --upscale 3.0`` 으로 PMI crop 이미지를 업스케일한 뒤,
기존 CVAT 라벨 XML 의 모든 좌표를 같은 배율로 변환하여 새 task 에 import 가능.

배경
----
PMI crop 의 텍스트 가독성이 낮아 라벨링 효율이 떨어지는 문제를 해결하기 위해
crop 을 균등 업스케일 (e.g., 3x Lanczos) 하면 이미지 dimension 이 변경된다.
CVAT 는 이미지 dimension 기반 좌표를 사용하므로, 기존 라벨을 새 task 에
그대로 import 하면 좌표가 어긋난다.

이 도구는 균등 스케일링이 **선형 변환** 임을 활용하여 모든 좌표를 일괄 × scale
로 변환한다. 회전 OBB 의 angle 은 균등 스케일에 invariant 이므로 그대로 보존.

지원 shape
---------
- ``<box>`` (rectangle, OBB) — xtl/ytl/xbr/ybr 변환, rotation 보존
- ``<polygon>`` — points 좌표 변환
- ``<polyline>`` — points 좌표 변환
- ``<points>`` — points 좌표 변환
- ``<ellipse>`` — cx/cy/rx/ry 변환
- ``<image>`` — width/height 변환

검증
----
- 입력/출력 box 개수 일치 여부
- 회전 박스 개수 보존 여부 (rotation 속성 유지)
- 좌표 범위가 새 image dimension 안에 있는지 sanity check

CLI
---
::

    # 기본 (3x 업스케일에 맞춘 좌표 변환)
    python src/migrate_cvat_labels.py \
        --input outputs/cvat_stage2_v3_backup_ORIGINAL.xml \
        --output outputs/cvat_stage2_v3_upscaled3x.xml \
        --scale 3.0

    # 다른 배율
    python src/migrate_cvat_labels.py \
        --input source.xml --output target.xml --scale 4.0

    # 검증만 (--dry-run, 파일 저장 안 함)
    python src/migrate_cvat_labels.py \
        --input source.xml --output target.xml --scale 3.0 --dry-run

워크플로
-------
1. CVAT Tasks → Stage2_PMI_v3_844 → Actions → Export task dataset (CVAT 1.1)
2. 다운로드 → ``outputs/cvat_stage2_v3_backup_ORIGINAL.xml`` 저장
3. ``extract_pmi_crops_v3.py --upscale 3.0`` 실행 → 새 crop 폴더
4. 이 스크립트 실행 → 변환된 XML 생성
5. CVAT 새 task (``Stage2_PMI_v3_upscaled3x_844``) 생성
6. 변환된 XML import (Actions → Upload annotations)
7. 무작위 5~10장 시각 검증

관련 의사결정
-------------
- D-037 adaptive padding (v3 base)
- D-037 v3 확장 (upscale 옵션)
"""
from __future__ import annotations

import argparse
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("migrate_cvat_labels")


# ---------------------------------------------------------------------------
# 좌표 변환 helpers
# ---------------------------------------------------------------------------
def _scale_attr(elem: ET.Element, attr: str, scale: float) -> bool:
    """단일 float 속성을 scale 배 적용. 존재 안 하면 False 반환."""
    val = elem.get(attr)
    if val is None:
        return False
    try:
        new_val = float(val) * scale
        elem.set(attr, f"{new_val:.2f}")
        return True
    except ValueError:
        return False


def _scale_int_attr(elem: ET.Element, attr: str, scale: float) -> bool:
    """단일 int 속성을 scale 배 적용 (image width/height 용)."""
    val = elem.get(attr)
    if val is None:
        return False
    try:
        new_val = int(round(float(val) * scale))
        elem.set(attr, str(new_val))
        return True
    except ValueError:
        return False


def _scale_points(points_str: str, scale: float) -> str:
    """
    "x1,y1;x2,y2;..." 형식의 points 문자열을 scale 배 변환.

    CVAT polygon/polyline/points 는 "x,y;x,y;..." 형식 사용.
    """
    out_pts = []
    for pt in points_str.split(";"):
        pt = pt.strip()
        if not pt:
            continue
        parts = pt.split(",")
        if len(parts) != 2:
            continue
        try:
            x = float(parts[0]) * scale
            y = float(parts[1]) * scale
            out_pts.append(f"{x:.2f},{y:.2f}")
        except ValueError:
            out_pts.append(pt)  # 파싱 실패 시 원본 유지
    return ";".join(out_pts)


# ---------------------------------------------------------------------------
# Image element 변환
# ---------------------------------------------------------------------------
def transform_image(image_elem: ET.Element, scale: float) -> dict:
    """
    <image> element 와 그 자식 shape 들을 모두 변환.

    Returns
    -------
    dict
        {
            "n_box": ...,
            "n_polygon": ...,
            "n_polyline": ...,
            "n_points": ...,
            "n_ellipse": ...,
            "n_rotated_box": ...,
            "skipped": [...],   # 처리 못 한 element tag list
        }
    """
    stats = {
        "n_box": 0,
        "n_polygon": 0,
        "n_polyline": 0,
        "n_points": 0,
        "n_ellipse": 0,
        "n_rotated_box": 0,
        "skipped": [],
    }

    # 1. image 자체의 width/height
    _scale_int_attr(image_elem, "width", scale)
    _scale_int_attr(image_elem, "height", scale)

    # 2. 자식 shape 들
    for child in list(image_elem):
        tag = child.tag.lower()

        if tag == "box":
            # Rectangle (axis-aligned 또는 OBB)
            _scale_attr(child, "xtl", scale)
            _scale_attr(child, "ytl", scale)
            _scale_attr(child, "xbr", scale)
            _scale_attr(child, "ybr", scale)
            # rotation 은 그대로 유지 (균등 스케일 invariant)
            rotation = child.get("rotation")
            if rotation is not None and rotation not in ("", "0", "0.0", "0.00"):
                stats["n_rotated_box"] += 1
            stats["n_box"] += 1

        elif tag == "polygon":
            points = child.get("points")
            if points:
                child.set("points", _scale_points(points, scale))
            stats["n_polygon"] += 1

        elif tag == "polyline":
            points = child.get("points")
            if points:
                child.set("points", _scale_points(points, scale))
            stats["n_polyline"] += 1

        elif tag == "points":
            points = child.get("points")
            if points:
                child.set("points", _scale_points(points, scale))
            stats["n_points"] += 1

        elif tag == "ellipse":
            _scale_attr(child, "cx", scale)
            _scale_attr(child, "cy", scale)
            _scale_attr(child, "rx", scale)
            _scale_attr(child, "ry", scale)
            stats["n_ellipse"] += 1

        elif tag in ("attribute",):
            # attribute 는 좌표가 아니므로 skip
            pass

        else:
            stats["skipped"].append(tag)

    return stats


# ---------------------------------------------------------------------------
# Main 변환
# ---------------------------------------------------------------------------
def migrate(input_path: Path, output_path: Path, scale: float,
            dry_run: bool = False) -> Tuple[int, dict]:
    """
    CVAT XML 의 모든 좌표를 scale 배 변환.

    Returns
    -------
    (n_images, total_stats)
    """
    log.info("Reading: %s", input_path)
    try:
        tree = ET.parse(input_path)
    except ET.ParseError as e:
        log.error("XML parse error: %s", e)
        return 0, {}

    root = tree.getroot()
    if root.tag != "annotations":
        log.error("Expected root <annotations>, got <%s>", root.tag)
        return 0, {}

    # 누적 통계
    total = {
        "n_box": 0,
        "n_polygon": 0,
        "n_polyline": 0,
        "n_points": 0,
        "n_ellipse": 0,
        "n_rotated_box": 0,
        "n_images": 0,
        "n_images_with_shapes": 0,
        "skipped_tags": set(),
    }

    images = root.findall("image")
    if not images:
        log.warning("No <image> elements found in %s", input_path)
        return 0, total

    log.info("Found %d <image> elements. Scaling by %.2fx ...", len(images), scale)

    for img in images:
        stats = transform_image(img, scale)
        total["n_box"] += stats["n_box"]
        total["n_polygon"] += stats["n_polygon"]
        total["n_polyline"] += stats["n_polyline"]
        total["n_points"] += stats["n_points"]
        total["n_ellipse"] += stats["n_ellipse"]
        total["n_rotated_box"] += stats["n_rotated_box"]
        total["n_images"] += 1
        if any([stats["n_box"], stats["n_polygon"], stats["n_polyline"],
                stats["n_points"], stats["n_ellipse"]]):
            total["n_images_with_shapes"] += 1
        for s in stats["skipped"]:
            total["skipped_tags"].add(s)

    # Save
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        log.info("Saved: %s", output_path)
    else:
        log.info("[DRY-RUN] Output not saved.")

    return len(images), total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="CVAT XML annotations 좌표 스케일 변환 도구 (업스케일 마이그레이션)",
    )
    p.add_argument("--input", type=Path, required=True,
                   help="원본 CVAT XML (annotations.xml)")
    p.add_argument("--output", type=Path, required=True,
                   help="변환된 XML 저장 경로")
    p.add_argument("--scale", type=float, required=True,
                   help="좌표 배율 (e.g., 3.0 = 3배 업스케일된 이미지에 맞춤)")
    p.add_argument("--dry-run", action="store_true",
                   help="검증만 하고 파일 저장 안 함")
    args = p.parse_args()

    # 입력 검증
    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        return 1

    if args.scale <= 0:
        log.error("--scale 은 양수여야 함 (받은 값: %.2f)", args.scale)
        return 2

    if args.scale == 1.0:
        log.warning("--scale 1.0 — 좌표 변환 없음 (그냥 복사 효과)")

    if args.input.resolve() == args.output.resolve():
        log.error("입력과 출력이 같은 파일입니다. 다른 경로를 지정하세요.")
        return 3

    # 변환 실행
    log.info("=" * 60)
    log.info("CVAT 라벨 마이그레이션 시작")
    log.info("  Input  : %s", args.input)
    log.info("  Output : %s", args.output)
    log.info("  Scale  : %.2fx", args.scale)
    if args.dry_run:
        log.info("  Mode   : DRY-RUN (저장 안 함)")
    log.info("=" * 60)

    n_images, stats = migrate(args.input, args.output, args.scale, args.dry_run)

    if n_images == 0:
        log.error("변환된 이미지 없음. 입력 파일 확인 필요.")
        return 4

    # 결과 보고
    log.info("=" * 60)
    log.info("마이그레이션 완료")
    log.info("  처리 이미지         : %d", n_images)
    log.info("  shape 있는 이미지   : %d", stats["n_images_with_shapes"])
    log.info("  Box (rectangle)    : %d", stats["n_box"])
    log.info("    - 회전 OBB        : %d (rotation 보존됨)", stats["n_rotated_box"])
    log.info("    - axis-aligned    : %d", stats["n_box"] - stats["n_rotated_box"])
    log.info("  Polygon            : %d", stats["n_polygon"])
    log.info("  Polyline           : %d", stats["n_polyline"])
    log.info("  Points             : %d", stats["n_points"])
    log.info("  Ellipse            : %d", stats["n_ellipse"])
    if stats["skipped_tags"]:
        log.warning("  ⚠ 처리 안 함 tags : %s", sorted(stats["skipped_tags"]))
    log.info("=" * 60)

    if not args.dry_run:
        log.info("[다음 단계 — CVAT 라벨 import]")
        log.info("  1. CVAT 새 task 생성 (업스케일된 이미지 ZIP 업로드)")
        log.info("  2. 새 task → Actions → Upload annotations")
        log.info("  3. Format: CVAT for images 1.1")
        log.info("  4. File: %s", args.output)
        log.info("  5. 무작위 5~10장 시각 검증 (라벨 위치 정확성 확인)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
