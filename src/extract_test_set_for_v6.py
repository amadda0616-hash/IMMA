"""
src/extract_test_set_for_v6.py

Phase 16c V6 검증용 test set 추출.

stage3_numerical.py 의 split 로직 (seed=42, group-aware D-024) 동일하게 적용하여
data/vlm/numerical/ 의 11,470 region 중 test set (~10%, 1,187 region 예상) 만 추출.

출력 구조 (★ stage3_numerical.py batch CLI 호환):
    outputs/test_set_v6/
        Measure/<id>.jpg
        GDT/<id>.jpg
        Roughness/<id>.jpg
        gt/<id>.json    ← V6 --gt 입력 (auto_filled completed=True 만)

CLI:
    PYTHONPATH=. python src/extract_test_set_for_v6.py
"""
from __future__ import annotations

import shutil
import sys
from collections import Counter
from pathlib import Path

# Project root bootstrap (D-049 패턴)
_PROJECT_ROOT_BOOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_BOOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_BOOT))

from src.stage3_numerical import discover_samples, split_samples  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_DIR = PROJECT_ROOT / "data" / "vlm" / "numerical"
OUT_DIR = PROJECT_ROOT / "outputs" / "test_set_v6"


def main() -> int:
    if not NUMERICAL_DIR.exists():
        print(f"ERROR: {NUMERICAL_DIR} not found")
        return 2

    print(f"Discovering samples under {NUMERICAL_DIR} ...")
    samples = discover_samples(NUMERICAL_DIR)
    print(f"  total: {len(samples)}")

    # Phase 16b 학습과 동일 split (configs/donut_numerical.yaml: 70/20/10, seed=42)
    train_s, val_s, test_s = split_samples(samples, ratios=(0.7, 0.2, 0.1), seed=42)
    print(f"  split: train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    # ★ completed=True 만 사용 (auto_fill 성공한 region — V6 GT 신뢰성)
    test_completed = [
        s for s in test_s
        if s.json_data.get("_review", {}).get("completed", False)
    ]
    print(f"  test completed (auto_filled GT): {len(test_completed)} / {len(test_s)} "
          f"({len(test_completed)/len(test_s)*100:.1f}%)")

    # 클래스 분포
    cls_counter = Counter(s.region_class for s in test_completed)
    print(f"  test completed 클래스 분포:")
    for cls, n in sorted(cls_counter.items()):
        print(f"    {cls}: {n}")

    # ---- 출력 ----
    if OUT_DIR.exists():
        print(f"  ★ {OUT_DIR} 이미 존재 — 기존 파일 덮어씀")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt_dir = OUT_DIR / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    skipped = 0
    for s in test_completed:
        cls_dir = OUT_DIR / s.region_class
        cls_dir.mkdir(parents=True, exist_ok=True)

        # 이미지 copy
        try:
            shutil.copy2(s.image_path, cls_dir / s.image_path.name)
        except OSError as e:
            print(f"  ★ skip image {s.image_path.name}: {e}")
            skipped += 1
            continue

        # GT JSON copy
        json_src = s.image_path.with_suffix(".json")
        if json_src.exists():
            shutil.copy2(json_src, gt_dir / json_src.name)
        else:
            print(f"  ★ no JSON for {s.image_path.name}")

        counts[s.region_class] += 1

    print(f"\n★ Test set V6 추출 완료: {OUT_DIR}")
    print(f"  Measure: {counts.get('Measure', 0)}")
    print(f"  GDT: {counts.get('GDT', 0)}")
    print(f"  Roughness: {counts.get('Roughness', 0)}")
    print(f"  Total: {sum(counts.values())}")
    if skipped:
        print(f"  skipped: {skipped}")
    print(f"\n다음 명령:")
    print(f"  # 1. Stage 3-N batch 추론")
    print(f"  PYTHONPATH=. python src/stage3_numerical.py batch \\")
    print(f"      --input-dir {OUT_DIR} \\")
    print(f"      --ckpt checkpoints/donut_numerical/final \\")
    print(f"      --device 0 \\")
    print(f"      --out-dir outputs/stage3n_baseline_v1_predictions/")
    print(f"")
    print(f"  # 2. V6 검증")
    print(f"  PYTHONPATH=. python src/validate/check_stage3n_numerical.py \\")
    print(f"      --predictions outputs/stage3n_baseline_v1_predictions/ \\")
    print(f"      --gt {gt_dir} \\")
    print(f"      --reports-dir reports/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
