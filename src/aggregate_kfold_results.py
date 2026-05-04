"""
src/aggregate_kfold_results.py

Stage 2 OBB K-fold Cross-Validation 결과 집계.

배경
----
- ``train_kfold.py`` 실행 후 5 fold 의 ``results.csv`` + ``best.pt`` 분석
- mAP@0.5 / Precision / Recall 등의 mean ± std 계산
- Best fold 식별 (validation mAP@0.5 최고)
- 최종 단일 모델 weight 결정 → ``checkpoints/yolo_obb.pt`` 자동 복사

처리 단계
---------
1. ``checkpoints/yolo_obb_runs/yolo_obb_v3_kfold_{0..K-1}/results.csv`` 파싱
2. 각 fold 의 final epoch 메트릭 추출
   - metrics/mAP50(B)
   - metrics/mAP50-95(B)
   - metrics/precision(B)
   - metrics/recall(B)
3. 평균/표준편차 계산
4. Best fold 식별 (mAP@0.5 기준)
5. ``outputs/kfold_summary.csv``, ``outputs/kfold_summary.json``,
   ``outputs/kfold_best_fold.txt`` 생성
6. Best fold 의 ``best.pt`` → ``checkpoints/yolo_obb.pt`` 복사 (선택)

CLI
---
::

    # 표준 집계
    python src/aggregate_kfold_results.py \\
        --runs-dir checkpoints/yolo_obb_runs \\
        --k 5 \\
        --output-dir outputs/

    # Best fold 의 best.pt 자동 복사 (★ pipeline.py 통합용)
    python src/aggregate_kfold_results.py \\
        --runs-dir checkpoints/yolo_obb_runs \\
        --k 5 \\
        --output-dir outputs/ \\
        --copy-best-to checkpoints/yolo_obb.pt

산출물
------
::

    outputs/
    ├── kfold_summary.csv      ← per-fold 메트릭 + mean/std 행
    ├── kfold_summary.json     ← 동일 (JSON)
    └── kfold_best_fold.txt    ← Best fold 번호 + best.pt 경로

관련 의사결정
-------------
- D-023 critical 임계값 (Measure missing < 8% / GDT < 5% / Roughness < 30%)
- Option F K-fold CV (Best fold 선정)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aggregate_kfold")

# 추출할 메트릭 컬럼 이름 (ultralytics results.csv 표준)
METRIC_COLUMNS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]
# 보조 컬럼 (선택)
EXTRA_COLUMNS = [
    "epoch",
    "train/box_loss",
    "train/cls_loss",
    "val/box_loss",
    "val/cls_loss",
]


# ---------------------------------------------------------------------------
# results.csv 파싱
# ---------------------------------------------------------------------------
def parse_fold_results(results_csv: Path) -> Optional[Dict[str, float]]:
    """
    한 fold 의 results.csv 마지막 epoch 메트릭 추출.

    Returns
    -------
    dict[str, float] or None
        키: METRIC_COLUMNS + EXTRA_COLUMNS
    """
    if not results_csv.exists():
        return None

    try:
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:  # noqa: BLE001
        log.error("Parse error %s: %s", results_csv, e)
        return None

    if not rows:
        return None

    # 마지막 epoch (최신 row)
    last_row = rows[-1]

    # mAP50 가 가장 높은 epoch (best epoch) 도 같이 추출
    best_row = last_row
    try:
        best_row = max(
            rows,
            key=lambda r: float(r.get("metrics/mAP50(B)", 0) or 0),
        )
    except (ValueError, KeyError):
        pass

    out: Dict[str, float] = {}
    for col in METRIC_COLUMNS:
        # ultralytics 컬럼명 변동 가능성: 공백 trim
        col_stripped = col.strip()
        # 우선 best_row, 없으면 last_row
        for row in (best_row, last_row):
            for k, v in row.items():
                if k.strip() == col_stripped:
                    try:
                        out[col] = float(v)
                        break
                    except (ValueError, TypeError):
                        continue
            if col in out:
                break

    # 보조 컬럼 (마지막 epoch 기준)
    for col in EXTRA_COLUMNS:
        for k, v in last_row.items():
            if k.strip() == col.strip():
                try:
                    out[col] = float(v)
                except (ValueError, TypeError):
                    pass
                break

    # best epoch 번호도 기록
    try:
        out["best_epoch"] = float(best_row.get("epoch", -1) or -1)
    except (ValueError, TypeError):
        pass

    return out


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------
def aggregate(
    fold_results: Dict[int, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    fold 별 메트릭 → mean/std/min/max 집계.

    Returns
    -------
    dict
        {
            "mAP50": {"mean": ..., "std": ..., "min": ..., "max": ...},
            ...
        }
    """
    summary: Dict[str, Dict[str, float]] = {}
    for col in METRIC_COLUMNS:
        values = [
            r[col] for r in fold_results.values() if col in r
        ]
        if not values:
            continue
        summary[col] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    return summary


def find_best_fold(
    fold_results: Dict[int, Dict[str, float]], metric: str = "metrics/mAP50(B)",
) -> Optional[Tuple[int, float]]:
    """가장 높은 metric 값을 가진 fold 식별."""
    candidates = [
        (fid, r[metric]) for fid, r in fold_results.items() if metric in r
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_summary_csv(
    out_path: Path,
    fold_results: Dict[int, Dict[str, float]],
    summary: Dict[str, Dict[str, float]],
    best_fold: Optional[Tuple[int, float]],
) -> None:
    """kfold_summary.csv 생성."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["fold"] + METRIC_COLUMNS + ["best_epoch"]
        writer.writerow(header)

        # Per-fold
        for fid in sorted(fold_results):
            r = fold_results[fid]
            row = [fid]
            for col in METRIC_COLUMNS:
                row.append(f"{r[col]:.4f}" if col in r else "—")
            row.append(int(r.get("best_epoch", -1)))
            writer.writerow(row)

        # 빈 행
        writer.writerow([])

        # Mean/std 행
        for stat in ("mean", "std", "min", "max"):
            row = [stat]
            for col in METRIC_COLUMNS:
                if col in summary:
                    row.append(f"{summary[col][stat]:.4f}")
                else:
                    row.append("—")
            row.append("")
            writer.writerow(row)

        # Best fold 행
        if best_fold:
            writer.writerow([])
            writer.writerow(["best_fold", best_fold[0], "mAP50:", f"{best_fold[1]:.4f}"])


def write_summary_json(
    out_path: Path,
    fold_results: Dict[int, Dict[str, float]],
    summary: Dict[str, Dict[str, float]],
    best_fold: Optional[Tuple[int, float]],
) -> None:
    """kfold_summary.json 생성."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "source": "aggregate_kfold_results.py",
            "decision": "Option F (5-fold CV)",
            "metric_for_best": "metrics/mAP50(B)",
        },
        "per_fold": {str(fid): r for fid, r in fold_results.items()},
        "summary": summary,
        "best_fold": {
            "fold": best_fold[0] if best_fold else None,
            "mAP50": best_fold[1] if best_fold else None,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def write_best_fold_txt(
    out_path: Path, best_fold: Optional[Tuple[int, float]], best_pt: Optional[Path],
) -> None:
    """kfold_best_fold.txt 생성."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not best_fold:
        out_path.write_text("(no fold available)\n", encoding="utf-8")
        return
    fid, mAP = best_fold
    content = (
        f"Best fold: {fid}\n"
        f"mAP@0.5: {mAP:.4f}\n"
        f"best.pt path: {best_pt}\n"
        f"\n"
        f"# pipeline.py 통합 시 사용:\n"
        f"#   --weights {best_pt}\n"
    )
    out_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage 2 OBB K-fold 결과 집계 (Option F)",
    )
    p.add_argument("--runs-dir", type=Path,
                   default=Path("checkpoints/yolo_obb_runs"),
                   help="ultralytics run 폴더 (default: checkpoints/yolo_obb_runs)")
    p.add_argument("--name-prefix", default="yolo_obb_v3_kfold_",
                   help="run name prefix (default: yolo_obb_v3_kfold_)")
    p.add_argument("--k", type=int, default=5, help="fold 수 (default: 5)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs"),
                   help="결과 저장 폴더 (default: outputs/)")
    p.add_argument("--copy-best-to", type=Path, default=None,
                   help="Best fold 의 best.pt 를 이 경로로 복사 "
                        "(e.g., checkpoints/yolo_obb.pt)")
    args = p.parse_args()

    if not args.runs_dir.exists():
        log.error("Runs dir not found: %s", args.runs_dir)
        return 1

    log.info("=" * 60)
    log.info("Stage 2 OBB K-fold 결과 집계")
    log.info("=" * 60)
    log.info("Runs dir   : %s", args.runs_dir)
    log.info("Name prefix: %s", args.name_prefix)
    log.info("K          : %d", args.k)
    log.info("Output dir : %s", args.output_dir)
    log.info("=" * 60)

    # --- Step 1: 각 fold 결과 파싱 ---
    fold_results: Dict[int, Dict[str, float]] = {}
    fold_best_pts: Dict[int, Path] = {}

    for fid in range(args.k):
        run_name = f"{args.name_prefix}{fid}"
        run_dir = args.runs_dir / run_name
        results_csv = run_dir / "results.csv"
        best_pt = run_dir / "weights" / "best.pt"

        if not results_csv.exists():
            log.warning("Fold %d: results.csv 미발견 — skip", fid)
            continue

        metrics = parse_fold_results(results_csv)
        if metrics is None:
            log.warning("Fold %d: results.csv 파싱 실패 — skip", fid)
            continue

        fold_results[fid] = metrics
        if best_pt.exists():
            fold_best_pts[fid] = best_pt

    if not fold_results:
        log.error("파싱 가능한 fold 결과 없음. 종료.")
        return 2

    # --- Step 2: 집계 ---
    summary = aggregate(fold_results)
    best_fold = find_best_fold(fold_results)

    # --- Step 3: 출력 ---
    log.info("")
    log.info("=== Per-fold 메트릭 ===")
    log.info("%-6s %-12s %-12s %-12s %-12s %-12s",
             "Fold", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Best epoch")
    log.info("-" * 78)
    for fid in sorted(fold_results):
        r = fold_results[fid]
        log.info("%-6d %-12.4f %-12.4f %-12.4f %-12.4f %-12d",
                 fid,
                 r.get("metrics/precision(B)", 0),
                 r.get("metrics/recall(B)", 0),
                 r.get("metrics/mAP50(B)", 0),
                 r.get("metrics/mAP50-95(B)", 0),
                 int(r.get("best_epoch", -1)))

    log.info("")
    log.info("=== 통계 ===")
    log.info("%-12s %-12s %-12s %-12s %-12s",
             "Metric", "Mean", "Std", "Min", "Max")
    log.info("-" * 64)
    for col in METRIC_COLUMNS:
        if col not in summary:
            continue
        s = summary[col]
        log.info("%-22s %-12.4f %-12.4f %-12.4f %-12.4f",
                 col.replace("metrics/", "").replace("(B)", ""),
                 s["mean"], s["std"], s["min"], s["max"])

    if best_fold:
        log.info("")
        log.info("=" * 60)
        log.info("★ Best Fold: %d (mAP@0.5 = %.4f)",
                 best_fold[0], best_fold[1])
        log.info("=" * 60)

    # --- Step 4: 파일 저장 ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "kfold_summary.csv"
    summary_json = args.output_dir / "kfold_summary.json"
    best_fold_txt = args.output_dir / "kfold_best_fold.txt"

    write_summary_csv(summary_csv, fold_results, summary, best_fold)
    write_summary_json(summary_json, fold_results, summary, best_fold)

    best_pt_path = fold_best_pts.get(best_fold[0]) if best_fold else None
    write_best_fold_txt(best_fold_txt, best_fold, best_pt_path)

    log.info("")
    log.info("저장된 파일:")
    log.info("  CSV  : %s", summary_csv)
    log.info("  JSON : %s", summary_json)
    log.info("  Best : %s", best_fold_txt)

    # --- Step 5: Best 모델 복사 (선택) ---
    if args.copy_best_to and best_pt_path:
        args.copy_best_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt_path, args.copy_best_to)
        log.info("")
        log.info("Best 모델 복사: %s → %s", best_pt_path, args.copy_best_to)

    # --- Step 6: 다음 단계 안내 ---
    log.info("")
    log.info("=" * 60)
    log.info("[다음 단계 — Phase 14: V3-B 검증 (D-023 critical)]")
    log.info("=" * 60)
    if best_pt_path:
        log.info("  python -m src.validate.check_stage2_model \\")
        log.info("      --weights %s \\", args.copy_best_to or best_pt_path)
        log.info("      --data data/annotation_kfold/fold_%d/data.yaml \\",
                 best_fold[0])
        log.info("      --device 0 --iou 0.5 --conf 0.25")
    log.info("")
    log.info("D-023 critical 임계값:")
    log.info("  - missing_rate[Measure] < 0.08")
    log.info("  - missing_rate[GDT] < 0.05  ★ 위험 (88 labels)")
    log.info("  - drawing_level_recall ≥ 0.85")

    return 0


if __name__ == "__main__":
    sys.exit(main())
