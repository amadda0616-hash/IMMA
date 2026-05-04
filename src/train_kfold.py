"""
src/train_kfold.py

Stage 2 OBB K-fold Cross-Validation 순차 학습 (Option F).

배경
----
- 5 fold 학습 = 5 × 1.74h ≈ 8.7h (overnight 적정)
- 각 fold 독립 학습 → fold 단위 실패 격리 (전체 중단 방지)
- ``--start-fold N`` 옵션 — N 번 fold 부터 시작 (앞 fold 완료 시 건너뛰기)
- Best fold 자동 식별은 ``aggregate_kfold_results.py`` 에서 처리

처리 흐름
---------
::

    fold 0 학습 (250 epoch, ~1.74h)
       ↓ 완료 시 결과 보존, 실패 시 로그 + 다음 fold
    fold 1 학습
       ↓
    fold 2 학습
       ↓
    fold 3 학습
       ↓
    fold 4 학습
       ↓
    완료 ✓

CLI
---
::

    # 표준 5-fold 학습 (overnight)
    python src/train_kfold.py \\
        --kfold-dir data/annotation_kfold \\
        --k 5 \\
        --model yolo11l-obb.pt \\
        --imgsz 1280 --batch 4 \\
        --epochs 250 --patience 120 \\
        --device 0 --save-period 50

    # 일부 fold 만 재실행 (예: fold 3 부터)
    python src/train_kfold.py \\
        --kfold-dir data/annotation_kfold \\
        --start-fold 3 \\
        ... (다른 옵션 동일)

    # 학습 중단된 fold 만 재개 (각 fold 내부 --resume 활용)
    python src/train_kfold.py \\
        --kfold-dir data/annotation_kfold \\
        --start-fold 2 \\
        --resume-current-fold \\
        ... (다른 옵션 동일)

산출물
------
::

    checkpoints/yolo_obb_runs/
    ├── yolo_obb_v3_kfold_0/
    │   └── weights/best.pt
    ├── yolo_obb_v3_kfold_1/
    │   └── weights/best.pt
    ├── yolo_obb_v3_kfold_2/
    │   └── weights/best.pt
    ├── yolo_obb_v3_kfold_3/
    │   └── weights/best.pt
    └── yolo_obb_v3_kfold_4/
        └── weights/best.pt

    outputs/kfold_train_log.txt   ← 진행 로그 + 시간 + 실패 fold

관련 의사결정
-------------
- D-024 group-aware split (per fold, prepare_kfold_dataset.py 에서 검증)
- D-039 Stage 3-A 별도 (PaddleOCR-VL-1.5)
- Phase 8 Option B
- Option F K-fold CV
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_kfold")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_RUN_DIR = PROJECT_ROOT / "checkpoints" / "yolo_obb_runs"
DEFAULT_LOG_FILE = PROJECT_ROOT / "outputs" / "kfold_train_log.txt"


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------
def train_fold(
    fold_idx: int,
    fold_dir: Path,
    args: argparse.Namespace,
    log_file: Path,
) -> Dict:
    """
    한 fold 학습 실행 (subprocess 로 stage2_annotation.py 호출).

    Returns
    -------
    dict
        {
            "fold": ...,
            "status": "success" | "failed" | "skipped",
            "duration_sec": ...,
            "best_pt": Path | None,
            "error": (있을 때) str
        }
    """
    run_name = f"yolo_obb_v3_kfold_{fold_idx}"
    data_yaml = fold_dir / "data.yaml"

    if not data_yaml.exists():
        return {
            "fold": fold_idx,
            "status": "failed",
            "duration_sec": 0.0,
            "best_pt": None,
            "error": f"data.yaml not found: {data_yaml}",
        }

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "stage2_annotation.py"),
        "train",
        "--data", str(data_yaml),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--device", str(args.device),
        "--name", run_name,
        "--save-period", str(args.save_period),
    ]

    # --resume-current-fold 옵션 시 현재 fold 만 resume
    is_resume = (
        args.resume_current_fold and fold_idx == args.start_fold
    )
    if is_resume:
        # last.pt 존재 확인
        last_pt = CHECKPOINTS_RUN_DIR / run_name / "weights" / "last.pt"
        if last_pt.exists():
            cmd.append("--resume")
            log.info("Fold %d: --resume mode (last.pt 발견)", fold_idx)
        else:
            log.warning("Fold %d: --resume-current-fold 지정됐으나 last.pt 미발견 → fresh", fold_idx)
            cmd.extend(["--model", args.model])
    else:
        cmd.extend(["--model", args.model])

    log.info("=" * 60)
    log.info("Fold %d / %d 학습 시작", fold_idx, args.k - 1)
    log.info("=" * 60)
    log.info("Command:")
    log.info("  %s", " ".join(cmd))

    # 학습 실행
    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=False)
        rc = result.returncode
    except KeyboardInterrupt:
        log.warning("Fold %d KeyboardInterrupt — 중단", fold_idx)
        return {
            "fold": fold_idx,
            "status": "interrupted",
            "duration_sec": time.time() - t0,
            "best_pt": None,
            "error": "KeyboardInterrupt",
        }
    except Exception as e:  # noqa: BLE001
        log.error("Fold %d 예외: %s", fold_idx, e)
        return {
            "fold": fold_idx,
            "status": "failed",
            "duration_sec": time.time() - t0,
            "best_pt": None,
            "error": str(e),
        }

    duration = time.time() - t0
    best_pt = CHECKPOINTS_RUN_DIR / run_name / "weights" / "best.pt"

    if rc == 0 and best_pt.exists():
        log.info("Fold %d 완료 ✅ (%.1f분)", fold_idx, duration / 60)
        return {
            "fold": fold_idx,
            "status": "success",
            "duration_sec": duration,
            "best_pt": best_pt,
            "error": None,
        }
    else:
        log.error("Fold %d 실패 (rc=%d, best.pt=%s)",
                  fold_idx, rc, "exists" if best_pt.exists() else "MISSING")
        return {
            "fold": fold_idx,
            "status": "failed",
            "duration_sec": duration,
            "best_pt": best_pt if best_pt.exists() else None,
            "error": f"return code {rc}",
        }


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def append_log(log_file: Path, msg: str) -> None:
    """진행 로그 파일에 한 줄 append."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage 2 OBB K-fold Cross-Validation 순차 학습 (Option F)",
    )
    p.add_argument("--kfold-dir", type=Path, required=True,
                   help="K-fold 데이터 폴더 (e.g., data/annotation_kfold)")
    p.add_argument("--k", type=int, default=5, help="fold 수 (default: 5)")
    p.add_argument("--model", default="yolo11l-obb.pt",
                   help="베이스 모델 (default: yolo11l-obb.pt)")
    p.add_argument("--imgsz", type=int, default=1280, help="default: 1280")
    p.add_argument("--batch", type=int, default=4, help="default: 4")
    p.add_argument("--epochs", type=int, default=250, help="default: 250")
    p.add_argument("--patience", type=int, default=120, help="default: 120")
    p.add_argument("--device", default="0", help="GPU id (default: 0)")
    p.add_argument("--save-period", type=int, default=50, help="default: 50")
    p.add_argument("--start-fold", type=int, default=0,
                   help="시작 fold 번호 (default: 0). 앞 fold 이미 완료 시 건너뛰기.")
    p.add_argument("--resume-current-fold", action="store_true",
                   help="--start-fold 의 현재 fold 만 --resume (last.pt 자동 감지)")
    p.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE,
                   help=f"진행 로그 파일 (default: {DEFAULT_LOG_FILE})")
    args = p.parse_args()

    if not args.kfold_dir.exists():
        log.error("K-fold dir not found: %s", args.kfold_dir)
        return 1
    if args.start_fold < 0 or args.start_fold >= args.k:
        log.error("--start-fold 는 [0, %d) 범위여야 함", args.k)
        return 2

    log.info("=" * 60)
    log.info("Stage 2 OBB K-fold 순차 학습 시작")
    log.info("=" * 60)
    log.info("K-fold dir : %s", args.kfold_dir)
    log.info("K          : %d", args.k)
    log.info("Model      : %s", args.model)
    log.info("imgsz      : %d / batch: %d", args.imgsz, args.batch)
    log.info("epochs     : %d / patience: %d", args.epochs, args.patience)
    log.info("save_period: %d / device: %s", args.save_period, args.device)
    log.info("Start fold : %d", args.start_fold)
    if args.resume_current_fold:
        log.info("Resume     : 현재 fold 만 --resume")
    log.info("Log file   : %s", args.log_file)
    log.info("=" * 60)

    append_log(args.log_file,
               f"=== K-fold 학습 시작: K={args.k}, model={args.model}, "
               f"imgsz={args.imgsz}, batch={args.batch}, epochs={args.epochs} ===")

    # --- Fold 순차 실행 ---
    results: List[Dict] = []
    overall_start = time.time()

    for fold_idx in range(args.start_fold, args.k):
        fold_dir = args.kfold_dir / f"fold_{fold_idx}"
        if not fold_dir.exists():
            log.error("Fold dir not found: %s — skip", fold_dir)
            results.append({
                "fold": fold_idx,
                "status": "skipped",
                "duration_sec": 0.0,
                "best_pt": None,
                "error": "fold dir missing",
            })
            append_log(args.log_file, f"Fold {fold_idx} SKIP: fold dir missing")
            continue

        result = train_fold(fold_idx, fold_dir, args, args.log_file)
        results.append(result)

        # 진행 로그 append
        append_log(
            args.log_file,
            f"Fold {fold_idx}: {result['status']} "
            f"({result['duration_sec'] / 60:.1f}분)"
            + (f" — error: {result['error']}" if result.get("error") else ""),
        )

        # KeyboardInterrupt 시 전체 중단
        if result["status"] == "interrupted":
            log.warning("KeyboardInterrupt — 학습 중단")
            break

    # --- 요약 ---
    overall_duration = time.time() - overall_start
    n_success = sum(1 for r in results if r["status"] == "success")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_interrupted = sum(1 for r in results if r["status"] == "interrupted")

    log.info("")
    log.info("=" * 60)
    log.info("K-fold 학습 종합")
    log.info("=" * 60)
    log.info("Total time : %.1f시간 (%.1f분)",
             overall_duration / 3600, overall_duration / 60)
    log.info("Success    : %d / %d", n_success, len(results))
    log.info("Failed     : %d", n_failed)
    log.info("Skipped    : %d", n_skipped)
    log.info("Interrupted: %d", n_interrupted)
    log.info("")
    log.info("=== Fold 결과 ===")
    log.info("%-8s %-12s %-12s %-12s",
             "Fold", "Status", "Duration", "best.pt")
    log.info("-" * 56)
    for r in results:
        marker = "✅" if r["status"] == "success" else "❌"
        log.info("%-8d %-12s %-12s %-12s",
                 r["fold"],
                 f"{r['status']} {marker}",
                 f"{r['duration_sec'] / 60:.1f}분",
                 r["best_pt"].name if r["best_pt"] else "-")

    append_log(args.log_file,
               f"=== K-fold 학습 종료: success={n_success}/{len(results)}, "
               f"total={overall_duration / 3600:.1f}h ===")

    if n_failed > 0 or n_interrupted > 0 or n_skipped > 0:
        log.warning("")
        log.warning("[차후 액션 — 실패/중단된 fold 재실행]")
        for r in results:
            if r["status"] != "success":
                log.warning("  Fold %d (%s): --start-fold %d 로 재실행",
                            r["fold"], r["status"], r["fold"])

    if n_success > 0:
        log.info("")
        log.info("[다음 단계 — Phase 14: K-fold 결과 집계]")
        log.info("  python src/aggregate_kfold_results.py \\")
        log.info("      --runs-dir checkpoints/yolo_obb_runs \\")
        log.info("      --k %d \\", args.k)
        log.info("      --output-dir outputs/")

    return 0 if n_failed == 0 and n_interrupted == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
