"""
src/pipeline.py

Step 7 — End-to-end JPG → 통합 Structured JSON

Orchestrates Stage 1 → Stage 2 → Stage 3-A / 3-N → Stage 4 (merge) into a
single ``Pipeline`` class. All models load once; per-drawing inference
target ≤ 30s on RTX 5080 (D-021).

Schema (HANDOFF §5.5)
---------------------
::

    {
      "drawing_id": "...",
      "image_path": "...",
      "image_size": [W, H],
      "title_block": {...},                 # Stage 3-A
      "notes": [...],                       # Stage 3-A
      "views": [
        {
          "view_id": "view_0",
          "bbox": [x1,y1,x2,y2],
          "annotations": [
            {
              "class": "Measure",
              "obb_global": [[x,y]*4],      # back to original image coords
              "obb_local":  [[x,y]*4],      # view-crop coords
              "angle": 12.5,
              "conf": 0.93,
              "parsed": {...}               # Stage 3-N JSON
            }
          ]
        }
      ],
      "meta": {
        "model_versions": {...},
        "timing_seconds": {...},
        "timestamp": "ISO8601"
      }
    }

Skip flags (incremental dev)
----------------------------
- ``--skip-numerical``   : Stage 3-N off (예: Donut fine-tune 미완료)
- ``--skip-alphabetical``: Stage 3-A off
- 두 flag 모두 → YOLO 결과만 (구조 파악용)

CLI
---
::

    # 단일 도면
    python src/pipeline.py run --image dataset/sample.jpg \\
        --out outputs/sample.json

    # 배치 (전체 dataset/)
    python src/pipeline.py batch --input-dir dataset/ \\
        --out-dir outputs/json --device 0

    # YOLO 만 (Donut 학습 전)
    python src/pipeline.py run --image dataset/sample.jpg \\
        --skip-numerical --skip-alphabetical
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Project root bootstrap — allow `python src/pipeline.py` direct execution.
# (Without this, lazy `from src.xxx import ...` calls fail with
# ModuleNotFoundError because src/ is not on sys.path.)
_PROJECT_ROOT_BOOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_BOOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_BOOT))

# Stage 1/2/3 imports are deferred (lazy) inside Pipeline methods so that
# CLI --help works without heavy deps (ultralytics, transformers, torch).

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DET_WEIGHTS = PROJECT_ROOT / "checkpoints" / "yolo_det.pt"
DEFAULT_OBB_WEIGHTS = PROJECT_ROOT / "checkpoints" / "yolo_obb.pt"
DEFAULT_DONUT_NUMERICAL = PROJECT_ROOT / "checkpoints" / "donut_numerical" / "final"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
PIPELINE_TMP_ROOT = PROJECT_ROOT / "outputs" / "_pipeline_tmp"

# ---- Stage 2 5-Fold Ensemble (D-040, 2026-05-04) -----------------------
# V3-B 단일 모델 Measure missing 0.101 FAIL → 5-fold ensemble 채택.
# D-023 PASS 확인 (Measure/GDT/Roughness missing = 0.000).
DEFAULT_ENSEMBLE_CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "yolo_obb_runs"
DEFAULT_ENSEMBLE_FOLD_PATTERN = "yolo_obb_v3_kfold_{i}"
DEFAULT_N_FOLDS = 5
DEFAULT_IOU_NMS = 0.5

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:  # noqa: BLE001
        log.error("imread failed for %s: %s", path, e)
        return None


def obb_local_to_global(obb_local: List[List[float]],
                        view_bbox: List[int]) -> List[List[float]]:
    """Translate OBB 4 points from view-crop to original-drawing coords."""
    x0, y0 = view_bbox[0], view_bbox[1]
    return [[round(x + x0, 2), round(y + y0, 2)] for x, y in obb_local]


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------
class Pipeline:
    """End-to-end orchestrator. Loads all 4 models once, processes JPGs."""

    def __init__(self,
                 det_weights: Path = DEFAULT_DET_WEIGHTS,
                 obb_weights: Path = DEFAULT_OBB_WEIGHTS,
                 donut_num_ckpt: Optional[Path] = DEFAULT_DONUT_NUMERICAL,
                 device: Optional[str] = None,
                 conf_det: float = 0.25,
                 conf_obb: float = 0.25,
                 imgsz_det: int = 1280,
                 imgsz_obb: int = 1024,
                 skip_numerical: bool = False,
                 skip_alphabetical: bool = False,
                 keep_tmp: bool = False,
                 # ---- Stage 2 Ensemble params (D-040, 2026-05-04) ----
                 use_ensemble: bool = True,
                 ensemble_ckpt_root: Path = DEFAULT_ENSEMBLE_CKPT_ROOT,
                 ensemble_fold_pattern: str = DEFAULT_ENSEMBLE_FOLD_PATTERN,
                 n_folds: int = DEFAULT_N_FOLDS,
                 iou_nms: float = DEFAULT_IOU_NMS):
        self.det_weights = Path(det_weights)
        self.obb_weights = Path(obb_weights)
        self.donut_num_ckpt = Path(donut_num_ckpt) if donut_num_ckpt else None
        self.device = device
        self.conf_det = conf_det
        self.conf_obb = conf_obb
        self.imgsz_det = imgsz_det
        self.imgsz_obb = imgsz_obb
        self.keep_tmp = keep_tmp

        # ---- Stage 2 ensemble config (D-040) ----
        self.use_ensemble = use_ensemble
        self.ensemble_ckpt_root = Path(ensemble_ckpt_root)
        self.ensemble_fold_pattern = ensemble_fold_pattern
        self.n_folds = n_folds
        self.iou_nms = iou_nms
        self._fold_models = None   # lazy-loaded list

        # Auto-skip Stage 3-N if Donut Numerical not yet trained
        if not skip_numerical and (
            not self.donut_num_ckpt or not self.donut_num_ckpt.exists()
        ):
            log.warning("Donut Numerical checkpoint not found at %s — "
                        "auto skipping Stage 3-N", self.donut_num_ckpt)
            skip_numerical = True

        if not self.det_weights.exists():
            raise FileNotFoundError(
                f"Stage 1 weights not found: {self.det_weights}"
            )

        # Stage 2 weights validation: branch on ensemble vs single
        if self.use_ensemble:
            for i in range(self.n_folds):
                fold_dir = self.ensemble_ckpt_root / self.ensemble_fold_pattern.format(i=i)
                wp = fold_dir / "weights" / "best.pt"
                if not wp.exists():
                    raise FileNotFoundError(
                        f"Stage 2 ensemble fold {i} weights missing: {wp}\n"
                        f"  → use --no-ensemble for single best.pt mode, "
                        f"or run K-fold training (src/train_kfold.py) first."
                    )
        else:
            if not self.obb_weights.exists():
                raise FileNotFoundError(
                    f"Stage 2 single weights not found: {self.obb_weights}"
                )

        self.skip_numerical = skip_numerical
        self.skip_alphabetical = skip_alphabetical

        # Lazy-loaded VLM models
        self._donut_a = None    # (processor, model, device)
        self._donut_n = None    # (processor, model, device)

        obb_label = (
            f"5fold_ensemble@{self.ensemble_ckpt_root.name}"
            if self.use_ensemble else self.obb_weights.name
        )
        log.info("Pipeline init  det=%s  obb=%s  donut_n=%s  skip[a/n]=[%s/%s]",
                 self.det_weights.name, obb_label,
                 self.donut_num_ckpt.name if self.donut_num_ckpt else "—",
                 self.skip_alphabetical, self.skip_numerical)

    # -- Lazy loaders --------------------------------------------------
    def _ensure_alphabetical(self) -> None:
        if self.skip_alphabetical or self._donut_a is not None:
            return
        from src.stage3_alphabetical import load_model  # noqa: PLC0415
        log.info("Loading Donut Alphabetical (zero-shot) ...")
        self._donut_a = load_model(device=self.device)

    def _ensure_numerical(self) -> None:
        if self.skip_numerical or self._donut_n is not None:
            return
        from src.stage3_numerical import load_inference_model  # noqa: PLC0415
        log.info("Loading Donut Numerical (fine-tuned) ...")
        self._donut_n = load_inference_model(self.donut_num_ckpt, self.device)

    def _ensure_ensemble(self) -> None:
        """Lazy-load 5 fold OBB models for Stage 2 ensemble (D-040)."""
        if not self.use_ensemble or self._fold_models is not None:
            return
        from src.ensemble_predict import load_fold_models  # noqa: PLC0415
        log.info("Loading %d-fold Stage 2 ensemble (lazy)", self.n_folds)
        self._fold_models = load_fold_models(
            ckpt_root=self.ensemble_ckpt_root,
            n_folds=self.n_folds,
            fold_pattern=self.ensemble_fold_pattern,
            device=self.device,
        )

    # -- Stage runners -------------------------------------------------
    def _run_stage1(self, image_path: Path) -> Dict[str, Any]:
        from src.stage1_layout import predict_one as det_predict  # noqa: PLC0415
        return det_predict(
            image_path, weights=self.det_weights,
            conf_thr=self.conf_det, imgsz=self.imgsz_det, device=self.device,
        )

    def _run_stage2(self, view_image: Path,
                    parent_bbox: List[int]) -> Dict[str, Any]:
        # ---- D-040: 5-Fold Ensemble (default) ----
        if self.use_ensemble:
            self._ensure_ensemble()
            from src.ensemble_predict import predict_one_schema  # noqa: PLC0415
            return predict_one_schema(
                models=self._fold_models,
                image_path=view_image,
                conf=self.conf_obb,
                iou_nms=self.iou_nms,
                imgsz=self.imgsz_obb,
                device=self.device,
                parent_bbox=parent_bbox,
            )
        # ---- Legacy single-model path (--no-ensemble) ----
        from src.stage2_annotation import predict_one as obb_predict  # noqa: PLC0415
        return obb_predict(
            view_image, weights=self.obb_weights,
            conf_thr=self.conf_obb, imgsz=self.imgsz_obb, device=self.device,
            parent_bbox=parent_bbox,
        )

    def _run_stage3_alphabetical(self,
                                  patch_path: Path,
                                  region_type: str,
                                  language_hint: Optional[str] = None
                                  ) -> Optional[Dict[str, Any]]:
        if self.skip_alphabetical:
            return None
        self._ensure_alphabetical()
        from src.stage3_alphabetical import predict_one  # noqa: PLC0415
        processor, model, device = self._donut_a
        try:
            return predict_one(
                image_path=patch_path,
                region_type=region_type,
                processor=processor, model=model, device=device,
                language_hint=language_hint,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Stage 3-A failed on %s: %s", patch_path.name, e)
            return {"type": region_type, "error": str(e)}

    def _run_stage3_numerical(self,
                               patch_path: Path,
                               region_class: str
                               ) -> Optional[Dict[str, Any]]:
        if self.skip_numerical:
            return None
        self._ensure_numerical()
        from src.stage3_numerical import predict_one  # noqa: PLC0415
        processor, model, device = self._donut_n
        try:
            return predict_one(
                image_path=patch_path, region_class=region_class,
                processor=processor, model=model, device=device,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Stage 3-N failed on %s: %s", patch_path.name, e)
            return {"type": region_class, "error": str(e)}

    # -- Per-drawing pipeline -----------------------------------------
    def run(self,
            image_path: Path,
            language_hint: Optional[str] = None) -> Dict[str, Any]:
        """Process a single drawing → unified JSON."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        timing: Dict[str, float] = {}
        t0 = time.perf_counter()

        # Per-drawing temp dir for crops/warps
        drawing_id = image_path.stem
        tmp_dir = PIPELINE_TMP_ROOT / drawing_id
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # ---- Stage 1 ------------------------------------------------
        t = time.perf_counter()
        det_rec = self._run_stage1(image_path)
        timing["stage1"] = round(time.perf_counter() - t, 3)

        regions = det_rec.get("regions", [])
        views_meta = [r for r in regions if r["class"] == "View"]
        tb_meta    = [r for r in regions if r["class"] == "TitleBlock"]
        notes_meta = [r for r in regions if r["class"] == "Notes"]

        # ---- Stage 1 crop (필요한 영역만) ----------------------------
        from src.stage1_layout import crop_regions as det_crop  # noqa: PLC0415
        t = time.perf_counter()
        crops_dir = tmp_dir / "stage1_crops"
        all_relevant = views_meta + tb_meta + notes_meta
        crops = det_crop(image_path, all_relevant, crops_dir, padding=5) \
            if all_relevant else []
        timing["stage1_crop"] = round(time.perf_counter() - t, 3)

        # Build lookup: class → list of (path, bbox, conf)
        crops_by_cls: Dict[str, List[Dict[str, Any]]] = {}
        for c in crops:
            crops_by_cls.setdefault(c["class"], []).append(c)

        # ---- Stage 3-A (TitleBlock + Notes) -------------------------
        title_block_out: Dict[str, Any] = {}
        notes_out: List[str] = []

        t = time.perf_counter()
        for c in crops_by_cls.get("TitleBlock", []):
            rec = self._run_stage3_alphabetical(
                Path(c["path"]), region_type="titleblock",
                language_hint=language_hint,
            )
            if rec and isinstance(rec.get("fields"), dict):
                # Merge multiple TBs (rare): later one overwrites
                title_block_out.update({
                    k: v for k, v in rec["fields"].items() if v
                })

        for c in crops_by_cls.get("Notes", []):
            rec = self._run_stage3_alphabetical(
                Path(c["path"]), region_type="notes",
                language_hint=language_hint,
            )
            if rec and isinstance(rec.get("items"), list):
                notes_out.extend(rec["items"])
        timing["stage3_alphabetical"] = round(time.perf_counter() - t, 3)

        # ---- Stage 2 (per View) + Stage 3-N -------------------------
        views_out: List[Dict[str, Any]] = []
        t_stage2 = 0.0
        t_stage3n = 0.0
        for v_idx, v in enumerate(crops_by_cls.get("View", [])):
            view_path = Path(v["path"])
            view_bbox = v["bbox"]

            # Stage 2
            ts = time.perf_counter()
            obb_rec = self._run_stage2(view_path, parent_bbox=view_bbox)
            t_stage2 += time.perf_counter() - ts

            ann_global: List[Dict[str, Any]] = []
            annotations = obb_rec.get("annotations", [])

            # Per-OBB warp + Stage 3-N
            from src.stage2_annotation import crop_obb_regions as obb_crop  # noqa: PLC0415
            warps_dir = tmp_dir / "stage2_warps" / f"view_{v_idx}"
            warps_dir.mkdir(parents=True, exist_ok=True)
            warp_records = obb_crop(view_path, annotations, warps_dir) \
                if annotations else []

            ts = time.perf_counter()
            # Match warp_records with annotations by class+order
            cls_idx_counter: Dict[str, int] = {}
            warp_lookup: Dict[tuple, str] = {}
            for w in warp_records:
                cls = w["class"]
                idx = cls_idx_counter.get(cls, 0)
                warp_lookup[(cls, idx)] = w["path"]
                cls_idx_counter[cls] = idx + 1

            cls_idx_counter = {}
            for ann in annotations:
                cls = ann["class"]
                idx = cls_idx_counter.get(cls, 0)
                cls_idx_counter[cls] = idx + 1
                wpath = warp_lookup.get((cls, idx))

                parsed: Optional[Dict[str, Any]] = None
                if wpath and not self.skip_numerical:
                    rec = self._run_stage3_numerical(Path(wpath), region_class=cls)
                    if rec:
                        parsed = rec.get("parsed")

                obb_local = ann["obb"]
                obb_global = obb_local_to_global(obb_local, view_bbox)
                ann_global.append({
                    "class": cls,
                    "obb_global": obb_global,
                    "obb_local":  obb_local,
                    "angle":      ann.get("angle"),
                    "conf":       ann.get("conf"),
                    "parsed":     parsed,
                })
            t_stage3n += time.perf_counter() - ts

            views_out.append({
                "view_id": f"view_{v_idx}",
                "bbox":    view_bbox,
                "conf":    v.get("conf"),
                "annotations": ann_global,
            })

        timing["stage2"] = round(t_stage2, 3)
        timing["stage3_numerical"] = round(t_stage3n, 3)
        timing["total"] = round(time.perf_counter() - t0, 3)

        # ---- Stage 4 merge ------------------------------------------
        unified: Dict[str, Any] = {
            "drawing_id":  drawing_id,
            "image_path":  str(image_path),
            "image_size":  det_rec.get("image_size", [0, 0]),
            "title_block": title_block_out,
            "notes":       notes_out,
            "views":       views_out,
            "meta": {
                "model_versions": {
                    "yolo_det":    self.det_weights.name,
                    "yolo_obb":    (
                        f"5fold_ensemble (kfold_0..{self.n_folds-1}, "
                        f"iou_nms={self.iou_nms}, conf={self.conf_obb})"
                        if self.use_ensemble else self.obb_weights.name
                    ),
                    "donut_alpha": (
                        "donut-base-finetuned-docvqa"
                        if not self.skip_alphabetical else "skipped"
                    ),
                    "donut_num":   (
                        str(self.donut_num_ckpt)
                        if not self.skip_numerical else "skipped"
                    ),
                },
                "timing_seconds": timing,
                "timestamp": now_iso(),
                "language_hint": language_hint,
            },
        }

        # Cleanup
        if not self.keep_tmp:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except OSError:
                pass

        return unified

    # -- Batch ---------------------------------------------------------
    def run_batch(self,
                  image_paths: List[Path],
                  out_dir: Path,
                  language_hint: Optional[str] = None
                  ) -> Dict[str, Any]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        log_rows: List[Dict[str, Any]] = []
        ok = err = 0
        t0 = time.perf_counter()
        for i, img in enumerate(image_paths, 1):
            try:
                rec = self.run(img, language_hint=language_hint)
                json_path = out_dir / f"{img.stem}.json"
                write_json(json_path, rec)
                ok += 1
                log_rows.append({
                    "image": str(img),
                    "json":  str(json_path),
                    "status": "ok",
                    "total_s": rec["meta"]["timing_seconds"].get("total"),
                })
            except Exception as e:  # noqa: BLE001
                err += 1
                log.error("Failed on %s: %s", img.name, e)
                log_rows.append({
                    "image": str(img), "json": "",
                    "status": "error", "error": str(e),
                })
            if i % 5 == 0 or i == len(image_paths):
                log.info("[%d/%d] ok=%d err=%d", i, len(image_paths), ok, err)

        total_s = round(time.perf_counter() - t0, 1)
        summary = {
            "n_total": len(image_paths),
            "n_ok": ok,
            "n_err": err,
            "total_seconds": total_s,
            "avg_seconds_per_drawing": round(total_s / max(1, len(image_paths)), 2),
            "log": log_rows,
        }
        write_json(out_dir / "_pipeline_summary.json", summary)
        log.info("Batch complete  total=%ds  avg=%.2fs/drawing  ok=%d  err=%d",
                 total_s, summary["avg_seconds_per_drawing"], ok, err)
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    p = Pipeline(
        det_weights=Path(args.det_weights),
        obb_weights=Path(args.obb_weights),
        donut_num_ckpt=Path(args.donut_num) if args.donut_num else None,
        device=args.device,
        conf_det=args.conf_det,
        conf_obb=args.conf_obb,
        imgsz_det=args.imgsz_det,
        imgsz_obb=args.imgsz_obb,
        skip_numerical=args.skip_numerical,
        skip_alphabetical=args.skip_alphabetical,
        keep_tmp=args.keep_tmp,
        use_ensemble=args.use_ensemble,
        ensemble_ckpt_root=Path(args.ensemble_ckpt_root),
        ensemble_fold_pattern=args.ensemble_fold_pattern,
        n_folds=args.n_folds,
        iou_nms=args.iou_nms,
    )
    rec = p.run(Path(args.image), language_hint=args.language)

    out = Path(args.out) if args.out else (
        DEFAULT_OUTPUT_DIR / f"{Path(args.image).stem}.json"
    )
    write_json(out, rec)
    log.info("Saved: %s", out)
    log.info("Timing: %s", rec["meta"]["timing_seconds"])
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        log.error("input dir not found: %s", in_dir)
        return 2

    images = sorted(p for p in in_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        log.warning("No images under %s", in_dir)
        return 0

    p = Pipeline(
        det_weights=Path(args.det_weights),
        obb_weights=Path(args.obb_weights),
        donut_num_ckpt=Path(args.donut_num) if args.donut_num else None,
        device=args.device,
        conf_det=args.conf_det,
        conf_obb=args.conf_obb,
        imgsz_det=args.imgsz_det,
        imgsz_obb=args.imgsz_obb,
        skip_numerical=args.skip_numerical,
        skip_alphabetical=args.skip_alphabetical,
        keep_tmp=args.keep_tmp,
        use_ensemble=args.use_ensemble,
        ensemble_ckpt_root=Path(args.ensemble_ckpt_root),
        ensemble_fold_pattern=args.ensemble_fold_pattern,
        n_folds=args.n_folds,
        iou_nms=args.iou_nms,
    )
    out_dir = Path(args.out_dir)
    summary = p.run_batch(images, out_dir, language_hint=args.language)
    return 0 if summary["n_err"] == 0 else 1


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--det-weights", type=str, default=str(DEFAULT_DET_WEIGHTS))
    p.add_argument("--obb-weights", type=str, default=str(DEFAULT_OBB_WEIGHTS),
                   help="Stage 2 single best.pt (used only with --no-ensemble)")
    p.add_argument("--donut-num", type=str,
                   default=str(DEFAULT_DONUT_NUMERICAL),
                   help="Donut Numerical fine-tuned ckpt (final/). "
                        "미존재 시 자동 skip.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--conf-det", type=float, default=0.25)
    p.add_argument("--conf-obb", type=float, default=0.25)
    p.add_argument("--imgsz-det", type=int, default=1280)
    p.add_argument("--imgsz-obb", type=int, default=1024)
    p.add_argument("--skip-numerical", action="store_true",
                   help="Stage 3-N (Donut Numerical) off")
    p.add_argument("--skip-alphabetical", action="store_true",
                   help="Stage 3-A (Donut Alphabetical zero-shot) off")
    p.add_argument("--language", type=str, default=None,
                   help="Optional language hint (en/ko/ja/ru) for Stage 3-A")
    p.add_argument("--keep-tmp", action="store_true",
                   help="outputs/_pipeline_tmp 보존 (디버깅)")
    # ---- Stage 2 5-Fold Ensemble (D-040, default ON) ----
    p.add_argument("--use-ensemble", dest="use_ensemble",
                   action="store_true", default=True,
                   help="★ Stage 2 5-fold ensemble (default, D-040 PASS)")
    p.add_argument("--no-ensemble", dest="use_ensemble",
                   action="store_false",
                   help="단일 best.pt 사용 (디버깅/legacy, --obb-weights 필요)")
    p.add_argument("--ensemble-ckpt-root", type=str,
                   default=str(DEFAULT_ENSEMBLE_CKPT_ROOT),
                   help="K-fold checkpoints 루트 (yolo_obb_v3_kfold_{i}/)")
    p.add_argument("--ensemble-fold-pattern", type=str,
                   default=DEFAULT_ENSEMBLE_FOLD_PATTERN)
    p.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    p.add_argument("--iou-nms", type=float, default=DEFAULT_IOU_NMS,
                   help="Cross-fold rotated NMS IoU threshold (default 0.5)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 7 — End-to-end Engineering Drawing → Structured JSON",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="단일 도면 처리")
    add_common_args(pr)
    pr.add_argument("--image", type=str, required=True)
    pr.add_argument("--out", type=str, default=None)
    pr.set_defaults(func=cmd_run)

    pb = sub.add_parser("batch", help="폴더 일괄 처리")
    add_common_args(pb)
    pb.add_argument("--input-dir", type=str, required=True)
    pb.add_argument("--out-dir", type=str, required=True)
    pb.add_argument("--limit", type=int, default=0,
                    help="처리할 도면 최대 개수 (0 = 전체)")
    pb.set_defaults(func=cmd_batch)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
