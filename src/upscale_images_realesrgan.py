"""
src/upscale_images_realesrgan.py

Phase 15b 4차 평가용 — Real-ESRGAN AI 4x upscale (★ D-047 후속, 2026-05-05).

Background
----------
Phase 15b 3차 (D-046 fix) 결과 V5 미통과 (avg char acc ~0.50). 원인 중 하나는
sample 5장의 해상도 부족 (640×640):
- en (640) / ko (640) / ru (640) / zh (640) — 4장 모두 1280 미달
- ja (3334) — 이미 충분

→ Real-ESRGAN 4x upscale 로 1280+ 만들어 4차 평가 시도.

Pipeline
--------
1. 입력 디렉토리 자동 스캔 (jpg/png)
2. Real-ESRGAN x4plus model 로드 (자동 다운, ~64MB)
3. 각 이미지:
   - min(W, H) >= TARGET_MIN: copy (이미 충분)
   - 그 외: 4x upscale → 저장
4. 결과 통계 + 검증

★ Notes
-------
- Real-ESRGAN 은 일반 사진 학습 → 도면 (선화) 에서 artifact 가능
- 안 좋으면 PIL Lanczos fallback 활용
- 처리 시간: ~30s/이미지 (RTX 5080, 4x scale)

Usage
-----
::

    # 별도 venv (Phase 15 전용)
    source .venv-paddleocr/bin/activate

    # 의존성 설치
    uv pip install realesrgan basicsr

    # 5장 4x upscale (default)
    python src/upscale_images_realesrgan.py

    # 입력/출력 경로 지정
    python src/upscale_images_realesrgan.py \\
        --input data/stage3a_eval_samples/ \\
        --output data/stage3a_eval_samples_realesrgan/

    # PIL Lanczos fallback (Real-ESRGAN 미설치 시)
    python src/upscale_images_realesrgan.py --backend lanczos
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root bootstrap (직접 실행 호환)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_INPUT  = _PROJECT_ROOT / "data" / "stage3a_eval_samples"
DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "stage3a_eval_samples_realesrgan"
DEFAULT_WEIGHTS_DIR = _PROJECT_ROOT / "weights"

# Real-ESRGAN x4plus 공식 weights URL
REALESRGAN_X4PLUS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
REALESRGAN_X4PLUS_FILENAME = "RealESRGAN_x4plus.pth"

DEFAULT_SCALE = 4
DEFAULT_TARGET_MIN = 1280   # min(W, H) >= TARGET_MIN 이면 upscale skip
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
def ensure_realesrgan_weights(weights_dir: Path) -> Path:
    """Download Real-ESRGAN x4plus weights if not present."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / REALESRGAN_X4PLUS_FILENAME

    if weights_path.exists() and weights_path.stat().st_size > 1_000_000:
        log(f"  [Weights] cached: {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB)")
        return weights_path

    log(f"  [Weights] downloading: {REALESRGAN_X4PLUS_URL}")
    log(f"            → {weights_path}")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded / total_size * 100)
                if block_num % 50 == 0:
                    log(f"            {downloaded/1e6:.1f}/{total_size/1e6:.1f} MB ({pct:.0f}%)")

        urllib.request.urlretrieve(
            REALESRGAN_X4PLUS_URL, weights_path, reporthook=_progress,
        )
        log(f"  [Weights] downloaded: {weights_path.stat().st_size / 1e6:.1f} MB")
    except Exception as e:  # noqa: BLE001
        if weights_path.exists():
            weights_path.unlink()
        raise RuntimeError(
            f"Real-ESRGAN weights download failed: {e}\n"
            f"수동 다운로드: {REALESRGAN_X4PLUS_URL}\n"
            f"저장 위치:    {weights_path}"
        ) from e

    return weights_path


# ---------------------------------------------------------------------------
# Real-ESRGAN backend
# ---------------------------------------------------------------------------
def init_realesrgan(weights_path: Path,
                     scale: int = 4,
                     device: str = "cuda",
                     half: bool = False) -> Any:
    """Initialize Real-ESRGAN upscaler (xinntao/realesrgan + basicsr).

    Returns
    -------
    RealESRGANer instance
    """
    try:
        from realesrgan import RealESRGANer  # noqa: PLC0415
        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Real-ESRGAN 미설치. 다음 명령으로 설치:\n"
            "  uv pip install realesrgan basicsr\n"
            "또는 PIL Lanczos fallback 사용:\n"
            "  python src/upscale_images_realesrgan.py --backend lanczos"
        ) from e

    # x4plus model architecture (RRDBNet 23 blocks)
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=scale,
    )

    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(weights_path),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=device,
    )
    return upsampler


def upscale_realesrgan(upsampler, image_path: Path, out_path: Path,
                        outscale: int = 4) -> Tuple[int, int]:
    """Upscale one image with Real-ESRGAN. Returns new (W, H)."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Real-ESRGAN expects BGR (OpenCV convention) but works with RGB too
    output, _ = upsampler.enhance(img_np, outscale=outscale)

    out_img = Image.fromarray(output)
    out_img.save(out_path, quality=95)
    return out_img.size  # (W, H)


# ---------------------------------------------------------------------------
# PIL Lanczos fallback (★ Real-ESRGAN 미설치 시)
# ---------------------------------------------------------------------------
def upscale_lanczos(image_path: Path, out_path: Path,
                     target_min: int = 1280) -> Tuple[int, int]:
    """Fallback: PIL Lanczos resize until min(W, H) >= target_min."""
    from PIL import Image  # noqa: PLC0415

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if min(w, h) >= target_min:
        # 이미 충분 — copy
        shutil.copy2(image_path, out_path)
        return (w, h)

    # min(W, H) 가 target_min 이 되도록 scale 계산
    scale = target_min / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    upscaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    upscaled.save(out_path, quality=95)
    return (new_w, new_h)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def find_image_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    return files


def run_pipeline(input_dir: Path,
                  output_dir: Path,
                  backend: str = "realesrgan",
                  scale: int = 4,
                  target_min: int = 1280,
                  weights_dir: Path = DEFAULT_WEIGHTS_DIR,
                  device: str = "cuda",
                  half: bool = False,
                  skip_existing: bool = False) -> Dict[str, Any]:
    """Process all images in input_dir → output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = find_image_files(input_dir)

    log(f"=== Real-ESRGAN Upscaling Pipeline ===")
    log(f"  input:        {input_dir}")
    log(f"  output:       {output_dir}")
    log(f"  backend:      {backend}")
    log(f"  scale:        {scale}x")
    log(f"  target_min:   {target_min} (이미지 min(W,H) >= 이면 copy)")
    log(f"  device:       {device}")
    log(f"  half:         {half}")
    log(f"  found:        {len(images)} images")

    if backend == "realesrgan":
        log(f"\n  [Init] Real-ESRGAN backend")
        weights_path = ensure_realesrgan_weights(weights_dir)
        upsampler = init_realesrgan(
            weights_path, scale=scale, device=device, half=half,
        )
        log(f"  [Init] Real-ESRGAN ready")
    elif backend == "lanczos":
        log(f"\n  [Init] PIL Lanczos backend (fallback)")
        upsampler = None
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Per-image processing
    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for i, p in enumerate(images, 1):
        out_path = output_dir / p.name

        if skip_existing and out_path.exists():
            log(f"  [{i}/{len(images)}] {p.name}: skip (exists)")
            continue

        # Read original size
        try:
            from PIL import Image  # noqa: PLC0415
            with Image.open(p) as im:
                orig_w, orig_h = im.size
        except Exception as e:  # noqa: BLE001
            log(f"  [{i}/{len(images)}] {p.name}: ERROR (read): {e}")
            results.append({
                "filename": p.name, "error": f"read: {e}",
            })
            continue

        # Skip if already big enough
        if min(orig_w, orig_h) >= target_min:
            shutil.copy2(p, out_path)
            log(f"  [{i}/{len(images)}] {p.name}: {orig_w}x{orig_h} → copy (이미 충분)")
            results.append({
                "filename": p.name,
                "orig_size": [orig_w, orig_h],
                "new_size":  [orig_w, orig_h],
                "action":    "copy",
                "time_s":    0.0,
            })
            continue

        # Upscale
        ts = time.perf_counter()
        try:
            if backend == "realesrgan":
                new_w, new_h = upscale_realesrgan(
                    upsampler, p, out_path, outscale=scale,
                )
            else:
                new_w, new_h = upscale_lanczos(p, out_path, target_min=target_min)

            elapsed = time.perf_counter() - ts
            log(
                f"  [{i}/{len(images)}] {p.name}: "
                f"{orig_w}x{orig_h} → {new_w}x{new_h}  "
                f"({elapsed:.1f}s, {backend})"
            )
            results.append({
                "filename": p.name,
                "orig_size": [orig_w, orig_h],
                "new_size":  [new_w, new_h],
                "action":    f"upscale_{backend}",
                "time_s":    round(elapsed, 2),
            })
        except Exception as e:  # noqa: BLE001
            log(f"  [{i}/{len(images)}] {p.name}: ERROR (upscale): {e}")
            results.append({
                "filename": p.name,
                "orig_size": [orig_w, orig_h],
                "error":    f"upscale: {e}",
            })

    total_t = time.perf_counter() - t0

    # Summary
    n_ok = sum(1 for r in results if "error" not in r)
    n_err = sum(1 for r in results if "error" in r)
    n_upscaled = sum(1 for r in results if r.get("action", "").startswith("upscale_"))
    n_copied = sum(1 for r in results if r.get("action") == "copy")

    log(f"\n=== Summary ===")
    log(f"  total time:    {total_t:.1f}s ({total_t / max(1, len(images)):.1f}s/image)")
    log(f"  processed:     {n_ok}/{len(images)}")
    log(f"  upscaled:      {n_upscaled}")
    log(f"  copied (skip): {n_copied}")
    log(f"  errors:        {n_err}")

    # 검증 — 모두 1280+ 인지
    log(f"\n  [Validation] min(W, H) >= {target_min}:")
    for r in results:
        if "error" in r:
            continue
        new_w, new_h = r["new_size"]
        mark = "★ HIGH-RES" if min(new_w, new_h) >= target_min else "❌ LOW-RES"
        log(f"    {r['filename']}: {new_w}x{new_h}  {mark}")

    return {
        "input_dir":  str(input_dir),
        "output_dir": str(output_dir),
        "backend":    backend,
        "scale":      scale,
        "target_min": target_min,
        "n_total":    len(images),
        "n_upscaled": n_upscaled,
        "n_copied":   n_copied,
        "n_errors":   n_err,
        "total_time_s": round(total_t, 2),
        "results":    results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 15b — Real-ESRGAN 4x upscale (★ D-047 후속)",
    )
    p.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"입력 디렉토리 (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"출력 디렉토리 (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--backend", choices=["realesrgan", "lanczos"], default="realesrgan",
        help="Upscaling backend. realesrgan=AI (권장), lanczos=PIL fallback",
    )
    p.add_argument(
        "--scale", type=int, default=DEFAULT_SCALE,
        help=f"Upscale factor (default: {DEFAULT_SCALE}, x4plus model 기준)",
    )
    p.add_argument(
        "--target-min", type=int, default=DEFAULT_TARGET_MIN,
        help=f"min(W,H) >= TARGET_MIN 이면 upscale skip (copy). default: {DEFAULT_TARGET_MIN}",
    )
    p.add_argument(
        "--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR,
        help=f"weights cache (default: {DEFAULT_WEIGHTS_DIR})",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="device (default: cuda). cpu 도 가능하지만 느림.",
    )
    p.add_argument(
        "--half", action="store_true",
        help="FP16 추론 (메모리 절약, RTX 5080 16GB 충분하므로 default OFF)",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="출력 디렉토리에 이미 존재하는 파일 skip",
    )
    p.add_argument(
        "--output-json", type=Path, default=None,
        help="결과 summary JSON 경로 (옵션)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    summary = run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        backend=args.backend,
        scale=args.scale,
        target_min=args.target_min,
        weights_dir=args.weights_dir,
        device=args.device,
        half=args.half,
        skip_existing=args.skip_existing,
    )

    if args.output_json:
        import json  # noqa: PLC0415
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log(f"\n  [Output JSON] {args.output_json}")

    return 0 if summary["n_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
