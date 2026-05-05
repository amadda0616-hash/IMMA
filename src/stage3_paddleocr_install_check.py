"""
src/stage3_paddleocr_install_check.py

Phase 15a — PaddleOCR-VL-1.5 환경 설치 자동 검증 (★ D-039 박제 검증 도구).

Background
----------
D-039 (Stage 3-A → PaddleOCR-VL-1.5 채택, 2026-05-03 박제) 의 실제 환경
작동 여부를 자동 확인. 별도 venv (`.venv-paddleocr`) 에서 실행.

★ Critical workaround
---------------------
transformers 5.x 의 native paddleocr_vl 구현이 ``config.text_config`` 속성 접근
시 ``AttributeError: 'PaddleOCRVLConfig' object has no attribute 'text_config'``
발생 — 5.x 의 PreTrainedConfig 가 ``get_text_config()`` 로 옮겨짐.

해결 (monkey-patch, 1줄):
    config.text_config = config.get_text_config()

이 스크립트는 자동으로 patch 적용 + 모든 후속 코드 (`stage3_alphabetical.py`,
`pipeline.py` 등) 도 동일 패턴 적용해야 함.

Usage
-----
::

    # 별도 venv 활성화 (Phase 15 전용)
    source .venv-paddleocr/bin/activate

    # 검증 실행
    python src/stage3_paddleocr_install_check.py

    # 빠른 모드 (inference 생략, env + load 만)
    python src/stage3_paddleocr_install_check.py --skip-inference

    # 출력 JSON 경로 지정
    python src/stage3_paddleocr_install_check.py \\
        --output outputs/stage3a_install_check.json

Environment requirements
------------------------
- Python 3.10+
- torch 2.11+cu128 (Blackwell sm_120 호환, D-030)
- transformers 5.0.0  (★ 5.0~5.7 모두 동일 monkey-patch 필요)
- accelerate, sentencepiece, protobuf, einops, pillow
- ★ 모델 자동 다운로드: PaddlePaddle/PaddleOCR-VL-1.5 (~1.92 GB)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Project root bootstrap (직접 실행 호환)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
DEFAULT_OUTPUT = _PROJECT_ROOT / "outputs" / "stage3a_install_check.json"

# Pass criteria
MIN_PARAMS_B = 0.5      # 0.9B 모델 기준 — 0.5B 이상이어야 PASS
MAX_LOAD_S = 120.0      # 모델 로드 타임아웃 (cold cache 첫 다운로드 후 재실행 기준)
MAX_INFERENCE_S = 60.0  # 더미 inference 타임아웃


# ---------------------------------------------------------------------------
# Step 1 — 환경 정보 수집
# ---------------------------------------------------------------------------
def collect_env_info() -> Dict[str, Any]:
    """torch / transformers / GPU 환경 수집."""
    info: Dict[str, Any] = {"python": sys.version.split()[0], "platform": sys.platform}

    try:
        import torch  # noqa: PLC0415
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            info["cuda_capability"] = f"({cap[0]}, {cap[1]})"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info()
            info["gpu_total_gb"] = round(total_b / 1e9, 2)
            info["gpu_free_gb_initial"] = round(free_b / 1e9, 2)
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        import transformers  # noqa: PLC0415
        info["transformers"] = transformers.__version__
    except Exception as e:  # noqa: BLE001
        info["transformers_error"] = f"{type(e).__name__}: {e}"

    return info


# ---------------------------------------------------------------------------
# Step 2 — Model 로드 (★ monkey-patch 적용)
# ---------------------------------------------------------------------------
def load_paddleocr_vl(verbose: bool = True) -> Tuple[Any, Any, Any]:
    """Load PaddleOCR-VL-1.5 with monkey-patch workaround.

    Returns
    -------
    (processor, model, config)
    """
    from transformers import (  # noqa: PLC0415
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
    )
    import torch  # noqa: PLC0415

    if verbose:
        print(f"[Load 1/3] AutoConfig: {MODEL_ID}")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    # ★ Critical workaround (transformers 5.x compat)
    if not hasattr(config, "text_config") and hasattr(config, "get_text_config"):
        config.text_config = config.get_text_config()
        if verbose:
            print("           ★ Patch applied: config.text_config = config.get_text_config()")
    else:
        if verbose:
            print(f"           text_config attr already present: {hasattr(config, 'text_config')}")

    if verbose:
        print("[Load 2/3] AutoProcessor")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    if verbose:
        print("[Load 3/3] AutoModelForImageTextToText (★ D-046 bfloat16)")
    # ★ D-046: bfloat16 (NOT float16 — numerical instability 회피)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).eval()   # ★ D-046: eval() 모드

    if torch.cuda.is_available():
        model = model.to("cuda:0")
        if verbose:
            print("           Moved to cuda:0 (bfloat16, eval mode)")

    return processor, model, config


# ---------------------------------------------------------------------------
# Step 3 — GPU 메모리 측정 (model load 후)
# ---------------------------------------------------------------------------
def measure_gpu_after_load() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            used_b = total_b - free_b
            info["gpu_free_gb_after_load"] = round(free_b / 1e9, 2)
            info["gpu_used_gb_by_model"] = round(used_b / 1e9, 2)
            info["gpu_total_gb"] = round(total_b / 1e9, 2)
    except Exception as e:  # noqa: BLE001
        info["gpu_measure_error"] = f"{type(e).__name__}: {e}"
    return info


# ---------------------------------------------------------------------------
# Step 4 — 더미 이미지 inference
# ---------------------------------------------------------------------------
def run_dummy_inference(processor, model, verbose: bool = True) -> Dict[str, Any]:
    """Run inference on a small dummy image to verify model end-to-end.

    ★ D-046 (2026-05-05): README BLOCK 3 권장 호출 방식 적용.
       - messages 안에 image 직접 binding
       - apply_chat_template(... tokenize=True, return_dict=True,
         return_tensors="pt", images_kwargs={...}) 통합 호출
       - "OCR:" task keyword (자연어 prompt 폐기)
       - processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    """
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    info: Dict[str, Any] = {}
    dummy = Image.new("RGB", (256, 128), color=(255, 255, 255))

    if verbose:
        print(f"[Infer]    Dummy image: 256x128 white (★ D-046 task keyword)")

    # ★ D-046: messages 안에 image 직접 binding + task keyword
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy},   # ★ image 직접
                {"type": "text", "text": "OCR:"},    # ★ README task keyword
            ],
        }
    ]

    # ★ D-046: apply_chat_template 통합 호출 (text + image + tokenize)
    try:
        min_pixels = getattr(
            processor.image_processor, "min_pixels", 4 * 28 * 28,
        )
        max_pixels = 1280 * 28 * 28
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": min_pixels,
                    "longest_edge":  max_pixels,
                }
            },
        )
    except Exception as e:  # noqa: BLE001
        info["error"] = f"apply_chat_template failed: {type(e).__name__}: {e}"
        return info

    # CUDA 이동
    if torch.cuda.is_available():
        inputs = inputs.to("cuda:0")

    # input 토큰 길이 (decode 슬라이스용)
    input_len = int(inputs["input_ids"].shape[1])

    # ★ D-046: Pure generate (D-045 의 추가 파라미터 모두 폐기)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
            )
        info["inference_time_s"] = round(time.perf_counter() - t0, 2)
    except Exception as e:  # noqa: BLE001
        info["error"] = f"Generate failed: {type(e).__name__}: {e}"
        info["inference_time_s"] = round(time.perf_counter() - t0, 2)
        return info

    # ★ D-046: input 부분 슬라이스 + processor.decode
    try:
        gen_only = output_ids[0][input_len:]
        if hasattr(processor, "decode"):
            decoded = processor.decode(gen_only, skip_special_tokens=True)
        elif hasattr(processor, "tokenizer"):
            decoded = processor.tokenizer.decode(gen_only, skip_special_tokens=True)
        elif hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode([gen_only], skip_special_tokens=True)[0]
        else:
            decoded = "<no decoder available>"
        info["output_preview"] = decoded[:200]
    except Exception as e:  # noqa: BLE001
        info["decode_error"] = f"{type(e).__name__}: {e}"

    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _write_result(result: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Wrote: {output_path}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 15a — PaddleOCR-VL-1.5 환경 검증",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Result JSON 경로 (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--skip-inference", action="store_true",
        help="더미 이미지 inference 건너뛰기 (env + load 만 검증)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="간소화 출력",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet

    print("=" * 72)
    print("  Phase 15a — PaddleOCR-VL-1.5 환경 검증")
    print(f"  Model: {MODEL_ID}")
    print("=" * 72)

    result: Dict[str, Any] = {
        "phase": "15a",
        "model_id": MODEL_ID,
        "status": "pending",
    }

    # --------- Step 1: Env ---------
    print("\n[Step 1] 환경 정보")
    env = collect_env_info()
    result["env"] = env
    for k, v in env.items():
        print(f"  {k:<25} {v}")

    # --------- Step 2: Model load ---------
    print("\n[Step 2] PaddleOCR-VL-1.5 로드 (monkey-patch 적용)")
    t0 = time.perf_counter()
    try:
        processor, model, config = load_paddleocr_vl(verbose=verbose)
        load_time = time.perf_counter() - t0
        result["load_time_s"] = round(load_time, 2)
        nparams = sum(p_.numel() for p_ in model.parameters()) / 1e9
        result["model_params_b"] = round(nparams, 2)
        result["config_class"] = type(config).__name__

        print(f"  Model params:  {nparams:.2f}B")
        print(f"  Config class:  {type(config).__name__}")
        print(f"  Load time:     {load_time:.1f}s")

        if load_time > MAX_LOAD_S:
            print(f"  ⚠ Load time exceeded {MAX_LOAD_S}s threshold")
    except Exception as e:  # noqa: BLE001
        result["status"] = "FAIL_LOAD"
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"  ❌ FAIL: {e}")
        _write_result(result, args.output)
        return 1

    # --------- Step 3: GPU after load ---------
    print("\n[Step 3] GPU 메모리 (모델 로드 후)")
    gpu_info = measure_gpu_after_load()
    result["gpu"] = gpu_info
    for k, v in gpu_info.items():
        print(f"  {k:<25} {v}")

    # --------- Step 4: Inference ---------
    if args.skip_inference:
        print("\n[Step 4] 더미 inference 건너뛰기 (--skip-inference)")
        result["inference"] = {"skipped": True}
    else:
        print("\n[Step 4] 더미 이미지 inference")
        try:
            inf_info = run_dummy_inference(processor, model, verbose=verbose)
            result["inference"] = inf_info
            for k, v in inf_info.items():
                if isinstance(v, str) and len(v) > 80:
                    print(f"  {k:<25} {v[:80]}...")
                else:
                    print(f"  {k:<25} {v}")
        except Exception as e:  # noqa: BLE001
            result["inference"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  ❌ Inference FAIL: {e}")

    # --------- Step 5: 판정 ---------
    print("\n[Step 5] 판정")
    pass_conditions = {
        "cuda_available": result.get("env", {}).get("cuda_available", False),
        "model_params_above_min": (
            result.get("model_params_b", 0.0) >= MIN_PARAMS_B
        ),
        "inference_ok": (
            args.skip_inference
            or (
                isinstance(result.get("inference"), dict)
                and "error" not in result["inference"]
                and result["inference"].get("inference_time_s", 0) > 0
            )
        ),
    }

    for k, v in pass_conditions.items():
        print(f"  {k:<25} {'✅' if v else '❌'}")

    if all(pass_conditions.values()):
        result["status"] = "PASS"
        print("\n  ★ PASS — Phase 15a 완료")
    else:
        result["status"] = "FAIL"
        print("\n  ❌ FAIL — 위 조건 중 ❌ 재확인 필요")

    print("=" * 72)

    _write_result(result, args.output)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
