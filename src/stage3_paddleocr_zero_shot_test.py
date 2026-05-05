"""
src/stage3_paddleocr_zero_shot_test.py

Phase 15b — PaddleOCR-VL-1.5 다국어 zero-shot 평가 (★ D-039 / D-043 검증).

Background
----------
Phase 15a (`stage3_paddleocr_install_check.py`) 환경 검증 PASS 후, 실제
다국어 도면 5장 (한/영/일/중/러) 으로 zero-shot 정성 평가.

Sample 도면 (★ 사용자 확정 2026-05-04):
- en_drawing.jpg : MOTOR MTG. PLATE (인도 SV ROBOTICS)
- ja_drawing.jpg : 브쉬 (BSBM, TT-10CW, 東洋自動機)
- ko_drawing.jpg : 수도전기공업고등학교 [42 과제] (학습용)
- ru_drawing.jpg : FNINI.732214.001 / Корпус
- zh_drawing.jpg : 规格零件图 (간체)

D-043 박제: 영어/한국어 1장 → low/한정 confidence,
            일본어/중국어 → high confidence (data 풍부),
            러시아어 → mid, 독일어 별도 (~10장 미분류).

Pipeline (per drawing)
----------------------
1. Image load (PIL)
2. 3 prompt zero-shot 추론:
   a. TitleBlock — JSON 추출
   b. Notes — bullet list 추출
   c. Full text — 전체 transcribe (정성 검토용)
3. JSON / Notes parse 시도 (fenced code block / pattern)
4. 결과 누적 → JSON + MD 보고서

★ Critical workaround (D-042)
-----------------------------
모든 모델 로드 시 동일한 monkey-patch 적용:

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if not hasattr(config, "text_config") and hasattr(config, "get_text_config"):
        config.text_config = config.get_text_config()

Usage
-----
::

    # 별도 venv (Phase 15 전용)
    source .venv-paddleocr/bin/activate

    # 5장 평가
    python src/stage3_paddleocr_zero_shot_test.py

    # 일부 prompt 만 (빠른 모드)
    python src/stage3_paddleocr_zero_shot_test.py --prompts titleblock,notes

    # 다른 디렉토리 (예: 독일어 후속)
    python src/stage3_paddleocr_zero_shot_test.py \\
        --samples-dir data/stage3a_eval_samples_de/

    # 출력 경로 지정
    python src/stage3_paddleocr_zero_shot_test.py \\
        --output-json outputs/stage3a_zero_shot_eval.json \\
        --output-md   outputs/stage3a_zero_shot_eval.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root bootstrap (직접 실행 호환)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
DEFAULT_SAMPLES_DIR = _PROJECT_ROOT / "data" / "stage3a_eval_samples"
DEFAULT_OUTPUT_JSON = _PROJECT_ROOT / "outputs" / "stage3a_zero_shot_eval.json"
DEFAULT_OUTPUT_MD   = _PROJECT_ROOT / "outputs" / "stage3a_zero_shot_eval.md"

# 언어 prefix → 표시명 매핑
LANG_FROM_PREFIX: Dict[str, str] = {
    "en_": "English",
    "ja_": "Japanese",
    "ko_": "Korean",
    "ru_": "Russian",
    "zh_": "Chinese",
    "de_": "German",
}

# ---------------------------------------------------------------------------
# 표준 TitleBlock Schema (★ ISO 7200:2004 + KS A 0005 + 첨부 이미지 통합, 23 fields)
# ---------------------------------------------------------------------------
TITLEBLOCK_STANDARD_SCHEMA: Dict[str, List[str]] = {
    "identification": [
        "drawing_no",       # 도면번호 (ISO 7200 mandatory)
        "project_id",       # 프로젝트/조립체 ID (★ 첨부 이미지)
        "title",            # 도면 제목 (ISO 7200 mandatory)
        "sheet",            # 시트번호 (예: 1 OF 1) (ISO 7200 mandatory)
        "revision",         # 개정번호 (ISO 7200 optional)
    ],
    "descriptive": [
        "part_name",        # 부품명 (KS A 0005)
        "material",         # 재질
        "mass",             # 질량 (★ 첨부 이미지)
        "scale",            # 척도 (ISO 동적)
        "projection",       # 투상법 (1각/3각, KS A 0005)
        "paper_size",       # 용지 크기 (A3 등, ISO optional)
        "quantity",         # 수량
        "surface_treatment",# 표면 처리
        "heat_treatment",   # 열처리
        "general_tolerance",# 일반 공차 (ISO 동적)
    ],
    "administrative": [
        "company",          # 회사/법인 (ISO 7200 mandatory: legal_owner)
        "department",       # 책임 부서 (ISO 7200 optional)
        "drawn_by",         # 작성자 (ISO 7200 mandatory: creator)
        "designed_by",      # 설계자 (★ 첨부 이미지에서 drawn_by 와 분리)
        "checked_by",       # 검도자 (KS A 0005)
        "approved_by",      # 승인자 (ISO 7200 mandatory)
        "date",             # 발행일 (ISO 7200 mandatory)
        "state",            # 도면 상태 — Released/Draft (★ 첨부 이미지, ISO 7200 optional)
    ],
}
TITLEBLOCK_FIELD_NAMES: List[str] = [
    f for cat in TITLEBLOCK_STANDARD_SCHEMA.values() for f in cat
]


# Prompt 정의 (★ D-046, 2026-05-05): PaddleOCR-VL README BLOCK 3 task keyword.
# 자연어 prompt → task keyword 로 전환.
#
# README PROMPTS = {
#     "ocr":      "OCR:",                  # 전체 텍스트
#     "table":    "Table Recognition:",    # 표 → markdown
#     "formula":  "Formula Recognition:",  # 수식 → LaTeX
#     "chart":    "Chart Recognition:",    # 차트 → 데이터
#     "spotting": "Spotting:",             # 텍스트 + bbox
#     "seal":     "Seal Recognition:",     # 도장
# }
#
# 우리 use case 매핑:
# - titleblock (TitleBlock = 표) → "Table Recognition:"
# - notes  (자유 텍스트 list)  → "OCR:"
# - full_text (전체 transcribe) → "OCR:"
PROMPTS: Dict[str, str] = {
    "titleblock": "Table Recognition:",
    "notes":      "OCR:",
    "full_text":  "OCR:",
}

# ---------------------------------------------------------------------------
# Generation parameters (★ D-046: README BLOCK 3 — pure model.generate() 호출)
# D-045 의 repetition_penalty / no_repeat_ngram_size / pad_token_id 제거.
# ★ Task keyword 자체가 well-defined trigger 라 degenerate 우려 적음.
# ---------------------------------------------------------------------------
DEFAULT_MAX_NEW_TOKENS = 512

# ---------------------------------------------------------------------------
# Image processing (★ D-046: max_pixels 제어)
# ---------------------------------------------------------------------------
# README BLOCK 3:
#   - 'spotting' task:    2048 * 28 * 28 = 1605632 pixels
#   - other (ocr/table):  1280 * 28 * 28 = 1003520 pixels (~1M)
DEFAULT_MAX_PIXELS = 1280 * 28 * 28          # ~1M pixels (default)
SPOTTING_MAX_PIXELS = 2048 * 28 * 28         # ~1.6M pixels (spotting task only)

DEFAULT_PROMPTS = ["titleblock", "notes", "full_text"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Model loading (★ D-042 monkey-patch)
# ---------------------------------------------------------------------------
def load_model_and_processor(device: str = "cuda:0",
                              dtype_str: str = "bfloat16"
                              ) -> Tuple[Any, Any]:
    """Load PaddleOCR-VL-1.5 with D-042 monkey-patch + D-046 bfloat16.

    ★ D-046 (2026-05-05): README BLOCK 3 권장 dtype = bfloat16 (NOT float16).
       float16 사용 시 numerical instability (emoji hallucination 등 발생).

    Returns
    -------
    (processor, model)
    """
    from transformers import (  # noqa: PLC0415
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
    )
    import torch  # noqa: PLC0415

    log(f"[Load] AutoConfig: {MODEL_ID}")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    # ★ D-042 Critical workaround
    if not hasattr(config, "text_config") and hasattr(config, "get_text_config"):
        config.text_config = config.get_text_config()
        log("       ★ D-042 patch applied: config.text_config = config.get_text_config()")

    log("[Load] AutoProcessor")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    log(f"[Load] AutoModelForImageTextToText (★ D-046 dtype={dtype_str})")
    dtype = getattr(torch, dtype_str, torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        dtype=dtype,
    ).eval()   # ★ D-046: eval() 모드
    if torch.cuda.is_available() and device.startswith("cuda"):
        model = model.to(device)
        log(f"       Moved to {device}")

    return processor, model


# ---------------------------------------------------------------------------
# Inference (per prompt)
# ---------------------------------------------------------------------------
def infer_one_prompt(processor,
                     model,
                     image,
                     prompt: str,
                     max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                     max_pixels: int = DEFAULT_MAX_PIXELS,
                     device: str = "cuda:0"
                     ) -> Dict[str, Any]:
    """Run one image + one prompt through the model.

    ★ D-046 (2026-05-05): README BLOCK 3 권장 호출 방식 적용.
       - messages 안에 image 직접 binding ({"type": "image", "image": image})
       - apply_chat_template(... tokenize=True, return_dict=True,
         return_tensors="pt", images_kwargs={"size": {...}}) 통합 호출
       - Pure model.generate(**inputs, max_new_tokens=512) — D-045 의
         repetition_penalty / no_repeat_ngram_size / pad_token_id 모두 폐기
       - processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    Returns dict with raw_output, output_chars, inference_time_s.
    """
    import torch  # noqa: PLC0415

    # ★ D-046: messages 안에 image 객체 직접 binding
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},   # ★ image 직접
            {"type": "text", "text": prompt},
        ],
    }]

    # ★ D-046: apply_chat_template 통합 호출 (text + image + tokenize)
    try:
        # min_pixels: image processor 의 기본값 사용 (~12544)
        min_pixels = getattr(
            processor.image_processor, "min_pixels", 4 * 28 * 28,
        )
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
        return {"error": f"apply_chat_template: {type(e).__name__}: {e}"}

    if torch.cuda.is_available() and device.startswith("cuda"):
        inputs = inputs.to(device)

    # input 토큰 길이 (decode 시 슬라이스용)
    input_len = int(inputs["input_ids"].shape[1])

    # ★ D-046: Pure generate — D-045 의 추가 파라미터 모두 폐기
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )
    except Exception as e:  # noqa: BLE001
        return {
            "error": f"Generate: {type(e).__name__}: {e}",
            "inference_time_s": round(time.perf_counter() - t0, 2),
        }
    inf_t = time.perf_counter() - t0

    # ★ D-046: Decode — input 부분 슬라이스 + processor.decode
    try:
        gen_only = output_ids[0][input_len:]
        if hasattr(processor, "decode"):
            decoded = processor.decode(gen_only, skip_special_tokens=True)
        elif hasattr(processor, "tokenizer"):
            decoded = processor.tokenizer.decode(
                gen_only, skip_special_tokens=True,
            )
        else:
            decoded = processor.batch_decode(
                [gen_only], skip_special_tokens=True,
            )[0]
    except Exception as e:  # noqa: BLE001
        return {
            "error": f"Decode: {type(e).__name__}: {e}",
            "inference_time_s": round(inf_t, 2),
        }

    # ★ D-046: input 슬라이스 후 decode → 그대로 answer (Assistant 추출 불필요)
    answer = decoded.strip()

    return {
        "raw_output": decoded,
        "answer": answer,
        "output_chars": len(answer),
        "inference_time_s": round(inf_t, 2),
    }


def _extract_assistant_answer(decoded: str) -> str:
    """Extract Assistant: response block from chat template output."""
    # 'Assistant:' 또는 '<assistant>' 등 다양한 형식 대응
    patterns = [
        r"[Aa]ssistant\s*:\s*(.+)$",
        r"<\|assistant\|>(.+?)(?:<\|.+?\|>|$)",
        r"<assistant>(.+?)</assistant>",
    ]
    for p in patterns:
        m = re.search(p, decoded, re.DOTALL)
        if m:
            return m.group(1).strip()
    # 폴백: prompt 이후 부분 또는 전체
    if "Assistant" in decoded:
        return decoded.split("Assistant", 1)[1].lstrip(":").strip()
    return decoded.strip()


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from text.

    Tries:
    1. ```json ... ``` fenced block
    2. First { ... } balanced block
    3. Direct json.loads on whole text
    """
    if not text:
        return None
    # 1. ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 2. First balanced { ... } 단순 매칭
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    # 3. Direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_notes_list(text: str) -> List[str]:
    """Extract note items from text. Supports numbered/bullet lists."""
    if not text:
        return []
    items: List[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        # 1. 패턴: "1." / "1)" / "- " / "• " / "* "
        m = re.match(
            r"^(?:\d+[\.\)]|[\-\*•·▶])\s+(.+)$",
            s,
        )
        if m:
            items.append(m.group(1).strip())
    return items


# ---------------------------------------------------------------------------
# Per-drawing pipeline
# ---------------------------------------------------------------------------
def process_one_drawing(processor,
                         model,
                         img_path: Path,
                         prompts: List[str],
                         max_new_tokens: int,
                         device: str
                         ) -> Dict[str, Any]:
    """Run all selected prompts on one drawing."""
    from PIL import Image  # noqa: PLC0415

    log(f"\n--- {img_path.name} ---")

    # 언어 prefix 감지
    lang = "Unknown"
    for prefix, name in LANG_FROM_PREFIX.items():
        if img_path.name.lower().startswith(prefix):
            lang = name
            break

    log(f"  Language (prefix): {lang}")

    # Image load
    try:
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
    except Exception as e:  # noqa: BLE001
        return {
            "filename": img_path.name,
            "language": lang,
            "error": f"Image load: {type(e).__name__}: {e}",
        }

    log(f"  Image size: {img_w}x{img_h}")

    result: Dict[str, Any] = {
        "filename": img_path.name,
        "language": lang,
        "image_size": [img_w, img_h],
    }

    # 각 prompt 실행
    for pkey in prompts:
        if pkey not in PROMPTS:
            log(f"  [warn] unknown prompt key: {pkey}")
            continue

        log(f"  [{pkey}] inference ...")
        out = infer_one_prompt(
            processor, model, image, PROMPTS[pkey],
            max_new_tokens=max_new_tokens, device=device,
        )

        # 후처리 — TitleBlock 은 JSON parse 시도, Notes 는 list parse
        if "answer" in out and not out.get("error"):
            if pkey == "titleblock":
                out["parsed_json"] = try_parse_json(out["answer"])
                out["json_parsed_ok"] = out["parsed_json"] is not None
            elif pkey == "notes":
                out["items"] = parse_notes_list(out["answer"])
                out["n_items"] = len(out["items"])

        # 콘솔 미리보기 (간단)
        preview = (out.get("answer") or out.get("error", ""))[:150]
        log(f"           time={out.get('inference_time_s', 0):.2f}s  "
            f"chars={out.get('output_chars', 0)}  "
            f"preview: {preview!r}")

        result[pkey] = out

    return result


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def build_markdown_report(summary: Dict[str, Any]) -> str:
    """Build a markdown report from the result dict."""
    lines: List[str] = []
    lines.append(f"# Phase 15b — PaddleOCR-VL-1.5 다국어 Zero-Shot 평가")
    lines.append("")
    lines.append(f"- **Model**: `{summary['model_id']}`")
    lines.append(f"- **Samples dir**: `{summary['samples_dir']}`")
    lines.append(f"- **Sample 수**: {summary['n_samples']}")
    lines.append(f"- **Prompts**: {', '.join(summary['prompts'])}")
    lines.append(f"- **Total inference time**: {summary['total_inference_time_s']:.2f}s")
    lines.append(f"- **Languages processed**: {', '.join(summary['languages_processed'])}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 결과 요약 표
    lines.append("## 결과 요약")
    lines.append("")
    lines.append("| Filename | Language | Image | Total Time | TB Parsed | Notes Items |")
    lines.append("|---|---|---|---|---|---|")
    for r in summary["results"]:
        if "error" in r and "filename" in r and len(r) <= 3:
            lines.append(
                f"| {r['filename']} | {r.get('language', '?')} | — | — | — | "
                f"❌ {r['error'][:40]} |"
            )
            continue
        sz = r.get("image_size", [0, 0])
        total_t = sum(
            (r.get(k, {}) or {}).get("inference_time_s", 0)
            for k in PROMPTS
            if isinstance(r.get(k), dict)
        )
        tb_ok = "✅" if (r.get("titleblock", {}) or {}).get("json_parsed_ok") else "⚠"
        n_notes = (r.get("notes", {}) or {}).get("n_items", "—")
        lines.append(
            f"| {r['filename']} | {r['language']} | {sz[0]}×{sz[1]} | "
            f"{total_t:.2f}s | {tb_ok} | {n_notes} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # 각 도면 상세
    for r in summary["results"]:
        lines.append(f"## {r.get('filename', '?')}")
        lines.append("")
        if "error" in r:
            lines.append(f"❌ **Error**: `{r['error']}`")
            lines.append("")
            continue

        lines.append(f"- Language: **{r['language']}**")
        sz = r.get("image_size", [0, 0])
        lines.append(f"- Image size: {sz[0]}×{sz[1]}")
        lines.append("")

        # TitleBlock
        if "titleblock" in r:
            tb = r["titleblock"]
            lines.append("### TitleBlock")
            lines.append("")
            if tb.get("error"):
                lines.append(f"❌ Error: `{tb['error']}`")
            else:
                lines.append(f"- Inference time: {tb.get('inference_time_s', 0):.2f}s")
                lines.append(f"- Output chars: {tb.get('output_chars', 0)}")
                lines.append(f"- JSON parsed: {'✅' if tb.get('json_parsed_ok') else '⚠ (raw)'}")
                lines.append("")
                if tb.get("parsed_json"):
                    lines.append("**Parsed JSON**:")
                    lines.append("```json")
                    lines.append(json.dumps(tb["parsed_json"], indent=2, ensure_ascii=False))
                    lines.append("```")
                    lines.append("")
                lines.append("**Raw answer**:")
                lines.append("```")
                lines.append((tb.get("answer", "") or "")[:1500])
                lines.append("```")
            lines.append("")

        # Notes
        if "notes" in r:
            nt = r["notes"]
            lines.append("### Notes")
            lines.append("")
            if nt.get("error"):
                lines.append(f"❌ Error: `{nt['error']}`")
            else:
                lines.append(f"- Inference time: {nt.get('inference_time_s', 0):.2f}s")
                lines.append(f"- Output chars: {nt.get('output_chars', 0)}")
                lines.append(f"- Items parsed: {nt.get('n_items', 0)}")
                lines.append("")
                items = nt.get("items", [])
                if items:
                    lines.append("**Parsed items**:")
                    for i, it in enumerate(items, 1):
                        lines.append(f"{i}. {it}")
                    lines.append("")
                lines.append("**Raw answer**:")
                lines.append("```")
                lines.append((nt.get("answer", "") or "")[:1500])
                lines.append("```")
            lines.append("")

        # Full text
        if "full_text" in r:
            ft = r["full_text"]
            lines.append("### Full text")
            lines.append("")
            if ft.get("error"):
                lines.append(f"❌ Error: `{ft['error']}`")
            else:
                lines.append(f"- Inference time: {ft.get('inference_time_s', 0):.2f}s")
                lines.append(f"- Output chars: {ft.get('output_chars', 0)}")
                lines.append("")
                lines.append("**Raw answer (first 2000 chars)**:")
                lines.append("```")
                lines.append((ft.get("answer", "") or "")[:2000])
                lines.append("```")
            lines.append("")

        lines.append("---")
        lines.append("")

    # 정성 평가 가이드
    lines.append("## 사용자 정성 평가 가이드")
    lines.append("")
    lines.append("각 도면별로 다음 5개 항목 1~5점으로 평가 권장:")
    lines.append("")
    lines.append("1. **TitleBlock 필드 정확도** (key + value 정확하게 추출?)")
    lines.append("2. **Notes 의미 보존** (원문 내용 누락/왜곡 없이?)")
    lines.append("3. **다국어 정확도** (해당 언어 글자 정확한 transcription?)")
    lines.append("4. **Hallucination 여부** (없는 텍스트 생성 안 함?)")
    lines.append("5. **JSON 형식 준수** (TitleBlock 만)")
    lines.append("")
    lines.append("**임계값 (D-013)**: 평균 char accuracy ≥ 0.85, field-level F1 ≥ 0.80, hallucination ≤ 0.05")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**박제 위치**: `history.md §A.12.x`, `PROJECT_HANDOFF.md §10`, `outputs/stage3a_zero_shot_eval.json`")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 15b — PaddleOCR-VL-1.5 다국어 zero-shot 평가",
    )
    p.add_argument(
        "--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR,
        help=f"평가 도면 디렉토리 (default: {DEFAULT_SAMPLES_DIR})",
    )
    p.add_argument(
        "--output-json", type=Path, default=DEFAULT_OUTPUT_JSON,
        help=f"결과 JSON 경로 (default: {DEFAULT_OUTPUT_JSON})",
    )
    p.add_argument(
        "--output-md", type=Path, default=DEFAULT_OUTPUT_MD,
        help=f"결과 MD 보고서 경로 (default: {DEFAULT_OUTPUT_MD})",
    )
    p.add_argument(
        "--prompts", type=str, default=",".join(DEFAULT_PROMPTS),
        help="실행할 prompt (콤마 구분, choices: titleblock, notes, full_text). "
             f"default: {','.join(DEFAULT_PROMPTS)}",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"max_new_tokens (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    p.add_argument(
        "--device", type=str, default="cuda:0",
        help="device (default: cuda:0)",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="평가할 도면 최대 개수 (0 = 전체)",
    )
    return p.parse_args(argv)


def find_image_files(samples_dir: Path) -> List[Path]:
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    files = sorted(
        p for p in samples_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    return files


def main(argv=None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("  Phase 15b — PaddleOCR-VL-1.5 다국어 Zero-Shot 평가")
    print(f"  Model:    {MODEL_ID}")
    print(f"  Samples:  {args.samples_dir}")
    print("=" * 72)

    # Sample 검색
    try:
        images = find_image_files(args.samples_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(
            f"\n해결: mkdir -p {args.samples_dir} 후 5장 도면 저장:\n"
            f"  - en_drawing.jpg / ja_drawing.jpg / ko_drawing.jpg / "
            f"ru_drawing.jpg / zh_drawing.jpg",
        )
        return 2

    if args.limit > 0:
        images = images[:args.limit]

    if not images:
        print(f"❌ No images found under {args.samples_dir}")
        return 2

    print(f"\nFound {len(images)} image(s):")
    for img in images:
        print(f"  - {img.name}")

    # Prompt 검증
    # Prompt 검증
    selected_prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    invalid = [p for p in selected_prompts if p not in PROMPTS]
    if invalid:
        print(f"ERROR: Unknown prompts: {invalid}. Available: {list(PROMPTS.keys())}")
        return 2
    print(f"\nPrompts: {selected_prompts}")

    # Model 로드 (한 번만)
    print()
    t0 = time.perf_counter()
    processor, model = load_model_and_processor(device=args.device)
    print(f"[Load] complete in {time.perf_counter() - t0:.1f}s")

    # 각 도면 처리
    results: List[Dict[str, Any]] = []
    eval_t0 = time.perf_counter()
    for i, img_path in enumerate(images, 1):
        print(f"\n[Drawing {i}/{len(images)}]")
        rec = process_one_drawing(
            processor, model, img_path,
            prompts=selected_prompts,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        results.append(rec)

    total_inf_t = time.perf_counter() - eval_t0
    print(f"\n[Total inference time] {total_inf_t:.2f}s")

    # Summary
    languages = sorted({r.get("language", "Unknown") for r in results})
    summary: Dict[str, Any] = {
        "phase": "15b",
        "model_id": MODEL_ID,
        "samples_dir": str(args.samples_dir),
        "n_samples": len(results),
        "prompts": selected_prompts,
        "device": args.device,
        # ★ D-046 generation parameters (README BLOCK 3 — pure call)
        # D-045 의 repetition_penalty / no_repeat_ngram_size / pad_token_id 폐기
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "max_pixels": DEFAULT_MAX_PIXELS,
            "dtype": "bfloat16",
            "prompt_style": "task_keyword",
        },
        "total_inference_time_s": round(total_inf_t, 2),
        "languages_processed": languages,
        "titleblock_schema": TITLEBLOCK_STANDARD_SCHEMA,
        "titleblock_field_count": len(TITLEBLOCK_FIELD_NAMES),
        "results": results,
    }

    # JSON 저장
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Output JSON] {args.output_json}")

    # MD 보고서
    md = build_markdown_report(summary)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[Output MD]   {args.output_md}")

    # 콘솔 요약
    print()
    print("=" * 72)
    print("  Summary")
    print(f"  Total drawings:  {len(results)}")
    print(f"  Total time:      {total_inf_t:.2f}s")
    print(f"  Avg per drawing: {total_inf_t / max(1, len(results)):.2f}s")
    print(f"  Languages:       {', '.join(languages)}")

    n_errors = sum(1 for r in results if "error" in r and len(r) <= 3)
    if n_errors:
        print(f"  Errors: {n_errors}")
    else:
        print(f"  All drawings processed")

    print("=" * 72)
    return 0 if n_errors == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
