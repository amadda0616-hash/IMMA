"""
src/stage5_enrichment.py

Step 9 (extension) — Metadata Enrichment

논문 범위 외 확장 모듈. 4-tier deterministic-first cascade
(Khan 2026 Context-Aware Mapping + Singh-Sadhu 2026 LLM Survey 참고).

Cascade
-------
::

    Stage 4 통합 JSON
            │
            ▼
    9.1 Gap Detection            schema diff → 누락/모호 필드 식별
            │
            ▼
    9.2 Deterministic Lookup     KB/표준 직접 매칭   (conf 0.95+)
            │ unresolved
            ▼
    9.3 Engineering Heuristics   도메인 룰 (Ø, M, n×) (conf ~0.80)
            │ unresolved
            ▼
    9.4 RAG-Augmented LLM        Gemini / Qwen + KB   (conf < 0.95)
            │
            ▼
    9.5 HITL Flag Gate           conf < 0.70 → flagged_for_review=true

Output schema (per field, HANDOFF §11 D-022 / D-023)
---------------------------------------------------
::

    {
      "original":  ...,
      "suggested": ...,
      "alternatives": [...],
      "confidence": 0.86,
      "method": "deterministic" | "heuristic" | "llm" | "hitl",
      "source": "...",                # KB file or model name
      "rationale": "...",
      "evidence": [...],
      "flagged_for_review": false
    }

Decisions
---------
- D-019  Step 9 = 논문 범위 외 확장
- D-020  4-tier cascade
- D-021  KB 4종 (material/iso2768/roughness/gdt_priors) — material+iso 우선
- D-022  Provenance 필수
- D-023  HITL 임계값 = 0.70
- D-024  Provider adapter (gemini / qwen / claude / mock)

CLI
---
::

    # 단일
    python src/stage5_enrichment.py enrich \\
        --json outputs/sample.json \\
        --image dataset/sample.jpg \\
        --provider mock --categories all

    # 배치
    python src/stage5_enrichment.py batch \\
        --json-dir outputs/json/ --provider gemini

Categories
----------
- material               : "stainless" → "SUS304 No.2D" 등
- tolerance_general      : null → "ISO 2768-mK"
- surface_roughness_default : null → "Ra 3.2 μm"
- process_sequence       : null → ["선삭", "드릴", ...]
- qc_checklist           : null → critical 치수/공차
- all                    : 위 5개 모두
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
KB_DIR = PROJECT_ROOT / "data" / "kb"

HITL_CONF_THRESHOLD = 0.70   # D-023
DEFAULT_PROVIDER = "mock"    # API key 없어도 동작

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage5_enrichment")


# ---------------------------------------------------------------------------
# Knowledge Base (inline defaults; data/kb/*.json overrides if present)
# ---------------------------------------------------------------------------
INLINE_MATERIAL_KB: Dict[str, Dict[str, Any]] = {
    "stainless": {
        "primary": "SUS304 No.2D",
        "alternatives": [
            {"value": "SUS304 BA",     "weight": 0.30, "note": "외관 / 반사 중요"},
            {"value": "SUS316L No.2D", "weight": 0.15, "note": "내식성 강화"},
            {"value": "SUS430 No.2D",  "weight": 0.10, "note": "원가 절감"},
        ],
        "rationale": "박판/판금 + 한국 시장 → SUS304 2D 마감이 가장 흔함",
        "source": "KS D 3705 / cross_standard_map",
        "confidence": 0.78,
    },
    "stainless steel": {  # alias
        "primary": "SUS304 No.2D",
        "alternatives": [{"value": "SUS304 BA", "weight": 0.30}],
        "confidence": 0.78,
        "source": "KS D 3705",
        "rationale": "박판 일반 grade",
    },
    "carbon steel": {
        "primary": "SS400",
        "alternatives": [
            {"value": "S45C", "weight": 0.25, "note": "기계가공·열처리"},
            {"value": "SM490A", "weight": 0.10, "note": "구조용"},
        ],
        "confidence": 0.72,
        "source": "KS D 3503",
        "rationale": "일반 강재",
    },
    "aluminum": {
        "primary": "AL6061-T6",
        "alternatives": [
            {"value": "AL5052", "weight": 0.35, "note": "판금 가공"},
            {"value": "AL7075-T6", "weight": 0.20, "note": "고강도 항공"},
        ],
        "confidence": 0.74,
        "source": "AMS-QQ-A-250",
        "rationale": "기계가공 + 일반 산업",
    },
    "brass": {
        "primary": "C2680",
        "alternatives": [{"value": "C2600", "weight": 0.30}],
        "confidence": 0.70,
        "source": "KS D 5201",
        "rationale": "전기 단자 / 장식 부품",
    },
}

ISO_2768_DEFAULTS: Dict[str, Any] = {
    "default": "ISO 2768-mK",
    "rationale": "general tolerance 미명시 → ISO 2768 medium-coarse 가 기계가공 표준",
    "source": "iso_2768_defaults.json",
    "confidence": 0.95,
}

ROUGHNESS_DEFAULTS: Dict[str, Any] = {
    "default": "Ra 3.2 μm",
    "alternatives": [
        {"value": "Ra 1.6 μm", "note": "정밀 가공면"},
        {"value": "Ra 6.3 μm", "note": "거친 가공면"},
    ],
    "rationale": "표면거칠기 표시 없는 가공면 표준값 (일반 선삭/밀링)",
    "source": "machining_roughness_table.json",
    "confidence": 0.65,
}

# Process sequence templates (process_combination_catalog.json placeholder)
PROCESS_TEMPLATES: Dict[str, List[str]] = {
    "sheet_metal_thin": ["fiber_laser_cutting", "press_brake_bending", "deburring"],
    "machined_aluminum": ["cnc_milling_3axis", "cnc_drilling", "deburring", "anodize"],
    "shaft_S45C": [
        "cnc_turning_rough", "cnc_turning_finish",
        "induction_hardening", "tempering",
        "cylindrical_grinding",
    ],
    "default_machined": ["cnc_milling", "cnc_drilling", "deburring"],
}


def load_kb(name: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Load data/kb/<name>.json if exists, else use inline fallback."""
    kb_path = KB_DIR / f"{name}.json"
    if kb_path.exists():
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            log.warning("Failed loading %s: %s — using inline KB", kb_path, e)
    return fallback


# ---------------------------------------------------------------------------
# Provider adapter
# ---------------------------------------------------------------------------
class EnrichmentProvider(Protocol):
    """Adapter protocol for LLM providers (Gemini / Qwen / Claude / Mock)."""
    name: str
    def query(self, prompt: str,
              image_path: Optional[Path] = None,
              context: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]: ...


class MockProvider:
    """Test provider — returns predictable responses with mid confidence.
    Useful for CI / no-API environments."""
    name = "mock"

    def __init__(self, hitl_threshold: float = HITL_CONF_THRESHOLD):
        self.hitl_threshold = hitl_threshold

    def query(self, prompt: str,
              image_path: Optional[Path] = None,
              context: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
        # Look at prompt to make response a bit more meaningful
        suggested = "mock_suggestion"
        confidence = 0.65
        rationale = "Mock provider fallback (no real LLM call)"
        if context and context.get("category") == "material":
            suggested = "SUS304 No.2D"
            confidence = 0.65
            rationale = "Mock — would normally call Gemini/Qwen with image+KB context"
        return {
            "suggested": suggested,
            "alternatives": [],
            "confidence": confidence,
            "rationale": rationale,
            "evidence": [],
        }


class GeminiProvider:
    """Google Gemini API. Requires ``GEMINI_API_KEY`` env var."""
    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash",
                 api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY env var not set. "
                "Use --provider mock for testing."
            )
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "google-generativeai not installed. "
                "pip install google-generativeai"
            ) from e
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self.model)

    def query(self, prompt: str,
              image_path: Optional[Path] = None,
              context: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
        self._ensure_client()

        # Build content list (multi-modal)
        from PIL import Image  # noqa: PLC0415
        contents: List[Any] = [prompt]
        if image_path and image_path.exists():
            contents.append(Image.open(image_path))

        response = self._client.generate_content(contents)
        text = response.text if hasattr(response, "text") else str(response)

        return _parse_llm_json_response(text)


class QwenProvider:
    """Local Qwen2.5-VL via HuggingFace transformers (stub)."""
    name = "qwen"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (  # noqa: PLC0415
                AutoProcessor, AutoModelForVision2Seq,
            )
            import torch  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "transformers / torch not installed. "
                "Use --provider mock for testing."
            ) from e

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, torch_dtype="auto",
        ).to(self.device).eval()

    def query(self, prompt: str,
              image_path: Optional[Path] = None,
              context: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
        self._ensure_loaded()
        from PIL import Image  # noqa: PLC0415

        msgs: List[Dict[str, Any]] = [{"role": "user", "content": []}]
        if image_path and image_path.exists():
            msgs[0]["content"].append(
                {"type": "image", "image": Image.open(image_path)}
            )
        msgs[0]["content"].append({"type": "text", "text": prompt})

        text = self._processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text], padding=True, return_tensors="pt",
        ).to(self.device)
        outputs = self._model.generate(**inputs, max_new_tokens=512)
        response = self._processor.batch_decode(outputs)[0]
        return _parse_llm_json_response(response)


def _parse_llm_json_response(text: str) -> Dict[str, Any]:
    """Try to extract a JSON dict from the LLM's response text."""
    # Strip ``` fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        # Last-resort: find {...} blob
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {
                "suggested": text[:200], "alternatives": [],
                "confidence": 0.4, "rationale": "raw text fallback",
                "evidence": [],
            }
        try:
            d = json.loads(m.group())
        except json.JSONDecodeError:
            return {
                "suggested": text[:200], "alternatives": [],
                "confidence": 0.4, "rationale": "raw text fallback",
                "evidence": [],
            }
    return {
        "suggested":    d.get("suggested") or d.get("answer") or "",
        "alternatives": d.get("alternatives", []),
        "confidence":   float(d.get("confidence", 0.5)),
        "rationale":    d.get("rationale", ""),
        "evidence":     d.get("evidence", []),
    }


def make_provider(name: str) -> EnrichmentProvider:
    name = (name or DEFAULT_PROVIDER).lower()
    if name == "mock":
        return MockProvider()
    if name == "gemini":
        return GeminiProvider()
    if name == "qwen":
        return QwenProvider()
    raise ValueError(f"Unknown provider: {name!r} (mock|gemini|qwen)")


# ---------------------------------------------------------------------------
# 9.1 Gap Detection
# ---------------------------------------------------------------------------
DEFAULT_CATEGORIES = [
    "material",
    "tolerance_general",
    "surface_roughness_default",
    "process_sequence",
    "qc_checklist",
]

AMBIGUOUS_MATERIAL_KEYWORDS = [
    "stainless", "carbon", "aluminum", "brass", "steel",
    "스테인레스", "스테인리스", "철", "알루미늄",
]


def detect_gaps(unified: Dict[str, Any],
                categories: List[str]) -> Dict[str, Dict[str, Any]]:
    """Find which fields need enrichment. Returns {category: {original, reason}}."""
    gaps: Dict[str, Dict[str, Any]] = {}
    tb = unified.get("title_block") or {}
    notes = unified.get("notes") or []

    if "material" in categories:
        mat = tb.get("material")
        if not mat:
            gaps["material"] = {"original": None, "reason": "missing"}
        elif _is_ambiguous(mat):
            gaps["material"] = {"original": mat, "reason": "ambiguous"}

    if "tolerance_general" in categories:
        tol = tb.get("tolerance")
        if not tol:
            gaps["tolerance_general"] = {
                "original": None, "reason": "missing",
            }

    if "surface_roughness_default" in categories:
        # Look for Roughness annotations in any view
        has_roughness = False
        for v in unified.get("views", []):
            for a in v.get("annotations", []):
                if a.get("class") == "Roughness":
                    has_roughness = True
                    break
            if has_roughness:
                break
        if not has_roughness:
            gaps["surface_roughness_default"] = {
                "original": None, "reason": "no_roughness_annotation",
            }

    if "process_sequence" in categories:
        gaps["process_sequence"] = {"original": None, "reason": "always_suggest"}

    if "qc_checklist" in categories:
        gaps["qc_checklist"] = {"original": None, "reason": "always_suggest"}

    return gaps


def _is_ambiguous(material: str) -> bool:
    s = material.strip().lower()
    if len(s) < 4:
        return True
    if any(kw in s for kw in [k.lower() for k in AMBIGUOUS_MATERIAL_KEYWORDS]):
        # If it already has a specific grade like "SUS304" then not ambiguous
        if re.search(r"\b(SUS|SS|S\d|AL|AISI|GG|FCD|SKD)[A-Z0-9-]*",
                     material, flags=re.IGNORECASE):
            return False
        return True
    return False


# ---------------------------------------------------------------------------
# 9.2 Deterministic Lookup
# ---------------------------------------------------------------------------
def deterministic_lookup(category: str,
                         original: Any,
                         unified: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """KB direct match. Returns enrichment dict or None."""
    if category == "tolerance_general":
        kb = load_kb("iso_2768_defaults", ISO_2768_DEFAULTS)
        return {
            "original": original,
            "suggested": kb["default"],
            "alternatives": kb.get("alternatives", []),
            "confidence": kb["confidence"],
            "method": "deterministic",
            "source": kb["source"],
            "rationale": kb["rationale"],
            "evidence": [],
        }

    if category == "material" and original:
        kb = load_kb("material_catalog", INLINE_MATERIAL_KB)
        s = original.strip().lower()
        if s in kb:
            entry = kb[s]
            return {
                "original": original,
                "suggested": entry["primary"],
                "alternatives": entry.get("alternatives", []),
                "confidence": entry.get("confidence", 0.75),
                "method": "deterministic",
                "source": entry.get("source", "material_catalog.json"),
                "rationale": entry.get("rationale", ""),
                "evidence": [{"type": "kb_entry", "key": s}],
            }

    return None


# ---------------------------------------------------------------------------
# 9.3 Engineering Heuristics
# ---------------------------------------------------------------------------
def heuristic_suggest(category: str,
                      original: Any,
                      unified: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Apply domain rules. Returns enrichment dict or None."""

    if category == "surface_roughness_default":
        # Heuristic: if annotations include precision Measure (h6/k6/H7),
        # default Ra is finer (1.6); else 3.2.
        has_precision = False
        for v in unified.get("views", []):
            for a in v.get("annotations", []):
                if a.get("class") == "Measure":
                    p = a.get("parsed") or {}
                    tc = p.get("tolerance_class") or ""
                    if any(c in tc for c in ("h6", "h7", "k6", "H6", "H7", "K6")):
                        has_precision = True
                        break
            if has_precision:
                break
        if has_precision:
            return {
                "original": None,
                "suggested": "Ra 1.6 μm",
                "alternatives": [{"value": "Ra 0.8 μm", "note": "베어링 시트"}],
                "confidence": 0.78,
                "method": "heuristic",
                "source": "heuristics.py:precision_fit_implies_ra1.6",
                "rationale": "h6/k6/H7 정밀 끼워맞춤 검출 → 표면거칠기 Ra 1.6 표준",
                "evidence": [],
            }
        # Fallback
        return {
            "original": None,
            "suggested": ROUGHNESS_DEFAULTS["default"],
            "alternatives": ROUGHNESS_DEFAULTS["alternatives"],
            "confidence": ROUGHNESS_DEFAULTS["confidence"],
            "method": "heuristic",
            "source": ROUGHNESS_DEFAULTS["source"],
            "rationale": ROUGHNESS_DEFAULTS["rationale"],
            "evidence": [],
        }

    if category == "process_sequence":
        # Heuristic: peek at material + annotation density
        tb = unified.get("title_block") or {}
        mat = (tb.get("material") or "").lower()
        n_measures = sum(
            1 for v in unified.get("views", [])
            for a in v.get("annotations", [])
            if a.get("class") == "Measure"
        )

        if "sus" in mat or "stainless" in mat:
            seq = PROCESS_TEMPLATES["sheet_metal_thin"]
            rationale = "SUS 박판 → 레이저 절단 + 벤딩"
        elif "al" in mat or "aluminum" in mat:
            seq = PROCESS_TEMPLATES["machined_aluminum"]
            rationale = "AL 6061 류 → CNC 가공 + 아노다이징"
        elif "s45c" in mat or "shaft" in mat:
            seq = PROCESS_TEMPLATES["shaft_S45C"]
            rationale = "S45C → 선삭+열처리+연삭"
        else:
            seq = PROCESS_TEMPLATES["default_machined"]
            rationale = f"기본 기계가공 (annotations={n_measures})"

        return {
            "original": None,
            "suggested": seq,
            "alternatives": [],
            "confidence": 0.70,
            "method": "heuristic",
            "source": "process_combination_catalog.json (inline)",
            "rationale": rationale,
            "evidence": [{"type": "material", "value": mat or "unknown"}],
        }

    return None


# ---------------------------------------------------------------------------
# 9.4 RAG-Augmented LLM
# ---------------------------------------------------------------------------
def llm_query(category: str,
              original: Any,
              unified: Dict[str, Any],
              provider: EnrichmentProvider,
              image_path: Optional[Path] = None) -> Dict[str, Any]:
    """Query LLM with category-specific prompt. Returns enrichment dict."""
    prompt = build_prompt(category, original, unified)
    context = {"category": category, "original": original}

    raw = provider.query(prompt, image_path=image_path, context=context)
    return {
        "original": original,
        "suggested": raw.get("suggested"),
        "alternatives": raw.get("alternatives", []),
        "confidence": float(raw.get("confidence", 0.5)),
        "method": "llm",
        "source": f"{provider.name}",
        "rationale": raw.get("rationale", ""),
        "evidence": raw.get("evidence", []),
    }


def build_prompt(category: str, original: Any, unified: Dict[str, Any]) -> str:
    """Category-specific RAG-style prompt."""
    tb = unified.get("title_block") or {}
    n_views = len(unified.get("views", []))
    n_measures = sum(
        1 for v in unified.get("views", [])
        for a in v.get("annotations", [])
        if a.get("class") == "Measure"
    )

    base = (
        f"You are an engineering drawing expert. "
        f"Drawing summary: title={tb.get('title') or '?'}, "
        f"material={tb.get('material') or '?'}, "
        f"scale={tb.get('scale') or '?'}, "
        f"views={n_views}, measures={n_measures}.\n\n"
    )

    if category == "material":
        base += (
            f"The drawing mentions material: '{original}'. "
            f"Suggest the most likely specific grade considering Korean / "
            f"Japanese / international standards. Return JSON:\n"
            "{ \"suggested\": \"<KS/JIS grade>\", "
            "\"alternatives\": [{\"value\": ..., \"weight\": 0.0-1.0}], "
            "\"confidence\": 0.0-1.0, \"rationale\": \"...\" }"
        )
    elif category == "qc_checklist":
        base += (
            "List the critical dimensions/tolerances that should be inspected "
            "during quality control. Return JSON:\n"
            "{ \"suggested\": [\"<item1>\", \"<item2>\", ...], "
            "\"confidence\": 0.0-1.0, \"rationale\": \"...\" }"
        )
    elif category == "process_sequence":
        base += (
            "Suggest a manufacturing process sequence for this part. Return JSON:\n"
            "{ \"suggested\": [\"<step1>\", ...], "
            "\"confidence\": 0.0-1.0, \"rationale\": \"...\" }"
        )
    else:
        base += f"Suggest a value for category '{category}'. Original: {original!r}."

    return base


# ---------------------------------------------------------------------------
# 9.5 HITL Gate
# ---------------------------------------------------------------------------
def apply_hitl_gate(enrichment: Dict[str, Any],
                    threshold: float = HITL_CONF_THRESHOLD) -> Dict[str, Any]:
    """Mark fields with confidence < threshold for human review."""
    conf = float(enrichment.get("confidence", 0.0))
    if conf < threshold:
        enrichment["flagged_for_review"] = True
        enrichment["hitl_reason"] = f"confidence {conf:.2f} < {threshold}"
    else:
        enrichment["flagged_for_review"] = False
    return enrichment


# ---------------------------------------------------------------------------
# Cascade orchestrator
# ---------------------------------------------------------------------------
def enrich_field(category: str,
                 gap_info: Dict[str, Any],
                 unified: Dict[str, Any],
                 provider: EnrichmentProvider,
                 image_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run the 4-tier cascade for one field. Always returns an enrichment dict."""
    original = gap_info.get("original")

    # Tier 1: deterministic
    res = deterministic_lookup(category, original, unified)
    if res is not None and res.get("suggested"):
        return apply_hitl_gate(res)

    # Tier 2: heuristic
    res = heuristic_suggest(category, original, unified)
    if res is not None and res.get("suggested"):
        return apply_hitl_gate(res)

    # Tier 3: LLM
    try:
        res = llm_query(category, original, unified, provider, image_path)
        if res.get("suggested"):
            return apply_hitl_gate(res)
    except Exception as e:  # noqa: BLE001
        log.warning("LLM query failed for %s: %s", category, e)

    # Tier 4: HITL fallback (no resolution)
    return {
        "original": original,
        "suggested": None,
        "alternatives": [],
        "confidence": 0.0,
        "method": "hitl",
        "source": "cascade_failed",
        "rationale": "All tiers failed — manual entry required",
        "evidence": [],
        "flagged_for_review": True,
        "hitl_reason": "no_suggestion_from_any_tier",
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def enrich(unified: Dict[str, Any],
           provider: EnrichmentProvider,
           image_path: Optional[Path] = None,
           categories: Optional[List[str]] = None,
           hitl_threshold: float = HITL_CONF_THRESHOLD) -> Dict[str, Any]:
    """Run the enrichment cascade. Returns the enriched JSON (deep copy)."""
    cats = list(categories) if categories else list(DEFAULT_CATEGORIES)
    enriched = json.loads(json.dumps(unified))   # deep copy

    gaps = detect_gaps(unified, cats)

    fields_out: Dict[str, Dict[str, Any]] = {}
    stats = {
        "fields_total":           0,
        "resolved_deterministic": 0,
        "resolved_heuristic":     0,
        "resolved_llm":           0,
        "flagged_hitl":           0,
    }

    for category, gap_info in gaps.items():
        result = enrich_field(category, gap_info, unified,
                              provider, image_path=image_path)
        fields_out[category] = result
        stats["fields_total"] += 1
        method = result.get("method", "hitl")
        stats[f"resolved_{method}"] = stats.get(f"resolved_{method}", 0) + 1
        if result.get("flagged_for_review"):
            stats["flagged_hitl"] += 1

    enriched["enrichment"] = {
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": provider.name,
        "stats": stats,
        "fields": fields_out,
    }
    return enriched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_enrich(args: argparse.Namespace) -> int:
    in_path = Path(args.json)
    if not in_path.exists():
        log.error("Input JSON not found: %s", in_path)
        return 2
    with open(in_path, "r", encoding="utf-8") as f:
        unified = json.load(f)

    cats = (args.categories or "all").split(",")
    if "all" in cats:
        cats = list(DEFAULT_CATEGORIES)

    provider = make_provider(args.provider)
    log.info("Provider: %s | categories: %s", provider.name, cats)

    image_path = Path(args.image) if args.image else None
    enriched = enrich(unified, provider, image_path=image_path,
                      categories=cats, hitl_threshold=args.hitl_threshold)

    out = Path(args.out) if args.out else in_path.with_suffix(".enriched.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)

    s = enriched["enrichment"]["stats"]
    log.info(
        "Stats: total=%d  det=%d  heur=%d  llm=%d  hitl=%d",
        s["fields_total"],
        s.get("resolved_deterministic", 0),
        s.get("resolved_heuristic", 0),
        s.get("resolved_llm", 0),
        s["flagged_hitl"],
    )
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    in_dir = Path(args.json_dir)
    if not in_dir.exists():
        log.error("Input dir not found: %s", in_dir)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "enriched"
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = (args.categories or "all").split(",")
    if "all" in cats:
        cats = list(DEFAULT_CATEGORIES)
    provider = make_provider(args.provider)

    files = sorted(p for p in in_dir.glob("*.json")
                   if p.name not in ("_pipeline_summary.json",))
    if args.limit > 0:
        files = files[:args.limit]
    log.info("Processing %d files with provider=%s", len(files), provider.name)

    ok = err = 0
    for i, fp in enumerate(files, 1):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                unified = json.load(f)
            image_path = None
            if "image_path" in unified:
                ip = Path(unified["image_path"])
                if ip.exists():
                    image_path = ip
            enriched = enrich(unified, provider, image_path=image_path,
                              categories=cats,
                              hitl_threshold=args.hitl_threshold)
            out = out_dir / f"{fp.stem}.enriched.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(enriched, f, ensure_ascii=False, indent=2)
            ok += 1
        except Exception as e:  # noqa: BLE001
            log.error("Failed on %s: %s", fp.name, e)
            err += 1
        if i % 5 == 0 or i == len(files):
            log.info("[%d/%d] ok=%d err=%d", i, len(files), ok, err)

    return 0 if err == 0 else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 9 — Metadata Enrichment (4-tier cascade)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("enrich", help="단일 JSON 처리")
    pe.add_argument("--json", type=str, required=True,
                    help="Stage 4 통합 JSON (pipeline.py output)")
    pe.add_argument("--image", type=str, default=None,
                    help="원본 도면 (LLM multimodal 입력, 옵션)")
    pe.add_argument("--out", type=str, default=None)
    pe.add_argument("--provider", type=str, default=DEFAULT_PROVIDER,
                    choices=["mock", "gemini", "qwen"])
    pe.add_argument("--categories", type=str, default="all",
                    help="comma-separated: material,tolerance_general,...,all")
    pe.add_argument("--hitl-threshold", type=float, default=HITL_CONF_THRESHOLD)
    pe.set_defaults(func=cmd_enrich)

    pb = sub.add_parser("batch", help="폴더 일괄")
    pb.add_argument("--json-dir", type=str, required=True)
    pb.add_argument("--out-dir", type=str, default=None)
    pb.add_argument("--provider", type=str, default=DEFAULT_PROVIDER,
                    choices=["mock", "gemini", "qwen"])
    pb.add_argument("--categories", type=str, default="all")
    pb.add_argument("--hitl-threshold", type=float, default=HITL_CONF_THRESHOLD)
    pb.add_argument("--limit", type=int, default=0)
    pb.set_defaults(func=cmd_batch)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
