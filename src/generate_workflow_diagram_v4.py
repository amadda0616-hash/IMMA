"""
src/generate_workflow_diagram_v4.py

Multi-Stage Hybrid Framework Pipeline v4 Diagram (한글)

v3 → v4 갱신 사항:
- Phase 4 추가: Stage 3-A PaddleOCR-VL-1.5 zero-shot (D-039, D-042, D-046, D-047)
- Phase 5 추가: Stage 3-N Donut Numerical fine-tune 1차 baseline (D-049, D-050, D-051)
- Real-ESRGAN 4x upscale 단계
- auto_fill_numerical_gt 단계
- Validation V0~V9 갱신 (V5 부분 PASS, V6 진행 중)
- Decision IDs D-001 ~ D-051
- KNOWN_LIMITATIONS.md 참조

출력: outputs/workflow_diagram_v4.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import koreanize_matplotlib  # noqa: F401  (한글 폰트 활성화)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "workflow_diagram_v4.png"

# ---------------------------------------------------------------------------
# Color palette (v3 와 동일 + v4 신규)
# ---------------------------------------------------------------------------
COLORS = {
    "dl_vlm":     "#FFF3C4",   # DL/VLM stage (노란)
    "hitl":       "#FCE4EC",   # HITL/human (분홍)
    "io":         "#FFFFFF",   # I/O artifact (흰색)
    "cluster":    "#E1F5FE",   # Cluster (하늘)
    "deterministic": "#EEEEEE",  # Deterministic (회색)
    "kb":         "#FFE0B2",   # Knowledge base (주황)
    "final":      "#C8E6C9",   # Final output (연두)
    "new_v3":     "#FFAB91",   # * New in v3 (주황 진하게)
    "new_v4":     "#FF7043",   # ★ New in v4 (주황 강조)
    "active_learning": "#E3F2FD",  # Active Learning (연파랑)
    "phase_bg":   "#F0F4F8",   # Phase 배경
    "limitation": "#FFCDD2",   # ★ NEW v4: Known Limitations (연빨강)
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_box(ax, x, y, w, h, text, color, fontsize=8, bold=False, edge="#666666"):
    """Add rounded box with text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color, edgecolor=edge, linewidth=0.8,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight, wrap=True,
    )

def add_phase_bg(ax, x, y, w, h, label):
    """Phase 배경 + 라벨."""
    bg = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor=COLORS["phase_bg"], edgecolor="#90A4AE", linewidth=1.0,
        alpha=0.8,
    )
    ax.add_patch(bg)
    ax.text(x + 0.15, y + h - 0.25, label, fontsize=11, fontweight="bold", color="#37474F")

def add_arrow(ax, x1, y1, x2, y2, color="#546E7A", lw=1.0):
    """Add arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_width=4,head_length=6",
        color=color, linewidth=lw,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)

# ---------------------------------------------------------------------------
# Main diagram
# ---------------------------------------------------------------------------
def main() -> int:
    fig, ax = plt.subplots(figsize=(16, 22))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 30)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Title ---
    ax.text(10, 29.3, "다단계 하이브리드 프레임워크 — Pipeline v4",
            ha="center", fontsize=18, fontweight="bold")
    ax.text(10, 28.85,
            "Khan 2025 base + Active Learning + 5-fold Ensemble (D-023) + Stage 3-A PaddleOCR-VL-1.5 (D-039) + Stage 3-N Donut 1차 baseline (D-051)",
            ha="center", fontsize=9, color="#555")
    ax.text(10, 28.55,
            "★ 2026-05-05 갱신: D-001 ~ D-051 / 6개 언어 / Phase 16a 진행 중",
            ha="center", fontsize=8, color="#777", style="italic")

    # --- Legend ---
    legend_y = 27.7
    legends = [
        (0.3,  "DL / VLM 학습",       COLORS["dl_vlm"]),
        (3.0,  "HITL / 사람 검수",    COLORS["hitl"]),
        (5.7,  "I/O 산출물",          COLORS["io"]),
        (8.0,  "클러스터",             COLORS["cluster"]),
        (10.2, "결정론 처리",          COLORS["deterministic"]),
        (12.6, "지식베이스",           COLORS["kb"]),
        (14.8, "최종 출력",            COLORS["final"]),
        (17.0, "★ v4 신규",           COLORS["new_v4"]),
    ]
    legends2 = [
        (0.3,  "Active Learning",     COLORS["active_learning"]),
        (3.5,  "★ Known Limitations", COLORS["limitation"]),
    ]
    for x, label, color in legends:
        rect = patches.Rectangle((x, legend_y), 0.4, 0.25, facecolor=color, edgecolor="#888", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x + 0.5, legend_y + 0.13, label, fontsize=7, va="center")
    for x, label, color in legends2:
        rect = patches.Rectangle((x, legend_y - 0.4), 0.4, 0.25, facecolor=color, edgecolor="#888", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x + 0.5, legend_y - 0.27, label, fontsize=7, va="center")

    # ============================================================
    # PHASE 1 — 데이터 준비
    # ============================================================
    add_phase_bg(ax, 0.2, 24.4, 19.6, 2.5, "Phase 1 — 데이터 준비")

    add_box(ax, 0.5, 25.3, 2.7, 0.9,
            "dataset/\n(5,839 JPG, 6개 언어\nKO/EN/JP/RU/CN/DE)\nD-024/D-025",
            COLORS["io"], fontsize=7)
    add_box(ax, 3.5, 25.3, 2.5, 0.9,
            "Step 1.5 (D-019)\nsort_by_titleblock.py\nOCR + line density",
            COLORS["deterministic"], fontsize=7)
    add_box(ax, 3.5, 24.5, 2.5, 0.7,
            "★ Step 1.6 (D-026)\nsort_by_drawing_type.py",
            COLORS["new_v3"], fontsize=7)
    add_box(ax, 6.3, 25.3, 2.7, 0.9,
            "Roboflow 업로드\n(Auto-Orient + Resize 1280)\n5 클래스 (D-028)",
            COLORS["io"], fontsize=7)
    add_box(ax, 9.3, 25.3, 2.7, 0.9,
            "Roboflow 수동 라벨링\n100 seed (D-024 group key)\n80/20 group-aware",
            COLORS["hitl"], fontsize=7)
    add_box(ax, 12.3, 25.3, 3.3, 0.9,
            "IMMA.v1i.yolov11/\ntrain 80 + valid 20\nIsometric/PMI/Table/Text/View",
            COLORS["new_v3"], fontsize=7, bold=True)
    add_box(ax, 16.0, 25.3, 3.5, 0.9,
            "★ exclude_groups.py\n5,839 → 5,793 + 검수 폴더\n(46 groups 제외)",
            COLORS["deterministic"], fontsize=7)

    # arrows
    add_arrow(ax, 3.2, 25.75, 3.5, 25.75)
    add_arrow(ax, 6.0, 25.75, 6.3, 25.75)
    add_arrow(ax, 9.0, 25.75, 9.3, 25.75)
    add_arrow(ax, 12.0, 25.75, 12.3, 25.75)
    add_arrow(ax, 15.6, 25.75, 16.0, 25.75)

    # ============================================================
    # PHASE 2 — Stage 1 학습 (Active Learning)
    # ============================================================
    add_phase_bg(ax, 0.2, 21.0, 19.6, 3.0, "Phase 2 — Stage 1 학습 (Active Learning)")

    add_box(ax, 0.5, 22.4, 3.0, 1.1,
            "★ V.A Quick Train\nyolo11m.pt + 100 seed\n50 epochs / 28.5분\nmAP@50=0.9364",
            COLORS["dl_vlm"], fontsize=7, bold=True)
    add_box(ax, 0.5, 21.3, 3.0, 0.9,
            "checkpoints/yolo_det.pt\n(V.A, 40.7MB)",
            COLORS["io"], fontsize=7)
    add_box(ax, 3.8, 22.7, 2.5, 0.8,
            "V2-A 라벨 검증\ncheck_labels_yolo.py",
            COLORS["deterministic"], fontsize=7)
    add_box(ax, 3.8, 21.8, 2.5, 0.7,
            "V2-B 모델 검증\n(D-029 매핑 fix)",
            COLORS["deterministic"], fontsize=7)
    add_box(ax, 6.6, 22.4, 3.2, 1.1,
            "★ Step 5.5 (NEW)\nauto_label_stage1.py\n5,739 자동 라벨\n4-tier 우선순위",
            COLORS["new_v3"], fontsize=7)
    add_box(ax, 6.6, 21.3, 3.2, 0.9,
            "outputs/auto_labels/\nmanifest.csv",
            COLORS["io"], fontsize=7)
    add_box(ax, 10.1, 22.4, 3.0, 1.1,
            "Roboflow Pre-annotation\n사람 검수 (~16h)\n우선순위 순",
            COLORS["hitl"], fontsize=7)
    add_box(ax, 13.4, 22.4, 3.4, 1.1,
            "★ V.B Full Retrain\n(★ 미진행 — Phase 17 후)\n5,839 imgs (4671/1168)\n100 epochs / ~5h",
            COLORS["dl_vlm"], fontsize=7)
    add_box(ax, 17.0, 21.3, 2.5, 1.0,
            "★ D-048 검증\nja_drawing 110 region\n자동 분리 ✅",
            COLORS["new_v4"], fontsize=7, bold=True)

    add_arrow(ax, 2.0, 22.4, 2.0, 22.2)
    add_arrow(ax, 3.5, 22.95, 3.8, 22.95)
    add_arrow(ax, 6.3, 22.95, 6.6, 22.95)
    add_arrow(ax, 8.2, 22.4, 8.2, 22.2)
    add_arrow(ax, 9.8, 22.95, 10.1, 22.95)
    add_arrow(ax, 13.1, 22.95, 13.4, 22.95)

    # ============================================================
    # PHASE 3 — Stage 2 학습 (D-034 계층적 + V3-B K-fold Ensemble)
    # ============================================================
    add_phase_bg(ax, 0.2, 17.4, 19.6, 3.2, "Phase 3 — Stage 2 학습 (D-034 계층적 + V3-B 5-fold Ensemble D-023 PASS)")

    add_box(ax, 0.5, 18.9, 3.0, 1.2,
            "Stage 1 추론 + Crop\nstage1_layout.py\n5 클래스 분리",
            COLORS["dl_vlm"], fontsize=7)
    add_box(ax, 3.8, 18.9, 3.0, 1.2,
            "outputs/crops/<id>/\n+ View / Isometric\n+ TitleBlock / Notes\n+ ★ PMI (Stage 2)",
            COLORS["io"], fontsize=7)
    add_box(ax, 7.1, 18.9, 3.0, 1.2,
            "extract_pmi_crops_v3.py\n844 PMI crops\n(adaptive padding)",
            COLORS["deterministic"], fontsize=7)
    add_box(ax, 10.4, 18.9, 3.2, 1.2,
            "CVAT OBB 라벨링\nMeasure 555 / GDT 88\n/ Roughness 106 + SKIP 277",
            COLORS["hitl"], fontsize=7)
    add_box(ax, 13.9, 18.9, 3.0, 1.2,
            "Stage 2 학습 (yolo11l-obb)\nimgsz 1280 / 200 ep / 5-fold\n★ overnight 8.7h",
            COLORS["dl_vlm"], fontsize=7)
    add_box(ax, 17.1, 18.9, 2.5, 1.2,
            "★ Ensemble (D-023 PASS)\ncheckpoints/yolo_obb.pt\nensemble_predict.py",
            COLORS["new_v3"], fontsize=7, bold=True)

    add_box(ax, 4.5, 17.6, 4.5, 1.0,
            "perspective-warp 회전 정렬\nstage2_annotation.crop (D-012)\nStage 3-N 입력 patch 생성",
            COLORS["deterministic"], fontsize=7)

    add_arrow(ax, 3.5, 19.5, 3.8, 19.5)
    add_arrow(ax, 6.8, 19.5, 7.1, 19.5)
    add_arrow(ax, 10.1, 19.5, 10.4, 19.5)
    add_arrow(ax, 13.6, 19.5, 13.9, 19.5)
    add_arrow(ax, 16.9, 19.5, 17.1, 19.5)
    add_arrow(ax, 11.5, 18.9, 11.5, 18.6)

    # ============================================================
    # PHASE 4 — Stage 3-A PaddleOCR-VL-1.5 (★ NEW v4)
    # ============================================================
    add_phase_bg(ax, 0.2, 13.0, 19.6, 4.0, "★ Phase 4 — Stage 3-A PaddleOCR-VL-1.5 Zero-shot (D-039, D-042, D-046, D-047)")

    add_box(ax, 0.5, 15.4, 3.0, 1.4,
            "★ D-042 monkey-patch\nconfig.text_config\n= config.get_text_config()\n(transformers 5.0.0)",
            COLORS["new_v4"], fontsize=7, bold=True)
    add_box(ax, 3.8, 15.4, 3.0, 1.4,
            "stage3_paddleocr_zero\n_shot_test.py (809 lines)\n★ D-046 호출 방식\n(task keyword + bf16)",
            COLORS["new_v4"], fontsize=7)
    add_box(ax, 7.1, 15.4, 3.0, 1.4,
            "1차 평가 (V5 미통과)\ndegenerate generation\n→ D-045 generation params",
            COLORS["dl_vlm"], fontsize=7)
    add_box(ax, 10.4, 15.4, 3.0, 1.4,
            "2~3차 평가 (V5 미통과)\nTitleBlock 부정확\nko hallucination",
            COLORS["dl_vlm"], fontsize=7)
    add_box(ax, 13.7, 15.4, 3.0, 1.4,
            "★ Real-ESRGAN 4x upscale\nupscale_images_realesrgan.py\n5장 4.9s / 640→2560",
            COLORS["new_v4"], fontsize=7, bold=True)
    add_box(ax, 17.0, 15.4, 2.5, 1.4,
            "★ 4차 평가\navg 0.69 (부분 PASS)\nko/zh 큰 향상",
            COLORS["new_v4"], fontsize=7, bold=True)

    add_box(ax, 0.5, 13.5, 4.5, 1.4,
            "TitleBlock crop / Notes crop\n(D-046 task keyword:\n\"Table Recognition:\" / \"OCR:\")\nbf16 + apply_chat_template\n+ processor.decode 슬라이스",
            COLORS["dl_vlm"], fontsize=7)
    add_box(ax, 5.3, 13.5, 4.5, 1.4,
            "★ V5 char accuracy 결과 (4차)\nko ~0.78 / zh ~0.75 (★★ 큰 향상)\nen ~0.65 / ru ~0.68\nja ~0.50 (다중 도면 한계)\nde 미평가\n→ V5 부분 PASS 인정",
            COLORS["dl_vlm"], fontsize=7)
    add_box(ax, 10.1, 13.5, 4.5, 1.4,
            "★ Known Limitations §3 추가\n• ja_drawing 다중 도면 영역별 미검증\n• V5 0.69 < 0.85 임계 미달\n• PaddleOCR-VL fine-tune 미진행",
            COLORS["limitation"], fontsize=7)
    add_box(ax, 14.9, 13.5, 4.6, 1.4,
            "★ D-048 후속 (다음 날)\nyolo_det.pt 110 region 분리\n→ 영역별 PaddleOCR-VL 평가\n(option, 30분 예상)",
            COLORS["active_learning"], fontsize=7)

    add_arrow(ax, 3.5, 16.1, 3.8, 16.1)
    add_arrow(ax, 6.8, 16.1, 7.1, 16.1)
    add_arrow(ax, 10.1, 16.1, 10.4, 16.1)
    add_arrow(ax, 13.4, 16.1, 13.7, 16.1)
    add_arrow(ax, 16.7, 16.1, 17.0, 16.1)

    # ============================================================
    # PHASE 5 — Stage 3-N Donut Numerical fine-tune (★ NEW v4, 진행 중)
    # ============================================================
    add_phase_bg(ax, 0.2, 8.4, 19.6, 4.4, "★ Phase 5 — Stage 3-N Donut Numerical Fine-tune 1차 baseline (D-049, D-050, D-051) — 진행 중 (2026-05-05)")

    add_box(ax, 0.5, 10.9, 3.4, 1.5,
            "Phase 16a (~1h 40분)\nprepare_vlm_dataset.py\nnumerical --limit 500\n★ D-049 sys.path bootstrap\n(★ uv pip install src 금지)",
            COLORS["new_v4"], fontsize=7, bold=True)
    add_box(ax, 4.2, 10.9, 3.4, 1.5,
            "data/vlm/numerical/\n~13,000 region 예상\nMeasure 86% / GDT 2.6%\n/ Roughness 11%\n+ ocr_hint (Tesseract)",
            COLORS["io"], fontsize=7)
    add_box(ax, 7.9, 10.9, 3.4, 1.5,
            "★ auto_fill_numerical_gt.py\n(★ 신규 452 lines)\nMeasure: ocr_numeric → nominal\nGDT: symbol regex (14)\nRoughness: Ra regex",
            COLORS["new_v4"], fontsize=7, bold=True)
    add_box(ax, 11.6, 10.9, 3.6, 1.5,
            "★ Auto-fill 결과 (dry-run)\nMeasure 62.2% ✅\nGDT 0% ❌\nRoughness 30% ⚠️\nOverall 57%",
            COLORS["new_v4"], fontsize=7)
    add_box(ax, 15.5, 10.9, 4.0, 1.5,
            "Phase 16b 1차 baseline\nstage3_numerical.py train\n--cfg configs/donut_numerical.yaml\nepochs 30 / batch 4 / lr 1e-6\novernight ~6h",
            COLORS["dl_vlm"], fontsize=7, bold=True)

    add_box(ax, 0.5, 9.0, 4.7, 1.7,
            "★ D-050 박제 — Tesseract OCR 한계 (Critical)\n• --psm 6 + kor+eng+rus+jpn\n• 도면 patch 작은 글자 (10~14px) + 한자/일본어/한글 혼재\n• tolerance ± 부호 인식 0%\n• GDT symbol (⌖/⏤/⊥) 인식 0%\n• Ra 키워드 인식 거의 0%",
            COLORS["limitation"], fontsize=7)
    add_box(ax, 5.5, 9.0, 4.7, 1.7,
            "★ D-051 박제 — 1차 baseline = Measure-only\n• Measure nominal extraction 에 한정\n• GDT 학습 사실상 불가 (sample 13/500 + auto-fill 0%)\n• Roughness 30% 제한적\n• Phase 17 e2e 검증의 자리만 채움\n• 후속 개선 우선순위 정량화",
            COLORS["limitation"], fontsize=7)
    add_box(ax, 10.5, 9.0, 4.7, 1.7,
            "★ D-049 박제 — sys.path bootstrap pattern\n• prepare_vlm_dataset.py / auto_fill_numerical_gt.py\n• pipeline.py (Task #92) 동일 적용\n• 후속: src/validate/check_*.py 9개 파일\n★ 절대 금지: pip install src (PyPI 무관 패키지)",
            COLORS["limitation"], fontsize=7)
    add_box(ax, 15.5, 9.0, 4.0, 1.7,
            "★ 후속 (Phase 18+)\n• 검수 도구 작성 (Streamlit/CVAT)\n• 사람 검수 ~3일\n• GDT crop 500 추가 라벨링\n• Stage 3-N full GT 재학습",
            COLORS["active_learning"], fontsize=7)

    add_arrow(ax, 3.9, 11.65, 4.2, 11.65)
    add_arrow(ax, 7.6, 11.65, 7.9, 11.65)
    add_arrow(ax, 11.3, 11.65, 11.6, 11.65)
    add_arrow(ax, 15.2, 11.65, 15.5, 11.65)

    # ============================================================
    # PHASE 6 — Pipeline + Step 9 Enrichment (Phase 17 e2e 미진행)
    # ============================================================
    add_phase_bg(ax, 0.2, 4.6, 19.6, 3.6, "Phase 6 — Pipeline + Step 9 Enrichment (★ Phase 17 e2e 미진행 — 2026-05-06 이후)")

    add_box(ax, 0.5, 6.7, 3.5, 1.3,
            "Stage 3-A: PaddleOCR-VL\nzero-shot\nTitleBlock/Notes crop\n★ D-046 fix",
            COLORS["new_v4"], fontsize=7)
    add_box(ax, 0.5, 5.3, 3.5, 1.3,
            "Stage 3-N: Donut Numerical\n★ Phase 16b 학습 후\nMeasure/GDT/Roughness JSON",
            COLORS["new_v4"], fontsize=7)
    add_box(ax, 4.3, 5.7, 3.0, 1.5,
            "Stage 4: JSON Merger\npipeline.py\nOBB local → global",
            COLORS["deterministic"], fontsize=7)
    add_box(ax, 7.6, 5.7, 3.5, 1.5,
            "Unified Structured JSON\n(HANDOFF §5.5)\nview / annotation\n/ TitleBlock / Notes",
            COLORS["io"], fontsize=7)
    add_box(ax, 11.4, 5.7, 4.0, 1.5,
            "★ Step 9 Enrichment (D-019/020)\nstage5_enrichment.py\n4-tier cascade:\n1. Deterministic (KB lookup)\n2. Heuristic (rule-based)\n3. RAG-LLM (Mock/Gemini/Qwen)\n4. HITL gate (low conf flag)",
            COLORS["new_v3"], fontsize=7, bold=True)
    add_box(ax, 15.7, 6.5, 3.8, 0.7,
            "Knowledge Base\nmaterial_catalog.json (incl CN)\niso_2768.json, roughness.json",
            COLORS["kb"], fontsize=7)
    add_box(ax, 15.7, 5.7, 3.8, 0.7,
            "HITL flag\n(low_conf / cost-controlled)",
            COLORS["hitl"], fontsize=7)

    add_box(ax, 7.6, 4.7, 7.8, 0.9,
            "★ Final Enriched JSON (per drawing)\nmaterial / tolerance / roughness / process / qc + provenance: source / confidence / cost (D-022)",
            COLORS["final"], fontsize=8, bold=True)

    add_arrow(ax, 4.0, 7.3, 4.3, 6.5)
    add_arrow(ax, 4.0, 5.95, 4.3, 6.5)
    add_arrow(ax, 7.3, 6.45, 7.6, 6.45)
    add_arrow(ax, 11.1, 6.45, 11.4, 6.45)
    add_arrow(ax, 15.4, 6.85, 15.7, 6.85)
    add_arrow(ax, 15.4, 6.05, 15.7, 6.05)
    add_arrow(ax, 13.4, 5.7, 13.4, 5.6)

    # ============================================================
    # Validation Framework V0~V9
    # ============================================================
    add_phase_bg(ax, 0.2, 2.6, 19.6, 1.7, "Validation Framework (V0 ~ V9)")

    val_x = 0.7
    validations = [
        ("V0\ncommon.py", COLORS["deterministic"], False),
        ("V1\nsort_titleblock", COLORS["deterministic"], False),
        ("V2-A\nlabels_yolo", COLORS["deterministic"], False),
        ("V2-B\nstage1_model\n(D-029 fix)", COLORS["new_v3"], True),
        ("V3-A\nlabels_obb", COLORS["deterministic"], False),
        ("V3-B\nstage2_model\n(D-023 ✅)", COLORS["deterministic"], False),
        ("★ V5\nstage3a\n부분 PASS", COLORS["new_v4"], True),
        ("V6\nstage3n\n(★ 진행)", COLORS["new_v4"], True),
        ("V7\npipeline_e2e\n(★ 미진행)", COLORS["limitation"], False),
        ("V9\nenrichment\n(★ 미진행)", COLORS["limitation"], False),
    ]
    for i, (label, color, bold) in enumerate(validations):
        add_box(ax, val_x + i * 1.92, 2.85, 1.7, 1.05, label, color, fontsize=7, bold=bold)

    # ============================================================
    # Known Limitations 박스
    # ============================================================
    add_phase_bg(ax, 0.2, 0.3, 19.6, 2.0, "★ Known Limitations 매트릭스 (docs/KNOWN_LIMITATIONS.md, 2026-05-05)")
    add_box(ax, 0.5, 0.6, 4.6, 1.5,
            "★★★ Critical\n• D-050 Tesseract OCR 한계\n• Stage 2 GDT 라벨 부족 (2.6%)",
            COLORS["limitation"], fontsize=7, bold=True)
    add_box(ax, 5.4, 0.6, 4.6, 1.5,
            "★★ High\n• Stage 3-A V5 0.69 < 0.85\n• ja_drawing 영역별 미검증\n• 검수 도구 부재",
            COLORS["limitation"], fontsize=7)
    add_box(ax, 10.3, 0.6, 4.6, 1.5,
            "★ Medium\n• stage1_fp_notes (Phase 15d)\n• D-026 가공/조립 분류 실패\n• Roughness Ra fallback 30%",
            COLORS["limitation"], fontsize=7)
    add_box(ax, 15.2, 0.6, 4.3, 1.5,
            "☆ Low / Resolved\n• D-036 V.A 회전 증강\n• D-042/D-046 Resolved\n• D-049 Resolved",
            COLORS["deterministic"], fontsize=7)

    ax.text(10, 0.05,
            "최신 학습: history.md §A.12.9 | 의사결정 박제: PROJECT_HANDOFF.md §11 (D-001 ~ D-051) | 한계: docs/KNOWN_LIMITATIONS.md",
            ha="center", fontsize=8, style="italic", color="#555")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
