"""Validation framework for the multi-stage hybrid pipeline.

See PROJECT_HANDOFF.md §14 (Validation Procedures) and
configs/validation_thresholds.yaml for severity thresholds.

Module map (see also README §6.5):

- common.py                  : CheckResult / ValidationReport / HTML / JSON
- check_step1_5_sorter.py    : V1
- check_labels_yolo.py       : V2 (Stage 1 labels)
- check_labels_obb.py        : V3 (Stage 2 labels)
- check_stage1_model.py      : V2 (mAP, per-class)
- check_stage1_crops.py      : V2 (crop quality)
- check_stage2_model.py      : V3 (★ missing rate, recall — D-023)
- check_stage2_warps.py      : V3 (perspective-warp legibility)
- check_stage3a_alphabetical.py : V5 (zero-shot Donut F1, hallucination)
- check_stage3n_numerical.py    : V6 (fine-tuned Donut F1, schema)
- check_pipeline_e2e.py      : V7
- check_enrichment.py        : V9
- run_all.py                 : orchestrator
"""

from .common import (
    CheckResult,
    ValidationReport,
    Severity,
    Status,
    load_thresholds,
)

__all__ = [
    "CheckResult",
    "ValidationReport",
    "Severity",
    "Status",
    "load_thresholds",
]
