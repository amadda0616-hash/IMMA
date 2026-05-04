# `src/pipeline.py`

> **Step 7** — End-to-end Engineering Drawing JPG → 통합 Structured JSON

## 1. 구현 요약

Stage 1 → Stage 2 → Stage 3-A / 3-N → Stage 4 (merge) 를 단일 `Pipeline` 클래스로 오케스트레이션.

**핵심 컴포넌트** (~600 lines)

| 영역 | 구현 |
|---|---|
| `Pipeline` 클래스 | 4개 모델을 1회 로드 후 재사용 |
| Stage 1 (lazy) | `_run_stage1()` — `stage1_layout.predict_one` |
| Stage 2 (lazy, per-View / per-PMI) | `_run_stage2()` — View 또는 PMI crop 위에서 OBB 검출 (D-028 신규: PMI 권장). **★ D-040 (2026-05-04): default = 5-Fold Ensemble** (단일 best.pt 는 `--no-ensemble` 시 fallback) |
| Stage 2 Ensemble | `_ensure_ensemble()` + `ensemble_predict.predict_one_schema()` — 5 fold best.pt 동시 추론 → class-wise NMS |
| Stage 3-A (lazy, optional) | `_run_stage3_alphabetical()` — Donut zero-shot |
| Stage 3-N (lazy, optional) | `_run_stage3_numerical()` — Donut fine-tuned |
| OBB 좌표 변환 | `obb_local_to_global()` — view-crop → 원본 도면 |
| Stage 4 merge | `run()` 메서드 끝에서 통합 JSON 생성 |
| Batch | `run_batch()` — timing/error log + summary JSON |
| Auto-skip | Donut Numerical 미존재 시 Stage 3-N 자동 건너뜀 |

**2개 CLI 서브커맨드**

```
run    ─ 단일 도면 → outputs/<id>.json
batch  ─ 폴더 일괄 → outputs/<id>.json + _pipeline_summary.json
```

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 모델 1회 로드 | 클래스 init 에서 모든 weights 가용성 검사, VLM 은 lazy load | RTX 5080 16GB, 배치 효율 |
| **★ Stage 2 Ensemble (default)** | 5 fold best.pt 동시 추론 + class-wise rotated NMS (`iou_nms=0.5`) | D-040: V3-B 단일 모델 Measure missing 0.101 → 0.000 (D-023 PASS) |
| Stage 2 weight 검증 | `use_ensemble=True` 시 5 fold best.pt 모두 존재 확인, False 시 단일 best.pt | 빠른 실패 + legacy 호환 |
| Ensemble fold lazy load | 첫 `_run_stage2()` 호출 시 `_ensure_ensemble()` 로 5 모델 로드 | CLI `--help` 빠른 응답 |
| **Lazy import** | stage1/2/3 모듈 import 를 메서드 내부로 | CLI `--help` 시 ultralytics/transformers 불필요 |
| Auto-skip Stage 3-N | donut_num_ckpt 미존재 → 자동 skip | 학습 전 부분 검증 가능 |
| Skip flags | `--skip-numerical` / `--skip-alphabetical` | 점진적 통합 검증 |
| 좌표 변환 | View bbox 의 (x0, y0) 더해 OBB 글로벌 좌표 | HANDOFF §5.5 schema |
| **OBB 두 좌표 보존** | `obb_global` + `obb_local` 모두 출력 | 다운스트림 (CAD 매핑 / 디버깅) 양쪽 활용 |
| 임시 파일 | `outputs/_pipeline_tmp/<drawing_id>/` 자동 생성·정리 | crop / warp 패치는 디스크 경유 (Stage 3 가 path 입력) |
| `--keep-tmp` | 디버깅 시 임시 산출물 보존 | 검증·재현 |
| timing 측정 | 단계별 + total 초 단위 기록 | D-021 임계값 (≤ 30s) 검증 |
| Batch 결과 | `_pipeline_summary.json` 에 ok/err/avg time 집계 | V7 검증기 입력 |
| 다국어 hint | `--language` 로 Stage 3-A 에 전달 | D-013 4개 언어 |
| 의존성 graceful | weights 미존재 → FileNotFoundError + 명확한 메시지 | 사용자 안내 |

## 3. 사용법

### CLI

```bash
# 단일 도면 (모든 단계)
python src/pipeline.py run \
    --image dataset/sample.jpg \
    --out outputs/sample.json --device 0

# 배치 처리
python src/pipeline.py batch \
    --input-dir dataset/ \
    --out-dir outputs/json --device 0

# Donut 학습 전 — YOLO 만 (Stage 3 skip)
python src/pipeline.py run \
    --image dataset/sample.jpg \
    --skip-numerical --skip-alphabetical

# ★ Stage 2 ensemble OFF (단일 best.pt, 디버깅/legacy)
python src/pipeline.py run \
    --image dataset/sample.jpg \
    --no-ensemble --obb-weights checkpoints/yolo_obb.pt

# ★ Stage 2 ensemble 커스텀 (top-3 fold 만, 속도 우선)
python src/pipeline.py run \
    --image dataset/sample.jpg \
    --n-folds 3 --iou-nms 0.5 --conf-obb 0.30

# 디버깅: 임시 파일 보존 + 처음 5장만
python src/pipeline.py batch \
    --input-dir dataset/ --out-dir outputs/json \
    --keep-tmp --limit 5
```

### CLI 인자 (Stage 2 Ensemble, ★ D-040)

```
--use-ensemble          (default ON) Stage 2 5-fold ensemble
--no-ensemble           단일 best.pt 사용 (--obb-weights 필요)
--ensemble-ckpt-root    checkpoints/yolo_obb_runs (default)
--ensemble-fold-pattern yolo_obb_v3_kfold_{i} (default)
--n-folds               5 (default)
--iou-nms               0.5 (default, cross-fold rotated NMS)
```

### Python import

```python
from src.pipeline import Pipeline
from pathlib import Path

p = Pipeline(
    det_weights="checkpoints/yolo_det.pt",
    obb_weights="checkpoints/yolo_obb.pt",   # only used if use_ensemble=False
    donut_num_ckpt="checkpoints/donut_numerical/final",
    device="0",
    # ---- Stage 2 5-Fold Ensemble (default ON, D-040) ----
    use_ensemble=True,
    ensemble_ckpt_root="checkpoints/yolo_obb_runs",
    n_folds=5,
    iou_nms=0.5,
)

# 단일 도면
rec = p.run(Path("dataset/sample.jpg"), language_hint="en")
# rec = HANDOFF §5.5 schema

# 배치
images = sorted(Path("dataset").glob("*.jpg"))
summary = p.run_batch(images, Path("outputs/json"))
# summary = {n_total, n_ok, n_err, total_seconds, avg_seconds_per_drawing, log[...]}
```

## 4. 검증 결과

### 4.1 단위 테스트 (lazy import)

```
✓ pipeline 모듈 로드 OK (stage1/2/3 의존성 없는 환경)
```

`/tmp/test_pipe2/` 깨끗한 디렉터리에서 `from src import pipeline` 정상 동작 — ultralytics / transformers 미설치 환경에서도 모듈 로드 가능. CLI `--help` 도 정상.

### 4.2 OBB 좌표 변환 검증

```python
view = [1000, 500, 2500, 1700]
local = [[100, 50], [300, 50], [300, 150], [100, 150]]
got = obb_local_to_global(local, view)
# [[1100, 550], [1300, 550], [1300, 650], [1100, 650]]  ✓
```

각 점에 view 의 (x0, y0) = (1000, 500) 가 더해져 정확히 글로벌 좌표로 환산.

### 4.3 CLI 서브커맨드

```
$ python -m src.pipeline --help
usage: pipeline.py [-h] {run,batch} ...

positional arguments:
  {run,batch}
    run        단일 도면 처리
    batch      폴더 일괄 처리
```

3개 weights 옵션 + 2개 skip flag + language/keep-tmp 모두 정상 파싱.

### 4.4 모델 학습 후 평가 (V7)

V7 (`check_pipeline_e2e.py`) 작성 후 자동 측정 항목:

| 항목 | 임계값 (D-021) | Severity |
|---|---|---|
| field-level F1 | ≥ 0.75 | critical |
| 도면당 추론 시간 | ≤ 30s | warning |
| 실패율 | < 1% | critical |
| GPU 메모리 peak | < 14 GB | warning |

## 5. 출력 형식

### 5.1 단일 도면 결과 JSON (HANDOFF §5.5)

```json
{
  "drawing_id": "drawing_001",
  "image_path": "dataset/drawing_001.jpg",
  "image_size": [4961, 3508],
  "title_block": {
    "drawing_no": "DWG-001-A",
    "material": "SS400",
    "scale": "1:2"
  },
  "notes": [
    "1. UNLESS OTHERWISE SPECIFIED",
    "2. ALL DIMENSIONS IN MM"
  ],
  "views": [
    {
      "view_id": "view_0",
      "bbox": [320, 480, 3850, 2780],
      "conf": 0.97,
      "annotations": [
        {
          "class": "Measure",
          "obb_global": [[770, 1000], [4020, 1000], [4020, 1040], [770, 1040]],
          "obb_local":  [[450, 520],  [3700, 520],  [3700, 560],  [450, 560]],
          "angle": 0.0,
          "conf": 0.93,
          "parsed": {
            "nominal": 480.0,
            "tolerance": null,
            "unit": "mm"
          }
        }
      ]
    }
  ],
  "meta": {
    "model_versions": {
      "yolo_det":    "yolo_det.pt",
      "yolo_obb":    "yolo_obb.pt",
      "donut_alpha": "donut-base-finetuned-docvqa",
      "donut_num":   "checkpoints/donut_numerical/final"
    },
    "timing_seconds": {
      "stage1":              1.21,
      "stage1_crop":         0.18,
      "stage2":              3.42,
      "stage3_alphabetical": 14.80,
      "stage3_numerical":    8.95,
      "total":               28.56
    },
    "timestamp": "2026-04-27T15:30:00+00:00",
    "language_hint": "en"
  }
}
```

### 5.2 Batch summary JSON

```json
{
  "n_total": 5839,
  "n_ok": 4571,
  "n_err": 16,
  "total_seconds": 134820.5,
  "avg_seconds_per_drawing": 29.39,
  "log": [
    {"image": "...", "json": "...", "status": "ok", "total_s": 28.56},
    {"image": "...", "json": "",    "status": "error", "error": "..."}
  ]
}
```

### 5.3 디렉터리 구조

```
outputs/
├── drawing_001.json                    ← 통합 결과 (단일 모드)
├── json/                               ← batch 모드
│   ├── drawing_001.json
│   ├── drawing_002.json
│   ├── ...
│   └── _pipeline_summary.json
└── _pipeline_tmp/                      ← 자동 정리 (--keep-tmp 미지정 시)
    └── <drawing_id>/
        ├── stage1_crops/
        └── stage2_warps/
```

## 6. 의존성

```
ultralytics (Stage 1/2)         # lazy import
transformers (Stage 3-A/3-N)    # lazy import
torch (CUDA 12.8)
opencv-python, numpy
shapely (Stage 2 OBB IoU 백업)  # 옵션
```

CLI `--help` 시점은 위 의존성 모두 불필요 (lazy import).

## 7. 관련 의사결정

- **D-001** 아키텍처 = 논문 (4단계 통합)
- **D-012** Stage 2 OBB de-rotation 적용 (warp 후 Stage 3-N)
- **D-013** 4개 언어 — `language_hint` 파라미터로 전달
- **D-018** Stage 3 모델 = Donut (alphabetical zero-shot, numerical fine-tune)
- **D-021** 추론 시간 ≤ 30s critical (timing 측정 의무)
- **D-022** Provenance — meta 블록에 model_versions / timing / timestamp 기록
- **D-024** Group-aware (출력에는 영향 없음, 학습 시점 적용)

## 8. 검증 모듈

[`check_pipeline_e2e.py`](./../README.md) — V7, 작성 예정

## 9. 업스트림 / 다운스트림

**업스트림 (이 모듈이 사용)**
- `stage1_layout.predict_one()` / `crop_regions()`
- `stage2_annotation.predict_one()` / `crop_obb_regions()`
- `stage3_alphabetical.load_model()` / `predict_one()`
- `stage3_numerical.load_inference_model()` / `predict_one()`

**다운스트림 (이 모듈을 사용)**
- **`stage5_enrichment.py`** (Step 9) — 통합 JSON + image → enriched JSON
- **`check_pipeline_e2e.py`** (V7) — field-level F1 / 시간 / 실패율 측정
- **`utils/metrics.py`** (Step 8) — 평가 지표

## 10. 성능 최적화 팁

| 상황 | 조치 |
|---|---|
| 추론 시간 > 30s | (1) `--skip-alphabetical` (DocVQA 14회 호출이 가장 비싼 부분) (2) `--imgsz-det 1024` |
| GPU 메모리 부족 | `--device cpu` (느려지나 동작) 또는 `--imgsz-obb 768` |
| 5,839 도면 일괄 처리 | `nohup python src/pipeline.py batch ... &` 백그라운드 권장 |
| 재시작 가능성 | 기존 출력 JSON 존재 시 skip 로직 추가 검토 (현재는 덮어씀) |

## 11. FAIL 발생 시 조치

| 증상 | 원인 / 해결 |
|---|---|
| `Stage 1 weights not found` | `python src/stage1_layout.py train` 먼저 |
| `Stage 2 weights not found` | Step 3 학습 후 |
| Stage 3-N auto-skipped | Donut Numerical fine-tune (Step 6) 미완 — 정상 |
| Stage 3-A 모델 로드 실패 | transformers / sentencepiece 설치 확인 |
| 추론 시간 너무 김 | DocVQA 14회 호출 → `--skip-alphabetical` 또는 batch 크기 ↓ |
| 한글 파일명 오류 | `np.fromfile 