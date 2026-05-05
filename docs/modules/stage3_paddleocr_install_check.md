# `src/stage3_paddleocr_install_check.py`

> **Phase 15a** — PaddleOCR-VL-1.5 환경 자동 검증 도구 (★ D-042 monkey-patch 자동 적용)

## 1. 구현 요약

D-039 (Stage 3-A → PaddleOCR-VL-1.5 채택) 의 실제 환경 작동 여부를 자동 확인. `.venv-paddleocr` 별도 venv 에서 실행. 7차에 걸친 시도-실패 끝에 발견된 **monkey-patch workaround** 를 자동 적용하여 환경 재현성 보장.

**5단계 검증 흐름**:
1. **환경 정보 수집** — torch / transformers / GPU
2. **Model 로드** (Config + monkey-patch + Processor + Model)
3. **GPU 메모리 측정** (모델 로드 후)
4. **더미 이미지 inference** (256×128, 32 토큰 generate)
5. **PASS / FAIL 자동 판정** — 3개 조건 + JSON 출력

**판정 조건**:
- `cuda_available` = True
- `model_params_b` >= 0.5 (0.9B 모델 기준)
- inference 정상 (또는 `--skip-inference`)

**종료 코드**: PASS = 0, FAIL = 1 (CI/CD 친화).

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 별도 venv (`.venv-paddleocr`) | Phase 14 ultralytics 와 분리 | transformers 4 → 5 충돌 회피 |
| transformers 버전 | **5.0.0** | 5.6+ rope deprecated, 5.0 안정적 |
| **★ monkey-patch (D-042)** | `config.text_config = config.get_text_config()` 1줄 | 5.x native paddleocr_vl 의 schema mismatch 우회 |
| Model 클래스 | `AutoModelForImageTextToText` (NOT `AutoModel`) | README 명시, multi-modal task 호환 |
| Processor 클래스 | `AutoProcessor` (NOT `AutoTokenizer`) | image + text 통합 처리 |
| dtype | `float16` | RTX 5080 16GB 충분 + 추론 속도 ↑ |
| 더미 이미지 | 256×128 흰 배경 | 빠른 검증 + chat template 작동 확인 |
| `max_new_tokens` | 32 | 검증 목적, 실 사용은 더 길게 |
| 출력 | JSON (env / gpu / model / inference) | CI/CD + 박제 통합 |

## 3. 사용법

### 3.1 환경 구축 (최초 1회)

```bash
cd /mnt/c/Users/user/github/Drawing

# 별도 venv 생성
uv venv --python 3.10 .venv-paddleocr
source .venv-paddleocr/bin/activate

# 의존성 (cu128 + Blackwell sm_120, D-030)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers==5.0.0" accelerate sentencepiece protobuf einops pillow
```

### 3.2 검증 실행

```bash
source .venv-paddleocr/bin/activate

# 전체 검증 (env + load + inference)
python src/stage3_paddleocr_install_check.py

# 빠른 모드 (inference 생략, 환경 + load 만)
python src/stage3_paddleocr_install_check.py --skip-inference

# 출력 JSON 경로 변경
python src/stage3_paddleocr_install_check.py \
    --output outputs/stage3a_install_check.json
```

**종료 코드**:
- `0` — PASS (Phase 15a 완료)
- `1` — FAIL (조건 미충족, JSON `error` 필드 확인)

### 3.3 CLI 인자

```
--output OUTPUT       Result JSON 경로 (default: outputs/stage3a_install_check.json)
--skip-inference      더미 이미지 inference 건너뛰기 (env + load 만)
--quiet               간소화 출력
```

## 4. 검증 결과 (★ 2026-05-04 실측)

```
========================================================================
  Phase 15a — PaddleOCR-VL-1.5 환경 검증 — PASS
========================================================================

[Step 1] 환경 정보
  python                    3.10.20
  platform                  linux
  torch                     2.11.0+cu128
  cuda_available            True
  cuda_capability           (12, 0)
  gpu_name                  NVIDIA GeForce RTX 5080
  gpu_total_gb              17.09
  gpu_free_gb_initial       15.68
  transformers              5.0.0

[Step 2] PaddleOCR-VL-1.5 로드 (monkey-patch 적용)
  ★ Patch applied: config.text_config = config.get_text_config()
  Model params:  0.91B
  Config class:  PaddleOCRVLConfig
  Load time:     39.7s

[Step 3] GPU 메모리 (모델 로드 후)
  gpu_free_gb_after_load    13.81
  gpu_used_gb_by_model      3.29
  gpu_total_gb              17.09

[Step 4] 더미 이미지 inference
  inference_time_s          2.26 ~ 3.47
  output_preview            "User: What is in this image?
                             Assistant:
                             The provided image is a logo or a graphi..."

[Step 5] 판정 — ★ PASS
========================================================================
```

## 5. 출력 형식 (`outputs/stage3a_install_check.json`)

```json
{
  "phase": "15a",
  "model_id": "PaddlePaddle/PaddleOCR-VL-1.5",
  "status": "PASS",
  "env": {
    "python": "3.10.20",
    "platform": "linux",
    "torch": "2.11.0+cu128",
    "cuda_available": true,
    "cuda_capability": "(12, 0)",
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "gpu_total_gb": 17.09,
    "gpu_free_gb_initial": 15.68,
    "transformers": "5.0.0"
  },
  "load_time_s": 39.7,
  "model_params_b": 0.91,
  "config_class": "PaddleOCRVLConfig",
  "gpu": {
    "gpu_free_gb_after_load": 13.81,
    "gpu_used_gb_by_model": 3.29,
    "gpu_total_gb": 17.09
  },
  "inference": {
    "inference_time_s": 2.26,
    "output_preview": "User: What is in this image?\nAssistant: ..."
  }
}
```

## 6. 의존성

- Python 3.10+
- torch 2.11+cu128 (Blackwell sm_120, D-030)
- transformers 5.0.0 (★ 정확한 버전, 다른 버전은 호환성 이슈)
- accelerate, sentencepiece, protobuf, einops, pillow
- HuggingFace Hub 접속 (모델 다운로드 — 첫 실행만)
- 디스크: ~2 GB (모델 cache)
- GPU: 4+ GB VRAM (float16)

## 7. 트러블슈팅

### Q1. `KeyError: 'default'` (ROPE_INIT_FUNCTIONS)
→ transformers 5.6+ 에서 ROPE API 변경. **5.0.0 으로 다운그레이드** + `AutoModelForImageTextToText` 사용 (NOT `AutoModel`).

### Q2. `ModuleNotFoundError: 'transformers.masking_utils'`
→ transformers 4.x 에서 발생. **5.0.0 이상** 필요. 단, 5.6+ 는 다른 호환성 이슈.

### Q3. `AttributeError: 'PaddleOCRVLConfig' object has no attribute 'text_config'`
→ ★ D-042 monkey-patch 미적용. 본 스크립트의 `load_paddleocr_vl()` 함수가 자동 처리하지만, 다른 코드에서 직접 모델 로드 시:
```python
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
if not hasattr(config, "text_config") and hasattr(config, "get_text_config"):
    config.text_config = config.get_text_config()
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, config=config, ...)
```

### Q4. `ValueError: Cannot instantiate this tokenizer from a slow version`
→ `sentencepiece` 누락. `uv pip install sentencepiece` 후 재시도.

### Q5. `ImportError: This modeling file requires einops`
→ `uv pip install einops`

## 8. 관련 파일

- `src/stage3_paddleocr_install_check.py` — 본 스크립트
- `outputs/stage3a_install_check.json` — 검증 결과
- `docs/PHASE15_CHECKLIST.md` — 15a 체크리스트
- `history.md §A.12.1 ~ §A.12.2` — 시도 매트릭스 + 결과
- `PROJECT_HANDOFF.md §11 D-042` — monkey-patch 박제

## 9. 차후 검토

| 조건 | 액션 |
|---|---|
| PaddleOCR-VL-2.0 출시 | monkey-patch 폐기 가능 여부 재확인 |
| transformers 5.x major 버전 업그레이드 | ROPE / masking API 호환성 재검증 |
| Phase 17 batch 단계 | vLLM 도입 ROI 측정 (transformers 7h vs vLLM 1.5h 추정) |
| native paddleocr_vl 버그 수정 | transformers PR 또는 직접 patch 제출 |

---

**Last updated**: 2026-05-04 (Phase 15a DONE)
