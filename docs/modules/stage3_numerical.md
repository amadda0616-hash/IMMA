# `src/stage3_numerical.py`

> **Step 6** — Donut Numerical VLM (fine-tune + inference)

## 1. 구현 요약

Stage 2 의 perspective-warped 패치 (Measure / GDT / Roughness) 를 입력받아 schema-defined JSON 을 출력하는 fine-tuned Donut 모듈.

**3개 서브커맨드**

```
train     ─ HF Trainer 기반 fine-tuning (FP16/BF16/8bit 옵션)
predict   ─ 단일 패치 → 스키마 JSON
batch     ─ Stage 2 crop 폴더 통째 처리
```

**핵심 컴포넌트** (~750 lines)

| 영역 | 함수 / 클래스 |
|---|---|
| JSON ↔ Donut 변환 | `json_to_donut`, `donut_to_json` (재귀, round-trip 검증 PASS) |
| Token 등록 | `register_special_tokens` (3 task + 16 schema field + sep) |
| 데이터 | `Sample` dataclass, `discover_samples`, `split_samples` (group-aware), `build_torch_dataset` |
| 학습 | `train(cfg, device, resume, load_in_8bit)` |
| 추론 | `load_inference_model`, `predict_one` |
| CLI | `cmd_train`, `cmd_predict`, `cmd_batch` |

**3개 task token**: `<s_measure>`, `<s_gdt>`, `<s_roughness>`

**Schema field tokens (16개)**: nominal, tolerance, upper, lower, unit, diameter, radius, thread, depth, symbol, datum, modifier, Ra, Rz, Rmax, type

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 베이스 모델 | `naver-clova-ix/donut-base` | D-001 / D-018 / 논문 |
| Fine-tuning | **수행** (Numerical 만, Alphabetical 은 zero-shot) | 논문 §4.3 — schema 정의 가능 |
| Epochs | **30** | 논문 §4.3 |
| Batch size | **4** | 논문 + 16GB VRAM 한계 |
| Optimizer | **AdamW**, lr=1e-6, cosine decay, no warm-up | 논문 그대로 |
| 입력 해상도 | [960, 1280] | configs/donut_numerical.yaml |
| 최대 토큰 길이 | 768 | 충분히 긴 JSON 대응 |
| Precision | **BF16** (RTX 5080 지원) > FP16 fallback | Blackwell 권장 |
| **Gradient checkpointing** | **활성** | RTX 5080 16GB 메모리 |
| 8-bit 옵션 | `--load-in-8bit` (bitsandbytes) | 메모리 부족 시 |
| Train/Val/Test split | **70 / 20 / 10** | 논문 §4.3 |
| **Group-aware split** | **`GroupShuffleSplit` 2단계** ★ | D-024 / Roboflow 증강 변형이 train·val 양쪽 섞이지 않게 |
| Group key 추출 | `<id>.split('.rf.')[0].split('__View_')[0]` 등 | Roboflow + Stage 1/2 prefix 모두 제거 |
| Decoder start token | 첫 task token | Donut convention |
| 빈 라벨 처리 | -100 으로 마스킹 | HF 표준 |
| 평가 메트릭 | `eval_loss` (best model 선택) | 학습 안정성 우선 |
| 토큰 라이브러리 | 자체 구현 (`json_to_donut`/`donut_to_json`) | `processor.token2json` 보다 robust |

## 3. 사용법

### CLI

```bash
# Fine-tune
python src/stage3_numerical.py train \
    --cfg configs/donut_numerical.yaml --device 0

# 메모리 부족 시 8-bit 로딩
python src/stage3_numerical.py train --load-in-8bit

# Resume from checkpoint
python src/stage3_numerical.py train --resume checkpoints/donut_numerical/checkpoint-1000

# 단일 패치 추론
python src/stage3_numerical.py predict \
    --image outputs/crops/sample/annotations/Measure/foo.jpg \
    --region-class Measure

# Stage 2 crop 폴더 일괄
python src/stage3_numerical.py batch \
    --input-dir outputs/crops/sample/annotations \
    --out-dir outputs/sample/numerical
```

### 공개 함수

```python
from src.stage3_numerical import (
    load_inference_model, predict_one,
    json_to_donut, donut_to_json,   # 변환기 (학습 데이터 만들 때도 사용)
)

processor, model, device = load_inference_model(Path("checkpoints/donut_numerical/final"))
rec = predict_one(Path("measure.jpg"), region_class="Measure",
                  processor=processor, model=model, device=device)
# rec["parsed"] = {"nominal": 25.0, "tolerance": {...}, "unit": "mm"}
```

## 4. 검증 결과

### 4.1 ★ JSON ↔ Donut Round-Trip (5 케이스 PASS)

```python
# 테스트한 5 케이스 모두 round-trip 성공
✓ Measure (단순)        : {nominal, tolerance{upper,lower}, unit}
✓ GDT (datum 리스트)    : {symbol, tolerance, datum:[A,B]}
✓ Roughness            : {Ra, unit}
✓ Measure-thread       : {thread:"M12x1.75", depth, unit}
✓ GDT-modifier         : {symbol, tolerance, datum:[A,B,C], modifier:"Ⓜ"}
```

검증 코드:
```python
seq = TASK_TOKENS["Measure"] + json_to_donut(d)
parsed = donut_to_json(seq)
assert parsed == {k: v for k, v in d.items() if k != "type"}
```

### 4.2 모델 검증 (V6, 작성 예정)

논문 베이스라인:

| 항목 | 논문 값 | 임계값 |
|---|---|---|
| Measure F1 | 0.923 | ≥ 0.90 critical |
| GDT F1 | 0.965 | ≥ 0.95 critical |
| Roughness F1 | 1.0 | ≥ 0.95 warning |
| Hallucination | 0.067 | < 0.10 critical |
| Numerical accuracy | — | ≥ 0.95 critical |

## 5. 출력 형식

### 5.1 학습 데이터 (사용자 작성, Step 4 자동 시드)

```
data/vlm/numerical/
├── <id>.jpg     # de-rotated patch (Stage 2 crop)
└── <id>.json    # ground truth schema
```

`<id>.json` 예시:

```json
{
  "type": "Measure",
  "nominal": 25.0,
  "tolerance": {"upper": 0.05, "lower": -0.05},
  "unit": "mm"
}
```

### 5.2 추론 출력 (HANDOFF §5.4)

```json
{
  "type": "Measure",
  "source": "outputs/crops/.../Measure/foo.jpg",
  "parsed": {
    "nominal": 25.0,
    "tolerance": {"upper": 0.05, "lower": -0.05},
    "unit": "mm"
  },
  "raw_seq": "<s_measure><s_nominal>25.0</s_nominal>..."
}
```

### 5.3 학습 산출물

```
checkpoints/donut_numerical/
├── checkpoint-100/   ← epoch 단위 자동 저장
├── checkpoint-200/
├── ...
├── final/            ← best model + processor (load_inference_model 가 사용)
└── logs/             ← TensorBoard
```

## 6. 의존성

```
torch (CUDA 12.8 / BF16 지원 권장)
transformers>=4.44.0
accelerate>=0.34.0
sentencepiece>=0.2.0
timm>=1.0.0
datasets>=2.20.0
scikit-learn>=1.5.0   # GroupShuffleSplit
pyyaml>=6.0
```

선택 (메모리 절감):
```
bitsandbytes        # 8-bit 로딩
flash-attn          # attention 가속
```

## 7. 관련 의사결정

- **D-001** 아키텍처 = Donut paper-faithful
- **D-005** 학습 하이퍼파라미터 (epoch 30 / AdamW / cosine 1e-6 / batch 4 / FP16)
- **D-013** 4개 언어 (Numerical 은 GDT 심볼·숫자 위주라 언어 영향 적음)
- **D-018** Donut 유지 — Step 7 평가 후 미달 시 재논의
- **D-022** Provenance 필수 (raw_seq 포함)
- **D-023** 사용자 필수 임계값 (Measure F1 ≥ 0.90 / Hallucination < 0.10)
- **D-024** Group-aware train/val/test split (Roboflow 증강 변형 누수 방지)

## 8. 검증 모듈

[`check_stage3n_numerical.py`](./../README.md) — V6, 작성 예정

## 9. 업스트림 / 다운스트림

**입력 (업스트림)**
- Stage 2 의 `crop_obb_regions()` 출력: `outputs/crops/<id>/annotations/{Measure,GDT,Roughness}/*.jpg`
- 학습 데이터 시드: Step 4 (`prepare_vlm_dataset.py`) 자동 생성 → 사람이 JSON 검수

**출력 (다운스트림)**
- `pipeline.py` (Step 7) 통합 JSON
- `utils/metrics.py` (Step 8) F1 측정

## 10. 학습 진행 모니터링

```bash
# TensorBoard
tensorboard --logdir checkpoints/donut_numerical/logs

# GPU 메모리
watch -n 1 nvidia-smi
```

학습 시간 예상 (RTX 5080 16GB):
- 13,000 패치 × 30 epoch / batch 4 → 약 8~12시간
