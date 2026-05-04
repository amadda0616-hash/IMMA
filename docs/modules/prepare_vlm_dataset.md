# `src/prepare_vlm_dataset.py`

> **Step 4** — Stage 3 (VLM) 학습/평가용 image-text pair 자동 시드 생성

## 1. 구현 요약

학습된 Stage 1 + Stage 2 모델로 자동 추론·crop·warp 한 뒤, 사람 검수용 **빈 JSON 템플릿** 을 자동 생성하는 브리지 모듈.

**3개 CLI 서브커맨드**

| 모드 | 입력 | 출력 | 의존 |
|---|---|---|---|
| `alphabetical` | dataset/ + Stage 1 모델 | `data/vlm/alphabetical/` (TitleBlock + Notes) | Stage 1 학습 후 |
| `numerical` | dataset/ + Stage 1 + Stage 2 모델 | `data/vlm/numerical/` (Measure + GDT + Roughness) | **★ Step 6 학습 데이터** |
| `all` | 위 둘 동시 | 위 둘 모두 | Stage 1+2 학습 후 |

**핵심 기능**

| 영역 | 함수 |
|---|---|
| Group key 추출 (D-024) | `extract_group_key(filename)` — Roboflow + Stage 1/2 prefix 처리 |
| OCR pre-fill (옵션) | `ocr_image()` + `extract_numeric_hint()` — Pytesseract 4개 언어 |
| 템플릿 빌더 | `build_numerical_template()`, `build_alphabetical_template()` |
| 도면 처리 | `process_drawing_alphabetical()`, `process_drawing_numerical()` |
| 매니페스트 | `write_manifest()` — UTF-8-SIG CSV |

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 출력 = **빈 템플릿** | 자동 추출이 아닌 사람 검수 시드 | Donut fine-tune 정확도 = GT 품질에 직결 |
| Schema 구조 | HANDOFF §5.3 / §5.4 그대로 | 다운스트림 호환 |
| `_review` 메타 블록 | `source_image`, `completed`, `ocr_hint` 등 | 검수 진행 추적 / 학습 시 제거 |
| OCR pre-fill | **옵션** (`--ocr-prefill`) | 검수 시간 단축, but ground truth 오염 방지 위해 별도 필드 |
| 숫자 hint 추출 | Measure 만 `ocr_numeric` 추가 | nominal 채워두지 않음 (검수 후 수동 입력) |
| **Group key 통합 추출** | Roboflow `.rf.` 와 Stage 1/2 `__View_/__Measure_` prefix 모두 처리 | D-024 group-aware split 보장 |
| 파일명 규약 | `<drawing>__View_<id>__<class>_<idx>.jpg` (numerical) | Step 6 에서 group_key 역산 가능 |
| 단계별 모드 분리 | Stage 1 학습 직후 alphabetical 만 가능 | 라벨링 병렬 진행 지원 |
| imread 처리 | `np.fromfile + cv2.imdecode` | 4개 언어 파일명 안전 |
| 매니페스트 | UTF-8-SIG CSV | Excel 다국어 호환 |
| `--limit N` | 디버깅/테스트용 처음 N장만 처리 | 5,839장 전체 실행 전 검증 가능 |
| Cleanup | 임시 subdir 자동 정리 | 산출물 평탄화 |

## 3. 사용법

### CLI

```bash
# 1) Stage 1 학습 직후 — TitleBlock/Notes 만
python src/prepare_vlm_dataset.py alphabetical \
    --dataset dataset/ \
    --det-weights checkpoints/yolo_det.pt \
    --ocr-prefill --device 0

# 2) ★ Stage 2 학습 후 — Numerical (Step 6 학습 데이터)
python src/prepare_vlm_dataset.py numerical \
    --dataset dataset/ \
    --det-weights checkpoints/yolo_det.pt \
    --obb-weights checkpoints/yolo_obb.pt \
    --ocr-prefill --device 0

# 3) 모두 (Stage 1+2 학습 후)
python src/prepare_vlm_dataset.py all \
    --det-weights checkpoints/yolo_det.pt \
    --obb-weights checkpoints/yolo_obb.pt \
    --ocr-prefill

# 4) 디버깅: 처음 10장만
python src/prepare_vlm_dataset.py numerical --limit 10
```

### 사람 검수 흐름

생성된 `<id>.json` 의 `null` 필드를 채움:

```json
// 자동 생성된 템플릿
{
  "type": "Measure",
  "nominal": null,           ← 사람이 채움
  "tolerance": null,         ← 옵션
  "unit": "mm",
  "_review": {
    "source_image": "...",
    "completed": false,      ← 검수 완료 시 true 로 변경
    "ocr_hint": "12.5",      ← 참고용 (자동)
    "ocr_numeric": 12.5      ← 참고용 (자동)
  }
}
```

검수 완료된 JSON 만 Step 6 학습이 사용 (manifest `status` 컬럼으로 추적).

## 4. 검증 결과

### 4.1 Group key 추출 단위 테스트 (8 케이스 PASS)

```
✓ 11_jpeg.rf.8b46c563d114.jpg       → '11_jpeg'
✓ drawing.rf.deadbeef.jpg           → 'drawing'
✓ plain_drawing.jpg                 → 'plain_drawing'
✓ foo__View_00.jpg                  → 'foo'
✓ foo__View_00__Measure_03.jpg      → 'foo'
✓ foo.rf.aaa__View_01__GDT_02.jpg   → 'foo'
✓ 11_jpeg.rf.bbb__View_03__Roughness_00.jpg → '11_jpeg'
✓ complex_name_v2.rf.ccc__TitleBlock_00.jpg → 'complex_name_v2'
```

Roboflow `.rf.<hash>` 와 Stage 1/2 `__<class>_<idx>` 양쪽 prefix 가 같이 있어도 정확히 원본 도면 stem 으로 환원.

### 4.2 템플릿 빌더 검증 (5종)

5개 템플릿 (Measure / GDT / Roughness / TitleBlock / Notes) 모두 정상 생성. Schema 가 HANDOFF §5.3 / §5.4 와 일치.

### 4.3 숫자 hint 추출 (5 케이스)

```
'12.5'        → 12.5
'Ø25.4 ±0.1'  → 25.4
'no number'   → None
'Ra 1.6 μm'   → 1.6
'M8 thread'   → 8.0
```

### 4.4 CLI 검증

3개 서브커맨드 (`all`/`alphabetical`/`numerical`) 정상 파싱. `--help` 동작 확인.

## 5. 출력 형식

### 5.1 디렉터리 구조

```
data/vlm/numerical/                       ← Step 6 학습용 (★)
├── <drawing>__View_00__Measure_00.jpg
├── <drawing>__View_00__Measure_00.json   ← 사람 검수 필요
├── <drawing>__View_00__GDT_00.jpg
├── <drawing>__View_00__GDT_00.json
├── <drawing>__View_01__Roughness_00.jpg
├── <drawing>__View_01__Roughness_00.json
└── manifest.csv

data/vlm/alphabetical/                    ← Stage 3-A V5 평가용 (옵션)
├── <drawing>__TitleBlock_00.jpg
├── <drawing>__TitleBlock_00.json
├── <drawing>__Notes_00.jpg
├── <drawing>__Notes_00.json
└── manifest.csv
```

### 5.2 매니페스트 CSV

```csv
filename,region_class,parent_drawing,parent_view,group_key,has_ocr_hint,json_path,status
foo__View_00__Measure_00.jpg,Measure,foo.jpg,View_00,foo,true,...,pending_review
foo__TitleBlock_00.jpg,TitleBlock,foo.jpg,-,foo,true,...,pending_review
```

`status` 값:
- `pending_review` — 자동 생성, 사람 검수 대기
- `completed` — 검수 완료 (수동 갱신)
- `skipped` — 검수 시 무시

### 5.3 Numerical 템플릿 예시

```json
{
  "type": "Measure",
  "nominal": null,
  "tolerance": null,
  "unit": "mm",
  "_review": {
    "source_image": "data/vlm/numerical/foo__View_00__Measure_00.jpg",
    "completed": false,
    "ocr_hint": "12.5",
    "ocr_numeric": 12.5,
    "obb": [[0.12, 0.34], [0.45, 0.34], [0.45, 0.40], [0.12, 0.40]]
  }
}
```

### 5.4 Alphabetical 템플릿 예시

```json
{
  "type": "TitleBlock",
  "fields": {
    "drawing_no": null,
    "title": null,
    "material": null,
    "scale": null,
    "revision": null,
    "date": null,
    ...
  },
  "_review": {
    "source_image": "...",
    "completed": false,
    "ocr_text": "DRAWING NO\nDWG-001-A\nMATERIAL\nSS400\nSCALE\n1:2\n..."
  }
}
```

## 6. 의존성

```
ultralytics>=8.3.0        # Stage 1/2 모델 import (lazy)
opencv-python, numpy
pytesseract               # --ocr-prefill 옵션 시
Pillow
```

`--ocr-prefill` 미사용 시 pytesseract 불필요 (graceful skip).

## 7. 관련 의사결정

- **D-001** 아키텍처 = 논문 (스키마는 §5.3/§5.4)
- **D-013** 4개 언어 OCR pre-fill (`kor+eng+rus+jpn`)
- **D-019** sort_by_titleblock 과 무관 — Stage 1 검출 결과 직접 사용
- **D-022** Provenance — `_review` 블록으로 source_image/ocr_hint 추적
- **D-024** Group key 추출 (Roboflow + Stage prefix 양쪽 처리, D-024 코드 예시 그대로)

## 8. 업스트림 / 다운스트림

**업스트림 (이 모듈이 사용)**
- `stage1_layout.predict_one() / crop_regions()`
- `stage2_annotation.predict_one() / crop_obb_regions()`
- (옵션) Pytesseract

**다운스트림 (이 모듈을 사용)**
- **`stage3_numerical.train()`** ← `data/vlm/numerical/` (사람 검수 후) → Step 6 fine-tune
- **`check_stage3a_alphabetical.py`** ← `data/vlm/alphabetical/` (V5 평가용 GT)
- **`check_stage3n_numerical.py`** ← `data/vlm/numerical/` test split (V6 평가)

## 9. 사람 검수 노동량 추정

**Numerical (Step 6 학습 필수)**

- 5,839 도면 × 평균 9 OBB ≈ **40,000+ 패치** 예상
- 1패치당 검수 시간 평균 30초 (OCR pre-fill 사용 시)
- 1인 작업 시 **약 350~500시간** (병목 단계)

OCR pre-fill 효과로 30~50% 시간 단축 기대.

**Alphabetical (Stage 3-A V5 평가, 옵션)**

- 5,839 도면 × 평균 1.2 패치 ≈ 5,500 패치
- 1패치당 검수 시간 ~2분 (TitleBlock 12 필드)
- 약 180~250시간

V5 검증을 위한 sampling (50~100 도면) 만 검수해도 충분 (전수 불필요).

## 10. 사용 시 흔한 이슈

| 증상 | 원인 / 해결 |
|---|---|
| `obb weights not found` | Stage 2 학습 미완료. `alphabetical` 모드만 실행 |
| OCR 결과가 빈 문자열 | Tesseract 언어팩 미설치 (`tesseract-ocr-{kor,jpn,rus}`) |
| 추론 매우 느림 | `--device 0` 명시 / `--imgsz-det 1024` 로 줄임 |
| 메모리 부족 | `--limit 100` 으로 분할 처리 |
| 같은 group_key 가 여러 split 에 누수 | Step 6 `split_samples` 가 GroupShuffleSplit 사용 — 자동 방지 |
| `_review` 블록이 학습에 들어감 | Step 6 `discover_samples()` 가 자동 제거 (`if k == "type": continue` 등) — 검증 후 제거 로직 추가 검토 |

## 11. 진행 흐름 요약

```
사용자 라벨링 → Stage 1 학습 → V2-B PASS
                        ↓
        prepare_vlm_dataset.py alphabetical (선택)
                        ↓
        사람 검수 (TitleBlock/Notes, V5 평가용)

사용자 OBB 라벨링 → Stage 2 학습 → V3-B PASS
                        ↓
        ★ prepare_vlm_dataset.py numerical
                        ↓
        ★ 사람 검수 (~40,000 패치, 가장 큰 노동량)
                        ↓
        Step 6 (stage3_numerical.py) train
                        ↓
        V6 검증 → Step 7 → ...
```
