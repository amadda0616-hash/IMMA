# `src/validate/check_step1_5_sorter.py`

> **V1** — `sort_by_titleblock.py` 분류기 정확도 검증

## 1. 구현 요약

`outputs/sort_titleblock_manifest.csv` 와 (선택) 사람 검수 GT 를 비교해 **분류기 정확도, 언어별 분포, confusion matrix** 측정.

**측정 항목**

| 항목 | 의존 |
|---|---|
| `manual_review_rate` | manifest only |
| `error_rate` (imread/OCR 실패) | manifest only |
| 언어별 decision 분포 (filename heuristic) | manifest only |
| keyword_hits 히스토그램 | manifest only |
| `classifier_accuracy` | GT 필요 |
| `precision_TB_present` / `recall_TB_present` / `f1_TB_present` | GT 필요 |
| Confusion matrix plot | GT 필요 |
| `per_language_accuracy[en/ko/ja/ru]` | GT 필요 |

**언어 자동 감지** (filename heuristic, Stage 3 의 정확한 언어 판정 전 임시):
- Hangul → ko
- Hiragana/Katakana/CJK → ja
- Cyrillic → ru
- 그 외 → en

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 입력 | `outputs/sort_titleblock_manifest.csv` (UTF-8-SIG) | sort_by_titleblock 출력 직접 사용 |
| GT 형식 | CSV `filename, has_titleblock_actual` (1/0) | 단순, Excel 작성 가능 |
| GT 필수 여부 | **선택** (descriptive only mode 지원) | 초기 데이터 적재 시 GT 없을 수 있음 |
| `manual_review` 처리 | 정확도 계산에서 **제외** | 사람 검수가 진짜 정답이므로 |
| 언어 감지 | 파일명 기반 heuristic | Stage 3 Donut 결과 전 임시 |
| 임계값 | `configs/validation_thresholds.yaml#step1_5` 동적 로드 | 사용자 조정 가능 |
| Confusion matrix | manual_review 제외 후 binary (TB-present / No-TB) | 명확한 해석 |

## 3. 사용법

```bash
# GT 없이 (descriptive stats only)
python -m src.validate.check_step1_5_sorter \
    --manifest outputs/sort_titleblock_manifest.csv

# GT 와 함께 (정확도 측정)
python -m src.validate.check_step1_5_sorter \
    --manifest outputs/sort_titleblock_manifest.csv \
    --gt data/validation_gt/step1_5_titleblock_gt.csv

# 임계값 커스터마이즈
python -m src.validate.check_step1_5_sorter \
    --thresholds configs/validation_thresholds.yaml \
    --reports-dir reports/

# 컬러 비활성 (CI 환경)
python -m src.validate.check_step1_5_sorter --no-color
```

### GT CSV 형식

```csv
filename,has_titleblock_actual
drawing_001_KO.jpg,1
drawing_002_EN.jpg,1
drawing_003_JP_図面.jpg,1
drawing_005_view_only.jpg,0
drawing_006_KO_view.jpg,0
```

`has_titleblock_actual`: `1` = TB 있음, `0` = TB 없음. 50장 정도 샘플 권장 (D-022).

## 4. 검증 결과

### 4.1 더미 데이터 검증 (8 manifest + 5 GT)

```
[ 1/8] manual_review_rate     0.2500   ! WARN  ≤ 0.2000  (2/8 → 25%)
[ 2/8] error_rate             0.1250   ! WARN  ≤ 0.0100  (1 imread 실패)
[ 3/8] classifier_accuracy    1.0000   ✓ PASS  ≥ 0.8500  (5건 GT 매칭)
[ 4/8] precision_TB_present   1.0000   · INFO  —         (TP=3 FP=0)
[ 5/8] recall_TB_present      1.0000   · INFO  —         (TP=3 FN=0)
[ 6/8] f1_TB_present          1.0000   · INFO  —
[ 7/8] per_language_acc[en]   1.0000   ✓ PASS  ≥ 0.8000  (n=4)
[ 8/8] per_language_acc[ja]   1.0000   ✓ PASS  ≥ 0.8000  (n=1)

Overall: WARN   PASS=3  WARN=2  FAIL=0  INFO=3  ERROR=0
```

### 4.2 산출물 검증

- `reports/2026-04-27_step1.5_sorter.html` — 90,725 bytes, plot 3건 임베디드
- `reports/2026-04-27_step1.5_sorter.json` — 3,942 bytes, schema 일관

## 5. 임계값 (`validation_thresholds.yaml#step1_5`)

```yaml
step1_5:
  classifier_accuracy:    {threshold: 0.85, severity: critical}
  per_language_min:       {threshold: 0.80, severity: warning}
  manual_review_rate_max: {threshold: 0.20, severity: warning}
```

## 6. 출력 형식

### 6.1 콘솔

위 §4.1 참조.

### 6.2 HTML

- Overall pill (PASS/WARN/FAIL 색상)
- 검증 항목 테이블 (severity 별 색상 구분)
- "Decision distribution by language" 표
- "Decision distribution" 막대 차트
- "Confusion matrix (TB-present)" — GT 제공 시
- "Per-language accuracy" 표
- "Keyword hits distribution" 히스토그램

### 6.3 JSON

```json
{
  "title": "Step 1.5 — TitleBlock Sorter Validation",
  "step": "step1.5_sorter",
  "overall_status": "WARN",
  "counts": {"PASS": 3, "WARN": 2, "FAIL": 0, ...},
  "checks": [...],
  "artifacts": [...],
  "metadata": {
    "manifest": "outputs/sort_titleblock_manifest.csv",
    "gt": "data/validation_gt/step1_5_titleblock_gt.csv",
    "n_rows": 8,
    "n_gt": 5
  }
}
```

## 7. 의존성

```
PyYAML
matplotlib  (via common.py)
jinja2      (via common.py)
```

(stdlib 만 사용 — csv / pathlib / collections)

## 8. 관련 의사결정

- **D-019** sort_by_titleblock 은 선택 분석 도구
- **D-020** 검증 의무화
- **D-021** severity 분류
- **D-022** 콘솔 + HTML + JSON 3종 출력

## 9. 검증 대상 모듈

[`sort_by_titleblock.md`](./sort_by_titleblock.md) — Step 1.5

## 10. 한계

- 언어 자동 감지가 파일명 기반 heuristic (이미지 OCR 결과 기반이 더 정확하지만 시간/리소스 큼)
- TB 부분 존재(영역 일부 잘림) 케이스 인식 안 함 — 사람 GT 가 binary 0/1
- 4개 언어 외 (e.g. 중국어) 도면이 섞이면 "en" 으로 분류됨
