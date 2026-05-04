# `src/validate/common.py`

> **V0** — 검증 프레임워크 공통 인프라 (CheckResult / ValidationReport / HTML+JSON 렌더)

## 1. 구현 요약

모든 `check_*.py` 검증기가 공유하는 **핵심 데이터 구조 + 출력 렌더링** 인프라.

**핵심 컴포넌트**

| 클래스 / 함수 | 역할 |
|---|---|
| `Severity` enum | `critical` / `warning` / `info` (D-021) |
| `Status` enum | `PASS` / `FAIL` / `WARN` / `INFO` / `ERROR` |
| `CheckResult` dataclass | 단일 검증 항목 (이름·값·임계값·방향·severity·상태) |
| `CheckResult.evaluate(...)` | 임계값 자동 비교 → status 산출 |
| `ValidationReport` | 다중 CheckResult 집계 + plot/table 첨부 |
| `report.add_eval(...)` | 검증 항목 추가 |
| `report.add_plot(title, png_bytes)` | base64 PNG 임베디드 |
| `report.add_table(title, rows, cols)` | 표 첨부 |
| `report.emit(...)` | 콘솔 + JSON + HTML 동시 출력 |
| `load_thresholds(yaml)` | YAML 임계값 로더 |
| `threshold_lookup(cfg, dotted)` | 중첩 임계값 조회 (`stage2_model.missing_rate_max.Measure`) |
| `make_bar_chart()` | matplotlib 막대 차트 |
| `make_confusion_matrix()` | matplotlib confusion matrix |
| `setup_logging()` | 통일된 로깅 포맷 |

**Jinja2 인라인 HTML 템플릿** — 외부 파일 의존 없음, fallback 미니 HTML 도 지원.

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| Severity 분류 | `critical` 차단 / `warning` 경고 / `info` 모니터 | D-021 |
| 출력 형식 | **콘솔 + JSON + HTML 3종 동시** | D-022 |
| HTML 라이브러리 | **Jinja2 인라인** (외부 파일 없음) | 단일 파일 portability |
| HTML 폴백 | jinja2 미설치 시 raw HTML 생성 | graceful degradation |
| Plot 임베딩 | base64 PNG → data URI | 단일 HTML 파일로 휴대 |
| matplotlib backend | `Agg` (headless) | 서버/CI 환경 호환 |
| JSON 출력 | UTF-8 + `ensure_ascii=False` | 4개 언어 출력 |
| HTML CSS | 인라인 `<style>` | 외부 의존 없음 |
| 색상 코드 | ANSI in console / CSS pill in HTML | 일관된 시각 표시 |
| Status 자동 판정 | direction 별 (`ge` / `le` / `eq` / `none`) | 룰 명시화 |
| 시계열 추적 | JSON 파일은 stable schema | 외부 모니터링 도구 연동 |
| 유니코드 | 한글/일본어/러시아어 모두 처리 | 다국어 데이터셋 |

## 3. 사용법 (다른 check_*.py 에서 import)

```python
from src.validate.common import (
    DEFAULT_THRESHOLDS_PATH, DEFAULT_REPORTS_DIR,
    Severity, Status, CheckResult, ValidationReport,
    load_thresholds, threshold_lookup,
    make_bar_chart, make_confusion_matrix,
    setup_logging,
)

log = setup_logging("my_check")

def run(...):
    thr = load_thresholds(DEFAULT_THRESHOLDS_PATH)
    report = ValidationReport(
        title="My check title",
        step="my_step",
        metadata={"...": "..."},
    )

    # 검증 항목 추가
    report.add_eval(
        "my_metric", value=0.92, threshold=0.85, direction="ge",
        severity=Severity.CRITICAL,
        message="optional context",
    )

    # 차트 첨부
    report.add_plot(
        "My chart",
        make_bar_chart(labels=["A","B"], values=[10, 20], title="..."),
    )

    return report

def main():
    args = parse_args()
    report = run(...)
    report.emit(reports_dir=DEFAULT_REPORTS_DIR)  # 콘솔 + JSON + HTML
    return 0 if report.overall_status in (Status.PASS, Status.WARN) else 1
```

## 4. 검증 결과

### 4.1 더미 리포트 출력 검증

`check_step1_5_sorter.py` 8건 더미로 실행 시:
- 콘솔: PASS / WARN / FAIL 컬러 출력 정상
- JSON: 3,942 bytes, schema 일관됨
- **HTML: 90,725 bytes, plot 3건 base64 임베디드 정상**

```
checks: 8
overall: WARN
counts: {PASS: 3, WARN: 2, FAIL: 0, INFO: 3, ERROR: 0}
artifacts: 3 plots + 2 tables
```

### 4.2 status 자동 판정 단위 테스트

| direction | value | threshold | 결과 |
|---|---|---|---|
| `ge` | 0.95 | 0.85 | PASS |
| `ge` | 0.80 | 0.85 | FAIL/WARN (severity에 따라) |
| `le` | 0.05 | 0.10 | PASS |
| `le` | 0.15 | 0.10 | FAIL/WARN |
| `none` | any | any | INFO |

## 5. 출력 형식

### 5.1 콘솔 (ANSI 컬러)

```
╔════════════════════════════════════════════════════════════════╗
║  Step 1.5 — TitleBlock Sorter Validation                       ║
╚════════════════════════════════════════════════════════════════╝

 [ 1/8] manual_review_rate         0.2500   ! WARN  ≤ 0.2000
 [ 3/8] classifier_accuracy        1.0000   ✓ PASS  ≥ 0.8500

 Overall: WARN   PASS=3  WARN=2  FAIL=0  INFO=3  ERROR=0
```

### 5.2 JSON (시계열 추적용)

```json
{
  "title": "...",
  "step": "step1.5_sorter",
  "timestamp": "2026-04-27T14:32:11",
  "overall_status": "WARN",
  "counts": {"PASS": 3, "WARN": 2, "FAIL": 0, "INFO": 3, "ERROR": 0},
  "checks": [
    {
      "name": "manual_review_rate",
      "value": 0.25,
      "threshold": 0.20,
      "direction": "le",
      "severity": "warning",
      "status": "WARN",
      "message": "...",
      "details": {}
    }
  ],
  "artifacts": [
    {"kind": "plot", "title": "..."},
    {"kind": "table", "title": "...", "columns": [...], "rows": [...]}
  ],
  "metadata": {...}
}
```

### 5.3 HTML (시각 검수용)

- 헤더 + 메타 정보
- Pill-style overall status (색상 구분)
- 검증 항목 테이블 (severity 별 색상)
- 차트/표 artifacts (base64 PNG embedded)
- Footer (D-020/D-021/D-022 참조)

## 6. 의존성

```
pyyaml>=6.0          # 임계값 YAML 로더
matplotlib>=3.9.0    # 차트
jinja2>=3.1.0        # HTML 템플릿
numpy>=1.26.0        # confusion matrix
```

`jinja2` 미설치 시 fallback HTML 사용 (degraded but functional).

## 7. 관련 의사결정

- **D-020** 검증 의무화 (각 step → check_*.py + reports/ 보관)
- **D-021** Severity 분류 (critical 차단 / warning 경고 / info 모니터)
- **D-022** 콘솔 + JSON + HTML 3종 동시
- **D-023** 사용자 필수 임계값 → `configs/validation_thresholds.yaml`

## 8. 사용 예 (실제 활용 검증기)

| 검증기 | common.py 활용 항목 |
|---|---|
| `check_step1_5_sorter.py` | `make_confusion_matrix`, `make_bar_chart`, language breakdown table |
| `check_labels_yolo.py` | `make_bar_chart` (class distribution), error sample tables |
| `check_stage1_model.py` | `make_confusion_matrix`, per-class accuracy chart |
| `check_labels_obb.py` | `make_bar_chart` (angle histogram, class distribution) |
| `check_stage2_model.py` | `make_confusion_matrix`, missing rate chart, drawing recall histogram |

## 9. 확장 가능성

- 임계값 정책 추가 (e.g. `between` direction, percentile-based)
- HTML 템플릿 외부화 (`templates/report.html.j2`)
- JSON Lines 출력 (대용량 결과)
- 시계열 비교 리포트 (`compare_reports.py`)
