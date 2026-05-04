# Sample Enriched JSON Files (Step 9 Output Mock)

본 폴더는 **Step 9 (Metadata Enrichment)** 의 출력 형태를 보여주는 **가상 메타데이터** 10건입니다.
실제 도면 없이도 enriched JSON 스키마를 검토·테스트할 수 있도록 만든 fixture/sample 입니다.

> 출처는 인터넷에 공개된 일반 산업 표준 + 한·일·미·러 시장의 통상적 사용 빈도 기준으로 작성된 가상 데이터.
> 실제 회사 사양과 다를 수 있으며, 제품 설계 직접 인용은 금지.

## 10개 샘플 목록

| # | 가공 조합 | 주재질 | 도면 언어 | 시장/특징 |
|---|---|---|---|---|
| 01 | 레이저 절단 + 판금 (벤딩) | **SUS304 No.2D 1.5t** | KO | 제어함, 일반 외장 판금 (한국 표준 KS) |
| 02 | 워터젯 절단 + CNC 머시닝(드릴/탭/카운터보어) + 용접 | **SUS316L 6t** | EN | 펌프 브래킷, 내식성 중요 |
| 03 | CNC 머시닝 + 아노다이징 | **AL6061-T6** | EN | 항공/방산 마운트 브래킷, 정밀 |
| 04 | CNC 선삭 + 열처리 + 연삭 | **S45C HRC58-62** | JA | 정밀 샤프트, 베어링 시트 |
| 05 | 판금 + 프레스 성형 + 분체도장 | **SECC 1.0t (SGCC)** | KO | 서버 랙 패널, 양산 외장 |
| 06 | 스탬핑 + 스폿 용접 | **SAPH440** | JA | 자동차 구조 부품 (JIS) |
| 07 | 주조 + CNC 머시닝 + 열처리 | **FCD450 (구상흑연주철)** | EN | 펌프/밸브 하우징 |
| 08 | 적층제조(DMLS) + CNC 후가공 | **Ti6Al4V (Grade 5)** | EN | 항공 터빈 부품 |
| 09 | 플라스마 절단 + 용접 + 도장 | **SM490A** | RU | 교량/중구조 GOST 표준 |
| 10 | 와이어 EDM + 표면 연삭 | **SKD11 HRC58-62** | JA | 사출 금형 코어 |

## 이 샘플들이 보여주는 것

각 JSON은 PROJECT_HANDOFF.md §5 (스키마) + Step 9 enrichment 스키마를 따르며,
다음 4-tier escalation 결과를 다양하게 시연합니다.

| Method | 의미 | 예시 |
|---|---|---|
| `deterministic` | KB/표준 직접 매칭 | "ISO 2768-mK" 일반 공차 |
| `heuristic` | 도메인 룰 적용 | Ø 심볼 → diameter feature |
| `llm` | RAG-augmented LLM 추론 | "stainless" → "SUS304 No.2D" |
| `hitl` | 신뢰도 < 0.70 → 사람 검수 필요 | 모호한 재질 grade |

## 제공 필드

각 파일은 다음 섹션을 모두 포함:

- `drawing_id`, `image_size`, `language_hint`
- `title_block` (도번, 제목, 재질, 척도, 공차, 개정, 작성자 등)
- `notes` (도면 노트 항목)
- `views[]` (각 view 의 annotation 리스트, OBB 글로벌 좌표 포함)
- `enrichment` (Step 9 결과: gap detection → 4-tier resolution → provenance)
- `meta` (모델 버전, 타임스탬프)

## 활용

- Step 9 스크립트 단위테스트 fixture
- 다운스트림 (CAD/ERP 연동) 모듈의 입력 mock
- 사용자 검수 UI 프로토타이핑
- 평가 지표 (precision/recall/F1) 베이스라인 데이터

## 면책

본 데이터는 **합성**이며, 특정 회사·도면을 모사하지 않음. 표준 등급명(SUS304 등)은 공개된 KS/JIS/ASTM/GOST 표준명으로, 실제 시장에서 통용되는 명칭임.
