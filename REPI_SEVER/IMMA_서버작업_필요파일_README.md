# IMMA 서버 작업 시 필요한 파일 및 역할 README

작성 목적: IMMA 서버를 수정하거나 배포할 때 어떤 파일을 확인해야 하는지, 각 파일이 어떤 역할을 하는지 빠르게 파악하기 위한 정리 문서입니다.

---

## 1. 서버 작업의 기본 기준

IMMA 서버는 FastAPI 기반으로 구성되어 있고, Railway에 배포된 PostgreSQL과 연결되어 동작합니다.

서버 작업을 할 때는 보통 아래 순서로 확인합니다.

```text
main.py
→ routers/
→ pipeline/
→ lookup_tables/
→ 환경변수
→ Swagger /docs
→ Railway 배포 로그
```

즉, 단순히 `main.py` 하나만 보는 것이 아니라, 라우터, DB 연결, 매칭 파이프라인, seed 데이터, 환경변수를 함께 확인해야 합니다.

---

## 2. 최상위 서버 파일

### main.py

서버의 진입점입니다.

주요 역할:

```text
FastAPI 앱 생성
CORS 설정
정적 UI 파일 연결
각 router include
기본 health check endpoint 제공
Swagger /docs 자동 생성
```

확인해야 하는 부분:

```text
app = FastAPI(...)
CORS 설정
include_router(...)
StaticFiles mount
GET /
GET /db-test
```

주의할 점:

```text
main.py는 전체 서버의 입구이므로, 잘못 수정하면 모든 API가 같이 깨질 수 있습니다.
수정 전에는 반드시 백업을 남기는 것이 좋습니다.
```

---

## 3. 라우터 폴더

### routers/

각 기능별 API가 분리되어 있는 폴더입니다.

대표 파일:

```text
routers/auth.py
routers/signup.py
routers/companies.py
routers/rfqs.py
routers/drawings.py
routers/matching.py
routers/orders.py
routers/quotes.py
routers/reviews.py
routers/notifications.py
routers/admin.py
routers/catalog.py
routers/legacy.py
routers/deps.py
```

---

## 4. routers/deps.py

DB 연결과 공통 의존성을 관리하는 파일입니다.

주요 역할:

```text
DATABASE_URL 읽기
SQLAlchemy engine 생성
SessionLocal 생성
SCHEMA 설정
비밀번호 해시 함수
공통 DB 연결 함수
```

서버가 DB에 연결되지 않을 때 가장 먼저 확인할 파일입니다.

확인할 것:

```text
DATABASE_URL 환경변수 존재 여부
SCHEMA 값
engine 생성 코드
pool_pre_ping=True 설정 여부
```

---

## 5. 회원가입/로그인 관련 파일

### routers/signup.py

회원가입 API를 담당합니다.

주요 역할:

```text
buyer 회원가입
supplier 회원가입
login_id 저장
password hash 저장
companies/users 테이블 연동
```

확인할 것:

```text
POST /signup
login_id 필드
role 값: buyer / supplier
DB 테이블 컬럼과 코드 필드가 일치하는지
```

주의:

```text
schema.sql에는 login_id가 있어도 Railway DB에 실제 컬럼이 없으면 오류가 날 수 있습니다.
이 경우 코드 문제가 아니라 DB 마이그레이션 문제일 수 있습니다.
```

### routers/auth.py

로그인과 인증 토큰 관련 API를 담당합니다.

주요 역할:

```text
로그인
JWT 발급
현재 사용자 확인
권한 검증
```

확인할 것:

```text
JWT_SECRET
ACCESS_TOKEN_EXPIRE_MINUTES
password 검증 로직
token payload의 sub 값
```

---

## 6. 업체/회사 관련 파일

### routers/companies.py

클라이언트와 가공업체 회사 정보를 조회하거나 관리하는 API입니다.

주요 역할:

```text
buyer 회사 조회
supplier 회사 조회
업체 상세 조회
업체 capability 조회
```

확인할 것:

```text
GET /companies
GET /companies/buyers
GET /companies/suppliers
GET /companies/{id}
```

---

## 7. RFQ / 견적 요청 관련 파일

### routers/rfqs.py

클라이언트가 견적 요청을 등록하고 조회하는 API입니다.

주요 역할:

```text
RFQ 생성
RFQ 목록 조회
RFQ 단건 조회
RFQ 상태 변경
```

확인할 것:

```text
POST /rfq
GET /rfqs
GET /rfq/{id}
status 값: open, matched, quoted, ordered 등
```

### routers/drawings.py

도면 업로드를 담당합니다.

주요 역할:

```text
도면 파일 업로드
uploads/ 폴더 저장
SHA-256 해시 생성
중복 업로드 확인
drawings 테이블 저장
```

확인할 것:

```text
POST /api/drawings/upload
multipart/form-data
uploads/ 경로
original_filename
file_uri
sha256
```

주의:

```text
같은 파일을 반복 업로드하면 SHA-256 중복으로 오류가 날 수 있습니다.
시연 전에는 새로운 파일명 또는 새로운 도면을 준비하는 것이 안전합니다.
```

---

## 8. 매칭/RAG 관련 파일

### routers/matching.py

업체 매칭 API를 담당합니다.

주요 역할:

```text
VLM 결과 JSON 입력
match input 변환
RAG/DB 매칭 파이프라인 호출
추천 업체 결과 반환
```

확인할 API:

```text
POST /api/match-v2
GET /match/{rfq_id}
```

### pipeline/

RAG/DB 매칭 파이프라인 코드가 들어 있는 폴더입니다.

대표 파일:

```text
pipeline/config.py
pipeline/db.py
pipeline/parse.py
pipeline/resolve.py
pipeline/match.py
pipeline/response.py
pipeline/pipeline_runner.py
```

각 파일 역할:

```text
config.py: schema 이름, lookup table 경로, 공통 설정
db.py: DB 연결 helper, lookup 데이터 조회
parse.py: VLM JSON 파싱, parts/material/process 추출
resolve.py: 재질 alias 해소, 표준명 변환
match.py: 업체 후보 계산, 재질/공정/장비 조건 비교
response.py: API 응답 형태 구성
pipeline_runner.py: 전체 파이프라인 실행 흐름
```

---

## 9. lookup_tables 폴더

### lookup_tables/

매칭에 필요한 기준 데이터와 DB schema가 들어 있습니다.

대표 파일:

```text
lookup_tables/schema.sql
lookup_tables/lookup_data.json
lookup_tables/equipment_catalog.json
```

### schema.sql

DB 테이블 구조 정의 파일입니다.

확인할 것:

```text
imma schema
users
companies
rfqs
drawings
vlm_rag_results
material_capabilities
process_capabilities
orders
quotes
reviews
```

주의:

```text
schema.sql에 컬럼이 있다고 해서 Railway DB에 반드시 적용된 것은 아닙니다.
실제 DB에 컬럼이 없으면 ALTER TABLE 또는 재마이그레이션이 필요합니다.
```

### lookup_data.json

재질, 공정, alias, 기준 데이터를 담는 파일입니다.

주요 역할:

```text
재질명 표준화
공정명 표준화
도면 분석 결과를 매칭 가능한 값으로 변환
```

### equipment_catalog.json

장비 카탈로그 데이터입니다.

주요 역할:

```text
업체 장비 보유 여부 확인
장비 기반 매칭 점수 계산
equipment_verified 판단
```

---

## 10. 주문/견적/리뷰 관련 파일

### routers/quotes.py

견적 생성과 조회를 담당합니다.

```text
업체 견적 제출
견적 목록 조회
견적 비교
```

### routers/orders.py

발주와 생산 진행 상태를 담당합니다.

```text
업체 선택 후 발주 생성
발주 상태 변경
생산 진행도 공유
납품 단계 전환
```

### routers/reviews.py

리뷰 등록과 조회를 담당합니다.

```text
클라이언트 리뷰 등록
업체 리뷰 확인
관리자 리뷰 관리
```

### routers/notifications.py

알림 흐름을 담당합니다.

```text
견적 요청 알림
업체 회신 알림
발주 상태 알림
생산/납품 알림
```

---

## 11. 관리자 관련 파일

### routers/admin.py

관리자 운영 화면에 필요한 데이터를 제공합니다.

주요 역할:

```text
AI 분석 로그 조회
매칭 관리
계약/결제 상태 확인
납품 정산 로그 확인
리뷰/평점 관리
운영 상태 확인
```

시연에서 관리자 화면은 직접 거래를 처리한다기보다, 전체 흐름이 정상적으로 쌓이는지 확인하는 관제 화면입니다.

---

## 12. 정적 UI 파일

### machhub_ui/ 또는 UI 폴더

FastAPI에서 정적 파일로 제공되는 프론트엔드입니다.

대표 파일:

```text
landing.html
client-register.html
client-dashboard.html
quote-request.html
matching.html
order-management.html
client-fulfillment.html
supplier-register.html
supplier-dashboard.html
supplier-workbench.html
admin-control-center.html
site-actions.js
admin-menu.js
imma-common.css
role-workflows.css
```

주의:

```text
site-actions.js가 전역 클릭/submit 이벤트를 가로채는 경우 실제 API 흐름과 충돌할 수 있습니다.
실제 API 연결용 코드와 시연용 demo script를 구분하는 것이 안전합니다.
```

---

## 13. 환경변수

Railway 서버에서 반드시 확인해야 하는 값입니다.

```text
DATABASE_URL
JWT_SECRET
REPLICATE_API_TOKEN
REPLICATE_MODEL_VERSION
V_E_DEMO_MODE
V_E_CLOUD_PROVIDER
SCHEMA
```

특히 중요:

```text
DATABASE_URL 없으면 DB 연결 실패
JWT_SECRET 없으면 로그인/토큰 오류
REPLICATE_API_TOKEN 없으면 VLM 클라우드 호출 실패
REPLICATE_MODEL_VERSION 틀리면 prediction 생성 실패
```

---

## 14. 시연 전에 확인할 것

```text
1. Railway 서버 접속 확인
2. GET / 정상 응답 확인
3. GET /db-test 정상 응답 확인
4. /docs Swagger 접속 확인
5. POST /signup 테스트
6. 도면 업로드 테스트
7. POST /api/match-v2 테스트
8. matching.html 화면 이동 확인
9. supplier-workbench 화면 확인
10. admin-control-center 화면 확인
```

---

## 15. 자주 터질 수 있는 문제

### DB 컬럼 불일치

증상:

```text
column login_id does not exist
column password does not exist
```

원인:

```text
schema.sql과 실제 Railway DB가 다름
```

대응:

```text
DB에서 직접 컬럼 확인
ALTER TABLE로 컬럼 추가
필요하면 새 DB 생성 후 schema.sql 재적용
```

### 도면 중복 업로드

증상:

```text
duplicate file
sha256 conflict
500 Internal Server Error
```

대응:

```text
새 파일명 또는 다른 도면 사용
중복 처리 로직 개선
```

### Replicate cold start

증상:

```text
AI 분석이 오래 걸림
prediction 대기
timeout 발생
```

대응:

```text
시연 전 warm-up call 실행
실시간 호출 대신 fixture JSON 준비
```

### UI와 API 상태 불일치

증상:

```text
화면에는 100개인데 API에는 30개
클라이언트에서 보낸 요청이 업체 화면에 바로 안 보임
```

원인:

```text
demo data와 실제 API data가 섞임
```

대응:

```text
시연용 fixture 값 통일
수량/재질/납기/업체명 고정
```

---

## 16. 서버 작업 추천 순서

```text
1. main.py 확인
2. routers/deps.py에서 DB 연결 확인
3. /db-test 확인
4. schema.sql과 실제 DB 컬럼 비교
5. routers/signup.py와 auth.py 확인
6. drawings.py 업로드 흐름 확인
7. matching.py와 pipeline/ 확인
8. Swagger에서 API 단위 테스트
9. UI 버튼과 실제 API 연결 확인
10. Railway 로그 확인
```

---

## 17. 서버 작업자가 팀원에게 받아야 할 파일

### VLM 팀원에게 받아야 할 것

```text
실제 VLM 결과 JSON 샘플
입력 도면 이미지/PDF
cog.yaml
predict.py
requirements 또는 python_packages 목록
모델 version 정보
Replicate prediction 성공/실패 로그
```

### RAG/DB 팀원에게 받아야 할 것

```text
schema.sql
lookup_data.json
equipment_catalog.json
pipeline/ 전체 폴더
sample_vlm_result.json
sample_match_v2_response.json
매칭 API 계약서
DB seed script
```

### UI/시연 담당이 준비해야 할 것

```text
최종 UI 폴더
시연 버튼 클릭 순서
고정 입력값
시연용 계정
시연용 도면
우회 URL 목록
스크린샷 백업
```

---

## 18. 한 줄 요약

IMMA 서버 작업에서 필요한 핵심 파일은 `main.py`, `routers/`, `pipeline/`, `lookup_tables/`, 정적 UI 폴더, Railway 환경변수입니다.

서버 작업자는 이 파일들을 기준으로 API, DB, 매칭 파이프라인, UI 흐름이 서로 맞게 연결되어 있는지 확인해야 합니다.
