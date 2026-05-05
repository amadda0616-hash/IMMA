IMMA 서버 연동 증거자료

목적:
VLM Result JSON을 /api/match-v2에 입력했을 때,
서버가 Match Input을 생성하고 업체 후보를 반환하는 흐름이 정상 동작함을 기록한다.

검증 일자:
2026-05-05

검증 서버:
https://fas-production-c5f2.up.railway.app/docs

검증 API:
POST /api/match-v2

입력:
sample_vlm_result.json

출력:
sample_match_v2_response.json 형태

스크린샷 설명:
- 스크린샷(1505).png: /api/match-v2 Request Body 입력 화면
- 스크린샷(1506).png: 200 OK 응답 및 match_input 생성 확인
- 스크린샷(1508).png: candidates 업체 후보 반환 확인

확인된 항목:
- VLM Result JSON 직접 입력 가능
- 서버 내부 Match Input 생성 가능
- PostgreSQL/RAG DB 기반 업체 후보 반환 가능
- company_name, match_reasons, overall_status, next_available_date, equipment_verified 반환 확인

현재 한계:
- 실제 VLM 결과가 아닌 sample/mock 기준 검증
- 실제 VLM 결과 수신 후 필드명 및 누락값 처리 조정 필요