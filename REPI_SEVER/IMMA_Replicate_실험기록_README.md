# IMMA Replicate 실험 기록 README

작성 목적:  
IMMA 프로젝트에서 Replicate/Cog를 이용해 VLM 도면 인식 모델을 실험하는 동안 발생한 성공, 실패, 오류, 판단, 대안 경로를 하나의 실험 기록으로 정리합니다.

작성 기준:  
이 문서는 프로젝트 중 남아 있는 대화 기록, 실험 산출물, 디버그 패키지, Replicate Prediction 화면, Path B/Path C 논의, VLM 서버 관련 README, QC 메모를 바탕으로 정리한 발표·인수인계용 기록입니다.

주의:  
일부 시간, version hash, prediction id, 정확한 로그 전문은 별도 스크린샷이나 Replicate 웹 콘솔에 남아 있을 수 있습니다. 이 README는 실험 흐름과 의사결정 중심으로 작성되었습니다. API token이나 key는 보안상 직접 적지 않습니다.

---

## 1. 한 줄 요약

IMMA의 Replicate 실험은 단순히 “VLM을 한 번 실행해봤다”가 아니라,  
**Qwen 계열 대형 VLM을 Cog로 패키징해 클라우드 추론을 시도하고, worker setup 실패·타입 오류·LoRA 설정 오류·cold start 지연을 겪은 뒤, vLLM 기반 고성능 경로와 Donut/YOLO 기반 안정성 경로를 나누어 검증한 실험 과정**입니다.

최종적으로 얻은 결론은 다음과 같습니다.

```text
1. cog push 성공과 실제 prediction 성공은 다르다.
2. 대형 VLM은 모델 성능보다 배포 안정성과 cold start가 더 큰 리스크가 될 수 있다.
3. vLLM + torch + Cog/Replicate worker 조합은 버전 충돌 가능성이 크다.
4. 시연에서는 실시간 VLM 호출보다 fixture JSON 또는 미리 성공한 Prediction 결과를 함께 준비해야 한다.
5. Replicate Prediction 로그는 처리 시간, 비용, 성공 여부, cold start 리스크를 기록하는 중요한 실험 이력이다.
```

---

## 2. 실험 배경

IMMA는 클라이언트가 도면을 업로드하면 AI가 도면에서 재질, 공정, 치수, 누락 정보 등을 추출하고, 이 결과를 매칭 API로 넘겨 적합한 가공업체를 추천하는 구조를 목표로 했습니다.

전체 목표 흐름은 다음과 같습니다.

```text
도면 이미지/PDF 입력
→ VLM/OCR 도면 분석
→ 구조화 JSON 생성
→ match input 변환
→ /api/match-v2 호출
→ 업체 추천 결과 반환
→ UI에서 후보 비교 및 견적 요청
```

여기서 Replicate 실험은 위 흐름 중 **도면 이미지/PDF 입력 → VLM/OCR 도면 분석 → 구조화 JSON 생성** 부분을 클라우드 GPU 환경에서 검증하기 위한 작업이었습니다.

---

## 3. 왜 Replicate를 사용했는가

초기 판단은 다음과 같았습니다.

```text
IMMA FastAPI 서버는 Railway에서 CPU 기반으로 운영한다.
대형 VLM은 Railway 서버에 직접 올리기 어렵다.
GPU 서버를 직접 운영하면 비용과 관리 부담이 크다.
Replicate는 사용한 만큼 비용을 내고, idle 비용을 줄일 수 있다.
Cog로 모델을 패키징하면 API 형태로 추론을 호출할 수 있다.
```

따라서 구조는 다음처럼 설계되었습니다.

```text
[IMMA FastAPI 서버]
  - 도면 업로드
  - DB 저장
  - match-v2 호출
  - UI 제공

[Replicate 클라우드]
  - VLM/OCR 모델 실행
  - 도면 분석 JSON 반환
```

이 방식의 장점은 다음과 같았습니다.

```text
GPU 서버를 직접 유지하지 않아도 된다.
시연/실험 단위로만 비용이 발생한다.
외부 API처럼 FastAPI에서 호출할 수 있다.
여러 모델 경로를 비교하기 쉽다.
```

하지만 실제 실험을 통해 다음 단점도 확인했습니다.

```text
cold start가 길 수 있다.
worker setup 실패 시 원인 추적이 어렵다.
모델/패키지 버전 호환성이 매우 중요하다.
대형 모델은 빌드와 실행 시간이 길다.
시연 중 실시간 호출은 불안정할 수 있다.
```

---

## 4. 실험 경로 요약

IMMA의 Replicate 실험은 크게 네 경로로 정리할 수 있습니다.

```text
V.B Demo v1
- Qwen3-VL-30B-A3B-Instruct-FP8
- vLLM 기반
- A100 80GB 권장
- 고성능 VLM 전체 도면 분석 목표

Path C
- 기존 Qwen3/V.B 구조를 유지하되 torch/vLLM/transformers 버전 조정
- worker setup 오류 해결 시도

Path B
- vLLM 제거
- Qwen2-VL 또는 더 작은 transformers 기반 구조 검토
- 배포 안정성 우선

V.E / Path B-3
- YOLO + Donut FT + Pydantic schema 검증 구조
- 도면 전체를 한 번에 읽기보다 영역 검출 + OCR/JSON 추출 방식
- Replicate Playground에서 JSON output 성공 사례 확인
```

---

## 5. 실험 1 — V.B Demo v1: Qwen3-VL-30B + vLLM 경로

### 5.1 목적

대형 Qwen 계열 VLM을 사용해 도면 이미지를 직접 분석하고, 재질, 공정, 치수, title block, notes 등을 구조화 JSON으로 추출할 수 있는지 확인하는 실험입니다.

목표 모델/구성은 대략 다음과 같았습니다.

```text
모델: Qwen3-VL-30B-A3B-Instruct-FP8
추론 엔진: vLLM
GPU: A100 80GB 권장
배포 방식: Cog → Replicate
출력 목표: V.B schema 기반 JSON
```

### 5.2 필요 파일

```text
cog.yaml
predict.py
README.md
requirements 또는 python_packages 목록
sample drawing image
schema definition
postprocess code
```

### 5.3 기대 흐름

```text
cog build
→ cog push r8.im/<owner>/<model>
→ Replicate version 생성
→ Playground에서 input schema 확인
→ 도면 이미지 업로드
→ Run 실행
→ JSON output 생성
```

### 5.4 실제 진행

Replicate 계정, billing, WSL2, Docker, Cog CLI, cog login 등 기본 환경은 준비되었습니다.

실험 과정에서 확인된 성공 항목은 다음과 같습니다.

```text
Replicate 계정 준비
billing/credit 준비
WSL2/Docker 환경 준비
Cog CLI 사용
cog login
private model 생성
A100 80GB 설정
cog push 일부 단계 성공
Playground input schema 표시까지 도달
```

즉, 완전히 시작도 못 한 것이 아니라, **Replicate 모델 페이지와 Playground까지는 도달한 실험**이었습니다.

### 5.5 발생 문제

그러나 실제 Run 실행 시 worker setup 단계에서 실패가 발생했습니다.

대표적인 오류 유형:

```text
worker setup failed
Setup failed
version disabled
vLLM + torch + Cog/Replicate worker setup 호환 문제
```

이 문제는 모델 자체의 추론 정확도 문제가 아니라, **Replicate worker가 모델 실행 환경을 구성하는 단계에서 실패한 문제**였습니다.

### 5.6 이 실험에서 얻은 교훈

```text
cog push 성공만으로는 모델이 실제 실행된다고 볼 수 없다.
Playground schema가 떠도 worker setup에서 실패할 수 있다.
대형 모델은 GPU만 맞으면 되는 것이 아니라, torch/vLLM/transformers/CUDA 조합이 맞아야 한다.
배포 안정성은 모델 성능과 별개의 실험 항목이다.
```

---

## 6. 실험 2 — Cog Input 타입 오류

### 6.1 발생한 오류

실험 중 다음과 같은 오류가 발생했습니다.

```text
infer_schema(func): Parameter input has unsupported type torch.Tensor
```

또는 유사하게:

```text
Parameter input has unsupported type torch.Tensor
```

### 6.2 원인

Cog는 `predict()` 함수의 입력 인자를 API schema로 변환합니다.  
이때 `torch.Tensor` 같은 내부 계산용 타입을 외부 입력 타입으로 쓰면 Cog가 schema를 생성할 수 없습니다.

문제 구조 예시:

```python
def predict(self, input: torch.Tensor):
    ...
```

이 방식은 Cog API input으로 적절하지 않습니다.

### 6.3 수정 방향

Cog input은 지원되는 타입으로 받아야 합니다.

예시:

```python
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def predict(
        self,
        image: Path = Input(description="Input drawing image")
    ):
        # 내부에서 PIL, numpy, torch tensor로 변환
        ...
```

즉, 외부 입력은 `Path`, `str`, `int`, `float`, `bool` 등 Cog가 schema화할 수 있는 타입으로 받고, torch tensor 변환은 내부에서 해야 합니다.

### 6.4 교훈

```text
모델 코드에서 쓰는 타입과 API 입력 타입은 다르다.
Replicate/Cog 배포에서는 predict 함수 signature가 매우 중요하다.
input schema 생성 단계에서 막히면 모델 로딩까지 가지 못한다.
```

---

## 7. 실험 3 — CogPath alias 문제

### 7.1 발생한 문제

실험 중 `CogPath` alias 관련 문제가 발생했습니다.

대표 형태:

```text
unsupported type: CogPath
```

### 7.2 원인 추정

다음과 같이 alias를 쓰는 경우 Cog schema inference에서 타입을 제대로 인식하지 못할 수 있습니다.

```python
from cog import Path as CogPath
```

그리고 predict 함수에서:

```python
def predict(self, image: CogPath):
    ...
```

처럼 사용하면, Cog가 내부적으로 지원 타입으로 판단하지 못할 가능성이 있습니다.

### 7.3 수정 방향

외부 입력은 Cog가 기대하는 형태로 단순하게 사용합니다.

```python
from cog import Path
```

내부 파일 경로 처리가 필요하면 Python 표준 library와 이름을 분리합니다.

```python
from pathlib import Path as LocalPath
from cog import Path
```

### 7.4 교훈

```text
Cog input 타입은 가능하면 문서 예시 그대로 쓰는 것이 안전하다.
alias를 과하게 쓰면 schema inference에서 예상치 못한 오류가 날 수 있다.
```

---

## 8. 실험 4 — LoRA Config 오류

### 8.1 발생한 오류

다음 오류가 발생했습니다.

```text
TypeError: LoraConfig.__init__() got an unexpected keyword argument 'peft_version'
```

### 8.2 원인

LoRA adapter config에 현재 설치된 `peft` 버전이 지원하지 않는 필드가 들어간 것으로 보입니다.

즉, adapter를 만든 환경과 Replicate worker에서 설치된 `peft`/`transformers` 버전이 맞지 않았을 가능성이 큽니다.

### 8.3 대응 방향

```text
peft 버전 확인
transformers 버전 확인
adapter_config.json에서 peft_version 필드 지원 여부 확인
지원하지 않는 키 제거 검토
학습/추론 환경의 peft 버전 맞추기
```

### 8.4 교훈

```text
LoRA adapter는 모델 weight만 있으면 끝나는 것이 아니다.
adapter config와 peft/transformers 버전이 맞아야 한다.
버전 고정이 실험 재현성에 매우 중요하다.
```

---

## 9. 실험 5 — Transformers architecture 미지원 문제

### 9.1 발생한 오류 유형

다음과 같은 문제가 언급되었습니다.

```text
architecture not recognized
model type qwen3_vl_moe not recognized
qwen2_vl out of date
```

### 9.2 원인

Transformers 버전이 해당 모델 architecture를 아직 지원하지 않거나, 모델 config의 `model_type`을 현재 설치된 transformers가 인식하지 못하는 상태입니다.

### 9.3 대응 방향

```text
transformers 버전 업데이트
모델이 요구하는 transformers commit/branch 확인
trust_remote_code 필요 여부 확인
지원되는 Qwen2-VL 계열로 임시 전환
```

### 9.4 교훈

```text
최신 모델일수록 라이브러리 지원 상태가 불안정할 수 있다.
모델 card에서 요구하는 transformers 버전을 반드시 확인해야 한다.
```

---

## 10. 실험 6 — pyairports / airports 관련 누락 모듈

### 10.1 발생한 문제

실험 중 다음과 같은 모듈 누락 문제가 언급되었습니다.

```text
ModuleNotFoundError: No module named 'pyairports'
```

또는 `airports` 패키지 관련 의존성 문제가 있었습니다.

### 10.2 원인

직접 사용하지 않는 것처럼 보여도, 설치된 패키지 중 하나가 내부적으로 추가 dependency를 요구했을 수 있습니다.

### 10.3 대응 방향

```text
requirements에 누락 패키지 추가
불필요한 패키지 제거
dependency tree 점검
실제 추론에 필요 없는 패키지는 제거해 build 단순화
```

### 10.4 교훈

```text
requirements가 많을수록 디버깅 난이도가 올라간다.
모델 추론에 꼭 필요한 패키지만 남기는 것이 Replicate worker 안정성에 유리하다.
```

---

## 11. 실험 7 — Path C: 버전 조정으로 기존 vLLM 구조 살리기

### 11.1 목적

Path C는 기존 Qwen3 + vLLM 구조를 최대한 유지하면서, torch/vLLM/transformers 버전을 조정해 worker setup 실패를 해결하려는 경로였습니다.

### 11.2 실험 방향

```text
vLLM 버전 변경
torch 버전 변경
transformers 버전 고정
qwen-vl-utils 버전 확인
pydantic 버전 확인
필요 system package 추가
```

언급된 후보 조합 예시:

```text
vllm==0.6.0
transformers==4.45.2
torch==2.3.1 또는 torch==2.4.0
pillow==10.4.0
pydantic==2.9.2
qwen-vl-utils==0.0.8
```

### 11.3 장점

```text
성공하면 기존 대형 VLM 구조를 유지할 수 있다.
성능 기대치가 높다.
```

### 11.4 단점

```text
버전 조합 탐색 비용이 크다.
빌드 시간이 길다.
실패할 때마다 Replicate worker setup 시간을 소모한다.
시연 전 안정성을 보장하기 어렵다.
```

### 11.5 판단

시연 일정이 가까운 상황에서는 Path C만 붙잡는 것이 위험했습니다.  
따라서 안정성 우선 경로인 Path B가 함께 검토되었습니다.

---

## 12. 실험 8 — Path B: vLLM 제거 + transformers 기반 전환

### 12.1 목적

Path B는 vLLM 기반 구조를 제거하고, transformers 기반으로 모델을 실행하는 대안입니다.

핵심 판단:

```text
가장 강한 모델보다 실제로 Replicate에서 도는 모델이 중요하다.
시연에서는 속도와 안정성이 정확도보다 중요할 수 있다.
의존성을 줄여 worker setup 성공 가능성을 높인다.
```

### 12.2 주요 변경

```text
vLLM 제거
transformers 기반 추론 사용
Qwen3 30B보다 작은 모델 검토
Qwen2-VL-7B-Instruct 같은 경량 후보 검토
GPU 요구량 완화
패키지 버전 고정
```

### 12.3 기대 효과

```text
Cog setup 성공 가능성 증가
GPU 메모리 요구량 감소
빌드/디버깅 난이도 감소
시연 안정성 증가
```

### 12.4 단점

```text
모델 성능이 낮아질 수 있다.
처리 속도가 vLLM보다 느릴 수 있다.
도면 이해력이 낮아질 수 있다.
```

### 12.5 교훈

```text
AI 모델 실험에서는 최고 성능 모델이 항상 최선이 아니다.
시연/서비스에서는 재현성, 안정성, 비용이 함께 중요하다.
```

---

## 13. 실험 9 — Replicate cold start 측정

### 13.1 관찰

Replicate Prediction 실행 중 cold start로 인해 5~6분대 대기/실행 시간이 관찰되었습니다.

기록된 사례:

```text
5분 46초 수준 cold start
6분 23초 수준 cold start
```

이전에는 60~150초 정도를 예상했지만, 실제로는 더 길게 걸리는 경우가 있었습니다.

### 13.2 문제

시연 중 실시간 분석으로 5~6분이 걸리면 발표 흐름이 끊깁니다.

특히 도면 업로드 후 AI 분석 결과가 나오기까지 관객이 기다려야 하면 시연 안정성이 크게 떨어집니다.

### 13.3 대응 전략

```text
시연 전 warm-up call 실행
Replicate Playground 결과 화면 미리 열어두기
분석 결과 JSON을 fixture로 준비
AI 분석 시작 화면만 보여주고 결과 화면으로 전환
실시간 호출 실패 시 샘플 결과로 진행
```

### 13.4 교훈

```text
실시간 AI 시연은 모델 정확도보다 latency 관리가 중요하다.
cold start는 기능 문제가 아니라 운영 리스크다.
```

---

## 14. 실험 10 — Replicate 비용 확인

### 14.1 관찰

Replicate Prediction 화면에서 실행별 비용이 표시되었습니다.

대략적인 관찰:

```text
건당 약 $0.01 ~ $0.03 수준의 prediction 비용
일부 실행은 모델/GPU/시간에 따라 달라짐
```

### 14.2 의미

시연 몇 회 수준에서는 비용이 크지 않을 수 있지만, 반복 실험과 실패 빌드가 누적되면 비용과 시간이 함께 증가합니다.

### 14.3 대응

```text
불필요한 Run 반복 줄이기
로컬 cog build로 사전 검증
작은 모델/fixture로 기본 흐름 검증
실패 로그를 보관해 같은 실험 반복 방지
```

### 14.4 교훈

```text
AI 실험은 정확도뿐 아니라 비용 이력도 실험 기록에 포함해야 한다.
```

---

## 15. 실험 11 — Replicate token / Gemini key 노출 리스크

### 15.1 관찰

프로젝트 중 Railway variables 또는 화면 출력 과정에서 API token/key가 평문으로 노출된 이력이 있었습니다.

이 README에는 보안상 실제 값을 적지 않습니다.

### 15.2 위험

```text
외부 사용자가 API를 무단 호출할 수 있음
예상치 못한 비용 발생 가능
계정 보안 위험
```

### 15.3 대응

```text
시연 후 token revoke/regenerate
Railway Variables 새 값으로 교체
GitHub/Discord/README/스크린샷에서 key 제거
문서에는 key 대신 [REDACTED] 사용
```

### 15.4 교훈

```text
실험 이력관리에는 성공/실패뿐 아니라 보안 이슈도 포함해야 한다.
```

---

## 16. 실험 12 — V.E / Path B-3: Donut FT + YOLO 대안

### 16.1 배경

Qwen 계열 대형 VLM을 전체 도면에 바로 적용하는 방식은 강력하지만, 배포와 비용, cold start 문제가 컸습니다.

따라서 다른 접근으로 다음 구조가 검토되었습니다.

```text
YOLO로 도면 내 영역 검출
→ Donut FT로 region별 OCR/JSON 추출
→ Pydantic schema 검증
→ JSON output 생성
```

### 16.2 목표

```text
전체 이미지를 대형 VLM에 한 번에 맡기지 않는다.
도면의 table, title block, note, view 영역을 나눠 처리한다.
schema validation을 통해 출력 형태를 통제한다.
Replicate에서 더 안정적으로 실행 가능한 구조를 검토한다.
```

### 16.3 Replicate Playground 성공 사례

Replicate Playground에서 Path B-3 / Donut FT demo가 실행되었고, JSON Output이 생성되었습니다.

확인된 항목:

```text
model_id
quantity
drawing_id
image_size
schema_version
inference_seconds
yolo_regions_detected
status: succeeded
```

예시 관찰:

```text
pathb3 donut-ft demo
Generated in 약 16.5 seconds
quantity: 100
YOLO regions detected
schema_version 확인
status: succeeded
```

### 16.4 의미

이 실험은 IMMA VLM 실험에서 중요한 의미가 있습니다.

```text
Qwen 대형 VLM 하나만 고집하지 않고 대안 구조를 검토했다.
실제로 Replicate Playground에서 JSON output까지 확인했다.
시연에는 더 짧은 실행 시간과 구조화 JSON 확인 화면을 보여줄 수 있다.
```

### 16.5 교훈

```text
AI 도면 인식은 단일 모델보다 pipeline architecture가 중요할 수 있다.
영역 검출 + OCR/JSON + schema 검증 조합이 실서비스에 더 안정적일 수 있다.
```

---

## 17. 실험 13 — VLM 결과 JSON과 Match API 연결

### 17.1 목적

Replicate에서 생성된 VLM output은 그대로 UI에 보여주기보다, `/api/match-v2`가 이해할 수 있는 match input으로 변환되어야 합니다.

### 17.2 필요한 필드

```text
material
processes
quantity
envelope
tolerances
post_treatment
warnings
part_name
drawing_id
```

### 17.3 실험 포인트

```text
VLM JSON이 schema validation을 통과하는가
match input으로 변환 가능한가
재질 alias를 해소할 수 있는가
공정명이 lookup table과 맞는가
quantity가 RFQ 입력값과 충돌하지 않는가
warnings가 UI에 표시되는가
```

### 17.4 발견된 문제

```text
VLM 출력의 quantity와 사용자 입력 quantity가 다를 수 있음
도면에서 재질이 불명확하면 입력 재질 우선순위가 필요함
schema는 맞지만 내용이 틀릴 수 있음
일부 table/BOM 정보는 누락될 수 있음
```

### 17.5 교훈

```text
VLM 결과는 최종 판단이 아니라 match pipeline의 입력 후보로 봐야 한다.
AI가 읽은 정보와 사용자가 입력한 정보의 우선순위 정책이 필요하다.
```

---

## 18. 실험 14 — 시연 정책 결정: 실시간 호출 vs fixture

### 18.1 문제

Replicate 실시간 호출은 다음 리스크가 있었습니다.

```text
cold start 5~6분
worker setup failed 가능성
모델 version disabled 가능성
API token 문제
네트워크 지연
출력 JSON 불안정
```

### 18.2 선택지

```text
1. 실시간 VLM 호출을 그대로 시연한다.
2. 실시간 호출은 시작만 보여주고 결과는 fixture JSON으로 넘긴다.
3. Replicate Playground의 성공 화면을 근거 자료로 보여준다.
4. 항상 켜져 있는 Deployment 유료 옵션을 검토한다.
```

### 18.3 최종에 가까운 판단

시연 안정성을 위해 다음 방식이 가장 안전하다고 판단했습니다.

```text
도면 업로드와 AI 분석 시작은 보여준다.
실시간 분석이 오래 걸리면 미리 준비된 결과 화면으로 넘어간다.
Replicate 성공 로그와 JSON Output 화면은 기술 근거 자료로 보여준다.
matching-ui는 고정 fixture 값과 맞춰 시연한다.
```

### 18.4 교훈

```text
시연은 기술 실험의 모든 과정을 실시간으로 재현하는 자리가 아니다.
검증된 실험 결과를 바탕으로 안정적인 사용자 흐름을 보여주는 것도 합리적인 선택이다.
```

---

## 19. 성공 사례 정리

Replicate 실험에서 성공으로 볼 수 있는 항목은 다음과 같습니다.

```text
Replicate 계정 및 모델 생성
Cog CLI 로그인
cog push를 통한 model/version 생성
Playground input schema 표시
Prediction 로그 축적
일부 prediction succeeded 확인
비용/시간/cold start 데이터 관찰
Path B-3 Donut FT demo JSON Output 확인
Replicate 화면을 PPT 기술 근거 자료로 활용
```

발표용 문장:

```text
Replicate에서 여러 차례 Prediction을 실행하며 성공/실패 로그를 축적했고, 실제 처리 시간과 비용, cold start 리스크를 확인했습니다. 특히 Path B-3 Donut FT 경로에서는 도면 이미지 입력 후 JSON Output 생성까지 확인했습니다.
```

---

## 20. 실패 사례 정리

Replicate 실험에서 실패 또는 위험으로 남은 항목은 다음과 같습니다.

```text
worker setup failed
version disabled
unsupported type torch.Tensor
CogPath alias 문제
LoRA peft_version 오류
transformers architecture 미지원
vLLM/torch/CUDA 호환성 문제
cold start 5~6분
실시간 시연 안정성 부족
API token/key 노출 리스크
```

발표용 문장:

```text
실패도 중요한 실험 이력이었습니다. 단순히 모델이 안 된 것이 아니라, vLLM, torch, Cog worker setup, LoRA config, cold start 같은 운영 환경 리스크를 확인했고, 이를 바탕으로 Path B와 fixture 시연 전략을 세웠습니다.
```

---

## 21. 실험 기록 표

| 실험 ID | 날짜/구간 | 목적 | 방식 | 결과 | 판단 |
|---|---|---|---|---|---|
| REP-001 | 5/11 전후 | Qwen V.B 모델 Replicate 배포 | cog.yaml + predict.py + cog push | model/version 생성 단계 일부 성공 | push와 prediction 성공은 별도 |
| REP-002 | 5/11~5/12 | V.B Playground 실행 | Replicate Run | worker setup failed | vLLM/torch/Cog 호환성 점검 필요 |
| REP-003 | 5/12 | Cog input schema 수정 | torch.Tensor/CogPath 타입 수정 | build 단계 진입 개선 | predict signature 중요 |
| REP-004 | 5/12 | LoRA config 검증 | PEFT/adapter config 확인 | peft_version 오류 | peft/transformers 버전 정합 필요 |
| REP-005 | 5/12 | Path C 검토 | torch/vLLM 버전 조정 | 불확실성 큼 | 일정상 단독 경로 위험 |
| REP-006 | 5/12~5/13 | Path B 검토 | vLLM 제거, transformers 기반 | 안정성 우선 대안 | 시연 안정성 측면 유리 |
| REP-007 | 5/13~5/14 | cold start 측정 | Prediction 반복 실행 | 5~6분대 지연 관찰 | 실시간 시연 리스크 |
| REP-008 | 5/14~5/15 | Donut/YOLO Path B-3 | Replicate Playground | JSON output succeeded 확인 | PPT/기술 근거로 사용 |
| REP-009 | 5/15 | 발표자료 반영 | Prediction 화면 PPT 삽입 | 기술 스택 10번 슬라이드 제작 | 실험 이력 시각화 |

---

## 22. 다음 사람이 이어받을 때 해야 할 일

### 22.1 Replicate 계정 확인

```text
model 목록 확인
private/public 여부 확인
latest version 확인
billing/credit 확인
token 재발급 여부 확인
```

### 22.2 모델별 상태 정리

```text
qwen3vl-v-b-demo-v1
- vLLM 기반
- worker setup 실패 이력
- 대형 모델/cold start 리스크

v-e-donut-ft-pathb3-demo
- Donut/YOLO 기반 Path B-3
- JSON output 성공 화면 존재
- schema_version, yolo_regions_detected 확인 가능
```

### 22.3 파일 확인

```text
cog.yaml
predict.py
README.md
PATH_B_GUIDE.md
replicate_debug_package.zip
sample input image
sample output JSON
```

### 22.4 시연 전 확인

```text
Playground가 열리는지
Run 버튼이 보이는지
이전 succeeded prediction이 남아 있는지
JSON output 화면 캡처가 있는지
FastAPI에서는 fixture fallback이 준비되어 있는지
```

---

## 23. 발표에서 쓰기 좋은 30초 설명

```text
VLM 도면 분석은 Replicate와 Cog를 이용해 클라우드 추론 배포를 실험했습니다. 처음에는 Qwen 계열 대형 VLM을 vLLM 기반으로 올리는 경로를 시도했지만, worker setup 실패, torch/vLLM 호환성, LoRA config 오류, cold start 지연 같은 운영 리스크를 확인했습니다. 이후 vLLM을 제거한 Path B와 Donut/YOLO 기반 대안 구조를 검토했고, Replicate Playground에서 도면 이미지 입력 후 JSON Output이 생성되는 사례까지 확인했습니다. 이 과정에서 단순 성공 여부뿐 아니라 처리 시간, 비용, 실패 원인, 시연 가능성까지 실험 이력으로 관리했습니다.
```

---

## 24. 발표에서 쓰기 좋은 1분 설명

```text
Replicate 실험은 IMMA의 AI 도면 분석 파트를 실제 클라우드 추론으로 검증하기 위한 과정이었습니다. 초기에는 Qwen3-VL-30B 계열 대형 VLM을 Cog로 패키징해 Replicate에 올리는 방식을 시도했습니다. 이 과정에서 cog push 자체는 진행되었지만, 실제 Prediction 단계에서는 worker setup failed, unsupported type torch.Tensor, LoRA config 오류, transformers architecture 미지원 같은 문제가 반복되었습니다.

이 실험을 통해 모델 성능뿐 아니라 배포 안정성, 패키지 버전, cold start, 비용이 모두 중요한 판단 요소라는 점을 확인했습니다. 그래서 vLLM을 제거한 Path B와 Donut/YOLO 기반 Path B-3 대안을 검토했고, Replicate Playground에서 JSON Output이 생성되는 결과를 확보했습니다. 최종 시연에서는 실시간 호출 리스크를 줄이기 위해 fixture와 성공 로그를 함께 활용하는 전략을 세웠습니다.
```

---

## 25. 최종 결론

IMMA의 Replicate 실험은 성공만 있었던 실험이 아닙니다.

오히려 중요한 것은 실패를 통해 다음 판단이 만들어졌다는 점입니다.

```text
Qwen 대형 VLM 실험
→ worker setup / vLLM 호환 문제 확인
→ Cog input 타입과 LoRA config 문제 수정
→ Path C 버전 조정 검토
→ Path B 안정성 우선 경로 검토
→ Donut/YOLO Path B-3 대안 확인
→ 실시간 호출 대신 fixture/성공 로그 병행 시연 전략 결정
```

따라서 이 실험의 의미는 다음과 같습니다.

```text
단순히 AI 모델을 호출한 것이 아니라,
어떤 모델이 실제 서비스와 시연에 적합한지,
어떤 구조가 비용과 안정성 면에서 더 현실적인지,
실패 로그를 바탕으로 다음 대안을 어떻게 선택했는지 기록했다.
```

한 줄로 정리하면:

```text
Replicate 실험은 IMMA가 AI 도면 분석을 실제 클라우드 추론으로 연결하기 위해 거친 성공·실패·대안 선택의 이력이다.
```
