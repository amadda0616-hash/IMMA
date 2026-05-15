# IMMA Replicate / Cog 빌드·Push·실행 README

작성 목적:  
IMMA 프로젝트에서 VLM 도면 인식 모델을 Replicate에 올리거나, Cog로 빌드·푸시·실행할 때 필요한 파일, 명령어, 점검 순서, 자주 발생한 오류를 정리한 문서입니다.

---

## 1. 이 README가 다루는 범위

이 문서는 FastAPI 서버 작업 README가 아니라, **Replicate + Cog 기반 VLM 모델 배포 작업**을 위한 README입니다.

다루는 작업:

```text
Cog 설치/로그인
cog.yaml 확인
predict.py 확인
로컬 cog build
로컬 cog predict
Replicate model 생성
cog push
Replicate version 확인
Playground 실행
Prediction 로그 확인
실패 시 디버그 패키지 정리
```

---

## 2. 전체 흐름

Replicate/Cog 배포 흐름은 아래 순서로 봅니다.

```text
VLM 모델 파일/코드 준비
→ cog.yaml 작성
→ predict.py 작성
→ 로컬에서 cog build
→ 가능하면 cog predict
→ Replicate 로그인
→ Replicate model 생성
→ cog push
→ Replicate version 생성 확인
→ Playground/API에서 Prediction 실행
→ Output JSON 확인
→ 로그/시간/비용 기록
```

중요한 점은 **cog push 성공 = 실제 추론 성공**이 아니라는 것입니다.

```text
cog push 성공
= Replicate에 image/version이 올라감

Prediction succeeded
= 실제 worker setup과 inference까지 성공
```

발표나 이력관리에서는 이 둘을 구분해서 기록해야 합니다.

---

## 3. 필요한 파일 목록

Replicate/Cog 작업을 하려면 최소한 아래 파일이 필요합니다.

```text
cog.yaml
predict.py
requirements.txt 또는 python_packages 목록
입력 테스트 이미지
샘플 output JSON
README 또는 실행 메모
```

모델 방식에 따라 추가로 필요한 것:

```text
LoRA adapter 파일
checkpoint 파일
tokenizer/config 파일
model config
학습된 weight 파일
schema 정의 파일
postprocess 코드
```

IMMA 기준으로는 아래 파일들이 중요했습니다.

```text
cog.yaml
predict.py
sample drawing image
sample_vlm_result.json
V.E / V.B schema
Replicate prediction log
Path B guide
debug report
```

---

## 4. cog.yaml의 역할

`cog.yaml`은 Replicate/Cog가 어떤 환경에서 모델을 실행할지 정의하는 파일입니다.

주로 들어가는 내용:

```yaml
build:
  gpu: true
  python_version: "3.11"
  python_packages:
    - torch
    - transformers
    - pillow
    - pydantic
predict: "predict.py:Predictor"
```

확인해야 할 것:

```text
python_version
python_packages
system_packages
gpu 사용 여부
predict 경로
CUDA/torch/vLLM/transformers 버전
```

주의:

```text
패키지 버전을 너무 느슨하게 쓰면, 나중에 같은 코드를 push해도 다른 버전이 설치되어 오류가 날 수 있습니다.
가능하면 주요 패키지는 버전을 고정하는 것이 안전합니다.
```

예시:

```yaml
build:
  gpu: true
  python_version: "3.11"
  python_packages:
    - torch==2.4.0
    - transformers==4.45.2
    - pillow==10.4.0
    - pydantic==2.9.2
predict: "predict.py:Predictor"
```

---

## 5. predict.py의 역할

`predict.py`는 Replicate에서 실제 추론을 실행하는 파일입니다.

기본 구조:

```python
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        # 모델 로드
        pass

    def predict(self, image: Path = Input(description="Input drawing image")):
        # 이미지 읽기
        # 모델 추론
        # JSON 반환
        return result
```

IMMA에서 확인해야 할 것:

```text
setup()에서 모델이 정상 로드되는지
predict() 입력 타입이 Cog에서 지원되는지
출력 JSON이 schema에 맞는지
도면 이미지가 PIL/OpenCV로 정상 열리는지
GPU 메모리를 너무 많이 쓰지 않는지
```

주의:

```text
Cog가 지원하지 않는 타입을 predict() 인자로 쓰면 schema 생성 단계에서 실패할 수 있습니다.
예: torch.Tensor를 Input 인자로 직접 받는 방식은 피하는 것이 안전합니다.
```

---

## 6. 로컬에서 Cog 빌드하기

작업 폴더에서 실행합니다.

```bash
cog build
```

성공하면 Docker image가 만들어집니다.

확인할 것:

```text
패키지 설치 성공 여부
CUDA/torch 충돌 여부
predict.py import 오류 여부
schema 생성 오류 여부
```

실패할 때 흔한 원인:

```text
패키지 버전 충돌
Python 버전 불일치
system package 누락
predict.py 문법 오류
Cog Input 타입 오류
vLLM/torch/CUDA 호환 문제
```

---

## 7. 로컬에서 Cog Predict 실행하기

가능하면 push 전에 로컬에서 실행합니다.

```bash
cog predict -i image=@sample.png
```

또는 입력 이름이 다르면:

```bash
cog predict -i input=@sample.png
```

실행 전 확인:

```text
predict.py의 Input 이름
테스트 이미지 경로
GPU 사용 가능 여부
모델 다운로드 시간
로컬 디스크 여유 공간
```

주의:

```text
Qwen3-VL-30B 같은 대형 모델은 로컬 PC에서 거의 실행이 어렵습니다.
이 경우 local cog predict는 생략하고, cog build와 push 중심으로 확인할 수 있습니다.
```

---

## 8. Replicate 로그인

먼저 Replicate 계정과 API token이 필요합니다.

```bash
cog login
```

또는 환경변수로 설정할 수 있습니다.

```bash
export REPLICATE_API_TOKEN=your_token_here
```

Windows PowerShell 예시:

```powershell
$env:REPLICATE_API_TOKEN="your_token_here"
```

주의:

```text
토큰은 GitHub에 올리면 안 됩니다.
README나 스크린샷에 노출되지 않게 주의해야 합니다.
```

---

## 9. Replicate 모델 생성

Replicate 웹에서 model을 먼저 만들거나, push 과정에서 지정한 경로에 맞춰 생성합니다.

예시 model name:

```text
rnjsxodms12-star/qwen3vl-v-b-demo-v1
rnjsxodms12-star/v-e-donut-ft-pathb3-demo
```

공개 범위:

```text
private 권장
시연용이면 private 유지
```

---

## 10. cog push 실행

기본 형식:

```bash
cog push r8.im/<owner>/<model-name>
```

IMMA 예시:

```bash
cog push r8.im/rnjsxodms12-star/qwen3vl-v-b-demo-v1
```

또 다른 예시:

```bash
cog push r8.im/rnjsxodms12-star/v-e-donut-ft-pathb3-demo
```

성공하면 Replicate에 새 version이 생성됩니다.

확인할 것:

```text
build 단계 통과
image push 완료
version hash 생성
Replicate model page에서 version 보이는지
Playground input schema가 뜨는지
```

---

## 11. Push 후 반드시 확인할 것

`cog push`가 끝난 뒤 바로 끝내면 안 됩니다.

Replicate 웹에서 확인:

```text
Model page 접속
Versions 확인
Playground 열기
Input form이 정상 표시되는지 확인
테스트 이미지 업로드
Run 실행
Prediction succeeded 여부 확인
Output JSON 확인
Setup logs 확인
Total duration / running time / cost 확인
```

기록할 항목:

```text
prediction id
model version
입력 이미지
실행 시간
queued time
running time
total duration
approximate cost
status: succeeded / failed
오류 로그
출력 JSON
```

---

## 12. Replicate Prediction 로그 확인

Replicate에서 `Predictions` 탭을 보면 실행 이력을 확인할 수 있습니다.

확인할 것:

```text
Status
Model or Deployment
Source
Queued
Running
Total duration
Approximate cost
Created time
```

IMMA 발표에서 쓸 수 있는 표현:

```text
Replicate Prediction 로그를 통해 실제 VLM 추론의 성공 여부, 처리 시간, 비용, cold start 리스크를 확인했습니다.
```

---

## 13. IMMA에서 겪은 주요 오류

### 1. unsupported type torch.Tensor

증상:

```text
infer_schema(func): Parameter input has unsupported type torch.Tensor
```

의미:

```text
Cog가 predict 함수의 입력 타입을 API schema로 변환하지 못한 상태입니다.
```

대응:

```text
predict() 인자에서 torch.Tensor 직접 사용하지 않기
image: Path 또는 image: str 등 Cog 지원 타입 사용
내부에서 PIL/torch Tensor로 변환하기
```

---

### 2. CogPath alias 문제

증상:

```text
unsupported type: CogPath
```

가능 원인:

```text
from cog import Path as CogPath 형태의 alias를 Cog schema가 제대로 인식하지 못함
```

대응:

```python
from cog import Path
```

또는 내부 파일 경로 처리는:

```python
from pathlib import Path as LocalPath
```

처럼 구분합니다.

---

### 3. worker setup failed

증상:

```text
worker setup failed
Setup failed
```

가능 원인:

```text
패키지 설치 실패
모델 로딩 실패
GPU 메모리 부족
vLLM/torch/CUDA 호환 문제
transformers 버전 문제
LoRA config 문제
```

대응:

```text
Setup logs 확인
torch/transformers/vLLM 버전 고정
불필요한 패키지 제거
모델 크기 줄이기
vLLM 제거 후 transformers 기반 Path B 검토
```

---

### 4. LoraConfig peft_version 오류

증상:

```text
TypeError: LoraConfig.__init__() got an unexpected keyword argument 'peft_version'
```

가능 원인:

```text
PEFT/transformers/adapter config 버전 불일치
```

대응:

```text
peft 버전 확인
LoRA config에서 지원하지 않는 키 제거
사용 모델과 adapter 생성 버전 맞추기
```

---

### 5. architecture not recognized

증상:

```text
model architecture not recognized
model type qwen3_vl_moe not recognized
```

가능 원인:

```text
transformers 버전이 해당 모델 구조를 지원하지 않음
```

대응:

```text
transformers 버전 업데이트
모델을 지원하는 branch/commit 사용
더 안정적인 Qwen2-VL 계열로 임시 전환
```

---

### 6. cold start 지연

증상:

```text
Prediction queued/running 시간이 길어짐
시연 중 1~6분 이상 대기
```

대응:

```text
시연 전 warm-up call
실시간 호출 대신 fixture JSON 준비
Playground 결과 화면 미리 열어두기
```

---

## 14. Path A / Path B / Path C 판단

IMMA VLM 배포 과정에서는 여러 대안을 검토했습니다.

### Path A: Cog CLI / 환경 업데이트

```text
Cog CLI 또는 관련 환경을 업데이트해서 기존 구조를 살리는 방향
```

장점:

```text
기존 코드 변경이 적음
```

단점:

```text
근본 원인이 vLLM/torch 호환이면 해결이 안 될 수 있음
```

---

### Path B: vLLM 제거 + transformers 기반 전환

```text
대형 Qwen3 + vLLM 구조를 줄이고, transformers 기반 모델로 안정성 우선 전환
```

장점:

```text
Cog setup 성공 가능성 증가
의존성 단순화
GPU 요구량 감소 가능
시연 안정성 증가
```

단점:

```text
속도나 성능이 낮아질 수 있음
```

발표용 설명:

```text
가장 큰 모델보다 실제로 시연에서 안정적으로 도는 구조를 우선했습니다.
```

---

### Path C: torch/vLLM 버전 고정 또는 다운그레이드

```text
vLLM 구조를 유지하되 torch, transformers, vLLM 버전을 조정하는 방향
```

장점:

```text
성공하면 기존 고성능 구조 유지 가능
```

단점:

```text
버전 조합 탐색 비용이 큼
반복 빌드 비용과 시간이 큼
```

---

## 15. V.B / V.E 구분

### V.B Demo

```text
Qwen 계열 VLM을 이용해 도면 전체를 한 번에 읽는 방식
```

특징:

```text
큰 모델 기반
도면 전체 이해 시도
Replicate/Cog 배포 실험
cold start와 setup 문제가 주요 리스크
```

### V.E / Path B-3

```text
Donut FT + YOLO 기반 대안 구조
```

특징:

```text
YOLO로 영역 검출
Donut으로 OCR/JSON 추출
Pydantic schema 검증
Replicate에서 JSON Output 성공 확인
16.5초 실행 사례 확인
```

발표용 표현:

```text
초기에는 Qwen 계열 VLM을 실험했지만, 배포 안정성과 비용 문제를 확인한 뒤 Donut/YOLO 기반 Path B 대안도 함께 검토했습니다.
```

---

## 16. Output JSON에서 확인할 것

Prediction이 성공하면 JSON Output을 확인합니다.

예시 확인 항목:

```text
status: succeeded
model_id
drawing_id
quantity
image_size
schema_version
inference_seconds
yolo_regions_detected
title_block
parts
warnings
```

IMMA에서 중요한 이유:

```text
VLM 결과 JSON이 match input으로 변환 가능한지 확인해야 하기 때문입니다.
```

---

## 17. FastAPI 서버와 Replicate 연결 시 필요한 값

FastAPI에서 Replicate를 호출하려면 환경변수가 필요합니다.

```text
REPLICATE_API_TOKEN
REPLICATE_MODEL_VERSION
V_E_DEMO_MODE=cloud
V_E_CLOUD_PROVIDER=replicate
```

서버 흐름 예시:

```text
클라이언트 도면 업로드
→ FastAPI 서버에서 이미지 저장
→ Replicate API 호출
→ Prediction 생성
→ Output JSON 수신
→ match input 변환
→ /api/match-v2 호출
→ 업체 추천 결과 반환
```

주의:

```text
시연에서는 실시간 Replicate 호출이 오래 걸릴 수 있으므로 fixture JSON fallback을 준비합니다.
```

---

## 18. 시연 전 Replicate 체크리스트

```text
1. Replicate billing/credit 상태 확인
2. API token 유효성 확인
3. model page 접속 확인
4. latest version 확인
5. Playground input form 확인
6. 테스트 이미지 업로드
7. Run 실행
8. Prediction succeeded 확인
9. Output JSON 복사/저장
10. 실행 시간과 비용 기록
11. 실패 로그 스크린샷 저장
12. 시연용 fixture JSON 준비
```

---

## 19. 디버그 패키지에 넣을 것

문제가 생겼을 때 팀원에게 넘길 패키지에는 아래를 넣습니다.

```text
cog.yaml
predict.py
requirements.txt
실행한 명령어
터미널 로그
Replicate setup logs
prediction id
입력 이미지
output JSON 또는 error JSON
수정 전후 파일
README_디버그메모.md
```

파일명 예시:

```text
replicate_debug_package_20260514.zip
replicate_v_b_demo_v1_수정_전_후.zip
path_b_donut_yolo_test_log.zip
```

---

## 20. 발표용 요약 문장

짧은 버전:

```text
VLM 도면 분석은 Replicate와 Cog를 이용해 클라우드 추론 배포를 실험했습니다. cog build, cog push, Playground 실행, Prediction 로그를 통해 실제 추론 성공 여부와 처리 시간, 비용, cold start 리스크를 확인했습니다.
```

조금 긴 버전:

```text
초기에는 Qwen 계열 VLM을 Cog로 패키징해 Replicate에 올리는 구조를 실험했습니다. 하지만 vLLM, torch, Cog worker setup 호환성 문제가 반복되어, 단순히 모델 성능만이 아니라 배포 안정성이 중요하다는 점을 확인했습니다. 이후 Path B로 vLLM을 제거하고, Donut/YOLO 기반 대안 구조를 검토하면서 실제 JSON Output 생성과 시연 가능성을 확인했습니다.
```

---

## 21. 한 줄 요약

Replicate/Cog 작업에서 중요한 것은 `cog push` 자체가 아니라, **빌드 → Push → Prediction 실행 → Output JSON 확인 → 로그 기록 → 다음 대안 판단**까지 이어지는 전체 이력입니다.
