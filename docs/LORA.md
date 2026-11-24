# 서빙 웹 화면에서 LoRA 선택

fxbX 0.9.7의 `fxbx serve MODEL --port 8000`은 해당 패키지가 선언한 LoRA 목록을 웹 화면에 표시합니다. 체크박스로 선택하고 적용 강도를 입력한 뒤 실행합니다. 결과 이미지 아래에는 실제 적용한 ID·강도가 표시됩니다. 아무것도 선택하지 않으면 기본 가중치를 사용합니다. 예제 입력 버튼도 LoRA 선택을 초기화합니다.

**기존 모델 패키지가 자동으로 LoRA 지원 모델로 바뀌지는 않습니다.** 어댑터가 LoRA를 지원하지 않거나 등록된 파일이 없으면 이유를 표시하며, 미지원 요청을 적용된 것처럼 반환하지 않습니다. HF 검색 결과는 후보일 뿐이며, 패키지 제작자가 호환성과 사용 조건을 확인한 항목만 등록합니다.

HTTP `/predict`도 같은 경로를 사용합니다. 아래 ID는 패키지가 실제 제공하는 ID로 바꿉니다.

```json
{
  "prompt": "a portrait, toy_face",
  "seed": 42,
  "output": "/tmp/result.png",
  "loras": [{"id": "toy-face", "scale": 0.8}]
}
```

응답의 `applied_loras`는 적용 후 추론이 성공했을 때만 반환합니다. 다음 요청에 `loras`가 없거나 빈 목록이면 기본 모델로 복원합니다. LoRA 수와 강도 범위는 패키지가 정합니다. 현재 HTTP 서버는 요청을 직렬 실행하며, Python Session도 가중치 변경과 추론을 같은 잠금으로 보호합니다. 서로 다른 LoRA를 한 NPU 배치에서 동시에 실행하는 기능은 아닙니다.

선택이 바뀌면 CPU에서 가중치를 병합하고 SDK를 통해 장치 가중치를 바꿉니다. 같은 선택을 반복하면 변경을 생략합니다. 변경 후 첫 요청은 준비 비용이 더해지므로 반복 추론 시간과 구분합니다. 장치 변경 실패 시 세션을 재적재해야 합니다.

## 패키지 어댑터 계약

신뢰된 어댑터는 다음 두 메서드를 제공해야 합니다.

- `lora_catalog()`: `id`, `name`, 선택적 `description`, `min_scale`, `max_scale`, `default_scale` 목록.
- `apply_loras(selection)`: 정규화된 `[{id, scale}]`를 완전히 적용하고 성공 후 반환. 빈 목록은 반드시 기본 가중치를 복원해야 합니다. 실패한 부분 적용을 정상으로 취급하면 안 됩니다.

최대 선택 수는 `max_active_loras` 속성으로 선언합니다(기본 1, 최대 8). `predict()`에는 `loras`를 제외한 본래 모델 입력만 전달합니다. 이 후크만 구현한 어댑터의 정확성은 제작자가 검증해야 하며, fxbX가 임의 모델의 LoRA 호환성을 추론하지 않습니다.

## 저장된 NPU 실행계획용 로더

`fxbex.lora_weights.BoundLoras`는 `lora.json`과 패키지의 `adapters/*.safetensors`에서 명시적으로 연결된 선형 A/B 텐서를 읽습니다. 변환·패키징은 FuryNo5/FHA의 역할입니다. fxbX는 소스 모델·TCL·EDF를 생성하거나 컴파일하지 않습니다.

지원 범위:

- 동일 shape·dtype·stride의 **완전한 parameter 입력**을 받는 stateless ExecutionFXB.
- 선형 LoRA `W + scale * alpha/rank * B@A`. FP32로 병합 후 선언된 실행 dtype으로 변환.
- 명시된 모든 A/B 텐서를 연결해야 함. 모르는 텐서를 조용히 무시하거나 부분 적용하지 않음.
- 고정 상수·frozen buffer·분할 checkpoint parameter·stateful plan·컨볼루션/양자화/DoRA 등 다른 표현은 이 로더에서 미지원.
- 원본 체크포인트와 EDF 해시는 유지. 변경 가중치는 별도 장치 버퍼에 적재하고 RunnableSequence의 바인딩을 재생성.
- LoRA CPU 캐시 기본 512MiB, 교체 가중치 staging 기본 2GiB. 어댑터는 실제 환경 예산을 검토한 경우에만 변경 가능.

등록된 LoRA와 베이스 모델의 라이선스를 각각 따라야 합니다. 예제용 파일의 사용 조건이 불명확한 경우 모델 가중치를 새 공개 릴리스로 재배포하지 않습니다.
