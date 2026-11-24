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


## 목록·프롬프트와 실행

웹의 목록 새로고침은 `/demo/loras`에서 현재 어댑터가 제공하는 목록을 다시 읽습니다. 등록된 목록의 이름·설명·트리거를 검색할 수 있습니다. HF 검색 결과를 무조건 실행 가능한 항목으로 취급하지 않습니다.

카탈로그 항목에 `trigger_words`(최대 16개), `example_prompt`(최대 4,000자), `source_url`(HF 모델 주소)을 선언할 수 있습니다. 트리거 버튼은 사용자가 눌렀을 때만 입력에 추가됩니다. 추천 프롬프트 사용은 해당 LoRA만 선택하고 프롬프트를 채웁니다. "이 LoRA만 실행"은 다른 선택을 해제한 뒤 현재 프롬프트로 실행하며, 입력이 비어 있으면 해당 LoRA의 추천 프롬프트를 사용합니다. 입력을 보고 LoRA를 몰래 바꾸지는 않습니다.

### Hugging Face 검색과 가져오기

웹의 **Hugging Face에서 LoRA 가져오기**는 패키지 어댑터가
`lora_hub = HfLoraLibrary(bound_loras, base_models=[...], convert=callback)`를
제공할 때 활성화됩니다. 검색은 공식 Hub API의 `lora`·`base_model` 메타데이터를
사용하며 `owner/model` 직접 입력도 가능합니다. 메타데이터 일치는 후보 조건일 뿐,
실행 가능성과 이미지 품질을 보증하지 않습니다.

사용자가 파일을 골라 **다운로드·호환 검사**를 누르면 immutable revision의
safetensors만 받습니다. 신뢰하는 패키지의 converter는 `(tensor_dict,
component_bindings)`를 반환해야 합니다. 모든 텐서의 대상·shape·dtype·alpha를
검사하고, 일부만 적용하거나 원격 Python/pickle 코드를 실행하지 않습니다.
검사 통과 후에도 자동 적용하지 않으며 목록에서 선택해야 합니다.

가져온 파일과 checksum·출처는 `~/.cache/fxbx/loras/<execution-contract-hash>/`에
별도로 보관합니다. 원본 FXBX와 manifest는 변경하지 않습니다. 파일당 기본
512 MiB, 로컬 라이브러리 2 GiB·32개 제한, 원자적 인덱스 교체와 프로세스 잠금을
사용합니다. 재시작 시 검증 후 복원합니다. 패키지에 converter가 없으면 가져오기
UI는 숨기며 기존 등록 LoRA 선택은 계속 사용할 수 있습니다.

LoRA는 선택 사항입니다. 처음 화면과 초기화는 LoRA를 선택하지 않으며,
목록·Hub 가져오기·강도 설정은 기본적으로 접힌 **LoRA 추가 옵션** 안에 있습니다.
LoRA 없이 프롬프트만 입력하고 바로 실행할 수 있습니다.
