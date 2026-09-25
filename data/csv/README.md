# EXAONE 최종 학습용 데이터 분할

이 파일들은 `aics_data.zip` 안의 원본 `aics_data/cs_data.csv`를 기준으로 다시 분할했습니다.

## 왜 다시 나눴나요?

기존 팀원 split은 일반적인 8:1:1 분할로는 사용할 수 있었지만,
앞서 진행한 1차/2차 모델 선정 실험과 Final Test를 일관되게 유지하기 위해
2차 실험 당시의 질문-group split 규칙을 다시 적용했습니다.

## 파일

- `train_11147.csv`
  - EXAONE 최종 QLoRA 학습용
  - 원본 행 11,147개

- `validation_1413.csv`
  - Trainer validation loss 확인용
  - 원본 행 1,413개

- `validation_canonical_981.csv`
  - 튜닝 과정의 답변 생성 평가용
  - 동일 질문을 1개로 묶은 981개 질문
  - 여러 정답이 존재하면 `reference_responses_json`에 모두 보존

- `final_test_1389.csv`
  - 원본 형태로 보관한 Final Test
  - 모델 튜닝에 사용하지 말 것

- `final_test_canonical_981.csv`
  - 최종 성능 평가용 권장 파일
  - 981개의 서로 다른 질문
  - 모든 하이퍼파라미터 결정이 끝난 뒤 딱 한 번 사용

- `split_manifest.json`
  - 데이터 개수와 중복 검증 결과

## 중복 검증

- Train ↔ Validation 질문 중복: 0
- Train ↔ Final Test 질문 중복: 0
- Validation ↔ Final Test 질문 중복: 0
- 1차 사용 질문 ↔ Validation: 0
- 1차 사용 질문 ↔ Final Test: 0

## 권장 순서

EXAONE Base
→ train_11147.csv로 QLoRA
→ validation_1413.csv / validation_canonical_981.csv로 튜닝
→ 설정 확정
→ final_test_canonical_981.csv를 마지막에 한 번 평가

주의: Final Test는 학습이나 하이퍼파라미터 선택에 사용하지 마세요.
