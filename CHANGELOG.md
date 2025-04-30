# UFlow PyTorch 구현 변경 사항

## 2025-05-01 - SequenceLoss 및 FullSequentialLoss 제거
- 논문에 명시적으로 언급되지 않은 SequenceLoss 관련 코드 제거
- 복잡한 FullSequentialLoss 클래스 삭제 및 간단한 MultiScaleUFlowLoss 사용
- 훈련 파이프라인 단순화:
  - 연속 3프레임 대신 2프레임만 사용하도록 수정
  - sequential_dataloader 대신 기본 dataloader 사용
  - 불필요한 코드 복잡성 감소
- 시퀀스 손실 관련 매개변수 제거

## 2025-04-30 - 가려짐(occlusion) 추정 기능 개선
- `compute_range_map` 함수를 재작성하여 TensorFlow 구현과 일치하도록 변경
- 여러 가려짐 추정 방법(`wang`, `wang4`, `uflow` 등)에 대한 일관된 출력 제공
- 테스트 코드 작성 및 실행하여 올바른 가려짐 마스크 생성 확인

## 2025-04-30 - 시퀀스 손실(Sequence Loss) 강화
- 시퀀셜 프레임 손실 계산 개선
- TensorFlow 구현과 유사한 시퀀스 손실 기능 구현
- 주요 개선 내용:
  1. `compute_range_map` 함수 재구현 - PyTorch 1.10 이상 버전 호환성 지원
  2. `compute_fb_squared_diff`, `compute_fb_sum_squared` 함수 추가
  3. 다양한 마스킹 방식 지원:
     - `gaussian`: 가우시안 기반 일관성 마스크
     - `advection`: 가려짐 마스크 기반
     - `ddflow`: 임계값 기반 이진 마스크
  4. `SequenceLoss` 클래스 강화:
     - 다양한 마스킹 옵션 지원
     - 시간적 일관성 손실 계산 방법 개선
  5. `FullSequentialLoss` 클래스 업데이트:
     - 다중 스케일 손실과 시퀀스 손실 통합
     - TensorFlow 구현과 유사한 가중치 시스템 지원

- 테스트 코드 추가 및 검증
- 시퀀스 손실이 포함된 전체 손실 함수 완성 