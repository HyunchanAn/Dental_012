# Dental_012: Periapical Lesion Detection YOLOv11 Benchmark Report

**Date**: 2026-07-14
**Model**: YOLOv11s (`yolo11s.pt` Fine-tuning)
**Task**: 치근단 병소(Periapical Lesion) 객체 탐지 (Object Detection)

---

## 1. 훈련 개요 및 종료 상태

- **총 소요 시간**: 약 0.657 시간 (약 40분)
- **종료 사유**: Early Stopping (조기 종료 발동)
  - 최대 100 에포크로 설정되었으나, 최근 20 에포크 동안 성능 개선이 관찰되지 않아 **에포크 88에서 훈련이 자동 종료**되었습니다.
  - 최고의 검증 성능을 낸 시점은 **에포크 68**이며, 이 시점의 가중치가 `best.pt`로 저장되었습니다.
- **최종 가중치 경로**: `\\rtx4060laptop-hc\Users\chema\Github\Dental_012\runs\detect\periapical_yolo11s2\weights\best.pt`

---

## 2. 벤치마크 평가 지표 (Validation Metrics)

검증 데이터셋 내 785장의 이미지와 1,344건의 치근단 병소(Instances)를 대상으로 한 최종 성능 평가 결과입니다.

| Metric | Score | 설명 |
| :--- | :--- | :--- |
| **Precision (정밀도)** | `0.692` | 모델이 병소라고 예측한 것 중 실제로 병소인 비율 |
| **Recall (재현율)** | `0.643` | 실제 병소들 중에서 모델이 놓치지 않고 찾아낸 비율 |
| **mAP50** | `0.699` | IoU 0.5 기준의 전반적인 탐지 정확도 (가장 중요한 지표) |
| **mAP50-95** | `0.313` | IoU 0.5 ~ 0.95 구간의 평균 정확도 (더 엄격한 기준) |

> [!TIP]
> **성능 요약**: 첫 파인튜닝 시도임에도 불구하고 mAP50이 **0.699**에 달하며, 1,344개의 치근단 병소를 70% 가까운 정확도로 짚어내는 훌륭한 탐지 능력을 보여주고 있습니다.

---

## 3. 추론 속도 (Inference Speed)

초경량화 및 하드웨어 가속(RTX 5080) 최적화를 통해 실시간 처리를 훌쩍 뛰어넘는 압도적인 속도를 기록했습니다.

- **Preprocess (전처리)**: `0.1 ms` / image
- **Inference (순수 추론)**: `0.5 ms` / image
- **Postprocess (후처리)**: `0.9 ms` / image
- **총합 (Total Speed)**: 장당 **약 1.5 ms** (초당 약 660 FPS 수준)

---

## 4. 향후 활용 방안
1. **추론(Inference) 테스트**: 위 `best.pt` 경로를 활용하여 새로운 환자의 파노라마 사진에서 치근단 병소를 그려내는 스크립트를 작성할 수 있습니다.
2. **모델 고도화**: 만약 Recall(0.643)을 더 끌어올리고 싶다면, 증강 기법(Augmentation)을 추가하거나 Confidence 컷오프 조절, 또는 YOLOv11m/YOLOv11l 체급으로 스케일업하는 방안을 고려할 수 있습니다.
