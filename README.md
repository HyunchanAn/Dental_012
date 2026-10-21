# Dental_012: Periapical Lesion Detection Module

![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![Backend](https://img.shields.io/badge/Backend-YOLO11s%20%2B%20Apex%20Attribution-red) ![License](https://img.shields.io/badge/License-MIT-green) ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github)

## 개요
- [학습 환경 사양]: 실질적 모델 학습은 RTX 5080 + 라이젠9-6 9900X 환경에서 진행되었습니다.

이 레포지토리는 파노라마 방사선 사진에서 치근단 병소(Periapical Lesion: 치근단 육아종, 치근단 낭종, 치근단 농양 등 방사선 투과성 병변)를 자동으로 탐지하고, `Dental_008` (치아 분할) 모듈의 치아 컨투어/바운딩 박스와 연계하여 다근치(Multi-rooted molar) 해부학적 치근단(Apex) 앵커점 기준으로 원인 치아(FDI 번호)를 매칭하는 모듈입니다.

## 데이터셋
본 모듈은 Mendeley Data에 공개된 치근단 병소 파노라마 데이터셋을 기반으로 음성 배경(정상 해부학적 골 투과상: 이공·상악동 기저부 등) 증강을 결합하여 사용합니다.
- Dataset: Panoramic radiographs with periapical lesions Dataset
- URL: [Mendeley Data Link](https://data.mendeley.com/datasets/kx52tk2ddj/3)
- DOI: `10.17632/kx52tk2ddj.3`
- 라이선스: CC BY 4.0

### 설치 및 데이터 다운로드 방법
1. 상기 링크에 접속하여 데이터셋 압축 파일(Zip)을 다운로드합니다.
2. 다운로드한 파일을 `data/raw/` 폴더에 압축 해제합니다.
3. `src/prepare_data.py`를 통해 YOLO 포맷(`data/yolo_dataset_augmented`, `data.yaml` 상대 경로 설정)으로 변환합니다.

## 파이프라인 및 모듈 구성
1. 데이터 전처리 (`src/prepare_data.py`): 다운로드한 XML Annotation 파일을 YOLO 정규화 좌표 형식으로 파싱 및 변환합니다.
2. 모델 학습 (`src/train.py`): `data.yaml`(`path: data/yolo_dataset_augmented`)을 이용하여 YOLO11s 모델을 학습합니다 (`torch.cuda.is_available()` 자동 감지 기반 CUDA/CPU 폴백 지원).
3. 단독 추론 및 다근치 치근단 FDI 매칭 (`src/inference.py` / `src/predict.py`):
   - `PeriapicalPredictor`: YOLO11s 치근단 병소 바운딩 박스 추론 및 임상 이중 임계값(`DEFINITE >= 0.50`, `REVIEW_REQUIRED >= 0.25`) 분류를 수행합니다.
   - `match_lesions_to_teeth`: 상악(`11~28`, 상단 치근단 `ty1`) 및 하악(`31~48`, 하단 치근단 `ty2`)의 해부학적 치근단 방향과 대구치(`#6, #7, #8`) 다근(근심·중앙·원심 치근단) 후보 앵커점을 계산하여 가장 인접한 치아 FDI 번호를 할당합니다. 치아 치근단과 무관한 고립 투과상은 `isolated_anatomical_suspicion=True` 안전 플래그를 부여합니다.

## 벤치마크 검증 성능 (Validation Metrics)
- 검증 데이터셋: 환자 분리(Patient-disjoint) 독립 검증 세트 `825장` (정상 해부학적 음영 Negative Background 포함, 정답 병소 1,344개)
- 프로덕션 가중치 식별자: `best.onnx` (SHA-256: `ee43dc87...`)
- 정량 지표:
  - `mAP@0.5`: `73.8%` (Negative Background 증강 재학습 기준 `73.7%` ~ `73.8%`, 구형 양성 단독 학습 `66.8%` 대비 `+7.0%p` 향상)
  - `Precision`: `70.9%` ~ `71.7%`
  - `Recall`: `66.9%` ~ `68.2%`
  - `mAP@0.5:0.95`: `34.9%`

## 설치 및 실행 방법
```bash
pip install -r requirements.txt
pytest tests/test_inference.py -v
```

## References
- Data in Brief (2024) - "A Dataset of apical periodontitis lesions in panoramic radiographs for deep-learning-based classification and detection"
- Mendeley Data: [Panoramic radiographs with periapical lesions Dataset](https://data.mendeley.com/datasets/kx52tk2ddj/3)

## 가중치 백업 안내
오래된 이전 가중치 파일(Legacy Backups)은 통일성을 위해 `Emergency_Weights_Vault/Dental_012/Backup_Checkpoints` 폴더에 별도로 안전하게 보존되어 있습니다.

## 연구용 실행 주의사항
- 기본 실행 주소: 127.0.0.1 (로컬 전용)
- 외부 공개/상용 배포 금지
- API 키·토큰은 환경변수로만 관리
- 오픈 데이터셋 외 민감 의료데이터 사용 금지
- 외부 접근이 필요하면 ALLOW_EXTERNAL_ACCESS=true 로만 명시적 허용
