![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen "Status") ![Python](https://img.shields.io/badge/Python-3.12%2B-blue "Python") ![Backend](https://img.shields.io/badge/Backend-YOLO11s%20%2B%20Apex%20Attribution-red "Backend") ![License](https://img.shields.io/badge/License-MIT-green "License") ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github "CI/CD Pipeline")

# Dental_012: Periapical Lesion Detection Module (치근단 병소 탐지 및 다근치 치근단 매핑)

## 개요

`Dental_012`는 파노라마 방사선 사진에서 치근단 병소(Periapical Lesion: 치근단 육아종, 치근단 낭종, 치근단 농양 등 방사선 투과성 병변)를 자동으로 탐지하고, `Dental_008` (치아 분할) 모듈의 치아 컨투어/바운딩 박스와 연계하여 다근치(Multi-rooted molar) 해부학적 치근단(Apex) 앵커점 기준으로 원인 치아(FDI 번호)를 매칭하는 모듈입니다.

- [학습 및 추론 하드웨어 사양]:
  - Training Env (메인 워크스테이션): AMD Ryzen 9 9900X, NVIDIA GeForce RTX 5080 (16GB VRAM), 64GB RAM
  - Development / Inference Env (랩탑 워크스테이션): Intel Core i5-14450HX, NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), 16GB RAM

---

## 주요 기능 및 아키텍처

### 1. 파이프라인 및 모듈 구성
1. [데이터 전처리 (`src/prepare_data.py`)]: 다운로드한 XML Annotation 파일을 YOLO 정규화 좌표 형식으로 파싱 및 변환합니다.
2. [모델 학습 (`src/train.py`)]: `data.yaml`(`path: data/yolo_dataset_augmented`)을 이용하여 `YOLO11s` 모델을 학습합니다 (`torch.cuda.is_available()` 자동 감지 기반 CUDA/CPU 폴백 지원).
3. [단독 추론, 2-Stage 치근단 패치 크롭 및 다근치 치근단 FDI 매칭 (`src/inference.py` / `src/predict.py`)]:
   - `PeriapicalPredictor`: `YOLO11s` 치근단 병소 바운딩 박스 추론, 2-Stage 고해상도 치근단(Apex) 패치 크롭·역투영(`extract_apex_roi_patches`, `project_patch_bbox_to_global`, `predict_two_stage_apex_patches`), 그리고 임상 이중 임계값(`DEFINITE >= 0.50`, `REVIEW_REQUIRED >= 0.25`) 분류를 수행합니다.
   - `match_lesions_to_teeth`: 상악(`11~28`, 상단 치근단 `ty1`) 및 하악(`31~48`, 하단 치근단 `ty2`)의 해부학적 치근단 방향과 대구치(`#6, #7, #8`) 다근(근심·중앙·원심 치근단) 후보 앵커점을 계산하여 가장 인접한 치아 FDI 번호를 할당합니다. 치아 치근단과 무관한 고립 투과상은 `isolated_anatomical_suspicion=True` 안전 플래그를 부여합니다.

---

## 벤치마크 및 검증 성능

- [검증 데이터셋]: 환자 분리(Patient-disjoint) 독립 검증 세트 `825장` (정상 해부학적 음영 Negative Background 340장 포함, 정답 병소 `1,344개`)
- [프로덕션 가중치 식별자]: `best.onnx` (SHA-256: `ee43dc87...`)
- [정량 지표 (`Dental_000/reports/260917_2105_Dental012_Negative_Aug_Benchmark_Report.md`)]:
  - `mAP@0.5`: `73.7%` ~ `73.8%` (구형 양성 단독 학습 `66.8%` 대비 `+7.0%p` 향상)
  - `Precision`: `70.9%` ~ `71.7%` (이공·상악동 기저부 등 정상 해부학적 골 투과상 오탐 억제)
  - `Recall`: `66.9%` ~ `68.2%`
  - `mAP@0.5:0.95`: `34.9%`

---

## 데이터셋 출처 및 라이선스

- [데이터셋 출처]: Panoramic radiographs with periapical lesions Dataset (Mendeley Data)
  - URL: [https://data.mendeley.com/datasets/kx52tk2ddj/3](https://data.mendeley.com/datasets/kx52tk2ddj/3)
  - DOI: `10.17632/kx52tk2ddj.3`
  - 라이선스: `CC BY 4.0`
- [소스 코드 라이선스]: 본 프로젝트의 소스 코드는 `MIT License` ([LICENSE](LICENSE), `Copyright (c) 2026 Hyunchan An`) 하에 배포됩니다.

---

## 설치 및 실행 방법

1. 데이터 준비:
   - Mendeley Data 링크에서 압축 파일을 다운로드하여 `data/raw/`에 압축 해제한 뒤 `python src/prepare_data.py`를 실행합니다.
2. 패키지 설치 및 단위 테스트 실행:
   ```bash
   pip install -r requirements.txt
   pytest tests/test_inference.py -v
   ```

---

## 참고 문헌 및 연구용 실행 주의사항

- Data in Brief (2024) - "A Dataset of apical periodontitis lesions in panoramic radiographs for deep-learning-based classification and detection"
- Mendeley Data: [Panoramic radiographs with periapical lesions Dataset](https://data.mendeley.com/datasets/kx52tk2ddj/3)
- [가중치 백업 안내]: 오래된 이전 가중치 파일(Legacy Backups)은 `Emergency_Weights_Vault/Dental_012/Backup_Checkpoints` 폴더에 별도로 보존되어 있습니다.
- [연구용 실행 주의사항]:
  - 기본 실행 주소: `127.0.0.1` (로컬 전용)
  - 외부 공개/상용 배포 금지 및 오픈 데이터셋 외 민감 의료데이터 사용 금지
  - API 키·토큰은 환경변수로만 관리하며, 외부 접근이 필요하면 `ALLOW_EXTERNAL_ACCESS=true`로만 명시적 허용
