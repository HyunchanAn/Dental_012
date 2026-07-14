![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![Backend](https://img.shields.io/badge/Backend-YOLOv8-red) ![UI](https://img.shields.io/badge/UI-Streamlit-orange) ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github)

# Dental_012: Periapical Lesion Detection Module

## 개요
> **[학습 환경 사양]** 실질적 모델 학습은 **RTX 5080 + 라이젠9-6 9900x** 환경에서 진행되었습니다.

이 레포지토리는 파노라마 방사선 사진에서 **치근단 병소(Periapical Lesion)** 를 자동으로 탐지하고 분류하는 딥러닝 기반 모듈을 포함합니다. 

기존 `Dental_002` 모듈이 전반적인 구강 병소를 포괄적으로 탐지하는 반면, 이 모듈은 치근단 병소 탐지에 특화되어 구축됩니다. 또한, `Dental_008` (치아 분할) 모듈과 연동하여 발견된 병소가 어느 치아(FDI 번호)에 해당하는지 자동으로 매칭하는 기능을 목표로 합니다.

## 데이터셋
본 모듈은 Mendeley Data에 공개된 치근단 병소 파노라마 데이터셋을 사용합니다.
- **Dataset**: Panoramic radiographs with periapical lesions Dataset
- **URL**: [Mendeley Data Link](https://data.mendeley.com/datasets/kx52tk2ddj/3)
- **DOI**: `10.17632/kx52tk2ddj.3`
- **라이선스**: CC BY 4.0

### 설치 및 데이터 다운로드 방법
1. 상기 링크에 접속하여 데이터셋의 전체 압축 파일(Zip)을 다운로드합니다.
2. 다운로드한 파일을 `data/raw/` 폴더에 압축 해제합니다.
3. 압축이 풀리면 `Original JPG Images`, `Image Annotations` 등의 폴더가 생성됩니다.

## 파이프라인
1. **데이터 전처리 (`src/prepare_data.py`)**: 다운로드한 Annotation 파일을 YOLO 학습 형식으로 파싱 및 변환합니다.
2. **모델 학습 (`src/train.py`)**: 전처리된 데이터셋을 이용하여 YOLOv11 모델을 파인튜닝합니다.
3. **E2E 추론 및 FDI 연동 (`Dental_Panoramic_Reader`)**: 치아 위치 마스크(008)와 병소 바운딩 박스(012) 간의 IoU 및 거리 계산을 통해 최종 FDI 번호를 도출합니다.

## References
- Data in Brief (2024) – "A Dataset of apical periodontitis lesions in panoramic radiographs for deep-learning-based classification and detection"
- Mendeley Data: [Panoramic radiographs with periapical lesions Dataset](https://data.mendeley.com/datasets/kx52tk2ddj/3)

## 설치 및 실행 방법
```bash
pip install -r requirements.txt
```
