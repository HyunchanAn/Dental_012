# [ 261125 1530 랩탑 개발팀 ] (Dental_012#4 브리지 정합화, 컬러 스페이스 통일 및 HiTL 중복 제거 계획서)

## 1. 대상 이슈 및 위치

- 대상 이슈: [Dental_012#4](https://github.com/HyunchanAn/Dental_012/issues/4) (`[P1][브리지] 유치/과잉치 상·하악 판별, BGR/RGB 통일, HiTL 중복 제거 및 README 보완`)
- 상위 Umbrella: [Dental_000#18](https://github.com/HyunchanAn/Dental_000/issues/18) (`Dental_000/plans_replies_walkthrough/261125_1530_Dental_Series_Remediation_Umbrella_Plan.md` Track 2)
- 선행 권장: `Dental_Core#7` (`clinical_status` 및 `lesionSubtype` 스키마 규약 통일)

---

## 2. 즉시 실행 가능 작업 (`[READY]`)

### 2.1 `src/inference.py` 상·하악 판별 확장 및 컬러 스페이스 통일
- 대상 라인: `63`, `196-197`, `232-234`:
  - 상/하악 판별 로직을 영구치 범위(`11..28` / `31..48`)에서 `Dental_Core` 규약(`quadrant in (1, 2, 5, 6)` -> 상악, `(3, 4, 7, 8)` -> 하악)으로 확장하여 유치(`51..85`) 및 과잉치(`19/29/39/49`)의 치근단 탐색 방향 오분류 수정.
- 대상 라인: `86`, `89`:
  - `Dental_Core`/`Dental_008`의 OpenCV BGR 표준과 어긋나는 `COLOR_RGB2LAB`/`COLOR_LAB2RGB` 호출을 BGR 기준으로 명시 통일(또는 입력 포맷 정규화).

### 2.2 HiTL 교정 병변 중복 적재 방지
- 대상 라인: `src/inference.py:340-359`
- 작업 내용:
  - `clinician_corrected` 병변을 보존할 때 동일한 FDI/좌표의 AI 탐지 결과가 중복 `append`되지 않도록 IoU 또는 `(fdi, category)` 기반 중복 제거 로직 구현 및 단위 테스트 추가.

### 2.3 `export_onnx.py` 및 `README.md` 정비
- `export_onnx.py:6-8`: 템플릿 `REPO_ID`(`HyunchanAn/Dental_012`) 및 경로를 실제 `chemahc94/...` 네임스페이스와 모델 경로에 맞게 정비.
- `README.md`: L30 미존재 `Dental_000` 리포트 경로 정정 및 Python 배지/CI 버전 정합화.

---

## 3. `[FREEZE]` 보류 항목 (치과의사 임상 판단 참고)

| 항목 | 상태 | 동결 사유 및 치과의사(안현찬) 역할 |
| :--- | :--- | :--- |
| 유치(`51..85`) 치근 흡수기 및 과잉치(`19/29/39/49`) 치근단 병소 임상 귀속 정책 | `[FREEZE / 임상 판단 필요]` | 코드상 상·하악 방향 판별 오류 수정은 즉시 진행 가능하나, 혼합치열기 유치의 생리적 치근 흡수를 병소로 오판하지 않기 위한 임상 예외 규칙 및 `clinical_criteria_v1.json` 업데이트 여부는 치과의사(안현찬)가 판단해야 함. |

---

## 4. 검증 기준

- 유치(`#51~#85`) 및 과잉치(`#19/#29/#39/#49`) 상·하악 판별, BGR/LAB 변환, HiTL 중복 방지 단위 테스트 통과 (`pytest tests -v`).
- 원격 GitHub Actions CI 통과 확인.
