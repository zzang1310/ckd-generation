# config 폴더 개요

v11 프롬프트 생성에서 사용하는 **공통 설정·정의**를 모아둔 패키지입니다.  
이전 버전(v8 등)에서는 이 내용이 각 스크립트에 흩어져 있었고, v11부터 `config/`로 분리

---

## 폴더 구조

```
config/
├── __init__.py           # 패키지 초기화
├── feature_definitions.py # Lab/Vital itemid, CHART_FEATURES 등
├── comorbidity_definitions.py  # 동반질환, CCI, 체중, RRT, 시술, VP/수액 등
└── README.md             # 이 문서
```

---

## 1. feature_definitions.py

**역할**: MIMIC-IV itemid 매핑 및 피처 분류

### 주요 내용

| 상수/함수 | 설명 |
|-----------|------|
| `FEATURE_ITEMIDS` | 피처명 → itemid 리스트 (creatinine, bun, sbp, temperature, weight 등) |
| `CHART_FEATURES` | chartevents 기반 피처 (sbp, heart_rate, temperature, resp_rate, spo2, map, **weight**) |
| `BODY_METRIC_FEATURES` | 체중, 신장 |
| `LAB_FEATURES` | Lab 피처 그룹 (kidney_core, electrolytes, metabolism 등) |
| `VITAL_FEATURES` | Vital 피처 그룹 (icu_core, body_metrics) |
| `get_weight_itemids()` | 체중 itemid 반환 |
| `get_egfr_itemids()` | eGFR itemid 반환 |

### v11 관련 변경

- **weight**를 `CHART_FEATURES`에 추가 → Vitals 시계열에 일별 체중 포함
- 체중 itemid: `224639`, `226512`, `226531` (226531은 lbs 단위, 변환 필요)

---

## 2. comorbidity_definitions.py

**역할**: 환자 배경 정보 추출·포맷팅 (동반질환, CCI, 체중, RRT, 시술, 승압제, 수액 등)

### 2-1. 동반질환·CCI

| 함수/상수 | 설명 |
|-----------|------|
| `COMORBIDITY_ICD_MAP` | ICD-9/10 코드 → 당뇨, 고혈압, 심부전, COPD 등 매핑 |
| `extract_comorbidities()` | 진단 코드에서 동반질환 추출 |
| `compute_cci()` | Charlson Comorbidity Index 계산 |
| `extract_ckd_stage()` | ICD 기반 CKD Stage 추출 |
| `build_patient_background()` | 배경 정보 통합 (demographics, CCI, comorbidities, admission 등) |
| `format_background_for_prompt()` | 배경 정보를 프롬프트용 텍스트로 변환 |

### 2-2. 체중

| 함수 | 설명 |
|------|------|
| `extract_weight_info()` | 최근 체중 + lookback 기간 내 변화량 (단일 스냅샷) |
| `format_weight_for_prompt()` | 체중 정보 포맷 (현재는 Vitals 시계열로 대체되어 사용 안 함) |
| `extract_weight_trajectory()` | **입원별 장기 체중 궤적** (Kidney history 스타일) |
| `format_weight_trajectory()` | `82->78->75->65 kg (losing)` 형식 |

**단위 변환**: itemid 226531은 lbs → `×0.453592`로 kg 변환

### 2-3. eGFR 궤적

| 함수 | 설명 |
|------|------|
| `extract_egfr_trajectory()` | 입원별 첫 48h Cr → CKD-EPI 2021 eGFR 궤적 |
| `format_egfr_trajectory()` | `eGFR 90->49->39 (declining), Stage 3b` 형식 |
| `compute_egfr_ckd_epi_2021()` | Cr → eGFR 계산 |

### 2-4. RRT·시술·조영제

| 함수 | 설명 |
|------|------|
| `extract_rrt_status()` | 투석/CRRT 이력 (HD, PD, CRRT) |
| `extract_major_procedures()` | 심장수술, 요로시술, 기계환기(주석) 등 |
| `format_major_procedures()` | 시술명 + 시점 정보 포맷 |
| `extract_contrast_exposure()` | 조영제 노출 여부 |

### 2-5. 승압제·수액

| 함수 | 설명 |
|------|------|
| `extract_vasopressor_status()` | input_events에서 승압제 사용 여부·종류·시점 |
| `extract_fluid_status()` | IV 수액 투여량·시점 |

### 2-6. 약물 키워드

| 상수 | 설명 |
|------|------|
| `RX_KEYWORDS` | EMAR/Input/Prescriptions에서 매칭할 약물 키워드 (insulin, furosemide, ACE/ARB, NSAIDs, SGLT2i 등) |

---

## 3. 사용처

| 사용 스크립트 | config 사용 |
|---------------|-------------|
| `build_prompt_dataset_v11_Prognosis.py` | `feature_definitions`, `comorbidity_definitions` 전부 |
| `extract_with_egfr_full_fast.py` | `feature_definitions` (LAB_FEATURES, VITAL_FEATURES) |

---

## 4. 기존 대비 변경 요약

- **config 폴더 신설**: itemid·동반질환·추출 로직을 한 곳에서 관리
- **동반질환·CCI**: ICD 기반 추출, CCI와 comorbidities 별도 표기
- **체중**: 장기 궤적 + Vitals 시계열, lbs→kg 변환
- **체온**: °F→°C 변환 (itemid 223761)
- **승압제·수액·시술**: input_events/procedures_icd 기반 추출
- **약물**: input_events에 d_items.label 병합 후 키워드 매칭

---

## 5. 전체 파이프라인 (실행 순서)

**1. 신장질환 코호트 추출** → `kidney_cohort`
```bash
python generate_kidney_cohort.py --mimic-path data/mimic-iv-3.1 --output-path processed_data/intermediate
```

**2. 멀티모달 pkl 생성** → `full_multimodal_kidney_cohort_with_egfr_complete_fast.pkl`
```bash
python extract_with_egfr_full_fast.py
```

**3. 환자 선별** → `patient_selection/patient_selection.pkl`
```bash
python prepare_patient_selection.py --processed-data-path processed_data --data-file intermediate/full_multimodal_kidney_cohort_with_egfr_complete_fast.pkl --output processed_data/patient_selection/patient_selection.pkl
```

**4. 프롬프트 생성** → `CKD_llm_prompts_*.json`
```bash
python build_prompt_dataset_v11_Prognosis.py --processed-data-path processed_data --data-file intermediate/full_multimodal_kidney_cohort_with_egfr_complete_fast.pkl --patient-selection patient_selection/patient_selection.pkl --max-patients 1000 --tasks egfr,aki
```

**5. LoRA 파인튜닝** → 학습 체크포인트
```bash
cd Me-LLaMA && bash finetune.sh
```
※ `finetune.sh` 내 `DATA_FILE` 등 변수 수정 후 실행

---

## 6. 의존성

- `pandas`, `numpy`
- `build_prompt_dataset_v11_Prognosis.py`에서 `from config.comorbidity_definitions import ...` 형태로 import
