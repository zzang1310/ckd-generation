"""동반질환(Comorbidity) 정의 및 계산 모듈.

MIMIC-IV의 ICD-9/ICD-10 진단 코드로부터:
1. 주요 동반질환 목록 추출
2. Charlson Comorbidity Index (CCI) 계산
3. CKD Stage 추출
4. 입원 맥락 정보 구성

사용 예시:
    from config.comorbidity_definitions import (
        extract_comorbidities,
        compute_cci,
        extract_ckd_stage,
        build_patient_background,
    )

    bg = build_patient_background(
        patient_id=12345,
        diagnoses_df=diagnoses_icd,
        admissions_df=admissions,
        target_time=pd.Timestamp("2200-03-15 08:00"),
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any
import math
import pandas as pd
import numpy as np

# ============================================================
# 1. 주요 동반질환 ICD 코드 매핑
#    - 신장질환 환자에게 임상적으로 의미 있는 동반질환
#    - ICD-9/ICD-10 모두 지원
# ============================================================


# condition_type 분류:
#   "chronic"  — 한번 진단되면 지속 (당뇨, 고혈압, COPD 등)
#   "acute"    — 에피소드성, 해당 입원에서만 의미 (패혈증 등)
#   "variable" — 상태 변화 가능, 재진단 여부가 의미 있음 (비만, 빈혈, 악성종양 등)

COMORBIDITY_ICD_MAP: Dict[str, Dict[str, Any]] = {
    "diabetes": {
        "label_en": "Diabetes Mellitus",
        "label_ko": "당뇨병",
        "condition_type": "chronic",
        "icd10": ["E10", "E11", "E12", "E13", "E14"],
        "icd9": ["250"],
        "clinical_note": "CKD 최대 원인질환, 혈당 조절과 신기능 직접 연관",
    },
    "diabetes_with_complications": {
        "label_en": "Diabetes with complications",
        "label_ko": "합병증 동반 당뇨병",
        "condition_type": "chronic",
        "icd10": ["E102", "E103", "E104", "E105", "E106", "E107",
                   "E112", "E113", "E114", "E115", "E116", "E117"],
        "icd9": ["2502", "2503", "2504", "2505", "2506", "2507"],
        "clinical_note": "당뇨합병증(신증, 망막증 등) 동반 → 더 공격적 신손상 진행",
    },
    "hypertension": {
        "label_en": "Hypertension",
        "label_ko": "고혈압",
        "condition_type": "chronic",
        "icd10": ["I10", "I11", "I12", "I13", "I15"],
        "icd9": ["401", "402", "403", "404", "405"],
        "clinical_note": "CKD 2대 원인, 사구체 내 압력 상승 → 신기능 악화",
    },
    "heart_failure": {
        "label_en": "Heart Failure",
        "label_ko": "심부전",
        "condition_type": "chronic",
        "icd10": ["I50"],
        "icd9": ["428"],
        "clinical_note": "심신증후군(cardiorenal syndrome), 신장 관류 저하",
    },
    "coronary_artery_disease": {
        "label_en": "Coronary Artery Disease",
        "label_ko": "관상동맥질환",
        "condition_type": "chronic",
        "icd10": ["I20", "I21", "I22", "I23", "I24", "I25"],
        "icd9": ["410", "411", "412", "413", "414"],
        "clinical_note": "CKD 환자의 심혈관 사망 위험 증가",
    },
    "atrial_fibrillation": {
        "label_en": "Atrial Fibrillation",
        "label_ko": "심방세동",
        "condition_type": "chronic",
        "icd10": ["I48"],
        "icd9": ["4273"],
        "clinical_note": "항응고제 사용 → 출혈/조영제 노출 위험, 혈역학 불안정",
    },
    "peripheral_vascular_disease": {
        "label_en": "Peripheral Vascular Disease",
        "label_ko": "말초혈관질환",
        "condition_type": "chronic",
        "icd10": ["I70", "I71", "I73", "I77", "I79"],
        "icd9": ["440", "441", "443", "447", "557"],
        "clinical_note": "혈관접근로(투석) 합병증 위험, 전신 동맥경화 지표",
    },
    "cerebrovascular_disease": {
        "label_en": "Cerebrovascular Disease",
        "label_ko": "뇌혈관질환",
        "condition_type": "chronic",
        "icd10": ["I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"],
        "icd9": ["430", "431", "432", "433", "434", "435", "436", "437", "438"],
        "clinical_note": "CKD 환자 뇌졸중 고위험군",
    },
    "copd": {
        "label_en": "COPD",
        "label_ko": "만성폐쇄성폐질환",
        "condition_type": "chronic",
        "icd10": ["J44"],
        "icd9": ["496"],
        "clinical_note": "저산소증 → 신관류 저하, 이뇨제 반응 변화",
    },
    "liver_disease": {
        "label_en": "Liver Disease",
        "label_ko": "간질환",
        "condition_type": "chronic",
        "icd10": ["K70", "K71", "K72", "K73", "K74", "K75", "K76"],
        "icd9": ["571"],
        "clinical_note": "간신증후군(hepatorenal syndrome) 위험",
    },
    "liver_disease_severe": {
        "label_en": "Severe Liver Disease",
        "label_ko": "중증 간질환",
        "condition_type": "chronic",
        "icd10": ["K721", "K729", "K766", "K767"],
        "icd9": ["5722", "5723", "5724", "5728"],
        "clinical_note": "간경변 + 복수 → 간신증후군 고위험",
    },
    "malignancy": {
        "label_en": "Malignancy",
        "label_ko": "악성종양",
        "condition_type": "variable",
        "icd10": ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
        "icd9": ["14", "15", "16", "17", "18", "19", "20"],
        "clinical_note": "화학요법 신독성, 종양용해증후군 → AKI. 관해 가능",
    },
    "sepsis": {
        "label_en": "Sepsis",
        "label_ko": "패혈증",
        "condition_type": "acute",
        "icd10": ["A40", "A41", "R652"],
        "icd9": ["038", "9959"],
        "clinical_note": "AKI의 가장 흔한 원인(ICU), 에피소드성",
    },
    "obesity": {
        "label_en": "Obesity",
        "label_ko": "비만",
        "condition_type": "variable",
        "icd10": ["E66"],
        "icd9": ["2780"],
        "clinical_note": "비만 관련 사구체병증, 체중 변화에 따라 상태 변동",
    },
    "anemia": {
        "label_en": "Anemia",
        "label_ko": "빈혈",
        "condition_type": "variable",
        "icd10": ["D50", "D51", "D52", "D53", "D63", "D64"],
        "icd9": ["280", "281", "282", "283", "284", "285"],
        "clinical_note": "CKD 빈혈(EPO 부족), 치료에 따라 호전/악화 가능",
    },
}


# ============================================================
# 2. Charlson Comorbidity Index (CCI) 정의
#    - Quan et al. (2005) ICD-10 매핑 기반
#    - 가중치: 원본 Charlson (1987) 기준
# ============================================================

CCI_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "myocardial_infarction": {
        "weight": 1,
        "icd10": ["I21", "I22", "I252"],
        "icd9": ["410", "412"],
    },
    "congestive_heart_failure": {
        "weight": 1,
        "icd10": ["I099", "I110", "I130", "I132", "I255", "I420",
                   "I425", "I426", "I427", "I428", "I429", "I43", "I50", "P290"],
        "icd9": ["39891", "40201", "40211", "40291", "40401", "40403",
                 "40411", "40413", "40491", "40493", "4254", "4255",
                 "4257", "4258", "4259", "428"],
    },
    "peripheral_vascular_disease": {
        "weight": 1,
        "icd10": ["I70", "I71", "I731", "I738", "I739", "I771",
                   "I790", "I792", "K551", "K558", "K559", "Z958", "Z959"],
        "icd9": ["0930", "4373", "440", "441", "4431", "4432",
                 "4438", "4439", "4471", "5571", "5579", "V434"],
    },
    "cerebrovascular_disease": {
        "weight": 1,
        "icd10": ["G45", "G46", "I60", "I61", "I62", "I63",
                   "I64", "I65", "I66", "I67", "I68", "I69", "H340"],
        "icd9": ["36234", "430", "431", "432", "433", "434", "435", "436", "437", "438"],
    },
    "dementia": {
        "weight": 1,
        "icd10": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
        "icd9": ["290", "2941", "3312"],
    },
    "chronic_pulmonary_disease": {
        "weight": 1,
        "icd10": ["I278", "I279", "J40", "J41", "J42", "J43",
                   "J44", "J45", "J46", "J47", "J60", "J61",
                   "J62", "J63", "J64", "J65", "J66", "J67",
                   "J684", "J701", "J703"],
        "icd9": ["4168", "4169", "490", "491", "492", "493",
                 "494", "495", "496", "500", "501", "502",
                 "503", "504", "505", "5064", "5081", "5088"],
    },
    "rheumatic_disease": {
        "weight": 1,
        "icd10": ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"],
        "icd9": ["4465", "7100", "7101", "7102", "7103", "7104", "7140", "7141", "7142", "7148", "725"],
    },
    "peptic_ulcer_disease": {
        "weight": 1,
        "icd10": ["K25", "K26", "K27", "K28"],
        "icd9": ["531", "532", "533", "534"],
    },
    "mild_liver_disease": {
        "weight": 1,
        "icd10": ["B18", "K700", "K701", "K702", "K703", "K709",
                   "K713", "K714", "K715", "K717", "K73", "K74",
                   "K760", "K762", "K763", "K764", "K768", "K769", "Z944"],
        "icd9": ["07022", "07023", "07032", "07033", "07044", "07054",
                 "0706", "0709", "570", "571", "5733", "5734", "5738", "5739", "V427"],
    },
    "diabetes_without_complications": {
        "weight": 1,
        "icd10": ["E100", "E101", "E106", "E108", "E109",
                   "E110", "E111", "E116", "E118", "E119",
                   "E120", "E121", "E126", "E128", "E129",
                   "E130", "E131", "E136", "E138", "E139",
                   "E140", "E141", "E146", "E148", "E149"],
        "icd9": ["2500", "2501", "2502", "2503", "2508", "2509"],
    },
    "diabetes_with_complications": {
        "weight": 2,
        "icd10": ["E102", "E103", "E104", "E105", "E107",
                   "E112", "E113", "E114", "E115", "E117",
                   "E122", "E123", "E124", "E125", "E127",
                   "E132", "E133", "E134", "E135", "E137",
                   "E142", "E143", "E144", "E145", "E147"],
        "icd9": ["2504", "2505", "2506", "2507"],
    },
    "hemiplegia_or_paraplegia": {
        "weight": 2,
        "icd10": ["G041", "G114", "G801", "G802", "G81", "G82",
                   "G830", "G831", "G832", "G833", "G834", "G839"],
        "icd9": ["3341", "342", "343", "3440", "3441", "3442",
                 "3443", "3444", "3445", "3446", "3449"],
    },
    "renal_disease": {
        "weight": 2,
        "icd10": ["I120", "I131", "N032", "N033", "N034", "N035",
                   "N036", "N037", "N052", "N053", "N054", "N055",
                   "N056", "N057", "N18", "N19", "N250", "Z490",
                   "Z491", "Z492", "Z940", "Z992"],
        "icd9": ["40301", "40311", "40391", "40402", "40403", "40412",
                 "40413", "40492", "40493", "582", "5830", "5831",
                 "5832", "5834", "5836", "5837", "585", "586", "5880", "V420", "V451", "V56"],
    },
    "malignancy": {
        "weight": 2,
        "icd10": ["C0", "C1", "C2", "C3", "C40", "C41", "C43",
                   "C45", "C46", "C47", "C48", "C49", "C5",
                   "C6", "C70", "C71", "C72", "C73", "C74",
                   "C75", "C76", "C81", "C82", "C83", "C84",
                   "C85", "C88", "C90", "C91", "C92", "C93",
                   "C94", "C95", "C96", "C97"],
        "icd9": ["14", "15", "16", "17", "18", "19", "200", "201",
                 "202", "203", "204", "205", "206", "207", "208", "2386"],
    },
    "moderate_or_severe_liver_disease": {
        "weight": 3,
        "icd10": ["I850", "I859", "I864", "I982", "K704",
                   "K711", "K721", "K729", "K765", "K766", "K767"],
        "icd9": ["4560", "4561", "4562", "5722", "5723", "5724", "5728"],
    },
    "metastatic_solid_tumor": {
        "weight": 6,
        "icd10": ["C77", "C78", "C79", "C80"],
        "icd9": ["196", "197", "198", "199"],
    },
    "aids_hiv": {
        "weight": 6,
        "icd10": ["B20", "B21", "B22", "B24"],
        "icd9": ["042", "043", "044"],
    },
}


# ============================================================
# 3. CKD Stage ICD 코드 매핑
# ============================================================

CKD_STAGE_MAP: Dict[str, Dict[str, List[str]]] = {
    "Stage 1": {"icd10": ["N181"], "icd9": ["5851"]},
    "Stage 2": {"icd10": ["N182"], "icd9": ["5852"]},
    "Stage 3": {"icd10": ["N183"], "icd9": ["5853"]},
    "Stage 3a": {"icd10": ["N1830"], "icd9": []},
    "Stage 3b": {"icd10": ["N1831"], "icd9": []},
    "Stage 4": {"icd10": ["N184"], "icd9": ["5854"]},
    "Stage 5": {"icd10": ["N185"], "icd9": ["5855"]},
    "ESRD": {"icd10": ["N186"], "icd9": ["5856"]},
    "Unspecified": {"icd10": ["N18", "N189"], "icd9": ["585", "5859"]},
}


# ============================================================
# 4. 핵심 함수들
# ============================================================

def _code_matches(icd_code: str, prefix_list: List[str]) -> bool:
    """ICD 코드가 prefix 목록 중 하나와 매칭되는지 확인."""
    code = str(icd_code).strip().upper().replace(".", "")
    return any(code.startswith(p.upper().replace(".", "")) for p in prefix_list)


def _get_hadm_ids_up_to(
    patient_id: int,
    admissions_df: Optional[pd.DataFrame],
    target_time: Optional[pd.Timestamp],
) -> Optional[Set[int]]:
    """target_time 이전 입원의 hadm_id 집합을 반환. None이면 필터 없음."""
    if admissions_df is None or target_time is None:
        return None
    if admissions_df.empty or "admittime" not in admissions_df.columns:
        return None
    pt_adm = admissions_df[admissions_df["subject_id"] == patient_id].copy()
    if pt_adm.empty:
        return None
    pt_adm["admittime"] = pd.to_datetime(pt_adm["admittime"], errors="coerce")
    valid = pt_adm[pt_adm["admittime"] <= target_time]
    if valid.empty:
        return None
    return set(valid["hadm_id"].dropna().astype(int))


def extract_comorbidities(
    patient_id: int,
    diagnoses_df: pd.DataFrame,
    admissions_df: Optional[pd.DataFrame] = None,
    target_time: Optional[pd.Timestamp] = None,
) -> Dict[str, Dict[str, Any]]:
    """환자의 ICD 코드로부터 주요 동반질환을 추출 (시간 인식, 입원별 이력).

    condition_type에 따른 해석:
    - chronic: 한번 진단 → 지속. first_seen 이후 계속 존재로 간주.
    - acute: 에피소드성. 해당 입원에서만 활성. last_seen 입원이 현재가 아니면 "이력"
    - variable: 재진단 여부가 의미. last_seen이 현재 입원이 아니면 상태 불확실

    Args:
        patient_id: 환자 ID
        diagnoses_df: MIMIC-IV diagnoses_icd DataFrame
        admissions_df: admissions DataFrame (시간 필터링 + 입원별 이력)
        target_time: 프롬프트 기준 시점 (None이면 전체 이력)

    Returns:
        Dict[str, Dict]: 각 동반질환의 상세 정보
        {
            "diabetes": {
                "present": True,
                "condition_type": "chronic",
                "first_seen": "01-01",   # MM-DD (비식별 연도 생략)
                "last_seen": "06-15",
                "in_current_adm": True,  # 현재 입원에서도 진단됨
                "n_admissions": 3,       # 총 몇 번 입원에서 진단
                "status": "active",      # active / historical / resolved
            },
            ...
        }
    """
    pt_diag = diagnoses_df[diagnoses_df["subject_id"] == patient_id]

    # 입원별 시간 정보 구성
    adm_time_map: Dict[int, pd.Timestamp] = {}
    current_hadm_id: Optional[int] = None

    if admissions_df is not None and not admissions_df.empty:
        pt_adm = admissions_df[admissions_df["subject_id"] == patient_id].copy()
        if not pt_adm.empty and "admittime" in pt_adm.columns:
            pt_adm["admittime"] = pd.to_datetime(pt_adm["admittime"], errors="coerce")
            if "dischtime" in pt_adm.columns:
                pt_adm["dischtime"] = pd.to_datetime(pt_adm["dischtime"], errors="coerce")

            if target_time is not None:
                pt_adm = pt_adm[pt_adm["admittime"] <= target_time]

            for _, row in pt_adm.iterrows():
                hid = row.get("hadm_id")
                if pd.notna(hid):
                    adm_time_map[int(hid)] = row["admittime"]

            if target_time is not None and not pt_adm.empty:
                current_mask = (
                    (pt_adm["admittime"] <= target_time)
                    & (pt_adm["dischtime"].isna() | (pt_adm["dischtime"] >= target_time))
                )
                current_rows = pt_adm[current_mask]
                if not current_rows.empty:
                    current_hadm_id = int(current_rows.iloc[-1]["hadm_id"])
                else:
                    latest = pt_adm.sort_values("admittime").iloc[-1]
                    current_hadm_id = int(latest["hadm_id"])

    # hadm_id 필터
    valid_hadms = set(adm_time_map.keys()) if adm_time_map else None
    if valid_hadms is not None:
        pt_diag = pt_diag[pt_diag["hadm_id"].isin(valid_hadms)]

    if pt_diag.empty:
        return {
            name: {"present": False, "condition_type": info.get("condition_type", "chronic")}
            for name, info in COMORBIDITY_ICD_MAP.items()
        }

    result: Dict[str, Dict[str, Any]] = {}

    for name, info in COMORBIDITY_ICD_MAP.items():
        all_prefixes = info.get("icd10", []) + info.get("icd9", [])
        cond_type = info.get("condition_type", "chronic")

        matched_hadms: List[int] = []
        for hadm_id in pt_diag["hadm_id"].unique():
            adm_codes = set(
                pt_diag.loc[pt_diag["hadm_id"] == hadm_id, "icd_code"]
                .astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
            )
            if any(_code_matches(c, all_prefixes) for c in adm_codes):
                matched_hadms.append(int(hadm_id))

        if not matched_hadms:
            result[name] = {"present": False, "condition_type": cond_type}
            continue

        matched_hadms_sorted = sorted(matched_hadms, key=lambda h: adm_time_map.get(h, pd.Timestamp.min))
        first_hadm = matched_hadms_sorted[0]
        last_hadm = matched_hadms_sorted[-1]
        first_time = adm_time_map.get(first_hadm)
        last_time = adm_time_map.get(last_hadm)
        in_current = current_hadm_id is not None and current_hadm_id in matched_hadms

        # status 결정
        if cond_type == "chronic":
            status = "active"
        elif cond_type == "acute":
            status = "active" if in_current else "historical"
        else:  # variable
            status = "active" if in_current else "last seen"

        entry: Dict[str, Any] = {
            "present": True,
            "condition_type": cond_type,
            "first_seen": first_time.strftime("%m-%d") if pd.notna(first_time) else "",
            "last_seen": last_time.strftime("%m-%d") if pd.notna(last_time) else "",
            "in_current_adm": in_current,
            "n_admissions": len(matched_hadms),
            "status": status,
        }
        result[name] = entry

    return result


def compute_cci(
    patient_id: int,
    diagnoses_df: pd.DataFrame,
    age: Optional[int] = None,
    admissions_df: Optional[pd.DataFrame] = None,
    target_time: Optional[pd.Timestamp] = None,
) -> Tuple[int, Dict[str, bool]]:
    """Charlson Comorbidity Index를 계산 (시간 인식).

    Quan et al. (2005) ICD-10 매핑, 원본 Charlson (1987) 가중치.
    target_time까지의 입원 진단만 사용.

    Args:
        patient_id: 환자 ID
        diagnoses_df: MIMIC-IV diagnoses_icd DataFrame
        age: 환자 나이 (있으면 연령 보정 적용: 50세 이상 10년당 +1점)
        admissions_df: admissions DataFrame (시간 필터링용)
        target_time: 프롬프트 기준 시점

    Returns:
        (total_score, component_dict) — 총점과 각 항목별 해당 여부
    """
    mask = diagnoses_df["subject_id"] == patient_id

    valid_hadms = _get_hadm_ids_up_to(patient_id, admissions_df, target_time)
    if valid_hadms is not None:
        mask = mask & diagnoses_df["hadm_id"].isin(valid_hadms)

    patient_codes = set(
        diagnoses_df.loc[mask, "icd_code"]
        .astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
    )

    total = 0
    components: Dict[str, bool] = {}
    for name, info in CCI_COMPONENTS.items():
        all_prefixes = info.get("icd10", []) + info.get("icd9", [])
        present = any(_code_matches(code, all_prefixes) for code in patient_codes)
        components[name] = present
        if present:
            total += info["weight"]

    # 당뇨: 합병증 동반 시 합병증만 카운트 (중복 방지)
    if components.get("diabetes_with_complications") and components.get("diabetes_without_complications"):
        total -= CCI_COMPONENTS["diabetes_without_complications"]["weight"]
        components["diabetes_without_complications"] = False

    # 간질환: 중증 시 경증은 제외 (중복 방지)
    if components.get("moderate_or_severe_liver_disease") and components.get("mild_liver_disease"):
        total -= CCI_COMPONENTS["mild_liver_disease"]["weight"]
        components["mild_liver_disease"] = False

    # 종양: 전이 시 일반 종양은 제외 (중복 방지)
    if components.get("metastatic_solid_tumor") and components.get("malignancy"):
        total -= CCI_COMPONENTS["malignancy"]["weight"]
        components["malignancy"] = False

    # 연령 보정 (선택)
    if age is not None and age >= 50:
        age_score = (age - 40) // 10
        total += age_score

    return total, components


CKD_STAGE_SEVERITY: Dict[str, int] = {
    "Stage 1": 1, "Stage 2": 2, "Stage 3": 3, "Stage 3a": 3,
    "Stage 3b": 4, "Stage 4": 5, "Stage 5": 6, "ESRD": 7, "Unspecified": 0,
}


def _match_ckd_stage(icd_codes: Set[str]) -> Optional[str]:
    """ICD 코드 집합에서 가장 심한 CKD Stage를 반환."""
    severity_order = [
        "ESRD", "Stage 5", "Stage 4", "Stage 3b", "Stage 3a",
        "Stage 3", "Stage 2", "Stage 1", "Unspecified",
    ]
    for stage in severity_order:
        prefixes = CKD_STAGE_MAP[stage]["icd10"] + CKD_STAGE_MAP[stage]["icd9"]
        if any(_code_matches(code, prefixes) for code in icd_codes):
            return stage
    return None


def extract_ckd_stage(
    patient_id: int,
    diagnoses_df: pd.DataFrame,
    admissions_df: Optional[pd.DataFrame] = None,
    target_time: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """시간 인식(time-aware) CKD Stage 추출.

    임상의사 관점: 현재 시점까지 알 수 있는 진단만 사용하며,
    과거→현재의 stage 변화 이력(trajectory)도 함께 반환.

    Args:
        patient_id: 환자 ID
        diagnoses_df: MIMIC-IV diagnoses_icd DataFrame
        admissions_df: admissions DataFrame (시간 필터링용, None이면 전체 사용)
        target_time: 프롬프트의 기준 시점 (None이면 전체 이력)

    Returns:
        {
            "current_stage": "Stage 3",         # 가장 최근 입원에서의 stage
            "highest_stage": "Stage 3",         # target_time까지의 최고 stage
            "trajectory": "Stage 2->3",         # 변화 이력 (compact 표기)
            "first_stage": "Stage 2",           # 최초 진단 stage
            "stage_transitions": [              # 단계 전환 이력 (기간 포함)
                {"stage": "Stage 2", "start": "2200-01-01",
                 "duration_months": 5, "is_current": False},
                {"stage": "Stage 3", "start": "2200-06-15",
                 "duration_months": 3, "is_current": True},
            ],
            "stage_at_admissions": [             # 입원별 상세 (디버깅/분석용)
                {"hadm_id": 100, "admittime": "2200-01-01", "stage": "Stage 2"},
                {"hadm_id": 200, "admittime": "2200-06-15", "stage": "Stage 3"},
            ]
        }
        CKD 진단 없으면 모든 값이 None/"" 인 dict 반환
    """
    result: Dict[str, Any] = {
        "current_stage": None,
        "highest_stage": None,
        "trajectory": "",
        "first_stage": None,
        "stage_transitions": [],
        "stage_at_admissions": [],
    }

    pt_diag = diagnoses_df[diagnoses_df["subject_id"] == patient_id]
    if pt_diag.empty:
        return result

    # admissions 정보가 있으면 시간순으로 입원별 stage 추출
    if admissions_df is not None and not admissions_df.empty:
        pt_adm = admissions_df[admissions_df["subject_id"] == patient_id].copy()
        if not pt_adm.empty and "admittime" in pt_adm.columns:
            pt_adm["admittime"] = pd.to_datetime(pt_adm["admittime"], errors="coerce")

            # target_time까지의 입원만 필터
            if target_time is not None:
                pt_adm = pt_adm[pt_adm["admittime"] <= target_time]

            if not pt_adm.empty:
                pt_adm = pt_adm.sort_values("admittime")
                stage_history: List[Dict[str, Any]] = []
                seen_stages: List[str] = []

                for _, adm_row in pt_adm.iterrows():
                    hadm_id = adm_row["hadm_id"]
                    adm_codes = set(
                        pt_diag.loc[pt_diag["hadm_id"] == hadm_id, "icd_code"]
                        .astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
                    )
                    stage = _match_ckd_stage(adm_codes)
                    if stage and stage != "Unspecified":
                        entry = {
                            "hadm_id": int(hadm_id),
                            "admittime": str(adm_row["admittime"].date()) if pd.notna(adm_row["admittime"]) else "",
                            "stage": stage,
                        }
                        stage_history.append(entry)
                        if not seen_stages or seen_stages[-1] != stage:
                            seen_stages.append(stage)

                if stage_history:
                    result["stage_at_admissions"] = stage_history
                    result["current_stage"] = stage_history[-1]["stage"]
                    result["first_stage"] = stage_history[0]["stage"]
                    result["highest_stage"] = max(
                        [e["stage"] for e in stage_history],
                        key=lambda s: CKD_STAGE_SEVERITY.get(s, 0),
                    )

                    # --- stage_transitions: 단계 전환 시점 + 기간 ---
                    transitions: List[Dict[str, Any]] = []
                    prev_tr_stage: Optional[str] = None
                    for sh_entry in stage_history:
                        if sh_entry["stage"] != prev_tr_stage:
                            transitions.append({
                                "stage": sh_entry["stage"],
                                "start": sh_entry["admittime"],
                            })
                            prev_tr_stage = sh_entry["stage"]

                    for idx, tr in enumerate(transitions):
                        start_dt = pd.Timestamp(tr["start"]) if tr["start"] else None
                        tr["is_current"] = (idx == len(transitions) - 1)
                        if start_dt is None:
                            tr["duration_months"] = None
                            continue
                        if idx + 1 < len(transitions):
                            next_start = pd.Timestamp(transitions[idx + 1]["start"])
                            end_dt = next_start if pd.notna(next_start) else (target_time or start_dt)
                        else:
                            end_dt = target_time if target_time else start_dt
                        dur_days = (end_dt - start_dt).days
                        tr["duration_months"] = max(int(round(dur_days / 30.44)), 0)

                    result["stage_transitions"] = transitions

                    # trajectory: compact 표기 (Stage 2->3->4)
                    def _stage_short(s: str) -> str:
                        return s.replace("Stage ", "") if s.startswith("Stage") else s

                    if len(seen_stages) > 1:
                        result["trajectory"] = "->".join(_stage_short(s) for s in seen_stages)
                    else:
                        result["trajectory"] = seen_stages[0]
                    return result

    # Fallback: admissions 정보 없이 전체 진단에서 추출
    all_codes = set(
        pt_diag["icd_code"]
        .astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
    )
    stage = _match_ckd_stage(all_codes)
    if stage and stage != "Unspecified":
        result["current_stage"] = stage
        result["highest_stage"] = stage
        result["trajectory"] = stage
        result["first_stage"] = stage

    return result


def get_admission_context(
    patient_id: int,
    admissions_df: pd.DataFrame,
    target_time: pd.Timestamp,
) -> Dict[str, Any]:
    """프롬프트 시점에 해당하는 입원 맥락 정보를 구성.

    Args:
        patient_id: 환자 ID
        admissions_df: MIMIC-IV admissions DataFrame
        target_time: 프롬프트의 기준 시점

    Returns:
        Dict with admission_type, los_days, admission_location 등
    """
    pt_adm = admissions_df[admissions_df["subject_id"] == patient_id].copy()
    if pt_adm.empty:
        return {}

    for col in ["admittime", "dischtime"]:
        if col in pt_adm.columns:
            pt_adm[col] = pd.to_datetime(pt_adm[col], errors="coerce")

    if "admittime" not in pt_adm.columns:
        return {}

    # target_time이 속하는 입원 찾기
    current_adm = pt_adm[
        (pt_adm["admittime"] <= target_time)
        & (
            pt_adm["dischtime"].isna()
            | (pt_adm["dischtime"] >= target_time)
        )
    ]

    if current_adm.empty:
        current_adm = pt_adm[pt_adm["admittime"] <= target_time]
        if current_adm.empty:
            return {}
        current_adm = current_adm.sort_values("admittime").tail(1)

    row = current_adm.iloc[-1]
    ctx: Dict[str, Any] = {}

    hadm_id = row.get("hadm_id")
    if pd.notna(hadm_id):
        ctx["hadm_id"] = int(hadm_id)

    adm_type = row.get("admission_type", "")
    if pd.notna(adm_type) and str(adm_type).strip():
        ctx["admission_type"] = str(adm_type).strip()

    adm_loc = row.get("admission_location", "")
    if pd.notna(adm_loc) and str(adm_loc).strip():
        ctx["admission_location"] = str(adm_loc).strip()

    admit_time = row.get("admittime")
    if pd.notna(admit_time):
        los = (target_time - pd.Timestamp(admit_time)).total_seconds() / 86400.0
        ctx["los_days"] = round(max(los, 0), 1)

    return ctx


# ============================================================
# 5. 통합 함수: 환자 배경 정보 구성 (프롬프트 빌더에서 호출)
# ============================================================

def build_patient_background(
    patient_id: int,
    diagnoses_df: pd.DataFrame,
    admissions_df: Optional[pd.DataFrame] = None,
    target_time: Optional[pd.Timestamp] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    lang: str = "en",
) -> Dict[str, Any]:
    """프롬프트에 삽입할 환자 배경 정보를 종합 구성.

    Returns:
        {
            "demographics": {"age": 65, "gender": "M"},
            "ckd_stage": "Stage 3",
            "cci_score": 5,
            "cci_components": {"diabetes_with_complications": True, ...},
            "comorbidities": {"diabetes": True, "hypertension": True, ...},
            "comorbidity_list": ["Diabetes Mellitus", "Hypertension", "Heart Failure"],
            "admission_context": {"admission_type": "EMERGENCY", "los_days": 3.5, ...},
        }
    """
    result: Dict[str, Any] = {}

    # Demographics
    demo: Dict[str, Any] = {}
    if age is not None:
        demo["age"] = age
    if gender is not None:
        demo["gender"] = gender
    result["demographics"] = demo

    # CKD Stage (시간 인식: target_time까지의 진단만 사용, 변화 이력 포함)
    if admissions_df is not None and target_time is not None:
        adm_ctx = get_admission_context(patient_id, admissions_df, target_time)
        result["admission_context"] = adm_ctx
    else:
        result["admission_context"] = {}

    ckd_info = extract_ckd_stage(
        patient_id, diagnoses_df,
        admissions_df=admissions_df,
        target_time=target_time,
    )
    result["ckd_stage"] = ckd_info.get("current_stage")
    result["ckd_trajectory"] = ckd_info.get("trajectory", "")
    result["ckd_first_stage"] = ckd_info.get("first_stage")
    result["ckd_stage_transitions"] = ckd_info.get("stage_transitions", [])

    # CCI (시간 인식: target_time까지의 진단만 사용)
    cci_score, cci_components = compute_cci(
        patient_id, diagnoses_df, age=age,
        admissions_df=admissions_df, target_time=target_time,
    )
    result["cci_score"] = cci_score
    result["cci_components"] = cci_components

    # Comorbidities (시간 인식: target_time까지의 진단만 사용)
    comorbidities = extract_comorbidities(
        patient_id, diagnoses_df,
        admissions_df=admissions_df, target_time=target_time,
    )
    result["comorbidities"] = comorbidities

    return result


# ============================================================
# 6. 체중 트렌드 유틸리티 (chartevents에서 추출)
# ============================================================

def extract_weight_info(
    chart_events: pd.DataFrame,
    target_time: pd.Timestamp,
    lookback_days: int = 30,
    weight_itemids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """프롬프트 시점 기준으로 체중 정보를 추출.

    Args:
        chart_events: 환자의 chartevents DataFrame (charttime, itemid, valuenum 필수)
        target_time: 프롬프트 기준 시점
        lookback_days: 체중 변화 추적 기간 (기본 30일)
        weight_itemids: 체중 itemid 목록 (None이면 feature_definitions에서 로드)

    Returns:
        {
            "latest_weight_kg": 75.3,
            "latest_weight_time": "2200-03-14",
            "weight_change_kg": -2.1,       # 기간 내 변화량
            "weight_change_period_days": 14, # 변화 측정 기간
            "measurements_count": 5,         # 기간 내 측정 횟수
        }
    """
    if weight_itemids is None:
        from config.feature_definitions import get_weight_itemids
        weight_itemids = get_weight_itemids()

    if chart_events.empty or not weight_itemids:
        return {}

    ce = chart_events.copy()
    if "charttime" in ce.columns:
        ce["charttime"] = pd.to_datetime(ce["charttime"], errors="coerce")
    else:
        return {}

    wt = ce[
        (ce["itemid"].isin(weight_itemids))
        & (ce["valuenum"].notna())
        & (ce["valuenum"] > 0)
        & (ce["charttime"] <= target_time)
    ].copy()

    if wt.empty:
        return {}

    # itemid 226531: 단위 NaN → 실제 lbs 기록 (MIMIC-IV 확인)
    # 같은 환자의 kg 기록 대비 ratio ≈ 2.2 = lbs/kg 변환 계수와 일치
    lbs_mask = wt["itemid"] == 226531
    wt.loc[lbs_mask, "valuenum"] = wt.loc[lbs_mask, "valuenum"] * 0.453592

    # 변환 후 생리학적 범위 필터 (20~300 kg)
    wt = wt[(wt["valuenum"] >= 20) & (wt["valuenum"] <= 300)]
    wt = wt.sort_values("charttime")

    if wt.empty:
        return {}

    # 최근 체중
    latest = wt.iloc[-1]
    result: Dict[str, Any] = {
        "latest_weight_kg": round(float(latest["valuenum"]), 1),
        "latest_weight_time": str(latest["charttime"].date()),
    }

    # lookback 기간 내 변화량
    window_start = target_time - pd.Timedelta(days=lookback_days)
    wt_window = wt[wt["charttime"] >= window_start]

    result["measurements_count"] = len(wt_window)

    if len(wt_window) >= 2:
        first_wt = float(wt_window.iloc[0]["valuenum"])
        last_wt = float(wt_window.iloc[-1]["valuenum"])
        period = (wt_window.iloc[-1]["charttime"] - wt_window.iloc[0]["charttime"]).days
        result["weight_change_kg"] = round(last_wt - first_wt, 1)
        result["weight_change_period_days"] = max(period, 1)

    return result


def format_weight_for_prompt(
    weight_info: Dict[str, Any],
    lang: str = "en",
) -> str:
    """체중 정보를 프롬프트용 텍스트로 변환.

    예시: "Weight: 75.3 kg (Δ -2.1 kg over 14 days)"
    """
    if not weight_info:
        return ""

    wt = weight_info.get("latest_weight_kg")
    if wt is None:
        return ""

    label = "Weight" if lang == "en" else "체중"
    text = f"{label}: {wt} kg"

    change = weight_info.get("weight_change_kg")
    period = weight_info.get("weight_change_period_days")
    if change is not None and period is not None:
        sign = "+" if change > 0 else ""
        day_label = "days" if lang == "en" else "일"
        text += f" (Δ {sign}{change} kg / {period} {day_label})"

    return text


def format_background_for_prompt(
    background: Dict[str, Any],
    lang: str = "en",
) -> str:
    """build_patient_background 결과를 Me-LLaMA 파인튜닝에 적합한 의학 텍스트로 변환.

    표현 원칙:
      - 의학 문헌/임상 노트에서 자연스러운 영어 표현 사용
      - 'dx' (diagnosed), 'resolved', 'last dx' 등 표준 의학 약어 활용
      - CKD Stage는 compact 표기: Stage 2(5mo)->3(6mo)->4(current)
      - 토큰 효율을 위해 불필요한 반복 제거

    예시 (en):
        Patient: 65M
        CKD progression: Stage 2(5mo)->3(6mo)->4(2mo, current)
        Comorbidities (CCI=5): DM (dx 01-01), HTN, CHF; past: Sepsis (resolved 06-15)
        Admission: EMERGENCY | HD 3
    """
    lines: List[str] = []

    # --- Demographics: "Patient: 65M" 스타일 ---
    demo = background.get("demographics", {})
    age = demo.get("age")
    gender = demo.get("gender", "")
    if age is not None or gender:
        age_str = str(age) if age is not None else ""
        gender_str = str(gender).strip().upper()[:1] if gender else ""
        lines.append(f"Patient: {age_str}{gender_str}".strip())

    # --- eGFR trajectory는 뒤쪽 별도 블록에서 출력 (ICD CKD Stage 제거됨) ---

    # --- Comorbidities + CCI: 의학적 표준 표현 ---
    comorbidities = background.get("comorbidities", {})
    cci = background.get("cci_score", 0)
    label_key = "label_en" if lang == "en" else "label_ko"

    active_items: List[str] = []
    past_items: List[str] = []

    for name, detail in comorbidities.items():
        if not isinstance(detail, dict) or not detail.get("present"):
            continue

        display_name = COMORBIDITY_ICD_MAP.get(name, {}).get(label_key, name)
        cond_type = detail.get("condition_type", "chronic")
        status = detail.get("status", "active")
        first_seen = detail.get("first_seen", "")

        if status == "active":
            if cond_type == "chronic" and first_seen:
                active_items.append(f"{display_name} (dx {first_seen})")
            else:
                active_items.append(display_name)
        elif status == "historical":
            last_seen = detail.get("last_seen", "")
            if last_seen:
                past_items.append(f"{display_name} (resolved {last_seen})")
            else:
                past_items.append(f"{display_name} (resolved)")
        else:  # "last seen" — variable condition
            last_seen = detail.get("last_seen", "")
            if last_seen:
                past_items.append(f"{display_name} (last dx {last_seen})")
            else:
                past_items.append(f"{display_name} (last dx unknown)")

    if active_items or past_items:
        header = "Comorbidities" if lang == "en" else "동반질환"
        segments = []
        if active_items:
            segments.append(", ".join(active_items))
        if past_items:
            segments.append(f"past: {', '.join(past_items)}")
        lines.append(f"{header}: {'; '.join(segments)}")

    if cci > 0:
        lines.append(f"CCI: {cci}")

    # --- Admission context ---
    adm = background.get("admission_context", {})
    adm_parts: List[str] = []
    if "admission_type" in adm:
        adm_parts.append(adm["admission_type"])
    if "los_days" in adm:
        adm_parts.append(f"HD {adm['los_days']:.0f}")
    if adm_parts:
        lines.append(f"Admission: {' | '.join(adm_parts)}")

    # --- eGFR trajectory (Cr-derived) ---
    egfr_traj = background.get("egfr_trajectory")
    if egfr_traj:
        traj_text = format_egfr_trajectory(egfr_traj)
        if traj_text:
            lines.append(traj_text)

    # --- Weight trajectory (입원별 장기 체중 추이) ---
    wt_traj = background.get("weight_trajectory")
    if wt_traj:
        wt_traj_text = format_weight_trajectory(wt_traj)
        if wt_traj_text:
            lines.append(wt_traj_text)

    # --- RRT status ---
    rrt = background.get("rrt_status", {})
    if rrt.get("has_rrt"):
        rrt_type = rrt.get("rrt_type", "HD")
        last_date = rrt.get("last_session_date", "")
        n_sessions = rrt.get("total_sessions", 0)
        rrt_parts = [f"RRT: {rrt_type}"]
        if n_sessions > 1:
            rrt_parts.append(f"{n_sessions} sessions")
        if last_date:
            rrt_parts.append(f"last {last_date}")
        lines.append(", ".join(rrt_parts))

    # --- Contrast exposure ---
    contrast = background.get("contrast_exposure", {})
    if contrast.get("exposed"):
        c_date = contrast.get("date", "")
        c_days = contrast.get("days_ago")
        if c_date and c_days is not None:
            lines.append(f"Contrast exposure: {c_date} ({c_days}d ago)")
        elif c_date:
            lines.append(f"Contrast exposure: {c_date}")

    # --- Vasopressor ---
    vp = background.get("vasopressor_status", {})
    if vp.get("has_vasopressor"):
        agents = vp.get("agents", [])
        days = vp.get("days_before")
        agent_str = ", ".join(agents) if agents else "vasopressor"
        if days is not None and days <= 0:
            lines.append(f"Vasopressor: {agent_str} (same day)")
        elif days is not None:
            lines.append(f"Vasopressor: {agent_str} ({days}d before)")
        else:
            lines.append(f"Vasopressor: {agent_str}")

    # --- IV Fluid ---
    fl = background.get("fluid_status", {})
    if fl.get("has_fluid"):
        total_ml = fl.get("total_ml", 0)
        days = fl.get("days_before")
        period = fl.get("period_days", 3)
        vol_str = f"{total_ml:.0f}mL" if total_ml < 10000 else f"{total_ml/1000:.1f}L"
        if days is not None and days <= 0:
            lines.append(f"IV Fluid: {vol_str}/{period}d (ongoing)")
        elif days is not None:
            lines.append(f"IV Fluid: {vol_str}/{period}d (last {days}d before)")
        else:
            lines.append(f"IV Fluid: {vol_str}/{period}d")

    # --- Major procedures (시술 이력) ---
    major_proc = background.get("major_procedures", {})
    if major_proc:
        target_time = background.get("_target_time")
        proc_text = format_major_procedures(major_proc, target_time=target_time, lang=lang)
        if proc_text:
            lines.append(proc_text)

    return "\n".join(lines)


# ============================================================
# 7. Cr-derived eGFR 궤적 (CKD-EPI 2021 Refit)
# ============================================================

def compute_egfr_ckd_epi_2021(
    cr: float, age: int, is_female: bool,
) -> Optional[float]:
    """CKD-EPI 2021 (race-free refit) eGFR 계산.

    Reference: Inker et al., NEJM 2021.
    eGFR = 142 × min(Scr/κ,1)^α × max(Scr/κ,1)^(-1.200) × 0.9938^age × [1.012 if female]
    κ = 0.7 (F) / 0.9 (M), α = -0.241 (F) / -0.302 (M)
    """
    if cr is None or cr <= 0 or age is None or age <= 0:
        return None
    k = 0.7 if is_female else 0.9
    a = -0.241 if is_female else -0.302
    cr_k = cr / k
    t1 = min(cr_k, 1.0) ** a
    t2 = max(cr_k, 1.0) ** (-1.200)
    egfr = 142.0 * t1 * t2 * (0.9938 ** age)
    if is_female:
        egfr *= 1.012
    return egfr


def _egfr_to_ckd_stage(egfr: float) -> str:
    """eGFR → CKD Stage 문자열."""
    if egfr >= 90:
        return "1"
    elif egfr >= 60:
        return "2"
    elif egfr >= 45:
        return "3a"
    elif egfr >= 30:
        return "3b"
    elif egfr >= 15:
        return "4"
    else:
        return "5"


_MAX_TRAJECTORY_POINTS = 8


def extract_egfr_trajectory(
    patient_id: int,
    lab_events: pd.DataFrame,
    admissions_df: pd.DataFrame,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    anchor_age: Optional[int] = None,
    anchor_year: Optional[int] = None,
    target_time: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """입원별 첫 Cr → CKD-EPI 2021 eGFR 궤적을 추출.

    Args:
        patient_id: 환자 ID
        lab_events: 해당 환자의 lab_events (또는 전체)
        admissions_df: admissions DataFrame
        age: 현재 나이 (fallback)
        gender: 'M' or 'F'
        anchor_age, anchor_year: MIMIC de-identification anchors
        target_time: 프롬프트 기준시점 (이 시점까지만)

    Returns:
        {
            "points": [{"date": Timestamp, "egfr": 86, "stage": "2"}, ...],
            "current_egfr": 34,
            "current_stage": "3b",
            "baseline_egfr": 106,
            "baseline_stage": "1",
            "n_admissions": 7,
            "span_months": 153,
            "trend": "declining",  # declining/stable/improving/fluctuating
        }
        데이터 부족 시 빈 dict
    """
    CR_ITEMIDS = {50912, 52546}
    is_female = str(gender).strip().upper().startswith("F")

    result: Dict[str, Any] = {}

    # --- 환자 Cr ---
    if "subject_id" in lab_events.columns:
        pt_cr = lab_events[
            (lab_events["subject_id"] == patient_id)
            & (lab_events["itemid"].isin(CR_ITEMIDS))
        ]
    else:
        pt_cr = lab_events[lab_events["itemid"].isin(CR_ITEMIDS)]
    if pt_cr.empty:
        return result

    pt_cr = pt_cr[pt_cr["valuenum"].notna() & (pt_cr["valuenum"] > 0) & (pt_cr["valuenum"] < 30)].copy()
    if pt_cr.empty:
        return result

    if not pd.api.types.is_datetime64_any_dtype(pt_cr["charttime"]):
        pt_cr["charttime"] = pd.to_datetime(pt_cr["charttime"], errors="coerce")
    pt_cr = pt_cr.dropna(subset=["charttime"])
    if pt_cr.empty:
        return result

    # --- 환자 admissions ---
    pt_adm = admissions_df[admissions_df["subject_id"] == patient_id].copy()
    if pt_adm.empty:
        return result
    if not pd.api.types.is_datetime64_any_dtype(pt_adm.get("admittime")):
        pt_adm["admittime"] = pd.to_datetime(pt_adm["admittime"], errors="coerce")
    pt_adm = pt_adm.dropna(subset=["admittime"])

    if target_time is not None:
        pt_adm = pt_adm[pt_adm["admittime"] <= target_time]
    if pt_adm.empty:
        return result

    pt_adm = pt_adm.sort_values("admittime")

    # --- 입원별 첫 48h Cr → eGFR ---
    points: List[Dict[str, Any]] = []
    for _, adm_row in pt_adm.iterrows():
        admit = adm_row["admittime"]
        first_cr = pt_cr[
            (pt_cr["charttime"] >= admit)
            & (pt_cr["charttime"] <= admit + pd.Timedelta(hours=48))
        ]
        if first_cr.empty:
            continue
        cr_val = float(first_cr.sort_values("charttime").iloc[0]["valuenum"])

        if anchor_age is not None and anchor_year is not None:
            pt_age = int(anchor_age) + (admit.year - int(anchor_year))
        elif age is not None:
            pt_age = int(age)
        else:
            continue

        egfr = compute_egfr_ckd_epi_2021(cr_val, pt_age, is_female)
        if egfr is None:
            continue

        egfr_rounded = int(round(egfr))
        points.append({
            "date": admit,
            "egfr": egfr_rounded,
            "stage": _egfr_to_ckd_stage(egfr_rounded),
        })

    if not points:
        return result

    # --- trend 계산 ---
    egfr_vals = [p["egfr"] for p in points]
    first_egfr = egfr_vals[0]
    last_egfr = egfr_vals[-1]
    delta = last_egfr - first_egfr

    if len(points) == 1:
        trend = "single"
    else:
        diffs = [egfr_vals[i + 1] - egfr_vals[i] for i in range(len(egfr_vals) - 1)]
        n_up = sum(1 for d in diffs if d > 5)
        n_down = sum(1 for d in diffs if d < -5)
        if n_up == 0 and n_down == 0:
            trend = "stable"
        elif n_down > 0 and n_up == 0:
            trend = "declining"
        elif n_up > 0 and n_down == 0:
            trend = "improving"
        elif n_down >= n_up and abs(delta) > 10:
            trend = "declining"
        elif n_up > n_down and delta > 10:
            trend = "improving"
        else:
            trend = "fluctuating"

    span_days = (points[-1]["date"] - points[0]["date"]).days
    span_months = max(int(round(span_days / 30.44)), 0)

    result = {
        "points": points,
        "current_egfr": last_egfr,
        "current_stage": _egfr_to_ckd_stage(last_egfr),
        "baseline_egfr": first_egfr,
        "baseline_stage": _egfr_to_ckd_stage(first_egfr),
        "n_admissions": len(points),
        "span_months": span_months,
        "trend": trend,
    }
    return result


def format_egfr_trajectory(traj: Dict[str, Any]) -> str:
    """extract_egfr_trajectory 결과 → 프롬프트 텍스트.

    Examples:
        Kidney history (7 adm/12yr): eGFR 106->90->63->34 (declining), Stage 3b
        Kidney history: eGFR 58 (stable), Stage 3a
    """
    if not traj or "points" not in traj:
        return ""

    points = traj["points"]
    if not points:
        return ""

    n_adm = traj.get("n_admissions", len(points))
    span_mo = traj.get("span_months", 0)
    trend = traj.get("trend", "")
    current_stage = traj.get("current_stage", "")

    egfr_vals = [p["egfr"] for p in points]

    # 궤적 포인트 압축: > _MAX_TRAJECTORY_POINTS이면 키포인트만
    if len(egfr_vals) > _MAX_TRAJECTORY_POINTS:
        n = len(egfr_vals)
        indices = [0]
        step = (n - 1) / (_MAX_TRAJECTORY_POINTS - 1)
        for i in range(1, _MAX_TRAJECTORY_POINTS - 1):
            indices.append(int(round(i * step)))
        indices.append(n - 1)
        indices = sorted(set(indices))
        display_vals = [egfr_vals[i] for i in indices]
    else:
        display_vals = egfr_vals

    vals_str = "->".join(str(v) for v in display_vals)

    # 기간 표현
    if span_mo >= 12:
        span_str = f"{span_mo // 12}yr"
        remainder = span_mo % 12
        if remainder >= 3:
            span_str += f"{remainder}mo"
    elif span_mo > 0:
        span_str = f"{span_mo}mo"
    else:
        span_str = ""

    # 조립
    if len(points) == 1:
        return f"Kidney history: eGFR {egfr_vals[0]}, Stage {current_stage}"

    header_parts = []
    if n_adm > 1:
        header_parts.append(f"{n_adm} adm")
    if span_str:
        header_parts.append(span_str)
    header = f" ({'/'.join(header_parts)})" if header_parts else ""

    trend_str = f" ({trend})" if trend and trend != "single" else ""

    return f"Kidney history{header}: eGFR {vals_str}{trend_str}, Stage {current_stage}"


# ============================================================
# 7-b. Weight trajectory (입원별 체중 궤적)
# ============================================================

_MAX_WEIGHT_POINTS = 10
_WEIGHT_ITEMIDS = [224639, 226512, 226531]
_LBS_ITEMID = 226531


def extract_weight_trajectory(
    chart_events: pd.DataFrame,
    admissions_df: pd.DataFrame,
    patient_id: int,
    target_time: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """입원별 중앙값 체중으로 장기 체중 궤적을 추출.

    Returns:
        {
            "points": [{"date": Timestamp, "weight_kg": 78.2}, ...],
            "current_kg": 65.0,
            "baseline_kg": 82.0,
            "n_admissions": 7,
            "span_months": 15,
            "trend": "losing",  # losing/gaining/stable/fluctuating
        }
        데이터 부족 시 빈 dict
    """
    if chart_events is None or chart_events.empty:
        return {}

    ce = chart_events.copy()
    if "charttime" not in ce.columns or "valuenum" not in ce.columns:
        return {}
    ce["charttime"] = pd.to_datetime(ce["charttime"], errors="coerce")
    ce = ce.dropna(subset=["charttime", "valuenum"])

    wt = ce[
        (ce["subject_id"] == patient_id)
        & (ce["itemid"].isin(_WEIGHT_ITEMIDS))
        & (ce["valuenum"] > 0)
    ].copy()
    if wt.empty:
        return {}

    # lbs → kg 변환 (itemid 226531)
    lbs_mask = wt["itemid"] == _LBS_ITEMID
    wt.loc[lbs_mask, "valuenum"] = wt.loc[lbs_mask, "valuenum"] * 0.453592
    wt = wt[(wt["valuenum"] >= 20) & (wt["valuenum"] <= 300)]
    if wt.empty:
        return {}

    if target_time is not None:
        wt = wt[wt["charttime"] <= target_time]
    if wt.empty:
        return {}

    # admissions
    pt_adm = admissions_df[admissions_df["subject_id"] == patient_id].copy()
    if pt_adm.empty:
        return {}
    pt_adm["admittime"] = pd.to_datetime(pt_adm["admittime"], errors="coerce")
    if "dischtime" in pt_adm.columns:
        pt_adm["dischtime"] = pd.to_datetime(pt_adm["dischtime"], errors="coerce")
    pt_adm = pt_adm.dropna(subset=["admittime"])
    if target_time is not None:
        pt_adm = pt_adm[pt_adm["admittime"] <= target_time]
    pt_adm = pt_adm.sort_values("admittime")
    if pt_adm.empty:
        return {}

    # 입원별 중앙값 체중
    points: List[Dict[str, Any]] = []
    for _, adm_row in pt_adm.iterrows():
        admit = adm_row["admittime"]
        disch = adm_row.get("dischtime")
        if pd.isna(disch):
            disch = admit + pd.Timedelta(days=30)
        adm_wt = wt[(wt["charttime"] >= admit) & (wt["charttime"] <= disch)]
        if adm_wt.empty:
            continue
        median_kg = round(float(adm_wt["valuenum"].median()), 1)
        points.append({"date": admit, "weight_kg": median_kg})

    if not points:
        return {}

    # trend 판정
    vals = [p["weight_kg"] for p in points]
    if len(vals) >= 3:
        first_third = sum(vals[: len(vals) // 3]) / max(len(vals) // 3, 1)
        last_third = sum(vals[-(len(vals) // 3):]) / max(len(vals) // 3, 1)
        diff_pct = (last_third - first_third) / first_third * 100 if first_third else 0
        if diff_pct < -5:
            trend = "losing"
        elif diff_pct > 5:
            trend = "gaining"
        else:
            std_pct = (max(vals) - min(vals)) / (sum(vals) / len(vals)) * 100 if vals else 0
            trend = "fluctuating" if std_pct > 15 else "stable"
    elif len(vals) == 2:
        diff_pct = (vals[-1] - vals[0]) / vals[0] * 100 if vals[0] else 0
        if diff_pct < -5:
            trend = "losing"
        elif diff_pct > 5:
            trend = "gaining"
        else:
            trend = "stable"
    else:
        trend = "single"

    span_months = 0
    if len(points) >= 2:
        span_days = (points[-1]["date"] - points[0]["date"]).days
        span_months = max(span_days // 30, 1)

    return {
        "points": points,
        "current_kg": vals[-1],
        "baseline_kg": vals[0],
        "n_admissions": len(points),
        "span_months": span_months,
        "trend": trend,
    }


def format_weight_trajectory(traj: Dict[str, Any]) -> str:
    """extract_weight_trajectory 결과 → 프롬프트 텍스트.

    Examples:
        Weight history (5 adm/1yr3mo): 82->78->75->71->65 kg (losing)
        Weight history: 78 kg
    """
    if not traj or "points" not in traj:
        return ""

    points = traj["points"]
    if not points:
        return ""

    n_adm = traj.get("n_admissions", len(points))
    span_mo = traj.get("span_months", 0)
    trend = traj.get("trend", "")
    wt_vals = [p["weight_kg"] for p in points]

    # 궤적 포인트 압축
    if len(wt_vals) > _MAX_WEIGHT_POINTS:
        n = len(wt_vals)
        indices = [0]
        step = (n - 1) / (_MAX_WEIGHT_POINTS - 1)
        for i in range(1, _MAX_WEIGHT_POINTS - 1):
            indices.append(int(round(i * step)))
        indices.append(n - 1)
        indices = sorted(set(indices))
        display_vals = [wt_vals[i] for i in indices]
    else:
        display_vals = wt_vals

    vals_str = "->".join(str(v) for v in display_vals)

    # 기간 표현
    if span_mo >= 12:
        span_str = f"{span_mo // 12}yr"
        remainder = span_mo % 12
        if remainder >= 3:
            span_str += f"{remainder}mo"
    elif span_mo > 0:
        span_str = f"{span_mo}mo"
    else:
        span_str = ""

    if len(points) == 1:
        return f"Weight history: {wt_vals[0]} kg"

    header_parts = []
    if n_adm > 1:
        header_parts.append(f"{n_adm} adm")
    if span_str:
        header_parts.append(span_str)
    header = f" ({'/'.join(header_parts)})" if header_parts else ""

    trend_str = f" ({trend})" if trend and trend != "single" else ""

    return f"Weight history{header}: {vals_str} kg{trend_str}"


# ============================================================
# 8. RRT (Renal Replacement Therapy) 상태 추출
# ============================================================

# ICD codes for dialysis (procedures_icd)
_RRT_ICD_CODES = {
    # ICD-9 Procedure
    "3995": "HD", "5498": "PD",
    # ICD-10 PCS
    "5A1D70Z": "HD", "5A1D60Z": "HD", "5A1D90Z": "HD", "5A1D00Z": "HD",
    "5A1D80Z": "HD",
    "5A1955Z": "CRRT", "5A1945Z": "CRRT",
    "3E1M39Z": "PD", "3E1M38Z": "PD", "3E1M38X": "PD",
}

# Contrast agent procedure codes
_CONTRAST_ICD_CODES = {
    "8854", "8855", "8856", "8857", "8872",
    "B211YZZ", "B215YZZ", "B21GYZZ", "B31FYZZ",
}


def extract_rrt_status(
    patient_id: int,
    procedures_df: pd.DataFrame,
    admissions_df: Optional[pd.DataFrame] = None,
    target_time: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """환자의 RRT(투석/CRRT) 상태 추출.

    Returns:
        {
            "has_rrt": True,
            "rrt_type": "HD",          # HD / CRRT / PD / None
            "last_session_date": "03-15",
            "total_sessions": 5,
        }
    """
    result: Dict[str, Any] = {
        "has_rrt": False,
        "rrt_type": None,
        "last_session_date": None,
        "total_sessions": 0,
    }

    if not isinstance(procedures_df, pd.DataFrame) or procedures_df.empty:
        return result

    pt_proc = procedures_df[procedures_df["subject_id"] == patient_id]
    if pt_proc.empty:
        return result

    code_col = "icd_code" if "icd_code" in pt_proc.columns else None
    if code_col is None:
        return result

    codes = pt_proc[code_col].astype(str).str.strip().str.upper()
    rrt_mask = codes.isin(_RRT_ICD_CODES)
    rrt_proc = pt_proc[rrt_mask.values].copy()
    if rrt_proc.empty:
        return result

    time_col = "chartdate" if "chartdate" in rrt_proc.columns else None
    if time_col is None:
        return result

    if not pd.api.types.is_datetime64_any_dtype(rrt_proc[time_col]):
        rrt_proc[time_col] = pd.to_datetime(rrt_proc[time_col], errors="coerce")
    rrt_proc = rrt_proc.dropna(subset=[time_col])

    if target_time is not None:
        rrt_proc = rrt_proc[rrt_proc[time_col] <= target_time]

    if rrt_proc.empty:
        return result

    rrt_proc = rrt_proc.sort_values(time_col)

    rrt_types = [
        _RRT_ICD_CODES.get(c.strip().upper(), "HD")
        for c in rrt_proc[code_col].astype(str)
    ]
    last_type = rrt_types[-1] if rrt_types else "HD"
    last_date = rrt_proc[time_col].iloc[-1]

    result["has_rrt"] = True
    result["rrt_type"] = last_type
    result["last_session_date"] = last_date.strftime("%m-%d") if pd.notna(last_date) else None
    result["total_sessions"] = len(rrt_proc)

    return result


# ============================================================
# 9. Major Procedure History (주요 시술 이력) 추출
#    - CCI는 '진단'만 반영, '시술'은 반영하지 않음
#    - 비뇨기 시술: CCI 통제 후에도 전 구간 유의 (p<0.003, Cr range +2.7~+6.4)
#    - 심장수술(CVC 제외): CCI 6+ 구간에서 유의 (p<0.001, Cr range +1.1)
#    - 기계환기: CCI 6+ 구간에서만 유의 (현재 비활성, 향후 재평가)
# ============================================================

# CVC(Central Venous Catheter) insertion → 단순 시술, 심장수술로 분류하지 않음
_CVC_CODES = frozenset([
    "02HV33Z", "02H633Z", "02HV03Z", "02HK33Z", "02HN33Z",
    "02HP32Z", "02HQ32Z", "02H63JZ", "02HK3JZ", "02HK3KZ",
    "02PA", "02PY",  # prefix: device removal
])

_URINARY_PROC_DESC = {
    "0T25": "ureteral stent",
    "0TY0": "kidney transplant",
    "0TY1": "kidney transplant",
    "0TB0": "kidney biopsy",
    "0TB1": "kidney biopsy",
    "0T90": "kidney drainage",
    "0T91": "kidney drainage",
    "0T93": "kidney drainage",
    "0TP9": "ureteral device",
    "0T77": "ureteral dilation",
    "0T78": "ureteral dilation",
    "0T76": "ureteral dilation",
    "0TJB": "bladder inspection",
    "0T9B": "bladder drainage",
    "0TCB": "bladder extraction",
    "0TQB": "bladder repair",
    "0TTB": "bladder resection",
    "0T2B": "bladder change",
}

_CARDIAC_SURGERY_DESC = {
    "0210": "CABG",
    "0211": "CABG",
    "027":  "coronary dilation/stent",
    "02R":  "valve replacement",
    "02Q":  "valve repair",
    "02U":  "valve supplement",
    "02N":  "cardiac release",
    "02B":  "cardiac excision",
    "02C":  "cardiac extraction",
    "02L":  "cardiac occlusion",
    "025":  "cardiac destruction",
    "02W":  "cardiac revision",
    "02S":  "cardiac reposition",
    "02H4": "pacemaker/ICD lead",
    "02HN": "pacemaker/ICD lead",
}


def _is_cvc_code(code: str) -> bool:
    """CVC 삽입/제거 코드인지 판별."""
    code = code.upper().strip()
    if code in _CVC_CODES:
        return True
    for prefix in ["02PA", "02PY"]:
        if code.startswith(prefix):
            return True
    return False


def _describe_procedure(code: str, desc_map: dict) -> str:
    """ICD-10-PCS 코드에서 가장 구체적인 설명을 반환."""
    code = code.upper().strip()
    for prefix_len in [4, 3]:
        key = code[:prefix_len]
        if key in desc_map:
            return desc_map[key]
    return ""


def extract_major_procedures(
    patient_id: int,
    procedures_df: pd.DataFrame,
    target_time: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """환자의 주요 시술 이력 추출 (target_time 이전).

    추출 대상:
      - urinary_procedures: 비뇨기 시술 (ICD-10-PCS 0T)
      - cardiac_surgery: 심장수술 (ICD-10-PCS 02, CVC 제외)
      - mechanical_ventilation: 기계환기 (ICD-10-PCS 5A19) [현재 비활성]

    Returns:
        {
            "urinary_procedures": [
                {"date": Timestamp, "description": "ureteral stent"},
                ...
            ],
            "cardiac_surgery": [
                {"date": Timestamp, "description": "CABG"},
            ],
            # "mechanical_ventilation": [...],  # 현재 비활성
        }
    """
    result: Dict[str, Any] = {
        "urinary_procedures": [],
        "cardiac_surgery": [],
        # 기계환기: CCI 통제 시 CCI 6+ 구간에서만 유의 (p<0.001).
        # 자체적으로 CCI-독립적 기여가 제한적이므로 현재 비활성.
        # 향후 ICU 환자 subgroup 분석이나 중증도 모델링 시 재평가.
        # "mechanical_ventilation": [],
    }

    if not isinstance(procedures_df, pd.DataFrame) or procedures_df.empty:
        return result

    pt_proc = procedures_df[procedures_df["subject_id"] == patient_id]
    if pt_proc.empty:
        return result

    code_col = "icd_code" if "icd_code" in pt_proc.columns else None
    if code_col is None:
        return result

    time_col = "chartdate" if "chartdate" in pt_proc.columns else None
    if time_col is None:
        return result

    pt_proc = pt_proc.copy()
    if not pd.api.types.is_datetime64_any_dtype(pt_proc[time_col]):
        pt_proc[time_col] = pd.to_datetime(pt_proc[time_col], errors="coerce")
    pt_proc = pt_proc.dropna(subset=[time_col])

    if target_time is not None:
        pt_proc = pt_proc[pt_proc[time_col] <= target_time]
    if pt_proc.empty:
        return result

    pt_proc = pt_proc.sort_values(time_col)

    for _, row in pt_proc.iterrows():
        code = str(row[code_col]).strip().upper()
        date = row[time_col]

        # --- 비뇨기 시술 (0T) ---
        if code.startswith("0T"):
            desc = _describe_procedure(code, _URINARY_PROC_DESC) or "urinary procedure"
            result["urinary_procedures"].append({"date": date, "description": desc})

        # --- 심장수술 (02, CVC 제외) ---
        elif code.startswith("02") and not _is_cvc_code(code):
            desc = _describe_procedure(code, _CARDIAC_SURGERY_DESC) or "cardiac procedure"
            result["cardiac_surgery"].append({"date": date, "description": desc})

        # --- 기계환기 (5A19) --- 현재 비활성
        # 근거: CCI 통제 후 CCI 0-2 (p=0.105), CCI 3-5 (p=0.281)에서 유의하지 않음.
        # CCI 6+ (p<0.001)에서만 유의하나, 해당 구간은 CCI 자체가 중증도를 반영.
        # 따라서 CCI와 독립적인 추가 예측력이 제한적으로 판단하여 비활성.
        # 향후 활성화 시 아래 주석 해제:
        # elif code.startswith("5A19"):
        #     duration_map = {"5A1955Z": ">96h", "5A1945Z": "24-96h"}
        #     dur = duration_map.get(code, "")
        #     desc = f"mechanical ventilation ({dur})" if dur else "mechanical ventilation"
        #     result.setdefault("mechanical_ventilation", []).append(
        #         {"date": date, "description": desc}
        #     )

    return result


def format_major_procedures(
    proc_history: Dict[str, Any],
    target_time: Optional[pd.Timestamp] = None,
    lang: str = "en",
) -> str:
    """extract_major_procedures 결과 → 프롬프트 텍스트.

    Examples:
        Major procedures: cardiac surgery (CABG, 14d before); urinary (stent, 3d before)
        Major procedures: urinary (kidney biopsy, 7d before, kidney drainage, 45d before)
    """
    if not proc_history:
        return ""

    category_texts: List[str] = []

    for cat_key, cat_label in [
        ("cardiac_surgery", "cardiac"),
        ("urinary_procedures", "urinary"),
        # ("mechanical_ventilation", "mech. ventilation"),  # 비활성
    ]:
        entries = proc_history.get(cat_key, [])
        if not entries:
            continue

        seen_descs: Dict[str, dict] = {}
        for e in entries:
            desc = e.get("description", "procedure")
            if desc not in seen_descs:
                seen_descs[desc] = {"count": 1, "last_date": e.get("date")}
            else:
                seen_descs[desc]["count"] += 1
                d = e.get("date")
                if d is not None and (seen_descs[desc]["last_date"] is None or d > seen_descs[desc]["last_date"]):
                    seen_descs[desc]["last_date"] = d

        parts: List[str] = []
        for desc, info in seen_descs.items():
            timing = ""
            if target_time is not None and info["last_date"] is not None:
                days = (target_time - info["last_date"]).days
                if days < 0:
                    days = 0
                if days <= 365:
                    timing = f", {days}d before" if days > 0 else ", same day"
                else:
                    yr = days // 365
                    timing = f", {yr}yr before"

            count_str = f" x{info['count']}" if info["count"] > 1 else ""
            parts.append(f"{desc}{count_str}{timing}")

        if len(parts) > 3:
            parts = parts[:3]

        category_texts.append(f"{cat_label} ({'; '.join(parts)})")

    if not category_texts:
        return ""

    return f"Major procedures: {', '.join(category_texts)}"


def extract_contrast_exposure(
    patient_id: int,
    procedures_df: pd.DataFrame,
    target_time: Optional[pd.Timestamp] = None,
    lookback_days: int = 7,
) -> Dict[str, Any]:
    """target_time 이전 lookback_days 이내 조영제 노출 여부.

    Returns:
        {"exposed": True, "date": "03-10", "days_ago": 3}
    """
    result: Dict[str, Any] = {"exposed": False, "date": None, "days_ago": None}

    if not isinstance(procedures_df, pd.DataFrame) or procedures_df.empty:
        return result
    if target_time is None:
        return result

    pt_proc = procedures_df[procedures_df["subject_id"] == patient_id]
    if pt_proc.empty:
        return result

    code_col = "icd_code" if "icd_code" in pt_proc.columns else None
    if code_col is None:
        return result

    codes = pt_proc[code_col].astype(str).str.strip().str.upper()
    contrast_mask = codes.isin(_CONTRAST_ICD_CODES)
    c_proc = pt_proc[contrast_mask.values].copy()
    if c_proc.empty:
        return result

    time_col = "chartdate" if "chartdate" in c_proc.columns else None
    if time_col is None:
        return result

    if not pd.api.types.is_datetime64_any_dtype(c_proc[time_col]):
        c_proc[time_col] = pd.to_datetime(c_proc[time_col], errors="coerce")
    c_proc = c_proc.dropna(subset=[time_col])

    window_start = target_time - pd.Timedelta(days=lookback_days)
    recent = c_proc[(c_proc[time_col] >= window_start) & (c_proc[time_col] <= target_time)]
    if recent.empty:
        return result

    last = recent.sort_values(time_col).iloc[-1]
    last_date = last[time_col]
    days_ago = (target_time - last_date).days

    result["exposed"] = True
    result["date"] = last_date.strftime("%m-%d") if pd.notna(last_date) else None
    result["days_ago"] = int(days_ago)

    return result


# ============================================================
# 10. Vasopressor / IV Fluid 추출 (input_events 기반)
#
# 승압제: ΔeGFR p=1.15e-70, |ΔeGFR| p=6.22e-46
#   → hemodynamic instability → 신장관류↓ → AKI 위험
#   → 커버리지 13.9% (ICU 환자), 해당 환자에게 강력한 신호
#
# 수액: ΔeGFR p=2.13e-131, 용량-반응 관계 확인
#   → Cr 희석 효과 → eGFR 과추정 가능 → 보정 가치
#   → 커버리지 26.8% (ICU 환자)
# ============================================================

_VASOPRESSOR_ITEMIDS = {
    221906: "norepinephrine",
    221289: "epinephrine",
    222315: "vasopressin",
    221749: "phenylephrine",
    221662: "dopamine",
    221653: "dobutamine",
    221986: "milrinone",
}

_FLUID_CATEGORIES = frozenset([
    "02-Fluids (Crystalloids)",
    "03-IV Fluid Bolus",
    "04-Fluids (Colloids)",
])


def extract_vasopressor_status(
    patient_id: int,
    input_events_df: pd.DataFrame,
    target_time: Optional[pd.Timestamp] = None,
    lookback_days: int = 7,
) -> Dict[str, Any]:
    """target_time 이전 lookback_days 이내 승압제 사용 이력.

    Returns:
        {
            "has_vasopressor": True,
            "agents": ["norepinephrine", "vasopressin"],
            "last_date": Timestamp,
            "days_before": 3,
        }
    """
    result: Dict[str, Any] = {
        "has_vasopressor": False,
        "agents": [],
        "last_date": None,
        "days_before": None,
    }

    if not isinstance(input_events_df, pd.DataFrame) or input_events_df.empty:
        return result
    if target_time is None:
        return result

    pt_ie = input_events_df[input_events_df["subject_id"] == patient_id]
    if pt_ie.empty:
        return result

    vp = pt_ie[pt_ie["itemid"].isin(_VASOPRESSOR_ITEMIDS.keys())].copy()
    if vp.empty:
        return result

    time_col = "starttime"
    if time_col not in vp.columns:
        return result

    if not pd.api.types.is_datetime64_any_dtype(vp[time_col]):
        vp[time_col] = pd.to_datetime(vp[time_col], errors="coerce")
    vp = vp.dropna(subset=[time_col])

    window_start = target_time - pd.Timedelta(days=lookback_days)
    recent = vp[(vp[time_col] >= window_start) & (vp[time_col] <= target_time)]
    if recent.empty:
        return result

    agents = sorted(set(
        _VASOPRESSOR_ITEMIDS.get(iid, "vasopressor")
        for iid in recent["itemid"].unique()
    ))

    last_time = recent[time_col].max()
    days_before = (target_time - last_time).days

    result["has_vasopressor"] = True
    result["agents"] = agents
    result["last_date"] = last_time
    result["days_before"] = max(int(days_before), 0)

    return result


def extract_fluid_status(
    patient_id: int,
    input_events_df: pd.DataFrame,
    target_time: Optional[pd.Timestamp] = None,
    lookback_days: int = 3,
) -> Dict[str, Any]:
    """target_time 이전 lookback_days 이내 IV 수액 투여 상태.

    Returns:
        {
            "has_fluid": True,
            "total_ml": 3500.0,
            "last_date": Timestamp,
            "days_before": 1,
            "period_days": 3,
        }
    """
    result: Dict[str, Any] = {
        "has_fluid": False,
        "total_ml": 0.0,
        "last_date": None,
        "days_before": None,
        "period_days": lookback_days,
    }

    if not isinstance(input_events_df, pd.DataFrame) or input_events_df.empty:
        return result
    if target_time is None:
        return result

    pt_ie = input_events_df[input_events_df["subject_id"] == patient_id]
    if pt_ie.empty:
        return result

    cat_col = "ordercategoryname"
    if cat_col not in pt_ie.columns:
        return result

    fluids = pt_ie[pt_ie[cat_col].isin(_FLUID_CATEGORIES)].copy()
    if fluids.empty:
        return result

    time_col = "starttime"
    if time_col not in fluids.columns:
        return result

    if not pd.api.types.is_datetime64_any_dtype(fluids[time_col]):
        fluids[time_col] = pd.to_datetime(fluids[time_col], errors="coerce")
    fluids = fluids.dropna(subset=[time_col])

    window_start = target_time - pd.Timedelta(days=lookback_days)
    recent = fluids[(fluids[time_col] >= window_start) & (fluids[time_col] <= target_time)]
    if recent.empty:
        return result

    total_ml = recent["amount"].sum() if "amount" in recent.columns else 0.0
    if pd.isna(total_ml) or total_ml <= 0:
        return result

    last_time = recent[time_col].max()
    days_before = (target_time - last_time).days

    result["has_fluid"] = True
    result["total_ml"] = float(total_ml)
    result["last_date"] = last_time
    result["days_before"] = max(int(days_before), 0)

    return result
