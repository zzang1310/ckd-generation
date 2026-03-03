"""Central feature definitions for CKD-LLM.

This file is intended to be the single source of truth for:
- FEATURE_ITEMIDS: mapping of feature name -> MIMIC itemids used in prompt builders
- CHART_FEATURES: list of chart/vital feature names used in prompt builders

It also provides extraction-oriented grouped feature maps:
- LAB_FEATURES / VITAL_FEATURES
- helper getters to flatten itemids

Note:
- This file was reconstructed to match the current prompt builder defaults.
- Update here to change feature sets across scripts consistently.

향후 서비스 환경 (웨어러블 센서 + 스마트폰 기반 데이터 수집 플랫폼):
- 활성 피처: 웨어러블/가정에서 취득 가능한 항목 (체중, 혈압, 심박수, SpO2, 체온, 기본 혈액검사)
- 주석 피처: ICU/입원 환경에서만 얻을 수 있어 서비스에서 사용 불가한 항목
"""

from __future__ import annotations

from typing import Dict, List

# ============================================================
# Prompt feature definitions (활성 피처)
# Lab + chart itemids (prompt generation uses CHART_FEATURES to split lab vs chart)
# ============================================================

FEATURE_ITEMIDS: Dict[str, List[int]] = {
    # -- Lab features --
    "creatinine": [50912, 52546],
    "bun": [51006, 52647],
    "egfr": [53161],
    "sodium": [50983, 52623],
    "potassium": [50971, 52610],
    "glucose": [50931, 52569],
    "albumin": [50862, 53085, 53138],
    "phosphate": [50970],
    # -- Vital features (chartevents) --
    "sbp": [220050, 220179, 51],
    "heart_rate": [220045, 211],
    "temperature": [223761, 223762, 678],
    "resp_rate": [220210, 618],
    "spo2": [220277],
    "map": [220052, 456],
    # -- Body metrics (chartevents) --
    "weight": [224639, 226512, 226531],   # Daily Weight, Admit Wt, Daily Wt (kg)
    "height": [226730],                    # Height (cm), BMI 계산용
}


# Chart/vital feature names used in prompts (Lab이 아닌 chartevents 기반 피처)
CHART_FEATURES: List[str] = [
    "sbp",
    "heart_rate",
    "temperature",
    "resp_rate",
    "spo2",
    "map",
    "weight",
]

# Body metric feature names (chartevents 기반, 별도 카테고리)
BODY_METRIC_FEATURES: List[str] = [
    "weight",
    "height",
]


# ============================================================
# 추가 Lab 피처 (주석 처리 — 서비스 환경에서 취득 불가)
#
# 아래 피처들은 임상적으로 CKD/AKI 예측에 매우 유용하지만,
# 웨어러블 센서 + 스마트폰 기반 서비스 환경에서는 채혈이 필요하여
# 실시간 취득이 불가능합니다.
# 연구 목적 또는 입원 환경 모델 구축 시 활성화하세요.
# ============================================================

# FEATURE_ITEMIDS_HOSPITAL_ONLY: Dict[str, List[int]] = {
#     # CKD 빈혈 평가 — EPO 부족으로 인한 빈혈은 CKD Stage 3부터 흔함
#     "hemoglobin": [51222],
#     "hematocrit": [51221],
#
#     # CKD-MBD (뼈-미네랄 대사 이상) — CKD 3대 합병증 중 하나
#     "calcium": [50893],
#
#     # 대사성 산증 — CKD 진행의 핵심 지표, 투석 결정 기준
#     "bicarbonate": [50882],
#
#     # 조직 관류/중증도 — AKI 원인 감별 및 중증도 판단
#     "lactate": [50813],
#
#     # 감염/염증 — AKI의 주요 원인(패혈증) 감별
#     "wbc": [51301],
#     "crp": [50889],          # C-reactive protein
#
#     # 전해질 균형 — 근육 경련, 부정맥 위험 평가
#     "magnesium": [50960],
#
#     # 통풍/신장결석 위험 — CKD 환자에서 흔한 합병증
#     "uric_acid": [51007],
#
#     # 심장 손상 마커 — 심신증후군(cardiorenal syndrome) 감별
#     "troponin": [51003],
#
#     # 응고/간기능 — 간신증후군(hepatorenal syndrome) 감별
#     "pt_inr": [51237, 51275],
# }


# ============================================================
# 소변량 (주석 처리 — 입원 환자/도뇨관 삽입 환자만 해당)
#
# KDIGO AKI 기준에서 소변량(0.5 mL/kg/h 미만 6시간 이상)은
# 핵심 진단 기준이지만, 정확한 측정에는 유치 도뇨관(Foley catheter)이
# 필요하여 외래/가정 환경에서는 사용 불가합니다.
# 입원 환경 AKI 예측 모델 구축 시 활성화하세요.
# ============================================================

# URINE_OUTPUT_ITEMIDS: List[int] = [
#     226559,  # Foley (가장 정확, ICU)
#     226560,  # Void (자가배뇨)
#     226561,  # Condom Cath
#     226584,  # Ileoconduit
#     226563,  # Suprapubic
#     227488,  # GU Irrigant/Urine Volume Out
#     227489,  # GU Irrigant Volume In
# ]


# ============================================================
# Extraction feature definitions (grouped)
# extract_with_egfr_full_fast.py 호환용
# ============================================================

LAB_FEATURES: Dict[str, Dict[str, List[int]]] = {
    "kidney_core": {
        "creatinine": FEATURE_ITEMIDS["creatinine"],
        "bun": FEATURE_ITEMIDS["bun"],
        "egfr": FEATURE_ITEMIDS["egfr"],
    },
    "electrolytes": {
        "sodium": FEATURE_ITEMIDS["sodium"],
        "potassium": FEATURE_ITEMIDS["potassium"],
        "phosphate": FEATURE_ITEMIDS["phosphate"],
    },
    "metabolism": {
        "glucose": FEATURE_ITEMIDS["glucose"],
    },
    "kidney_specific": {
        "albumin": FEATURE_ITEMIDS["albumin"],
    },
}


VITAL_FEATURES: Dict[str, Dict[str, List[int]]] = {
    "icu_core": {
        "sbp": FEATURE_ITEMIDS["sbp"],
        "heart_rate": FEATURE_ITEMIDS["heart_rate"],
        "map": FEATURE_ITEMIDS["map"],
        "resp_rate": FEATURE_ITEMIDS["resp_rate"],
        "spo2": FEATURE_ITEMIDS["spo2"],
        "temperature": FEATURE_ITEMIDS["temperature"],
    },
    "body_metrics": {
        "weight": FEATURE_ITEMIDS["weight"],
        "height": FEATURE_ITEMIDS["height"],
    },
}


# ============================================================
# Helper getters
# ============================================================

def get_all_lab_itemids() -> List[int]:
    """Return flattened unique lab itemids used for extraction filtering."""
    out: List[int] = []
    seen = set()
    for category in LAB_FEATURES.values():
        for itemids in category.values():
            for x in itemids:
                if x not in seen:
                    seen.add(x)
                    out.append(int(x))
    return out


def get_all_vital_itemids() -> List[int]:
    """Return flattened unique vital/chart itemids used for extraction filtering."""
    out: List[int] = []
    seen = set()
    for category in VITAL_FEATURES.values():
        for itemids in category.values():
            for x in itemids:
                if x not in seen:
                    seen.add(x)
                    out.append(int(x))
    return out


def get_egfr_itemids() -> List[int]:
    """Return eGFR itemids."""
    return list(FEATURE_ITEMIDS.get("egfr", [53161]))


def get_weight_itemids() -> List[int]:
    """Return body weight itemids."""
    return list(FEATURE_ITEMIDS.get("weight", []))


def get_height_itemids() -> List[int]:
    """Return height itemids."""
    return list(FEATURE_ITEMIDS.get("height", []))
