#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학습용 프롬프트 JSONL 생성 스크립트 - 미래 예측(Prognosis) 통합 버전

VERSION: v11.0.0 (2026.02.09)
PURPOSE:
- v8 Prognosis 기반 + v10 compact 약물 포맷 + 환자 배경정보 통합
- 현재 시점의 진단(Diagnosis)이 아닌, 미래 시점(Prognosis)의 신장 기능 및 AKI 위험 예측
- "Target Anchor & Back-shift" 전략 적용
- Trajectory Prediction 지원

v11 주요 개선사항:
- 환자 배경정보: 동반질환(ICD 기반), CCI, CKD Stage, 입원맥락, 체중 트렌드
- compact 약물 포맷: v10의 range-notation ("0,6,20-21 (4/28d)") 도입으로 토큰 절약
- 체중 피처: chartevents에서 Daily Weight 자동 추출

CHANGELOG:
- v11.0.0: v8 기반 + v10 compact rx + 환자 배경정보 통합
- v8.0.0: Quality Gating 추가 (앵커 시점 기반 과거 윈도우 품질 보장)
- v7.0.0: Initial Fork from v5.0.0. Added Prognosis/Future Prediction logic.
- v5.0.0: AKI 탐지 프롬프트 정교화(KDIGO 규칙 명시), 입력 문맥 명확화
- v2.0.0: EMAR/Input Events/Rx 3단계 처방 + Procedures 필터링 + 성능 최적화
- v1.x.x: 기본 멀티모달 특징 추출 및 KDIGO AKI 탐지

입력: processed_data의 완전 멀티모달 코호트(pkl) + (선택) 추천 환자 pkl
출력: instruction/input/output/metadata 구조의 JSONL (eGFR 회귀 + AKI 이진 분류)

사용 예시:
  python build_prompt_dataset_v5.py \
    --processed-data-path "D:/CKD-LLM/processed_data" \
    --data-file full_multimodal_kidney_cohort_with_egfr_complete.pkl \
    --patient-selection valid_patients_selection.pkl \
    --max-patients 30 \
    --max-labels-per-patient -1 \
    --tasks egfr,aki \
    --output processed_data/kidney_llm_prompts.jsonl

실제 예시:
python build_prompt_dataset_v5.py --max-patients 1000 --output-lang en  로 수행 시 적용되는 디폴트 동작 요약:

>데이터 경로
--processed-data-path: D:/CKD-LLM/processed_data
--data-file: intermediate/full_multimodal_kidney_cohort_with_egfr_complete_fast.pkl
--patient-selection: patient_selection/patient_selection.pkl

>생성 범위/태스크
--max-patients: 1000(명령문으로 지정함)
--output-lang: en(명령문으로 지정함)

--tasks: egfr,aki(디폴트)
--max-labels-per-patient: -1(디폴트)
--output-format: json (JSON 배열)(디폴트)
--file-format: json-pretty (가독성 좋은 JSON 배열)(디폴트)
--aki-label-source: hybrid (하이브리드)(디폴트)
--hybrid-policy: priority (진단 우선)(디폴트)
--aki-neg-pos-ratio: 1.0(디폴트)
--workers: 24(디폴트)
--aki-require-kdigo-consistency: True(디폴트)
--aki-scan-all-admissions: False(디폴트)
--kdigo-baseline-quantile: 0.3(디폴트)
--procedure-extend-after-by-window: False(디폴트)

>출력 경로
--output: CKD_llm_prompts_날짜_P1000_{LALL|LEGFR|LAKI}_HHMMSS.json
  LALL=egfr+aki, LEGFR=egfr만, LAKI=aki만 (--tasks 기준)

"""



__version__ = "5.0.0-stable"
__author__ = "CKD-LLM Project"
__date__ = "2025.10.13"

import argparse
import json
import pickle
import os
import hashlib
from datetime import timedelta, datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import traceback as _tb
import concurrent.futures as _fut
import math
from numbers import Number

import pandas as pd

# 공통 피처 정의 import
try:
    from config.feature_definitions import (
        FEATURE_ITEMIDS, CHART_FEATURES, BODY_METRIC_FEATURES,
        get_weight_itemids,
    )
    from config.comorbidity_definitions import (
        build_patient_background,
        format_background_for_prompt,
        extract_weight_info,
        format_weight_for_prompt,
        extract_weight_trajectory,
        format_weight_trajectory,
        extract_egfr_trajectory,
        extract_rrt_status,
        extract_contrast_exposure,
        extract_major_procedures,
        extract_vasopressor_status,
        extract_fluid_status,
    )
except ImportError as e:
    raise ImportError(
        "config 모듈 import에 실패했습니다. "
        "프로젝트 루트에 `config/feature_definitions.py`, `config/comorbidity_definitions.py`가 "
        "존재해야 하며, 작업 디렉토리를 리포지토리 루트(`/home/noh/CKD-LLM`)에서 실행하세요."
    ) from e

# ======= 빠른 네비게이션 =======
# 1) CLI 인자/상수: 경로, tasks(egfr,aki), 출력 형식(text), 언어(ko/en), AKI 옵션(라벨 소스·음성:양성 비율·KDIGO 일치·전체 입원 스캔)
# 2) 텍스트 템플릿/포맷 도우미: ko/en 지시문, 출력 prefix, 공통 문자열 유틸
# 3) 유틸/전처리: 안전 접근, 시간 변환, 긴 윈도우 추세 RLE 압축 옵션
# 4) 일자별 맵 사전계산: Lab/Chart/Rx 최신값 인덱싱
# 5) 약물 특징 추출: EMAR → Input Events → Prescriptions 하이브리드(소스 태깅)
# 6) 시술/수술 기반 제외: 투석/이식 등 특정 기간 라벨 제외 로직
# 7) 윈도우별 특징 추출: 과거/최근 윈도우 집계, eGFR 누설 방지(exclude)
# 8) 요약 포맷팅: 트렌드 텍스트, 최근 스냅샷, 약물 요약
# 9) 라벨 생성: eGFR 현재시점 타겟, AKI 양/음성(진단/KDIGO/하이브리드, 비율 제어, 일관성 옵션)
# 10) 프롬프트 빌더: eGFR 회귀, AKI 분류(ko/en)
# 11) 메인: 데이터 로드 → 환자 반복 → 프롬프트 생성/저장 → 통계 출력



def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다. 동작 변경 없음(가독성 주석만 추가).

    Returns:
        argparse.Namespace: 사용자 지정 옵션(경로, 태스크, 출력 등)
    """
    # 현재 시점으로 기본 파일명 생성
    current_time = datetime.now().strftime("%m%d_%H%M")
    default_output = f"processed_data/generated_prompts/CKD_llm_prompts_{current_time}.json"

    # OS별 기본 processed_data 경로
    # - Windows: 기존 프로젝트 관행 유지
    # - Linux/WSL: 리포 내 상대경로가 기본
    default_processed_data_path = "D:/CKD-LLM/processed_data" if os.name == "nt" else "processed_data"
    
    parser = argparse.ArgumentParser(description="선별 환자 기반 eGFR/AKI 프롬프트 JSONL 생성")
    parser.add_argument("--processed-data-path", type=str, default=default_processed_data_path)
    parser.add_argument("--data-file", type=str, default="intermediate/full_multimodal_kidney_cohort_with_egfr_complete_fast.pkl")
    parser.add_argument("--patient-selection", type=str, default="patient_selection/patient_selection.pkl")
    parser.add_argument("--max-patients", type=int, default=100)
    parser.add_argument("--max-labels-per-patient", type=int, default=-1, help="환자당 최대 라벨 수 (-1이면 제한 없음)")
    parser.add_argument("--tasks", type=str, default="egfr,aki", help="생성할 태스크: egfr,aki 중 콤마로 구분")
    parser.add_argument("--output", type=str, default=default_output)
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["text", "json"],
        help="출력 형식: text(자연어) 또는 json(구조화). 학습/평가 자동화에는 json 권장"
    )
    parser.add_argument(
        "--file-format",
        type=str,
        default="json-pretty",
        choices=["jsonl", "json-pretty"],
        help="파일 저장 형식: jsonl(한줄씩) 또는 json-pretty(가독성 좋은 JSON Array)"
    )
    parser.add_argument(
        "--output-lang",
        type=str,
        default="ko",
        choices=["ko", "en"],
        help="출력 언어: ko(한국어) 또는 en(English). instruction/output의 고정 문구에 적용"
    )
    # AKI 라벨 소스 옵션
    parser.add_argument(
        "--aki-label-source",
        type=str,
        default="hybrid",
        choices=["diagnosis", "kdigo", "hybrid"],
        help="AKI 라벨 소스 선택: diagnosis(진단코드), kdigo(SCr 기반), hybrid(진단 우선 후 KDIGO 보완)"
    )
    parser.add_argument(
        "--hybrid-policy",
        type=str,
        default="priority",
        choices=["priority", "union", "intersection"],
        help="hybrid 정책: priority(진단 우선, 없으면 KDIGO), union(합집합), intersection(교집합)"
    )
    parser.add_argument(
        "--aki-neg-pos-ratio",
        type=float,
        default=1.0,
        help="AKI 음성:양성 비율. 예) 1.0이면 양성과 동일 수의 음성 라벨 생성"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="환자 단위 병렬 처리 스레드 수(0이면 병렬화하지 않음)"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--aki-require-kdigo-consistency",
        dest="aki_require_kdigo_consistency",
        action="store_true",
        help="hybrid/diagnosis 앵커의 AKI 양성은 KDIGO(48h +0.3 또는 7d 1.5x) 근거가 있을 때만 생성"
    )
    grp.add_argument(
        "--no-aki-require-kdigo-consistency",
        dest="aki_require_kdigo_consistency",
        action="store_false",
        help="KDIGO 근거 일관성 요구 비활성화"
    )
    parser.set_defaults(aki_require_kdigo_consistency=True)
    parser.add_argument(
        "--aki-scan-all-admissions",
        action="store_true",
        help="KDIGO 일치 시점 확장 시 동일 환자의 모든 입원창을 스캔(끄면 해당 진단 hadm 창에서만 스캔)"
    )
    parser.add_argument(
        "--kdigo-baseline-quantile",
        type=float,
        default=0.3,
        help="7일 baseline을 최솟값 대신 분위수로 사용할 때 0<q<1 설정(예: 0.3). None이면 최솟값 사용"
    )
    parser.add_argument(
        "--procedure-extend-after-by-window",
        action="store_true",
        help="프로시저 제외 after 시간을 윈도우 최대길이(28d)+회복기간만큼 추가 연장"
    )
    parser.add_argument(
        "--medication-format",
        type=str,
        default="compact",
        choices=["legacy", "compact"],
        help="Medications 표기: legacy(pat=...;last=...;days=...;tag=...), compact(0-1,20-21 (n/28d) — 토큰 절약)",
    )

    # Prognosis 전용 옵션
    parser.add_argument("--prediction-windows", type=str, default="0,1,2,3,4,5,6,7",
                        help="예측할 미래 시점 목록 (단위: 일). 예: '0,1,3,7' -> 현재(0일) + 1일, 3일, 7일 전 예측")

    # Prognosis 품질 게이팅(앵커(cutoff_time) 기준 과거 윈도우 품질 보장)
    # - 기본 적용: pred_days>0(미래예측) 샘플에만 적용
    # - 필요 시 --quality-gate-include-current 로 pred_days=0에도 적용
    parser.add_argument(
        "--min-scr-count",
        type=int,
        default=2,
        help="(품질 게이팅) 과거 윈도우 내 Creatinine(SCr) 최소 측정 개수. 0이면 비활성화",
    )
    parser.add_argument(
        "--min-scr-days",
        type=int,
        default=2,
        help="(품질 게이팅) 과거 윈도우 내 Creatinine(SCr) 최소 관측 일수(서로 다른 날짜 수). 0이면 비활성화",
    )
    parser.add_argument(
        "--min-observed-days",
        type=int,
        default=3,
        help="(품질 게이팅) 과거 윈도우 내 전체 관측일수 최소값(랩/바이탈/약물 중 어떤 형태로든 관측된 날짜 수). 0이면 비활성화",
    )
    parser.add_argument(
        "--quality-gate-include-current",
        action="store_true",
        help="품질 게이팅을 pred_days=0(현재시점) 샘플에도 적용",
    )

    # eGFR history(eGFR 과거 추세) 입력 포함 옵션
    # - 목표가 SCr 등 다른 피처 기반 추론이라면 기본 OFF 권장
    # - 실사용에서 과거 eGFR을 함께 제공할 계획이면 dropout으로 섞어 학습(robust) 가능
    parser.add_argument(
        "--egfr-history",
        type=str,
        default="off",
        choices=["off", "on", "dropout"],
        help="eGFR 프롬프트에 과거 eGFR Trend 입력을 포함할지. off=미포함, on=항상 포함, dropout=확률적으로 포함",
    )
    parser.add_argument(
        "--egfr-history-dropout-p",
        type=float,
        default=0.5,
        help="--egfr-history=dropout일 때, eGFR history를 '제거'할 확률 p (0~1). 예: 0.5면 절반은 제거",
    )

    return parser.parse_args()


# 전역 캐시 (읽기 전용으로 안전)
############################################################
# (1) CONFIG: 상수/매핑/전역설정
############################################################

# Global caches and feature maps  
PRECOMPUTED_DAILIES: Dict[int, Dict[str, Any]] = {}
KDIGO_CACHE: Dict[int, Dict[str, Any]] = {}

# 피처 정의는 config.feature_definitions에서 단일 소스로 관리

# Prescriptions (1D) keyword 사전: MIMIC-IV free-text(drug/medication)에 str.contains 기반 매칭.
# - EMAR: medication 컬럼, Prescriptions: drug 컬럼. Input Events: label/category 있으면 동일 키워드 사용.
# - 소문자 변환 후 매칭하므로 대소문자 무관. 브랜드명/제네릭 모두 넣어 두는 것이 좋음.
# - 신장/eGFR 관련: 이뇨제, ACE/ARB, NSAID, 승압제, 신장독성약, SGLT2i, 메트폼민, MRA, CNI 등.
RX_KEYWORDS: Dict[str, List[str]] = {
    # 당뇨/혈당: 신장 기능에 따른 인슐린/메트폼민 조절
    "insulin": ["insulin", "humalog", "novolog", "apidra", "lantus", "levemir", "tresiba", "humulin", "novolin"],
    # Loop 이뇨제: 신부전/체액, 전해질, AKI 위험
    "furosemide": ["furosemide", "lasix"],
    # 승압제: 혈역학/신관류 (input_events에도 별도 추출 있음)
    "pressors": ["norepinephrine", "epinephrine", "vasopressin", "phenylephrine", "dopamine", "dobutamine"],
    # ACEi/ARB: 신보호·고칼륨·혈역학
    "ace_arb": [
        "lisinopril", "enalapril", "captopril", "ramipril", "benazepril", "quinapril", "perindopril",
        "fosinopril", "trandolapril", "moexipril",
        "losartan", "valsartan", "irbesartan", "olmesartan", "candesartan", "telmisartan", "azilsartan", "eprosartan",
    ],
    # NSAID: 신장독성, 체액/전해질
    "nsaids": ["ibuprofen", "ketorolac", "naproxen", "diclofenac", "indomethacin", "celecoxib", "meloxicam", "nabumetone", "oxaprozin", "piroxicam"],
    # 신장독성 항생제 (AKI/D-AKI 위험)
    "aminoglycosides": ["gentamicin", "tobramycin", "amikacin", "neomycin", "streptomycin"],
    "vancomycin": ["vancomycin"],
    # 기타 이뇨제: 체액/전해질/신기능
    "diuretics_other": ["torsemide", "bumetanide", "hydrochlorothiazide", "chlorthalidone", "metolazone", "demadex", "zaroxolyn"],
    # SGLT2i: 신보호, eGFR 변동
    "sglt2i": ["empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin", "jardiance", "farxiga", "invokana"],
    # 메트폼민: CKD에서 용량 조절/중단
    "metformin": ["metformin", "glucophage"],
    # MRA: 칼륨, 신장
    "mra": ["spironolactone", "eplerenone", "aldactone"],
    # CNI (이식 후): 신장독성
    "cni": ["tacrolimus", "cyclosporine", "cyclosporin", "prograf", "neoral", "fk506"],
}

# Procedure 기반 데이터 제외 기준: 일상생활 예측에 부적합한 의료개입들
# - 각 항목은 keywords, exclusion_hours_before/after, reason을 포함
# - should_exclude_timepoint_for_procedures(...)에서 이 설정을 사용하여
#   exclusion_start = proc_time - before_hours
#   exclusion_end   = proc_time + after_hours
#   target_time이 [start, end]에 포함되면 해당 라벨 시점을 제외
# - 옵션 --procedure-extend-after-by-window ON이면 after에
#   WINDOW_MAX_DAYS(=28일)×24시간을 추가하여, 라벨 시점의 과거 최대 4주 창이 시술/수술일을 포함하지 않도록 보호(수술시점 후 28일 이후까지 옵션 ON/OFF)

# 최대 라벨 윈도우 길이(일)
# - 단순 가정: 현 버전은 28일(=4주) 고정
# - --procedure-extend-after-by-window 옵션 ON이면, after ← after + WINDOW_MAX_DAYS(=28일)
#   → 라벨 시점의 과거 최대 4주 윈도우가 시술/수술일을 포함하지 않도록 보장
WINDOW_MAX_DAYS: int = 0

EXCLUSION_PROCEDURES: Dict[str, Dict[str, Any]] = {
    "dialysis": {
        "keywords": ["39.95", "54.98", "5A1D", "3E1M", "dialysis", "hemodialysis", "peritoneal"],
        "exclusion_hours_before": 24,  # 투석 전 24시간
        "exclusion_hours_after": 24,   # 투석 후 24시간
        "reason": "투석은 신장기능에 직접적 영향을 미치는 의료개입"
    },
    "major_kidney_surgery": {
        "keywords": ["55.", "0T", "nephrectomy", "transplant", "kidney surgery", "renal surgery"],
        "exclusion_hours_before": 72,  # 수술 전 72시간
        "exclusion_hours_after": 168,  # 수술 후 7일
        "reason": "신장 수술은 장기간에 걸쳐 신장기능에 영향"
    },
    "kidney_biopsy": {
        "keywords": ["55.23", "0DTM", "0DTN", "biopsy", "kidney", "renal"],
        "exclusion_hours_before": 12,  # 생검 전 12시간
        "exclusion_hours_after": 48,   # 생검 후 48시간
        "reason": "신장 생검은 신장기능 평가에 일시적 영향"
    },
    "major_vascular_access": {
        "keywords": ["39.27", "39.93", "03HG", "05H", "access", "graft", "fistula", "central line"],
        "exclusion_hours_before": 12,  # 시술 전 12시간
        "exclusion_hours_after": 24,   # 시술 후 24시간
        "reason": "혈관접근로는 혈역학 및 신장기능에 영향"
    },
    "major_cardiac_surgery": {
        "keywords": ["36.", "02H", "02R", "cardiac surgery", "heart surgery", "bypass", "stent"],
        "exclusion_hours_before": 48,  # 수술 전 48시간
        "exclusion_hours_after": 120,  # 수술 후 5일
        "reason": "심장 수술은 전신 혈역학에 광범위한 영향"
    },
    "major_abdominal_surgery": {
        "keywords": ["47.", "48.", "49.", "50.", "51.", "52.", "53.", "54.", "0DT", "0F", "0B", "hepatectomy", "cholecystectomy", "gastrectomy", "colectomy", "bowel", "liver", "pancreas"],
        "exclusion_hours_before": 24,  # 수술 전 24시간
        "exclusion_hours_after": 72,   # 수술 후 3일
        "reason": "주요 복부수술은 수액균형과 신장관류에 영향"
    },
    "major_thoracic_surgery": {
        "keywords": ["32.", "33.", "34.", "0BB", "0BC", "0BD", "lung", "thoracic", "pneumonectomy", "lobectomy"],
        "exclusion_hours_before": 24,  # 수술 전 24시간
        "exclusion_hours_after": 48,   # 수술 후 2일
        "reason": "흉부수술은 수액관리와 혈역학에 영향"
    },
    "major_orthopedic_surgery": {
        "keywords": ["81.", "0QS", "0QR", "joint replacement", "hip", "knee", "spine fusion", "fracture repair"],
        "exclusion_hours_before": 12,  # 수술 전 12시간
        "exclusion_hours_after": 48,   # 수술 후 2일
        "reason": "주요 정형외과 수술은 출혈과 수액균형에 영향"
    },
    "neurosurgery": {
        "keywords": ["01.", "02.", "00.", "0015", "0016", "craniotomy", "brain", "spinal", "neurosurgery"],
        "exclusion_hours_before": 24,  # 수술 전 24시간
        "exclusion_hours_after": 72,   # 수술 후 3일
        "reason": "신경외과 수술은 수액관리와 혈압조절에 영향"
    },
    "emergency_surgery": {
        "keywords": ["trauma", "emergency", "urgent", "emergent", "stat"],
        "exclusion_hours_before": 6,   # 응급상황 전 6시간
        "exclusion_hours_after": 48,   # 수술 후 2일
        "reason": "응급수술은 전신상태와 신장기능에 급격한 영향"
    }
}

FALLBACK_LOOKBACK_HOURS: int = 24
MISSING_DISPLAY: str = "NA"  # 결측 표시 토큰 (짧은 창도 'NA'로 표기)

# === Trend compression options (RLE for long windows) ===
TREND_RLE_ENABLED: bool = True          # 긴 창에서 결측 런-렝스 압축 사용
TREND_RLE_MIN_RUN: int = 3              # 최소 연속 길이
TREND_RLE_APPLY_DAYS: int = 14          # 이 이상 일수 창에만 적용 (e.g., 14, 28)


TIME_WINDOWS: Dict[str, int] = {
    "current": 1,
    "3days": 3,
    "7days": 7,
    "2weeks": 14,
    "4weeks": 28,
}


AKI_CODES = [
    # ICD-10
    "N17", "N170", "N171", "N172", "N178", "N179",
    # ICD-9
    "584", "5840", "5841", "5845", "5846", "5847", "5848", "5849",
]

############################################################
# (2) UTILS: 공통 유틸(날짜/타입/출력/RLE)
############################################################

def to_date(obj) -> Optional[date]:
    """Timestamp/str/date → date 변환(실패 시 None).
    
    NOTE: Behavior preserved. No logic change.
    """
    if obj is None:
        return None
    if isinstance(obj, date):
        return obj
    if isinstance(obj, pd.Timestamp):
        return obj.date()
    if isinstance(obj, datetime):
        return obj.date()
    try:
        return pd.to_datetime(obj).date()
    except:
        return None

def safe_to_float(x) -> Optional[float]:
    """float 변환 실패 시 None.
    
    NOTE: Behavior preserved. No logic change.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return float(x)
    except Exception:
        return None

def rle_missing(seq: List[str], min_run: int) -> List[str]:
    """'NA' 연속구간 NAxN 압축.
    
    NOTE: Behavior preserved. No logic change.
    """
    if not seq:
        return seq
    compact: List[str] = []
    i = 0
    n = len(seq)
    while i < n:
        token = seq[i]
        if token == MISSING_DISPLAY:
            j = i
            while j < n and seq[j] == MISSING_DISPLAY:
                j += 1
            run_len = j - i
            if run_len >= min_run:
                compact.append(f"NAx{run_len}")
            else:
                compact.extend([MISSING_DISPLAY] * run_len)
            i = j
        else:
            compact.append(token)
            i += 1
    return compact

def groupby_daily_mean(df: pd.DataFrame, patient_id: int, time_col: str, value_col: str, itemids: List[int]) -> Dict[date, float]:
    """Lab/Chart 공통 패턴: subject_id 필터 → 시간 파싱 → notna → date 컬럼 → itemid 필터 → groupby mean.
    
    NOTE: Behavior preserved. No logic change.
    """
    if df.empty:
        return {}
    sub = df[df["subject_id"] == patient_id]
    if sub.empty:
        return {}
    sub = sub.copy()
    ensure_datetime(sub, [time_col])
    sub = sub[pd.notna(sub[time_col])]
    if sub.empty:
        return {}
    sub["date"] = sub[time_col].dt.date
    if itemids:
        sub = sub[sub["itemid"].isin(itemids)]
    if sub.empty:
        return {}
    sub = sub[pd.notna(sub[value_col])]
    if sub.empty:
        return {}
    return dict_keys_to_date(sub.groupby("date")[value_col].mean().to_dict())
def dict_keys_to_date(d: dict) -> Dict[date, Optional[float]]:
    """키를 date로 정규화(값은 float/None; NaN→None).
    
    NOTE: Behavior preserved. No logic change.
    """
    out: Dict[date, Optional[float]] = {}
    for k, v in d.items():
        date_key = to_date(k)
        if date_key is not None:
            if pd.isna(v):
                out[date_key] = None
            else:
                try:
                    out[date_key] = float(v) if v is not None else None
                except:
                    out[date_key] = None
    return out

def ensure_datetime(df: pd.DataFrame, cols: List[str]) -> None:
    """DataFrame 컬럼 파싱 로직을 하나의 유틸로 분리.
    
    NOTE: Behavior preserved. No logic change.
    """
    for col in cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
            except TypeError:
                df[col] = pd.to_datetime(df[col], errors="coerce")

def _sort_by_time(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """시간 키("time") 기준 오름차순 정렬. 빈 리스트는 그대로 반환."""
    return sorted(events, key=lambda e: e["time"]) if events else events

def format_window_display(window_name: Optional[str], window_days: Optional[int], lang: str = "ko") -> str:
    """윈도우 표기를 사용자 언어로 반환. 예: "7days (7d)".
    동작 변경 없음.
    """
    if window_name == "current":
        return "Current" if lang == "en" else "현재"
    if window_name is None:
        return ""
    return f"{window_name} ({window_days}d)"


def _format_missing(token: str) -> str:
    return token.replace("NA", ".")


def get_texts(lang: str) -> Dict[str, str]:
    """언어별 고정 문구 템플릿 반환(간단 버전)."""
    if lang == "en":
        return {
            "patient": "Patient ID",
            "index_time": "Index time",
            # 입력은 항상 '평가/현재 시점'까지의 과거 윈도우 요약이므로 명시적으로 표기
            "window": "Past Window",
            "basic": "Demographics",
            "history": "Past window summary",
            "window_note": (
                "Past window ending at index time. Lab/Vitals: daily averages, one value per day (left=oldest, right=recent)."
            ),
            "instr_egfr": (
                "Based on the patient's clinical data up to the current timepoint, "
                "predict the patient's current eGFR. "
                "Respond with only the numeric value."
            ),
            "instr_egfr_json": (
                "Based on the patient's clinical data up to the current timepoint, "
                "predict the patient's current eGFR. "
                "Output JSON only: {{\"eGFR\": <number>}}"
            ),
            "instr_egfr_prognosis": (
                "Based on the patient's clinical data up to the current timepoint, "
                "predict the patient's eGFR {horizon_days} days in the future. "
                "Respond with only the numeric value."
            ),
            "instr_egfr_prognosis_json": (
                "Based on the patient's clinical data up to the current timepoint, "
                "predict the patient's eGFR {horizon_days} days in the future. "
                "Output JSON only: {{\"eGFR\": <number>}}"
            ),
            "instr_aki": (
                "Based on the patient's current clinical data, determine whether the patient has AKI (0/1). "
                "Return only a single token: 1 (AKI present) or 0 (no AKI). Do not explain."
            ),
            "instr_aki_json": (
                "Based on the patient's current clinical data, determine whether the patient has AKI (0/1). "
                "Output JSON only: {{\"AKI\": 0|1}}"
            ),
            "instr_aki_prognosis": (
                "Based on the patient's clinical history available up to the current timepoint, "
                "predict whether the patient will have AKI (0/1) {horizon_days} days in the future.\n"
                "Return only a single token: 1 (AKI present) or 0 (no AKI). Do not explain."
            ),
            "instr_aki_prognosis_json": (
                "Based on the patient's clinical history available up to the current timepoint, "
                "predict whether the patient will have AKI (0/1) {horizon_days} days in the future.\n"
                "Output JSON only: {{\"AKI\": 0|1}}"
            ),
            "out_egfr_prefix": "Predicted eGFR:",
            "out_aki_prefix": "AKI prediction:",
            "aki_pos": "POS",
            "aki_neg": "NEG",
            "rationale": "Rationale: interpretation based on recent values and window trends",
        }
    # ko (default)
    return {
        "patient": "환자 ID",
        "index_time": "평가 시점",
        # 입력은 항상 '평가/현재 시점'까지의 과거 윈도우 요약이므로 명시적으로 표기
        "window": "과거 윈도우",
        "basic": "기본 정보",
        "history": "과거 윈도우 요약",
        "window_note": (
            "아래 값들은 ‘평가 시점’을 끝점으로 하는 과거 윈도우 기간의 요약입니다. "
            "Lab·바이탈: 일별 평균, 값 1개=1일 (왼쪽=과거, 오른쪽=최근)."
        ),
        "instr_egfr": (
            "현재 시점까지의 환자 임상 데이터를 바탕으로, 현재 eGFR을 예측하세요. "
            "수치만 답하세요."
        ),
        "instr_egfr_json": (
            "현재 시점까지의 환자 임상 데이터를 바탕으로, 현재 eGFR을 예측하세요. "
            "JSON만 출력: {{\"eGFR\": 숫자}}"
        ),
        "instr_egfr_prognosis": (
            "현재 시점까지의 환자 임상 데이터를 바탕으로, {horizon_days}일 후의 eGFR을 예측하세요. "
            "수치만 답하세요."
        ),
        "instr_egfr_prognosis_json": (
            "현재 시점까지의 환자 임상 데이터를 바탕으로, {horizon_days}일 후의 eGFR을 예측하세요. "
            "JSON만 출력: {{\"eGFR\": 숫자}}"
        ),
        "instr_aki": (
            "현재 시점의 환자 임상 데이터를 바탕으로 AKI(0/1) 발생 여부를 판단하세요. "
            "설명 없이 한 글자만 출력하세요: 1(AKI 있음) 또는 0(AKI 없음)."
        ),
        "instr_aki_json": (
            "현재 시점의 환자 임상 데이터를 바탕으로 AKI(0/1) 발생 여부를 판단하세요. "
            "JSON만 출력: {{\"aki\": 0|1}}"
        ),
        "instr_aki_prognosis": (
            "현재 시점까지의 환자 임상 병력을 바탕으로, {horizon_days}일 후의 AKI(0/1) 발생 여부를 예측하세요.\n"
            "설명 없이 한 글자만 출력하세요: 1(AKI 있음) 또는 0(AKI 없음)."
        ),
        "instr_aki_prognosis_json": (
            "현재 시점까지의 환자 임상 병력을 바탕으로, {horizon_days}일 후의 AKI(0/1) 발생 여부를 예측하세요.\n"
            "JSON만 출력: {{\"aki\": 0|1}}"
        ),
        "out_egfr_prefix": "예측 eGFR:",
        "out_aki_prefix": "예측 AKI 여부:",
        "aki_pos": "양성",
        "aki_neg": "음성",
        "rationale": "근거: 최근 측정값과 윈도우별 추이 기반의 임상적 해석",
    }


def _rx_classify_pattern(bits: str, recent3_sum: int, longest_streak: int, lang: str = "ko") -> str:
    if recent3_sum >= max(1, bits.count("1")) * 0.6:
        return "recent-focused" if lang == "en" else "recent-focused"
    if longest_streak >= 4:
        return "streak" if lang == "en" else "streak"
    # 간단 규칙: 산발적
    return "sporadic" if lang == "en" else "sporadic"


def _format_rx_compact(flags_seq: List[int], total_days: int) -> str:
    """flags_seq를 compact range-notation으로 변환 (v10 도입, 토큰 절약).

    0=가장 과거(윈도우 시작), N-1=가장 최근.
    반환: "26-27 (2/28d)" 또는 "0,6,20-21 (4/28d)"
    """
    n = len(flags_seq)
    if n == 0:
        return f"(0/{total_days}d)"
    days_with_med = [i for i, v in enumerate(flags_seq) if v]
    if not days_with_med:
        return f"(0/{total_days}d)"
    days_with_med.sort()
    ranges: List[Tuple[int, int]] = []
    start = days_with_med[0]
    end = start
    for d in days_with_med[1:]:
        if d == end + 1:
            end = d
        else:
            ranges.append((start, end))
            start = end = d
    ranges.append((start, end))
    parts = []
    for a, b in ranges:
        parts.append(str(a) if a == b else f"{a}-{b}")
    return f"{','.join(parts)} ({len(days_with_med)}/{total_days}d)"


############################################################
# (4) DAILY: 일별 집계(precompute_patient_daily_maps)
############################################################

def precompute_patient_daily_maps(clinical_data: Dict[str, Any], patient_id: int) -> Dict[str, Any]:
    """환자 단위 일별 집계 사전을 계산한다.
    반환 키:
      - lab_daily: { feature: {date: mean} }
      - chart_daily: { feature: {date: mean} }
      - rx_daily_flags: { rx_name: {date: 0/1} }
      - rx_events: { rx_name: [ {time,dose,unit,route}, ... ] }

    변경점(중요):
    - 📌 일별 맵의 키 타입을 모두 date로 '정규화'하여
      추후 윈도우 슬라이싱(3/7/14/28일)에서 날짜 매칭 문제가 발생하지 않도록 함.    
    """
    results: Dict[str, Any] = {
        "lab_daily": {},
        "chart_daily": {},
        "rx_daily_flags": {},
        "rx_events": {},
    }

    # ======================
    # Lab 일별 평균 집계
    # ======================
    lab_data = clinical_data.get("lab_events")
    lab_df = pd.DataFrame(lab_data) if isinstance(lab_data, dict) else lab_data

    if isinstance(lab_df, pd.DataFrame) and not lab_df.empty:
        # v2 방식으로 되돌림: 간단하고 직접적인 처리
        df = lab_df[(lab_df["subject_id"] == patient_id) & (pd.notna(lab_df.get("valuenum")))]
        if not df.empty:
            df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df["charttime"]):
                df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
            df = df[pd.notna(df["charttime"])]
            df["date"] = df["charttime"].dt.date
            for feature_name, itemids in FEATURE_ITEMIDS.items():
                # Lab 섹션: CHART_FEATURES는 제외
                if feature_name in CHART_FEATURES:
                    continue
                sub = df[df["itemid"].isin(itemids)]
                if not sub.empty:
                    daily = sub.groupby("date")["valuenum"].mean()
                    if len(daily) > 0:
                        results["lab_daily"][feature_name] = daily.to_dict()

    # ======================
    # Chart(바이탈) 일별 평균 집계
    # ======================
    chart_df: pd.DataFrame = clinical_data.get("chart_events")
    if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
        time_col = "event_time" if "event_time" in chart_df.columns else (
            "charttime" if "charttime" in chart_df.columns else None
        )
        if time_col and "valuenum" in chart_df.columns:
            # v2 방식으로 되돌림: 간단하고 직접적인 처리
            df = chart_df[(chart_df["subject_id"] == patient_id) & (pd.notna(chart_df["valuenum"]))].copy()
            if not df.empty:
                if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                df = df[pd.notna(df[time_col])]
                df["date"] = df[time_col].dt.date
                for feature_name, itemids in FEATURE_ITEMIDS.items():
                    # Chart 섹션: CHART_FEATURES만 포함
                    if feature_name not in CHART_FEATURES:
                        continue
                    sub = df[df["itemid"].isin(itemids)]
                    if not sub.empty:
                        if feature_name == "temperature":
                            sub = sub.copy()
                            # itemid 223761 = °F 기록 → °C 로 변환
                            f_mask = sub["itemid"] == 223761
                            sub.loc[f_mask, "valuenum"] = (
                                sub.loc[f_mask, "valuenum"] - 32
                            ) * 5 / 9
                            sub = sub[
                                (sub["valuenum"] >= 25) & (sub["valuenum"] <= 45)
                            ]
                        elif feature_name == "weight":
                            sub = sub.copy()
                            # itemid 226531 = lbs 기록(valueuom NaN) → kg 변환
                            lbs_mask = sub["itemid"] == 226531
                            sub.loc[lbs_mask, "valuenum"] = (
                                sub.loc[lbs_mask, "valuenum"] * 0.453592
                            )
                            sub = sub[
                                (sub["valuenum"] >= 20) & (sub["valuenum"] <= 300)
                            ]
                        daily = sub.groupby("date")["valuenum"].mean()
                        if len(daily) > 0:
                            results["chart_daily"][feature_name] = daily.to_dict()

    # ======================
    # Rx(EMAR → Input Events → Prescriptions) 하이브리드 추출
    # ======================
    _extract_hybrid_rx_features(clinical_data, patient_id, results)
    
    # NOTE: Procedures는 이제 프롬프트 피처가 아닌 데이터 필터링 기준으로 사용
    # should_exclude_timepoint_for_procedures() 함수를 환자 선별 단계에서 활용

    # ✅ lab_daily / chart_daily는 이미 date 키로 정규화되어 반환됨
    # rx_daily_flags는 값이 0/1 이므로 int로 유지, 키만 date로 통일
    norm_rx_flags: Dict[str, Dict[date, int]] = {}
    for rx, m in results["rx_daily_flags"].items():
        norm_map: Dict[date, int] = {}
        for k, v in dict(m).items():
            try:
                kk = pd.to_datetime(k).date()
                norm_map[kk] = int(bool(v))
            except Exception:
                pass
        norm_rx_flags[rx] = norm_map
    results["rx_daily_flags"] = norm_rx_flags

    return results


############################################################
# (5) RX: EMAR/Input/Prescriptions 추출
############################################################

def _extract_hybrid_rx_features(clinical_data: Dict[str, Any], patient_id: int, results: Dict[str, Any]) -> None:
    """
    3단계 처방 특징 추출: EMAR 우선 → Input Events 보완 → Prescriptions 백업
    
    Args:
        clinical_data: 임상 데이터 딕셔너리
        patient_id: 환자 ID
        results: 결과 저장용 딕셔너리 (rx_daily_flags, rx_events 업데이트)
    """
    
    # 1. EMAR 기반 추출 시도 (실제 투약 기록 - 최우선)
    emar_success = _extract_rx_from_emar(clinical_data, patient_id, results)
    
    # 2. EMAR 실패 시 Input Events 시도 (계획된 투여 기록)
    input_success = False
    if not emar_success:
        input_success = _extract_rx_from_input_events(clinical_data, patient_id, results)
    
    # 3. EMAR, Input Events 모두 실패 시 Prescriptions 백업 (처방 추정)
    if not emar_success and not input_success:
        _extract_rx_from_prescriptions(clinical_data, patient_id, results)


def _extract_rx_from_emar(clinical_data: Dict[str, Any], patient_id: int, results: Dict[str, Any]) -> bool:
    """
    EMAR 데이터에서 실제 투약 기록 추출
    
    Returns:
        bool: 성공 여부
    """
    emar_df: pd.DataFrame = clinical_data.get("emar")
    if not isinstance(emar_df, pd.DataFrame) or emar_df.empty:
        return False
    
    # 환자 데이터 필터링
    patient_emar = emar_df[emar_df["subject_id"] == patient_id].copy()
    if patient_emar.empty:
        return False
    
    # 시간 컬럼 처리
    time_col = "charttime" if "charttime" in patient_emar.columns else None
    if not time_col:
        return False
    
    # 시간 파싱
    ensure_datetime(patient_emar, [time_col])
    
    patient_emar = patient_emar[pd.notna(patient_emar[time_col])]
    if patient_emar.empty:
        return False
    
    patient_emar["date"] = patient_emar[time_col].dt.date
    
    # 약물 컬럼 확인
    drug_col = "medication" if "medication" in patient_emar.columns else None
    if not drug_col:
        return False
    
    # 약물별 일별 플래그 생성
    drug_series = patient_emar[drug_col].astype(str).str.lower()
    found_any = False
    
    for rx_name, keywords in RX_KEYWORDS.items():
        # 키워드 매칭
        mask = False
        for keyword in keywords:
            mask = mask | drug_series.str.contains(keyword, na=False)
        
        matched_records = patient_emar[mask]
        if matched_records.empty:
            continue
        
        # 일별 투여 플래그 (1 = 실제 투여됨)
        daily_flags = (matched_records.groupby("date").size() > 0).astype(int)
        results["rx_daily_flags"][rx_name] = daily_flags.to_dict()
        
        # 투여 이벤트 저장
        events = []
        for _, row in matched_records.iterrows():
            events.append({
                "time": row[time_col],
                "dose": row.get("medication", ""),  # EMAR에는 정확한 용량 정보가 제한적
                "unit": "",
                "route": "",
                "source": "emar"  # 데이터 소스 표시
            })
        
        results["rx_events"][rx_name] = _sort_by_time(events)
        found_any = True
    
    return found_any


def _extract_rx_from_input_events(clinical_data: Dict[str, Any], patient_id: int, results: Dict[str, Any]) -> bool:
    """
    Input Events 데이터에서 약물 투여 기록 추출 (EMAR 보완용)
    
    Returns:
        bool: 성공 여부
    """
    input_df: pd.DataFrame = clinical_data.get("input_events")
    if not isinstance(input_df, pd.DataFrame) or input_df.empty:
        return False
    
    # 환자 데이터 필터링
    patient_input = input_df[input_df["subject_id"] == patient_id].copy()
    if patient_input.empty:
        return False
    
    # 시간 컬럼 처리
    time_col = "starttime" if "starttime" in patient_input.columns else "storetime"
    if time_col not in patient_input.columns:
        return False
    
    # 시간 파싱
    ensure_datetime(patient_input, [time_col])
    
    patient_input = patient_input[pd.notna(patient_input[time_col])]
    if patient_input.empty:
        return False
    
    patient_input["date"] = patient_input[time_col].dt.date
    
    # 약물 컬럼 확인 (Input Events의 경우 'label' 또는 'category' 사용)
    drug_col = None
    for col in ["label", "category", "itemid"]:
        if col in patient_input.columns:
            drug_col = col
            break
    
    if not drug_col:
        return False
    
    # 신장질환 관련 약물 매칭 (EMAR과 동일한 키워드 사용)
    drug_series = patient_input[drug_col].astype(str).str.lower()
    
    found_any = False
    for rx_name, keywords in RX_KEYWORDS.items():
        if results["rx_events"].get(rx_name):  # 이미 EMAR에서 추출된 경우 건너뛰기
            continue
            
        # 키워드 매칭 (EMAR과 동일한 방식)
        mask = False
        for keyword in keywords:
            mask = mask | drug_series.str.contains(keyword, na=False)
        
        matched_records = patient_input[mask]
        if matched_records.empty:
            continue
        
        found_any = True
        events = []
        
        for _, row in matched_records.iterrows():
            events.append({
                "time": row[time_col],
                "date": row["date"],
                "dose": row.get("amount", ""),  # Input Events의 용량 정보
                "unit": row.get("amountuom", ""),  # 용량 단위
                "route": row.get("ordercategoryname", ""),  # 투여 경로
                "source": "input_events"  # 데이터 소스 표시
            })
        
        # 시간순 정렬
        results["rx_events"][rx_name] = _sort_by_time(events)
    
    return found_any


############################################################
# (6) PROCEDURES: 시술 기반 제외 로직
############################################################

def should_exclude_timepoint_for_procedures(clinical_data: Dict[str, Any], patient_id: int, target_time: pd.Timestamp) -> Dict[str, Any]:
    """
    시술/수술로 인해 해당 시점 데이터를 제외해야 하는지 확인
    
    Returns:
        Dict with exclusion info: {"exclude": bool, "reasons": List[str], "procedures": List[Dict]}
    """
    procedures_df: pd.DataFrame = clinical_data.get("procedures")
    if not isinstance(procedures_df, pd.DataFrame) or procedures_df.empty:
        return {"exclude": False, "reasons": [], "procedures": []}
    
    # 환자 데이터 필터링
    patient_procedures = procedures_df[procedures_df["subject_id"] == patient_id].copy()
    if patient_procedures.empty:
        return {"exclude": False, "reasons": [], "procedures": []}
    
    # 시간 컬럼 처리
    time_col = "chartdate" if "chartdate" in patient_procedures.columns else None
    if not time_col:
        return {"exclude": False, "reasons": [], "procedures": []}
    
    # 날짜 파싱
    if not pd.api.types.is_datetime64_any_dtype(patient_procedures[time_col]):
        try:
            patient_procedures[time_col] = pd.to_datetime(patient_procedures[time_col], errors="coerce", format="mixed")
        except TypeError:
            patient_procedures[time_col] = pd.to_datetime(patient_procedures[time_col], errors="coerce")
    
    patient_procedures = patient_procedures[pd.notna(patient_procedures[time_col])]
    if patient_procedures.empty:
        return {"exclude": False, "reasons": [], "procedures": []}
    
    # ICD 코드 컬럼 확인
    code_col = "icd_code" if "icd_code" in patient_procedures.columns else None
    if not code_col:
        return {"exclude": False, "reasons": [], "procedures": []}
    
    code_series = patient_procedures[code_col].astype(str).str.lower()
    exclusion_reasons = []
    conflicting_procedures = []
    
    # 각 제외 기준별로 확인
    for proc_name, config in EXCLUSION_PROCEDURES.items():
        # 키워드 소문자화하여 대소문자 불일치 제거
        keywords = [str(k).lower() for k in config["keywords"]]
        hours_before = int(config["exclusion_hours_before"]) or 0
        hours_after = int(config["exclusion_hours_after"]) or 0
        # 옵션: 라벨 윈도우 최대길이(28d) + 회복기간을 after에 가산
        # (구 옵션) 회복기간 확장 개념은 제거. 필요 시 after 값을 직접 상향 조정
        extend = bool(globals().get("_procedure_extend_after_flag", False))
        if extend:
            hours_after += WINDOW_MAX_DAYS * 24
        reason = config["reason"]
        
        # 키워드 매칭
        mask = False
        for keyword in keywords:
            mask = mask | code_series.str.contains(keyword, na=False)
        
        matched_procedures = patient_procedures[mask]
        if matched_procedures.empty:
            continue
        
        # 시점 기준으로 제외 구간 확인
        for _, row in matched_procedures.iterrows():
            proc_time = row[time_col]
            window_start = proc_time - pd.Timedelta(hours=hours_before)
            window_end = proc_time + pd.Timedelta(hours=hours_after)
            
            # target_time이 제외 구간 안에 있는지 확인
            if window_start <= target_time <= window_end:
                exclusion_reasons.append(f"{proc_name}: {reason}")
                conflicting_procedures.append({
                    "procedure": proc_name,
                    "time": proc_time,
                    "icd_code": row.get(code_col, ""),
                    "window_start": window_start,
                    "window_end": window_end,
                    "reason": reason
                })
    
    return {
        "exclude": len(exclusion_reasons) > 0,
        "reasons": exclusion_reasons,
        "procedures": conflicting_procedures
    }


    


def _extract_rx_from_prescriptions(clinical_data: Dict[str, Any], patient_id: int, results: Dict[str, Any]) -> None:
    """Prescriptions 데이터에서 처방 기록을 추정한다(백업 경로, 동작 동일)."""
    rx_df: pd.DataFrame = clinical_data.get("prescriptions")
    if not isinstance(rx_df, pd.DataFrame) or rx_df.empty:
        return
    
    # 기존 로직 (처방 시작일 기반)
    tcol = "starttime" if "starttime" in rx_df.columns else ("event_time" if "event_time" in rx_df.columns else None)
    if not tcol:
        return
    
    df = rx_df[rx_df["subject_id"] == patient_id].copy()
    if df.empty:
        return
    
    # 시간 파싱
    ensure_datetime(df, [tcol])
    
    df = df[pd.notna(df[tcol])]
    if df.empty:
        return
    
    df["date"] = df[tcol].dt.date
    
    # 약물 컬럼 확인
    drug_col = None
    for c in ["drug", "drug_name", "medication"]:
        if c in df.columns:
            drug_col = c
            break
    
    if not drug_col:
        return
    
    # 약물별 처방 패턴 생성
    drug_series = df[drug_col].astype(str).str.lower()
    
    for rx_name, keywords in RX_KEYWORDS.items():
        # 이미 EMAR에서 추출된 약물은 건너뛰기
        if rx_name in results["rx_daily_flags"]:
            continue
        
        # 키워드 매칭
        mask = False
        for keyword in keywords:
            mask = mask | drug_series.str.contains(keyword, na=False)
        
        matched_records = df[mask]
        if matched_records.empty:
            continue
        
        # 처방 시작일 기반 플래그 (추정)
        daily_flags = (matched_records.groupby("date").size() > 0).astype(int)
        results["rx_daily_flags"][rx_name] = daily_flags.to_dict()
        
        # 처방 이벤트 저장
        events = []
        for _, row in matched_records.iterrows():
            events.append({
                "time": row[tcol],
                "dose": row.get("dose_val_rx", ""),
                "unit": row.get("dose_unit_rx", ""),
                "route": row.get("route", ""),
                "source": "prescription"  # 데이터 소스 표시
            })
        
        results["rx_events"][rx_name] = _sort_by_time(events)

############################################################
# (8) SNAPSHOT: 최근 스냅샷/원시값 조회
############################################################

def build_recent_snapshot(
    clinical_data: Dict[str, Any],
    patient_id: int,
    target_time: pd.Timestamp,
    lookback_hours: int = FALLBACK_LOOKBACK_HOURS,
    exclude_lab_features: Optional[List[str]] = None,
    hours_candidates: Optional[List[int]] = None,
    allow_equal_end: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Optional[int]]:
    """최근 lookback_hours 내 가장 최근 측정값 스냅샷을 구축한다.
    - lab: FEATURE_ITEMIDS 중 차트 피처와 exclude_lab_features를 제외
    - chart: CHART_FEATURES만
    반환: { 'lab': {feature: value}, 'chart': {feature: value} }
    """
    if exclude_lab_features is None:
        exclude_lab_features = []

    snapshot: Dict[str, Dict[str, float]] = {"lab": {}, "chart": {}}
    hour_options: List[int] = hours_candidates if hours_candidates else [lookback_hours]

    for hours in hour_options:
        snapshot = {"lab": {}, "chart": {}}
        start_time = target_time - pd.Timedelta(hours=hours)

        # Lab snapshot
        lab_df: pd.DataFrame = clinical_data.get("lab_events")
        if isinstance(lab_df, pd.DataFrame) and not lab_df.empty:
            lab_mask = (
                (lab_df["subject_id"] == patient_id)
                & (lab_df["charttime"] >= start_time)
                & ((lab_df["charttime"] <= target_time) if allow_equal_end else (lab_df["charttime"] < target_time))
                & (pd.notna(lab_df["valuenum"]))
            )
            lab_window = lab_df[lab_mask]
            if not lab_window.empty:
                for feature_name, itemids in FEATURE_ITEMIDS.items():
                    if feature_name in CHART_FEATURES or feature_name in exclude_lab_features:
                        continue
                    sub = lab_window[lab_window["itemid"].isin(itemids)]
                    if not sub.empty:
                        latest = sub.sort_values("charttime").iloc[-1]
                        try:
                            snapshot["lab"][feature_name] = float(latest["valuenum"])
                        except Exception:
                            pass

        # Chart snapshot
        chart_df: pd.DataFrame = clinical_data.get("chart_events")
        if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
            time_col = "event_time" if "event_time" in chart_df.columns else ("charttime" if "charttime" in chart_df.columns else None)
            if time_col:
                chart_mask = (
                    (chart_df["subject_id"] == patient_id)
                    & (chart_df[time_col] >= start_time)
                    & ((chart_df[time_col] <= target_time) if allow_equal_end else (chart_df[time_col] < target_time))
                    & (pd.notna(chart_df["valuenum"]))
                )
                chart_window = chart_df[chart_mask]
                if not chart_window.empty:
                    for feature_name, itemids in FEATURE_ITEMIDS.items():
                        if feature_name not in CHART_FEATURES:
                            continue
                        sub = chart_window[chart_window["itemid"].isin(itemids)]
                        if not sub.empty:
                            latest = sub.sort_values(time_col).iloc[-1]
                            try:
                                snapshot["chart"][feature_name] = float(latest["valuenum"])
                            except Exception:
                                pass

        # return if we captured anything
        if snapshot["lab"] or snapshot["chart"]:
            return snapshot, hours

    return snapshot, None


def get_recent_raw_lab_values(
    clinical_data: Dict[str, Any],
    patient_id: int,
    feature_name: str,
    target_time: pd.Timestamp,
    lookback_hours: int = 48,
    allow_equal_end: bool = True,
) -> List[float]:
    """최근 lookback_hours 내 특정 랩 피처의 모든 원시값을 시간 오름차순으로 반환한다."""
    itemids = FEATURE_ITEMIDS.get(feature_name, [])
    if not itemids:
        return []
    lab_df: pd.DataFrame = clinical_data.get("lab_events")
    if not isinstance(lab_df, pd.DataFrame) or lab_df.empty:
        return []
    df = lab_df[(lab_df["subject_id"] == patient_id) & (lab_df["itemid"].isin(itemids))].copy()
    if df.empty:
        return []
    if not pd.api.types.is_datetime64_any_dtype(df.get("charttime")):
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    start_time = target_time - pd.Timedelta(hours=lookback_hours)
    mask = (df["charttime"] >= start_time) & ((df["charttime"] <= target_time) if allow_equal_end else (df["charttime"] < target_time))
    window = df[mask]
    if window.empty:
        return []
    window = window.sort_values("charttime")
    values: List[float] = []
    for v in window["valuenum"]:
        if pd.notna(v):
            try:
                values.append(float(v))
            except Exception:
                pass
    return values


def compute_kdigo_evidence_at_time(
    clinical_data: Dict[str, Any],
    patient_id: int,
    target_time: pd.Timestamp,
    baseline_quantile: Optional[float] = None,
) -> Dict[str, Any]:
    """
    KDIGO 표 기준 요약 근거:
      - 진단(positive): Δ48h ≥ 0.3 mg/dL  OR  7d ratio ≥ 1.5
      - 스테이징: ratio 1.5–1.9 →1, 2.0–2.9 →2, ≥3.0 →3
        (단, SCr ≥4.0 mg/dL은 '진단이 성립된 뒤' stage 3로 격상시키는 보조 기준)
    """
    lab_df: pd.DataFrame = clinical_data.get("lab_events")
    creat_ids = FEATURE_ITEMIDS.get("creatinine", [])
    if not isinstance(lab_df, pd.DataFrame) or lab_df.empty or not creat_ids:
        return {"last_time": None, "last_scr": None, "min48": None, "delta48": None, "meet48": False,
                "baseline7d": None, "ratio7": None, "meet7": False, "diagnose": False, "stage": 0}

    df = lab_df[(lab_df["subject_id"] == patient_id) & (lab_df["itemid"].isin(creat_ids)) & (pd.notna(lab_df.get("valuenum")))].copy()
    if df.empty:
        return {"last_time": None, "last_scr": None, "min48": None, "delta48": None, "meet48": False,
                "baseline7d": None, "ratio7": None, "meet7": False, "diagnose": False, "stage": 0}

    if not pd.api.types.is_datetime64_any_dtype(df["charttime"]):
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df[pd.notna(df["charttime"]) & (df["charttime"] <= target_time)].sort_values("charttime")
    if df.empty:
        return {"last_time": None, "last_scr": None, "min48": None, "delta48": None, "meet48": False,
                "baseline7d": None, "ratio7": None, "meet7": False, "diagnose": False, "stage": 0}

    last_time = df.iloc[-1]["charttime"]
    last_scr = float(df.iloc[-1]["valuenum"]) if pd.notna(df.iloc[-1]["valuenum"]) else None

    # 48h 기준
    prior_48 = df[(df["charttime"] >= target_time - pd.Timedelta(hours=48)) & (df["charttime"] < target_time)]
    min48 = float(prior_48["valuenum"].min()) if (not prior_48.empty and pd.notna(prior_48["valuenum"]).any()) else None
    delta48 = (last_scr - min48) if (last_scr is not None and min48 is not None) else None
    meet48 = (delta48 is not None) and (delta48 >= 0.3)

    # 7d 기준
    prior_7d = df[(df["charttime"] >= target_time - pd.Timedelta(days=7)) & (df["charttime"] < target_time)]
    if not prior_7d.empty and pd.notna(prior_7d["valuenum"]).any():
        if baseline_quantile is not None and 0 < baseline_quantile < 1:
            baseline7d = float(prior_7d["valuenum"].quantile(baseline_quantile))
        else:
            baseline7d = float(prior_7d["valuenum"].min())
    else:
        baseline7d = None
    ratio7 = (last_scr / baseline7d) if (last_scr is not None and baseline7d is not None and baseline7d > 0) else None
    meet7 = (ratio7 is not None) and (ratio7 >= 1.5)

    # 진단 여부
    diagnose = bool(meet48 or meet7)

    # 스테이징(진단 성립 시에만)
    stage_ratio = 0
    if ratio7 is not None:
        if ratio7 >= 3.0:
            stage_ratio = 3
        elif ratio7 >= 2.0:
            stage_ratio = 2
        elif ratio7 >= 1.5:
            stage_ratio = 1
    stage_48 = 1 if meet48 else 0
    stage = max(stage_ratio, stage_48) if diagnose else 0
    # SCr ≥4.0 mg/dL은 진단이 이미 성립된 경우 stage 3로 격상
    if diagnose and (last_scr is not None) and (last_scr >= 4.0):
        stage = max(stage, 3)

    return {
        "last_time": str(last_time) if pd.notna(last_time) else None,
        "last_scr": last_scr,
        "min48": min48,
        "delta48": delta48,
        "meet48": meet48,
        "baseline7d": baseline7d,
        "ratio7": ratio7,
        "meet7": meet7,
        "diagnose": diagnose,
        "stage": int(stage),
    }


def get_patient_demographics(complete_data: Dict[str, Any], patient_id: int) -> Dict[str, Any]:
    """환자 성별/나이 등 기본 정보를 반환. 없으면 비워서 반환."""
    base_info = complete_data.get("base_info", {})
    patients_info = base_info.get("patients_info")
    demographics: Dict[str, Any] = {}
    if isinstance(patients_info, pd.DataFrame) and not patients_info.empty:
        if "subject_id" in patients_info.columns:
            rows = patients_info[patients_info["subject_id"] == patient_id]
            if len(rows) > 0:
                row = rows.iloc[0]
                # 성별
                gender = row.get("gender") if "gender" in rows.columns else row.get("sex")
                if pd.notna(gender):
                    demographics["gender"] = str(gender)
                # 나이 (anchor_age 또는 age)
                age_val = None
                if "anchor_age" in rows.columns:
                    age_val = rows.iloc[0]["anchor_age"]
                elif "age" in rows.columns:
                    age_val = rows.iloc[0]["age"]
                if pd.notna(age_val):
                    try:
                        demographics["age"] = int(age_val)
                    except Exception:
                        pass
    return demographics


def get_patient_background(
    complete_data: Dict[str, Any],
    clinical_data: Dict[str, Any],
    patient_id: int,
    target_time: Optional[pd.Timestamp] = None,
    lang: str = "en",
) -> Dict[str, Any]:
    """환자의 종합 배경 정보를 구성 (v11 신규).

    Returns:
        Dict with demographics, ckd_stage, cci_score, comorbidity_list,
        admission_context, weight_info.
    """
    base_info = complete_data.get("base_info", {})

    # 기본 demographics (기존 로직 재사용)
    demographics = get_patient_demographics(complete_data, patient_id)
    age = demographics.get("age")
    gender = demographics.get("gender")

    # diagnoses_icd DataFrame
    diagnoses_df = base_info.get("diagnoses")
    if diagnoses_df is None:
        diagnoses_df = base_info.get("diagnoses_icd")
    if not isinstance(diagnoses_df, pd.DataFrame):
        diagnoses_df = pd.DataFrame()

    # admissions DataFrame
    admissions_df = base_info.get("admissions")
    if not isinstance(admissions_df, pd.DataFrame):
        admissions_df = pd.DataFrame()

    # build_patient_background (config.comorbidity_definitions)
    if not diagnoses_df.empty:
        bg = build_patient_background(
            patient_id=patient_id,
            diagnoses_df=diagnoses_df,
            admissions_df=admissions_df if not admissions_df.empty else None,
            target_time=target_time,
            age=age,
            gender=gender,
            lang=lang,
        )
    else:
        bg = {
            "demographics": demographics,
            "ckd_stage": None,
            "cci_score": 0,
            "cci_components": {},
            "comorbidities": {},
            "comorbidity_list": [],
            "admission_context": {},
        }

    # 체중 정보 (chartevents에서)
    chart_data = clinical_data.get("chart_events")
    if chart_data is not None:
        chart_df = pd.DataFrame(chart_data) if isinstance(chart_data, dict) else chart_data
        if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
            wi = extract_weight_info(chart_df, target_time) if target_time else {}
        else:
            wi = {}
    else:
        wi = {}
    bg["weight_info"] = wi

    # Weight trajectory (입원별 장기 체중 추이)
    if isinstance(chart_df, pd.DataFrame) and not chart_df.empty and not admissions_df.empty:
        bg["weight_trajectory"] = extract_weight_trajectory(
            chart_events=chart_df,
            admissions_df=admissions_df,
            patient_id=patient_id,
            target_time=target_time,
        )
    else:
        bg["weight_trajectory"] = {}

    # eGFR trajectory (Cr-derived, CKD-EPI 2021 Refit)
    lab_data = clinical_data.get("lab_events")
    pts_info = base_info.get("patients_info")
    anchor_age_val = None
    anchor_year_val = None
    if isinstance(pts_info, pd.DataFrame) and not pts_info.empty:
        pt_row = pts_info[pts_info["subject_id"] == patient_id]
        if not pt_row.empty:
            anchor_age_val = pt_row.iloc[0].get("anchor_age")
            anchor_year_val = pt_row.iloc[0].get("anchor_year")

    if isinstance(lab_data, pd.DataFrame) and not lab_data.empty and not admissions_df.empty:
        egfr_traj = extract_egfr_trajectory(
            patient_id=patient_id,
            lab_events=lab_data,
            admissions_df=admissions_df,
            age=age,
            gender=gender,
            anchor_age=anchor_age_val,
            anchor_year=anchor_year_val,
            target_time=target_time,
        )
    else:
        egfr_traj = {}
    bg["egfr_trajectory"] = egfr_traj

    # RRT (투석/CRRT) 상태
    proc_data = clinical_data.get("procedures")
    if isinstance(proc_data, pd.DataFrame) and not proc_data.empty:
        bg["rrt_status"] = extract_rrt_status(
            patient_id, proc_data,
            admissions_df=admissions_df if not admissions_df.empty else None,
            target_time=target_time,
        )
        bg["contrast_exposure"] = extract_contrast_exposure(
            patient_id, proc_data,
            target_time=target_time,
            lookback_days=7,
        )
        bg["major_procedures"] = extract_major_procedures(
            patient_id, proc_data,
            target_time=target_time,
        )
    else:
        bg["rrt_status"] = {}
        bg["contrast_exposure"] = {}
        bg["major_procedures"] = {}

    # Vasopressor / IV Fluid (input_events 기반, ICU 환자)
    ie_data = clinical_data.get("input_events")
    _ie_is_df = isinstance(ie_data, pd.DataFrame)
    _ie_empty = ie_data.empty if _ie_is_df else True
    if _ie_is_df and not _ie_empty:
        bg["vasopressor_status"] = extract_vasopressor_status(
            patient_id, ie_data,
            target_time=target_time,
            lookback_days=7,
        )
        bg["fluid_status"] = extract_fluid_status(
            patient_id, ie_data,
            target_time=target_time,
            lookback_days=3,
        )
    else:
        bg["vasopressor_status"] = {}
        bg["fluid_status"] = {}

    bg["_target_time"] = target_time

    return bg


def format_patient_background_for_prompt(
    bg: Dict[str, Any],
    lang: str = "en",
) -> str:
    """환자 배경 정보를 프롬프트 문자열로 변환.

    config.comorbidity_definitions.format_background_for_prompt에 위임 +
    체중 정보 추가 (v11 전용).
    """
    base = format_background_for_prompt(bg, lang=lang)

    # weight는 이제 Vitals 시계열(일별 평균)에 포함되므로 background에서 제거

    return base if base else ("No information" if lang == "en" else "정보 없음")


def format_prompt_time(ts: Any) -> str:
    """프롬프트 표시용 시간 포맷. 비식별 연도 혼선을 줄이기 위해 연도는 생략한다."""
    try:
        if isinstance(ts, pd.Timestamp):
            return ts.strftime("%m-%d %H:%M")
        # 문자열 형태라면 pandas로 파싱 시도
        parsed = pd.to_datetime(ts)
        if pd.notna(parsed):
            return parsed.strftime("%m-%d %H:%M")
    except Exception:
        pass
    return str(ts)





def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_kdigo_onset_for_patient(clinical_data: Dict[str, Any], patient_id: int) -> Optional[Dict[str, Any]]:
    """크레아티닌(SCr) 기반 KDIGO onset 계산(시간 제약 엄격 적용).
    - 48시간 내 +0.3 mg/dL 상승 또는
      최근 7일 기준 baseline 대비 1.5x/2.0x/3.0x 상승을 최초로 만족하는 시점을 onset으로 간주
    - ratio 기반 stage 산정: 1.5x→1, 2.0x→2, 3.0x→3 (48h +0.3은 stage=1 최소 보장)
    - baseline은 "타깃 시점 이전 7일" 구간의 최솟값을 사용(시점 포함하지 않음)
    - 반환: { 'onset_time': Timestamp, 'kdigo_stage': int(1/2/3), 'baseline_scr': float }
    """
    lab_df: pd.DataFrame = clinical_data.get("lab_events")
    if not isinstance(lab_df, pd.DataFrame) or lab_df.empty:
        return None
    scr_df = lab_df[(lab_df["subject_id"] == patient_id) & (lab_df["itemid"].isin(FEATURE_ITEMIDS.get("creatinine", [])))].copy()
    if scr_df.empty:
        return None
    scr_df["charttime"] = pd.to_datetime(scr_df["charttime"])
    scr_df = scr_df.sort_values("charttime")
    # 48h/7d 창 탐색 (baseline은 각 시점의 직전 7일 최소)
    for _, row in scr_df.iterrows():
        t = row["charttime"]
        v_raw = row["valuenum"]
        if not pd.notna(v_raw):
            continue
        v = float(v_raw)

        # 7일 baseline: (t-7d, t) 구간의 최소값
        prior_7d = scr_df[(scr_df["charttime"] >= t - pd.Timedelta(days=7)) & (scr_df["charttime"] < t)]
        baseline_7d = float(prior_7d["valuenum"].min()) if (not prior_7d.empty and pd.notna(prior_7d["valuenum"]).any()) else None

        # ratio 기반 stage
        stage_ratio = 0
        if baseline_7d is not None and baseline_7d > 0:
            ratio = v / baseline_7d
            if ratio >= 3.0:
                stage_ratio = 3
            elif ratio >= 2.0:
                stage_ratio = 2
            elif ratio >= 1.5:
                stage_ratio = 1

        # 48h +0.3 기준 (이전 48h의 최소와 비교)
        prior_48h = scr_df[(scr_df["charttime"] >= t - pd.Timedelta(hours=48)) & (scr_df["charttime"] < t)]
        stage_48h = 0
        if not prior_48h.empty and pd.notna(prior_48h["valuenum"]).any():
            min48 = float(prior_48h["valuenum"].min())
            if (v - min48) >= 0.3:
                stage_48h = 1

        stage = max(stage_ratio, stage_48h)
        if stage >= 1:
            return {"onset_time": t, "kdigo_stage": int(stage), "baseline_scr": baseline_7d if baseline_7d is not None else float('nan')}
    return None


def compute_kdigo_events_in_window(
    clinical_data: Dict[str, Any],
    patient_id: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    baseline_quantile: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    주어진 창 안에서 KDIGO 충족(진단 성립) 시점을 모두 반환.
    - 진단: Δ48h ≥ 0.3 mg/dL  OR  7d ratio ≥ 1.5
    - 스테이징: ratio(1/2/3)와 Δ48h(최소 1) 중 큰 값. 진단 성립 시 SCr ≥4.0 → stage 3로 격상.
    ※ 절대값(≥4.0)만으로 양성으로 판단하지 않음.
    """
    lab_df: pd.DataFrame = clinical_data.get("lab_events")
    creat_ids = FEATURE_ITEMIDS.get("creatinine", [])
    events: List[Dict[str, Any]] = []
    if not isinstance(lab_df, pd.DataFrame) or lab_df.empty or not creat_ids:
        return events

    df = lab_df[(lab_df["subject_id"] == patient_id) & (lab_df["itemid"].isin(creat_ids)) & (pd.notna(lab_df.get("valuenum")))].copy()
    if df.empty:
        return events
    if not pd.api.types.is_datetime64_any_dtype(df["charttime"]):
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df[pd.notna(df["charttime"])].sort_values("charttime")

    # 창 내부 후보만 검사
    df_in = df[(df["charttime"] >= window_start) & (df["charttime"] <= window_end)]
    for _, row in df_in.iterrows():
        t = row["charttime"]
        v = float(row["valuenum"]) if pd.notna(row["valuenum"]) else None
        if v is None:
            continue

        prior = df[df["charttime"] < t]

        # 48h
        prior_48 = prior[prior["charttime"] >= t - pd.Timedelta(hours=48)]
        min48 = float(prior_48["valuenum"].min()) if (not prior_48.empty and pd.notna(prior_48["valuenum"]).any()) else None
        delta48 = (v - min48) if (min48 is not None) else None
        meet48 = (delta48 is not None) and (delta48 >= 0.3)
        stage_48 = 1 if meet48 else 0

        # 7d baseline  (절대값 조건 제거)
        prior_7d = prior[prior["charttime"] >= t - pd.Timedelta(days=7)]
        if not prior_7d.empty and pd.notna(prior_7d["valuenum"]).any():
            if baseline_quantile is not None and 0 < baseline_quantile < 1:
                baseline7 = float(prior_7d["valuenum"].quantile(baseline_quantile))
            else:
                baseline7 = float(prior_7d["valuenum"].min())
        else:
            baseline7 = None
        ratio = (v / baseline7) if (baseline7 is not None and baseline7 > 0) else None

        stage_ratio = 0
        meet7 = False
        if ratio is not None:
            if ratio >= 3.0:
                stage_ratio = 3
            elif ratio >= 2.0:
                stage_ratio = 2
            elif ratio >= 1.5:
                stage_ratio = 1
            meet7 = ratio >= 1.5

        # 진단 성립 여부
        diagnose = bool(meet48 or meet7)
        if not diagnose:
            continue  # 이 시점은 AKI 양성이 아님

        # 스테이징
        stage = max(stage_48, stage_ratio)
        if v is not None and v >= 4.0:
            stage = max(stage, 3)

        events.append({
            "onset_time": pd.Timestamp(t),
            "kdigo_stage": int(stage),
            "baseline_scr": baseline7,
            "evidence": {
                "last_scr": v,
                "min48": min48,
                "delta48": delta48,
                "meet48": meet48,
                "baseline7d": baseline7,
                "ratio7": ratio,
                "meet7": meet7,
                "diagnose": True,
            }
        })

    return events



def get_test_patients(complete_data: Dict[str, Any], processed_path: Path, selection_file: str, max_patients: int) -> List[int]:
    selection_path = processed_path / selection_file
    if selection_path.exists():
        info = load_pickle(selection_path)
        recommended = list(info.get("recommended_patients", []))
        return recommended[:max_patients] if max_patients else recommended

    # fallback: eGFR 측정 수 기준 상위 환자 선별
    clinical_data = complete_data["clinical_data"]
    lab_df: pd.DataFrame = clinical_data["lab_events"]
    egfr_itemids = set(FEATURE_ITEMIDS["egfr"])
    egfr_df = lab_df[(lab_df["itemid"].isin(egfr_itemids)) & (pd.notna(lab_df["valuenum"]))]
    patient_ids = egfr_df["subject_id"].unique()

    egfr_counts: Dict[int, int] = {}
    for patient_id in patient_ids:
        egfr_counts[patient_id] = len(
            egfr_df[(egfr_df["subject_id"] == patient_id)]
        )

    sorted_patients = sorted(egfr_counts.items(), key=lambda x: x[1], reverse=True)
    selected = [pid for pid, _ in sorted_patients[: (max_patients or 50)]]
    return selected


def get_valid_measurement_timepoints(
    patient_id: int,
    clinical_data: Dict[str, Any],
    data_type: str = "egfr",
    max_labels: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    if data_type != "egfr":
        return [], 0

    lab_df: pd.DataFrame = clinical_data["lab_events"]
    egfr_itemids = set(FEATURE_ITEMIDS["egfr"])

    patient_egfr = lab_df[
        (lab_df["subject_id"] == patient_id)
        & (lab_df["itemid"].isin(egfr_itemids))
        & (pd.notna(lab_df["valuenum"]))
    ].copy()

    if len(patient_egfr) == 0:
        return [], 0

    patient_egfr = patient_egfr.sort_values("charttime").drop_duplicates(subset=["charttime"], keep="first")

    timepoints: List[Dict[str, Any]] = []
    excluded_by_procedures = 0
    count_limit = max_labels if isinstance(max_labels, int) and max_labels > 0 and max_labels != -1 else None
    
    for row in patient_egfr.itertuples(index=False):
        """
        특정 시점에서 eGFR 데이터를 판단하는 함수.
        - 제외: 시술/수술 기간 내 데이터
        """
        measurement_time = getattr(row, "charttime")
        
        # 🚫 Procedure 제외 조건 확인 (일상생활 기반 예측)
        exclusion_info = should_exclude_timepoint_for_procedures(clinical_data, patient_id, measurement_time)
        if exclusion_info["exclude"]:
            excluded_by_procedures += 1
            # 전역 집계에 반영 (메인 함수에서 초기화된 dict 사용)
            stats = globals().get("procedure_exclusion_stats")
            if isinstance(stats, dict):
                stats["excluded_count"] = stats.get("excluded_count", 0) + 1
                stats.setdefault("excluded_patients", set()).add(int(patient_id))
                reasons_map = stats.setdefault("exclusion_reasons", {})
                for reason in exclusion_info.get("reasons", []):
                    reasons_map[reason] = reasons_map.get(reason, 0) + 1
            continue
        
        timepoints.append(
            {
                "patient_id": int(patient_id),
                "measurement_time": measurement_time,
                "target_value": float(getattr(row, "valuenum")),
                "itemid": int(getattr(row, "itemid")),
                "measurement_type": "egfr",
            }
        )
        if count_limit is not None and len(timepoints) >= count_limit:
            break
    
    # 통계 정보는 상위 레벨에서 처리하므로 여기서는 출력하지 않음
    
    return timepoints, excluded_by_procedures


def get_aki_status_at_timepoint(complete_data: Dict[str, Any], patient_id: int, target_time: pd.Timestamp) -> bool:
    """
    특정 시점에서 AKI 상태를 판단하는 함수.
    - 진단: 특정 ICD 코드 포함
    - 입원: 특정 기간 내 입원
    """
    base_info = complete_data.get("base_info", {})
    diag_df: pd.DataFrame = base_info.get("diagnoses_icd")
    if not isinstance(diag_df, pd.DataFrame):
        return False

    patient_diag = diag_df[diag_df["subject_id"] == patient_id]
    if len(patient_diag) == 0:
        return False

    aki_diagnoses: List[pd.Series] = []
    for _, row in patient_diag.iterrows():
        icd_code = str(row.get("icd_code", ""))
        if any(code in icd_code for code in AKI_CODES):
            aki_diagnoses.append(row)

    if not aki_diagnoses:
        return False

    admissions: pd.DataFrame = base_info.get("admissions")
    if not isinstance(admissions, pd.DataFrame):
        # 진단만 있고 입원정보가 없으면 보수적으로 False
        return False

    for diag in aki_diagnoses:
        hadm_id = diag.get("hadm_id")
        if not hadm_id:
            continue
        admission = admissions[(admissions["subject_id"] == patient_id) & (admissions["hadm_id"] == hadm_id)]
        if len(admission) == 0:
            continue
        adm_record = admission.iloc[0]
        admit_time = pd.to_datetime(adm_record.get("admittime"))
        discharge_time = pd.to_datetime(adm_record.get("dischtime"))
        if pd.notna(admit_time) and pd.notna(discharge_time) and admit_time <= target_time <= discharge_time:
            return True

    return False

############################################################
# (9) FEATURES: 윈도우 피처 슬라이싱
############################################################

def extract_windowed_features(clinical_data: Dict[str, Any], patient_id: int, target_time: pd.Timestamp, window_days: int, lang: str = "ko", end_time: Optional[pd.Timestamp] = None, rx_format: str = "compact") -> Dict[str, Dict]:
    """특정 윈도우 기간의 피처들을 캐시에서 슬라이스하여 반환
    
    Args:
        target_time: 윈도우의 기준 시점 (start_time 계산용)
        window_days: 윈도우 일수
        end_time: 데이터 추출 종료 시점 (None이면 target_time 사용, Prognosis 모드에서는 cutoff_time 전달)
        rx_format: "compact"(range-notation, 토큰 절약) 또는 "legacy"(기존 상세 형식)
    """
    if window_days == 0:
        return {}

    # 캐시 준비 (전역 캐시, 읽기 전용으로 안전)
    if patient_id not in PRECOMPUTED_DAILIES:
        PRECOMPUTED_DAILIES[patient_id] = precompute_patient_daily_maps(clinical_data, patient_id)
    cache = PRECOMPUTED_DAILIES[patient_id]

    # end_time이 지정되지 않으면 target_time 사용 (기존 동작)
    actual_end_time = end_time if end_time is not None else target_time

    start_time = target_time - pd.Timedelta(days=window_days - 1) # 원본 방식으로 복원
    # 날짜 범위는 actual_end_time까지만 생성 (Prognosis 모드에서 블라인드 구간 제외)
    date_index = pd.date_range(start_time.date(), actual_end_time.date(), freq="D").date

    features: Dict[str, Dict] = {}

    # Lab/chart 슬라이스
    for group, prefix in [(cache.get("lab_daily", {}), "lab_"), (cache.get("chart_daily", {}), "chart_")]:
        for feat, daily_map in group.items():
            # 윈도우 날짜만 추출
            series_vals = {d: v for d, v in daily_map.items() if d in date_index}
            if series_vals:
                features[f"{prefix}{feat}"] = series_vals

    # Rx 개별 약물 토큰 생성 (간소화)
    rx_flags: Dict[str, Dict] = cache.get("rx_daily_flags", {})
    rx_events: Dict[str, List[Dict]] = cache.get("rx_events", {})
    
    for rx_name, daily_map in rx_flags.items():
        # 윈도우 플래그 시퀀스
        flags_seq = [1 if d in daily_map and daily_map[d] else 0 for d in date_index]
        if sum(flags_seq) == 0:
            continue
        bits = "".join(["1" if x else "0" for x in flags_seq])
        # 최근24h 대표 투여 (actual_end_time 기준)
        recent_token = "."
        data_source = ""
        events = [e for e in rx_events.get(rx_name, []) if (e["time"] >= actual_end_time - pd.Timedelta(hours=24)) and (e["time"] <= actual_end_time)]
        if events:
            last_e = _sort_by_time(events)[-1]
            dose = str(last_e.get("dose", "") or "").strip()
            unit = str(last_e.get("unit", "") or "").strip()
            route = str(last_e.get("route", "") or "").strip()
            recent_token = f"{dose}{unit}{route}" if dose else route or "."
            # 데이터 소스 표시 (EMAR = 실제 투여, Input Events = 계획된 투여, Prescription = 처방 추정)
            source = last_e.get("source", "prescription")
            if source == "emar":
                data_source = "; src=emar"
            elif source == "input_events":
                data_source = "; src=input"
            else:
                data_source = "; src=rx"
        # last dose h ago
        all_events = [e for e in rx_events.get(rx_name, []) if e["time"] <= end_time]
        if all_events:
            last_time = _sort_by_time(all_events)[-1]["time"]
            last_ago_h = int((end_time - last_time).total_seconds() // 3600)
        else:
            last_ago_h = 9999
        # 패턴 지표
        longest = 0
        cur = 0
        for b in flags_seq:
            if b:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0
        recent3 = sum(flags_seq[-3:]) if len(flags_seq) >= 3 else sum(flags_seq)
        tag = _rx_classify_pattern(bits, recent3, longest, lang)
        
        if rx_format == "compact":
            token = _format_rx_compact(flags_seq, len(flags_seq))
        else:
            if lang == "en":
                token = f"pat={bits}; last={last_ago_h}h; 24h={recent_token}; days={sum(flags_seq)}/{len(flags_seq)}; tag={tag}{data_source}"
            else:
                token = f"패턴={bits}; 최종={last_ago_h}시간전; 24시간={recent_token}; 일수={sum(flags_seq)}/{len(flags_seq)}; 유형={tag}{data_source}"
        
        features[f"rx_{rx_name}"] = token

    # 윈도우 스킵 규칙: 데이터가 전혀 없으면 빈 dict 반환 (호출부에서 생략)
    has_any = False
    for k, v in features.items():
        if isinstance(v, dict) and len(v) > 0:
            has_any = True
            break
    return features if has_any else {}


############################################################
# (9.5) QUALITY GATE: 윈도우 품질 필터(앵커 기준)
############################################################

def _record_quality_skip(reason: str, patient_id: int) -> None:
    """품질 게이팅으로 스킵된 샘플 통계를 전역 dict에 기록."""
    stats = globals().get("quality_gate_stats")
    if not isinstance(stats, dict):
        return
    stats["skipped_count"] = int(stats.get("skipped_count", 0)) + 1
    stats.setdefault("skipped_patients", set()).add(int(patient_id))
    reasons = stats.setdefault("reasons", {})
    reasons[reason] = int(reasons.get(reason, 0)) + 1


def _ensure_daily_cache(clinical_data: Dict[str, Any], patient_id: int) -> Dict[str, Any]:
    if patient_id not in PRECOMPUTED_DAILIES:
        PRECOMPUTED_DAILIES[patient_id] = precompute_patient_daily_maps(clinical_data, patient_id)
    return PRECOMPUTED_DAILIES[patient_id]


def _count_scr_in_window(
    clinical_data: Dict[str, Any],
    patient_id: int,
    end_time: pd.Timestamp,
    window_days: int,
) -> Tuple[int, int]:
    """(count, distinct_days) within [end_time-(window_days-1)d, end_time]."""
    lab_data = clinical_data.get("lab_events")
    lab_df = pd.DataFrame(lab_data) if isinstance(lab_data, dict) else lab_data
    if not isinstance(lab_df, pd.DataFrame) or lab_df.empty:
        return 0, 0

    # charttime 보정
    if "charttime" not in lab_df.columns:
        return 0, 0
    if not pd.api.types.is_datetime64_any_dtype(lab_df["charttime"]):
        lab_df = lab_df.copy()
        lab_df["charttime"] = pd.to_datetime(lab_df["charttime"], errors="coerce")

    scr_itemids = set((FEATURE_ITEMIDS or {}).get("creatinine", []))
    if not scr_itemids:
        return 0, 0

    start_time = pd.Timestamp(end_time) - pd.Timedelta(days=window_days - 1)
    rng = lab_df[
        (lab_df.get("subject_id") == patient_id)
        & (lab_df.get("itemid").isin(scr_itemids))
        & (pd.notna(lab_df.get("valuenum")))
        & (pd.notna(lab_df["charttime"]))
        & (lab_df["charttime"] >= start_time)
        & (lab_df["charttime"] <= end_time)
    ]
    if rng.empty:
        return 0, 0
    scr_count = int(rng["valuenum"].notna().sum())
    scr_days = int(pd.to_datetime(rng["charttime"]).dt.date.nunique())
    return scr_count, scr_days


def _count_observed_days_in_window(
    clinical_data: Dict[str, Any],
    patient_id: int,
    end_time: pd.Timestamp,
    window_days: int,
) -> int:
    """랩/차트/약물 플래그 중 어떤 형태로든 관측된 날짜 수."""
    cache = _ensure_daily_cache(clinical_data, patient_id)
    start_date = (pd.Timestamp(end_time) - pd.Timedelta(days=window_days - 1)).date()
    end_date = pd.Timestamp(end_time).date()
    date_set = set(pd.date_range(start_date, end_date, freq="D").date)

    observed: set = set()
    for group in (cache.get("lab_daily", {}), cache.get("chart_daily", {})):
        if isinstance(group, dict):
            for daily_map in group.values():
                if isinstance(daily_map, dict):
                    for d in daily_map.keys():
                        if d in date_set:
                            observed.add(d)
    rx_flags = cache.get("rx_daily_flags", {})
    if isinstance(rx_flags, dict):
        for daily_map in rx_flags.values():
            if isinstance(daily_map, dict):
                for d, flag in daily_map.items():
                    if d in date_set and bool(flag):
                        observed.add(d)
    return int(len(observed))


def passes_quality_gate(
    clinical_data: Dict[str, Any],
    patient_id: int,
    cutoff_time: pd.Timestamp,
    window_days: int,
    *,
    min_scr_count: int,
    min_scr_days: int,
    min_observed_days: int,
) -> bool:
    """앵커(cutoff_time) 기준 과거 윈도우 품질 검증.

    - Creatinine(SCr) 측정 개수/일수
    - 전체 관측일수(랩/차트/약물 중 어떤 형태로든 관측된 날짜)
    """
    # 0이면 해당 조건 비활성화
    min_scr_count = int(min_scr_count or 0)
    min_scr_days = int(min_scr_days or 0)
    min_observed_days = int(min_observed_days or 0)

    if min_scr_count <= 0 and min_scr_days <= 0 and min_observed_days <= 0:
        return True

    # 관측일수
    if min_observed_days > 0:
        obs_days = _count_observed_days_in_window(clinical_data, int(patient_id), pd.Timestamp(cutoff_time), int(window_days))
        if obs_days < min_observed_days:
            _record_quality_skip(f"min_observed_days<{min_observed_days}", int(patient_id))
            return False

    # SCr 개수/일수
    scr_count, scr_days = _count_scr_in_window(clinical_data, int(patient_id), pd.Timestamp(cutoff_time), int(window_days))
    if min_scr_count > 0 and scr_count < min_scr_count:
        _record_quality_skip(f"min_scr_count<{min_scr_count}", int(patient_id))
        return False
    if min_scr_days > 0 and scr_days < min_scr_days:
        _record_quality_skip(f"min_scr_days<{min_scr_days}", int(patient_id))
        return False

    return True


def format_trend_pattern(
    daily_values: Dict[date, Optional[float]],
    window_days: int,
    anchor_date: Any,
    lang: str = "ko",
) -> str:
    """
    시퀀스의 끝을 항상 'anchor_date'로 고정하여 창을 생성.
    - daily_values: 키가 date 로 통일되어 있다고 가정
    - anchor_date: date 또는 Timestamp/str (내부에서 date로 변환)
    - 14/28일 창: 결측 연속구간은 NAxN으로 압축(옵션), 구분자는 ','
    - 3/7일 창: RLE 미적용, 결측은 'NA'로 표기, 쉼표(',')
    """
    if not daily_values:
        return "No data" if lang == "en" else "데이터 없음"

    # anchor_date → date
    try:
        if isinstance(anchor_date, pd.Timestamp):
            anchor_date = anchor_date.date()
        elif not isinstance(anchor_date, date):
            anchor_date = pd.to_datetime(anchor_date).date()
    except Exception:
        return "No data" if lang == "en" else "데이터 없음"

    start_date = anchor_date - timedelta(days=window_days - 1)

    # 창 길이에 맞춰 날짜 순회
    seq: List[str] = []
    for i in range(window_days):
        d = start_date + timedelta(days=i)
        v = daily_values.get(d)
        if v is None or (isinstance(v, float) and not pd.notna(v)):
            # 짧은 창(≤7일)에서도 결측은 'NA'로 명시
            seq.append("NA")
        else:
            try:
                seq.append(f"{float(v):.1f}")
            except Exception:
                seq.append("NA")

    # 긴 창(>= TREND_RLE_APPLY_DAYS)에서만 결측 run-length 압축
    if window_days >= TREND_RLE_APPLY_DAYS and TREND_RLE_ENABLED:
        compact: List[str] = []
        i, n = 0, len(seq)
        while i < n:
            if seq[i] == "NA":
                j = i
                while j < n and seq[j] == "NA":
                    j += 1
                run_len = j - i
                if run_len >= TREND_RLE_MIN_RUN:
                    compact.append(f"NAx{run_len}")
                else:
                    compact.extend(["NA"] * run_len)
                i = j
            else:
                compact.append(seq[i])
                i += 1
        return ",".join(compact)

    return ",".join(seq)


############################################################
# (10) TRENDS: 트렌드 포맷(앵커고정/RLE)
############################################################

def format_recent_snapshot(snapshot: Dict[str, Dict[str, float]], lookback_hours: int, lang: str = "ko") -> str:
    """
    최근 스냅샷 데이터를 프롬프트 형식으로 포맷팅.
    snapshot 구조는 {"lab": {feat: value}, "chart": {feat: value}} 형태를 가정한다.
    """
    if not snapshot:
        return "No recent data" if lang == "en" else "최근 데이터 없음"

    lines: List[str] = []

    # Lab
    lab_map = snapshot.get("lab", {}) or {}
    if lab_map:
        lines.append("Lab Features:")
        for feat, val in lab_map.items():
            if val is not None and pd.notna(val):
                try:
                    lines.append(f"  - {feat}: {float(val):.1f}")
                except Exception:
                    lines.append(f"  - {feat}: {val}")

    # Chart
    chart_map = snapshot.get("chart", {}) or {}
    if chart_map:
        lines.append("Chart Features (ICU):")
        for feat, val in chart_map.items():
            if val is not None and pd.notna(val):
                try:
                    lines.append(f"  - {feat}: {float(val):.1f}")
                except Exception:
                    lines.append(f"  - {feat}: {val}")

    no_data_text = (
        f"No data in recent {lookback_hours} hours" if lang == "en" else f"최근 {lookback_hours}시간 데이터 없음"
    )
    return "\n".join(lines) if (lab_map or chart_map) else no_data_text

def format_medication_summary(rx_events: Dict[str, List[Dict]], target_time: pd.Timestamp, lookback_days: int = 7, lang: str = "ko") -> str:
    """
    최근 투약 정보 요약 (일상생활 기반 예측을 위해 시술 정보는 제외)
    """
    lines: List[str] = []
    cutoff_time = target_time - pd.Timedelta(days=lookback_days)
    
    # EMAR/처방 정보만 포함
    if rx_events:
        recent_meds = []
        for med_name, events in rx_events.items():
            recent_events = [e for e in events if pd.to_datetime(e["time"]) >= cutoff_time]
            if recent_events:
                source = recent_events[0].get("source", "prescription")
                if source == "emar":
                    source_tag = "emar"
                elif source == "input_events":
                    source_tag = "input"
                else:
                    source_tag = "rx"
                recent_meds.append(f"{med_name} ({len(recent_events)}회; src={source_tag})")
        
        if recent_meds:
            med_title = "Recent Medications:" if lang == "en" else "최근 투약:"
            lines.append(med_title)
            lines.extend([f"  - {med}" for med in recent_meds])
    
    return "\n".join(lines) if lines else ""


_TEMP_RECENT_DAYS = 7

def _format_temperature_compact(
    daily_map: Dict, window_days: int, anchor_date
) -> str:
    """체온을 compact 형식으로: avg + recent N일 트렌드.

    예: "avg 37.1, recent 7d: 37.3,36.9,37.0,36.7,36.5,36.8,36.9"
    """
    from datetime import timedelta
    dates = sorted(daily_map.keys())
    vals = [float(daily_map[d]) for d in dates if daily_map[d] is not None]
    if not vals:
        return "N/A"

    avg = sum(vals) / len(vals)

    # anchor_date 기준 최근 N일 값 추출
    recent_vals = []
    for offset in range(_TEMP_RECENT_DAYS, 0, -1):
        d = anchor_date - timedelta(days=offset - 1)
        v = daily_map.get(d)
        if v is not None:
            recent_vals.append(f"{float(v):.1f}")
        else:
            recent_vals.append("NA")

    recent_str = ",".join(recent_vals)
    return f"avg {avg:.1f}, recent {_TEMP_RECENT_DAYS}d: {recent_str}"


def summarize_windowed_features(windowed_features: Dict[str, Dict], lang: str = "ko", anchor_date=None) -> str:
    """
    주어진 윈도우(window_name)별 임상 피처 요약을 사람이 읽기 쉽게 문자열로 변환합니다.

    - 기존에는 각 줄마다 '3days - lab_xxx'처럼 윈도우명과 접두어가 반복적으로 붙었음
    - 개선된 버전은 윈도우명 반복을 제거하고, Lab / Vitals / Medications 섹션으로 묶어 표시
    - chart 피처는 최대 6개, lab 피처는 최대 10개까지만 출력 (너무 길어지는 것 방지)
    - eGFR 값은 target이므로 input 프롬프트에 포함하지 않음   
    - 출력은 Lab / Vitals / Medications 섹션으로 나뉘어 표시됩니다.
    - 변경점(중요):
    - 📌 일별 맵의 키 타입을 모두 datetime.date로 '정규화'하여
      추후 윈도우 슬라이싱(3/7/14/28일)에서 날짜 매칭 문제가 발생하지 않도록 함.    

    Args:
        windowed_features: {window_name: {feature_name: values}}
            - window_name: "current", "3days", "7days", "2weeks", "4weeks" 등
            - feature_name: "lab_creatinine", "chart_sbp", "rx_furosemide" 등
            - values: dict(date→값) 또는 단일 값 (float/str)
        lang: 출력 언어 ("ko" | "en")
        anchor_date: 시퀀스의 끝 날짜 (= target_time.date()).
                     이 값으로 시퀀스의 끝을 고정하여 3/7/14/28일 창 간 정렬 일관성을 보장.

    Returns:
        문자열: "📊 Lab: ..." "💓 Vitals: ..." "💊 Medications: ..." 형식의 다중 라인 요약
    """

    # 윈도우 피처가 없으면 바로 종료
    if not isinstance(windowed_features, dict) or not windowed_features:
        return "(No past window features)" if lang == "en" else "(과거 윈도우 피처 없음)"

    # 섹션 타이틀 / 레전드 문구
    if lang == "en":
        title_lab, title_chart, title_rx = "Lab", "Vitals", "Medications"
        no_data_text = "(No past window features)"
        legend_text = "(*) NAxN = N days unmeasured (not 0)."
    else:
        title_lab, title_chart, title_rx = "Lab", "Vitals", "투약"
        no_data_text = "(과거 윈도우 피처 없음)"
        legend_text = "(*) NAxN = N일 미측정(0 아님)."

    # 내부 helper: 접두어 제거 (lab_, chart_, rx_)
    def _disp_name(k: str) -> str:
        if k.startswith("lab_"): return k[4:]
        if k.startswith("chart_"): return k[6:]
        if k.startswith("rx_"): return k[3:]
        return k

    # 보통 한 레코드에 하나의 윈도우만 있음 → 첫 번째 항목만 처리
    window_name, features = next(iter(windowed_features.items()))
    window_days = TIME_WINDOWS.get(window_name, None)

    # anchor_date가 없으면 fallback으로 오늘 날짜 사용 (실제로는 호출부에서 넘겨줘야 함)
    if anchor_date is None:
        try:
            anchor_date = pd.Timestamp.today().date()
        except Exception:
            return no_data_text

    # 섹션별 라인 리스트
    lab_lines: List[str] = []
    chart_lines: List[str] = []
    rx_lines: List[str] = []

    # RLE 레전드 출력 여부 추적
    rle_used_lab = False
    rle_used_chart = False

    # 최대 표시 개수 제한 (너무 길어지는 것 방지)
    MAX_CHART = 8
    MAX_LAB = 10
    shown_chart = 0
    shown_lab = 0
    rx_is_compact = False

    # === 피처 반복 처리 ===
    for feature_name, value in features.items():
        # egfr은 target label → input에서 제외
        if feature_name == "lab_egfr":
            continue
        # rx_summary는 따로 처리되므로 제외
        if feature_name == "rx_summary":
            continue

        # (1) 시계열 데이터(dict)라면 → format_trend_pattern 호출
        if isinstance(value, dict) and window_days:
            if feature_name == "chart_temperature":
                trend = _format_temperature_compact(value, window_days, anchor_date)
            else:
                trend = format_trend_pattern(value, window_days, anchor_date, lang)
        # (2) 단일 값이라면 → 숫자/문자 그대로 표시
        else:
            if isinstance(value, Number):
                try:
                    trend = f"{float(value):.1f}"
                except Exception:
                    trend = str(value)
            else:
                trend = str(value)

        # 표시명 (접두어 제거)
        name = _disp_name(feature_name)

        # 분류: chart / lab / rx
        if feature_name.startswith("chart_"):
            if shown_chart < MAX_CHART:
                chart_lines.append(f"  - {name}: {trend}")
                if window_days and window_days >= TREND_RLE_APPLY_DAYS and "NAx" in trend:
                    rle_used_chart = True
                shown_chart += 1

        elif feature_name.startswith("lab_"):
            if shown_lab < MAX_LAB:
                lab_lines.append(f"  - {name}: {trend}")
                if window_days and window_days >= TREND_RLE_APPLY_DAYS and "NAx" in trend:
                    rle_used_lab = True
                shown_lab += 1

        elif feature_name.startswith("rx_"):
            rx_lines.append(f"  - {name}: {trend}")
            if isinstance(value, str) and "pat=" not in value and "/" in value and "d)" in value:
                rx_is_compact = True

        else:
            # 접두어 없는 경우 → lab 섹션으로 fallback
            if shown_lab < MAX_LAB:
                lab_lines.append(f"  - {name}: {trend}")
                shown_lab += 1

    # === 출력 조립 ===
    out_lines: List[str] = []

    if lab_lines:
        out_lines.append(f"{title_lab}:")
        if TREND_RLE_ENABLED and window_days and window_days >= TREND_RLE_APPLY_DAYS and rle_used_lab:
            out_lines.append(legend_text)
        out_lines.extend(lab_lines)

    if chart_lines:
        out_lines.append(f"{title_chart}:")
        if TREND_RLE_ENABLED and window_days and window_days >= TREND_RLE_APPLY_DAYS and rle_used_chart:
            out_lines.append(legend_text)
        out_lines.extend(chart_lines)

    if rx_lines:
        out_lines.append(f"{title_rx}:")
        out_lines.extend(rx_lines)
    elif lab_lines or chart_lines:
        # 과거 윈도우에 추적 약물이 없을 때 명시 (누락이 아님을 표시)
        out_lines.append(f"{title_rx}:")
        out_lines.append("  (none in past window)" if lang == "en" else "  (과거 윈도우 내 없음)")

    return "\n".join(out_lines) if out_lines else no_data_text


############################################################
# (11) BUILDERS: eGFR/AKI 프롬프트 생성
############################################################

def build_prompt_for_egfr(label: Dict[str, Any], output_format: str = "text", output_lang: str = "ko") -> Dict[str, Any]:
    """eGFR 회귀 태스크용 프롬프트를 구성한다(동작 동일).

    Args:
        label: 윈도우별 피처와 타겟 시간을 포함한 레코드
        output_format: "text" 또는 "json"
        output_lang: "ko" 또는 "en"

    Returns:
        Dict[str, Any]: instruction/input/output/metadata 구조
    """
    patient_id = label["patient_id"]
    target_time = label["target_time"]
    target_egfr = float(label["target_egfr"]) if pd.notna(label["target_egfr"]) else None
    # eGFR는 현재 데이터셋에서 정수 단위로만 존재(소수 의미 없음) → 정수로 통일
    target_egfr_int: Optional[int] = int(round(target_egfr)) if target_egfr is not None else None
    windowed_features = label["windowed_features"]

    # Prognosis 모드 확인: cutoff_time을 anchor_date로 사용하여 날짜 범위 제한
    pred_days = label.get("pred_days", 0)
    cutoff_time = label.get("cutoff_time", target_time)

    # 출력에만 정답 포함하도록 input에서는 현재시점 실제값 노출하지 않음
    # anchor_date를 cutoff_time으로 설정하여 블라인드 구간 이후 데이터가 표시되지 않도록 함
    anchor_date_for_summary = cutoff_time if pred_days > 0 else target_time
    trend_summary = summarize_windowed_features(windowed_features, output_lang, anchor_date_for_summary)
    # 과거 윈도우 요약이 비어 있는 경우 최근 스냅샷 보강 (egfr 누설 방지: egfr 제외)
    snapshot_text = ""
    no_features_text = "(No past window features)" if output_lang == "en" else "(과거 윈도우 피처 없음)"
    if not trend_summary or trend_summary.strip() == no_features_text:
        # Prognosis 모드에서는 cutoff_time까지만 스냅샷 생성
        snapshot_time = cutoff_time if pred_days > 0 else target_time
        snapshot, used_h = build_recent_snapshot(
            clinical_data=label.get("clinical_data_for_snapshot", {}),
            patient_id=int(patient_id),
            target_time=snapshot_time,
            lookback_hours=FALLBACK_LOOKBACK_HOURS,
            hours_candidates=[24, 48, 72],
            exclude_lab_features=["egfr"],
            allow_equal_end=True,
        )
        extra = f" (used {used_h}h)" if used_h else ""
        snapshot_text = format_recent_snapshot(snapshot, FALLBACK_LOOKBACK_HOURS, output_lang) + extra
        
        # EMAR 투약 정보 추가 (일상생활 기반 예측에 적합)
        # Prognosis 모드에서는 cutoff_time까지만 약물 정보 표시
        dailies = label.get("dailies", {})
        rx_events = dailies.get("rx_events", {})
        pred_days_temp = label.get("pred_days", 0)
        cutoff_time_temp = label.get("cutoff_time", target_time)
        medication_time = cutoff_time_temp if pred_days_temp > 0 else target_time
        medication_summary = format_medication_summary(rx_events, medication_time, 7, output_lang)
        if medication_summary:
            snapshot_text += "\n" + medication_summary

    T = get_texts(output_lang)
    
    # Prognosis 모드 확인: pred_days가 0보다 크면 과거로 돌아가서 예측 모드
    pred_days = label.get("pred_days", 0)
    cutoff_time = label.get("cutoff_time", target_time)
    
    if output_format == "json":
        if pred_days > 0:
            instruction = T["instr_egfr_prognosis_json"].format(horizon_days=pred_days)
        else:
            instruction = T["instr_egfr_json"].format()
    else:
        if pred_days > 0:
            # Prognosis 모드: 과거 시점에서 미래 예측 Instruction 사용
            instruction = T["instr_egfr_prognosis"].format(horizon_days=pred_days)
        else:
            # 기존 모드: 현재 시점 예측
            instruction = T["instr_egfr"]
    
    # 윈도우 분리 생성: label에 윈도우 이름/일수 포함됨
    window_name = label.get("window_name")
    window_days = label.get("window_days")
    # 환자 배경 정보 (v11: 동반질환/CCI/CKD Stage/입원맥락/체중 포함)
    patient_bg = label.get("patient_background", {})
    demo_str = format_patient_background_for_prompt(patient_bg, lang=output_lang)

    # current 창 사용 시간(표시용) - Prognosis 모드에서는 cutoff_time 사용
    used_h_display: Optional[int] = None
    if window_name == "current":
        snapshot_time_for_display = cutoff_time if pred_days > 0 else target_time
        _, used_h_display = build_recent_snapshot(
            clinical_data=label.get("clinical_data_for_snapshot", {}),
            patient_id=int(patient_id),
            target_time=snapshot_time_for_display,
            lookback_hours=FALLBACK_LOOKBACK_HOURS,
            hours_candidates=[24, 48, 72],
            exclude_lab_features=["egfr"],
            allow_equal_end=True,
        )

    # 윈도우 표시: current는 실사용 시간(24h/48h/72h)로 표기
    window_display = (
        f"{used_h_display}h" if (window_name == "current" and used_h_display) else format_window_display(window_name, window_days, output_lang)
    )
    current_text = "Current" if output_lang == "en" else "현재"
    hist_clean = (trend_summary if trend_summary else current_text)
    snap_clean = (snapshot_text) if snapshot_text else ""
    
    # History Trend 텍스트 추가 (있는 경우)
    history_trend = windowed_features.get("history_trend", "")
    if history_trend:
        hist_clean = history_trend + "\n\n" + hist_clean
    
    # Prognosis 모드: cutoff_time을 표시하고 target_time을 예측 목표로 명시
    if pred_days > 0:
        cutoff_time_display = format_prompt_time(cutoff_time)
        target_time_display = format_prompt_time(target_time)
        horizon_text = f"{pred_days} days" if output_lang == "en" else f"{pred_days}일"
        # 미래 예측 모드용 Window Note
        window_note_prognosis = (
            "Past window ending at current timepoint. Lab/Vitals: daily averages, one value per day (left=oldest, right=recent)."
            if output_lang == "en" else
            "현재 시점까지 과거 윈도우 요약. Lab·바이탈: 일별 평균, 값 1개=1일 (왼쪽=과거, 오른쪽=최근)."
        )
        # 한글/영어 라벨 설정
        current_timepoint_label = "Current timepoint" if output_lang == "en" else "현재 시점"
        prediction_target_label = "Prediction target" if output_lang == "en" else "예측 타겟"
        future_text = f"({horizon_text} in the future)" if output_lang == "en" else f"({horizon_text} 후)"
        
        input_text = (
            f"{T['patient']}: {patient_id}\n"
            f"{current_timepoint_label}: {cutoff_time_display}\n"
            f"{prediction_target_label}: {target_time_display} {future_text}\n"
            f"{T['window']}: {window_display}\n"
            f"{window_note_prognosis}\n"
            f"{demo_str}\n"
            f"{T['history']}:\n{hist_clean}" + ("\n" + snap_clean if snap_clean else "")
        )
    else:
        # 기존 모드 (현재 시점)
        input_text = (
            f"{T['patient']}: {patient_id}\n"
            f"{T['index_time']}: {format_prompt_time(target_time)}\n"
            f"{T['window']}: {window_display}\n"
            f"{T['window_note']}\n"
            f"{demo_str}\n"
            f"{T['history']}:\n{hist_clean}" + ("\n" + snap_clean if snap_clean else "")
        )
    if output_format == "json":
        output_text = json.dumps({
            "eGFR": int(target_egfr_int) if target_egfr_int is not None else None
        }, ensure_ascii=False)
    else:  # text
        # 직접적인 수치만 출력
        output_text = f"{target_egfr_int:d}" if target_egfr_int is not None else "N/A"

    meta: Dict[str, Any] = {
        "task_type": "egfr_prediction",
        "label_type": label.get("label_type"),
        "patient_id": int(patient_id),
        "target_time": format_prompt_time(target_time),
        "target_egfr": target_egfr_int,
        "window_name": window_name,
        "window_days": window_days,
        "window_hours": (int(used_h_display) if (window_name == "current" and used_h_display) else None),
        "demographics": patient_bg.get("demographics", {}),
        "cci_score": patient_bg.get("cci_score", 0),
        "ckd_stage": patient_bg.get("ckd_stage"),
    }
    # v7 산출물에서 빠져있던 prognosis 메타데이터 보강(분석/필터링 용이)
    if pred_days and int(pred_days) > 0:
        meta["cutoff_time"] = format_prompt_time(cutoff_time)
        meta["prediction_horizon"] = int(pred_days)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": meta,
    }


def extract_history_features(clinical_data: Dict[str, Any], patient_id: int, cutoff_time: pd.Timestamp, output_lang: str = "ko") -> str:
    """
    과거 eGFR/Creatinine 수치 이력을 시간순으로 추출하여 텍스트로 반환.
    Prognosis(미래 예측) 태스크에서 추세(Trend) 정보를 제공하기 위함.
    """
    history_lines = []
    
    # Lab 데이터에서 Creatinine/eGFR 추출 (v5와 동일하게 lab_events 사용)
    lab_data = clinical_data.get("lab_events", pd.DataFrame())
    lab_df = pd.DataFrame(lab_data) if isinstance(lab_data, dict) else lab_data
    
    if isinstance(lab_df, pd.DataFrame) and lab_df.empty:
        return ""
    
    if not isinstance(lab_df, pd.DataFrame):
        return ""

    # 관련 ItemID 필터링 (eGFR만 - 사용자 요청사항)
    # FEATURE_ITEMIDS는 config에서 가져옴
    target_itemids = []
    if FEATURE_ITEMIDS:
        target_itemids.extend(FEATURE_ITEMIDS.get("egfr", []))  # eGFR만 표시
    
    # 해당 환자의 cutoff_time 이전 데이터만 조회
    # charttime 컬럼이 datetime이 아닐 수 있으므로 변환
    if not pd.api.types.is_datetime64_any_dtype(lab_df.get("charttime")):
        lab_df = lab_df.copy()
        lab_df["charttime"] = pd.to_datetime(lab_df["charttime"], errors="coerce")
    
    # cutoff_time 이전만 추출 (cutoff_time 시점의 데이터는 제외하여 Data Leakage 방지)
    # 현재 시점(pred_days=0)에서는 cutoff_time = target_time이므로, target_time 시점의 정답값이 포함되지 않도록 < 사용
    patient_lab = lab_df[
        (lab_df["subject_id"] == patient_id) & 
        (lab_df["itemid"].isin(target_itemids)) &
        (pd.notna(lab_df.get("valuenum"))) &
        (lab_df["charttime"] < cutoff_time)  # <= 대신 < 사용하여 cutoff_time 시점 제외
    ].sort_values("charttime")

    if patient_lab.empty:
        return ""

    # 날짜별로 그룹화하여 출력 (최신순이 아니라 과거->현재 순으로)
    # 너무 많으면 최근 10개만? -> 아니요, 추세가 중요하므로 다 보여주되 길이 제한 고려
    # 여기서는 최근 N개로 제한(상한). 실제 출력 개수는 환자/시점마다 다를 수 있음.
    # 표기에는 날짜/시간(연도 비식별화 이슈 포함)을 노출하지 않고, 앵커(cutoff_time) 기준 상대일수(D-#)로만 표기한다.
    # 같은 날짜에 여러 번 측정된 경우가 있으므로, 일자별로 1개(그날의 가장 최근 측정값)로 압축한다.
    max_points = 20
    patient_lab = patient_lab.tail(max_points)

    cutoff_date = pd.Timestamp(cutoff_time).date()
    # key: days_ago(int), value: egfr_int (해당 day의 마지막 측정값)
    per_day_latest: Dict[int, int] = {}

    for _, row in patient_lab.iterrows():
        ts = row["charttime"]
        val = row["valuenum"]
        if pd.isna(ts) or pd.isna(val):
            continue
        try:
            ts_date = pd.Timestamp(ts).date()
            days_ago = int((cutoff_date - ts_date).days)
            v_int = int(round(float(val)))
        except Exception:
            continue
        if days_ago < 0:
            # cutoff_time 이전만 남겨뒀으므로 이 케이스는 원칙적으로 없어야 함
            continue
        # charttime 오름차순이므로 같은 days_ago에서는 뒤에 올수록 더 최신값 → overwrite 하면 "해당일 최신"이 됨
        per_day_latest[days_ago] = v_int

    # 과거(큰 D) → 현재(작은 D) 순으로 출력
    for days_ago in sorted(per_day_latest.keys(), reverse=True):
        history_lines.append(f"- D-{days_ago}: {per_day_latest[days_ago]}")

    if not history_lines:
        return ""
        
    if output_lang == "en":
        header = "Patient History (eGFR Trend; D-# days before current timepoint, oldest→newest):"
    else:
        # 한국어 프롬프트에 영문 헤더가 섞이지 않도록 분리
        header = "환자 과거 eGFR 추세(D-#=현재 시점 기준 며칠 전, 과거→현재):"
    return header + "\n" + "\n".join(history_lines)


def _deterministic_keep(p_drop: float, patient_id: int, cutoff_time: pd.Timestamp, pred_days: int) -> bool:
    """확률적 옵션을 재현 가능하게 적용하기 위한 결정적 keep 함수.

    - p_drop: 제거 확률(0~1). keep 확률은 (1-p_drop)
    - seed: (patient_id, cutoff_time, pred_days) 조합으로 고정
    """
    try:
        p_drop = float(p_drop)
    except Exception:
        p_drop = 0.0
    p_drop = 0.0 if p_drop < 0 else (1.0 if p_drop > 1 else p_drop)
    keep_p = 1.0 - p_drop
    key = f"{int(patient_id)}|{pd.Timestamp(cutoff_time).isoformat()}|{int(pred_days)}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    # 0 ~ 0.999999 범위
    u = (int(h[:12], 16) % 1_000_000) / 1_000_000.0
    return u < keep_p


def build_prompt_for_aki(label: Dict[str, Any], output_format: str = "text", output_lang: str = "ko") -> Dict[str, Any]:
    """AKI 분류 태스크용 프롬프트를 구성한다(동작 동일).

    Args:
        label: 윈도우별 피처와 타겟 시간을 포함한 레코드
        output_format: "text" 또는 "json"
        output_lang: "ko" 또는 "en"

    Returns:
        Dict[str, Any]: instruction/input/output/metadata 구조
    """
    patient_id = label["patient_id"]
    target_time = label["target_time"]
    target_aki = bool(label["target_aki"])  # True/False
    windowed_features = label["windowed_features"]

    # Prognosis 모드 확인: cutoff_time을 anchor_date로 사용하여 날짜 범위 제한
    pred_days = label.get("pred_days", 0)
    cutoff_time = label.get("cutoff_time", target_time)

    # input에서는 정답(현재시점 실제 AKI) 노출하지 않음
    # anchor_date를 cutoff_time으로 설정하여 블라인드 구간 이후 데이터가 표시되지 않도록 함
    anchor_date_for_summary = cutoff_time if pred_days > 0 else target_time
    trend_summary = summarize_windowed_features(windowed_features, output_lang, anchor_date_for_summary)

    T = get_texts(output_lang)
    
    # Prognosis 모드 확인: pred_days가 0보다 크면 과거로 돌아가서 예측 모드
    if output_format == "json":
        if pred_days > 0:
            instruction = T["instr_aki_prognosis_json"].format(horizon_days=pred_days)
        else:
            instruction = T["instr_aki_json"].format()
    else:
        if pred_days > 0:
            # Prognosis 모드: 과거 시점에서 미래 예측 Instruction 사용
            instruction = T["instr_aki_prognosis"].format(horizon_days=pred_days)
        else:
            # 기존 모드: 현재 시점 예측
            instruction = T["instr_aki"]
    
    window_name = label.get("window_name")
    window_days = label.get("window_days")
    patient_bg = label.get("patient_background", {})
    demo_str = format_patient_background_for_prompt(patient_bg, lang=output_lang)

    # 과거 요약이 비면 최근 스냅샷 보강
    snapshot_text = ""
    no_features_text = "(No past window features)" if output_lang == "en" else "(과거 윈도우 피처 없음)"
    if not trend_summary or trend_summary.strip() == no_features_text:
        # Prognosis 모드에서는 cutoff_time까지만 스냅샷 생성 (미래 정보 누출 방지)
        snapshot_time = cutoff_time if pred_days > 0 else target_time
        snapshot, used_h = build_recent_snapshot(
            clinical_data=label.get("clinical_data_for_snapshot", {}),
            patient_id=int(patient_id),
            target_time=snapshot_time,
            lookback_hours=FALLBACK_LOOKBACK_HOURS,
            hours_candidates=[24, 48, 72],
            exclude_lab_features=[],
            allow_equal_end=True,
        )
        extra = f" (used {used_h}h)" if used_h else ""
        snapshot_text = format_recent_snapshot(snapshot, FALLBACK_LOOKBACK_HOURS, output_lang) + extra
        
        # EMAR 투약 정보 추가 (일상생활 기반 예측에 적합)
        # Prognosis 모드에서는 cutoff_time까지만 약물 정보 표시
        dailies = label.get("dailies", {})
        rx_events = dailies.get("rx_events", {})
        medication_time = cutoff_time if pred_days > 0 else target_time
        medication_summary = format_medication_summary(rx_events, medication_time, 7, output_lang)
        if medication_summary:
            snapshot_text += "\n" + medication_summary

    # 최근 원시 SCr 값들(표시 전용): Prognosis 모드에서는 cutoff_time 사용
    lookback_hours_for_display = 48  # v7에서는 28일 윈도우만 사용하므로 항상 48h
    scr_time = cutoff_time if pred_days > 0 else target_time
    recent_scr_vals: List[float] = get_recent_raw_lab_values(
        clinical_data=label.get("clinical_data_for_snapshot", {}),
        patient_id=int(patient_id),
        feature_name="creatinine",
        target_time=scr_time,
        lookback_hours=lookback_hours_for_display,
        allow_equal_end=True,
    )
    tail_label = f"recent{lookback_hours_for_display}h_scr"
    scr_tail = f"\n- {tail_label}: [{', '.join(f'{v:.2f}' for v in recent_scr_vals)}]" if recent_scr_vals else ""

    # v7에서는 28일 윈도우만 사용하므로 used_h_display는 None
    used_h_display: Optional[int] = None

    # 윈도우 표시: v7에서는 28일 윈도우만 사용
    window_display = format_window_display(window_name, window_days, output_lang)
    
    # Prognosis 모드: Input 텍스트에 현재 시점과 예측 타겟 명시
    if pred_days > 0:
        current_timepoint_label = "Current timepoint" if output_lang == "en" else "현재 시점"
        prediction_target_label = "Prediction target" if output_lang == "en" else "예측 타겟"
        window_note_prognosis = (
            "Past window ending at current timepoint. Lab/Vitals: daily averages, one value per day (left=oldest, right=recent)."
            if output_lang == "en" else
            "현재 시점에서 종료되는 과거 윈도우. Lab·바이탈: 일별 평균, 값 1개=1일 (왼쪽=과거, 오른쪽=최근)."
        )

        # 언어별 "예측 타겟" 라인 (조건식 우선순위 버그 방지)
        if output_lang == "en":
            day_word = "day" if pred_days == 1 else "days"
            prediction_target_line = (
                f"{prediction_target_label}: {format_prompt_time(target_time)} "
                f"({pred_days} {day_word} in the future)\n"
            )
        else:
            prediction_target_line = f"{prediction_target_label}: {format_prompt_time(target_time)} ({pred_days}일 후)\n"
        
        input_text = (
            f"{T['patient']}: {patient_id}\n"
            f"{current_timepoint_label}: {format_prompt_time(cutoff_time)}\n"
            f"{prediction_target_line}"
            f"{T['window']}: {window_display}\n"
            f"{window_note_prognosis}\n"
            f"{demo_str}\n"
            f"{T['history']}:\n{trend_summary if trend_summary else '(No past window features)' if output_lang == 'en' else '(과거 윈도우 피처 없음)'}{scr_tail}"
        )
    else:
        # 기존 모드 (현재 시점 예측)
        current_text = "Current" if output_lang == "en" else "현재"
        hist_clean = (trend_summary if trend_summary else current_text)
        tail_clean = (scr_tail)
        input_text = (
            f"{T['patient']}: {patient_id}\n"
        f"{T['index_time']}: {format_prompt_time(target_time)}\n"
        f"{T['window']}: {window_display}\n"
        f"{T['window_note']}\n"
        f"{demo_str}\n"
        f"{T['history']}:\n{hist_clean}{tail_clean}"
    )
    if output_format == "json":
        # 간단한 라벨만 출력 (향후 CoT 확장: reasoning 필드 추가 예정)
        output_text = json.dumps({
            "aki": 1 if target_aki else 0
        }, ensure_ascii=False)
    else:  # text
        # 직접적인 라벨만 출력
        output_text = "1" if target_aki else "0"

    meta = {
        "task_type": "aki_detection",
        "label_type": label.get("label_type"),
        "patient_id": int(patient_id),
        "target_time": format_prompt_time(target_time),
        "cutoff_time": format_prompt_time(cutoff_time) if pred_days > 0 else None,
        "prediction_horizon": pred_days if pred_days > 0 else None,
        "target_aki": target_aki,
        "window_name": window_name,
        "window_days": window_days,
        "window_hours": None,
        "demographics": patient_bg.get("demographics", {}),
        "cci_score": patient_bg.get("cci_score", 0),
        "ckd_stage": patient_bg.get("ckd_stage"),
    }
    # 원천 라벨 정보(label_source, kdigo_stage 등) 병합하여 통계/분석에 활용 가능하게 노출
    extra_meta = label.get("metadata", {})
    if isinstance(extra_meta, dict):
        meta.update(extra_meta)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": meta,
    }

############################################################
# (12) MAIN: CLI/파이프라인
############################################################

def main() -> None:
    """엔트리 포인트: 데이터 로드 → 라벨 생성 → 프롬프트 저장(동작 동일)."""
    args = parse_args()
    # should_exclude_timepoint_for_procedures 에서 접근할 수 있도록 플래그만 글로벌로 노출
    globals()["_procedure_extend_after_flag"] = bool(getattr(args, "procedure_extend_after_by_window", False))
    
    # ===============================
    # 실행 설정 출력
    # ===============================
    # Windows 콘솔 인코딩 문제로 인해 이모지 제거
    print("[START] 프롬프트 데이터셋 생성 시작")
    print("=" * 60)
    print("[CONFIG] 현재 설정된 조건들:")
    print(f"   [PATH] 데이터 경로: {args.processed_data_path}")
    print(f"   [FILE] 데이터 파일: {args.data_file}")
    print(f"   [PATIENTS] 환자 선별: {args.patient_selection}")
    print(f"   [TARGET] 최대 환자 수: {args.max_patients}명 {'(전체)' if args.max_patients == 0 else ''}")
    print(f"   [LABELS] 환자당 최대 라벨: {args.max_labels_per_patient}개 {'(제한없음)' if args.max_labels_per_patient == -1 else ''}")
    print(f"   [TASKS] 생성 태스크: {args.tasks}")
    print(f"   [RX] medication_format: {getattr(args, 'medication_format', 'compact')}")
    print(f"   [BG] patient background: comorbidities + CCI + CKD Stage + admission + weight")
    print(f"   [LANG] 출력 언어: {args.output_lang} {'(한국어)' if args.output_lang == 'ko' else '(영어)'}")
    print(f"   [FORMAT] 출력 형식: {args.output_format} ({args.file_format})")
    print(f"   [AKI_SOURCE] AKI 라벨 소스: {args.aki_label_source}")
    print(f"   [RATIO] AKI 음성:양성 비율: {args.aki_neg_pos_ratio}:1")
    print(f"   [WORKERS] 병렬 처리: {args.workers}개 워커")
    print()
    print("[FEATURES] 사용할 핵심 Features:")
    
    # 활성화된 lab features 출력
    active_lab_features = []
    for feature, itemids in FEATURE_ITEMIDS.items():
        active_lab_features.append(f"{feature}({','.join(map(str, itemids))})")
    print(f"   [LAB] Lab Features ({len(FEATURE_ITEMIDS)}개): {', '.join(active_lab_features)}")
    
    # Chart features 출력
    print(f"   [VITAL] Vital Features ({len(CHART_FEATURES)}개): {', '.join(CHART_FEATURES)}")
    
    # Prescription features 출력  
    prescription_categories = list(RX_KEYWORDS.keys())
    print(f"   [RX] 처방 Features ({len(prescription_categories)}개): {', '.join(prescription_categories)}")

    
    print("=" * 60)
    print()

    processed_path = Path(args.processed_data_path)
    data_path = processed_path / args.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    complete_data: Dict[str, Any] = load_pickle(data_path)
    clinical_data = complete_data["clinical_data"]

    # 동반질환 추출을 위해 MIMIC-IV 원본 diagnoses_icd 로드
    # (pkl에는 신장관련 ICD 코드만 포함되어 있어 DM/HTN 등이 누락됨)
    _mimic_diag_path = Path("data/mimic-iv-3.1/hosp/diagnoses_icd.csv")
    _diag_by_patient: Dict[int, pd.DataFrame] = {}
    if _mimic_diag_path.exists():
        _patient_ids = complete_data.get("patient_ids", set())
        _full_diag = pd.read_csv(_mimic_diag_path, dtype={"icd_code": str})
        _full_diag = _full_diag[_full_diag["subject_id"].isin(_patient_ids)]
        _diag_by_patient = {pid: grp for pid, grp in _full_diag.groupby("subject_id")}
        complete_data["base_info"]["diagnoses_icd"] = _full_diag
        complete_data["base_info"]["diagnoses"] = _full_diag
        complete_data["_diag_by_patient"] = _diag_by_patient
        print(f"[DIAG] 원본 diagnoses_icd 로드: {len(_full_diag)}건 ({len(_diag_by_patient)}명 indexed)")
        del _full_diag
    else:
        print(f"[WARN] {_mimic_diag_path} 없음 — pkl 내 diagnoses 사용 (동반질환 누락 가능)")

    # 📊 Procedure 제외 통계 준비
    procedure_exclusion_stats = {
        "total_candidates": 0,
        "excluded_count": 0,
        "exclusion_reasons": {},
        "excluded_patients": set()
    }
    # 🧹 품질 게이팅 통계(앵커 기준 윈도우 품질 미달로 스킵)
    quality_gate_stats = {
        "skipped_count": 0,
        "reasons": {},
        "skipped_patients": set(),
    }
    # 라벨 생성 함수들이 globals()로 통계를 접근하므로, 전역에 바인딩
    globals()["procedure_exclusion_stats"] = procedure_exclusion_stats
    globals()["quality_gate_stats"] = quality_gate_stats

    # 시간 컬럼을 한 번만 datetime으로 변환하여 반복 변환 비용을 줄임
    # Series 딕셔너리에서 DataFrame으로 변환 필요
    lab_data = clinical_data.get("lab_events")
    if isinstance(lab_data, dict):
        # 딕셔너리 형태(Series들)에서 DataFrame으로 변환
        lab_df = pd.DataFrame(lab_data)
    else:
        lab_df = lab_data
    
    if isinstance(lab_df, pd.DataFrame) and not lab_df.empty:
        if not pd.api.types.is_datetime64_any_dtype(lab_df["charttime"]):
            lab_df["charttime"] = pd.to_datetime(lab_df["charttime"])
        clinical_data["lab_events"] = lab_df

    chart_data = clinical_data.get("chart_events")
    if isinstance(chart_data, dict):
        # 딕셔너리 형태(Series들)에서 DataFrame으로 변환
        chart_df = pd.DataFrame(chart_data)
    else:
        chart_df = chart_data
        
    if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
        time_col = "event_time" if "event_time" in chart_df.columns else ("charttime" if "charttime" in chart_df.columns else None)
        if time_col and not pd.api.types.is_datetime64_any_dtype(chart_df[time_col]):
            chart_df[time_col] = pd.to_datetime(chart_df[time_col])
        clinical_data["chart_events"] = chart_df

    # 처방 시간 컬럼을 한 번만 datetime으로 변환 (혼합 포맷 대비)
    rx_data = clinical_data.get("prescriptions")
    if isinstance(rx_data, dict):
        # 딕셔너리 형태(Series들)에서 DataFrame으로 변환
        rx_df = pd.DataFrame(rx_data)
    else:
        rx_df = rx_data
        
    if isinstance(rx_df, pd.DataFrame) and not rx_df.empty:
        for col in ["starttime", "stoptime", "event_time"]:
            if col in rx_df.columns and not pd.api.types.is_datetime64_any_dtype(rx_df[col]):
                try:
                    rx_df[col] = pd.to_datetime(rx_df[col], errors="coerce", format="mixed")
                except TypeError:
                    # pandas <2.0: format="mixed" 미지원 → 일반 파서 사용
                    rx_df[col] = pd.to_datetime(rx_df[col], errors="coerce")
        clinical_data["prescriptions"] = rx_df

    # 환자 선정
    test_patients = get_test_patients(complete_data, processed_path, args.patient_selection, args.max_patients)
    # numpy 정수 등을 Python int로 통일
    test_patients = [int(pid) for pid in test_patients]
    print(f"[GOAL] **4단계 목표**: recommended_patients 우선순위 순서로 프롬프트 생성")
    print(f"[PATIENTS] 대상 환자: {len(test_patients)}명 (상위 우선순위)")
    print(f"   [1ST] 처리 1순위: ID {test_patients[0]}")
    print(f"   [2ND] 처리 2순위: ID {test_patients[1] if len(test_patients) > 1 else 'N/A'}")
    print(f"   [3RD] 처리 3순위: ID {test_patients[2] if len(test_patients) > 2 else 'N/A'}")
    if len(test_patients) > 10:
        print(f"   ... 총 {len(test_patients)}명 순서대로 처리")

    # 태스크 설정
    task_flags = set([t.strip().lower() for t in args.tasks.split(",") if t.strip()])
    do_egfr = "egfr" in task_flags
    do_aki = "aki" in task_flags

    generated: List[Dict[str, Any]] = []

    def _prefilter_clinical_for_patient(full_clinical: Dict[str, Any], pid: int) -> Dict[str, Any]:
        """환자별 clinical_data 사전 필터링 (28M rows → 수천 rows)."""
        filtered = {}
        for key, df in full_clinical.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                filtered[key] = df
                continue
            if "subject_id" in df.columns:
                filtered[key] = df[df["subject_id"] == pid]
            else:
                filtered[key] = df
        return filtered

    def create_current_target_egfr_labels(patient_id: int, max_labels: int) -> Tuple[List[Dict[str, Any]], int]:
        # --- 성능 최적화: 환자별 DataFrame 사전 필터링 ---
        _pt_clinical = _prefilter_clinical_for_patient(clinical_data, patient_id)

        timepoints, excluded_count = get_valid_measurement_timepoints(patient_id, _pt_clinical, "egfr", max_labels)
        if len(timepoints) == 0:
            return [], excluded_count
        labels: List[Dict[str, Any]] = []
        
        # 예측 시점 목록: --prediction-windows 사용 (AKI와 동일하게 통일)
        prediction_days_str = getattr(args, "prediction_windows", "0,1,2,3,4,5,6,7")
        prediction_days = [int(x.strip()) for x in prediction_days_str.split(",") if x.strip().isdigit()]

        # base_info도 환자별 필터링 (comorbidity/CKD stage 추출 최적화)
        _base = complete_data.get("base_info", {})
        _diag_lookup = complete_data.get("_diag_by_patient", {})
        _adm_full = _base.get("admissions")
        _pt_diag = _diag_lookup.get(patient_id, pd.DataFrame())
        _pt_adm = _adm_full[_adm_full["subject_id"] == patient_id] if isinstance(_adm_full, pd.DataFrame) else pd.DataFrame()
        _local_base = dict(_base)
        _local_base["diagnoses_icd"] = _pt_diag
        _local_base["diagnoses"] = _pt_diag
        _local_base["admissions"] = _pt_adm
        _local_complete = dict(complete_data)
        _local_complete["base_info"] = _local_base
        _bg_cache: Dict[str, Dict[str, Any]] = {}

        for timepoint in timepoints if max_labels == -1 else timepoints[:max_labels]:
            target_time = timepoint["measurement_time"]
            target_egfr = timepoint["target_value"]
            target_itemid = int(timepoint["itemid"])
            
            for pred_days in prediction_days:
                cutoff_time = target_time - timedelta(days=pred_days)
                
                windowed_features: Dict[str, Any] = {}
                window_name = "4weeks"
                window_days = 28

                apply_gate = (pred_days > 0) or bool(getattr(args, "quality_gate_include_current", False))
                if apply_gate:
                    if not passes_quality_gate(
                        clinical_data=_pt_clinical,
                        patient_id=int(patient_id),
                        cutoff_time=pd.Timestamp(cutoff_time),
                        window_days=int(window_days),
                        min_scr_count=int(getattr(args, "min_scr_count", 0)),
                        min_scr_days=int(getattr(args, "min_scr_days", 0)),
                        min_observed_days=int(getattr(args, "min_observed_days", 0)),
                    ):
                        continue
                
                feats = extract_windowed_features(
                    _pt_clinical, patient_id, cutoff_time, window_days, args.output_lang,
                    end_time=cutoff_time, rx_format=getattr(args, "medication_format", "compact"),
                )
                if feats:
                    windowed_features[window_name] = feats
                else:
                    continue

                _ct_date = str(pd.Timestamp(cutoff_time).date())
                if _ct_date not in _bg_cache:
                    _bg_cache[_ct_date] = get_patient_background(
                        _local_complete, _pt_clinical, int(patient_id),
                        target_time=pd.Timestamp(cutoff_time), lang=args.output_lang,
                    )
                patient_bg = _bg_cache[_ct_date]

                include_hist = False
                mode = str(getattr(args, "egfr_history", "off")).lower()
                if mode == "on":
                    include_hist = True
                elif mode == "dropout":
                    include_hist = _deterministic_keep(
                        p_drop=float(getattr(args, "egfr_history_dropout_p", 0.5)),
                        patient_id=int(patient_id),
                        cutoff_time=pd.Timestamp(cutoff_time),
                        pred_days=int(pred_days),
                    )
                if include_hist:
                    history_text = extract_history_features(_pt_clinical, int(patient_id), cutoff_time, args.output_lang)
                    if history_text:
                        windowed_features["history_trend"] = history_text

                labels.append(
                    {
                        "patient_id": int(patient_id),
                        "target_time": target_time,
                        "cutoff_time": cutoff_time,
                        "pred_days": pred_days,
                        "target_egfr": float(target_egfr) if target_egfr is not None else None,
                        "itemid": target_itemid,
                        "windowed_features": windowed_features,
                        "label_type": "egfr_prognosis" if pred_days > 0 else "egfr_current",
                        "window_name": window_name,
                        "window_days": window_days,
                        "patient_background": patient_bg,
                        "clinical_data_for_snapshot": _pt_clinical,
                    }
                )
        return labels, excluded_count

    def create_current_target_aki_labels(patient_id: int, max_labels: int) -> Tuple[List[Dict[str, Any]], int]:
        label_source = args.aki_label_source.lower()
        policy = args.hybrid_policy.lower()
        base_info = complete_data.get("base_info", {})
        diag_df: pd.DataFrame = base_info.get("diagnoses_icd")
        admissions: pd.DataFrame = base_info.get("admissions")

        # --- 성능 최적화: 환자별 DataFrame 사전 필터링 ---
        _pt_clinical_aki = _prefilter_clinical_for_patient(clinical_data, patient_id)

        _diag_lookup_aki = complete_data.get("_diag_by_patient", {})
        _diag_aki = _diag_lookup_aki.get(patient_id, pd.DataFrame()) if _diag_lookup_aki else (diag_df[diag_df["subject_id"] == patient_id] if isinstance(diag_df, pd.DataFrame) else pd.DataFrame())
        _adm_aki = admissions[admissions["subject_id"] == patient_id] if isinstance(admissions, pd.DataFrame) else pd.DataFrame()
        _local_base_aki = dict(base_info)
        _local_base_aki["diagnoses_icd"] = _diag_aki
        _local_base_aki["diagnoses"] = _diag_aki
        _local_base_aki["admissions"] = _adm_aki
        _local_complete_aki = dict(complete_data)
        _local_complete_aki["base_info"] = _local_base_aki
        _bg_cache_aki: Dict[str, Dict[str, Any]] = {}

        labels: List[Dict[str, Any]] = []
        excluded_by_procedures_aki = 0

        # admissions 시간 컬럼 보정
        if isinstance(admissions, pd.DataFrame) and not admissions.empty:
            admissions = admissions.copy()
            for col in ["admittime", "dischtime"]:
                if col in admissions.columns and not pd.api.types.is_datetime64_any_dtype(admissions[col]):
                    admissions[col] = pd.to_datetime(admissions[col], errors="coerce")

        # ========== (1) 진단 기반 앵커 수집 ==========
        diag_anchors: List[pd.Timestamp] = []
        if isinstance(diag_df, pd.DataFrame) and isinstance(admissions, pd.DataFrame):
            patient_diag = diag_df[diag_df["subject_id"] == patient_id]
            for _, row in patient_diag.iterrows():
                icd_code = str(row.get("icd_code", ""))
                if any(code in icd_code for code in AKI_CODES):
                    hadm_id = row.get("hadm_id")
                    admission = admissions[(admissions["subject_id"] == patient_id) & (admissions["hadm_id"] == hadm_id)]
                    if len(admission) == 0:
                        continue
                    adm_record = admission.iloc[0]
                    admit_time = pd.to_datetime(adm_record.get("admittime"))
                    if pd.notna(admit_time):
                        diag_anchors.append(admit_time)

        # ========== (2) KDIGO onset(전역) ==========
        kdigo_info = compute_kdigo_onset_for_patient(_pt_clinical_aki, patient_id) if label_source in ["kdigo", "hybrid"] else None
        kdigo_anchor = kdigo_info.get("onset_time") if kdigo_info else None

        # ========== (3) 앵커 합성(정책) ==========
        anchor_pairs: List[Tuple[pd.Timestamp, str]] = []
        if label_source == "diagnosis":
            anchor_pairs = [(pd.Timestamp(t), "diagnosis") for t in diag_anchors]
        elif label_source == "kdigo":
            if kdigo_anchor is not None:
                anchor_pairs = [(pd.Timestamp(kdigo_anchor), "kdigo")]
        else:  # hybrid
            if policy == "priority":
                if len(diag_anchors) > 0:
                    anchor_pairs = [(pd.Timestamp(t), "diagnosis") for t in diag_anchors]
                elif kdigo_anchor is not None:
                    anchor_pairs = [(pd.Timestamp(kdigo_anchor), "kdigo")]
            elif policy == "union":
                anchor_pairs = [(pd.Timestamp(t), "diagnosis") for t in diag_anchors]
                if kdigo_anchor is not None:
                    anchor_pairs.append((pd.Timestamp(kdigo_anchor), "kdigo"))
                # dedup by timestamp
                uniq: Dict[pd.Timestamp, str] = {}
                for t, src in anchor_pairs:
                    uniq[t] = src if t not in uniq else uniq[t]
                anchor_pairs = sorted([(t, src) for t, src in uniq.items()])
            else:  # intersection
                if kdigo_anchor is not None and len(diag_anchors) > 0:
                    nearest = min([abs(pd.Timestamp(a) - pd.Timestamp(kdigo_anchor)) for a in diag_anchors])
                    if nearest <= pd.Timedelta(hours=48):
                        anchor_pairs = [(pd.Timestamp(kdigo_anchor), "intersection")]

        # ========== (4) 진단 앵커에 대해 입원창 KDIGO 이벤트 확장(옵션) ==========
        if args.aki_require_kdigo_consistency and len(anchor_pairs) > 0:
            admissions_df: pd.DataFrame = complete_data.get("base_info", {}).get("admissions")
            if isinstance(admissions_df, pd.DataFrame) and len(admissions_df) > 0:
                expanded: List[Tuple[pd.Timestamp, str, Dict[str, Any]]] = []
                for t, src in anchor_pairs:
                    if args.aki_scan_all_admissions:
                        adm = admissions_df[(admissions_df["subject_id"] == patient_id)]
                    else:
                        adm_mask = (
                            (admissions_df["subject_id"] == patient_id)
                            & (pd.to_datetime(admissions_df["admittime"], errors="coerce") <= pd.Timestamp(t))
                            & (pd.to_datetime(admissions_df["dischtime"], errors="coerce") >= pd.Timestamp(t))
                        )
                        adm = admissions_df[adm_mask]
                    if len(adm) == 0:
                        continue
                    for _, row_adm in adm.iterrows():
                        adm_start = pd.to_datetime(row_adm.get("admittime"))
                        adm_end = pd.to_datetime(row_adm.get("dischtime"))
                        if pd.notna(adm_start) and pd.notna(adm_end):
                            evs = compute_kdigo_events_in_window(
                                clinical_data,
                                int(patient_id),
                                adm_start,
                                adm_end,
                                baseline_quantile=getattr(args, "kdigo_baseline_quantile", None),
                            )
                            for ev in evs:
                                expanded.append((pd.Timestamp(ev["onset_time"]), "diag_kdigo_onset", ev.get("evidence", {})))
                if len(expanded) > 0:
                    uniq = {}
                    for tt, src, evd in expanded:
                        uniq[tt] = (src, evd)
                    anchor_pairs = [(t, uniq[t][0]) for t in sorted(uniq.keys())]
                    anchor_evidence_map = {t: uniq[t][1] for t in uniq.keys()}
                else:
                    anchor_evidence_map = {}
            else:
                anchor_evidence_map = {}
        else:
            anchor_evidence_map = {}

        # 정렬 및 상한
        anchor_pairs = sorted(anchor_pairs) if max_labels == -1 else sorted(anchor_pairs)[:max_labels]
        anchor_times = [t for t, _ in anchor_pairs]

        # 예측 시점 목록: 현재(0일) + 1일~7일 전 (eGFR과 동일)
        prediction_days_str = getattr(args, "prediction_windows", "0,1,2,3,4,5,6,7")
        prediction_days = [int(x.strip()) for x in prediction_days_str.split(",") if x.strip().isdigit()]

        # ===== 헬퍼: SCr 밀도 체크 (음성 샘플 생성용) =====
        lab_df_all: pd.DataFrame = clinical_data.get("lab_events")
        scr_df = None
        if isinstance(lab_df_all, pd.DataFrame) and not lab_df_all.empty:
            scr_df = lab_df_all[
                (lab_df_all["subject_id"] == patient_id)
                & (lab_df_all["itemid"].isin(FEATURE_ITEMIDS.get("creatinine", [])))
                & (pd.notna(lab_df_all.get("valuenum")))
            ].copy()
            if not scr_df.empty and not pd.api.types.is_datetime64_any_dtype(scr_df["charttime"]):
                scr_df["charttime"] = pd.to_datetime(scr_df["charttime"], errors="coerce")
            if not scr_df.empty:
                scr_df = scr_df[pd.notna(scr_df["charttime"])].sort_values("charttime")

        # ========== (5) 양성 라벨 생성: 라벨은 고정, 윈도우는 gating ==========
        positives: List[Dict[str, Any]] = []
        for target_time, anchor_src in anchor_pairs:
            # 🚫 Procedure 제외 조건 확인 (일상생활 기반 예측)
            exclusion_info = should_exclude_timepoint_for_procedures(_pt_clinical_aki, patient_id, target_time)
            if exclusion_info["exclude"]:
                excluded_by_procedures_aki += 1
                stats = globals().get("procedure_exclusion_stats")
                if isinstance(stats, dict):
                    stats["excluded_count"] = stats.get("excluded_count", 0) + 1
                    stats.setdefault("excluded_patients", set()).add(int(patient_id))
                    reasons_map = stats.setdefault("exclusion_reasons", {})
                    for reason in exclusion_info.get("reasons", []):
                        reasons_map[reason] = reasons_map.get(reason, 0) + 1
                continue
            
            # (옵션) diagnosis/hybrid에서도 KDIGO 일치 강제면, 일치 안 하면 스킵
            ev_for_label = compute_kdigo_evidence_at_time(
                _pt_clinical_aki,
                int(patient_id),
                target_time,
                baseline_quantile=getattr(args, "kdigo_baseline_quantile", None),
            )
            if args.aki_require_kdigo_consistency and anchor_src in ["diagnosis", "union", "priority"]:
                if not (ev_for_label.get("meet48") or ev_for_label.get("meet7")):
                    continue

            # 각 예측 시점(0일=현재, 1~7일=과거로 돌아가서 예측)에 대해 샘플 생성
            for pred_days in prediction_days:
                # Input Cutoff Time: target_time에서 N일 전
                cutoff_time = target_time - timedelta(days=pred_days)
                
                # v7 Prognosis 모드: 28일 윈도우만 사용 (eGFR과 동일)
                windowed_features: Dict[str, Any] = {}
                window_name = "4weeks"
                window_days = 28

                # 품질 게이팅(앵커 기준): pred_days>0 샘플에서만 기본 적용
                apply_gate = (pred_days > 0) or bool(getattr(args, "quality_gate_include_current", False))
                if apply_gate:
                    if not passes_quality_gate(
                        clinical_data=_pt_clinical_aki,
                        patient_id=int(patient_id),
                        cutoff_time=pd.Timestamp(cutoff_time),
                        window_days=int(window_days),
                        min_scr_count=int(getattr(args, "min_scr_count", 0)),
                        min_scr_days=int(getattr(args, "min_scr_days", 0)),
                        min_observed_days=int(getattr(args, "min_observed_days", 0)),
                    ):
                        continue
                
                feats = extract_windowed_features(
                    _pt_clinical_aki, patient_id, cutoff_time, window_days, args.output_lang,
                    end_time=cutoff_time, rx_format=getattr(args, "medication_format", "compact"),
                )
                if feats:
                    windowed_features[window_name] = feats
                else:
                    continue

                if not windowed_features:
                    continue

                _ct_date_aki = str(pd.Timestamp(cutoff_time).date())
                if _ct_date_aki not in _bg_cache_aki:
                    _bg_cache_aki[_ct_date_aki] = get_patient_background(
                        _local_complete_aki, _pt_clinical_aki, int(patient_id),
                        target_time=pd.Timestamp(cutoff_time), lang=args.output_lang,
                    )
                patient_bg = _bg_cache_aki[_ct_date_aki]

                positives.append(
                    {
                        "patient_id": int(patient_id),
                        "target_time": target_time,
                        "cutoff_time": cutoff_time,
                        "pred_days": pred_days,
                        "target_aki": True,
                        "windowed_features": windowed_features,
                        "label_type": "aki_prognosis" if pred_days > 0 else "aki_current",
                        "window_name": window_name,
                        "window_days": window_days,
                        "patient_background": patient_bg,
                        "clinical_data_for_snapshot": _pt_clinical_aki,
                        "metadata": {
                            "label_source": label_source,
                            "anchor_source": anchor_src,
                            "kdigo_stage": (kdigo_info.get("kdigo_stage") if kdigo_info else None),
                            "baseline_scr": (kdigo_info.get("baseline_scr") if kdigo_info else None),
                            "kdigo_evidence": ev_for_label,
                        },
                    }
                )

        labels.extend(positives)

        # ========== (6) 음성 샘플 생성: Prognosis 모드 적용 ==========
        target_neg_labels = int(len(positives) * max(args.aki_neg_pos_ratio, 0.0))
        if target_neg_labels <= 0:
            return labels, excluded_by_procedures_aki

        candidate_times: List[pd.Timestamp] = []
        if scr_df is not None and not scr_df.empty:
            candidate_times = list(pd.to_datetime(scr_df["charttime"]).drop_duplicates().tolist())

        anchors_set = set(pd.to_datetime(anchor_times))
        exclusion_days = 2

        def far_from_anchors(ts: pd.Timestamp) -> bool:
            return all(abs(ts - a) > pd.Timedelta(days=exclusion_days) for a in anchors_set)

        added_neg = 0
        used_times: set = set()

        for t in candidate_times:
            if added_neg >= target_neg_labels:
                break
            t = pd.Timestamp(t)
            if not far_from_anchors(t):
                continue
            if t in used_times:
                continue

            exclusion_info = should_exclude_timepoint_for_procedures(_pt_clinical_aki, patient_id, t)
            if exclusion_info["exclude"]:
                excluded_by_procedures_aki += 1
                stats = globals().get("procedure_exclusion_stats")
                if isinstance(stats, dict):
                    stats["excluded_count"] = stats.get("excluded_count", 0) + 1
                    stats.setdefault("excluded_patients", set()).add(int(patient_id))
                    reasons_map = stats.setdefault("exclusion_reasons", {})
                    for reason in exclusion_info.get("reasons", []):
                        reasons_map[reason] = reasons_map.get(reason, 0) + 1
                continue

            ev_t = compute_kdigo_evidence_at_time(
                _pt_clinical_aki,
                int(patient_id),
                t,
                baseline_quantile=getattr(args, "kdigo_baseline_quantile", None),
            )
            if ev_t.get("meet48") or ev_t.get("meet7"):
                continue

            for pred_days in prediction_days:
                cutoff_time = t - timedelta(days=pred_days)
                
                windowed_features: Dict[str, Any] = {}
                window_name = "4weeks"
                window_days = 28

                apply_gate = (pred_days > 0) or bool(getattr(args, "quality_gate_include_current", False))
                if apply_gate:
                    if not passes_quality_gate(
                        clinical_data=_pt_clinical_aki,
                        patient_id=int(patient_id),
                        cutoff_time=pd.Timestamp(cutoff_time),
                        window_days=int(window_days),
                        min_scr_count=int(getattr(args, "min_scr_count", 0)),
                        min_scr_days=int(getattr(args, "min_scr_days", 0)),
                        min_observed_days=int(getattr(args, "min_observed_days", 0)),
                    ):
                        continue
                
                feats = extract_windowed_features(
                    _pt_clinical_aki, patient_id, cutoff_time, window_days, args.output_lang,
                    end_time=cutoff_time, rx_format=getattr(args, "medication_format", "compact"),
                )
                if feats:
                    windowed_features[window_name] = feats
                else:
                    continue

                if not windowed_features:
                    continue

                _ct_date_neg = str(pd.Timestamp(cutoff_time).date())
                if _ct_date_neg not in _bg_cache_aki:
                    _bg_cache_aki[_ct_date_neg] = get_patient_background(
                        _local_complete_aki, _pt_clinical_aki, int(patient_id),
                        target_time=pd.Timestamp(cutoff_time), lang=args.output_lang,
                    )
                patient_bg_neg = _bg_cache_aki[_ct_date_neg]

                labels.append(
                    {
                        "patient_id": int(patient_id),
                        "target_time": t,
                        "cutoff_time": cutoff_time,
                        "pred_days": pred_days,
                        "target_aki": False,
                        "windowed_features": windowed_features,
                        "label_type": "aki_prognosis" if pred_days > 0 else "aki_current",
                        "window_name": window_name,
                        "window_days": window_days,
                        "patient_background": patient_bg_neg,
                        "clinical_data_for_snapshot": _pt_clinical_aki,
                        "metadata": {"label_source": "negative_random_nonpositive"},
                    }
                )
                added_neg += 1
                if added_neg >= target_neg_labels:
                    break

            used_times.add(t)

        return labels, excluded_by_procedures_aki




    def _process_one(pid: int) -> Dict[str, Any]:
        try:
            local_records: List[Dict[str, Any]] = []
            eg, ak = 0, 0
            eg_timepoints, ak_timepoints = 0, 0
            proc_ex_egfr, proc_ex_aki = 0, 0
            
            if do_egfr:
                egfr_labels, egfr_exclusions = create_current_target_egfr_labels(pid, args.max_labels_per_patient)
                # 프롬프트 수 (기존 방식)
                eg = len(egfr_labels)
                # 시점 수 (새로 추가)
                eg_timepoints = len({rec["target_time"] for rec in egfr_labels})
                proc_ex_egfr += egfr_exclusions
                for label in egfr_labels:
                    local_records.append(build_prompt_for_egfr(label, args.output_format, args.output_lang))
            if do_aki:
                aki_labels, aki_exclusions = create_current_target_aki_labels(pid, args.max_labels_per_patient)
                # 프롬프트 수 (기존 방식)
                ak = len(aki_labels)
                # 시점 수 (새로 추가)
                ak_timepoints = len({rec["target_time"] for rec in aki_labels})
                proc_ex_aki += aki_exclusions
                for label in aki_labels:
                    local_records.append(build_prompt_for_aki(label, args.output_format, args.output_lang))
            
            # 시술/수술 제외 통계: create_current_target_egfr_labels에서 egfr_exclusions로 반환됨
            
            return {
                "records": local_records, 
                "eg": eg, "ak": ak,
                "eg_timepoints": eg_timepoints, "ak_timepoints": ak_timepoints,
                "proc_ex_egfr": proc_ex_egfr, "proc_ex_aki": proc_ex_aki,
            }
        except Exception as e:
            print(f"ERROR processing patient {pid}: {e}", flush=True)
            print(_tb.format_exc(), flush=True)
            return {"records": [], "eg": 0, "ak": 0, "eg_timepoints": 0, "ak_timepoints": 0, "procedure_exclusions": 0}

    total_egfr_labels = 0
    total_aki_labels = 0
    start_time = datetime.now()
    
    if args.workers and args.workers > 0:
        print(f"[PARALLEL] 병렬 실행: workers={args.workers}, 대상 환자: {len(test_patients)}명")
        with _fut.ThreadPoolExecutor(max_workers=args.workers) as ex:
            # 최고 성능: 완료되는 대로 바로 처리 (데이터 안전성은 보장됨)
            futures = {ex.submit(_process_one, pid): pid for pid in test_patients}
            for i, fut in enumerate(_fut.as_completed(futures), 1):
                pid = futures[fut]
                try:
                    res = fut.result()
                    generated.extend(res["records"])
                    total_egfr_labels += res["eg"]
                    total_aki_labels += res["ak"]
                    
                    # 진행상황 계산
                    progress_pct = (i / len(test_patients)) * 100
                    elapsed = datetime.now() - start_time
                    avg_time_per_patient = elapsed.total_seconds() / i
                    remaining_patients = len(test_patients) - i
                    eta_seconds = avg_time_per_patient * remaining_patients
                    eta = timedelta(seconds=int(eta_seconds))
                    
                    # 첫 번째 결과 특별 메시지
                    if i == 1:
                        print(f"[OK] 첫 번째 환자 처리 완료! (ID: {pid}) - 이제 본격적인 병렬 처리 시작")
                    
                    # 매 1명마다 출력
                    if True:
                        prompts_per_patient = len(generated) / i if i > 0 else 0
                        current_patient_prompts = len(res["records"])
                        
                        # 프롬프트 수와 시점 수를 구분하여 표시  
                        prompt_info = f"프롬프트 {current_patient_prompts}개 (eGFR:{res.get('eg', 0)}, AKI:{res.get('ak', 0)})"
                        timepoint_info = f"시점(eGFR:{res.get('eg_timepoints', 0)}, AKI:{res.get('ak_timepoints', 0)})"
                        
                        exclusion_info = ""
                        pe_e = res.get("proc_ex_egfr", 0)
                        pe_a = res.get("proc_ex_aki", 0)
                        if (pe_e + pe_a) > 0:
                            exclusion_info = f" | Proc_제외[eGFR:{pe_e}, AKI:{pe_a}]"
                        
                        print(f"[PROGRESS] {i}/{len(test_patients)}명 ({progress_pct:.1f}%) | "
                              f"ID: {pid} | "
                              f"누적: {len(generated)}개 (Avg: {prompts_per_patient:.1f}) | "
                              f"{prompt_info} {timepoint_info} | "
                              f"경과: {str(elapsed).split('.')[0]} | "
                              f"예상완료: {eta}{exclusion_info}", flush=True)
                except Exception as e:
                    print(f"[ERROR] 환자 {pid} 처리 실패: {e}", flush=True)
    else:
        print(f"[SEQUENTIAL] 순차 실행: 대상 환자: {len(test_patients)}명")
        for idx, patient_id in enumerate(test_patients):
            res = _process_one(patient_id)
            generated.extend(res["records"])
            total_egfr_labels += res["eg"]
            total_aki_labels += res["ak"]
            
            # 진행상황 계산
            progress_pct = ((idx + 1) / len(test_patients)) * 100
            elapsed = datetime.now() - start_time
            avg_time_per_patient = elapsed.total_seconds() / (idx + 1)
            remaining_patients = len(test_patients) - (idx + 1)
            eta_seconds = avg_time_per_patient * remaining_patients
            eta = timedelta(seconds=int(eta_seconds))
            
            # 매 1명마다 출력 (순차는 더 자주)
            prompts_per_patient = len(generated) / (idx + 1) if (idx + 1) > 0 else 0
            
            # 시술/수술 제외 정보 추가(환자별 eGFR/AKI 분리 표기)
            exclusion_info = ""
            pe_e = res.get("proc_ex_egfr", 0)
            pe_a = res.get("proc_ex_aki", 0)
            if (pe_e + pe_a) > 0:
                exclusion_info = f" | Proc_제외[eGFR:{pe_e}, AKI:{pe_a}]"
            
            print(f"[PROGRESS] 환자 {idx+1}/{len(test_patients)} ({progress_pct:.1f}%) | "
                  f"ID: {patient_id} | 프롬프트: {len(generated)}개 ({prompts_per_patient:.1f}/환자) | "
                  f"경과: {str(elapsed).split('.')[0]} | 예상완료: {eta}{exclusion_info}", flush=True)

    # 저장 (파일명에 시간 및 설정 정보 추가)
    original_path = Path(args.output)
    
    # 상대 경로이고 디렉토리가 지정되지 않은 경우 processed_data/generated_prompts 폴더 사용
    if not original_path.is_absolute() and original_path.parent == Path('.'):
        original_path = Path("processed_data/generated_prompts") / original_path.name
        print(f"[OUTPUT] 출력 폴더 자동 설정: {original_path}")
    
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    
    # 환자 수 표기 (0은 전체를 의미)
    patient_str = "ALL" if args.max_patients == 0 else f"P{args.max_patients}"
    # LALL = egfr+aki 모두, LEGFR = egfr만, LAKI = aki만 (태스크 기준)
    if do_egfr and do_aki:
        label_str = "LALL"
    elif do_egfr:
        label_str = "LEGFR"
    elif do_aki:
        label_str = "LAKI"
    else:
        label_str = "LALL"  # fallback
    
    # 확장자 분리하여 정보 삽입
    stem = original_path.stem  # 파일명 (확장자 제외)
    suffix = original_path.suffix  # 확장자
    timestamped_name = f"{stem}_{patient_str}_{label_str}_{timestamp}{suffix}"
    output_path = original_path.parent / timestamped_name
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # === (추가) AKI 양/음성 개수 집계 ===
    aki_pos = sum(
        1 for r in generated
        if r.get("metadata", {}).get("task_type") == "aki_detection"
        and r["metadata"].get("target_aki") is True
    )
    aki_neg = sum(
        1 for r in generated
        if r.get("metadata", {}).get("task_type") == "aki_detection"
        and r["metadata"].get("target_aki") is False
    )
    aki_ratio = (aki_neg / aki_pos) if aki_pos else float("inf")

    # 파일 저장 형식 결정 (확장자 자동 감지 포함)
    file_format = args.file_format
    
    # 확장자 기반 자동 변환은 하지 않음 (기본값 json-pretty 우선)
    # 사용자가 명시적으로 --file-format jsonl을 지정한 경우에만 JSONL 형식 사용
    if output_path.suffix.lower() == '.json':
        if args.file_format == "json-pretty":  # 기본값인 경우만
            print(f"[OUTPUT] 파일 확장자(.json)에 따라 JSON Array 형식으로 저장")
    
    # 파일 저장
    if file_format == "json-pretty":
        # 가독성 좋은 JSON Array 형식
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(generated, f, ensure_ascii=False, indent=2)
    else:
        # JSONL 형식
        with open(output_path, "w", encoding="utf-8") as f:
            for record in generated:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

    total_elapsed = datetime.now() - start_time
    prompts_per_sec = len(generated) / total_elapsed.total_seconds() if total_elapsed.total_seconds() > 0 else 0
    
    print(f"\n[COMPLETE] 프롬프트 생성 완료!")
    print(f"[STATS] 총 프롬프트 수: {len(generated)}개")
    print(f" - eGFR 레이블: {total_egfr_labels}개")
    print(f" - AKI 레이블: {total_aki_labels}개 (양성: {aki_pos}개, 음성: {aki_neg}개)")
    print(f" - AKI 양성/음성 비율: {aki_ratio:.2f}")
    print(f"[TIME] 총 소요시간: {str(total_elapsed).split('.')[0]}")
    print(f"[SPEED] 처리 속도: {prompts_per_sec:.2f} 프롬프트/초")
    print(f"[SAVED] 저장 완료: {output_path}")

    # 📊 Procedure 제외 요약 출력 (사유 포함)
    stats = globals().get("procedure_exclusion_stats")
    if isinstance(stats, dict):
        total_ex = int(stats.get("excluded_count", 0))
        uniq_patients = len(stats.get("excluded_patients", set()))
        reasons = stats.get("exclusion_reasons", {})
        if total_ex > 0:
            print("\n🧪 Procedure-based exclusions summary")
            print(f" - Excluded timepoints: {total_ex}")
            print(f" - Affected patients: {uniq_patients}")
            if reasons:
                print(" - Reasons (desc):")
                for reason, cnt in sorted(reasons.items(), key=lambda pair: pair[1], reverse=True):
                    print(f"    · {reason}: {cnt}")

    # 🧹 품질 게이팅 요약 출력
    qstats = globals().get("quality_gate_stats")
    if isinstance(qstats, dict):
        skipped = int(qstats.get("skipped_count", 0))
        uniq = len(qstats.get("skipped_patients", set()))
        reasons = qstats.get("reasons", {})
        if skipped > 0:
            print("\n🧹 Quality gate summary (anchor-based)")
            print(f" - Skipped samples: {skipped}")
            print(f" - Affected patients: {uniq}")
            if reasons:
                print(" - Reasons (desc):")
                for reason, cnt in sorted(reasons.items(), key=lambda pair: pair[1], reverse=True):
                    print(f"    · {reason}: {cnt}")


if __name__ == "__main__":
    main()


