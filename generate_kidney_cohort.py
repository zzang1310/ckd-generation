#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏥 MIMIC-IV 신장질환 환자 코호트 추출 스크립트

이 스크립트는 MIMIC-IV v3.1 데이터베이스에서 CKD(만성신장질환) 및 AKI(급성신장손상) 
환자들을 ICD 코드 기반으로 추출하여 기본 코호트를 생성합니다.

📋 생성되는 데이터:
- patient_ids: 신장질환 환자 ID 목록
- diagnoses: 신장질환 진단 데이터
- patients_info: 환자 기본 정보 (성별, 나이 등)
- admissions: 입원 기록
- 추출 메타데이터

🎯 사용처: extract_with_egfr_full_fast.py의 필수 사전 요구사항

📋 분석 완료: Load_MIMIC-IV_v0.1.ipynb와 v0.2.ipynb 비교 분석 후 
v0.2의 확장 기능은 이미 extract_with_egfr_full_fast.py에 더 발전된 형태로 구현됨

Author: CKD-LLM Project Team
Version: 1.0.0
Date: 2025-08-27
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import argparse

warnings.filterwarnings('ignore')

def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="MIMIC-IV 신장질환 환자 코호트 추출")
    parser.add_argument("--mimic-path", type=str, default="D:/CKD-LLM/mimic-iv-3.1",
                       help="MIMIC-IV 데이터 루트 경로")
    parser.add_argument("--output-path", type=str, default="processed_data/intermediate",
                       help="결과 저장 경로")
    parser.add_argument("--force-reload", action="store_true",
                       help="기존 저장된 데이터 무시하고 새로 추출")
    parser.add_argument("--test-mode", action="store_true",
                       help="테스트 모드 (진단 데이터 일부만 처리)")
    return parser.parse_args()

# =====================================================================
# ICD 코드 정의 (CKD/AKI 관련)
# =====================================================================

# ICD-10 신장질환 코드
CKD_ICD10_CODES = [
    'N18',    # Chronic kidney disease
    'N181',   # CKD stage 1
    'N182',   # CKD stage 2
    'N183',   # CKD stage 3
    'N184',   # CKD stage 4
    'N185',   # CKD stage 5
    'N186',   # End stage renal disease
    'N189'    # CKD unspecified
]

# AKI 관련 코드 (ICD-10)
AKI_ICD10_CODES = [
    'N17',    # Acute kidney failure
    'N170',   # AKI with tubular necrosis
    'N171',   # AKI with acute cortical necrosis
    'N172',   # AKI with medullary necrosis
    'N178',   # Other AKI
    'N179'    # AKI unspecified
]

# ICD-9 신장질환 코드 (MIMIC-IV에서 혼용)
CKD_ICD9_CODES = [
    '585',    # Chronic kidney disease
    '5851',   # CKD stage 1
    '5852',   # CKD stage 2
    '5853',   # CKD stage 3
    '5854',   # CKD stage 4
    '5855',   # CKD stage 5
    '5856',   # End stage renal disease
    '5859'    # CKD unspecified
]

AKI_ICD9_CODES = [
    '584',    # Acute kidney failure
    '5840',   # AKI unspecified
    '5841',   # AKI with acute cortical necrosis
    '5845',   # AKI with tubular necrosis
    '5846',   # AKI with acute kidney failure, prerenal
    '5847',   # AKI with acute kidney failure, postrenal
    '5848',   # AKI with other specified acute kidney failure
    '5849'    # AKI unspecified
]

# 전체 신장질환 코드 통합
ALL_KIDNEY_CODES = CKD_ICD10_CODES + AKI_ICD10_CODES + CKD_ICD9_CODES + AKI_ICD9_CODES

# =====================================================================
# 유틸리티 함수들
# =====================================================================

def load_mimic_table(table_name, data_path, sample_rows=None, compression='infer'):
    """
    MIMIC-IV 테이블을 효율적으로 로딩하는 함수
    
    Args:
        table_name: 테이블명 (예: 'patients', 'admissions')
        data_path: 데이터 경로 (HOSP_PATH 또는 ICU_PATH)
        sample_rows: 샘플링할 행 수 (None이면 전체)
        compression: 압축 형식 ('infer', 'gzip', None)
    
    Returns:
        pandas DataFrame 또는 None
    """
    file_path = Path(data_path) / f"{table_name}.csv"
    
    # 압축 파일 확인
    if not file_path.exists():
        file_path = Path(data_path) / f"{table_name}.csv.gz"
        if file_path.exists():
            compression = 'gzip'
    
    if not file_path.exists():
        print(f"❌ {table_name} 파일을 찾을 수 없습니다: {file_path}")
        return None
    
    try:
        print(f"📁 로딩 중: {table_name} ({'샘플 ' + str(sample_rows) + '행' if sample_rows else '전체'})")
        
        if sample_rows:
            df = pd.read_csv(file_path, compression=compression, nrows=sample_rows)
        else:
            df = pd.read_csv(file_path, compression=compression)
        
        print(f"✅ {table_name}: {df.shape[0]:,}행 × {df.shape[1]}열")
        return df
        
    except Exception as e:
        print(f"❌ {table_name} 로딩 실패: {e}")
        return None

def save_processed_data(data, filepath, description=""):
    """전처리된 데이터를 pkl 형식으로 저장"""
    try:
        # 디렉토리 생성
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        file_size = filepath.stat().st_size / (1024**2)  # MB
        print(f"✅ 저장 완료 ({file_size:.1f}MB): {filepath}")
        if description:
            print(f"   📋 {description}")
        return True
        
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        return False

def load_processed_data(filepath):
    """저장된 전처리 데이터 로딩"""
    if filepath.exists():
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ {filepath.name} 로딩 실패: {e}")
            return None
    return None

# =====================================================================
# 신장질환 코호트 추출 함수
# =====================================================================

def extract_kidney_cohort_with_chunks(diagnoses_file, test_mode=False):
    """
    대용량 진단 데이터를 청크 단위로 처리하여 신장질환 환자 추출
    
    Args:
        diagnoses_file: diagnoses_icd.csv 파일 경로
        test_mode: True시 일부 청크만 처리 (테스트용)
    
    Returns:
        tuple: (kidney_patients, kidney_diagnoses)
    """
    print("🔍 진단 데이터에서 신장질환 환자 추출 중...")
    
    chunk_size = 50000
    kidney_patients = set()
    kidney_diagnoses_list = []
    chunk_count = 0
    max_chunks = 10 if test_mode else None
    
    try:
        for chunk in pd.read_csv(diagnoses_file, chunksize=chunk_size):
            chunk_count += 1
            print(f"  📦 청크 {chunk_count} 처리 중... (행수: {len(chunk):,})")
            
            # 정확한 매칭과 접두어 매칭 모두 사용
            kidney_chunk = chunk[
                chunk['icd_code'].isin(ALL_KIDNEY_CODES) |
                chunk['icd_code'].str.startswith(('N17', 'N18', '584', '585'), na=False)
            ].copy()
            
            if not kidney_chunk.empty:
                kidney_patients.update(kidney_chunk['subject_id'].unique())
                kidney_diagnoses_list.append(kidney_chunk)
                
                print(f"    ✅ 이 청크에서 신장질환 진단 {len(kidney_chunk)}건 발견")
                
                # 상위 5개 ICD 코드 출력
                top_codes = kidney_chunk['icd_code'].value_counts().head(5)
                print(f"    📋 주요 ICD 코드: {dict(top_codes)}")
            
            # 테스트 모드에서 청크 제한
            if max_chunks and chunk_count >= max_chunks:
                print(f"  ⏰ 테스트 모드: {chunk_count}개 청크만 처리")
                break
                
    except Exception as e:
        print(f"❌ 진단 데이터 처리 중 오류: {e}")
        return None, None
    
    # 결과 통합
    if kidney_diagnoses_list:
        kidney_diagnoses = pd.concat(kidney_diagnoses_list, ignore_index=True)
        print(f"\n📊 신장질환 환자 추출 결과:")
        print(f"  👥 신장질환 환자 수: {len(kidney_patients):,}명")
        print(f"  📋 신장질환 진단 건수: {len(kidney_diagnoses):,}건")
        
        return list(kidney_patients), kidney_diagnoses
    else:
        print("❌ 신장질환 환자를 찾을 수 없습니다!")
        return None, None

def extract_kidney_cohort(mimic_path, output_path, force_reload=False, test_mode=False):
    """
    신장질환 환자 코호트를 추출하고 저장
    
    Args:
        mimic_path: MIMIC-IV 데이터 루트 경로
        output_path: 결과 저장 경로
        force_reload: True면 기존 저장된 데이터 무시하고 새로 추출
        test_mode: True면 테스트 모드로 실행
    
    Returns:
        dict: 필터링된 환자 데이터 또는 None
    """
    
    # 경로 설정
    mimic_path = Path(mimic_path)
    output_path = Path(output_path)
    hosp_path = mimic_path / "hosp"
    
    kidney_cohort_file = output_path / "kidney_cohort.pkl"
    
    # 기존 저장된 코호트 데이터가 있으면 로딩
    if not force_reload:
        saved_cohort = load_processed_data(kidney_cohort_file)
        if saved_cohort is not None:
            print("✅ 기존 저장된 신장질환 코호트 데이터 로딩 완료")
            print(f"   👥 환자 수: {saved_cohort['total_patients']:,}명")
            print(f"   📋 진단 건수: {saved_cohort['total_diagnoses']:,}건")
            
            # Stage 분포도 같이 출력
            kidney_diagnoses = saved_cohort['diagnoses']
            print("\n📊 CKD Stage별 분포 (ICD-9 + ICD-10):")
            stage_groups = {
                "Stage 1": {'icd10': ['N181'], 'icd9': ['5851']},
                "Stage 2": {'icd10': ['N182'], 'icd9': ['5852']},
                "Stage 3": {'icd10': ['N183'], 'icd9': ['5853']},
                "Stage 4": {'icd10': ['N184'], 'icd9': ['5854']},
                "Stage 5": {'icd10': ['N185'], 'icd9': ['5855']},
            }

            # 전체 합산을 위해 누적
            total_patients_all = 0
            total_diags_all = 0
            stage_stats = {}

            for stage, codes in stage_groups.items():
                # ICD-10
                mask10 = kidney_diagnoses['icd_code'].isin(codes['icd10'])
                pts10 = kidney_diagnoses.loc[mask10, 'subject_id'].nunique()
                diags10 = mask10.sum()

                # ICD-9
                mask9 = kidney_diagnoses['icd_code'].isin(codes['icd9'])
                pts9 = kidney_diagnoses.loc[mask9, 'subject_id'].nunique()
                diags9 = mask9.sum()

                # 합계
                total_patients = pts10 + pts9
                total_diags = diags10 + diags9

                total_patients_all += total_patients
                total_diags_all += total_diags

                stage_stats[stage] = {
                    "patients": total_patients, "diags": total_diags,
                    "icd10": (pts10, diags10), "icd9": (pts9, diags9)
                }

            # 최종 출력
            for stage, stats in stage_stats.items():
                pct_pat = stats["patients"] / total_patients_all * 100 if total_patients_all > 0 else 0
                pct_diag = stats["diags"] / total_diags_all * 100 if total_diags_all > 0 else 0
                print(f"  {stage}: {stats['patients']:,}명 ({stats['diags']:,}건) "
                    f"[환자 {pct_pat:.1f}%, 진단 {pct_diag:.1f}%]")
                print(f"     • ICD-10: {stats['icd10'][0]:,}명 ({stats['icd10'][1]:,}건)")
                print(f"     • ICD-9:  {stats['icd9'][0]:,}명 ({stats['icd9'][1]:,}건)")


            # diagnoses 테이블에서 CKD 코드를 범주별로 분해해 환자 수 비교
            dx = saved_cohort["diagnoses"].copy()
            dx["icd_code"] = dx["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)

            # 카테고리 정의
            is_ckd10_any = dx["icd_code"].str.startswith("N18")
            is_ckd9_any  = dx["icd_code"].str.startswith("585")

            # 스테이지 명시 (Stage 1~5)
            is_stage10 = dx["icd_code"].str.match(r"^N18[1-5]")
            is_stage9  = dx["icd_code"].str.match(r"^585[1-5]")

            # 미상세 CKD (unspecified)
            is_unspec10 = (dx["icd_code"] == "N18") | dx["icd_code"].str.match(r"^N189")
            is_unspec9  = (dx["icd_code"] == "585") | dx["icd_code"].str.match(r"^5859")

            # ESRD / Dialysis
            is_esrd10 = dx["icd_code"].str.match(r"^N18(0|6)")
            is_esrd9  = dx["icd_code"].str.match(r"^5856")

            # 환자 유니온(중복 제거) 카운트
            patients_stage   = dx.loc[is_stage10 | is_stage9, "subject_id"].nunique()
            patients_unspec  = dx.loc[is_unspec10 | is_unspec9, "subject_id"].nunique()
            patients_esrd    = dx.loc[is_esrd10  | is_esrd9,  "subject_id"].nunique()
            patients_ckd_any = dx.loc[is_ckd10_any | is_ckd9_any, "subject_id"].nunique()

            print("\n📊 CKD 환자 카테고리별(환자 유니온) 비교")
            print(f"  • CKD 전체(모든 N18*/585*): {patients_ckd_any:,}명")
            print(f"    ├─ 스테이지 명시(N181–N185, 5851–5855): {patients_stage:,}명")
            print(f"    ├─ 미상세(N18/N189, 585/5859):          {patients_unspec:,}명")
            print(f"    └─ ESRD/투석(N180/N186, 5856):          {patients_esrd:,}명")

            # 참고: 범주 간 중복 존재 → 단순 합산 ≠ CKD 전체
            # 겹침을 보고 싶다면 환자 레벨로 멀티라벨 집합을 만들어 교집합도 출력 가능
                
            return saved_cohort
    
    print("🔄 신장질환 환자 코호트 새로 추출 중...")
    print(f"📂 MIMIC-IV 경로: {mimic_path}")
    print(f"📂 출력 경로: {output_path}")
    
    # 경로 존재 여부 확인
    if not hosp_path.exists():
        print(f"❌ HOSP 경로가 존재하지 않습니다: {hosp_path}")
        return None
    
    # 1. 진단 데이터에서 신장질환 환자 추출 (청크 단위 처리)
    diagnoses_file = hosp_path / 'diagnoses_icd.csv'
    if not diagnoses_file.exists():
        print(f"❌ 진단 파일이 존재하지 않습니다: {diagnoses_file}")
        return None
    
    kidney_patients, kidney_diagnoses = extract_kidney_cohort_with_chunks(
        diagnoses_file, test_mode=test_mode
    )
    
    if kidney_patients is None or kidney_diagnoses is None:
        print("❌ 신장질환 환자 추출 실패")
        return None
    
    # 2. 환자 기본 정보 로딩
    print("\n📋 환자 기본 정보 매칭 중...")
    patients_df = load_mimic_table('patients', hosp_path)
    if patients_df is not None:
        kidney_patients_info = patients_df[patients_df['subject_id'].isin(kidney_patients)].copy()
        print(f"✅ 환자 기본 정보 {len(kidney_patients_info):,}명 매칭")
    else:
        print("⚠️ 환자 기본 정보 로딩 실패")
        kidney_patients_info = None
    
    # 3. 입원 정보 로딩
    print("\n🏥 입원 정보 매칭 중...")
    admissions_df = load_mimic_table('admissions', hosp_path)
    if admissions_df is not None:
        kidney_admissions = admissions_df[admissions_df['subject_id'].isin(kidney_patients)].copy()
        print(f"✅ 신장질환 환자 입원 {len(kidney_admissions):,}건 매칭")
    else:
        print("⚠️ 입원 정보 로딩 실패")
        kidney_admissions = None
    
    # 4. 코호트 데이터 패키징
    cohort_data = {
        'patient_ids': kidney_patients,
        'diagnoses': kidney_diagnoses,
        'patients_info': kidney_patients_info,
        'admissions': kidney_admissions,
        'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_patients': len(kidney_patients),
        'total_diagnoses': len(kidney_diagnoses),
        'icd_codes_used': ALL_KIDNEY_CODES,
        'test_mode': test_mode
    }
    
    # 5. 데이터 저장
    description = f"신장질환 환자 {len(kidney_patients):,}명, 진단 {len(kidney_diagnoses):,}건"
    if test_mode:
        description += " (테스트 모드)"
    
    if save_processed_data(cohort_data, kidney_cohort_file, description):
        print(f"\n✅ 코호트 추출 완료!")
    else:
        print(f"\n❌ 코호트 저장 실패")
        return None
    
    # 6. 결과 분석 및 요약
    print("\n📊 진단 코드별 분포 (상위 10개):")
    diagnosis_counts = kidney_diagnoses['icd_code'].value_counts().head(10)
    for code, count in diagnosis_counts.items():
        # ICD 코드 타입 확인
        if code in CKD_ICD10_CODES:
            code_type = "CKD (ICD-10)"
        elif code in AKI_ICD10_CODES:
            code_type = "AKI (ICD-10)"
        elif code in CKD_ICD9_CODES:
            code_type = "CKD (ICD-9)"
        elif code in AKI_ICD9_CODES:
            code_type = "AKI (ICD-9)"
        else:
            code_type = "기타"
        
        print(f"  {code}: {count:,}건 ({code_type})")
    
    # CKD vs AKI 분류 통계
    print(f"\n📈 진단 유형별 분포:")
    ckd_diagnoses = kidney_diagnoses[
        kidney_diagnoses['icd_code'].str.startswith(('N18', '585'), na=False)
    ]
    aki_diagnoses = kidney_diagnoses[
        kidney_diagnoses['icd_code'].str.startswith(('N17', '584'), na=False)
    ]
    
    print(f"  🔵 CKD 관련 진단: {len(ckd_diagnoses):,}건")
    print(f"  🔴 AKI 관련 진단: {len(aki_diagnoses):,}건")
    print(f"  ⚪ 기타 신장질환: {len(kidney_diagnoses) - len(ckd_diagnoses) - len(aki_diagnoses):,}건")


    # CKD 스테이지별 환자 분포
    print(f"\n📊 CKD Stage별 분포 (ICD 코드 기준):")
    stage_map = {
        'N181': 'Stage 1 (ICD-10)',
        'N182': 'Stage 2 (ICD-10)',
        'N183': 'Stage 3 (ICD-10)',
        'N184': 'Stage 4 (ICD-10)',
        'N185': 'Stage 5 (ICD-10)',
        '5851': 'Stage 1 (ICD-9)',
        '5852': 'Stage 2 (ICD-9)',
        '5853': 'Stage 3 (ICD-9)',
        '5854': 'Stage 4 (ICD-9)',
        '5855': 'Stage 5 (ICD-9)',
    }

    for code, label in stage_map.items():
        stage_patients = kidney_diagnoses[kidney_diagnoses['icd_code'] == code]['subject_id'].nunique()
        stage_diagnoses = (kidney_diagnoses['icd_code'] == code).sum()
        if stage_patients > 0:
            print(f"  {label}: {stage_patients:,}명 ({stage_diagnoses:,}건)")

    # CKD Stage 통합 분포 (ICD-9 + ICD-10)
    print(f"\n📊 CKD Stage별 통합 분포 (ICD-9 + ICD-10):")
    stage_groups = {
        "Stage 1": ['N181', '5851'],
        "Stage 2": ['N182', '5852'],
        "Stage 3": ['N183', '5853'],
        "Stage 4": ['N184', '5854'],
        "Stage 5": ['N185', '5855'],
    }
    for stage, codes in stage_groups.items():
        stage_patients = kidney_diagnoses[kidney_diagnoses['icd_code'].isin(codes)]['subject_id'].nunique()
        stage_diagnoses = kidney_diagnoses[kidney_diagnoses['icd_code'].isin(codes)].shape[0]
        print(f"  {stage}: {stage_patients:,}명 ({stage_diagnoses:,}건)")
    
    # diagnoses 테이블에서 CKD 코드를 범주별로 분해해 환자 수 비교
    dx = kidney_diagnoses.copy()
    dx["icd_code"] = dx["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)

    # 카테고리 정의
    is_ckd10_any = dx["icd_code"].str.startswith("N18")
    is_ckd9_any  = dx["icd_code"].str.startswith("585")

    # 스테이지 명시 (Stage 1~5)
    is_stage10 = dx["icd_code"].str.match(r"^N18[1-5]")
    is_stage9  = dx["icd_code"].str.match(r"^585[1-5]")

    # 미상세 CKD (unspecified)
    is_unspec10 = (dx["icd_code"] == "N18") | dx["icd_code"].str.match(r"^N189")
    is_unspec9  = (dx["icd_code"] == "585") | dx["icd_code"].str.match(r"^5859")

    # ESRD / Dialysis
    is_esrd10 = dx["icd_code"].str.match(r"^N18(0|6)")
    is_esrd9  = dx["icd_code"].str.match(r"^5856")

    # 환자 유니온(중복 제거) 카운트
    patients_stage   = dx.loc[is_stage10 | is_stage9, "subject_id"].nunique()
    patients_unspec  = dx.loc[is_unspec10 | is_unspec9, "subject_id"].nunique()
    patients_esrd    = dx.loc[is_esrd10  | is_esrd9,  "subject_id"].nunique()
    patients_ckd_any = dx.loc[is_ckd10_any | is_ckd9_any, "subject_id"].nunique()

    print("\n📊 CKD 환자 카테고리별(환자 유니온) 비교")
    print(f"  • CKD 전체(모든 N18*/585*): {patients_ckd_any:,}명")
    print(f"    ├─ 스테이지 명시(N181–N185, 5851–5855): {patients_stage:,}명")
    print(f"    ├─ 미상세(N18/N189, 585/5859):          {patients_unspec:,}명")
    print(f"    └─ ESRD/투석(N180/N186, 5856):          {patients_esrd:,}명")
    # 참고: 범주 간 중복 존재 → 단순 합산 ≠ CKD 전체
    # 겹침을 보고 싶다면 환자 레벨로 멀티라벨 집합을 만들어 교집합도 출력 가능
  
    
    # ICD 버전별 분포
    if 'icd_version' in kidney_diagnoses.columns:
        print(f"\n📊 ICD 버전별 분포:")
        version_dist = kidney_diagnoses['icd_version'].value_counts()
        for version, count in version_dist.items():
            print(f"  ICD-{version}: {count:,}건")
    
    return cohort_data

def main():
    """메인 실행 함수"""
    
    # 명령행 인수 파싱 (도움말 처리를 위해 가장 먼저 실행)
    args = parse_args()
    
    print("🏥 MIMIC-IV 신장질환 환자 코호트 추출 시작")
    print("=" * 60)
    
    print(f"🎯 **1단계 목표**: ICD 진단 코드 기반 신장질환 환자 풀 생성")
    print(f"📋 실행 설정:")
    print(f"  📂 MIMIC-IV 경로: {args.mimic_path}")
    print(f"  📂 출력 경로: {args.output_path}")
    print(f"  🔄 강제 재추출: {args.force_reload}")
    print(f"  🧪 테스트 모드: {args.test_mode}")
    
    print(f"\n📊 ICD 코드 정의:")
    print(f"  - 🔵 CKD (만성신장질환) ICD-10: {len(CKD_ICD10_CODES)}개 (N18.x 계열)")
    print(f"  - 🔴 AKI (급성신장손상) ICD-10: {len(AKI_ICD10_CODES)}개 (N17.x 계열)")
    print(f"  - 🔵 CKD ICD-9: {len(CKD_ICD9_CODES)}개 (585.x 계열)")
    print(f"  - 🔴 AKI ICD-9: {len(AKI_ICD9_CODES)}개 (584.x 계열)")
    print(f"  - 📋 총 {len(ALL_KIDNEY_CODES)}개 코드로 진단 기록 필터링")
    print(f"  - ℹ️  주의: Lab 데이터(eGFR 측정값)와 무관하게 진단 코드만 사용")
    
    # 신장질환 코호트 추출 실행
    try:
        result = extract_kidney_cohort(
            mimic_path=args.mimic_path,
            output_path=args.output_path,
            force_reload=args.force_reload,
            test_mode=args.test_mode
        )
        
        if result:
            print(f"\n🎉 성공적으로 완료!")
            print(f"📁 저장 위치: {Path(args.output_path) / 'kidney_cohort.pkl'}")
            print(f"🎯 다음 단계: extract_with_egfr_full_fast.py로 멀티모달 데이터 추출")
            print(f"   (2단계에서 실제 eGFR 측정값 추출 및 유효성 검증 수행)")
            
        else:
            print(f"\n❌ 코호트 추출 실패")
            return 1
            
    except Exception as e:
        print(f"\n💥 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
