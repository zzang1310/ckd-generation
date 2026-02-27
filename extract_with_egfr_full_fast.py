#!/usr/bin/env python3
"""
🚀 eGFR itemid 추가 후 FULL 멀티모달 데이터 추출 (빠른 버전)

eGFR 측정값과 AKI 진단이 포함된 완전한 전체 데이터셋을 추출합니다.
"""

import sys
import os
import pickle
import json
import re
from pathlib import Path
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 공통 피처 정의 import
try:
    from config.feature_definitions import (
        LAB_FEATURES, VITAL_FEATURES,
        get_all_lab_itemids, get_all_vital_itemids, get_egfr_itemids
    )
except ImportError as e:
    raise ImportError(
        "config.feature_definitions import에 실패했습니다. "
        "프로젝트 루트에 `config/feature_definitions.py`가 존재해야 하며, "
        "작업 디렉토리를 리포지토리 루트(`/home/noh/CKD-LLM`)에서 실행하세요."
    ) from e

# MIMIC-IV 경로: 환경변수 MIMIC_PATH 우선, 없으면 후보 경로 순서대로 탐색
def _resolve_mimic_path():
    env = os.environ.get("MIMIC_PATH")
    if env and Path(env).exists():
        return Path(env)
    for candidate in ["D:/CKD-LLM/mimic-iv-3.1", "data/mimic-iv-3.1", "mimic-iv-3.1"]:
        p = Path(candidate)
        if p.exists() and (p / "hosp").is_dir():
            return p
    return Path("D:/CKD-LLM/mimic-iv-3.1")  # 기본값 (실행 시 존재 여부 확인)

MIMIC_PATH = _resolve_mimic_path()
HOSP_PATH = MIMIC_PATH / "hosp"
ICU_PATH = MIMIC_PATH / "icu"

# 저장 경로
PROCESSED_PATH = Path("processed_data/intermediate")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

def load_processed_data(data_name):
    """저장된 데이터 로딩"""
    file_path = PROCESSED_PATH / f"{data_name}.pkl"
    if file_path.exists():
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ {data_name} 로딩 실패: {e}")
            return None
    return None

def save_processed_data(data, data_name, description=""):
    """데이터 저장"""
    file_path = PROCESSED_PATH / f"{data_name}.pkl"
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        file_size = file_path.stat().st_size / (1024**2)  # MB
        print(f"✅ {data_name} 저장 완료 ({file_size:.1f}MB): {description}")
        return True
    except Exception as e:
        print(f"❌ {data_name} 저장 실패: {e}")
        return False

# 🔥 피처 정의는 config.feature_definitions에서 단일 소스로 관리
ALL_LAB_ITEMIDS = get_all_lab_itemids()
ALL_VITAL_ITEMIDS = get_all_vital_itemids()
egfr_itemids = get_egfr_itemids()

print("🔥 피처 정의 로딩 완료 (공통 설정 사용)")
print(f"📊 피처 현황:")
print(f"   🧪 Lab 피처: {len(ALL_LAB_ITEMIDS)}개 itemid")
print(f"   💓 Vital 피처: {len(ALL_VITAL_ITEMIDS)}개 itemid")

# eGFR itemid 확인
egfr_in_lab = [itemid for itemid in egfr_itemids if itemid in ALL_LAB_ITEMIDS]
print(f"   ✅ eGFR itemid 포함 상태: {len(egfr_in_lab)}/{len(egfr_itemids)}개")
print(f"       포함된 eGFR itemid: {egfr_in_lab}")

# ICD 코드 정의 (기존 유지)
CKD_ICD10_CODES = ['N18', 'N180', 'N181', 'N182', 'N183', 'N184', 'N185', 'N186']
AKI_ICD10_CODES = ['N17', 'N170', 'N171', 'N172', 'N178', 'N179']
CKD_ICD9_CODES = ['585', '5851', '5852', '5853', '5854', '5855', '5856', '5859']
AKI_ICD9_CODES = ['584', '5840', '5841', '5845', '5846', '5847', '5848', '5849']

ALL_KIDNEY_CODES = CKD_ICD10_CODES + AKI_ICD10_CODES + CKD_ICD9_CODES + AKI_ICD9_CODES

print(f"\n🏥 신장질환 ICD 코드: {len(ALL_KIDNEY_CODES)}개")
print(f"   - AKI 코드: {len(AKI_ICD10_CODES + AKI_ICD9_CODES)}개")

def get_cached_row_count_fast(csv_path):
    """빠른 캐시 기반 행 수 조회 (계산 안함)"""
    cache_path = csv_path.parent / f"{csv_path.stem}_rows.cache"
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            return cache_data.get('row_count', 0)
        except:
            pass
    return None  # 캐시 없으면 None 리턴

def extract_table_data(table_name, file_path, patients, filter_itemids=None, max_chunks=None, show_progress_every=10):
    """⚡ 빠른 테이블 데이터 추출 (캐시 기반 퍼센트 표시)"""
    try:
        chunk_size = 100000
        chunk_count = 0
        data_list = []
        total_records = 0
        
        print(f"  📁 파일 정보 확인 중...")
        file_size_mb = file_path.stat().st_size / (1024**2)
        print(f"  📊 파일 크기: {file_size_mb:.1f}MB")
        
        # ⚡ 캐시 기반 빠른 진행률 계산
        cached_rows = get_cached_row_count_fast(file_path)
        if cached_rows:
            actual_total_chunks = (cached_rows // chunk_size) + (1 if cached_rows % chunk_size > 0 else 0)
            print(f"  💾 캐시에서 총 행 수: {cached_rows:,}행 → 예상 청크: {actual_total_chunks}개")
        else:
            actual_total_chunks = None
            print(f"  📏 캐시 없음 - 퍼센트 없이 진행")
        
        # ⚡ 간단한 CSV 읽기 (최소한의 설정)
        csv_kwargs = {
            'chunksize': chunk_size,
            'engine': 'c',
            'low_memory': False
        }
        
        try:
            csv_reader = pd.read_csv(file_path, **csv_kwargs)
        except Exception as e:
            print(f"  ⚠️ C 엔진 실패, Python 엔진으로 fallback: {e}")
            csv_kwargs['engine'] = 'python'
            csv_reader = pd.read_csv(file_path, **csv_kwargs)
        
        for chunk in csv_reader:
            chunk_count += 1
            
            # ⚡ 캐시 기반 진행률 표시
            if chunk_count % show_progress_every == 0 or chunk_count <= 10:
                if actual_total_chunks:
                    progress_pct = (chunk_count / actual_total_chunks * 100)
                    print(f"  📦 {table_name} 청크 {chunk_count}/{actual_total_chunks} 처리 중... ({progress_pct:.1f}% 진행, 누적: {total_records:,}건)")
                else:
                    print(f"  📦 {table_name} 청크 {chunk_count} 처리 중... (누적: {total_records:,}건)")
            
            # ⚡ 원본 순서: 환자 필터링 → itemid 필터링
            patient_chunk = chunk[chunk['subject_id'].isin(patients)]
            
            if not patient_chunk.empty:
                # ItemID 필터링 (있는 경우)
                if filter_itemids and 'itemid' in patient_chunk.columns:
                    filtered_chunk = patient_chunk[patient_chunk['itemid'].isin(filter_itemids)]
                else:
                    filtered_chunk = patient_chunk
                
                if not filtered_chunk.empty:
                    # ⚡ 간단한 시간 정보 표준화
                    time_cols = {
                        'labevents': 'charttime',
                        'chartevents': 'charttime',
                        'prescriptions': 'starttime',
                        'emar': 'charttime',
                        'inputevents': 'starttime',
                        'outputevents': 'charttime',
                        'procedures_icd': 'chartdate',
                    }
                    time_col = time_cols.get(table_name, 'charttime')
                    
                    if time_col in filtered_chunk.columns:
                        filtered_chunk = filtered_chunk.copy()
                        filtered_chunk['event_time'] = pd.to_datetime(filtered_chunk[time_col])
                    
                    data_list.append(filtered_chunk)
                    total_records += len(filtered_chunk)
            
            # 청크 제한 확인 (테스트용)
            if max_chunks and chunk_count >= max_chunks:
                print(f"  ⏰ {max_chunks}개 청크 제한으로 중단")
                break
        
        # ⚡ 빠른 데이터 통합
        if data_list:
            print(f"  🔗 {len(data_list)}개 청크 데이터 통합 중...")
            combined_data = pd.concat(data_list, ignore_index=True)
            print(f"  ✅ {table_name} 추출 완료: {len(combined_data):,}건 (총 {chunk_count}개 청크 처리)")
            return combined_data
        else:
            print(f"  ❌ {table_name}: 추출된 데이터 없음")
            return None
            
    except Exception as e:
        print(f"  ❌ {table_name} 처리 중 오류: {e}")
        return None


def attach_d_items_label_to_input_events(input_df: pd.DataFrame, icu_path: Path) -> pd.DataFrame:
    """Input Events에 d_items의 label(약물/투여명) 컬럼을 left-join으로 추가.
    프롬프트 빌드 시 RX_KEYWORDS 기반 약물 매칭을 위해 필요.
    d_items 없거나 조인 실패 시 원본 그대로 반환.
    """
    if input_df is None or input_df.empty:
        return input_df
    d_items_path = icu_path / "d_items.csv"
    if not d_items_path.exists():
        d_items_path = icu_path / "d_items.csv.gz"
    if not d_items_path.exists():
        print(f"  ⚠️ d_items 파일 없음 ({icu_path}) — input_events에 label 컬럼 미추가 (키워드 매칭은 EMAR/Prescriptions만 사용)")
        return input_df
    try:
        usecols = ["itemid", "label"]
        d_items = pd.read_csv(d_items_path, usecols=usecols)
        d_items = d_items.drop_duplicates(subset=["itemid"])  # itemid당 하나의 label
        d_items["itemid"] = d_items["itemid"].astype(input_df["itemid"].dtype)  # merge 호환
        out = input_df.merge(d_items, on="itemid", how="left")
        n_with_label = out["label"].notna().sum()
        print(f"  ✅ input_events + d_items 조인 완료: label 컬럼 추가, {n_with_label:,}/{len(out):,}건에 label 존재")
        return out
    except Exception as e:
        print(f"  ⚠️ d_items 조인 실패: {e} — input_events 원본 유지")
        return input_df


def main():
    print("🚀 FULL eGFR 포함 멀티모달 신장질환 데이터 추출 시작 (⚡ 빠른 버전)")
    print("=" * 70)
    print("🎯 **2단계 목표**: 1단계 환자들의 실제 Lab/Chart/EMAR/InputEvents/Procedures 데이터 추출")
    print("⚡ 주의: 속도 최적화 버전 - 약 30-40분 소요 예상")
    print("📋 이 단계에서 실제 eGFR 측정값 개수 확인 (NaN 제외)")
    print("=" * 70)
    
    # 1. 기본 코호트 로딩
    print("\n1. 기본 코호트 로딩...")
    base_cohort = load_processed_data('kidney_cohort')
    if base_cohort is None:
        print("❌ 기본 코호트가 없습니다. 먼저 기본 추출을 실행하세요.")
        return
    
    kidney_patients = set(base_cohort['patient_ids'])
    print(f"👥 대상 환자 수: {len(kidney_patients):,}명")
    
    # 2. 전체 모드 설정
    test_mode = False
    max_chunks_per_table = None
    print("⚡ 빠른 전체 추출 모드")
    
    # 3. 멀티모달 데이터 추출
    multimodal_data = {
        'patient_ids': kidney_patients,
        'base_info': {
            'diagnoses_icd': base_cohort.get('diagnoses', None),
            'patients_info': base_cohort.get('patients_info', None),
            'admissions': base_cohort.get('admissions', None)
        },
        'clinical_data': {},
        'extraction_meta': {
            'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_patients': len(kidney_patients),
            'test_mode': test_mode,
            'max_chunks_per_table': max_chunks_per_table,
            'egfr_itemids_included': egfr_itemids,
            'total_lab_itemids': len(ALL_LAB_ITEMIDS),
            'total_vital_itemids': len(ALL_VITAL_ITEMIDS),
            'full_extraction': True,
            'fast_mode': True
        }
    }
    
    extraction_summary = {}
    
    # 3-1. Lab Events (eGFR 포함)
    print(f"\n🧪 Lab Events 전체 추출 시작 (eGFR 포함)...")
    print(f"   📋 필터링 대상 itemid: {len(ALL_LAB_ITEMIDS)}개")
    lab_data = extract_table_data(
        table_name='labevents',
        file_path=HOSP_PATH / 'labevents.csv',
        patients=kidney_patients,
        filter_itemids=ALL_LAB_ITEMIDS,
        max_chunks=max_chunks_per_table,
        show_progress_every=10
    )
    multimodal_data['clinical_data']['lab_events'] = lab_data
    extraction_summary['lab_events'] = len(lab_data) if lab_data is not None else 0
    
    # 3-2. Chart Events
    print(f"\n💓 Chart Events 전체 추출 시작...")
    print(f"   📋 필터링 대상 itemid: {len(ALL_VITAL_ITEMIDS)}개")
    chart_data = extract_table_data(
        table_name='chartevents',
        file_path=ICU_PATH / 'chartevents.csv',
        patients=kidney_patients,
        filter_itemids=ALL_VITAL_ITEMIDS,
        max_chunks=max_chunks_per_table,
        show_progress_every=10
    )
    multimodal_data['clinical_data']['chart_events'] = chart_data
    extraction_summary['chart_events'] = len(chart_data) if chart_data is not None else 0
    
    # 3-3. Prescriptions
    print(f"\n💊 Prescriptions 전체 추출 시작...")
    prescription_data = extract_table_data(
        table_name='prescriptions',
        file_path=HOSP_PATH / 'prescriptions.csv',
        patients=kidney_patients,
        filter_itemids=None,
        max_chunks=max_chunks_per_table,
        show_progress_every=10
    )
    multimodal_data['clinical_data']['prescriptions'] = prescription_data
    extraction_summary['prescriptions'] = len(prescription_data) if prescription_data is not None else 0
    
    # 3-3b. EMAR - 실제 투약 기록
    print(f"\n💉 EMAR (Electronic Medication Administration Record) 전체 추출 시작...")
    emar_data = extract_table_data(
        table_name='emar',
        file_path=HOSP_PATH / 'emar.csv',
        patients=kidney_patients,
        filter_itemids=None,
        max_chunks=max_chunks_per_table,
        show_progress_every=10
    )
    multimodal_data['clinical_data']['emar'] = emar_data
    extraction_summary['emar'] = len(emar_data) if emar_data is not None else 0
    
    # 3-4. Input Events (약물 투여 정보만 - EMAR 보완용)
    print(f"\n💊 Input Events 약물 투여 기록 추출 시작...")
    print(f"   📋 목적: EMAR 보완용 중요 약물 투여 정보")
    input_data = extract_table_data(
        table_name='inputevents',
        file_path=ICU_PATH / 'inputevents.csv',
        patients=kidney_patients,
        filter_itemids=None,
        max_chunks=max_chunks_per_table,
        show_progress_every=10
    )
    # d_items와 조인하여 label(약물/투여명) 컬럼 추가 — RX_KEYWORDS 기반 약물 매칭용
    if input_data is not None:
        input_data = attach_d_items_label_to_input_events(input_data, ICU_PATH)
    multimodal_data['clinical_data']['input_events'] = input_data
    extraction_summary['input_events'] = len(input_data) if input_data is not None else 0
    
    # 3-5. Output Events (일상생활 모델에서 불필요 - 추출 생략)
    print(f"\n🚰 Output Events - 일상생활 모델용도로 추출 생략 (ICU 배액 데이터)")
    # output_data = extract_table_data(
    #     table_name='outputevents',
    #     file_path=ICU_PATH / 'outputevents.csv',
    #     patients=kidney_patients,
    #     filter_itemids=None,
    #     max_chunks=max_chunks_per_table,
    #     show_progress_every=10
    # )
    multimodal_data['clinical_data']['output_events'] = None
    extraction_summary['output_events'] = 0
    
    # 3-6. Procedures
    print(f"\n🏥 Procedures 전체 추출 시작...")
    procedure_data = extract_table_data(
        table_name='procedures_icd',
        file_path=HOSP_PATH / 'procedures_icd.csv',
        patients=kidney_patients,
        filter_itemids=None,
        max_chunks=max_chunks_per_table,
        show_progress_every=10
    )
    multimodal_data['clinical_data']['procedures'] = procedure_data
    extraction_summary['procedures'] = len(procedure_data) if procedure_data is not None else 0
    
    # 4. 최종 요약
    print(f"\n📊 전체 추출 완료 요약:")
    print(f"   - 대상 환자: {len(kidney_patients):,}명")
    print(f"   - Lab Events: {extraction_summary.get('lab_events', 0):,}건")
    print(f"   - Chart Events: {extraction_summary.get('chart_events', 0):,}건")
    print(f"   - Prescriptions: {extraction_summary.get('prescriptions', 0):,}건")
    print(f"   - EMAR: {extraction_summary.get('emar', 0):,}건")
    print(f"   - Input Events: {extraction_summary.get('input_events', 0):,}건")
    print(f"   - Output Events: {extraction_summary.get('output_events', 0):,}건")
    print(f"   - Procedures: {extraction_summary.get('procedures', 0):,}건")
    
    # 환자 분류 분석 (진단 코드 기반)
    if 'diagnoses' in base_cohort and base_cohort['diagnoses'] is not None:
        diagnoses_df = base_cohort['diagnoses']
        
        # CKD 환자 (만성 신장질환) - 정확한 매칭 + 접두어 매칭
        ckd_patients = set(diagnoses_df[
            diagnoses_df['icd_code'].isin(CKD_ICD10_CODES + CKD_ICD9_CODES) |
            diagnoses_df['icd_code'].str.startswith(('N18', '585'), na=False)
        ]['subject_id'].unique())
        
        # AKI 환자 (급성 신손상) - 정확한 매칭 + 접두어 매칭
        aki_patients = set(diagnoses_df[
            diagnoses_df['icd_code'].isin(AKI_ICD10_CODES + AKI_ICD9_CODES) |
            diagnoses_df['icd_code'].str.startswith(('N17', '584'), na=False)
        ]['subject_id'].unique())
        
        # 중복 환자 (CKD + AKI 모두 진단받은 환자)
        overlap_patients = ckd_patients & aki_patients
        
        # 기타 신장질환 환자 (CKD도 AKI도 아닌 환자)
        other_patients = kidney_patients - (ckd_patients | aki_patients)
        
        print(f"   - 👥 환자 분류:")
        print(f"     • CKD 환자: {len(ckd_patients):,}명")
        print(f"     • AKI 환자: {len(aki_patients):,}명")
        print(f"     • CKD+AKI 중복: {len(overlap_patients):,}명")
        print(f"     • 순수 CKD: {len(ckd_patients - aki_patients):,}명")
        print(f"     • 순수 AKI: {len(aki_patients - ckd_patients):,}명")
        print(f"     • 기타 신장질환: {len(other_patients):,}명")
        print(f"     • 검증: {len(ckd_patients - aki_patients) + len(aki_patients - ckd_patients) + len(overlap_patients) + len(other_patients):,}명 = {len(kidney_patients):,}명")
    
    # eGFR 요약 (유효한 값만) - 핵심 검증
    if lab_data is not None:
        egfr_total = int(lab_data['itemid'].isin(egfr_itemids).sum())
        egfr_valid = int((lab_data['itemid'].isin(egfr_itemids) & lab_data['valuenum'].notna()).sum())
        egfr_nan_count = egfr_total - egfr_valid
        print(f"   - 🧪 eGFR 관련 분석:")
        print(f"     • 전체 eGFR 레코드: {egfr_total:,}건")
        print(f"     • 유효한 eGFR 측정값: {egfr_valid:,}건")
        print(f"     • NaN/무효 값: {egfr_nan_count:,}건 ({egfr_nan_count/egfr_total*100:.1f}%)")
        
        # 환자별 eGFR 보유 현황
        egfr_patients = len(lab_data[lab_data['itemid'].isin(egfr_itemids) & lab_data['valuenum'].notna()]['subject_id'].unique())
        print(f"     • eGFR 측정값 보유 환자: {egfr_patients:,}명")
    
    multimodal_data['extraction_summary'] = extraction_summary
    
    # 5. 데이터 저장
    data_name = 'full_multimodal_kidney_cohort_with_egfr_complete_fast'
    description = f"⚡ 빠른 추출: eGFR 포함 멀티모달 신장질환 데이터 ({len(kidney_patients):,}명)"
    
    print(f"\n💾 데이터 저장 중...")
    if save_processed_data(multimodal_data, data_name, description):
        print(f"\n✅ ⚡ 2단계 완료: {data_name}.pkl")
        print(f"🎯 다음 단계: prepare_patient_selection.py로 고품질 환자 선별")
        print(f"   (3단계에서 eGFR+AKI 합집합 환자 우선순위 정렬 수행)")
    else:
        print(f"\n❌ 저장 실패")

if __name__ == "__main__":
    main()
