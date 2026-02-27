import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# EGFR_ITEMIDS = [53180, 53161, 50920, 52026, 51770]  # 모든 eGFR 측정 방법 포함
EGFR_ITEMIDS = [53161]  # 1가지만 사용
AKI_ICD_PREFIXES = [
    # ICD-10
    "N17",
    # ICD-9
    "584",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare patient selection and summary (EGFR, AKI diagnosis)")
    p.add_argument("--processed-data-path", required=True, help="processed_data 디렉토리 경로")
    p.add_argument("--data-file", required=True, help="complete_data 피클 파일명")
    p.add_argument("--output", default="processed_data/patient_selection/patient_selection.pkl", help="저장 경로(pkl)")
    return p.parse_args()


def load_complete_data(path: Path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_numeric_from_value(s: str):
    """텍스트 값에서 숫자를 추출합니다 (>60, ≥90 등에서)"""
    if pd.isna(s): 
        return None
    _num_re = re.compile(r'[-+]?\d+(\.\d+)?')
    m = _num_re.search(str(s))
    return float(m.group(0)) if m else None


def compute_egfr_counts(lab_df: pd.DataFrame) -> Tuple[Dict[int, int], List[int], int]:
    """개선된 eGFR 카운팅: valuenum + value(텍스트)에서 숫자 파싱"""
    df = lab_df[lab_df["itemid"].isin(EGFR_ITEMIDS)].copy()
    
    # value 컬럼이 없으면 None으로 초기화
    if "value" not in df.columns:
        df["value"] = None
    
    # valuenum이 NaN이지만 value에 텍스트가 있는 경우 숫자 파싱
    need_parsing = df["valuenum"].isna() & df["value"].notna()
    if need_parsing.any():
        parsed = df.loc[need_parsing, "value"].apply(parse_numeric_from_value)
        valid_parsed = need_parsing & parsed.notna()
        df.loc[valid_parsed, "valuenum"] = parsed[valid_parsed].astype(float)
    
    # 최종적으로 valuenum이 있고 합리적 범위인 것만 사용
    df = df[(df["valuenum"].notna()) & (df["valuenum"] >= 0) & (df["valuenum"] <= 250)]
    
    # 환자별 측정 횟수 집계
    counts: Dict[int, int] = {}
    for subject_id, g in df.groupby("subject_id"):
        counts[int(subject_id)] = int(len(g))
    patients = list(counts.keys())
    total_measurements = int(len(df))
    return counts, patients, total_measurements


def code_is_aki(icd_code: str) -> bool:
    if not icd_code:
        return False
    code = icd_code.strip()
    return any(code.startswith(prefix) for prefix in AKI_ICD_PREFIXES)


def compute_aki_diag_counts(diag_df: pd.DataFrame) -> Tuple[Dict[int, int], List[int], int]:
    if diag_df is None or len(diag_df) == 0:
        return {}, [], 0
    diag_df = diag_df.copy()
    diag_df["icd_code"] = diag_df["icd_code"].astype(str)
    aki_rows = diag_df[diag_df["icd_code"].apply(code_is_aki)]
    counts: Dict[int, int] = {}
    for subject_id, g in aki_rows.groupby("subject_id"):
        counts[int(subject_id)] = int(len(g))
    patients = list(counts.keys())
    total_diagnoses = int(len(aki_rows))
    return counts, patients, total_diagnoses


def build_recommended_union(egfr_counts: Dict[int, int], aki_counts: Dict[int, int]) -> List[int]:
    # 전수 리스트(상위 N 아님). 정렬은 일관성을 위해 egfr_count desc, aki_count desc, subject_id asc
    union_ids = set(egfr_counts.keys()) | set(aki_counts.keys())
    def sort_key(pid: int):
        return (
            -int(egfr_counts.get(pid, 0)),
            -int(aki_counts.get(pid, 0)),
            int(pid),
        )
    return sorted(union_ids, key=sort_key)


def main() -> None:
    print("📊 환자 선별 및 우선순위 정렬 시작")
    print("=" * 50)
    print("🎯 **3단계 목표**: eGFR 또는 AKI (합집합) 환자 우선순위 정렬")
    print("📋 선별 기준:")
    print("   • eGFR 측정값 보유 환자 (유효한 valuenum)")
    print("   • AKI 진단 보유 환자 (N17.x, 584.x 계열)")
    print("   • 합집합 방식: 둘 중 하나만 있어도 포함")
    print("   • 우선순위: eGFR수 → AKI수 → 환자ID 순")
    print()
    
    args = parse_args()
    processed_path = Path(args.processed_data_path)
    data_path = processed_path / args.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    complete_data = load_complete_data(data_path)
    clinical_data = complete_data["clinical_data"]
    base_info = complete_data.get("base_info", {})

    lab_df: pd.DataFrame = clinical_data.get("lab_events")
    diag_df: pd.DataFrame = base_info.get("diagnoses_icd")

    # 디버그: 원본 eGFR 행수 확인
    egfr_all = lab_df[lab_df["itemid"].isin(EGFR_ITEMIDS)]
    egfr_valuenum_only = lab_df[(lab_df["itemid"].isin(EGFR_ITEMIDS)) & (pd.notna(lab_df["valuenum"]))]
    
    print(f"🔍 eGFR 데이터 분석:")
    print(f"   • 전체 eGFR 레코드: {len(egfr_all):,}건 (NaN 포함)")
    print(f"   • 유효한 eGFR 값: {len(egfr_valuenum_only):,}건 (NaN 제외)")
    nan_ratio = (len(egfr_all) - len(egfr_valuenum_only)) / len(egfr_all) * 100 if len(egfr_all) > 0 else 0
    print(f"   • NaN 비율: {nan_ratio:.1f}%")
    
    # 요약 계산 (개선된 파싱 포함)
    egfr_counts, egfr_patients, egfr_total = compute_egfr_counts(lab_df)
    aki_counts, aki_patients, aki_total = compute_aki_diag_counts(diag_df)

    # 요약 출력
    print(f"\n📈 환자별 집계 결과:")
    print(f"   🧪 eGFR 보유 환자: {len(egfr_patients):,}명 (총 측정 {egfr_total:,}건)")
    print(f"   🏥 AKI 진단 보유 환자: {len(aki_patients):,}명 (총 진단 {aki_total:,}건)")

    # 전수 추천 리스트(교집합 아님, 점수 상위 아님)
    recommended_patients = build_recommended_union(egfr_counts, aki_counts)
    
    # 합집합 분석
    egfr_only = len(set(egfr_patients) - set(aki_patients))
    aki_only = len(set(aki_patients) - set(egfr_patients))
    both = len(set(egfr_patients) & set(aki_patients))
    
    print(f"\n🔗 합집합 구성 분석:")
    print(f"   • eGFR만 보유: {egfr_only:,}명")
    print(f"   • AKI만 보유: {aki_only:,}명")
    print(f"   • 둘 다 보유: {both:,}명")
    print(f"   • 총 합집합: {len(recommended_patients):,}명")
    print(f"   • 검증: {egfr_only + aki_only + both} = {len(recommended_patients)} ✅")

    # 저장
    selection = {
        "egfr_counts": egfr_counts,
        "aki_diagnosis_counts": aki_counts,
        "egfr_patients": egfr_patients,
        "aki_patients": aki_patients,
        "egfr_total_measurements": egfr_total,
        "aki_total_diagnoses": aki_total,
        # build_prompt_dataset.py가 기대하는 키를 그대로 제공하여 바로 사용 가능
        "recommended_patients": recommended_patients,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(selection, f)
    print(f"\n✅ 3단계 완료: {out_path}")
    print(f"📋 생성된 recommended_patients: {len(recommended_patients):,}명 우선순위 리스트")
    print(f"   🥇 1순위 환자: ID {recommended_patients[0]} (eGFR {egfr_counts.get(recommended_patients[0], 0)}회, AKI {aki_counts.get(recommended_patients[0], 0)}회)")
    print(f"   🥈 2순위 환자: ID {recommended_patients[1]} (eGFR {egfr_counts.get(recommended_patients[1], 0)}회, AKI {aki_counts.get(recommended_patients[1], 0)}회)")
    print(f"   🥉 3순위 환자: ID {recommended_patients[2]} (eGFR {egfr_counts.get(recommended_patients[2], 0)}회, AKI {aki_counts.get(recommended_patients[2], 0)}회)")
    print(f"🎯 다음 단계: build_prompt_dataset.py로 이 순서대로 프롬프트 생성")
    print(f"   (상위 N명 선택 시 recommended_patients[0:N] 순서로 처리)")


if __name__ == "__main__":
    main()


