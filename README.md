# CKD-LLM

MIMIC-IV 기반 신장질환(eGFR/AKI) LLM 학습 데이터 생성 파이프라인.

## 빠른 시작

**파이프라인 실행 순서 및 명령어** → [config/README.md](config/README.md) 5절 참고

1. 코호트 추출 → 2. 멀티모달 pkl → 3. 환자 선별 → 4. 프롬프트 생성

## 사전 준비

- MIMIC-IV v3.1 (PhysioNet 인증 필요)
- Python 3.9+, pandas, numpy

## 환경 변수

- `MIMIC_PATH`: MIMIC-IV 데이터 루트 경로

## 상세 문서

- [config/README.md](config/README.md) — 설정·파이프라인 상세
