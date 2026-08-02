# --- START OF FILE clean_dataset.py ---

import pandas as pd
import numpy as np
import warnings

# 경고 메시지 숨김 처리
warnings.filterwarnings('ignore')

# 1. 실제 데이터 로드
file_path = 'uci-secom.csv'
try:
    df = pd.read_csv(file_path)
    print(f"[최초 데이터 로드] 총 레코드 수: {df.shape[0]}건, 변수 수: {df.shape[1]}개\n")
except FileNotFoundError:
    print(f"Error: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

df_clean = df.copy()

# ==========================================
# 1. 유일성 (Uniqueness) 검증
# ==========================================
# 가이드라인: 동일한 시간(Time)에 수집된 중복 기록은 로깅 오류이므로 최초 기록만 남기고 제거
print("[1. 유일성 평가]")
if 'Time' in df_clean.columns:
    duplicates = df_clean.duplicated(subset=['Time'], keep=False)
    print(f" - 타임스탬프(Time) 중복 기록 데이터 수: {duplicates.sum()}건 발견")
    df_clean = df_clean.drop_duplicates(subset=['Time'], keep='first')
print("중복 데이터 제거 완료\n")


# ==========================================
# 2. 완전성 (Completeness) 검증
# ==========================================
# 가이드라인: 결측률 50% 이상인 무의미한 센서 변수(컬럼) 제거, 이후 결측치 20% 이상인 불량 관측치(행) 제거
print("[2. 완전성 평가]")
# 컬럼(변수) 기준 검증
missing_col_ratios = df_clean.isnull().mean()
high_missing_cols = missing_col_ratios[missing_col_ratios > 0.5].index
print(f" - 결측률 50% 이상인 센서 변수(컬럼) 수: {len(high_missing_cols)}개 발견")
df_clean = df_clean.drop(columns=high_missing_cols)

# 행(관측치) 기준 검증
missing_row_ratios = df_clean.isnull().mean(axis=1)
high_missing_rows = missing_row_ratios[missing_row_ratios > 0.2]
print(f" - 잔여 변수 중 결측률 20% 이상인 불량 관측치(행) 수: {len(high_missing_rows)}건 발견")
df_clean = df_clean.drop(index=high_missing_rows.index)
print("결측치 과다 변수 및 관측치 제거 완료\n")


# ==========================================
# 3. 유효성 (Validity) 검증
# ==========================================
# 가이드라인: 타겟 라벨(Pass/Fail)이 -1 또는 1인지 확인하고, Time 변수가 정상 날짜 포맷인지 검증
print("[3. 유효성 평가]")
if 'Pass/Fail' in df_clean.columns:
    invalid_labels = ~df_clean['Pass/Fail'].isin([-1, 1])
    print(f" - 유효하지 않은 타겟 라벨(Pass/Fail) 수: {invalid_labels.sum()}건 발견")
    df_clean = df_clean[~invalid_labels]

if 'Time' in df_clean.columns:
    parsed_time = pd.to_datetime(df_clean['Time'], errors='coerce')
    invalid_time = parsed_time.isna()
    print(f" - 유효하지 않은 타임스탬프 형식 수: {invalid_time.sum()}건 발견")
    df_clean = df_clean[~invalid_time]
print("유효성 위배 데이터 제거 완료\n")


# ==========================================
# 4. 일관성 (Consistency) 검증
# ==========================================
# 가이드라인: 장비 미가동이나 센서 단선 등으로 인해 모든 행이 동일한 값(분산=0)을 가지는 무의미한 센서 제거
print("[4. 일관성 평가]")
# 숫자형 변수들만 선택
num_cols = df_clean.select_dtypes(include=[np.number]).columns
# 고유값이 1개 이하(분산이 0)인 컬럼 추출
zero_variance_cols =[col for col in num_cols if df_clean[col].nunique() <= 1]
print(f" - 분산이 0(단일 값)인 무의미한 센서 변수 수: {len(zero_variance_cols)}개 발견")

df_clean = df_clean.drop(columns=zero_variance_cols)
print("일관성 없는(변화 없는) 변수 제거 완료\n")


# ==========================================
# 5. 정확성 (Accuracy) 검증
# ==========================================
# 가이드라인: Z-score 기준 5 표준편차를 벗어나는 극단적 이상치(센서 노이즈)를 NaN으로 치환 후, 중앙값으로 정밀 보간
print("[5. 정확성 평가]")
# 타겟 변수 제외하고 센서 변수만 선택
sensor_cols = df_clean.select_dtypes(include=[np.number]).columns.difference(['Pass/Fail'])

outlier_count = 0
for col in sensor_cols:
    mean, std = df_clean[col].mean(), df_clean[col].std()
    if std > 0:
        z_scores = np.abs((df_clean[col] - mean) / std)
        outliers = z_scores > 5
        outlier_count += outliers.sum()
        # 발견된 이상치를 NaN으로 치환
        df_clean.loc[outliers, col] = np.nan

print(f" - Z-score > 5 인 극단적 이상치(센서 노이즈): 총 {outlier_count}건 발견 및 제거")

# 최종적으로 남아있는 결측치(원래 있던 결측치 + 이상치 제거된 자리)를 변수별 중앙값(Median)으로 보간
df_clean[sensor_cols] = df_clean[sensor_cols].fillna(df_clean[sensor_cols].median())
print("극단적 이상치 정제 및 잔여 결측치 보간(Imputation) 완료\n")


# ==========================================
# [최종 결과] 정제된 고품질 데이터셋 저장
# ==========================================
print("=" * 60)
print("[최종 확보된 고품질 데이터셋 (Golden Dataset)]")
print(f" - 최초 원본 데이터: {df.shape[0]}건 (관측치), {df.shape[1]}개 (변수)")
print(f" - 최종 확보 데이터: {df_clean.shape[0]}건 (관측치), {df_clean.shape[1]}개 (변수)")
print("=" * 60)

# 정제된 데이터를 새로운 CSV 파일로 저장
df_clean.to_csv('uci-secom_clean.csv', index=False)
print("파일이 'uci-secom_clean.csv'로 성공적으로 저장되었습니다.")

# --- END OF FILE clean_dataset.py ---