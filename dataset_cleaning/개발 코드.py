import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def refine_secom_dataset(file_path, save_path):
    print("데이터 로딩 중...")
    # 데이터 로드 (첫 번째 컬럼: Time, 마지막 컬럼: Pass/Fail 가정)
    df = pd.read_csv(file_path)
    
    # 편의상 컬럼 식별: Time 변수 및 Target 변수 이름 찾기
    time_col = df.columns[0]
    target_col = df.columns[-1]
    
    # ---------------------------------------------------------
    # 1. [유일성(Uniqueness)] 확보
    # ---------------------------------------------------------
    print("1/5 유일성 확보: 중복 데이터 제거 중...")
    # 데이터 행 전체가 완전히 일치하는 중복 관측치 제거
    df = df.drop_duplicates(keep='first')
    
    # 타임스탬프 기준 중복 행 검사 및 제거 (같은 시간에 기록된 중복 로그 방지)
    df = df.drop_duplicates(subset=[time_col], keep='first')

    # ---------------------------------------------------------
    # 2. [유효성(Validity)] 확보
    # ---------------------------------------------------------
    print("2/5 유효성 확보: 데이터 형식 표준화 중...")
    # Time 컬럼을 표준 datetime 객체로 변환
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    # Target(라벨) 컬럼을 머신러닝에 적합한 0과 1로 변환 (기존 -1: 정상, 1: 불량)
    df[target_col] = df[target_col].replace({-1: 0})
    
    # 이후 처리를 위해 수치형 센서 변수 리스트만 별도 추출 (타겟 변수 제외)
    sensor_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in sensor_cols:
        sensor_cols.remove(target_col)

    # ---------------------------------------------------------
    # 3. [일관성(Consistency)] 확보
    # ---------------------------------------------------------
    print("3/5 일관성 확보: 무의미한 단일값(Zero-Variance) 변수 제거 중...")
    # 분산이 0인 (모든 관측치에서 값이 똑같은) 센서 컬럼 제거 
    nunique_counts = df[sensor_cols].nunique()
    zero_var_cols = nunique_counts[nunique_counts <= 1].index.tolist()
    
    df = df.drop(columns=zero_var_cols)
    sensor_cols =[col for col in sensor_cols if col not in zero_var_cols]

    # ---------------------------------------------------------
    # 4.[완전성(Completeness)] 확보
    # ---------------------------------------------------------
    print("4/5 완전성 확보: 결측치 제거 및 KNN 보간 중...")
    # 1) 결측치 비율이 40% 이상인 센서 변수 열 삭제
    missing_ratio = df[sensor_cols].isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio >= 0.4].index.tolist()
    
    df = df.drop(columns=high_missing_cols)
    sensor_cols = [col for col in sensor_cols if col not in high_missing_cols]
    
    # 2) KNN Imputer를 활용한 잔여 결측치 정밀 보간 (K=5)
    imputer = KNNImputer(n_neighbors=5)
    df[sensor_cols] = imputer.fit_transform(df[sensor_cols])

    # ---------------------------------------------------------
    # 5. [정확성(Accuracy)] 확보
    # ---------------------------------------------------------
    print("5/5 정확성 확보: 이상치 클리핑(Clipping) 적용 중...")
    # 결측치가 해결된 상태에서, IQR 기반 상/하한선으로 이상치 노이즈 완화
    for col in sensor_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 정상 데이터 판단 상/하한선
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 경계를 넘어서는 노이즈 값을 상/하한선 최대/최소값으로 깎아냄(Clipping)
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        
    print(f"데이터 정제 완료! (최종 데이터 형태: {df.shape})")
    # ---------------------------------------------------------
    # 6. CSV 파일로 저장
    # ---------------------------------------------------------
    if save_path:
        # index=False 옵션을 주어야 불필요한 행 번호(0, 1, 2...)가 새 컬럼으로 저장되지 않습니다.
        df.to_csv(save_path, index=False)
        print(f"정제된 데이터가 '{save_path}'에 저장되었습니다.")

    return df


file_path = 'uci-secom.csv'
save_path = 'uci-secom_cleaned.csv'
cleaned_dataset = refine_secom_dataset(file_path, save_path)
