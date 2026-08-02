# 라이브러리 임포트 및 데이터 로드
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("라이브러리 로드 완료")

# 데이터 로드 (EDA 단계와 동일한 원본 데이터)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)
print(f"원본 데이터 로드 완료: {df.shape}")

# 데이터 전처리 (식별자 제거, 인코딩, 데이터 누수 방지)
print("\n데이터 전처리 시작...")

def add_engineered_features(dataframe):
    engineered = dataframe.copy()
    engineered['Temperature difference [K]'] = engineered['Process temperature [K]'] - engineered['Air temperature [K]']
    engineered['Torque x Speed'] = engineered['Torque [Nm]'] * engineered['Rotational speed [rpm]']
    engineered['Wear / Speed'] = engineered['Tool wear [min]'] / (engineered['Rotational speed [rpm]'] + 1e-6)
    engineered['Torque / Speed'] = engineered['Torque [Nm]'] / (engineered['Rotational speed [rpm]'] + 1e-6)
    engineered['Wear x Torque'] = engineered['Tool wear [min]'] * engineered['Torque [Nm]']
    return engineered

# 1. 불필요한 식별자 컬럼 제거
df_processed = df.drop(columns=['UDI', 'Product ID'])

# 2. 범주형 변수(Type) One-Hot Encoding
df_processed = pd.get_dummies(df_processed, columns=['Type'], drop_first=True)

# 2-1. 파생 특징 생성
df_processed = add_engineered_features(df_processed)

# 3. 피처(X)와 타겟(y) 분리 및 데이터 누수(Leakage) 방지
target_col = 'Machine failure'
leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF'] # 세부 고장 모드 제거 필수

X = df_processed.drop(columns=[target_col] + leakage_cols)
y = df_processed[target_col]

# stratify 적용하여 클래스 비율 유지
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 연속형 센서 변수 스케일링 (Train 데이터 기준으로만 fit 적용)
scaler = StandardScaler()
num_cols = [
    'Air temperature [K]', 'Process temperature [K]', 
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'Temperature difference [K]', 'Torque x Speed', 'Wear / Speed', 'Torque / Speed', 'Wear x Torque'
]

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print(f"Train 피처 형태: {X_train.shape}")
print(f"Test 피처 형태: {X_test.shape}")

# PyTorch Dataset 정의
class ManufacturingDataset(Dataset):
    def __init__(self, features, labels):
        # DataFrame을 PyTorch FloatTensor로 변환
        self.X = torch.tensor(features.astype(np.float32).values, dtype=torch.float32)
        # 손실 함수(BCE Loss) 계산을 위해 타겟 형태를 [batch_size, 1]로 맞춤
        self.y = torch.tensor(labels.astype(np.float32).values, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ManufacturingDataset(X_train, y_train)
test_dataset = ManufacturingDataset(X_test, y_test)

print("\nPyTorch Dataset 생성 완료")

# DataLoader 구축 (증강/오버샘플링 없이 원본 분포 기반 학습)
batch_size = 64
loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, generator=loader_generator
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

print("DataLoader 구축 완료 (shuffle 기반 학습)")

# 배치 데이터 추출 테스트 및 오버샘플링 효과 확인
for batch_X, batch_y in train_loader:
    print(f"\n[Train DataLoader 배치 확인]")
    print(f"- X 텐서 형태: {batch_X.shape}")
    print(f"- y 텐서 형태: {batch_y.shape}")
    print(f"- 1개 배치({batch_size}개) 내 고장(1) 데이터 개수: {int(batch_y.sum().item())}개")
    break