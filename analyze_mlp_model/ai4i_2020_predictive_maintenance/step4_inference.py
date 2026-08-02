# 라이브러리 임포트 및 저장된 객체 불러오기
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

print("실시간 추론(Inference) 환경 준비 중...")

# 학습 장치(Device) 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# 모델 아키텍처 재정의 (학습 시와 동일해야 함)
class FaultDiagnosisMLP(nn.Module):
    def __init__(self, input_dim):
        super(FaultDiagnosisMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.15),    
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.15),    
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# 파일 경로 설정 (train_model.py에서 저장한 경로와 일치)
project_dir = Path(__file__).resolve().parent
model_path = project_dir / 'models' / 'fault_diagnosis_mlp.pth'
scaler_path = project_dir / 'models' / 'sensor_scaler.pkl'
threshold_path = project_dir / 'models' / 'decision_threshold.txt'
feature_columns_path = project_dir / 'models' / 'feature_columns.pkl'

# 저장된 최적 임계값 로드 (없으면 0.5 기본값 사용)
decision_threshold = 0.5
if threshold_path.exists():
    try:
        decision_threshold = float(threshold_path.read_text(encoding='utf-8').strip())
        print(f"의사결정 임계값 로드 완료: {decision_threshold:.4f}")
    except ValueError:
        print("경고: 저장된 임계값 파일 파싱 실패, 기본값 0.5 사용")

# 1. 스케일러 로드
try:
    scaler = joblib.load(str(scaler_path))
    print(f"스케일러 로드 완료: {scaler_path}")
except FileNotFoundError:
    print(f"에러: '{scaler_path}' 파일을 찾을 수 없습니다. 모델 학습을 먼저 진행해주세요.")
    exit()

def add_engineered_features(dataframe):
    engineered = dataframe.copy()
    engineered['Temperature difference [K]'] = engineered['Process temperature [K]'] - engineered['Air temperature [K]']
    engineered['Torque x Speed'] = engineered['Torque [Nm]'] * engineered['Rotational speed [rpm]']
    engineered['Wear / Speed'] = engineered['Tool wear [min]'] / (engineered['Rotational speed [rpm]'] + 1e-6)
    engineered['Torque / Speed'] = engineered['Torque [Nm]'] / (engineered['Rotational speed [rpm]'] + 1e-6)
    engineered['Wear x Torque'] = engineered['Tool wear [min]'] * engineered['Torque [Nm]']
    return engineered

# 1-1. 학습 시 저장한 피처 순서 로드
try:
    feature_columns = joblib.load(str(feature_columns_path))
    print(f"피처 순서 로드 완료: {feature_columns_path}")
except FileNotFoundError:
    print(f"에러: '{feature_columns_path}' 파일을 찾을 수 없습니다. 모델 학습을 먼저 진행해주세요.")
    exit()

# 2. 모델 가중치 로드
# 입력 차원은 학습 시 저장된 피처 순서를 기준으로 결정
INPUT_DIM = len(feature_columns)
model = FaultDiagnosisMLP(INPUT_DIM).to(device)

try:
    # weights_only=True를 통해 보안 경고 방지 및 안전한 로드 수행
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.eval() # 필수: 추론 모드로 전환하여 BatchNorm 동작을 고정
    print(f"모델 로드 및 평가 모드(eval) 전환 완료: {model_path}")
except FileNotFoundError:
    print(f"에러: '{model_path}' 파일을 찾을 수 없습니다.")
    exit()

# 현장 설비 실시간 센서 데이터 수집 (시뮬레이션)
# 현장의 PLC나 OPC-UA 서버를 통해 실시간으로 1건의 데이터가 들어왔다고 가정
incoming_data = {
    'Type': 'H',                       # 제품 등급 (L, M, H)
    'Air temperature [K]': 302.5,
    'Process temperature [K]': 311.2,
    'Rotational speed [rpm]': 1350,    # 평소보다 속도가 비정상적으로 떨어짐
    'Torque [Nm]': 70.0,               # 평소보다 토크가 높음 (과부하 징후)
    'Tool wear [min]': 215             # 공구 마모가 꽤 진행됨
}

df_new = pd.DataFrame([incoming_data])
print("\n[수집된 실시간 센서 데이터]")
# 환경에 따라 display가 없으면 print로 대체
print(df_new) if 'display' in globals() else print(df_new)

# 추론을 위한 데이터 전처리 (파이프라인)
# 학습 모델이 기대하는 피처와 순서를 정확히 맞춰야 합니다.

# 1. 범주형 변수(Type) One-Hot Encoding 수동 처리
df_new['Type_L'] = 1 if incoming_data['Type'] == 'L' else 0
df_new['Type_M'] = 1 if incoming_data['Type'] == 'M' else 0
df_new = df_new.drop(columns=['Type'])

# 1-1. 파생 특징 생성
df_new = add_engineered_features(df_new)

# 2. 컬럼 순서 재배치 (학습 데이터와 100% 동일한 순서)
df_new = df_new.reindex(columns=feature_columns, fill_value=0)

# 3. 연속형 센서 데이터 스케일링
num_cols = [
    'Air temperature [K]', 'Process temperature [K]', 
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'Temperature difference [K]', 'Torque x Speed', 'Wear / Speed', 'Torque / Speed', 'Wear x Torque'
]
df_new[num_cols] = scaler.transform(df_new[num_cols])

# 4. PyTorch 텐서 변환
X_tensor = torch.tensor(df_new.astype(np.float32).values, dtype=torch.float32).to(device)
print("\n데이터 전처리 및 텐서 변환 완료")

# AI 모델 결함 진단 수행
with torch.no_grad(): 
    output = model(X_tensor)
    prob = torch.sigmoid(output).item() # 0 ~ 1 사이 확률값으로 변환
    is_fault = prob >= decision_threshold

print("\n" + "="*40)
print("[AI 설비 상태 판별 결과] ")
print("="*40)
print(f"▶ 결함 발생 확률: {prob * 100:.2f}%")

if is_fault:
    print("[경고] 비정상 패턴 감지! 즉시 설비 점검이 필요합니다.")
else:
    print("[정상] 설비가 안정적으로 가동 중입니다.")
print("="*40)