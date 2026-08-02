import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 시각화 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 데이터 로드
csv_path = Path(__file__).resolve().parent / 'ai4i2020.csv'
if not csv_path.exists():
    raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

df = pd.read_csv(csv_path)

print(f"데이터셋 형태: {df.shape}")
print(df.head())

# 결측치 및 데이터 타입 확인
df.info()

# 주요 센서 및 공정 변수
sensor_cols = [
    'Air temperature [K]', 'Process temperature [K]', 
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(sensor_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(f'Distribution of {col}')
    
# 남는 그래프 공간 숨기기
axes[5].set_visible(False)
plt.tight_layout()
plt.show()

# 센서 변수들 간의 피어슨 상관계수 계산
corr_matrix = df[sensor_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Correlation Matrix of Sensor Variables')
plt.show()

# Machine failure (0: 정상, 1: 고장) 분포 확인
failure_counts = df['Machine failure'].value_counts()
failure_ratio = failure_counts / len(df) * 100

print(f"정상 데이터 비율: {failure_ratio[0]:.2f}%")
print(f"고장 데이터 비율: {failure_ratio[1]:.2f}%\n")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Machine failure', palette='Set2')
plt.title('Distribution of Target Variable (Machine failure)')
plt.show()

# 세부 고장 유형 파악 (TWF, HDF, PWF, OSF, RNF)
failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
print("세부 고장 유형별 발생 건수:")
print(df[failure_modes].sum())