import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

print("1. 데이터 로딩 중...")
normal = pd.read_csv('normal_data.csv', index_col=0)
outlier = pd.read_csv('outlier_data.csv')

normal_data = normal.copy()
outlier_data = outlier.copy()
normal_data['FIN_JGMT'] = 0
outlier_data['FIN_JGMT'] = 1

use_col = ['DV_R', 'DA_R', 'AV_R', 'AA_R', 'PM_R']
X_normal = normal_data[use_col]
y_normal = normal_data['FIN_JGMT']
X_anomaly = outlier_data[use_col]
y_anomaly = outlier_data['FIN_JGMT']

# 가이드북 기준: 정상 데이터 60,000개 분리
X_train_normal = X_normal[:60000]
y_train_normal = y_normal[:60000]
X_test_normal = X_normal[60000:]
y_test_normal = y_normal[60000:]
X_test_anomaly = X_anomaly
y_test_anomaly = y_anomaly

print("2. 데이터 스케일링 적용 중...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_normal_scaled = scaler.transform(X_test_normal)
X_test_anomaly_scaled = scaler.transform(X_test_anomaly)

y_train_normal = np.array(y_train_normal)
y_test_normal = np.array(y_test_normal)
y_test_anomaly = np.array(y_test_anomaly)

print("3. 시계열 시퀀스(Sequence) 생성 중...")
sequence = 20

def create_sequences(X_scaled, y_data):
    X_seq, Y_seq, Y_idx = [], [], []
    for index in range(len(X_scaled) - sequence - 40):
        # Input: 현재부터 20-step (0~6초)
        X_seq.append(X_scaled[index : index + sequence])
        # Target: 일정 시간 뒤의 20-step (12~18초) 예측
        Y_seq.append(X_scaled[index + sequence + 20 : index + sequence + 40])
        Y_idx.append(y_data[index + sequence + 40])
    return np.array(X_seq), np.array(Y_seq), np.array(Y_idx)

X_train, Y_train, Y_tr_index = create_sequences(X_train_scaled, y_train_normal)
X_test_normal_arr, Y_test_normal_arr, Y_te_index_1 = create_sequences(X_test_normal_scaled, y_test_normal)
X_test_anomal_arr, Y_test_anomal_arr, Y_te_index_2 = create_sequences(X_test_anomaly_scaled, y_test_anomaly)

# 가이드북 기준 검증(Validation) 데이터셋 분리
X_valid_normal, Y_valid_normal = X_test_normal_arr[:4000], Y_test_normal_arr[:4000]
Y_val_index_1 = Y_te_index_1[:4000]
X_test_normal_arr, Y_test_normal_arr = X_test_normal_arr[4000:], Y_test_normal_arr[4000:]
Y_te_index_1 = Y_te_index_1[4000:]

X_valid_anomal, Y_valid_anomal = X_test_anomal_arr[:1200], Y_test_anomal_arr[:1200]
Y_val_index_2 = Y_te_index_2[:1200]
X_test_anomal_arr, Y_test_anomal_arr = X_test_anomal_arr[1200:], Y_test_anomal_arr[1200:]
Y_te_index_2 = Y_te_index_2[1200:]

X_valid = np.vstack((X_valid_normal, X_valid_anomal))
Y_valid = np.vstack((Y_valid_normal, Y_valid_anomal))
Y_val_index = np.hstack((Y_val_index_1, Y_val_index_2))

X_test = np.vstack((X_test_normal_arr, X_test_anomal_arr))
Y_test = np.vstack((Y_test_normal_arr, Y_test_anomal_arr))
Y_te_index = np.hstack((Y_te_index_1, Y_te_index_2))

# 학습용 검증 데이터 (정상 데이터만)
X_valid_0 = X_valid[Y_val_index == 0]
Y_valid_0 = Y_valid[Y_val_index == 0]

print("4. 전처리된 데이터를 PyTorch 형식으로 저장합니다...")
torch.save({
    'X_train': torch.FloatTensor(X_train), 'Y_train': torch.FloatTensor(Y_train),
    'X_valid_0': torch.FloatTensor(X_valid_0), 'Y_valid_0': torch.FloatTensor(Y_valid_0),
    'X_valid': torch.FloatTensor(X_valid), 'Y_valid': torch.FloatTensor(Y_valid), 'Y_val_index': Y_val_index,
    'X_test': torch.FloatTensor(X_test), 'Y_test': torch.FloatTensor(Y_test), 'Y_te_index': Y_te_index
}, 'processed_dataset.pt')
print("데이터셋 준비 완료! (processed_dataset.pt 생성)")