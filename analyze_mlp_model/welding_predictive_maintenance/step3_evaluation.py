import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import torch.nn as nn

# 1. 모델 클래스 재정의
class LSTM_AE(nn.Module):
    def __init__(self, n_features, seq_len):
        super(LSTM_AE, self).__init__()
        self.seq_len = seq_len
        self.enc1 = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)
        self.enc2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.dec1 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.dec2 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.out = nn.Linear(64, n_features)
    def forward(self, x):
        x, _ = self.enc1(x)
        x, (h_n, _) = self.enc2(x)
        x = h_n.transpose(0, 1).repeat(1, self.seq_len, 1)
        x, _ = self.dec1(x)
        x, _ = self.dec2(x)
        x = self.out(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. 데이터 및 모델 로딩 중...")
# 보안 경고 해결을 위해 weights_only=False 적용
dataset = torch.load('processed_dataset.pt', weights_only=False)
X_valid, Y_valid, Y_val_index = dataset['X_valid'], dataset['Y_valid'], dataset['Y_val_index']
X_test, Y_test, Y_te_index = dataset['X_test'], dataset['Y_test'], dataset['Y_te_index']

model = LSTM_AE(n_features=5, seq_len=20).to(device)
# 가중치 파일 로딩
model.load_state_dict(torch.load('best_lstm_ae.pth', weights_only=True))
model.eval()

def flatten_last_step(X):
    return X[:, -1, :].numpy()

# ---------------------------------------------------------
# 2. 검증셋을 통한 최적 임곗값(Threshold) 탐색 및 시각화
# ---------------------------------------------------------
print("2. 최적 임곗값 탐색 중...")
with torch.no_grad():
    valid_predictions = model(X_valid.to(device)).cpu()
    
mse_valid = np.mean(np.power(flatten_last_step(Y_valid) - flatten_last_step(valid_predictions), 2), axis=1)

thresholds = np.linspace(0, 20, 100)
precisions, recalls = [], []

for thr in thresholds:
    pred_val = [1 if e > thr else 0 for e in mse_valid]
    precisions.append(precision_score(Y_val_index, pred_val, zero_division=0))
    recalls.append(recall_score(Y_val_index, pred_val, zero_division=0))

# 정밀도와 재현율이 같아지는 교점 찾기 [cite: 337, 950]
index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precisions, recalls)) if p == r][0]
optimal_threshold = thresholds[index_cnt]
print(f"-> 설정된 최적 임곗값: {optimal_threshold:.4f}")

# [시각화 1] 정밀도-재현율 곡선 (Precision-Recall Curve) [cite: 334-351, 952-961]
plt.figure(figsize=(8, 5))
plt.title('Precision/Recall Curve for Threshold', fontsize=14)
plt.plot(thresholds, precisions, label='Precision', color='navy', linestyle='--')
plt.plot(thresholds, recalls, label='Recall', color='seagreen')
plt.plot(optimal_threshold, precisions[index_cnt], 'ro', label=f'Optimal Threshold ({optimal_threshold:.2f})')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() # 첫 번째 창 닫으면 다음으로 넘어감

# ---------------------------------------------------------
# 3. 테스트셋 최종 평가 및 결과 시각화
# ---------------------------------------------------------
print("3. 테스트셋 모델 평가 중...")

np.random.seed(42) # 실행할 때마다 동일한 형태의 섞임을 보기 위해 시드 고정
shuffle_idx = np.random.permutation(len(Y_te_index)) # 0부터 데이터 길이까지의 인덱스를 섞음

X_test_shuffled = X_test[shuffle_idx]
Y_test_shuffled = Y_test[shuffle_idx]
Y_te_index_shuffled = Y_te_index[shuffle_idx]

with torch.no_grad():
    test_predictions = model(X_test_shuffled.to(device)).cpu()

mse_test = np.mean(np.power(flatten_last_step(Y_test_shuffled) - flatten_last_step(test_predictions), 2), axis=1)
pred_y = [1 if e > optimal_threshold else 0 for e in mse_test]

print(f"\n[최종 평가 결과]")
print(f"정확도 (Accuracy): {accuracy_score(Y_te_index_shuffled, pred_y)*100:.2f}%")
print(f"F1-Score: {f1_score(Y_te_index_shuffled, pred_y):.4f}")

# 정상 데이터와 이상 데이터의 오차 분리
mse_normal = mse_test[Y_te_index_shuffled == 0]
mse_anomaly = mse_test[Y_te_index_shuffled == 1]
idx_normal = np.where(Y_te_index_shuffled == 0)[0]
idx_anomaly = np.where(Y_te_index_shuffled == 1)[0]

# [시각화 2] 재구성 오차 산점도 (Reconstruction Error Scatter Plot) [cite: 354-369, 993-1013]
plt.figure(figsize=(10, 5))
plt.title('Reconstruction Error by Class', fontsize=14)
plt.scatter(idx_normal, mse_normal, label='Normal', color='coral', alpha=0.6, s=15)
plt.scatter(idx_anomaly, mse_anomaly, label='Anomaly', color='steelblue', alpha=0.6, s=15)
plt.axhline(optimal_threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Data Point Index')
plt.ylabel('Reconstruction Error (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() # 두 번째 창

# [시각화 3] 재구성 오차 분포 (Histogram/Density) 
plt.figure(figsize=(8, 5))
plt.title('Distribution of Reconstruction Errors', fontsize=14)
sns.histplot(mse_normal, bins=50, color='coral', label='Normal', kde=True, stat="density")
sns.histplot(mse_anomaly, bins=50, color='steelblue', label='Anomaly', kde=True, stat="density")
plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show() # 세 번째 창

# [시각화 4] 오차 행렬 (Confusion Matrix) 
cm = confusion_matrix(Y_te_index_shuffled, pred_y)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", cbar=False, annot_kws={"size": 14})
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('Observed Labels', fontsize=12)
plt.title('Confusion Matrix of LSTM-AE', fontsize=14)
plt.xticks([0.5, 1.5], ['Normal (0)', 'Anomaly (1)'])
plt.yticks([0.5, 1.5], ['Normal (0)', 'Anomaly (1)'])
plt.tight_layout()
plt.show() # 네 번째 창