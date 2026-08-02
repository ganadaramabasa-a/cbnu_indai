# 라이브러리 임포트 및 전처리 데이터 불러오기
import os
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 로깅용
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

import joblib



# 재현 가능한 학습을 위한 시드 고정
SEED = 326
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 전처리 모듈 임포트
try:
    from step2_data_prep import train_loader, test_loader, X_train, y_train, scaler
    print("데이터 전처리 모듈(step2_data_prep) 로드 성공")
except ModuleNotFoundError:
    print("에러: 'step2_data_prep.py' 파일을 찾을 수 없습니다. 같은 폴더에 있는지 확인해주세요.")
    exit()

# 학습 장치(Device) 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"학습 장치(Device) 설정 완료: {device}")

# 모델 아키텍처 정의 및 초기화
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

input_dim = X_train.shape[1] 
model = FaultDiagnosisMLP(input_dim).to(device)

print(f"\n[모델 구조 확인]\n{model}")
print(f"재현성 설정 시드(SEED): {SEED}")

# Train/Validation 분리: 임계값과 조기 종료는 검증셋으로만 결정
X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
)

class ManufacturingDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features.astype(np.float32).values, dtype=torch.float32)
        self.y = torch.tensor(labels.astype(np.float32).values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset_main = ManufacturingDataset(X_train_main, y_train_main)
val_dataset = ManufacturingDataset(X_val, y_val)

train_loader_main = torch.utils.data.DataLoader(train_dataset_main, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# 손실 함수, 최적화 알고리즘 및 TensorBoard 설정
# 클래스 불균형을 완화하기 위한 손실 가중치 계산 (증강 없음)
pos_count = float((y_train_main == 1).sum())
neg_count = float((y_train_main == 0).sum())
pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# TensorBoard 기록을 위한 SummaryWriter 초기화 (저장 경로 지정)
project_dir = Path(__file__).resolve().parent
tb_root = project_dir / "runs"
tb_root.mkdir(parents=True, exist_ok=True)
log_dir = tb_root / "fault_diagnosis_experiment"

# 같은 경로에 파일이 있으면 로그 디렉터리를 만들 수 없으므로 파일명 변경
if log_dir.exists() and not log_dir.is_dir():
    backup_path = log_dir.with_name(f"{log_dir.name}_file_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_dir.rename(backup_path)
    print(f"경고: TensorBoard 로그 경로가 파일이라 백업 이름으로 변경했습니다: {backup_path}")

log_dir.mkdir(parents=True, exist_ok=True)
writer = None
try:
    writer = SummaryWriter(str(log_dir))
    print(f"TensorBoard 로그 디렉토리 설정: {log_dir}")
except Exception as e:
    fallback_dir = project_dir / "runs" / f"fault_diagnosis_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    try:
        writer = SummaryWriter(str(fallback_dir))
        print(f"경고: 기본 로그 경로 사용 실패({e}). 대체 경로 사용: {fallback_dir}")
    except Exception as e2:
        print(f"경고: TensorBoard 비활성화 (초기화 실패): {e2}")

# 모델 그래프(구조)를 TensorBoard에 기록
# (더미 데이터를 하나 통과시켜서 그래프를 그립니다)
dummy_input = torch.randn(1, input_dim).to(device)
if writer is not None:
    writer.add_graph(model, dummy_input)

# 모델 학습 및 검증 루프 (Training & Validation Loop)
epochs = 50
best_val_f1 = -1.0
best_threshold = 0.5
best_model_state = deepcopy(model.state_dict())
best_val_loss = float('inf')
patience = 8
patience_counter = 0
print("\n [모델 학습 시작]")

for epoch in range(epochs):
    # --- 1. Training Phase ---
    model.train() 
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader_main:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad() 
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()       
        optimizer.step()      
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader_main)
    
    # --- 2. Validation(Test) Phase ---
    model.eval()
    val_loss = 0.0
    all_probs, all_targets = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            v_loss = criterion(outputs, batch_y)
            val_loss += v_loss.item()
            
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.squeeze(1).cpu().numpy().tolist())
            all_targets.extend(batch_y.squeeze(1).cpu().numpy().tolist())

    avg_val_loss = val_loss / len(val_loader)

    # 임계값 스윕으로 Validation F1 최대화
    threshold_candidates = [i / 100 for i in range(10, 91)]
    threshold_to_f1 = {}
    for thr in threshold_candidates:
        preds_thr = (np.array(all_probs) >= thr).astype(int)
        threshold_to_f1[thr] = f1_score(all_targets, preds_thr, zero_division=0)

    epoch_best_thr = max(threshold_to_f1, key=threshold_to_f1.get)
    val_f1 = threshold_to_f1[epoch_best_thr]

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_threshold = epoch_best_thr
        best_model_state = deepcopy(model.state_dict())

    scheduler.step(val_f1)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"조기 종료: {epoch+1} 에폭에서 검증 손실 개선이 멈췄습니다.")
            break
    
    # --- 3. TensorBoard에 지표 기록 ---
    if writer is not None:
        writer.add_scalars('Loss', {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch)
        writer.add_scalar('Metrics/Validation_F1', val_f1, epoch)
        writer.add_scalar('Metrics/Best_Threshold', epoch_best_thr, epoch)
    
    # 진행 상황 출력
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:2d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1(best-thr): {val_f1:.4f} | Thr: {epoch_best_thr:.2f}")

model.load_state_dict(best_model_state)
print(f"최적 Validation F1: {best_val_f1:.4f}, 최적 임계값: {best_threshold:.2f}")

if writer is not None:
    writer.close()
    print("학습 및 TensorBoard 기록 완료!")
else:
    print("학습 완료! (TensorBoard 기록은 비활성화됨)")

# 최종 모델 평가 (Evaluation) 
model.eval() 
all_preds, all_probs, all_targets = [], [], []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = model(batch_X)
        probs = torch.sigmoid(outputs)
        preds = (probs >= best_threshold).float()
        
        all_probs.extend(probs.squeeze(1).cpu().numpy().tolist())
        all_preds.extend(preds.squeeze(1).cpu().numpy().tolist())
        all_targets.extend(batch_y.squeeze(1).cpu().numpy().tolist())

# 평가지표 계산
acc = accuracy_score(all_targets, all_preds)
prec = precision_score(all_targets, all_preds, zero_division=0)
rec = recall_score(all_targets, all_preds, zero_division=0)
f1 = f1_score(all_targets, all_preds, zero_division=0)
auc = roc_auc_score(all_targets, all_probs)

print("\n[최종 테스트 데이터셋 평가 결과]")
print(f"Accuracy (정확도):  {acc:.4f}")
print(f"Precision (정밀도): {prec:.4f}")
print(f"Recall (재현율):    {rec:.4f}")
print(f"F1-Score:           {f1:.4f}")
print(f"ROC-AUC:            {auc:.4f}")

# 모델 가중치 및 스케일러 저장
# 디렉토리가 없으면 생성
models_dir = project_dir / 'models'
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / 'fault_diagnosis_mlp.pth'
scaler_path = models_dir / 'sensor_scaler.pkl'
threshold_path = models_dir / 'decision_threshold.txt'
feature_columns_path = models_dir / 'feature_columns.pkl'

torch.save(model.state_dict(), str(model_path))
joblib.dump(scaler, str(scaler_path))
joblib.dump(list(X_train.columns), str(feature_columns_path))
with open(threshold_path, 'w', encoding='utf-8') as f:
    f.write(str(best_threshold))

print(f"\n[저장 완료] 모델 가중치('{model_path}')와 스케일러('{scaler_path}')가 저장되었습니다.")
print(f"[저장 완료] 최적 임계값('{threshold_path}'): {best_threshold:.4f}")
print(f"[저장 완료] 피처 순서('{feature_columns_path}')가 저장되었습니다.")

# 모델 평가 결과 시각화 (Confusion Matrix 및 평가지표)


print("\n평가 결과 시각화 그래프를 생성합니다...")

# 시각화 환경 및 레이아웃 설정 (1행 3열)
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 1. Confusion Matrix (혼동 행렬) ---
cm = confusion_matrix(all_targets, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal (0)', 'Fault (1)'],
            yticklabels=['Normal (0)', 'Fault (1)'],
            cbar=False, annot_kws={"size": 14})
axes[0].set_title('Confusion Matrix', fontsize=14, pad=10)
axes[0].set_ylabel('Actual Status', fontsize=12)
axes[0].set_xlabel('Predicted Status', fontsize=12)

# --- 2. 5가지 평가지표 요약 바 차트 ---
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [acc, prec, rec, f1, auc]

sns.barplot(x=metrics_names, y=metrics_values, ax=axes[1], palette='viridis')
axes[1].set_title('Evaluation Metrics Summary', fontsize=14, pad=10)
axes[1].set_ylim(0, 1.1) # y축 범위를 0~1.1로 고정하여 여백 확보

# 막대 그래프 위에 정확한 수치 텍스트 표시
for i, v in enumerate(metrics_values):
    axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# --- 3. ROC Curve (수신자 조작 특성 곡선) ---
fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
axes[2].plot(fpr, tpr, color='crimson', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
axes[2].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', alpha=0.7)
axes[2].set_xlim([-0.02, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('False Positive Rate (FPR)', fontsize=12)
axes[2].set_ylabel('True Positive Rate (TPR)', fontsize=12)
axes[2].set_title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=10)
axes[2].legend(loc="lower right", fontsize=11)

# 그래프 간격 조절 및 출력
plt.tight_layout()
plt.show()