# %% [1] 라이브러리 임포트 및 모델 로드
import os
import glob
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# 모델 아키텍처 정의 (Step 2와 동일)
class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 32)
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(32, 256 * 16 * 16),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Identity()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder_fc(encoded)
        decoded = decoded.view(-1, 256, 16, 16)
        reconstructed = self.decoder_conv(decoded)
        return reconstructed

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = AudioAutoencoder().to(device)
model.load_state_dict(torch.load('audio_models/audio_autoencoder_real.pth', map_location=device))
model.eval()
print("실제 밸브 학습 모델 로드 완료")

stats = np.load('audio_models/audio_mel_stats.npz')
MEL_MEAN = float(stats['mean'])
MEL_STD = float(stats['std'])
print(f"전역 Mel 통계 로드 완료: mean={MEL_MEAN:.4f}, std={MEL_STD:.4f}")

# %% [2] 평가용 데이터 경로 설정 (정상/비정상)
normal_test_dir = '0_dB_valve/valve/id_02/normal'
abnormal_test_dir = '0_dB_valve/valve/id_02/abnormal'

# 효율적인 실습을 위해 각 폴더에서 50개씩 샘플링 (파일이 많을 경우 조정 가능)
normal_test_files = glob.glob(os.path.join(normal_test_dir, '*.wav'))[:50]
abnormal_test_files = glob.glob(os.path.join(abnormal_test_dir, '*.wav'))[:50]

# %% [3] 파일별 복원 오차 계산 함수
def compute_errors(file_list, sr=16000, n_mels=512, max_frames=512):
    errors = []
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for path in file_list:
            y, _ = librosa.load(path, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = np.clip(mel_db, -80.0, 0.0)
            
            # [변경됨] 표준화 (Standardization): 학습셋 전역 평균/표준편차로 변환
            mel_db = (mel_db - MEL_MEAN) / MEL_STD
            
            # 128x128 크기로 맞춤
            mel_input = mel_db[:, :max_frames] if mel_db.shape[1] >= max_frames else np.pad(mel_db, ((0,0),(0, max_frames-mel_db.shape[1])))
            
            input_tensor = torch.tensor(mel_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            reconstructed = model(input_tensor)
            
            loss = criterion(reconstructed, input_tensor)
            errors.append(loss.item())
    return np.array(errors)

print("정상 및 비정상 데이터의 복원 오차 계산 중...")
normal_errors = compute_errors(normal_test_files)
abnormal_errors = compute_errors(abnormal_test_files)

# %% [4] 오차 분포 시각화 (Threshold 결정을 위한 분석)
plt.figure(figsize=(10, 6))
sns.histplot(normal_errors, kde=True, color='blue', label='Normal (Valve)')
sns.histplot(abnormal_errors, kde=True, color='red', label='Abnormal (Valve)')
plt.title('Reconstruction Error Distribution (MIMII Valve ID02)')
plt.xlabel('Mean Squared Error (MSE)')
plt.legend()
plt.show()

# %% [5] ROC-AUC 성능 평가
y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(abnormal_errors))])
y_scores = np.concatenate([normal_errors, abnormal_errors])

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# %% [6] F1-Score 및 최적 임계값(Optimal Threshold) 탐색
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Precision-Recall Curve 계산을 통해 최적의 임계값 찾기
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# F1-Score가 최대가 되는 임계값 찾기 (분모가 0이 되는 것을 방지하기 위해 1e-8 추가)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"\n최적의 이상치 탐지 임계값(Threshold): {optimal_threshold:.4f}")
print(f"해당 임계값에서의 평가지표 - 최고 F1-Score: {optimal_f1:.4f}")

# 최적 임계값을 기준으로 예측(Prediction) 라벨 생성 (임계값보다 크면 고장=1, 작으면 정상=0)
y_pred = (y_scores >= optimal_threshold).astype(int)

# Confusion Matrix 시각화
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title(f'Confusion Matrix (Optimal Threshold: {optimal_threshold:.4f})')
plt.show()

# %% [7] 원본 vs 재구성 스펙트로그램 시각화 비교 (Explainability)
import librosa.display

def visualize_reconstruction(file_path, title):
    y, sr = librosa.load(file_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, -80.0, 0.0)
    
    # 표준화
    mel_db_norm = (mel_db - MEL_MEAN) / MEL_STD
    max_frames = 256
    
    # 128 단위로 패딩 및 자르기
    if mel_db_norm.shape[1] >= max_frames:
        mel_input = mel_db_norm[:, :max_frames]
    else:
        mel_input = np.pad(mel_db_norm, ((0,0),(0, max_frames - mel_db_norm.shape[1])))
        
    input_tensor = torch.tensor(mel_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed_tensor = model(input_tensor)
        
    # 모델을 통과한 텐서를 넘파이 배열로 변환 - (128, 128)
    original_img = input_tensor.cpu().squeeze().numpy()
    reconstructed_img = reconstructed_tensor.cpu().squeeze().numpy()
    
    # 표준화된 값 그대로 비교
    original_db = original_img
    reconstructed_db = reconstructed_img
    
    plt.figure(figsize=(12, 4))
    
    # 원본 스펙트로그램 표시
    plt.subplot(1, 2, 1)
    librosa.display.specshow(original_db, sr=16000, x_axis='time', y_axis='mel')
    plt.title(f'{title} - Original (Standardized Input)')
    plt.colorbar(format='%+2.0f')
    
    # 모델이 복원한(재구성한) 스펙트로그램 표시
    plt.subplot(1, 2, 2)
    librosa.display.specshow(reconstructed_db, sr=16000, x_axis='time', y_axis='mel')
    plt.title(f'{title} - Reconstructed (Output)')
    plt.colorbar(format='%+2.0f')
    
    plt.tight_layout()
    plt.show()

print("\n정상 및 비정상 오디오 스펙트로그램 복원 결과 시각화 중...")
visualize_reconstruction(normal_test_files[0], "Normal Valve Sound")
visualize_reconstruction(abnormal_test_files[0], "Abnormal Valve Sound")