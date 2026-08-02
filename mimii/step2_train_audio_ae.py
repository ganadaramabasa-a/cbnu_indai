# %% [1] 라이브러리 임포트 및 장치 설정
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"학습 장치(Device) 설정 완료: {device}")

# %% [2] 실제 MIMII 데이터셋 경로 설정 및 커스텀 Dataset 구축
normal_dir = '0_dB_valve/valve/id_02/normal'
normal_files = glob.glob(os.path.join(normal_dir, '*.wav'))

if not normal_files:
    print(f"에러: '{normal_dir}' 경로에서 .wav 파일을 찾을 수 없습니다.")
    exit()

print(f"로드된 정상 학습 오디오 파일 개수: {len(normal_files)}개")

STATS_PATH = 'audio_models/audio_mel_stats.npz'


def compute_global_mel_stats(file_paths, sr=16000, n_mels=512):
    values = []
    for path in file_paths:
        y, _ = librosa.load(path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=1.0)
        mel_db = np.clip(mel_db, -80.0, 0.0)
        values.append(mel_db.reshape(-1))

    all_values = np.concatenate(values)
    return all_values.mean(), all_values.std() + 1e-8


global_mel_mean, global_mel_std = compute_global_mel_stats(normal_files)
print(f"전역 Mel 통계 계산 완료: mean={global_mel_mean:.4f}, std={global_mel_std:.4f}")

os.makedirs('audio_models', exist_ok=True)
np.savez(STATS_PATH, mean=global_mel_mean, std=global_mel_std)
print(f"전역 Mel 통계를 '{STATS_PATH}'에 저장했습니다.")

class MIMIIMelDataset(Dataset):
    def __init__(self, file_paths, sr=16000, n_mels=512, max_frames=512, mel_mean=0.0, mel_std=1.0):
        self.file_paths = file_paths
        self.sr = sr
        self.n_mels = n_mels
        self.max_frames = max_frames # CNN 입력 크기를 고정하기 위함 (512x512)
        self.mel_mean = mel_mean
        self.mel_std = mel_std

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. 파일 로드
        path = self.file_paths[idx]
        y, _ = librosa.load(path, sr=self.sr)
        
        # 2. Mel-Spectrogram 변환
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=1.0)
        mel_db = np.clip(mel_db, -80.0, 0.0)
        
        # 표준화 (Standardization): 학습셋 전역 평균/표준편차로 변환
        mel_db = (mel_db - self.mel_mean) / self.mel_std
        
        # 3. 텐서 크기 고정 (128x128)
        # 긴 파일은 항상 앞부분만 쓰지 않고 임의 구간을 잘라 학습해 일반화를 높입니다.
        if mel_db.shape[1] > self.max_frames:  
            start = np.random.randint(0, mel_db.shape[1] - self.max_frames + 1)
            mel_db = mel_db[:, start:start + self.max_frames]
        else:
            # 혹시 길이가 짧은 파일이 있다면 0으로 패딩(Padding)을 채웁니다.
            pad_width = self.max_frames - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # 4. PyTorch 텐서 변환 (Channel, Mels, Time) -> (1, 128, 128)
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        
        # 오토인코더는 입력 데이터를 그대로 정답(Target)으로 사용하므로 두 번 반환합니다.
        return mel_tensor, mel_tensor

# 데이터셋 및 DataLoader 생성
train_dataset = MIMIIMelDataset(normal_files, mel_mean=global_mel_mean, mel_std=global_mel_std)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print("학습용 DataLoader 구축 완료 (512x512 크기로 자동 전처리 적용)")

# %% [3] CNN 기반 오토인코더(Autoencoder) 아키텍처 정의
class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()
        
        # 인코더 (Encoder): 512x512 이미지를 압축
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
        
        # 디코더 (Decoder): 다시 원본 크기인 512x512으로 복원
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
        decoded = decoded.view(-1, 256, 16, 16) # 다시 2D 형태로 변환
        reconstructed = self.decoder_conv(decoded)
        return reconstructed

model = AudioAutoencoder().to(device)
print("\n[오토인코더 모델 준비 완료]")

# %% [4] 손실 함수 및 최적화 설정
# 픽셀 단위의 복원 오차를 측정하는 MSE 사용
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.005)

# %% [5] 오토인코더 학습 루프 (Training Loop)
epochs = 100
print("\n[모델 학습 시작 - 실제 밸브의 정상 작동 소리 학습]")
model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_x, batch_target in train_loader:
        batch_x = batch_x.to(device)
        batch_target = batch_target.to(device)
        
        # Forward, Loss, Backward, Step
        outputs = model(batch_x)
        loss = criterion(outputs, batch_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    
    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:2d}/{epochs}] | Reconstruction Loss (MSE): {avg_loss:.4f}")

print("학습 완료!")

# %% [6] 모델 가중치 저장
os.makedirs('audio_models', exist_ok=True)
model_path = 'audio_models/audio_autoencoder_real.pth'
torch.save(model.state_dict(), model_path)
print(f"\n[저장 완료] 밸브 정상음 학습 모델이 '{model_path}'에 저장되었습니다.")