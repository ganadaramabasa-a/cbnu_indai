# %% [1] 라이브러리 임포트 및 모델 로드
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
import glob
import os

# 모델 아키텍처 정의 (학습 시와 동일)
class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()
        
        # 인코더 (Encoder): 128x128 이미지를 압축
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
            nn.Linear(256 * 8 * 8, 32)
        )
        
        # 디코더 (Decoder): 다시 원본 크기인 128x128로 복원
        self.decoder_fc = nn.Sequential(
            nn.Linear(32, 256 * 8 * 8),
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
        decoded = decoded.view(-1, 256, 8, 8)
        reconstructed = self.decoder_conv(decoded)
        return reconstructed

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = AudioAutoencoder().to(device)
model.load_state_dict(torch.load('audio_models/audio_autoencoder_real.pth', map_location=device))
model.eval()

stats = np.load('audio_models/audio_mel_stats.npz')
MEL_MEAN = float(stats['mean'])
MEL_STD = float(stats['std'])

# 평가 시 결정된 임계값 (예시: 이전 단계 히스토그램 분석 결과에 따라 설정)
THRESHOLD = 0.0016
print(f"모델 로드 완료. 설정된 결함 임계값: {THRESHOLD}")

# %% [2] 추론용 비정상 오디오 파일 선택
abnormal_files = glob.glob('0_dB_valve/valve/id_02/abnormal/*.wav')
test_file = abnormal_files[0] # 첫 번째 고장 파일 선택
print(f"추론 대상 파일: {test_file}")

y, sr = librosa.load(test_file, sr=16000)

# %% [3] 슬라이딩 윈도우 추론 시뮬레이션
window_size = sr * 2 # 2초 단위 분석
hop_length = sr * 1  # 1초씩 이동
n_mels = 256

scores = []
times = []

print("\n실시간 모니터링 시뮬레이션 중...")
for start in range(0, len(y) - window_size, hop_length):
    window = y[start : start + window_size]
    
    # Mel-Spectrogram 변환
    mel = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, -80.0, 0.0)
    
    # 학습셋 전역 통계 기준 표준화
    mel_db = (mel_db - MEL_MEAN) / MEL_STD

    # 모델의 지정학습 형태인 256프레임 고정을 위한 자르기 및 패딩(Padding)
    max_frames = 256
    if mel_db.shape[1] > max_frames:
        mel_db = mel_db[:, :max_frames]
    else:
        pad_width = max_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    input_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed = model(input_tensor)
        loss = nn.MSELoss()(reconstructed, input_tensor).item()
    
    scores.append(loss)
    current_time = start / sr
    times.append(current_time)
    
    status = "[ABNORMAL]" if loss > THRESHOLD else "[NORMAL]"
    print(f"Time: {current_time:4.1f}s | Anomaly Score: {loss:.4f} | Status: {status}")

# %% [4] 시간에 따른 결함 점수 추이 시각화
plt.figure(figsize=(12, 5))
plt.plot(times, scores, marker='o', label='Anomaly Score')
plt.axhline(y=THRESHOLD, color='red', linestyle='--', label='Detection Threshold')
plt.title('Machine Status Monitoring over Time')
plt.xlabel('Time (sec)')
plt.ylabel('Reconstruction Error (MSE)')
plt.legend()
plt.show()