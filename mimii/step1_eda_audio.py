# %% [1] 라이브러리 임포트 및 시각화 설정
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd


# VS Code 대화형 창 그래프 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
print("라이브러리 로드 완료")

# %% [2] 실제 MIMII 오디오 데이터 로드
# 데이터셋이 위치한 실제 경로 설정
normal_dir = '0_dB_valve/valve/id_02/normal'
abnormal_dir = '0_dB_valve/valve/id_02/abnormal'

# 각 폴더에서 첫 번째 .wav 파일 경로 가져오기
normal_files = glob.glob(os.path.join(normal_dir, '*.wav'))
abnormal_files = glob.glob(os.path.join(abnormal_dir, '*.wav'))

# 파일 존재 여부 확인
if not normal_files or not abnormal_files:
    print("에러: 지정된 경로에서 .wav 파일을 찾을 수 없습니다. 경로 구조를 확인해주세요.")
    print(f" - 확인된 정상 경로: {os.path.abspath(normal_dir)}")
    print(f" - 확인된 비정상 경로: {os.path.abspath(abnormal_dir)}")
else:
    normal_audio_path = normal_files[0]
    abnormal_audio_path = abnormal_files[0]
    
    print(f"로드된 정상 데이터: {normal_audio_path}")
    print(f"로드된 비정상 데이터: {abnormal_audio_path}")

    # librosa를 사용해 16,000Hz로 샘플링하여 오디오 로드
    sr_target = 16000
    y_normal, sr = librosa.load(normal_audio_path, sr=sr_target)
    y_anomaly, _ = librosa.load(abnormal_audio_path, sr=sr_target)

    print(f"\n오디오 샘플링 레이트(sr): {sr} Hz")
    print(f"정상 데이터 길이: {len(y_normal)} 샘플 ({len(y_normal)/sr:.2f}초)")

# %% [3] 1. Waveform (시간 영역 파형) 시각화 및 청음
# 소리의 진폭을 시간에 따라 보여줍니다.
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(12, 6))

librosa.display.waveshow(y_normal, sr=sr, ax=ax[0], color='steelblue')
ax[0].set_title('Normal Valve Sound - Waveform')

librosa.display.waveshow(y_anomaly, sr=sr, ax=ax[1], color='crimson')
ax[1].set_title('Abnormal Valve Sound - Waveform')

plt.tight_layout()
plt.show()

print("정상 소리 재생 중...")
sd.play(y_normal, sr)
sd.wait() # 오디오 재생이 끝날 때까지 코드가 넘어가지 않도록 대기

print("비정상 소리 재생 중...")
sd.play(y_anomaly, sr)
sd.wait() # 대기

# %% [4] 2. STFT를 통한 Spectrogram (주파수 영역) 변환
# 시간에 따른 주파수 성분의 변화를 시각화합니다.
D_normal = librosa.stft(y_normal)
S_db_normal = librosa.amplitude_to_db(np.abs(D_normal), ref=np.max)

D_anomaly = librosa.stft(y_anomaly)
S_db_anomaly = librosa.amplitude_to_db(np.abs(D_anomaly), ref=np.max)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
img1 = librosa.display.specshow(S_db_normal, sr=sr, x_axis='time', y_axis='hz', ax=ax[0], cmap='magma')
ax[0].set_title('Spectrogram - Normal Valve')

img2 = librosa.display.specshow(S_db_anomaly, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], cmap='magma')
ax[1].set_title('Spectrogram - Abnormal Valve')

fig.colorbar(img1, ax=ax, format="%+2.0f dB")
plt.show()

# %% [5] 3. Mel-Spectrogram 추출 (AI 모델 입력용 핵심 피처)
# 주파수 대역을 비선형적으로 압축하여 CNN 모델에 들어갈 2D 이미지 형태로 변환합니다.
n_mels = 128 

mel_normal = librosa.feature.melspectrogram(y=y_normal, sr=sr, n_mels=n_mels)
mel_db_normal = librosa.power_to_db(mel_normal, ref=np.max)

mel_anomaly = librosa.feature.melspectrogram(y=y_anomaly, sr=sr, n_mels=n_mels)
mel_db_anomaly = librosa.power_to_db(mel_anomaly, ref=np.max)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
img1 = librosa.display.specshow(mel_db_normal, sr=sr, x_axis='time', y_axis='mel', ax=ax[0], cmap='viridis')
ax[0].set_title('Mel-Spectrogram - Normal Valve')

img2 = librosa.display.specshow(mel_db_anomaly, sr=sr, x_axis='time', y_axis='mel', ax=ax[1], cmap='viridis')
ax[1].set_title('Mel-Spectrogram - Abnormal Valve')

fig.colorbar(img1, ax=ax, format="%+2.0f dB")
plt.show()

print(f"변환된 2D Mel-Spectrogram 텐서 형태: {mel_db_normal.shape}")