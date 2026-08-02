"""
공기질 데이터 딥러닝 분석 및 예측 모델

이 스크립트는 AirQuality UCI 데이터셋을 활용하여:
1. 상관관계 분석을 통한 변수 그룹화
2. PyTorch GPU 기반 딥러닝 모델 구현 (1D-CNN, LSTM, GRU)  
3. 모델 성능 비교 및 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings('ignore')

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data(file_path):
    """
    AirQuality 데이터를 로드하고 전처리합니다.
    """
    # 데이터 불러오기
    df = pd.read_csv(file_path, sep=';', decimal=',')
    
    # 빈 열 제거
    df = df.dropna(how='all', axis=1)
    
    # 날짜-시간 합치기
    df["Time"] = df["Time"].astype(str).str.replace(".", ":", regex=False)
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors='coerce')
    df = df.dropna(subset=["Datetime"])
    df = df.set_index("Datetime")
    df = df.drop(columns=["Date", "Time"])
    
    # 숫자형 변환 및 결측치 처리
    df = df.apply(pd.to_numeric, errors='coerce')
    df[df == -200] = pd.NA
    df = df.dropna()
    
    return df


def analyze_correlations(df, features_of_interest):
    """
    상관관계 분석 및 시각화
    """
    # 상관관계 매트릭스 계산
    corr_matrix = df[features_of_interest].corr()
    
    # 상관관계 히트맵 시각화
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                square=True, linewidths=0.5)
    plt.title("주요 변수 간 상관관계 매트릭스")
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


def create_sequences(data, seq_length, features_only):
    """
    시계열 예측을 위한 시퀀스 데이터를 생성합니다.
    """
    X, y = [], []
    
    # features_only는 타겟 변수를 제외한 특성들만 포함
    # data는 [특성들..., 타겟] 순서로 구성됨
    for i in range(len(data) - seq_length):
        X.append(features_only[i:i + seq_length])  # 특성들만
        y.append(data[i + seq_length, -1])  # 마지막 컬럼이 타겟
    
    return np.array(X), np.array(y)


def prepare_data(df, features, target_feature, seq_length=24, test_size=0.2):
    """
    모델 학습을 위한 데이터를 준비합니다.
    """
    # 타겟이 특성에 포함되어 있다면 제거
    features_clean = [f for f in features if f != target_feature]
    
    # 필요한 컬럼만 선택 (특성 + 타겟)
    all_features = features_clean + [target_feature]
    data = df[all_features].copy()
    
    # 정규화
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # 특성과 타겟을 따로 정규화
    data_scaled = data.copy()
    data_scaled[features_clean] = scaler_X.fit_transform(data[features_clean])
    data_scaled[target_feature] = scaler_y.fit_transform(data[[target_feature]])    
    # 특성들만 따로 추출
    features_only = data_scaled[features_clean].values
    full_data = data_scaled.values
    
    # 시퀀스 생성
    X, y = create_sequences(full_data, seq_length, features_only)
    
    print(f"생성된 시퀀스 - X shape: {X.shape}, y shape: {y.shape}")
    print(f"특성 개수: {len(features_clean)}, 전체 컬럼: {list(data_scaled.columns)}")
    print(f"사용된 특성: {features_clean}")
    print(f"타겟 변수: {target_feature}")
    
    # 훈련/테스트 분할
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


# 딥러닝 모델 클래스들
class CNN1D(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size*2, hidden_size, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size*2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_length)
        
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.batch_norm3(self.conv3(x)))
        
        x = self.global_pool(x)  # (batch_size, hidden_size, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_size)
        x = self.fc(x)  # (batch_size, 1)
        x = x.squeeze(-1)  # (batch_size,)
        
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 마지막 시간 스텝의 출력 사용
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)  # (batch_size, 1)
        output = output.squeeze(-1)  # (batch_size,)
        return output


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        # 마지막 시간 스텝의 출력 사용
        last_output = gru_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)  # (batch_size, 1)
        output = output.squeeze(-1)  # (batch_size,)
        return output


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """
    모델을 훈련시킵니다.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(num_epochs):
        # 훈련 모드
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, scaler_y):
    """
    모델을 평가합니다.
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    # 정규화 해제
    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    # 메트릭 계산
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mse)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'predictions': predictions,
        'actuals': actuals
    }


def plot_results_comparison(results_high, results_low):
    """
    모델 성능 비교 시각화
    """
    # 성능 메트릭 정리
    model_names = ['1D-CNN', 'LSTM', 'GRU']
    
    # 높은 상관관계 그룹 결과
    high_corr_data = []
    for model in model_names:
        high_corr_data.append([
            results_high[model]['MSE'],
            results_high[model]['MAE'], 
            results_high[model]['RMSE'],
            results_high[model]['R2']
        ])
    
    # 낮은 상관관계 그룹 결과
    low_corr_data = []
    for model in model_names:
        low_corr_data.append([
            results_low[model]['MSE'],
            results_low[model]['MAE'],
            results_low[model]['RMSE'], 
            results_low[model]['R2']
        ])
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('모델 성능 비교 (높은 상관관계 vs 낮은 상관관계 그룹)', fontsize=16)
    
    # MSE 비교
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0,0].bar(x - width/2, [data[0] for data in high_corr_data], width, 
                  label='높은 상관관계', alpha=0.8, color='skyblue')
    axes[0,0].bar(x + width/2, [data[0] for data in low_corr_data], width, 
                  label='낮은 상관관계', alpha=0.8, color='lightcoral')
    axes[0,0].set_title('MSE 비교')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(model_names)
    axes[0,0].legend()
    
    # MAE 비교
    axes[0,1].bar(x - width/2, [data[1] for data in high_corr_data], width, 
                  label='높은 상관관계', alpha=0.8, color='skyblue')
    axes[0,1].bar(x + width/2, [data[1] for data in low_corr_data], width, 
                  label='낮은 상관관계', alpha=0.8, color='lightcoral')
    axes[0,1].set_title('MAE 비교')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(model_names)
    axes[0,1].legend()
    
    # RMSE 비교
    axes[1,0].bar(x - width/2, [data[2] for data in high_corr_data], width, 
                  label='높은 상관관계', alpha=0.8, color='skyblue')
    axes[1,0].bar(x + width/2, [data[2] for data in low_corr_data], width, 
                  label='낮은 상관관계', alpha=0.8, color='lightcoral')
    axes[1,0].set_title('RMSE 비교')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(model_names)
    axes[1,0].legend()
    
    # R2 비교
    axes[1,1].bar(x - width/2, [data[3] for data in high_corr_data], width, 
                  label='높은 상관관계', alpha=0.8, color='skyblue')
    axes[1,1].bar(x + width/2, [data[3] for data in low_corr_data], width, 
                  label='낮은 상관관계', alpha=0.8, color='lightcoral')
    axes[1,1].set_title('R² 비교')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(model_names)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_predictions(results_dict, group_name, n_samples=200):
    """
    예측 결과를 시각화합니다.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{group_name} - 예측 vs 실제 값 비교', fontsize=16)
    
    for i, (model_name, result) in enumerate(results_dict.items()):
        predictions = result['predictions'][:n_samples]
        actuals = result['actuals'][:n_samples]
        
        axes[i].plot(actuals, label='실제 값', alpha=0.7, linewidth=2)
        axes[i].plot(predictions, label='예측 값', alpha=0.7, linewidth=2)
        axes[i].set_title(f'{model_name}\nMSE: {result["MSE"]:.6f}')
        axes[i].set_xlabel('시간')
        axes[i].set_ylabel('C6H6 농도')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_detailed_results(results_high, results_low, high_corr_features, low_corr_features):
    """
    상세한 성능 결과를 출력합니다.
    """
    print("\n" + "="*60)
    print("모델 성능 상세 결과")
    print("="*60)
    
    print("\n[높은 상관관계 변수 그룹]")
    print(f"사용된 변수: {high_corr_features}")
    print("-"*50)
    
    for model_name in ['1D-CNN', 'LSTM', 'GRU']:
        result = results_high[model_name]
        print(f"{model_name:8} - MSE: {result['MSE']:.6f}, MAE: {result['MAE']:.6f}, "
              f"RMSE: {result['RMSE']:.6f}, R²: {result['R2']:.6f}")
    
    print("\n[낮은 상관관계 변수 그룹]")
    print(f"사용된 변수: {low_corr_features}")
    print("-"*50)
    
    for model_name in ['1D-CNN', 'LSTM', 'GRU']:
        result = results_low[model_name]
        print(f"{model_name:8} - MSE: {result['MSE']:.6f}, MAE: {result['MAE']:.6f}, "
              f"RMSE: {result['RMSE']:.6f}, R²: {result['R2']:.6f}")
    
    # 최고 성능 모델 찾기
    print("\n" + "="*60)
    print("최고 성능 모델 (MSE 기준)")
    print("="*60)
    
    # 높은 상관관계 그룹
    best_high = min(results_high.items(), key=lambda x: x[1]['MSE'])
    print(f"높은 상관관계 그룹: {best_high[0]} (MSE: {best_high[1]['MSE']:.6f})")
    
    # 낮은 상관관계 그룹
    best_low = min(results_low.items(), key=lambda x: x[1]['MSE'])
    print(f"낮은 상관관계 그룹: {best_low[0]} (MSE: {best_low[1]['MSE']:.6f})")


def main():
    """
    메인 실행 함수
    """
    print("공기질 데이터 딥러닝 분석 시작...")
    
    # 1. 데이터 로드
    df = load_and_preprocess_data("AirQualityUCI.csv")
    print(f"데이터 크기: {df.shape}")
    
    # 2. 상관관계 분석
    features_of_interest = [
        "CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)",
        "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)",
        "T", "RH", "AH"
    ]
    
    corr_matrix = analyze_correlations(df, features_of_interest)
    
    # 3. 변수 그룹화
    high_corr_features = ["C6H6(GT)", "PT08.S2(NMHC)", "PT08.S5(O3)", "CO(GT)", "PT08.S1(CO)"]
    low_corr_features = ["T", "RH", "AH"]
    
    print("높은 상관관계 변수 그룹:", high_corr_features)
    print("낮은 상관관계 변수 그룹:", low_corr_features)
    
    # 4. 모델 훈련 설정
    seq_length = 24
    target_feature = "C6H6(GT)"
    batch_size = 64
    
    # 5. 높은 상관관계 그룹 모델 훈련
    print("\n=== 높은 상관관계 변수 그룹 모델 훈련 ===")
    
    X_train_high, X_test_high, y_train_high, y_test_high, scaler_X_high, scaler_y_high = prepare_data(
        df, high_corr_features, target_feature, seq_length
    )
    
    # 데이터로더 생성
    train_dataset_high = TensorDataset(
        torch.FloatTensor(X_train_high), 
        torch.FloatTensor(y_train_high)
    )
    test_dataset_high = TensorDataset(
        torch.FloatTensor(X_test_high), 
        torch.FloatTensor(y_test_high)
    )
    
    # 검증 데이터 분할
    val_size = int(0.2 * len(train_dataset_high))
    train_size = len(train_dataset_high) - val_size
    train_dataset_split, val_dataset_high = torch.utils.data.random_split(
        train_dataset_high, [train_size, val_size]
    )
    
    train_loader_split = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    val_loader_high = DataLoader(val_dataset_high, batch_size=batch_size, shuffle=False)
    test_loader_high = DataLoader(test_dataset_high, batch_size=batch_size, shuffle=False)
    
    input_size = X_train_high.shape[2]
    
    # 모델 훈련
    models_high = {}
    results_high = {}
    
    # 1D-CNN
    print("\n1D-CNN 모델 훈련 중...")
    cnn_model_high = CNN1D(input_size).to(device)
    train_model(cnn_model_high, train_loader_split, val_loader_high, num_epochs=30)
    models_high['1D-CNN'] = cnn_model_high
    results_high['1D-CNN'] = evaluate_model(cnn_model_high, test_loader_high, scaler_y_high)
    
    # LSTM
    print("\nLSTM 모델 훈련 중...")
    lstm_model_high = LSTMModel(input_size).to(device)
    train_model(lstm_model_high, train_loader_split, val_loader_high, num_epochs=30)
    models_high['LSTM'] = lstm_model_high
    results_high['LSTM'] = evaluate_model(lstm_model_high, test_loader_high, scaler_y_high)
    
    # GRU
    print("\nGRU 모델 훈련 중...")
    gru_model_high = GRUModel(input_size).to(device)
    train_model(gru_model_high, train_loader_split, val_loader_high, num_epochs=30)
    models_high['GRU'] = gru_model_high
    results_high['GRU'] = evaluate_model(gru_model_high, test_loader_high, scaler_y_high)
    
    # 6. 낮은 상관관계 그룹 모델 훈련
    print("\n=== 낮은 상관관계 변수 그룹 모델 훈련 ===")
    
    X_train_low, X_test_low, y_train_low, y_test_low, scaler_X_low, scaler_y_low = prepare_data(
        df, low_corr_features, target_feature, seq_length
    )
    
    # 데이터로더 생성
    train_dataset_low = TensorDataset(
        torch.FloatTensor(X_train_low), 
        torch.FloatTensor(y_train_low)
    )
    test_dataset_low = TensorDataset(
        torch.FloatTensor(X_test_low), 
        torch.FloatTensor(y_test_low)
    )
    
    # 검증 데이터 분할
    val_size = int(0.2 * len(train_dataset_low))
    train_size = len(train_dataset_low) - val_size
    train_dataset_split_low, val_dataset_low = torch.utils.data.random_split(
        train_dataset_low, [train_size, val_size]
    )
    
    train_loader_split_low = DataLoader(train_dataset_split_low, batch_size=batch_size, shuffle=True)
    val_loader_low = DataLoader(val_dataset_low, batch_size=batch_size, shuffle=False)
    test_loader_low = DataLoader(test_dataset_low, batch_size=batch_size, shuffle=False)
    
    input_size_low = X_train_low.shape[2]
    
    # 모델 훈련
    models_low = {}
    results_low = {}
    
    # 1D-CNN
    print("\n1D-CNN 모델 훈련 중...")
    cnn_model_low = CNN1D(input_size_low).to(device)
    train_model(cnn_model_low, train_loader_split_low, val_loader_low, num_epochs=30)
    models_low['1D-CNN'] = cnn_model_low
    results_low['1D-CNN'] = evaluate_model(cnn_model_low, test_loader_low, scaler_y_low)
    
    # LSTM
    print("\nLSTM 모델 훈련 중...")
    lstm_model_low = LSTMModel(input_size_low).to(device)
    train_model(lstm_model_low, train_loader_split_low, val_loader_low, num_epochs=30)
    models_low['LSTM'] = lstm_model_low
    results_low['LSTM'] = evaluate_model(lstm_model_low, test_loader_low, scaler_y_low)
    
    # GRU
    print("\nGRU 모델 훈련 중...")
    gru_model_low = GRUModel(input_size_low).to(device)
    train_model(gru_model_low, train_loader_split_low, val_loader_low, num_epochs=30)
    models_low['GRU'] = gru_model_low
    results_low['GRU'] = evaluate_model(gru_model_low, test_loader_low, scaler_y_low)
    
    # 7. 결과 분석 및 시각화
    print_detailed_results(results_high, results_low, high_corr_features, low_corr_features)
    
    # 성능 비교 차트
    plot_results_comparison(results_high, results_low)
    
    # 예측 결과 시각화
    plot_predictions(results_high, "높은 상관관계 변수 그룹")
    plot_predictions(results_low, "낮은 상관관계 변수 그룹")
    
    print("\n분석 완료!")


if __name__ == "__main__":
    main()
