# filepath: c:\Users\lhw12\Desktop\advanced_power_forecasting.py
"""
전력 수요 예측 모델 비교 분석 - 고급 버전
1D-CNN, LSTM, ARIMA 모델 포함
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import matplotlib.font_manager as fm
import platform

# 운영체제별 한글 폰트 설정
if platform.system() == 'Windows':
    # Windows에서 사용 가능한 한글 폰트들 시도
    font_candidates = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
elif platform.system() == 'Darwin':  # macOS
    font_candidates = ['AppleGothic', 'Helvetica', 'DejaVu Sans']
else:  # Linux
    font_candidates = ['DejaVu Sans', 'Liberation Sans', 'Noto Sans CJK KR']

# 사용 가능한 폰트 찾기
available_fonts = [f.name for f in fm.fontManager.ttflist]
selected_font = 'DejaVu Sans'  # 기본값

for font in font_candidates:
    if font in available_fonts:
        selected_font = font
        break

plt.rcParams['font.family'] = selected_font
plt.rcParams['axes.unicode_minus'] = False

print(f"사용하는 폰트: {selected_font}")

def setup_korean_font():
    """한글 폰트 설정 함수"""
    try:
        # Windows 한글 폰트 경로들
        font_paths = [
            'C:/Windows/Fonts/malgun.ttf',  # 맑은 고딕
            'C:/Windows/Fonts/gulim.ttc',   # 굴림
            'C:/Windows/Fonts/batang.ttc',  # 바탕
            'C:/Windows/Fonts/NanumGothic.ttf'  # 나눔고딕 (설치된 경우)
        ]
        
        for font_path in font_paths:
            try:
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                print(f"한글 폰트 설정 성공: {font_prop.get_name()}")
                return True
            except:
                continue
                
        # 폰트 파일이 없는 경우 시스템 폰트 사용
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['axes.unicode_minus'] = False
        print(f"시스템 폰트 사용: {selected_font}")
        return True
        
    except Exception as e:
        print(f"폰트 설정 오류: {e}")
        # 최후의 수단: 폰트 없이 진행
        plt.rcParams['axes.unicode_minus'] = False
        return False

# 한글 폰트 설정 실행
setup_korean_font()

# TensorFlow/Keras는 설치된 경우에만 import
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow 사용 가능")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow 없음 - 기본 모델만 사용")

# Statsmodels는 설치된 경우에만 import
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
    print("Statsmodels 사용 가능")
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels 없음 - ARIMA 모델 제외")

class PowerDemandForecasting:
    def __init__(self, file_path='household_power_consumption.txt'):
        self.file_path = file_path
        self.df = None
        self.daily_power = None
        self.train_data = None
        self.test_data = None
        self.scaler = MinMaxScaler()
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        print("데이터 로딩 중...")
        
        try:
            # 실제 데이터 로드 시도
            self.df = pd.read_csv(
                self.file_path,
                sep=';',
                parse_dates={'Datetime': ['Date', 'Time']},
                infer_datetime_format=True,
                na_values='?',
                low_memory=False
            )
            
            print(f"원본 데이터 크기: {self.df.shape}")
            
            # Global_active_power를 float으로 변환
            self.df['Global_active_power'] = pd.to_numeric(self.df['Global_active_power'], errors='coerce')
            
            # 결측치 제거
            self.df = self.df.dropna()
            print(f"결측치 제거 후 데이터 크기: {self.df.shape}")
            
            # 시간 인덱스 설정
            self.df.set_index('Datetime', inplace=True)
            
            # 일별 평균 전력 소비량 계산
            self.daily_power = self.df['Global_active_power'].resample('D').mean()
            self.daily_power = self.daily_power.dropna()
            
        except Exception as e:
            print(f"실제 데이터 로드 실패: {e}")
            print("샘플 데이터를 생성합니다...")
            self.daily_power = self.generate_sample_data()
        
        print(f"일별 데이터 크기: {len(self.daily_power)}")
        print(f"데이터 기간: {self.daily_power.index.min()} ~ {self.daily_power.index.max()}")
        
        return self.daily_power
    
    def generate_sample_data(self):
        """샘플 데이터 생성"""
        from datetime import datetime, timedelta
        
        # 2007-2010년 4년간의 일별 데이터 생성
        start_date = datetime(2007, 1, 1)
        end_date = datetime(2010, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 복잡한 시뮬레이션된 전력 소비 패턴
        np.random.seed(42)
        
        # 기본 소비량
        base_consumption = 1.5
        
        # 연간 계절성 (여름/겨울 높음)
        yearly_cycle = 0.3 * np.cos(2 * np.pi * np.arange(len(dates)) / 365.25)
        
        # 주간 계절성 (주말 낮음)
        weekly_cycle = 0.15 * np.cos(2 * np.pi * np.arange(len(dates)) / 7)
        
        # 장기 트렌드 (약간 증가)
        trend = 0.0001 * np.arange(len(dates))
        
        # 무작위 노이즈
        noise = np.random.normal(0, 0.08, len(dates))
        
        # 특별한 이벤트 (불규칙적 피크)
        events = np.random.choice([0, 0.5], size=len(dates), p=[0.95, 0.05])
        
        # 전체 조합
        consumption = (base_consumption + yearly_cycle + weekly_cycle + 
                      trend + noise + events)
        
        # 최소값 보장
        consumption = np.maximum(consumption, 0.1)
        
        daily_power = pd.Series(consumption, index=dates)
        print(f"샘플 데이터 생성 완료: {len(daily_power)}일")
        
        return daily_power
    
    def train_test_split(self, train_ratio=0.8, start_date=None, end_date=None):
        """학습/테스트 데이터 분할"""
        if start_date and end_date:
            # 지정된 기간의 데이터만 사용
            mask = (self.daily_power.index >= start_date) & (self.daily_power.index <= end_date)
            selected_data = self.daily_power[mask]
            print(f"선택된 기간: {start_date} ~ {end_date}")
        else:
            selected_data = self.daily_power
        
        print(f"선택된 데이터 크기: {len(selected_data)}")
        
        # 80:20 비율로 분할
        split_idx = int(len(selected_data) * train_ratio)
        
        self.train_data = selected_data[:split_idx]
        self.test_data = selected_data[split_idx:]
        
        print(f"훈련 데이터: {len(self.train_data)}일")
        print(f"테스트 데이터: {len(self.test_data)}일")
        
        return self.train_data, self.test_data
    
    def create_sequences(self, data, sequence_length=30):
        """시계열 데이터를 시퀀스로 변환 (딥러닝용)"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_dl_data(self, sequence_length=30):
        """딥러닝 모델용 데이터 준비"""
        # 스케일링
        train_scaled = self.scaler.fit_transform(self.train_data.values.reshape(-1, 1)).flatten()
        test_scaled = self.scaler.transform(self.test_data.values.reshape(-1, 1)).flatten()
        
        # 시퀀스 생성
        X_train, y_train = self.create_sequences(train_scaled, sequence_length)
        X_test, y_test = self.create_sequences(test_scaled, sequence_length)
        
        return X_train, y_train, X_test, y_test
    
    def build_1d_cnn_model(self, input_shape):
        """1D-CNN 모델 구축"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_lstm_model(self, input_shape):
        """LSTM 모델 구축"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_1d_cnn(self, sequence_length=30, epochs=50, batch_size=32):
        """1D-CNN 모델 훈련"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow가 없어 1D-CNN 모델을 건너뜁니다.")
            return None, None, None
            
        print("1D-CNN 모델 훈련 중...")
        
        X_train, y_train, X_test, y_test = self.prepare_dl_data(sequence_length)
        
        # 1D-CNN을 위한 데이터 형태 변경
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        model = self.build_1d_cnn_model((sequence_length, 1))
        
        history = model.fit(
            X_train_cnn, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # 예측
        predictions_scaled = model.predict(X_test_cnn, verbose=1)
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
        
        # 실제값 스케일 복원
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        self.models['1D-CNN'] = model
        self.predictions['1D-CNN'] = predictions
        
        return predictions, y_test_original, history
    
    def train_lstm(self, sequence_length=30, epochs=50, batch_size=32):
        """LSTM 모델 훈련"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow가 없어 LSTM 모델을 건너뜁니다.")
            return None, None, None
            
        print("LSTM 모델 훈련 중...")
        
        X_train, y_train, X_test, y_test = self.prepare_dl_data(sequence_length)
        
        model = self.build_lstm_model((sequence_length, 1))
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # 예측
        predictions_scaled = model.predict(X_test, verbose=1)
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
        
        # 실제값 스케일 복원
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        self.models['LSTM'] = model
        self.predictions['LSTM'] = predictions
        
        return predictions, y_test_original, history
    
    def train_arima(self, order=(5, 1, 2)):
        """ARIMA 모델 훈련"""
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels가 없어 ARIMA 모델을 건너뜁니다.")
            return None, None
            
        print("ARIMA 모델 훈련 중...")
        
        try:
            # ARIMA 모델 훈련
            model = ARIMA(self.train_data, order=order)
            fitted_model = model.fit()
            
            # 예측
            forecast_steps = len(self.test_data)
            forecast = fitted_model.forecast(steps=forecast_steps)
            
            self.models['ARIMA'] = fitted_model
            self.predictions['ARIMA'] = forecast.values
            
            return forecast.values, self.test_data.values
            
        except Exception as e:
            print(f"ARIMA 모델 훈련 실패: {e}")
            return None, None
    
    # 기본 모델들 (라이브러리 없이도 실행 가능)
    def train_moving_average(self, window=30):
        """이동평균 모델"""
        print(f"{window}일 이동평균 모델 훈련 중...")
        
        # 마지막 window만큼의 평균으로 예측
        predictions = []
        for i in range(len(self.test_data)):
            if i == 0:
                # 첫 번째 예측은 훈련 데이터의 마지막 window 사용
                pred = self.train_data.tail(window).mean()
            else:
                # 이후 예측은 슬라이딩 윈도우 사용
                if i < window:
                    # 훈련 데이터 + 이전 예측값들
                    recent_data = list(self.train_data.tail(window - i).values) + predictions[:i]
                else:
                    # 이전 예측값들만 사용
                    recent_data = predictions[i-window:i]
                pred = np.mean(recent_data)
            predictions.append(pred)
        
        model_name = f'Moving Average ({window}일)'
        self.predictions[model_name] = np.array(predictions)
        
        return np.array(predictions), self.test_data.values
    
    def train_exponential_smoothing(self, alpha=0.3):
        """지수 평활법"""
        print(f"지수 평활법 (α={alpha}) 훈련 중...")
        
        # 지수 평활 계산
        smoothed = [self.train_data.iloc[0]]
        
        for i in range(1, len(self.train_data)):
            smooth_value = alpha * self.train_data.iloc[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(smooth_value)
        
        # 마지막 평활값으로 예측
        last_smooth = smoothed[-1]
        predictions = np.array([last_smooth] * len(self.test_data))
        
        model_name = f'Exp. Smoothing (α={alpha})'
        self.predictions[model_name] = predictions
        
        return predictions, self.test_data.values
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """평가 지표 계산"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        self.metrics[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }
        
        return self.metrics[model_name]
    
    def compare_models(self):
        """모델 성능 비교"""
        print("\n" + "="*50)
        print("모델 성능 비교 결과")
        print("="*50)
        
        # 메트릭 테이블 생성
        metrics_df = pd.DataFrame(self.metrics).T
        print(metrics_df.round(4))
        
        # 최고 성능 모델
        best_model = metrics_df['RMSE'].idxmin()
        print(f"\n최고 성능 모델 (RMSE 기준): {best_model}")
        print(f"RMSE: {metrics_df.loc[best_model, 'RMSE']:.4f}")
        print(f"R²: {metrics_df.loc[best_model, 'R²']:.4f}")
        
        return metrics_df
    def plot_results(self):
        """결과 시각화"""
        try:
            # 한글 폰트 재설정 (시각화 직전에 다시 확인)
            plt.rcParams['font.family'] = selected_font
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
              # 1. 예측 결과 비교 (상위 4개 모델만)
            ax1 = axes[0, 0]
            ax1.plot(self.test_data.index, self.test_data.values, 
                    label='Actual', linewidth=2, color='black')
            
            # RMSE 기준으로 정렬하여 상위 4개만 표시
            if self.metrics:
                sorted_models = sorted(self.metrics.items(), 
                                     key=lambda x: x[1]['RMSE'])[:4]
                colors = ['red', 'blue', 'green', 'orange']
                
                for i, (model_name, _) in enumerate(sorted_models):
                    if model_name in self.predictions:
                        pred = self.predictions[model_name]
                        # 딥러닝 모델의 경우 시퀀스 길이만큼 조정
                        if model_name in ['1D-CNN', 'LSTM'] and len(pred) < len(self.test_data):
                            test_idx = self.test_data.index[30:]  # 시퀀스 길이 30
                        else:
                            test_idx = self.test_data.index
                        
                        ax1.plot(test_idx, pred, label=model_name, 
                               alpha=0.8, color=colors[i])
            
            ax1.set_title('Model Prediction Comparison (Top 4)')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Power Consumption (kW)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. RMSE 비교
            ax2 = axes[0, 1]
            if self.metrics:
                metrics_df = pd.DataFrame(self.metrics).T
                rmse_values = metrics_df['RMSE']
                bars = ax2.bar(range(len(rmse_values)), rmse_values.values)
                ax2.set_title('RMSE Comparison')
                ax2.set_ylabel('RMSE')
                ax2.set_xticks(range(len(rmse_values)))
                ax2.set_xticklabels(rmse_values.index, rotation=45)
                ax2.grid(True, alpha=0.3)
            
            # 3. R² 비교
            ax3 = axes[1, 0]
            if self.metrics:
                r2_values = metrics_df['R²']
                bars = ax3.bar(range(len(r2_values)), r2_values.values)
                ax3.set_title('R² Score Comparison')
                ax3.set_ylabel('R² Score')
                ax3.set_xticks(range(len(r2_values)))
                ax3.set_xticklabels(r2_values.index, rotation=45)
                ax3.grid(True, alpha=0.3)
            
            # 4. 전체 데이터 시각화
            ax4 = axes[1, 1]
            ax4.plot(self.daily_power.index, self.daily_power.values, alpha=0.7)
            if self.train_data is not None and self.test_data is not None:
                ax4.axvline(x=self.train_data.index[-1], color='red', 
                           linestyle='--', label='Train/Test Split')
            ax4.set_title('Full Time Series Data')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Power Consumption (kW)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('advanced_power_forecasting_results_en.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"시각화 중 오류: {e}")
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("전력 수요 예측 모델 비교 분석 시작")
        print("="*50)
        
        # 1. 데이터 로드 및 전처리
        self.load_and_preprocess_data()
        
        # 2. 훈련/테스트 분할
        self.train_test_split()
        
        # 3. 모델 훈련 및 예측
        
        # 기본 모델들 (항상 실행)
        pred, true = self.train_moving_average(window=7)
        self.calculate_metrics(true, pred, 'Moving Average (7일)')
        
        pred, true = self.train_moving_average(window=30)
        self.calculate_metrics(true, pred, 'Moving Average (30일)')
        
        pred, true = self.train_exponential_smoothing(alpha=0.1)
        self.calculate_metrics(true, pred, 'Exp. Smoothing (α=0.1)')
        
        pred, true = self.train_exponential_smoothing(alpha=0.3)
        self.calculate_metrics(true, pred, 'Exp. Smoothing (α=0.3)')
        
        # 고급 모델들 (라이브러리가 있는 경우에만)
        if STATSMODELS_AVAILABLE:
            try:
                arima_pred, arima_true = self.train_arima()
                if arima_pred is not None:
                    self.calculate_metrics(arima_true, arima_pred, 'ARIMA')
            except Exception as e:
                print(f"ARIMA 모델 실행 실패: {e}")
        
        if TENSORFLOW_AVAILABLE:
            try:
                cnn_pred, cnn_true, _ = self.train_1d_cnn(epochs=20)
                if cnn_pred is not None:
                    self.calculate_metrics(cnn_true, cnn_pred, '1D-CNN')
                    
                lstm_pred, lstm_true, _ = self.train_lstm(epochs=20)
                if lstm_pred is not None:
                    self.calculate_metrics(lstm_true, lstm_pred, 'LSTM')
            except Exception as e:
                print(f"딥러닝 모델 실행 실패: {e}")
        
        # 4. 결과 비교 및 분석
        self.compare_models()
        
        # 5. 시각화
        self.plot_results()
        
        # 6. 분석 결론
        self.print_conclusion()
    
    def print_conclusion(self):
        """분석 결론 출력"""
        print("\n" + "="*50)
        print("분석 결론 및 해석")
        print("="*50)
        
        if not self.metrics:
            print("실행된 모델이 없습니다.")
            return
        
        metrics_df = pd.DataFrame(self.metrics).T
        best_model = metrics_df['RMSE'].idxmin()
        
        print(f"\n1. 최고 성능 모델: {best_model}")
        print(f"   - RMSE: {metrics_df.loc[best_model, 'RMSE']:.4f}")
        print(f"   - R²: {metrics_df.loc[best_model, 'R²']:.4f}")
        print(f"   - MAPE: {metrics_df.loc[best_model, 'MAPE']:.2f}%")
        
        print("\n2. 각 모델의 특징 및 성능 분석:")
        
        # 실행된 모델들에 대해서만 분석
        available_models = list(self.metrics.keys())
        
        if any('Moving Average' in model for model in available_models):
            print("\n   이동평균 모델:")
            print("   - 단순하고 빠른 계산")
            print("   - 최근 데이터의 평균을 이용한 예측")
            print("   - 급격한 변화나 트렌드 포착에 한계")
        
        if any('Exp. Smoothing' in model for model in available_models):
            print("\n   지수 평활법:")
            print("   - 최근 데이터에 더 높은 가중치 부여")
            print("   - α 값으로 반응성 조절 가능")
            print("   - 단순하면서도 효과적인 예측 방법")
        
        if 'ARIMA' in available_models:
            print("\n   ARIMA 모델:")
            print("   - 자기회귀와 이동평균의 조합")
            print("   - 시계열의 정상성과 상관관계 고려")
            print("   - 전통적이고 검증된 시계열 분석 방법")
        
        if '1D-CNN' in available_models:
            print("\n   1D-CNN 모델:")
            print("   - 지역적 패턴 인식에 강함")
            print("   - 컨볼루션을 통한 자동 특징 추출")
            print("   - 비교적 빠른 훈련 속도")
        
        if 'LSTM' in available_models:
            print("\n   LSTM 모델:")
            print("   - 장기 의존성 학습 가능")
            print("   - 복잡한 시계열 패턴 모델링")
            print("   - 비선형 관계 포착 능력")
        
        print("\n3. 성능 차이의 주요 원인:")
        print("   - 데이터의 복잡성과 패턴의 특성")
        print("   - 각 모델의 구조적 특징과 가정")
        print("   - 하이퍼파라미터 최적화 정도")
        print("   - 훈련 데이터의 양과 품질")
        print("   - 예측 대상의 시간 범위")

if __name__ == "__main__":
    # 분석 실행
    forecaster = PowerDemandForecasting()
    forecaster.run_analysis()
