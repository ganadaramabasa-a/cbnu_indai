# cbnu_indai

충북대학교 산업인공지능(Industrial AI) 수업의 실습·과제·프로젝트를 모아 둔 저장소입니다.
머신러닝 기초부터 제조 데이터 분석, 이미지·음향 이상 탐지, 예지보전, 생산 스케줄링 최적화, RAG 문서 검색, 애플리케이션 개발까지 다룹니다.

## 목차

- [폴더 구성](#폴더-구성)
- [머신러닝·딥러닝 기초](#머신러닝딥러닝-기초)
- [이상 탐지 (Anomaly Detection)](#이상-탐지-anomaly-detection)
- [예지보전·제조 데이터](#예지보전제조-데이터)
- [시계열 예측](#시계열-예측)
- [애플리케이션](#애플리케이션)
- [실행 시 참고사항](#실행-시-참고사항)

## 폴더 구성

| 폴더 | 주제 | 주요 기술 |
|---|---|---|
| [`titanic/`](titanic) | Kaggle 타이타닉 생존 예측 | pandas, scikit-learn |
| [`regression&classicification/`](regression&classicification) | 선형/로지스틱 회귀, MLP 분류 | TensorFlow/Keras |
| [`loss, vanishing, overfitting, etc/`](loss,%20vanishing,%20overfitting,%20etc) | 손실함수, 기울기 소실, 과적합, 정규화 | TensorFlow/Keras |
| [`card_pattern/`](card_pattern) | 신용카드 이상 결제 탐지 | scikit-learn, PyTorch |
| [`MVTec/`](MVTec) | 이미지 이상 탐지(오토인코더), 불량 분류 CNN | PyTorch |
| [`anomalib/`](anomalib) | anomalib 라이브러리 실습 (PatchCore, EfficientAd) | anomalib |
| [`mimii/`](mimii) | 설비 음향 이상 탐지 | PyTorch, librosa |
| [`analyze_mlp_model/`](analyze_mlp_model) | 설비 고장 예측 MLP, 용접 LSTM-AE | PyTorch |
| [`dataset_cleaning/`](dataset_cleaning) | SECOM 반도체 데이터 정제 | pandas, scikit-learn |
| [`manufacturing-data-optimization/`](manufacturing-data-optimization) | 작업 이상 예측 + 위험 기반 스케줄링 | scikit-learn, PuLP |
| [`air_quality/`](air_quality) | 대기질(벤젠 농도) 예측 | PyTorch |
| [`elec_power_pred/`](elec_power_pred) | 가정용 전력 수요 예측 | statsmodels, TensorFlow/Keras |
| [`capstone_project/`](capstone_project) | 사내 문서 RAG 검색·질의응답 | PostgreSQL + pgvector, LiteLLM |
| [`project_bank/`](project_bank) | 은행 입출금 웹앱 | Flask |
| [`React/reactgrid-study/`](React/reactgrid-study) | ReactGrid 학습 | React, TypeScript, Vite |
| [`SinglePrompter/`](SinglePrompter) | 음성 인식 프롬프터 안드로이드 앱 | Kotlin, Jetpack Compose |

---

## 머신러닝·딥러닝 기초

### `titanic/`
Kaggle Titanic 데이터로 EDA, 결측치 처리(Age·Fare 중앙값, Cabin `U`, Embarked 최빈값), 파생변수(Title, FamilySize)를 만든 뒤
KNN·Decision Tree·SVM·Random Forest를 StratifiedKFold 교차검증으로 비교합니다. 최종 제출은 SVC로 Kaggle 점수 **0.77033**입니다.

### `regression&classicification/`
TensorFlow/Keras로 회귀·분류 기초를 실습합니다.

| 노트북 | 내용 |
|---|---|
| `linear_regression.ipynb` | 합성 데이터(y = 3x + 2 + noise) 선형 회귀 |
| `linear_regression_california(feat1).ipynb` | California Housing, MedInc 단일 특성 회귀 |
| `linear_regression_california(feat8).ipynb` | California Housing, 8개 특성 + 표준화 |
| `logistic_regression.ipynb` | Iris 이진 분류와 결정 경계 |
| `logistic_regression_gd_moons.ipynb` | Batch / Stochastic / Mini-batch 경사하강법 비교 |
| `diabetes_nn_regression.ipynb` | Diabetes 데이터 MLP 회귀 |
| `mnist_nn_classification.ipynb` | MNIST MLP 분류 |

### `loss, vanishing, overfitting, etc/`
신경망 학습에서 자주 마주치는 문제를 하나씩 실험으로 확인합니다 (TensorFlow/Keras).

| 노트북 | 내용 |
|---|---|
| `1-1`, `1-2` | MSE / MAE / Huber 손실 비교 (California Housing), 미니배치 단위 수렴 속도 |
| `2`, `2-1` | 이진·다중 클래스·다중 레이블 분류, MNIST 다중 분류 |
| `3.vanishing_gradient` | 깊은 Sigmoid 네트워크의 기울기 소실과 ReLU 비교 |
| `4-1`, `4-2` | 데이터 크기·모델 복잡도에 따른 과적합/과소적합 |
| `5.class_imbalanced` | 90:10 불균형 데이터에서 정확도의 함정 |
| `6.residual_vs_plain` | Residual Block(스킵 연결)과 일반 깊은 네트워크 비교 |
| `7.l1_l2_regularization` | L1 / L2 / L1+L2 정규화 비교 |
| `8.dropout` | Dropout 적용 여부 비교 |
| `9.data_aug_class_weight` | CIFAR-10 고양이/개 불균형 데이터에서 소수 클래스 증강과 class_weight 비교 |

### `card_pattern/` — 신용카드 이상 결제 탐지
Kaggle Credit Card Fraud 데이터(284,807건, 사기 0.172%)로 여러 방법을 비교한 패턴인식 프로젝트입니다.

- **지도학습**: Logistic Regression, Random Forest (`class_weight='balanced'`)
- **전통적 이상 탐지**: One-Class SVM, Isolation Forest, LOF, DBSCAN
- **딥러닝(PyTorch)**: Autoencoder, VAE, GAN, LSTM-AE, RNN-AE. 정상 거래만으로 학습하고 재구성 오차 95 백분위수를 임계값으로 사용
- **결과**(발표 자료 기준): 최종 모델은 Random Forest (F1 0.84, AP 0.85)

`creditcard.csv`는 용량 때문에 저장소에 없으므로 [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)에서 받아야 합니다. 제안·완료 발표 자료(`.pptx`)가 함께 들어 있습니다.

---

## 이상 탐지 (Anomaly Detection)

### `MVTec/`

| 하위 폴더 | 내용 |
|---|---|
| `mvtec/` | MVTec AD `bottle` 데이터로 비지도 이미지 이상 탐지 |
| `hw_6th_2025254004_이한울/` | 같은 과제의 6주차 제출본 (단순 오토인코더) |
| `cromate/` | 정상/불량 이미지 지도학습 CNN 분류 + Grad-CAM |

**`mvtec/`** — 정상 이미지로만 학습한 오토인코더의 재구성 오차로 결함을 찾습니다.
- 데이터: `mvtec_ad/bottle` (train/good 209장, test 83장: good, broken_large, broken_small, contamination)
- 모델: 스킵 연결이 있는 U-Net형 Conv Autoencoder (256×256 입력), 디노이징 학습, 손실 0.6·L1 + 0.4·MSE, AdamW, Early Stopping
- 판정: 픽셀 MSE 맵을 가우시안 블러한 뒤 상위 1% 픽셀 평균을 이미지 점수로 사용. AUROC를 계산하고, PR 곡선에서 F1이 가장 높은 임계값을 선택

```bash
cd MVTec/mvtec
python step1_data_eda.py
python step2_train.py      # autoencoder_model.pth 생성
python step3_evaluate.py   # AUROC, F1, 재구성/오차 맵 시각화
```

**`hw_6th_.../`** — 스킵 연결이 없는 5단 Conv Autoencoder로 MSE만 사용해 200 에폭 학습합니다. 점수 방식(`max`, `p99_5`, `topk_mean_1pct`)을 골라 비교할 수 있습니다. 상위 폴더의 `autoencoder_model1~5.pth`가 이 구조로 학습한 가중치입니다.

**`cromate/`** — 2-layer CNN으로 이미지를 정상/불량으로 분류하고(`step1`~`step4`), 추론 시 Grad-CAM 히트맵을 겹쳐 보여 줍니다. 이미지 데이터(`./data/학습`, `./data/테스트`)는 저장소에 없습니다. 함께 있는 KAMP 센서 CSV(`kemp-abh-sensor-*.csv`, `Error Lot list.csv`)는 현재 어떤 스크립트에서도 사용하지 않습니다.

> MVTec AD 데이터셋은 [CC BY-NC-SA 4.0](MVTec/mvtec/mvtec_ad/license.txt) 라이선스이므로 비상업적 용도로만 사용할 수 있습니다.

### `anomalib/`
[anomalib](https://github.com/openvinotoolkit/anomalib) v2 API 실습입니다.

| 파일 | 내용 |
|---|---|
| `train.py` | MVTec AD `bottle` 자동 다운로드 후 EfficientAd 학습·테스트·내보내기 (PatchCore는 주석 처리) |
| `custom_dataset.py` | `Folder` 데이터모듈로 사용자 데이터셋(`hazelnut_toy`)에 PatchCore 학습 |
| `inference.py` | PatchCore 체크포인트로 단일 이미지 추론 |
| `visualization.py`, `visualization_callback.py` | 이상 맵과 예측 마스크, GT 마스크 시각화 |
| `main.py` | anomalib과 무관한 GPU/CPU 행렬곱 속도 테스트 |

`inference.py`와 시각화 스크립트는 `results/Patchcore/MVTecAD/bottle/latest/...` 체크포인트를 읽으므로, `train.py`의 모델을 PatchCore로 바꿔 먼저 학습해야 합니다.

### `mimii/` — 설비 음향 이상 탐지
MIMII 데이터셋(valve, id_02, 0dB)의 정상 소리로만 Mel-spectrogram 오토인코더를 학습하고, 재구성 오차로 이상 소리를 탐지합니다.

- 데이터: `0_dB_valve/valve/id_02/` (normal 708개, abnormal 120개 wav)
- 전처리: 16 kHz, 512 mel × 512 frame, dB를 [-80, 0]으로 자른 뒤 전역 평균/표준편차로 정규화
- 모델: 5단 Conv 인코더 → 32차원 잠재 벡터 → 대칭 ConvTranspose 디코더, MSE 손실

| 스크립트 | 내용 |
|---|---|
| `step1_eda_audio.py` | 파형, STFT, Mel-spectrogram 시각화 및 재생 (`sounddevice` 필요) |
| `step2_train_audio_ae.py` | 학습 → `audio_models/audio_autoencoder_real.pth`, `audio_mel_stats.npz` |
| `step3_eval_audio_ae.py` | 오차 분포, ROC-AUC, 최적 F1 임계값, 혼동행렬 |
| `step4_inference_audio_ae.py` | 2초 슬라이딩 윈도우 실시간 모니터링 시뮬레이션 |

`mimii/` 폴더 안에서 순서대로 실행합니다. 스크립트는 VS Code Interactive(`# %%` 셀) 형식입니다. `KAMP_sound/`(팬 소음 OK/Error)와 `dtc_sound_model.pkl`은 현재 스크립트에서 사용하지 않습니다.

---

## 예지보전·제조 데이터

### `analyze_mlp_model/`

**`ai4i_2020_predictive_maintenance/`** — UCI AI4I 2020 데이터(10,000행, 고장 약 3.4%)로 설비 고장을 이진 분류합니다.
- 누수를 막기 위해 고장 유형 컬럼(TWF/HDF/PWF/OSF/RNF)을 제거하고, 온도차·토크×속도 등 파생 특성 5개를 추가
- MLP(128-64-32-1, BatchNorm, Dropout), `BCEWithLogitsLoss(pos_weight)`로 불균형 보정
- 검증 F1이 최대인 결정 임계값을 탐색해 `models/decision_threshold.txt`에 저장, TensorBoard 로깅

```bash
cd analyze_mlp_model/ai4i_2020_predictive_maintenance
python step1_eda.py
python step3_train_model.py   # step2_data_prep.py를 import하며 UCI에서 CSV를 내려받음 (인터넷 필요)
python step4_inference.py     # 샘플 1건 고장 여부 추론
```

**`welding_predictive_maintenance/`** — 용접 센서 시계열 5채널(`DV_R`, `DA_R`, `AV_R`, `AA_R`, `PM_R`)을 길이 20 윈도우로 잘라
LSTM Autoencoder를 정상 데이터로만 학습합니다. 검증 데이터에서 Precision과 Recall이 같아지는 지점을 임계값으로 정해 이상을 판정합니다. 폴더 안에서 `step1` → `step2` → `step3` 순서로 실행합니다.

### `dataset_cleaning/` — SECOM 반도체 공정 데이터 정제
UCI SECOM(1,567행 × 592열) 데이터를 데이터 품질 관점에서 정제합니다.

- `clean_dataset.py`: 유일성(중복 Time), 완전성(결측 50% 초과 열 / 20% 초과 행 제거), 유효성(레이블·시간 형식), 일관성(분산 0 열 제거), 정확성(|z| > 5를 결측 처리 후 중앙값 대치). 결과는 `uci-secom_clean.csv`
- `개발 코드.py`: 결측 40% 이상 열 제거, KNNImputer(k=5), 1.5×IQR 클리핑. 결과는 `uci-secom_cleaned.csv` (1,534행 × 444열, 저장소에 포함)
- `결과보고서.pptx`: 결과 보고서

### `manufacturing-data-optimization/` — 작업 이상 예측 + 위험 기반 스케줄링
하이브리드 제조 작업 데이터(1,000건, 설비 M01~M05)에서 지연·실패 작업을 예측하고, 예측된 위험도를 반영해 작업 순서를 최적화하는 텀프로젝트입니다.

1. **예측**: 전처리 파이프라인 + RandomForest(`class_weight='balanced'`)로 `is_abnormal`(Delayed/Failed) 분류
2. **스케줄링**: 설비별 투입 시각 순서(Baseline)와, 위험도·처리시간·에너지·가용성을 가중합한 우선순위 휴리스틱을 비교. PuLP 기반 MIP 스케줄러(`schedule_with_mip`)도 구현되어 있으나 `main()`에서는 호출하지 않음
3. **KPI**: 위험 노출, 평균 리드타임, 에너지 비용 대리지표

세 버전은 우선순위 가중치만 다릅니다. Baseline 대비 전체 KPI 감소율(`artifacts/vN/quantitative_effect_summary.csv`)은 다음과 같습니다.

| | v1 | v2 | v3 |
|---|---:|---:|---:|
| 위험 노출 | 92.66% | 93.02% | 92.05% |
| 리드타임 | 14.98% | **21.72%** | 15.18% |
| 에너지 | −11.95% | 3.77% | **43.43%** |

자세한 비교는 [`v1_v2_v3_개선분석.md`](manufacturing-data-optimization/v1_v2_v3_개선분석.md)에 있습니다.

```bash
cd manufacturing-data-optimization
pip install -r requirements.txt
python code-v2.py   # 결과는 artifacts/v2/ 에 저장
```

> 분류 성능이 모든 지표에서 1.0으로 나오는데, 실제 시작/종료 시각에서 만든 지연 특성이 `Job_Status`를 사실상 담고 있어 데이터 누수일 가능성이 큽니다.

---

## 시계열 예측

### `air_quality/` — 벤젠(C6H6) 농도 예측
UCI Air Quality 데이터로 1D-CNN, LSTM, GRU를 비교합니다. 상관관계가 높은 센서 그룹(PT08.S2, PT08.S5, CO, PT08.S1)과 낮은 그룹(T, RH, AH)을 각각 입력으로 넣어
24-step 윈도우로 학습하고 MSE/MAE/RMSE/R²를 비교합니다. 노트북에서는 전처리와 상관관계 시각화를 합니다.

```bash
cd air_quality
python air_quality_deep_learning.py
```

### `elec_power_pred/` — 가정용 일별 전력 수요 예측
이동평균, 지수평활, ARIMA(5,1,2), 1D-CNN, LSTM을 비교하고 RMSE 기준으로 최적 모델을 고릅니다.
UCI Individual Household Electric Power Consumption(`household_power_consumption.txt`)이 필요하며, 파일이 없으면 2007–2010년 **합성 데이터**로 대체해 실행됩니다.

---

## 애플리케이션

### `capstone_project/` — 사내 문서 RAG 검색 시스템
공유 폴더의 문서를 파싱·청킹·임베딩해 PostgreSQL(pgvector)에 저장하고, 하이브리드 검색과 LLM으로 질의응답하는 캡스톤 프로젝트입니다.

- **수집** (`main.py` → `pipeline.py`): 파일 변경 감지(mtime/size → MD5), 처리 상태 추적(PROCESSING/COMPLETED/FAILED) 및 재시도
- **파싱** (`parsers.py`): PDF(pdfplumber, 슬라이드/스캔 페이지는 VLM으로 요약·OCR), PPTX, DOCX, XLSX, 그 외는 MarkItDown
- **청킹** (`chunker.py`): 부모 2,500자 / 자식 350자의 Parent-Child 구조
- **키워드 색인** (`nlp_utils.py`): Kiwi 형태소 분석으로 명사·영단어·숫자 추출 → PostgreSQL 전문 검색
- **검색·대화** (`chat/chat.py`): 벡터 유사도 0.75 + 전문 검색 0.25 하이브리드 점수, 출처(파일·페이지) 인용, 대화 이력 저장
- **평가** (`evaluator.py`, `test_evaluator.py`): LLM-as-a-judge 응답 정확도, HitRate@K, 범위 밖 질문 거절률

```bash
cd capstone_project
pip install -r requirements.txt
python main.py                                 # 문서 수집·색인
python chat/chat.py "질문" --mode auto --top-k 8   # 질의응답
python test_evaluator.py                       # 평가 (실제 LLM 호출은 --real)
```

실행 전에 `config.py`의 DB 접속 정보와 `SHARED_FOLDER`를 채우고, pgvector 확장과 테이블을 만들어 두어야 합니다. LLM과 임베딩은 LiteLLM 게이트웨이(OpenAI 호환 API)로 호출합니다.

### `project_bank/` — Flask 은행 웹앱
로그인/로그아웃(세션 30분), 잔액 조회, 입금·출금(유효성 검사 포함), 거래 내역을 제공하는 Flask 실습 앱입니다. 데이터는 메모리에만 저장됩니다.

```bash
cd project_bank
pip install flask
python app.py   # http://127.0.0.1:5000
```

### `React/reactgrid-study/` — ReactGrid 학습
Vite + React 18 + TypeScript 환경에서 [`@silevis/reactgrid`](https://reactgrid.com/)로 편집 가능한 표를 만드는 예제입니다.

```bash
cd React/reactgrid-study
npm install
npm run dev
```

### `SinglePrompter/` — 음성 인식 프롬프터 (Android)
대본을 붙여 넣고 읽으면 안드로이드 음성 인식(`SpeechRecognizer`, ko-KR)이 말한 내용을 대본과 맞춰 보고 자동으로 다음 줄로 넘어가는 프롬프터 앱입니다.
읽은 단어를 노래방처럼 강조하고, 이전/다음 버튼으로 수동 보정할 수 있으며, 가로/세로 레이아웃을 지원합니다.

- Kotlin 2.0, Jetpack Compose (Material3), minSdk 26 / targetSdk 35, JDK 17
- Android Studio에서 폴더를 열어 실행하거나 `gradlew assembleDebug`로 빌드
- 자세한 내용은 [`SinglePrompter/README.md`](SinglePrompter/README.md) 참고

---

## 실행 시 참고사항

- **Python 환경**: 대부분 Python 3.10 기준입니다. 저장소 전체를 아우르는 `requirements.txt`는 없으므로 프로젝트별로 필요한 패키지를 설치하세요.
  주로 쓰는 패키지는 `torch`, `torchvision`, `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `librosa`, `anomalib`입니다.
- **작업 디렉터리**: 대부분의 스크립트가 상대 경로로 데이터를 읽으므로 **해당 프로젝트 폴더로 이동한 뒤** 실행하세요.
- **한글 폰트**: 시각화 코드 일부가 `Malgun Gothic` 폰트를 사용하므로 Windows가 아닌 환경에서는 폰트 설정을 바꿔야 합니다.
- **저장소에 없는 데이터**: Credit Card Fraud(`card_pattern`), `cromate` 이미지, `anomalib`의 `hazelnut_toy`, 가정용 전력 데이터(`elec_power_pred`)는 직접 준비해야 합니다.
- **알려진 문제**
  - `MVTec/hw_6th_.../step3_evaluate.py`의 `MODEL_PATH`가 개인 PC의 절대 경로로 되어 있습니다.
  - `mimii/step4_inference_audio_ae.py`의 모델 정의(256×8×8)가 학습 모델(512 입력)과 달라 체크포인트 로드 시 크기 불일치가 납니다. `step3`의 재구성 시각화도 입력 크기가 256×256으로 되어 있어 같은 문제가 있습니다.
  - `mimii/step3`은 학습에 쓴 정상 파일 일부로 평가하므로 성능이 실제보다 높게 나올 수 있습니다.
