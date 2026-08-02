import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from step1_data_eda import MVTecDataset
from step2_train import ConvAutoencoder


def get_anomaly_score(error_map, top_percent=1.0):
    """오차맵 상위 퍼센트 픽셀 평균으로 이미지 레벨 이상치 점수를 계산합니다."""
    flat = error_map.flatten()
    k = max(1, int(len(flat) * (top_percent / 100.0)))
    topk_vals = np.partition(flat, -k)[-k:]
    return float(np.mean(topk_vals))

def evaluate_performance(model, test_loader, device):
    """테스트 데이터셋 전체를 평가하여 정량적 지표를 산출합니다."""
    model.eval()
    y_true = []
    y_scores = []
    
    print("전체 테스트 데이터셋 정량 평가를 진행합니다...")
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # 1. 픽셀 단위 오차 계산 및 노이즈 제거 (가우시안 블러)
            error = torch.mean((images - outputs) ** 2, dim=1) 
            error_map = error.squeeze().cpu().numpy()
            error_map = cv2.GaussianBlur(error_map, (7, 7), 0)
            
            # 2. 이미지 레벨 이상치 점수(Anomaly Score) 산출
            # 제조품은 아주 작은 결함 하나만 있어도 전체가 불량입니다. 
            # 따라서 오차 맵에서 '가장 오차가 큰 픽셀의 값(Max)'을 해당 이미지의 대표 불량 점수로 사용합니다.
            anomaly_score = get_anomaly_score(error_map, top_percent=1.0)
            
            y_scores.append(anomaly_score)
            y_true.append(labels.item()) # 0: 정상, 1: 불량

    # 3. 정량적 지표 계산
    # AUROC: 임계값에 상관없이 모델의 전반적인 정상/불량 분류 능력을 평가
    auroc = roc_auc_score(y_true, y_scores)
    
    # Precision-Recall 기반 최적 임계값 및 F1-Score 탐색
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # F1-Score 계산 (0으로 나누어지는 것 방지)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

    # precision_recall_curve 결과에서 thresholds 길이는 precisions/recalls보다 1 작습니다.
    if len(thresholds) > 0:
        valid_f1 = f1_scores[:-1]
        best_idx = int(np.argmax(valid_f1))
        best_f1 = float(valid_f1[best_idx])
        best_threshold = float(thresholds[best_idx])
    else:
        best_f1 = float(f1_scores[0])
        best_threshold = 0.0
    
    print("-" * 40)
    print(f"[전체 평가 결과]")
    print(f"AUROC Score          : {auroc:.4f}")
    print(f"Best F1-Score        : {best_f1:.4f}")
    print(f"Optimal Threshold    : {best_threshold:.4f}")
    print("-" * 40)
    
    return best_threshold

def visualize_anomaly(model, test_loader, device, threshold, num_samples=3):
    """결함 탐지 시각화 및 판정 결과를 출력합니다."""
    model.eval()
    samples_shown = 0
    
    print(f"\n최적 임계값({threshold:.4f})을 적용하여 시각화를 시작합니다.")
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            if labels.item() == 0: 
                continue
                
            images = images.to(device)
            outputs = model(images)
            
            error = torch.mean((images - outputs) ** 2, dim=1) 
            error_map = error.squeeze().cpu().numpy()
            error_map = cv2.GaussianBlur(error_map, (7, 7), 0)
            
            anomaly_score = get_anomaly_score(error_map, top_percent=1.0)
            
            # 산출된 Threshold를 바탕으로 불량(NG) / 정상(OK) 판정
            prediction = "NG (Defect)" if anomaly_score >= threshold else "OK (Normal)"
            
            error_map_norm = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(error_map_norm, cv2.COLORMAP_JET)
            
            img_np = images.squeeze().cpu().permute(1, 2, 0).numpy()
            out_np = outputs.squeeze().cpu().permute(1, 2, 0).numpy()
            
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted((img_np * 255).astype(np.uint8), 0.5, heatmap_rgb, 0.5, 0)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            # 타이틀에 예측 판정 결과 및 점수 표시
            axes[0].imshow(img_np); axes[0].set_title(f'Original\nScore: {anomaly_score:.4f} -> {prediction}')
            axes[1].imshow(out_np); axes[1].set_title('Reconstructed')
            axes[2].imshow(error_map, cmap='hot'); axes[2].set_title('Error Map')
            axes[3].imshow(overlay); axes[3].set_title('Overlay Heatmap')
            
            for ax in axes:
                ax.axis('off')
            plt.show()
            
            samples_shown += 1
            if samples_shown >= num_samples:
                break

def display_misclassified_samples(model, test_loader, device, threshold):
    """
    임계값을 기준으로 잘못 분류된 샘플을 모두 출력합니다.
    """
    model.eval()
    print("\n잘못 분류된 샘플을 출력합니다...")

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images)

            # 오차 맵 계산 및 이상치 점수 산출
            error = torch.mean((images - outputs) ** 2, dim=1)
            error_map = error.squeeze().cpu().numpy()
            error_map = cv2.GaussianBlur(error_map, (7, 7), 0)
            anomaly_score = get_anomaly_score(error_map, top_percent=1.0)

            # 판정 결과
            prediction = 1 if anomaly_score >= threshold else 0

            # 잘못 분류된 경우만 출력
            if prediction != labels.item():
                print(f"파일 경로: {paths[0]}")
                print(f"실제 라벨: {'정상' if labels.item() == 0 else '불량'}")
                print(f"예측 결과: {'정상' if prediction == 0 else '불량'}")
                print(f"이상치 점수: {anomaly_score:.4f}\n")

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mvtec_ad'))
    CATEGORY = 'bottle'
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'autoencoder_model.pth'))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    test_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # 1. 정량 평가 수행 및 최적 임계값 도출
    optimal_thresh = evaluate_performance(model, test_loader, device)
    
    # 2. 도출된 임계값을 시각화 함수에 전달하여 실제 판정 시뮬레이션
    visualize_anomaly(model, test_loader, device, optimal_thresh, num_samples=3)
    
    # 3. 임계값을 기준으로 잘못 분류된 샘플을 출력
    display_misclassified_samples(model, test_loader, device, optimal_thresh)