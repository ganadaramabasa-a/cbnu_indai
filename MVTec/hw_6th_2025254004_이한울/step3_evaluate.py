import random

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from step1_data_eda import MVTecDataset
from step2_train import ConvAutoencoder

def compute_anomaly_score(error_map, method="p99_5"):
    """오차 맵을 이미지 단위 이상치 점수로 변환합니다."""
    if method == "max":
        return float(np.max(error_map))
    if method == "p99_5":
        return float(np.percentile(error_map, 99.5))
    if method == "topk_mean_1pct":
        flat = error_map.reshape(-1)
        k = max(1, int(flat.size * 0.01))
        topk = np.partition(flat, -k)[-k:]
        return float(np.mean(topk))
    raise ValueError(f"Unknown score method: {method}")


def evaluate_performance(model, test_loader, device, score_method="p99_5"):
    """테스트 데이터셋 전체를 평가하여 정량적 지표를 산출합니다."""
    model.eval()
    y_true = []
    y_scores = []
    
    print("전체 테스트 데이터셋 정량 평가를 진행합니다...")
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)

            # 1. 픽셀 단위 오차 계산
            error = torch.mean((images - outputs) ** 2, dim=1)
            error_maps = error.detach().cpu().numpy()  # (B, H, W)
            labels_np = labels.detach().cpu().numpy()  # (B,)

            # 2. 샘플별 이상치 점수(Anomaly Score) 산출
            for error_map, label in zip(error_maps, labels_np):
                error_map = cv2.GaussianBlur(error_map, (15, 15), 0)
                anomaly_score = compute_anomaly_score(error_map, method=score_method)
                y_scores.append(anomaly_score)
                y_true.append(int(label))  # 0: 정상, 1: 불량

    # 3. 정량적 지표 계산
    # AUROC: 임계값에 상관없이 모델의 전반적인 정상/불량 분류 능력을 평가
    auroc = roc_auc_score(y_true, y_scores)
    
    # Precision-Recall 기반 최적 임계값 및 F1-Score 탐색
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # F1-Score 계산 (0으로 나누어지는 것 방지)
    # thresholds 길이는 precisions/recalls보다 1 작으므로 인덱스를 맞춰 계산합니다.
    if len(thresholds) == 0:
        best_threshold = float(np.mean(y_scores))
        best_f1 = 0.0
    else:
        f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = int(np.argmax(f1_scores))
        best_f1 = float(f1_scores[best_idx])
        best_threshold = float(thresholds[best_idx])
    
    print("-" * 40)
    print(f"[전체 평가 결과]")
    print(f"Score Method         : {score_method}")
    print(f"AUROC Score          : {auroc:.4f}")
    print(f"Best F1-Score        : {best_f1:.4f}")
    print(f"Optimal Threshold    : {best_threshold:.4f}")
    print("-" * 40)
    
    return best_threshold

def visualize_anomaly(model, test_loader, device, threshold, num_samples=3, score_method="p99_5"):
    """결함 탐지 시각화 및 판정 결과를 출력합니다."""
    model.eval()
    samples_shown = 0
    
    print(f"\n최적 임계값({threshold:.4f})을 적용하여 시각화를 시작합니다.")
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)

            error = torch.mean((images - outputs) ** 2, dim=1)
            error_maps = error.detach().cpu().numpy()  # (B, H, W)
            labels_np = labels.detach().cpu().numpy()  # (B,)
            images_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, C)
            outputs_np = outputs.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, C)

            for img_np, out_np, error_map, label in zip(images_np, outputs_np, error_maps, labels_np):
                # 불량 샘플만 시각화
                if int(label) == 0:
                    continue

                error_map = cv2.GaussianBlur(error_map, (15, 15), 0)
                anomaly_score = compute_anomaly_score(error_map, method=score_method)

                # 산출된 Threshold를 바탕으로 불량(NG) / 정상(OK) 판정
                prediction = "NG (Defect)" if anomaly_score >= threshold else "OK (Normal)"

                error_map_norm = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmap = cv2.applyColorMap(error_map_norm, cv2.COLORMAP_JET)

                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted((img_np * 255).astype(np.uint8), 0.5, heatmap_rgb, 0.5, 0)

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                # 타이틀에 예측 판정 결과 및 점수 표시
                axes[0].imshow(img_np)
                axes[0].set_title(f'Original\nScore: {anomaly_score:.4f} -> {prediction}')
                axes[1].imshow(out_np)
                axes[1].set_title('Reconstructed')
                axes[2].imshow(error_map, cmap='hot')
                axes[2].set_title('Error Map')
                axes[3].imshow(overlay)
                axes[3].set_title('Overlay Heatmap')

                for ax in axes:
                    ax.axis('off')
                plt.show()

                samples_shown += 1
                if samples_shown >= num_samples:
                    return

def display_misclassified_samples(model, test_loader, device, threshold, score_method="p99_5"):
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
            error_maps = error.detach().cpu().numpy()  # (B, H, W)
            labels_np = labels.detach().cpu().numpy()  # (B,)

            for error_map, label, path in zip(error_maps, labels_np, paths):
                error_map = cv2.GaussianBlur(error_map, (15, 15), 0)
                anomaly_score = compute_anomaly_score(error_map, method=score_method)

                # 판정 결과
                prediction = 1 if anomaly_score >= threshold else 0

                # 잘못 분류된 경우만 출력
                if prediction != int(label):
                    print(f"파일 경로: {path}")
                    print(f"실제 라벨: {'정상' if int(label) == 0 else '불량'}")
                    print(f"예측 결과: {'정상' if prediction == 0 else '불량'}")
                    print(f"이상치 점수: {anomaly_score:.4f}\n")

if __name__ == "__main__":
    ROOT_DIR = './mvtec_ad'
    CATEGORY = 'bottle'
    MODEL_PATH = r'C:\Users\lhw12\OneDrive\Documents\충북대\2학년 1학기\제조 데이터 분석과 최적화\6주차 과제\autoencoder_model5.pth'
    SCORE_METHOD = "p99_5"  # options: "max", "p99_5", "topk_mean_1pct"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 랜덤 시드 고정
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # 1. 정량 평가 수행 및 최적 임계값 도출
    optimal_thresh = evaluate_performance(model, test_loader, device, score_method=SCORE_METHOD)
    
    # 2. 도출된 임계값을 시각화 함수에 전달하여 실제 판정 시뮬레이션
    # visualize_anomaly(model, test_loader, device, optimal_thresh, num_samples=3, score_method=SCORE_METHOD)
    
    # 3. 임계값을 기준으로 잘못 분류된 샘플을 출력
    # display_misclassified_samples(model, test_loader, device, optimal_thresh, score_method=SCORE_METHOD)