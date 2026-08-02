# evaluate.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# train.py에 정의된 모델 구조 임포트
from step2_train import SimpleCNN 

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor() 
    ])

    test_dir = './data/테스트'
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    class_names = test_dataset.classes

    # 모델 불러오기
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
    model.eval()

    y_true, y_pred, y_scores = [], [], []

    print("=== 모델 평가 진행 중 ===")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probs[:, 1].cpu().numpy())

    # 결과 리포트 출력
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix 시각화
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('혼동 행렬 (Confusion Matrix)')
    plt.ylabel('실제 라벨')
    plt.xlabel('예측 라벨')
    plt.show()

    # ROC Curve 시각화
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'CNN (AUC = {roc_auc:.2f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    evaluate_model()