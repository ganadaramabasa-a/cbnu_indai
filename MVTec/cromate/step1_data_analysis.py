# data_analysis.py
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def analyze_data(train_dir, test_dir):
    # 단순 시각화를 위해 텐서로만 변환
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 데이터 로드
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    print("=== 데이터 분석 결과 ===")
    print(f"클래스 매핑: {train_dataset.class_to_idx}")
    print(f"학습 데이터 개수: {len(train_dataset)}개")
    print(f"테스트 데이터 개수: {len(test_dataset)}개")

    # 샘플 이미지 시각화 (첫 번째 배치 로드)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    images, labels = next(iter(train_loader))
    class_names = train_dataset.classes

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        # PyTorch 텐서 (C, H, W)를 Matplotlib 형태 (H, W, C)로 변환
        img = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {class_names[labels[i]]}")
        axes[i].axis('off')
        
    plt.suptitle("학습 데이터 샘플 확인")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    TRAIN_DIR = './data/학습'   # 실제 경로에 맞게 수정
    TEST_DIR = './data/테스트' # 실제 경로에 맞게 수정
    analyze_data(TRAIN_DIR, TEST_DIR)