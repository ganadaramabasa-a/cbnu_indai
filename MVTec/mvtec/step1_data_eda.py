# 01_data_eda.py
import os
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MVTecDataset(Dataset):
    """MVTec AD 데이터셋 로더"""
    def __init__(self, root_dir, category, is_train=True, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = [] # 0: Normal, 1: Anomaly
        
        if is_train:
            img_dir = os.path.join(root_dir, category, 'train', 'good')
            paths = glob.glob(os.path.join(img_dir, '*.png'))
            self.image_paths.extend(paths)
            self.labels.extend([0] * len(paths))
        else:
            test_dir = os.path.join(root_dir, category, 'test')
            for defect_type in os.listdir(test_dir):
                paths = glob.glob(os.path.join(test_dir, defect_type, '*.png'))
                self.image_paths.extend(paths)
                label = 0 if defect_type == 'good' else 1
                self.labels.extend([label] * len(paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

def show_eda_images(dataset, num_images=4):
    """데이터 샘플 시각화 함수"""
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img, label, _ = dataset[i]
        img_np = img.permute(1, 2, 0).numpy() # Tensor to Numpy
        title = "Normal" if label == 0 else "Anomaly"
        axes[i].imshow(img_np)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.suptitle("MVTec AD Dataset EDA", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 스크립트 파일 위치 기준으로 mvtec_ad 경로를 설정합니다.
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mvtec_ad'))
    CATEGORY = 'bottle'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    print("데이터셋 로딩 중...")
    test_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=False, transform=transform)
    print(f"테스트 데이터 개수: {len(test_dataset)}장")
    
    # 샘플 이미지 확인
    show_eda_images(test_dataset)