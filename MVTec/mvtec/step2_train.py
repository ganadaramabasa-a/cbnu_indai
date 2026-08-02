# 02_train.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from step1_data_eda import MVTecDataset # 앞서 만든 Dataset 클래스 임포트

class ConvAutoencoder(nn.Module):
    """합성곱 오토인코더 아키텍처"""
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        d2 = self.up2(x3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))
        return self.out_conv(d1)


def add_gaussian_noise(images, std=0.05):
    noisy = images + torch.randn_like(images) * std
    return torch.clamp(noisy, 0.0, 1.0)

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mvtec_ad'))
    CATEGORY = 'bottle'
    BATCH_SIZE = 16
    NUM_EPOCHS = 80
    VAL_SPLIT = 0.1
    PATIENCE = 12

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
    ])

    full_train_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=True, transform=transform)
    val_size = max(1, int(len(full_train_dataset) * VAL_SPLIT))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 디바이스: {device}")

    model = ConvAutoencoder().to(device)
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'autoencoder_model.pth'))

    print("모델 학습 시작...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, _, _ in train_loader:
            images = images.to(device)
            noisy_images = add_gaussian_noise(images)
            outputs = model(noisy_images)
            loss = 0.6 * l1_criterion(outputs, images) + 0.4 * mse_criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss = 0.6 * l1_criterion(outputs, images) + 0.4 * mse_criterion(outputs, images)
                val_loss += loss.item()

        train_loss = epoch_loss / max(1, len(train_loader))
        val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"조기 종료: {PATIENCE} epoch 동안 검증 손실 개선 없음")
                break

    print(f"모델 저장 완료: {SAVE_PATH}")