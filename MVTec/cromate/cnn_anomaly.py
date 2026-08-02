import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 가이드북에 명시된 대로 256x256 크기로 조정하고 픽셀 값을 0~1로 정규화합니다.
# PyTorch의 ToTensor()는 이미지를 Tensor로 변환하며 자동으로 0~1 사이로 정규화(/255) 해줍니다.
transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor()
])

# 데이터 경로 설정 (가이드북의 '학습', '테스트' 폴더 구조를 가정)
# 예: path_to_data/학습/정상, path_to_data/학습/불량
train_dir = './resized/학습'
test_dir = './resized/테스트'

# ImageFolder를 사용하면 폴더명('정상', '불량')을 바탕으로 자동으로 라벨링이 진행됩니다.
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# DataLoader 설정 (배치 단위로 데이터 분할)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"학습 데이터 개수: {len(train_dataset)}")
print(f"테스트 데이터 개수: {len(test_dataset)}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 256 -> 128
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64
        )
        
        # Dense (Fully Connected) Layer
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2) # 정상/불량 2개 클래스
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 모델, 손실 함수, 최적화 기법 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련 (Epoch는 3으로 설정 - 가이드북 기준)
epochs = 3

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad() # 기울기 초기화
        
        outputs = model(images) # 순방향 전파
        loss = criterion(outputs, labels) # 손실 계산
        loss.backward() # 역방향 전파
        optimizer.step() # 가중치 업데이트
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 모델 평가
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"테스트 데이터 정확도 (Accuracy): {100 * correct / total:.2f}%")