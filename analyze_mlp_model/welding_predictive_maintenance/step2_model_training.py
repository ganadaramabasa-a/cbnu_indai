import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. 저장된 데이터 로딩
print("데이터를 불러옵니다...")
dataset = torch.load('processed_dataset.pt', weights_only=False)
X_train, Y_train = dataset['X_train'], dataset['Y_train']
X_valid_0, Y_valid_0 = dataset['X_valid_0'], dataset['Y_valid_0']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
valid_loader = DataLoader(TensorDataset(X_valid_0, Y_valid_0), batch_size=32, shuffle=False)

# 2. 모델 구조 정의 (가이드북 구조 반영)
class LSTM_AE(nn.Module):
    def __init__(self, n_features, seq_len):
        super(LSTM_AE, self).__init__()
        self.seq_len = seq_len
        # Encoder
        self.enc1 = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)
        self.enc2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        # Decoder
        self.dec1 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.dec2 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.out = nn.Linear(64, n_features)

    def forward(self, x):
        x, _ = self.enc1(x)
        x, (h_n, _) = self.enc2(x)
        
        # RepeatVector 역할 (마지막 Hidden State를 시퀀스 길이만큼 복제)
        x = h_n.transpose(0, 1).repeat(1, self.seq_len, 1)
        
        x, _ = self.dec1(x)
        x, _ = self.dec2(x)
        x = self.out(x)
        return x

model = LSTM_AE(n_features=5, seq_len=20).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습률 감소(ReduceLROnPlateau) 스케줄러
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8)

# 3. 모델 학습 루프
print(f"[{device}] 환경에서 학습을 시작합니다...")
epochs = 50 # 가이드북은 200회

train_losses, valid_losses = [], []
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 17 # Early Stopping

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    # 검증 단계
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            valid_loss += loss.item()
            
    avg_train_loss = train_loss / len(train_loader)
    avg_valid_loss = valid_loss / len(valid_loader)
    train_losses.append(avg_train_loss)
    valid_losses.append(avg_valid_loss)
    
    scheduler.step(avg_valid_loss) # 스케줄러 업데이트
    
    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.5f} | Valid Loss: {avg_valid_loss:.5f}")
    
    # Early Stopping 로직
    if avg_valid_loss < best_val_loss - 0.00001:
        best_val_loss = avg_valid_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_lstm_ae.pth') # 최고 성능 모델 저장
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early Stopping 트리거 발동! 학습을 조기 종료합니다.")
            break

print("학습 완료! (best_lstm_ae.pth 저장됨)")