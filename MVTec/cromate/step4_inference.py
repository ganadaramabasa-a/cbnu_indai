# inference.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 # 히트맵 색상 처리를 위해 사용

# train.py에 정의된 모델 구조 임포트
from step2_train import SimpleCNN 

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# --- Grad-CAM 구현 클래스 ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 모델의 순방향, 역방향 진행 시 특징 맵과 기울기를 저장하도록 Hook 등록
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        self.model.eval()
        
        # 순방향 전파
        output = self.model(x)
        self.model.zero_grad()
        
        # 모델이 예측한 클래스의 점수에 대해 역전파 수행
        score = output[:, class_idx]
        score.backward(retain_graph=True)
        
        # 저장된 기울기와 특징 맵 가져오기
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # 기울기의 평균을 구하여 각 채널의 가중치(Weight) 계산
        weights = np.mean(gradients, axis=(1, 2))
        
        # 가중치와 특징 맵을 곱하여 조합 (Linear Combination)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # 양의 영향력을 주는 부분만 남기기 위해 ReLU 적용
        cam = np.maximum(cam, 0)
        
        # 0 ~ 1 사이로 정규화
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam
# -----------------------------

def predict_single_image_with_cam(image_path, model_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])
    
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"이미지 로드 실패: {e}")
        return
    
    img_tensor = transform(img_pil).unsqueeze(0).to(device) 
    
    # 모델 불러오기
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    # 1. 예측 수행
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        prob_max, predicted = torch.max(probs, 1)
        
        pred_idx = predicted.item()
        pred_class = class_names[pred_idx]
        confidence = prob_max.item() * 100
        
    print(f">>> 분석 결과: [{pred_class}] (확신도: {confidence:.2f}%)")

    # 2. Grad-CAM 추출
    # SimpleCNN의 마지막 Convolution 레이어를 타겟으로 설정합니다.
    # self.conv_layers 구조: 0:Conv2d, 1:ReLU, 2:MaxPool, 3:Conv2d, 4:ReLU, 5:MaxPool
    target_layer = model.conv_layers[3] 
    
    cam_extractor = GradCAM(model, target_layer)
    cam = cam_extractor(img_tensor, pred_idx)
    
    # 3. 원본 이미지와 Heatmap 합성 시각화
    img_np = np.array(img_pil.resize((256, 256))) # 원본 이미지를 256x256으로 변환
    cam_resized = cv2.resize(cam, (256, 256))     # 히트맵도 동일한 크기로 리사이즈
    
    # 히트맵에 컬러(Jet) 맵 씌우기
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # OpenCV의 BGR을 RGB로 변환
    
    # 원본 이미지와 히트맵을 6:4 비율로 투명하게 겹치기
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    # 화면에 띄우기
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("원본 제품 이미지")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title("판단 주요 영역 (히트맵)")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"합성 결과: {pred_class} ({confidence:.1f}%)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 실습 시 이 경로를 변경하도록 지도해 주세요.
    # TARGET_IMAGE = './data/평가/sample_ok.png' 
    TARGET_IMAGE = './data/테스트/불량/KEMP_IMG_DATA_Error_55.png' 

    MODEL_PATH = 'cnn_model.pth'
    CLASSES = ['불량', '정상'] 
    
    predict_single_image_with_cam(TARGET_IMAGE, MODEL_PATH, CLASSES)