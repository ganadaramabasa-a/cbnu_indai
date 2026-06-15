# config.py
import os

# PostgreSQL 데이터베이스 설정
DB_CONFIG = {
    "host": "",
    "database": "",
    "user": "",
    "password": "",
    "port": 5432
}

# 1. LiteLLM 게이트웨이 엔드포인트 설정
LITELLM_BASE_URL = "http://localhost/v1"
HARRIER_EMBEDDING_URL = f"{LITELLM_BASE_URL}/embeddings"
LITELLM_VLM_URL = f"{LITELLM_BASE_URL}/chat/completions"

# 2. [핵심] 게이트웨이 라우팅을 위한 필수 헤더 구성
LITELLM_HEADERS = {
    "Host": "mx-solution",
    "Content-Type": "application/json"
}

# 3. 사용할 임베딩 모델명 입력
EMBEDDING_MODEL_NAME = "harrier-0.6B"
VLM_MODEL_NAME = "gemma-4-26b-a4b-it"

# 사내 공유 폴더 스캔 대상 경로 및 이미지 추출 임시 저장소
SHARED_FOLDER = "/mnt/p_drive/Common/MX솔루션팀/문서검색"
IMAGE_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/rag_extracted_images"))

# 디렉토리 자동 생성
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
