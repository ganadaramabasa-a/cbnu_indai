# nlp_utils.py
import os

from kiwipiepy import Kiwi


def create_kiwi():
    """
    Kiwi 엔진을 초기화합니다.
    kiwipiepy 0.22.0부터 sbg/knlm 모델 파일은 기본 배포에서 제외되었습니다.
    """
    model_type = os.getenv("KIWI_MODEL_TYPE", "cong")
    try:
        return Kiwi(model_type=model_type)
    except OSError as exc:
        print(f"[!] Kiwi model_type='{model_type}' 초기화 실패: {exc}")
        print("[*] Kiwi 기본 모델로 fallback합니다.")
        return Kiwi()


# 💡 Kiwi 엔진을 모듈 최상단에서 단 한 번만 초기화 (싱글톤 효과)
kiwi = create_kiwi()

# 도메인 사전에 전문 용어 등록
# kiwi.add_user_word("화학물질안전원", "NNP")
# kiwi.add_user_word("연구용역과제", "NNG")
# kiwi.add_user_word("디지털정보기술", "NNP")

def preprocess_korean_fts(text: str) -> str:
    """
    텍스트에서 명사, 영문, 숫자만 정밀 추출하여 공백으로 연결하는 함수
    """
    if not text or not text.strip():
        return ""
    
    # 복합명사 분해 옵션 활성화
    tokens = kiwi.tokenize(text, split_complex=True)
    
    valid_keywords = []
    for t in tokens:
        # NNG(일반명사), NNP(고유명사), SL(외국어/영문), SN(숫자) 필터링
        if t.tag in ['NNG', 'NNP', 'SL', 'SN']:
            valid_keywords.append(t.form)
            
    return " ".join(valid_keywords)
