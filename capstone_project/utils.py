# utils.py
import os
import hashlib
import requests
import base64
from datetime import datetime, timezone
import pdfplumber
from config import HARRIER_EMBEDDING_URL, LITELLM_HEADERS, LITELLM_VLM_URL, VLM_MODEL_NAME, EMBEDDING_MODEL_NAME

def get_file_metadata(file_path):
    """파일의 수정 시간(mtime)과 크기를 가볍게 조회 (1차 필터용)"""
    stat = os.stat(file_path)
    return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc), stat.st_size

def calculate_file_hash(file_path):
    """파일 내용 전체의 MD5 해시값을 계산 (2차 필터용)"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def classify_pdf_document_type(file_path):
    """
    [공용 유틸] PDF의 메타데이터, 절대 치수, 텍스트 파편화 밀도를 종합하여
    Word/보고서형(REPORT)인지 PPT/슬라이드형(SLIDE)인지 판별합니다.
    """
    with pdfplumber.open(file_path) as pdf:
        # 1단계: 메타데이터 검증
        meta = pdf.metadata or {}
        creator = str(meta.get("Creator", "")).lower()
        producer = str(meta.get("Producer", "")).lower()
        
        if "powerpoint" in creator or "powerpoint" in producer or "keynote" in creator:
            return "SLIDE_TYPE"
        if "word" in creator or "word" in producer or "hwp" in creator:
            return "REPORT_TYPE"
            
        # 2단계: 첫 페이지 물리 치수(Points) 검증
        first_page = pdf.pages[0]
        p_x0, p_top, p_x1, p_bottom = first_page.bbox
        width = round(p_x1 - p_x0)
        height = round(p_bottom - p_top)
        
        if width in [960, 720] or height in [960, 720] or width == 540:
            return "SLIDE_TYPE"
            
        if width in [595, 842] or height in [595, 842]:
            if width <= height: 
                return "REPORT_TYPE"

        # 3단계: 구조적 특징 분석 (텍스트 파편화 밀도 검사)
        words = first_page.extract_words()
        if not words:
            return "REPORT_TYPE"
            
        unique_tops = set(round(w['top'], 1) for w in words)
        fragmentation_ratio = len(unique_tops) / len(words)
        
        if fragmentation_ratio > 0.4: 
            return "SLIDE_TYPE"
            
        return "REPORT_TYPE"

def get_harrier_embedding(text):
    """LiteLLM 게이트웨이를 경유하여 임베딩 벡터 생성"""
    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "input": text
    }
    try:
        # [핵심] headers=LITELLM_HEADERS를 반드시 추가하여 Host 정보를 주입합니다.
        response = requests.post(
            HARRIER_EMBEDDING_URL, 
            json=payload, 
            headers=LITELLM_HEADERS, 
            timeout=30
        )
        
        # 정상 응답 코드(200) 확인 후 데이터 파싱
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            print(f"[-] Gateway Error ({response.status_code}): {response.text}")
            return None
            
    except Exception as e:
        print(f"[-] Embedding API Connection Error: {e}")
        return None

def analyze_image_with_vlm(image_path):
    """LiteLLM 게이트웨이를 경유하여 표준 OpenAI Vision 포맷으로 이미지 분석 요청"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # LiteLLM / OpenAI 표준 멀티모달 페이로드 구조
        payload = {
            "model": VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "이 이미지는 사내 문서의 시각 자료입니다. 표, 그래프, 다이어그램을 마크다운 형태의 상세 텍스트로 설명하세요."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_string}"
                            }
                        }
                    ]
                }
            ],
            "stream": False
        }
        
        # 게이트웨이 필수 헤더를 포함하여 요청 전송
        response = requests.post(
            LITELLM_VLM_URL, 
            json=payload, 
            headers=LITELLM_HEADERS, 
            timeout=60
        )
        
        if response.status_code == 200:
            # OpenAI 스펙 응답 파싱 -> choices[0].message.content
            return response.json()['choices'][0]['message']['content']
        else:
            return f"[VLM 게이트웨이 에러 ({response.status_code}): {response.text}]"
            
    except Exception as e:
        return f"[VLM 분석 실패: {str(e)}]"
    
def analyze_image_with_vlm_fusion(image_path):
    """LiteLLM 게이트웨이를 통해 이미지 요약(VLM)과 문자 추출(OCR)을 원샷으로 수행"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 💡 비전 의미 분석과 하드웨어 레벨 OCR을 동시 유도하는 고도화된 프롬프트
        fusion_prompt = (
            "당신은 사내 문서 이미지 분석 및 OCR 추출기입니다. 안내 문구, 서론, 결론, 마크다운 제목 등 지정되지 않은 텍스트는 절대 출력하지 마세요.\n\n"
            "이 이미지는 사내 문서에서 추출한 시각 자료(그래프, 도표, 아키텍처, 사진 등)입니다."
            "반드시 다음 두 가지 파트를 명확히 구분하여 마크다운 스타일로 작성하세요.\n\n"
            "1. [시각자료 분석 요약]: 이미지의 핵심 의미, 추이, 도표가 나타내는 비즈니스적 의도를 분석가 관점에서 상세히 설명하세요.\n"
            "2. [시각자료 OCR 원문 추출]: 이미지 내에 존재하는 모든 텍스트, 숫자, 기호, 데이터 표의 셀 내용을 단 하나도 빠짐없이 보이는 그대로 전사(Transcribe)하세요.\n"
            "[공통 예외 규칙]\n"
            "- 만약 이미지에 의미 있는 정보가 없거나, 단순 도형/빈 배경 등으로 인해 분석 및 텍스트 추출이 모두 불가능한 경우, 단 한 글자도 출력하지 말고 오직 빈 문자열(공백)만 반환하세요.\n"
        )

        payload = {
            "model": VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": fusion_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_string}"}
                        }
                    ]
                }
            ],
            "stream": False
        }
        
        response = requests.post(
            LITELLM_VLM_URL, 
            json=payload, 
            headers=LITELLM_HEADERS, 
            timeout=180
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"\n[VLM 게이트웨이 에러 ({response.status_code})]\n"
            
    except Exception as e:
        return f"\n[VLM/OCR 융합 분석 실패: {str(e)}]\n"
    
def restore_and_analyze_page(image_path, native_text=None):
    """
    이미지와 깨진 디지털 텍스트를 대조하여, 
    누락된 기호/영문을 완벽히 복원한 '단일 통합 텍스트'를 반환합니다.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 기본 프롬프트 (정상 문서 내 도표용)
        prompt = (
            "이 이미지는 문서 내 시각 자료입니다. 오직 하나의 마크다운 텍스트로만 답변하세요.\n"
            "1. [시각자료 분석 요약]: 이미지의 핵심 의미와 도표 내용을 상세히 설명하세요.\n"
            "2. [시각자료 OCR 원문 추출]: 내부에 존재하는 데이터와 텍스트를 보이는 그대로 전사하세요."
        )
        
        # 💡 오염된 문서 복구 전용 프롬프트 (native_text가 주어졌을 때)
        if native_text and native_text.strip():
            prompt = (
                "당신은 문서 복원 전문가입니다. 나는 다음 두 가지 데이터를 제공합니다.\n"
                f"1) 디지털 파서로 추출했으나 괄호, 문단 기호, 영단어가 통째로 증발한 텍스트: \n\"\"\"\n{native_text.strip()}\n\"\"\"\n"
                "2) 해당 페이지의 실제 원본 이미지\n\n"
                "[미션]\n"
                "제공된 텍스트를 기준 프레임으로 삼되, 이미지를 눈으로 확인하여 "
                "증발해 버린 문단 기호(예: '가)' 등), 괄호, 영문 명칭(예: '(Labeling)' 등), 도표 수치들을 "
                "원래 있어야 할 위치에 정확히 끼워 넣은 **완벽히 복원된 하나의 통합 마크다운 문서**를 출력하세요.\n"
                "절대 분석 파트와 오염 파트를 나누지 말고, 처음부터 끝까지 하나의 자연스러운 본문 스트림으로 출력해야 합니다."
            )

        payload = {
            "model": VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_string}"}
                        }
                    ]
                }
            ],
            "stream": False
        }
        
        response = requests.post(LITELLM_VLM_URL, json=payload, headers=LITELLM_HEADERS, timeout=60)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"[VLM 복원 에러: {response.status_code}]"
            
    except Exception as e:
        return f"[복원 실패: {str(e)}]"
    
def get_synthesized_body_text(image_path, broken_text):
    """도표/차트를 제외하고, 오직 순수 본문 줄글의 유실된 기호와 영문만 복구합니다."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 💡 지침 3번을 통해 독립된 시각자료 내부 데이터는 생략하도록 제한
        synthesis_prompt = (
            "당신은 문서 본문 교정 전문가입니다. 제공된 [디지털 추출 텍스트]는 "
            "폰트 인코딩 문제로 인해 괄호, 문단 기호, 영문 용어가 유실된 상태입니다.\n\n"
            "[디지털 추출 텍스트]:\n"
            f"{broken_text}\n\n"
            "작업 지침:\n"
            "1. 첨부된 이미지를 참고하여 [디지털 추출 텍스트]에서 빠진 기호(예: 가), 1)), 괄호, 영문(예: Labeling)을 제자리에 복원하세요.\n"
            "2. 두 데이터를 대조하여 오타가 없는 가장 완벽한 형태의 본문 줄글을 마크다운 형태로 작성하세요.\n"
            "3. ⚠️ 중요: 문서 내에 독립적으로 존재하는 그림, 도표, 차트 내부의 세부 데이터는 절대 본문에 포함하지 마세요. (그것들은 별도 파이프라인에서 처리됩니다)\n"
            "4. 다른 부연 설명 없이 오직 복구된 본문 텍스트만 출력하세요."
        )

        payload = {
            "model": VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": synthesis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_string}"}
                        }
                    ]
                }
            ],
            "stream": False
        }
        
        response = requests.post(LITELLM_VLM_URL, json=payload, headers=LITELLM_HEADERS, timeout=90)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return broken_text
    except:
        return broken_text

def get_synthesized_body_text2(image_path, broken_text, num_images):
    """
    기존의 전면 페이지 복구 구조를 유지하되, 
    VLM이 본문 흐름 내에 이미지 위치 태그([시각자료 #idx])를 엄격하게 삽입하도록 프롬프트 강화
    """
    try:
        import base64
        import requests
        from config import LITELLM_VLM_URL, LITELLM_HEADERS, VLM_MODEL_NAME
        
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 💡 지시사항을 무시하지 않도록 엄격한 가이드라인 구조로 프롬프트 재설계
        prompt = (
            "당신은 고정밀 문서 레이아웃 복구 전문가입니다. "
            "주어진 [깨진 텍스트]는 PDF 인코딩 오류로 인해 기호, 괄호, 영문 등이 깨져 있는 상태입니다. "
            "함께 제공된 페이지 전체 스크린샷 이미지를 보고 본문 줄글을 완벽하게 마크다운 형태로 교정하세요.\n\n"
            
            f"⚠️ [필수 임무 - 시각자료 위치 마킹]\n"
            f"현재 페이지에는 총 {num_images}개의 그림/도표(시각자료)가 본문 흐름 사이에 존재합니다.\n"
            f"텍스트 복구 도중, 해당 시각자료들이 위치한 문단과 문단 사이에 반드시 "
            f"`[시각자료 #0]`, `[시각자료 #1]` ... 형태의 태그를 위에서부터 순서대로 정확히 삽입해야 합니다.\n\n"
            
            "[깨진 텍스트]\n"
            f"{broken_text}\n\n"
            
            "[출력 규칙]\n"
            f"1. 본문 흐름에 맞게 총 {num_images}개의 태그(`[시각자료 #0]`부터 `[시각자료 #{num_images-1}]`까지)를 원래 그림이 있던 자리에 반드시 포함하세요.\n"
            "2. 그림/도표 내부의 상세 글자는 여기서 추출하지 마세요. 오직 줄글 본문만 복구하고 그림 자리는 태그로 대체합니다.\n"
            "3. 다른 설명이나 인사말은 모두 배제하고, 복구 및 태그 삽입이 완료된 마크다운 본문만 반환하세요."
            "3. [예외 사항]: 표 형태의 경우 텍스트가 유실되더라도 태그로 대체하지 말고, 보이는 텍스트만 복구하여 그대로 유지하세요.(표는 시각자료가 아니므로 태그로 대체하지 않습니다)"
        )
        
        payload = {
            "model": VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}
                    ]
                }
            ],
            "stream": False
        }
        
        response = requests.post(LITELLM_VLM_URL, json=payload, headers=LITELLM_HEADERS, timeout=60)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return broken_text
    except:
        return broken_text
