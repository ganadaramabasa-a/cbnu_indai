# chunker.py
import uuid
import re
from nlp_utils import preprocess_korean_fts

def split_into_sentences(text: str):
    """문장 부호 기준 분리"""
    sentence_end = re.compile(r'(?<=[.!?])\s+')
    return [s.strip() for s in sentence_end.split(text) if s.strip()]

def build_parent_child_packets(raw_pages: list, 
                                 parent_size_chars: int = 2500, parent_overlap_chars: int = 500,
                                 child_size_chars: int = 350, child_overlap_chars: int = 100) -> tuple:
    """
    [성능 최우선] 페이지 경계를 무시하고 대형 부모 청크를 슬라이딩 윈도우로 생성하며,
    해당 부모 문맥이 걸쳐 있는 모든 페이지 번호를 배열(ex: [1, 2])로 추출합니다.
    """
    parent_data_list = []
    child_data_list = []
    
    if not raw_pages:
        return parent_data_list, child_data_list

    file_path = raw_pages[0]["file_path"]
    
    # 1. 문장 단위로 해체할 때 이미지 정보와 매핑 정보 유지
    all_sentences = []
    for page in raw_pages:
        # 해당 페이지에 소속된 전체 이미지 리스트 확보
        page_images = page.get("images", [])
        sentences = split_into_sentences(page["content"])
        
        for s in sentences:
            all_sentences.append({
                "text": s,
                "page": page["page_number"],
                "section": page.get("section_title", ""),
                "images": page_images  # 문장별로 부모 페이지의 이미지 포인터를 들고 다님
            })
            
    # 2. 부모 청크 슬라이딩 윈도우 가동
    p_idx = 0
    current_parent_sentences = []
    current_parent_chars = 0
    
    while p_idx < len(all_sentences):
        s_obj = all_sentences[p_idx]
        current_parent_sentences.append(s_obj)
        current_parent_chars += len(s_obj["text"])
        
        # 부모 조건 충족 (설정 크기 도달 또는 문서 끝)
        if current_parent_chars >= parent_size_chars or p_idx == len(all_sentences) - 1:
            parent_id = str(uuid.uuid4())
            parent_text = " ".join([s["text"] for s in current_parent_sentences])
            
            # 💡 [핵심] 이 부모 청크에 포함된 모든 페이지 번호를 중복 제거 후 정렬하여 배열로 추출
            parent_pages = sorted(list(set([s["page"] for s in current_parent_sentences])))
            main_section = current_parent_sentences[0]["section"]
            
            # 💡 [중요 핵심] 현재 부모 윈도우 버퍼 내에 속한 모든 문장의 이미지 리스트를 중복 없이 통합
            parent_images = []
            for s in current_parent_sentences:
                for img in s["images"]:
                    if img not in parent_images:
                        parent_images.append(img)
                        
            parent_data_list.append((
                parent_id,
                file_path,
                parent_pages,  # 👉 이제 데이터 한 줄에 [1, 2] 혹은 [2, 3]이 정상적으로 들어갑니다.
                main_section,
                parent_text,
                parent_images  # 이미지 정보도 함께 저장
            ))
            
            # 3. 🔍 생성된 부모 청크의 '내부 문장 스트림'을 기준으로 자식 청크 분할
            c_idx = 0
            current_child_sentences = []
            current_child_chars = 0
            
            while c_idx < len(current_parent_sentences):
                cs_obj = current_parent_sentences[c_idx]
                current_child_sentences.append(cs_obj["text"])
                current_child_chars += len(cs_obj["text"])
                
                if current_child_chars >= child_size_chars or c_idx == len(current_parent_sentences) - 1:
                    child_text = " ".join(current_child_sentences)
                    child_id = str(uuid.uuid4())
                    
                    # Kiwi FTS 정제
                    fts_source = preprocess_korean_fts(child_text)
                    
                    child_data_list.append({
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "content": child_text,
                        "fts_source": fts_source
                    })
                    
                    # 자식 오버랩 역산
                    c_overlap_chars = 0
                    c_overlap_count = 0
                    for cs in reversed(current_child_sentences):
                        if c_overlap_chars + len(cs) <= child_overlap_chars:
                            c_overlap_chars += len(cs)
                            c_overlap_count += 1
                        else:
                            break
                    
                    current_child_sentences = []
                    current_chars = 0
                    if c_overlap_count > 0 and c_idx < len(current_parent_sentences) - 1:
                        c_idx -= (c_overlap_count - 1)
                        
                c_idx += 1
            
            # 4. 부모 오버랩 역산 후 대형 윈도우 포인터 이동
            p_overlap_chars = 0
            p_overlap_count = 0
            for s in reversed(current_parent_sentences):
                if p_overlap_chars + len(s["text"]) <= parent_overlap_chars:
                    p_overlap_chars += len(s["text"])
                    p_overlap_count += 1
                else:
                    break
                    
            current_parent_sentences = []
            current_parent_chars = 0
            if p_overlap_count > 0 and p_idx < len(all_sentences) - 1:
                p_idx -= (p_overlap_count - 1)
                
        p_idx += 1
        
    return parent_data_list, child_data_list
