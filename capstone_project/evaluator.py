# evaluator.py
import json
import re
import requests
from typing import List, Dict, Any, Union, Optional
from config import LITELLM_VLM_URL, LITELLM_HEADERS, VLM_MODEL_NAME

class ResponseAccuracyEvaluator:
    """
    1. 응답 정확도 평가지표
    전문가 QA 기준 수치·규정·조건·절차 사실 일치 여부를 LLM-as-a-judge로 평가합니다.
    """
    def __init__(self, model_name: str = VLM_MODEL_NAME, api_url: str = LITELLM_VLM_URL, headers: dict = LITELLM_HEADERS):
        self.model_name = model_name
        self.api_url = api_url
        self.headers = headers

    def evaluate(self, question: str, reference_answer: str, generated_answer: str) -> Dict[str, Any]:
        prompt = (
            "당신은 RAG 시스템의 답변 정확도를 검증하는 전문 평가자입니다.\n"
            "아래의 [질문]과 [전문가 답변]을 기준으로 [모델 답변]이 사실적으로 일치하는지 평가해 주세요.\n\n"
            "평가 대상 범주:\n"
            "1. 수치(numbers): 모델 답변의 수치(금액, 기간, 수량, 날짜 등)가 전문가 답변과 일치하는가?\n"
            "2. 규정(regulations): 모델 답변이 언급한 규정, 규칙, 법령, 가이드라인이 전문가 답변과 일치하는가?\n"
            "3. 조건(conditions): 모델 답변이 제시한 전제 조건, 제약 사항이 전문가 답변과 일치하는가?\n"
            "4. 절차(procedures): 모델 답변이 기술한 단계나 수행 절차가 전문가 답변과 일치하는가?\n\n"
            "평가 규칙:\n"
            "- 각 범주에 대해 사실 일치 여부를 판단하세요. (일치하면 1, 불일치 또는 왜곡/누락 시 0, 해당 범주가 전문가 답변에 중요하게 다뤄지지 않거나 관련 없으면 null)\n"
            "- 각 판단에 대한 간결한 사유를 한 문장으로 작성하세요.\n"
            "- 전체적인 최종 일치 점수(overall_score, 0.0 ~ 1.0)를 산출하세요. 모든 해당 범주가 일치하면 1.0, 왜곡이나 거짓 정보가 있을 경우 감점합니다.\n\n"
            "출력 포맷: 반드시 아래 양식의 JSON 형식으로만 응답해야 하며, 마크다운 백틱(```json)이나 다른 설명 없이 순수 JSON만 반환하세요.\n"
            "{\n"
            '  "numbers": {"match": 1, 0, or null, "reason": "이유"},\n'
            '  "regulations": {"match": 1, 0, or null, "reason": "이유"},\n'
            '  "conditions": {"match": 1, 0, or null, "reason": "이유"},\n'
            '  "procedures": {"match": 1, 0, or null, "reason": "이유"},\n'
            '  "overall_score": 0.0~1.0,\n'
            '  "reason": "종합 평가 사유"\n'
            "}\n\n"
            f"[질문]:\n{question}\n\n"
            f"[전문가 답변]:\n{reference_answer}\n\n"
            f"[모델 답변]:\n{generated_answer}\n"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=45)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                # Markdown JSON 코드 블록 제거
                cleaned_content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
                cleaned_content = re.sub(r"\s*```$", "", cleaned_content)
                return json.loads(cleaned_content)
            else:
                return {
                    "error": f"API Error ({response.status_code}): {response.text}",
                    "overall_score": 0.0,
                    "reason": "평가 API 호출 실패"
                }
        except Exception as e:
            return {
                "error": str(e),
                "overall_score": 0.0,
                "reason": "평가 파싱 또는 통신 에러 발생"
            }


class RetrievalPrecisionEvaluator:
    """
    2. 검색 정밀도 평가지표 (HitRate@K)
    상위 k 결과 내 정답 근거(또는 관련 문서 정보)가 포함되어 있는지 비율을 측정합니다.
    """
    @staticmethod
    def calculate_hit(retrieved_chunks: List[Dict[str, Any]], ground_truth: Dict[str, Any], k: int) -> int:
        """
        단일 쿼리에 대해 상위 k개 결과 내에 ground_truth(정답 근거)가 포함되어 있는지 판단 (Hit: 1, Miss: 0)
        
        retrieved_chunks: 검색 결과 청크 리스트 (예: [{"file_path": "...", "page_number": 1, "content": "..."}])
        ground_truth: 정답 근거 메타데이터 정보 또는 텍스트 키워드
                      형식 예 1 (메타데이터): {"file_path": "a.pdf", "page_number": 3}
                      형식 예 2 (부모/자식 ID): {"parent_id": "uuid..."} 또는 {"child_id": "uuid..."}
                      형식 예 3 (텍스트 포함): {"text": "핵심 키워드"}
        k: 평가할 상위 검색 결과 개수
        """
        top_k_chunks = retrieved_chunks[:k]
        
        # Ground Truth 타입에 따라 적절한 매칭 기법 적용
        for chunk in top_k_chunks:
            # Case 1: parent_id / child_id 매칭
            if "parent_id" in ground_truth:
                if str(chunk.get("parent_id")) == str(ground_truth["parent_id"]):
                    return 1
            elif "child_id" in ground_truth:
                if str(chunk.get("child_id")) == str(ground_truth["child_id"]):
                    return 1
            elif "chunk_id" in ground_truth:
                if str(chunk.get("chunk_id")) == str(ground_truth["chunk_id"]):
                    return 1
            
            # Case 2: 파일 경로 및 페이지 정보 매칭
            gt_file = ground_truth.get("file_path")
            gt_page = ground_truth.get("page_number")
            if gt_file is not None:
                # 경로의 단순 파일명만 비교하거나 절대 경로 비교
                chunk_file = chunk.get("file_path", "")
                if (gt_file in chunk_file) or (chunk_file in gt_file):
                    if gt_page is None:
                        return 1
                    # 페이지 매칭 (단일 값 또는 배열 지원)
                    chunk_pages = chunk.get("page_number", [])
                    if isinstance(chunk_pages, list):
                        if gt_page in chunk_pages:
                            return 1
                    elif isinstance(chunk_pages, (int, str)):
                        if int(chunk_pages) == int(gt_page):
                            return 1
            
            # Case 3: 텍스트 기반 포함 검사
            gt_text = ground_truth.get("text")
            if gt_text:
                chunk_content = chunk.get("content", "")
                if gt_text in chunk_content:
                    return 1
                    
        return 0

    @classmethod
    def calculate_hit_rate(cls, all_retrieved_chunks: List[List[Dict[str, Any]]], all_ground_truths: List[Dict[str, Any]], k: int) -> float:
        """
        전체 테스트 세트에 대해 HitRate@K 계산
        """
        if not all_retrieved_chunks or not all_ground_truths:
            return 0.0
            
        hits = 0
        total = min(len(all_retrieved_chunks), len(all_ground_truths))
        
        for i in range(total):
            hits += cls.calculate_hit(all_retrieved_chunks[i], all_ground_truths[i], k)
            
        return hits / total


class HallucinationBlockingEvaluator:
    """
    3. 환각 차단률 평가지표
    지식베이스 외 질문(Out of Knowledge)에 대해 "모른다" 또는 "근거 없음" 등으로 적절히 답변했는지 판별합니다.
    """
    REFUSAL_KEYWORDS = [
        "모릅니다", "모르겠습니다", "알 수 없", "확인되지 않", "기록이 없", 
        "제시되지 않", "정보가 없", "근거가 없", "답변하기 어렵", "답변해 드릴 수 없", "답변을 드릴 수 없",
        "제공되지 않", "알 수 없는", "찾을 수 없", "포함되어 있지 않",
        "언급되지 않", "파악되지 않", "존재하지 않"
    ]

    def __init__(self, model_name: str = VLM_MODEL_NAME, api_url: str = LITELLM_VLM_URL, headers: dict = LITELLM_HEADERS):
        self.model_name = model_name
        self.api_url = api_url
        self.headers = headers

    def is_refusal_by_rules(self, generated_answer: str) -> bool:
        """룰 기반 키워드 매칭"""
        # 공백 제거 후 키워드 탐색
        normalized_answer = generated_answer.replace(" ", "")
        for kw in self.REFUSAL_KEYWORDS:
            if kw.replace(" ", "") in normalized_answer:
                return True
        return False

    def is_refusal_by_llm(self, question: str, generated_answer: str) -> bool:
        """LLM을 통한 지식 외 답변 거절 여부 정밀 분류"""
        prompt = (
            "아래는 RAG 시스템의 질문과 답변입니다. 모델이 본문/지식베이스에 해당 정보가 없어서 답변을 거부했거나(모른다고 답변), "
            "관련 내용을 찾을 수 없다고 답변했는지 판별해 주세요.\n\n"
            "평가 기준:\n"
            "- 질문에 대한 직접적인 지식 답변을 하지 않고 '알 수 없다', '언급되지 않았다', '근거가 없다', '정보가 없다' 등으로 답했다면 TRUE입니다.\n"
            "- 지식을 임의로 꾸며내어 답변(환각 답변)했거나, 거절하지 않고 그럴듯하게 설명했다면 FALSE입니다.\n\n"
            "출력 포맷: 반드시 아래 양식의 JSON 형식으로만 응답해야 하며, 마크다운 백틱(```json)이나 다른 설명 없이 순수 JSON만 반환하세요.\n"
            "{\n"
            '  "refused": true 또는 false,\n'
            '  "reason": "판단 근거 한 줄"\n'
            "}\n\n"
            f"[질문]:\n{question}\n\n"
            f"[모델 답변]:\n{generated_answer}\n"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=30)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                cleaned_content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
                cleaned_content = re.sub(r"\s*```$", "", cleaned_content)
                result = json.loads(cleaned_content)
                return bool(result.get("refused", False))
        except Exception as e:
            print(f"[-] Hallucination evaluator LLM error: {e}")
        return False

    def evaluate_query(self, question: str, generated_answer: str, use_llm_fallback: bool = True) -> Dict[str, Any]:
        """
        단일 Out-of-Knowledge 질의에 대해 거절 성공 여부 판단
        """
        by_rules = self.is_refusal_by_rules(generated_answer)
        if by_rules:
            return {"refused": True, "method": "rule_keywords", "reason": "지식 배제/거부 키워드가 검출됨"}
        
        if use_llm_fallback:
            by_llm = self.is_refusal_by_llm(question, generated_answer)
            return {"refused": by_llm, "method": "llm_evaluator", "reason": "LLM 판단 결과 기반"}
            
        return {"refused": False, "method": "rule_keywords", "reason": "키워드 검출 실패 및 LLM 미사용"}

    def calculate_blocking_rate(self, ood_test_cases: List[Dict[str, str]], use_llm_fallback: bool = True) -> Dict[str, Any]:
        """
        전체 지식베이스 외(Out-of-Domain/Knowledge) 테스트 케이스 목록에 대해 환각 차단률 산출
        ood_test_cases: [{"question": "...", "generated_answer": "..."}] 형태의 리스트
        """
        if not ood_test_cases:
            return {"blocking_rate": 0.0, "total_count": 0, "blocked_count": 0, "details": []}
            
        blocked_count = 0
        details = []
        
        for case in ood_test_cases:
            q = case["question"]
            ans = case["generated_answer"]
            eval_res = self.evaluate_query(q, ans, use_llm_fallback)
            
            is_blocked = eval_res["refused"]
            if is_blocked:
                blocked_count += 1
                
            details.append({
                "question": q,
                "generated_answer": ans,
                "is_blocked": is_blocked,
                "method": eval_res["method"],
                "reason": eval_res["reason"]
            })
            
        return {
            "blocking_rate": blocked_count / len(ood_test_cases),
            "total_count": len(ood_test_cases),
            "blocked_count": blocked_count,
            "details": details
        }


class RAGEvaluator:
    """
    종합 RAG 평가지표 오케스트레이터
    """
    def __init__(self, model_name: str = VLM_MODEL_NAME, api_url: str = LITELLM_VLM_URL, headers: dict = LITELLM_HEADERS):
        self.accuracy_evaluator = ResponseAccuracyEvaluator(model_name, api_url, headers)
        self.blocking_evaluator = HallucinationBlockingEvaluator(model_name, api_url, headers)

    def run_evaluation(self, 
                       qa_test_cases: List[Dict[str, Any]], 
                       retrieval_test_cases: Optional[List[Dict[str, Any]]] = None,
                       ood_test_cases: Optional[List[Dict[str, Any]]] = None,
                       k_list: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        qa_test_cases: [{"question": "...", "reference_answer": "...", "generated_answer": "..."}]
        retrieval_test_cases: [{"retrieved_chunks": [...], "ground_truth": {...}}]
        ood_test_cases: [{"question": "...", "generated_answer": "..."}]
        """
        report = {}

        # 1. 응답 정확도 평가
        if qa_test_cases:
            print("[*] Evaluating Response Accuracy...")
            acc_scores = []
            acc_details = []
            for case in qa_test_cases:
                q = case["question"]
                ref = case["reference_answer"]
                gen = case["generated_answer"]
                eval_res = self.accuracy_evaluator.evaluate(q, ref, gen)
                
                score = eval_res.get("overall_score", 0.0)
                acc_scores.append(score)
                acc_details.append({
                    "question": q,
                    "reference": ref,
                    "generated": gen,
                    "score": score,
                    "categories": {
                        "numbers": eval_res.get("numbers"),
                        "regulations": eval_res.get("regulations"),
                        "conditions": eval_res.get("conditions"),
                        "procedures": eval_res.get("procedures")
                    },
                    "reason": eval_res.get("reason", "")
                })
            
            report["response_accuracy"] = {
                "average_score": sum(acc_scores) / len(acc_scores) if acc_scores else 0.0,
                "total_count": len(acc_scores),
                "details": acc_details
            }

        # 2. 검색 정밀도 평가 (HitRate@K)
        if retrieval_test_cases:
            print("[*] Evaluating Retrieval Precision...")
            ret_report = {}
            all_chunks = [tc["retrieved_chunks"] for tc in retrieval_test_cases]
            all_gts = [tc["ground_truth"] for tc in retrieval_test_cases]
            
            for k in k_list:
                hit_rate = RetrievalPrecisionEvaluator.calculate_hit_rate(all_chunks, all_gts, k)
                ret_report[f"HitRate@{k}"] = hit_rate
                
            report["retrieval_precision"] = ret_report

        # 3. 환각 차단률 평가
        if ood_test_cases:
            print("[*] Evaluating Hallucination Blocking Rate...")
            block_report = self.blocking_evaluator.calculate_blocking_rate(ood_test_cases)
            report["hallucination_blocking"] = block_report

        return report
