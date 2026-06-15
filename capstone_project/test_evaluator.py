# test_evaluator.py
import sys
import unittest
from unittest.mock import MagicMock, patch
from evaluator import ResponseAccuracyEvaluator, RetrievalPrecisionEvaluator, HallucinationBlockingEvaluator, RAGEvaluator

# =========================================================================
# 1. 테스트용 더미 데이터셋 정의
# =========================================================================

# 1. 응답 정확도 테스트 케이스
QA_TEST_CASES = [
    {
        "question": "화학물질 배출량 조사 대상 기업의 연간 취급량 기준은 어떻게 되나요?",
        "reference_answer": "연간 화학물질 취급량이 I그룹(발암성 물질 등 20종)은 1톤 이상, II그룹(기타 독성 물질 등 395종)은 10톤 이상인 사업장이 조사 대상입니다.",
        "generated_answer": "조사 대상 기업의 연간 취급량 기준은 I그룹의 경우 1톤 이상, II그룹의 경우 10톤 이상 취급하는 사업장입니다.",
        "expected_description": "수치와 기준이 완벽히 일치하는 모범 답변"
    },
    {
        "question": "신규 연구개발 과제의 신청 및 접수 절차를 요약해 주세요.",
        "reference_answer": "1단계: RFP 공고 확인 -> 2단계: 사업계획서 작성 및 온라인 제출 -> 3단계: 연구개발성과실무조정위 사전 검토 -> 4단계: 대면 평가 및 선정.",
        "generated_answer": "신규 R&D 과제는 RFP 공고 확인 후 사업계획서를 작성하여 온라인으로 제출합니다. 이후 바로 대면 평가를 통해 선정되며, 별도의 사전 검토 위원회 단계는 없습니다.",
        "expected_description": "절차 중 3단계를 누락하고 부정한 왜곡 답변 (감점 대상)"
    }
]

# 2. 검색 정밀도 테스트 케이스 (HitRate@K)
RETRIEVAL_TEST_CASES = [
    {
        "query": "A사 클라우드 아키텍처 보안 가이드라인",
        "ground_truth": {"file_path": "cloud_security_guide.pdf", "page_number": 5},
        "retrieved_chunks": [
            {"file_path": "cloud_security_guide.pdf", "page_number": 5, "content": "보안 가이드라인 5페이지 내용..."},
            {"file_path": "network_topology.docx", "page_number": 1, "content": "네트워크 구성도..."},
            {"file_path": "company_rule.pdf", "page_number": 2, "content": "사내 보안 규정..."}
        ]
    },
    {
        "query": "임직원 특별 포상 기준 및 지급액",
        "ground_truth": {"parent_id": "8fa67123-b1d5-4a6c-9ad5-56ee79b9a674"},
        "retrieved_chunks": [
            {"file_path": "finance_rule.pdf", "page_number": 10, "content": "재무 규정..."},
            {"parent_id": "8fa67123-b1d5-4a6c-9ad5-56ee79b9a674", "content": "특별 포상금은 기본급의 200% 지급한다..."},
            {"file_path": "company_rule.pdf", "page_number": 12, "content": "인사 제도..."}
        ]
    },
    {
        "query": "신입사원 OJT 교육 일정표",
        "ground_truth": {"text": "OJT 교육 세부 시간표"},
        "retrieved_chunks": [
            {"file_path": "vacation_policy.docx", "page_number": 2, "content": "휴가 제도 규정..."},
            {"file_path": "org_chart.pptx", "page_number": 1, "content": "조직도..."},
            {"file_path": "ojt_schedule.pdf", "page_number": 3, "content": "신입사원 대상 OJT 교육 세부 시간표 및 강사진 안내"}
        ]
    }
]

# 3. 환각 차단률 테스트 케이스 (지식베이스 외 질문)
OOD_TEST_CASES = [
    {
        "question": "2026년도 화성 탐사선 발사 일정에 대해 알려주세요.",
        "generated_answer": "죄송합니다. 제공된 사내 문서 지식베이스에는 2026년도 화성 탐사선 발사에 관한 정보가 포함되어 있지 않아 답변을 드릴 수 없습니다.",
        "expected_refused": True,
        "description": "알려진 거절 문구를 포함한 정상 차단 사례"
    },
    {
        "question": "사내 식당 메뉴 중 월요일 특식 가격은 얼마인가요?",
        "generated_answer": "월요일 특식은 제육볶음이며 가격은 6,500원입니다. 맛있는 식사 되시기 바랍니다.",
        "expected_refused": False,
        "description": "사내 문서 지식베이스에 해당 정보가 없음에도 임의로 꾸며낸 답변 (환각 차단 실패)"
    }
]


# =========================================================================
# 2. Mock API 응답 모의 정의 (네트워크 단절 시를 위한 대비)
# =========================================================================

def mock_llm_post(url, json, headers, timeout):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code
            self.text = "Mock Error Context"

        def json(self):
            return self.json_data

    prompt = json["messages"][0]["content"]

    # 1. 응답 정확도 평가인 경우
    if "검증하는 전문 평가자" in prompt:
        if "I그룹의 경우 1톤 이상, II그룹의 경우 10톤 이상" in prompt:
            # 완벽히 일치하는 경우의 응답
            return MockResponse({
                "choices": [{
                    "message": {
                        "content": '{\n  "numbers": {"match": 1, "reason": "I그룹 1톤 이상, II그룹 10톤 이상 수치 일치함"},\n  "regulations": {"match": null, "reason": "특이 사항 없음"},\n  "conditions": {"match": 1, "reason": "해당 사업장 기준 일치"},\n  "procedures": {"match": null, "reason": "특이 사항 없음"},\n  "overall_score": 1.0,\n  "reason": "전문가 답변에 언급된 중요 수치 및 기준 조건과 완벽히 일치함"\n}'
                    }
                }]
            }, 200)
        else:
            # 왜곡 답변인 경우
            return MockResponse({
                "choices": [{
                    "message": {
                        "content": '{\n  "numbers": {"match": null, "reason": "특이 사항 없음"},\n  "regulations": {"match": null, "reason": "특이 사항 없음"},\n  "conditions": {"match": null, "reason": "특이 사항 없음"},\n  "procedures": {"match": 0, "reason": "사전 검토 단계가 없다는 왜곡된 절차가 포함됨"},\n  "overall_score": 0.4,\n  "reason": "전문가 답변에 명시된 3단계(사전 검토)를 누락하고 없는 위원회 단계로 설명하여 절차가 불일치함"\n}'
                    }
                }]
            }, 200)

    # 2. 환각 차단률 평가인 경우
    elif "거절했거나(모른다고 답변)" in prompt:
        if "화성 탐사선" in prompt:
            return MockResponse({
                "choices": [{
                    "message": {
                        "content": '{\n  "refused": true,\n  "reason": "정보가 없음을 명시적으로 밝히며 답변을 거부함"\n}'
                    }
                }]
            }, 200)
        else:
            return MockResponse({
                "choices": [{
                    "message": {
                        "content": '{\n  "refused": false,\n  "reason": "정보가 없는 질문에 대해 제육볶음 가격 6,500원이라는 임의 답변을 함"\n}'
                    }
                }]
            }, 200)

    return MockResponse({}, 400)


# =========================================================================
# 3. 유닛 테스트 및 실행 검증 클래스
# =========================================================================

class TestRAGEvaluator(unittest.TestCase):

    @patch('requests.post', side_effect=mock_llm_post)
    def test_response_accuracy(self, mock_post):
        """1. 응답 정확도(Factual Accuracy) 기능 및 점수 검증"""
        evaluator = ResponseAccuracyEvaluator()
        
        # 케이스 1: 완벽 일치 케이스
        res1 = evaluator.evaluate(
            QA_TEST_CASES[0]["question"],
            QA_TEST_CASES[0]["reference_answer"],
            QA_TEST_CASES[0]["generated_answer"]
        )
        self.assertEqual(res1["overall_score"], 1.0)
        self.assertEqual(res1["numbers"]["match"], 1)

        # 케이스 2: 절차 누락/왜곡 케이스
        res2 = evaluator.evaluate(
            QA_TEST_CASES[1]["question"],
            QA_TEST_CASES[1]["reference_answer"],
            QA_TEST_CASES[1]["generated_answer"]
        )
        self.assertEqual(res2["overall_score"], 0.4)
        self.assertEqual(res2["procedures"]["match"], 0)

    def test_retrieval_precision_hit_rate(self):
        """2. 검색 정밀도 (HitRate@K) 점수 계산 논리 검증"""
        # 케이스 1 (file_path 및 page_number 매칭)
        hit1_k1 = RetrievalPrecisionEvaluator.calculate_hit(
            RETRIEVAL_TEST_CASES[0]["retrieved_chunks"],
            RETRIEVAL_TEST_CASES[0]["ground_truth"],
            k=1
        )
        hit1_k3 = RetrievalPrecisionEvaluator.calculate_hit(
            RETRIEVAL_TEST_CASES[0]["retrieved_chunks"],
            RETRIEVAL_TEST_CASES[0]["ground_truth"],
            k=3
        )
        self.assertEqual(hit1_k1, 1)
        self.assertEqual(hit1_k3, 1)

        # 케이스 2 (parent_id 매칭)
        hit2_k1 = RetrievalPrecisionEvaluator.calculate_hit(
            RETRIEVAL_TEST_CASES[1]["retrieved_chunks"],
            RETRIEVAL_TEST_CASES[1]["ground_truth"],
            k=1
        )
        hit2_k2 = RetrievalPrecisionEvaluator.calculate_hit(
            RETRIEVAL_TEST_CASES[1]["retrieved_chunks"],
            RETRIEVAL_TEST_CASES[1]["ground_truth"],
            k=2
        )
        self.assertEqual(hit2_k1, 0) # 첫 번째 청크는 finance_rule.pdf 이므로 miss
        self.assertEqual(hit2_k2, 1) # 두 번째 청크에서 parent_id 매칭되므로 hit

        # 케이스 3 (text 매칭)
        hit3_k2 = RetrievalPrecisionEvaluator.calculate_hit(
            RETRIEVAL_TEST_CASES[2]["retrieved_chunks"],
            RETRIEVAL_TEST_CASES[2]["ground_truth"],
            k=2
        )
        hit3_k3 = RetrievalPrecisionEvaluator.calculate_hit(
            RETRIEVAL_TEST_CASES[2]["retrieved_chunks"],
            RETRIEVAL_TEST_CASES[2]["ground_truth"],
            k=3
        )
        self.assertEqual(hit3_k2, 0)
        self.assertEqual(hit3_k3, 1)

        # 전체 평균 HitRate@K 검증
        all_chunks = [tc["retrieved_chunks"] for tc in RETRIEVAL_TEST_CASES]
        all_gts = [tc["ground_truth"] for tc in RETRIEVAL_TEST_CASES]
        
        rate_k1 = RetrievalPrecisionEvaluator.calculate_hit_rate(all_chunks, all_gts, k=1)
        rate_k2 = RetrievalPrecisionEvaluator.calculate_hit_rate(all_chunks, all_gts, k=2)
        rate_k3 = RetrievalPrecisionEvaluator.calculate_hit_rate(all_chunks, all_gts, k=3)

        # k=1: Case 1만 Hit -> 1/3 = 33.3%
        # k=2: Case 1, Case 2 Hit -> 2/3 = 66.7%
        # k=3: Case 1, Case 2, Case 3 Hit -> 3/3 = 100.0%
        self.assertAlmostEqual(rate_k1, 1.0/3.0)
        self.assertAlmostEqual(rate_k2, 2.0/3.0)
        self.assertAlmostEqual(rate_k3, 1.0)

    @patch('requests.post', side_effect=mock_llm_post)
    def test_hallucination_blocking(self, mock_post):
        """3. 환각 차단률(지식베이스 외 질문 거절 비율) 검증"""
        evaluator = HallucinationBlockingEvaluator()

        # 케이스 1: 룰 기반 키워드로 모른다고 말해 바로 차단됨
        res1 = evaluator.evaluate_query(
            OOD_TEST_CASES[0]["question"],
            OOD_TEST_CASES[0]["generated_answer"],
            use_llm_fallback=True
        )
        self.assertTrue(res1["refused"])
        self.assertEqual(res1["method"], "rule_keywords")

        # 케이스 2: 거짓으로 꾸며낸 답변 (환각 차단 실패 - refused=False)
        res2 = evaluator.evaluate_query(
            OOD_TEST_CASES[1]["question"],
            OOD_TEST_CASES[1]["generated_answer"],
            use_llm_fallback=True
        )
        self.assertFalse(res2["refused"])

        # 전체 환각 차단률 산출 검증
        rate_report = evaluator.calculate_blocking_rate(OOD_TEST_CASES, use_llm_fallback=True)
        self.assertEqual(rate_report["blocking_rate"], 0.5)
        self.assertEqual(rate_report["blocked_count"], 1)
        self.assertEqual(rate_report["total_count"], 2)


# =========================================================================
# 4. 엔트리포인트 및 종합 데모 리포트 출력 함수
# =========================================================================

def run_evaluation_demo(use_mock=True):
    print("=====================================================================")
    print(f"[*] RAG 문서 검색 및 응답 성능 평가 엔진 작동 (Mock Mode: {use_mock})")
    print("=====================================================================")
    
    if use_mock:
        # Mock 패치를 사용해 실제 API 호출 없이 로직 구동 데모 수행
        with patch('requests.post', side_effect=mock_llm_post):
            orchestrator = RAGEvaluator()
            report = orchestrator.run_evaluation(
                qa_test_cases=QA_TEST_CASES,
                retrieval_test_cases=RETRIEVAL_TEST_CASES,
                ood_test_cases=OOD_TEST_CASES,
                k_list=[1, 2, 3]
            )
    else:
        # 실제 config.py에 적힌 API 및 LLM Gateway 연결을 사용하여 구동
        orchestrator = RAGEvaluator()
        report = orchestrator.run_evaluation(
            qa_test_cases=QA_TEST_CASES,
            retrieval_test_cases=RETRIEVAL_TEST_CASES,
            ood_test_cases=OOD_TEST_CASES,
            k_list=[1, 2, 3]
        )

    # 평가 리포트 마크다운 스타일 출력
    print("\n" + "="*70)
    print("                      [ RAG 종합 평가 리포트 ]")
    print("="*70)
    
    # 1. 응답 정확도 출력
    if "response_accuracy" in report:
        ra = report["response_accuracy"]
        print(f"\n[1] 응답 정확도 (Response Accuracy)")
        print(f"    - 종합 평균 사실 일치도 점수: {ra['average_score'] * 100:.1f}점 (만점 1.0)")
        print(f"    - 평가 진행 질문 개수: {ra['total_count']}건")
        print("    - 세부 검증 내역:")
        for idx, det in enumerate(ra["details"]):
            print(f"      * Q{idx+1}: {det['question']}")
            print(f"        -> 전문가: {det['reference']}")
            print(f"        -> 모델:   {det['generated']}")
            print(f"        -> 사실성 점수: {det['score']:.2f}")
            print(f"        -> 세부 판별: 수치({det['categories']['numbers']}), 규정({det['categories']['regulations']}), 조건({det['categories']['conditions']}), 절차({det['categories']['procedures']})")
            print(f"        -> 사유: {det['reason']}")

    # 2. 검색 정밀도 출력
    if "retrieval_precision" in report:
        rp = report["retrieval_precision"]
        print(f"\n[2] 검색 정밀도 (Retrieval Precision)")
        for metric_name, value in rp.items():
            print(f"    - {metric_name}: {value * 100:.1f}%")

    # 3. 환각 차단률 출력
    if "hallucination_blocking" in report:
        hb = report["hallucination_blocking"]
        print(f"\n[3] 환각 차단률 (Hallucination Blocking Rate)")
        print(f"    - 지식 외 질문 차단률: {hb['blocking_rate'] * 100:.1f}% (성공 {hb['blocked_count']}건 / 전체 {hb['total_count']}건)")
        print("    - 세부 차단 여부:")
        for idx, det in enumerate(hb["details"]):
            status_str = "차단성공(Refused)" if det["is_blocked"] else "차단실패(Hallucinated)"
            print(f"      * Q{idx+1}: {det['question']}")
            print(f"        -> 답변: {det['generated_answer']}")
            print(f"        -> 결과: {status_str} ({det['method']})")
            print(f"        -> 사유: {det['reason']}")
            
    print("="*70)


if __name__ == "__main__":
    # 인자로 '--real'을 전달하면 실제 API 게이트웨이 호출, 없으면 Mock으로 테스트 진행
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        run_evaluation_demo(use_mock=False)
    else:
        # 1. unittest 프레임워크 검증 실행
        print("[*] 유닛 테스트 검증 시작...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRAGEvaluator)
        runner = unittest.TextTestRunner(verbosity=2)
        test_result = runner.run(suite)
        
        # 2. 데모 출력 실행
        if test_result.wasSuccessful():
            run_evaluation_demo(use_mock=True)
        else:
            print("[-] 유닛 테스트 실패로 인해 데모 출력을 중단합니다.")
            sys.exit(1)
