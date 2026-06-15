import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psycopg2
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DB_CONFIG, LITELLM_HEADERS, LITELLM_VLM_URL, VLM_MODEL_NAME
from db_utils import format_pgvector
from nlp_utils import preprocess_korean_fts
from utils import get_harrier_embedding


SEARCH_INTENT_KEYWORDS = (
    "문서",
    "파일",
    "목록",
    "어디",
    "관련 자료",
    "관련자료",
    "찾아",
    "찾기",
    "검색",
    "위치",
)


@dataclass
class SearchResult:
    child_id: str
    parent_id: str
    file_path: str
    file_name: str
    extension: str
    page_number: List[int]
    section_title: str
    child_content: str
    parent_content: str
    image_paths: List[str] = field(default_factory=list)
    vector_distance: Optional[float] = None
    vector_score: float = 0.0
    keyword_score: float = 0.0
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "extension": self.extension,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "child_content": self.child_content,
            "parent_content": self.parent_content,
            "image_paths": self.image_paths,
            "vector_distance": self.vector_distance,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "score": self.score,
        }


class EnhancedDocumentChat:
    def __init__(self, db_config: Optional[dict] = None, top_k: int = 8):
        self.db_config = db_config or DB_CONFIG
        self.top_k = top_k

    def connect(self):
        return psycopg2.connect(**self.db_config)

    def ask(
        self,
        question: str,
        user_id: str,
        session_id: Optional[str] = None,
        mode: str = "auto",
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        selected_mode = self.resolve_mode(question, mode)
        started_at = time.perf_counter()
        search_results = self.search_documents(question, top_k=top_k or self.top_k)

        with self.connect() as conn:
            with conn.cursor() as cursor:
                if session_id is None:
                    session_id = self.create_session(cursor, user_id, question[:80])
                self.save_message(cursor, session_id, "user", question)

                if selected_mode == "search":
                    answer = self.format_search_results(search_results)
                    prompt_tokens = None
                    completion_tokens = None
                else:
                    answer_payload = self.answer_from_documents(question, search_results, session_id=session_id)
                    answer = answer_payload["answer"]
                    prompt_tokens = answer_payload.get("prompt_tokens")
                    completion_tokens = answer_payload.get("completion_tokens")

                latency_ms = int((time.perf_counter() - started_at) * 1000)
                referenced_chunk_ids = [result.child_id for result in search_results]
                self.save_message(
                    cursor,
                    session_id,
                    "assistant",
                    answer,
                    referenced_chunk_ids=referenced_chunk_ids,
                    used_model=VLM_MODEL_NAME,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                )
                conn.commit()

        return {
            "session_id": session_id,
            "mode": selected_mode,
            "answer": answer,
            "results": [result.to_dict() for result in search_results],
        }

    def resolve_mode(self, question: str, mode: str) -> str:
        if mode not in {"auto", "search", "answer"}:
            raise ValueError("mode must be one of: auto, search, answer")
        if mode != "auto":
            return mode
        return "search" if any(keyword in question for keyword in SEARCH_INTENT_KEYWORDS) else "answer"

    def search_documents(self, question: str, top_k: int = 8) -> List[SearchResult]:
        query_embedding = get_harrier_embedding(question)
        keyword_query = preprocess_korean_fts(question) or question
        candidates: Dict[str, SearchResult] = {}

        with self.connect() as conn:
            with conn.cursor() as cursor:
                if query_embedding:
                    self._collect_vector_results(cursor, candidates, query_embedding, top_k * 3)
                if keyword_query.strip():
                    self._collect_keyword_results(cursor, candidates, keyword_query, top_k * 3)

        ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        return self._dedupe_by_file_path(ranked)[:top_k]

    def _collect_vector_results(self, cursor, candidates: Dict[str, SearchResult], query_embedding, limit: int):
        vector_literal = format_pgvector(query_embedding)
        cursor.execute(
            """
            SELECT
                c.child_id::text,
                p.parent_id::text,
                p.file_path,
                ft.file_name,
                ft.extension,
                p.page_number,
                COALESCE(p.section_title, ''),
                c.content,
                p.content,
                COALESCE(p.image_paths, '{}'),
                (c.embedding <=> %s::vector) AS vector_distance
            FROM document_children c
            JOIN document_parents p ON p.parent_id = c.parent_id
            JOIN file_tracker ft ON ft.file_path = p.file_path
            WHERE c.embedding IS NOT NULL
              AND ft.status = 'COMPLETED'
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal, vector_literal, limit),
        )
        for row in cursor.fetchall():
            result = self._row_to_result(row)
            distance = result.vector_distance if result.vector_distance is not None else 1.0
            result.vector_score = max(0.0, 1.0 - float(distance))
            self._merge_candidate(candidates, result)

    def _collect_keyword_results(self, cursor, candidates: Dict[str, SearchResult], keyword_query: str, limit: int):
        cursor.execute(
            """
            WITH q AS (SELECT plainto_tsquery('simple', %s) AS query)
            SELECT
                c.child_id::text,
                p.parent_id::text,
                p.file_path,
                ft.file_name,
                ft.extension,
                p.page_number,
                COALESCE(p.section_title, ''),
                c.content,
                p.content,
                COALESCE(p.image_paths, '{}'),
                NULL::float AS vector_distance,
                ts_rank(c.fts_vector, q.query) AS keyword_rank
            FROM document_children c
            JOIN document_parents p ON p.parent_id = c.parent_id
            JOIN file_tracker ft ON ft.file_path = p.file_path
            CROSS JOIN q
            WHERE c.fts_vector @@ q.query
              AND ft.status = 'COMPLETED'
            ORDER BY keyword_rank DESC
            LIMIT %s
            """,
            (keyword_query, limit),
        )
        for row in cursor.fetchall():
            result = self._row_to_result(row[:11])
            result.keyword_score = min(float(row[11] or 0.0), 1.0)
            self._merge_candidate(candidates, result)

    def _row_to_result(self, row) -> SearchResult:
        return SearchResult(
            child_id=row[0],
            parent_id=row[1],
            file_path=row[2],
            file_name=row[3],
            extension=row[4],
            page_number=list(row[5] or []),
            section_title=row[6] or "",
            child_content=row[7],
            parent_content=row[8],
            image_paths=list(row[9] or []),
            vector_distance=row[10],
        )

    def _merge_candidate(self, candidates: Dict[str, SearchResult], incoming: SearchResult):
        incoming.score = (incoming.vector_score * 0.75) + (incoming.keyword_score * 0.25)
        current = candidates.get(incoming.parent_id)
        if current is None:
            candidates[incoming.parent_id] = incoming
            return

        current.vector_score = max(current.vector_score, incoming.vector_score)
        current.keyword_score = max(current.keyword_score, incoming.keyword_score)
        current.score = (current.vector_score * 0.75) + (current.keyword_score * 0.25)
        if incoming.score > current.score:
            incoming.vector_score = current.vector_score
            incoming.keyword_score = current.keyword_score
            incoming.score = current.score
            candidates[incoming.parent_id] = incoming

    def _dedupe_by_file_path(self, results: List[SearchResult]) -> List[SearchResult]:
        unique_results: Dict[str, SearchResult] = {}
        for result in results:
            current = unique_results.get(result.file_path)
            if current is None or result.score > current.score:
                unique_results[result.file_path] = result
        return sorted(unique_results.values(), key=lambda item: item.score, reverse=True)

    def answer_from_documents(
        self,
        question: str,
        results: List[SearchResult],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not results:
            return {
                "answer": "관련 문서를 찾지 못했습니다. 문서에서 확인되지 않음.",
                "prompt_tokens": None,
                "completion_tokens": None,
            }

        history = self.load_recent_history(session_id, limit=6) if session_id else []
        context = self.build_context(results)
        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 사내 문서 검색 기반 RAG 비서입니다. "
                    "반드시 제공된 문서 근거 안에서만 답하세요. "
                    "근거가 부족하면 '문서에서 확인되지 않음'이라고 명확히 말하세요. "
                    "답변은 한국어로 간결하고 업무적으로 작성하세요."
                ),
            }
        ]
        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": f"[사용자 질문]\n{question}\n\n[검색된 문서 근거]\n{context}",
            }
        )

        payload = {
            "model": VLM_MODEL_NAME,
            "messages": messages,
            "temperature": 0.2,
            "stream": False,
        }
        response = requests.post(
            LITELLM_VLM_URL,
            json=payload,
            headers=LITELLM_HEADERS,
            timeout=90,
        )
        if response.status_code != 200:
            raise RuntimeError(f"LLM gateway error ({response.status_code}): {response.text}")

        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        answer = f"{answer}\n\n{self.format_sources(results)}"
        usage = data.get("usage", {})
        return {
            "answer": answer,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    def build_context(self, results: List[SearchResult], max_docs: int = 5, max_chars_per_doc: int = 1800) -> str:
        chunks = []
        for idx, result in enumerate(results[:max_docs], start=1):
            pages = ", ".join(str(page) for page in result.page_number) or "-"
            content = result.parent_content[:max_chars_per_doc]
            chunks.append(
                "\n".join(
                    [
                        f"[문서 {idx}]",
                        f"파일명: {result.file_name}",
                        f"경로: {result.file_path}",
                        f"페이지: {pages}",
                        f"섹션: {result.section_title or '-'}",
                        f"parent_id: {result.parent_id}",
                        f"child_id: {result.child_id}",
                        f"내용:\n{content}",
                    ]
                )
            )
        return "\n\n".join(chunks)

    def format_sources(self, results: List[SearchResult]) -> str:
        lines = ["출처:"]
        for idx, result in enumerate(results[:5], start=1):
            pages = ", ".join(str(page) for page in result.page_number) or "-"
            section = result.section_title or "-"
            lines.append(
                f"{idx}. {result.file_name} | 페이지 {pages} | 섹션 {section} | "
                f"child_id={result.child_id} | parent_id={result.parent_id}"
            )
        return "\n".join(lines)

    def format_search_results(self, results: List[SearchResult]) -> str:
        if not results:
            return "관련 문서를 찾지 못했습니다."

        lines = ["관련 문서 검색 결과:"]
        for idx, result in enumerate(results, start=1):
            pages = ", ".join(str(page) for page in result.page_number) or "-"
            snippet = " ".join(result.child_content.split())[:240]
            lines.extend(
                [
                    f"\n{idx}. {result.file_name}",
                    f"   경로: {result.file_path}",
                    f"   페이지: {pages}",
                    f"   섹션: {result.section_title or '-'}",
                    f"   점수: {result.score:.4f}",
                    f"   요약: {snippet}",
                    f"   child_id: {result.child_id}",
                    f"   parent_id: {result.parent_id}",
                ]
            )
        return "\n".join(lines)

    def create_session(self, cursor, user_id: str, title: str = "새로운 대화") -> str:
        cursor.execute(
            """
            INSERT INTO chat_sessions (user_id, title)
            VALUES (%s, %s)
            RETURNING session_id::text
            """,
            (user_id, title or "새로운 대화"),
        )
        return cursor.fetchone()[0]

    def save_message(
        self,
        cursor,
        session_id: str,
        role: str,
        content: str,
        referenced_chunk_ids: Optional[List[str]] = None,
        used_model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        latency_ms: Optional[int] = None,
    ) -> str:
        cursor.execute(
            """
            INSERT INTO chat_messages (
                session_id, role, content, referenced_chunk_ids, used_model,
                prompt_tokens, completion_tokens, latency_ms
            )
            VALUES (%s, %s, %s, %s::uuid[], %s, %s, %s, %s)
            RETURNING message_id::text
            """,
            (
                session_id,
                role,
                content,
                referenced_chunk_ids,
                used_model,
                prompt_tokens,
                completion_tokens,
                latency_ms,
            ),
        )
        message_id = cursor.fetchone()[0]
        cursor.execute(
            "UPDATE chat_sessions SET updated_at = NOW() WHERE session_id = %s",
            (session_id,),
        )
        return message_id

    def load_recent_history(self, session_id: str, limit: int = 6) -> List[Dict[str, str]]:
        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT role, content
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (session_id, limit),
                )
                rows = cursor.fetchall()

        messages = [{"role": role, "content": content} for role, content in reversed(rows)]
        return [message for message in messages if message["role"] in {"user", "assistant"}]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhanced document management system chat CLI")
    parser.add_argument("question", nargs="*", help="사용자 질문")
    parser.add_argument("--user-id", default="local", help="채팅 사용자 ID")
    parser.add_argument("--session-id", help="기존 chat_sessions.session_id")
    parser.add_argument("--mode", choices=("auto", "search", "answer"), default="auto")
    parser.add_argument("--top-k", type=int, default=8)
    return parser


def run_interactive(chat: EnhancedDocumentChat, user_id: str, session_id: Optional[str], mode: str, top_k: int):
    print("Enhanced Document Chat. 종료하려면 exit 또는 quit을 입력하세요.")
    current_session_id = session_id
    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = chat.ask(question, user_id=user_id, session_id=current_session_id, mode=mode, top_k=top_k)
        current_session_id = result["session_id"]
        print(f"\n세션: {current_session_id}")
        print(result["answer"])


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    chat = EnhancedDocumentChat(top_k=args.top_k)
    question = " ".join(args.question).strip()

    if not question:
        run_interactive(chat, args.user_id, args.session_id, args.mode, args.top_k)
        return

    result = chat.ask(
        question,
        user_id=args.user_id,
        session_id=args.session_id,
        mode=args.mode,
        top_k=args.top_k,
    )
    print(f"세션: {result['session_id']}")
    print(result["answer"])


if __name__ == "__main__":
    main()
