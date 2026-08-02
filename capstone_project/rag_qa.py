import argparse
import json
import os
import re
from typing import Any, List

import psycopg2
import requests


DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'lhw_llm'),
    'user': os.getenv('DB_USER', 'dlit'),
    'password': os.getenv('DB_PASSWORD', '*Dlit7004#'),
    'host': os.getenv('DB_HOST', '10.1.55.226'),
    'port': int(os.getenv('DB_PORT', '5432')),
}

API_BASE_URL = os.getenv('API_BASE_URL', 'http://10.1.55.226:30000/v1')
EMBED_MODEL_NAME = os.getenv('EMBED_MODEL_NAME', 'bge-m3')
EMBED_API_URL = os.getenv('EMBED_API_URL', f'{API_BASE_URL}/embeddings')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'gemma-4-26b-a4b-it')
LLM_API_URL = os.getenv('LLM_API_URL', f'{API_BASE_URL}/chat/completions')
API_KEY = os.getenv('API_KEY', 'EMPTY')
NOT_FOUND_TOKEN = 'NOT_FOUND'
NOT_FOUND_MESSAGE = '문서에서 질문에 대한 근거를 찾지 못했습니다.'
MAX_DISTANCE_ENV = os.getenv('MAX_DISTANCE', '').strip()
DEBUG_MODE = os.getenv('DEBUG', '').lower() in {'1', 'true', 'yes'}
AUTO_DISTANCE_MARGIN = 0.15
AUTO_DISTANCE_CAP = 1.0
REQUIRE_CITATION = os.getenv('REQUIRE_CITATION', 'false').lower() in {'1', 'true', 'yes'}


def _extract_openai_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ''

    choices = payload.get('choices')
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get('message', {})
            if isinstance(message, dict):
                content = message.get('content')
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict):
                            text = part.get('text')
                            if isinstance(text, str):
                                parts.append(text)
                    return '\n'.join(parts).strip()

    content = payload.get('content')
    return content.strip() if isinstance(content, str) else ''


def _vector_to_literal(vector: List[float]) -> str:
    return '[' + ','.join(f'{x:.10f}' for x in vector) + ']'


def get_embedding(text: str) -> List[float]:
    headers = {'Content-Type': 'application/json'}
    if API_KEY and API_KEY.upper() != 'EMPTY':
        headers['Authorization'] = f'Bearer {API_KEY}'

    response = requests.post(
        EMBED_API_URL,
        headers=headers,
        json={'model': EMBED_MODEL_NAME, 'input': text},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return payload['data'][0]['embedding']


def search_relevant_chunks(question: str, top_k: int = 5) -> List[dict[str, Any]]:
    query_embedding = get_embedding(question)
    embedding_literal = _vector_to_literal(query_embedding)

    sql = """
        SELECT
            dc.content,
            dc.metadata,
            f.file_name,
            f.file_path,
            (dc.embedding <=> %s::vector) AS distance
        FROM document_chunks dc
        JOIN files f ON dc.file_id = f.id
        WHERE dc.embedding IS NOT NULL
        ORDER BY dc.embedding <=> %s::vector
        LIMIT %s
    """

    rows: List[dict[str, Any]] = []
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (embedding_literal, embedding_literal, top_k))
            for content, metadata, file_name, file_path, distance in cur.fetchall():
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {'raw': metadata}
                rows.append(
                    {
                        'content': content,
                        'metadata': metadata if isinstance(metadata, dict) else {},
                        'file_name': file_name,
                        'file_path': file_path,
                        'distance': float(distance) if distance is not None else None,
                    }
                )

    return rows


def build_context(chunks: List[dict[str, Any]], max_chars: int = 7000) -> str:
    parts: List[str] = []
    total = 0

    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get('metadata', {})
        chunk_index = metadata.get('chunk_index', '?') if isinstance(metadata, dict) else '?'
        source = f"{chunk.get('file_name', 'unknown')} | chunk_index={chunk_index}"
        text = str(chunk.get('content', '')).strip()
        block = f"[S{i}] SOURCE: {source}\n{text}\n"

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return '\n'.join(parts)


def ask_llm(question: str, context: str) -> str:
    headers = {'Content-Type': 'application/json'}
    if API_KEY and API_KEY.upper() != 'EMPTY':
        headers['Authorization'] = f'Bearer {API_KEY}'

    system_prompt = (
        'You are a helpful assistant for document QA. '
        'Answer only from the provided context. '
        f'If evidence is insufficient, output exactly "{NOT_FOUND_TOKEN}" and nothing else. '
        'If evidence is sufficient, answer in Korean and cite source ids like [S1], [S2]. '
        'Never use outside knowledge.'
    )
    user_prompt = (
        f'Question:\n{question}\n\n'
        f'Context:\n{context}\n\n'
        'Return a concise, factual answer in Korean.'
    )

    payload = {
        'model': LLM_MODEL_NAME,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': 0.2,
        'max_tokens': 1024,
    }

    response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return _extract_openai_text(response.json())


def ask_llm_with_retry(question: str, context: str) -> str:
    answer = ask_llm(question, context)
    if answer.upper() == NOT_FOUND_TOKEN or _contains_source_citation(answer):
        return answer

    retry_instruction = (
        'You must include citations like [S1], [S2] after each key claim. '
        f'If you cannot, output exactly "{NOT_FOUND_TOKEN}".'
    )
    headers = {'Content-Type': 'application/json'}
    if API_KEY and API_KEY.upper() != 'EMPTY':
        headers['Authorization'] = f'Bearer {API_KEY}'

    payload = {
        'model': LLM_MODEL_NAME,
        'messages': [
            {
                'role': 'system',
                'content': (
                    'You are a helpful assistant for document QA. '
                    'Answer only from the provided context. '
                    f'If evidence is insufficient, output exactly "{NOT_FOUND_TOKEN}" and nothing else. '
                    'Never use outside knowledge.'
                ),
            },
            {'role': 'user', 'content': f'Question:\n{question}\n\nContext:\n{context}'},
            {'role': 'user', 'content': retry_instruction},
        ],
        'temperature': 0.0,
        'max_tokens': 1024,
    }

    response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return _extract_openai_text(response.json())


def _contains_source_citation(text: str) -> bool:
    return bool(re.search(r'\[S\d+\]', text))


def _extract_cited_indices(text: str) -> List[int]:
    return sorted({int(match.group(1)) for match in re.finditer(r'\[S(\d+)\]', text)})


def _filter_chunks_by_citations(answer: str, chunks: List[dict[str, Any]]) -> List[dict[str, Any]]:
    indices = _extract_cited_indices(answer)
    if not indices:
        return []
    filtered: List[dict[str, Any]] = []
    for idx in indices:
        if 1 <= idx <= len(chunks):
            filtered.append(chunks[idx - 1])
    return filtered


def _parse_max_distance(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _adaptive_max_distance(chunks: List[dict[str, Any]]) -> float | None:
    distances = [chunk.get('distance') for chunk in chunks]
    numeric = [d for d in distances if isinstance(d, float)]
    if not numeric:
        return None
    best = min(numeric)
    return min(best + AUTO_DISTANCE_MARGIN, AUTO_DISTANCE_CAP)


def _normalize_answer(raw_answer: str) -> str:
    answer = raw_answer.strip()
    if not answer:
        if DEBUG_MODE:
            print('[DEBUG] 답변이 비어있음')
        return NOT_FOUND_MESSAGE

    if answer.upper() == NOT_FOUND_TOKEN:
        if DEBUG_MODE:
            print(f'[DEBUG] 모델이 NOT_FOUND 반환')
        return NOT_FOUND_MESSAGE

    # REQUIRE_CITATION이 true인 경우만 출처 인용 필수
    if REQUIRE_CITATION and not _contains_source_citation(answer):
        if DEBUG_MODE:
            print(f'[DEBUG] 출처 인용 없음 (required). 답변: {answer[:100]}...')
        return NOT_FOUND_MESSAGE

    return answer


def answer_question(
    question: str,
    top_k: int = 5,
    max_distance: float | None = None,
) -> tuple[str, List[dict[str, Any]]]:
    chunks = search_relevant_chunks(question, top_k=top_k)
    if not chunks:
        if DEBUG_MODE:
            print('[DEBUG] 검색 결과 없음')
        return NOT_FOUND_MESSAGE, []

    if DEBUG_MODE:
        distances = [chunk.get('distance') for chunk in chunks]
        print(f'[DEBUG] 검색된 청크: {len(chunks)}개, 거리: {[f"{d:.4f}" for d in distances]}')

    if max_distance is None:
        max_distance = _adaptive_max_distance(chunks)

    if DEBUG_MODE:
        distance_str = f'{max_distance:.4f}' if max_distance is not None else '없음'
        print(f'[DEBUG] 적용 거리 필터: {distance_str}')

    if max_distance is not None:
        filtered_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk.get('distance'), float) and chunk['distance'] <= max_distance
        ]
        if DEBUG_MODE:
            print(f'[DEBUG] 거리 필터 후: {len(filtered_chunks)}개 청크')
        if not filtered_chunks:
            return NOT_FOUND_MESSAGE, []
        chunks = filtered_chunks

    context = build_context(chunks)
    if not context.strip():
        if DEBUG_MODE:
            print('[DEBUG] 컨텍스트 구성 실패')
        return NOT_FOUND_MESSAGE, chunks

    if DEBUG_MODE:
        print(f'[DEBUG] LLM 호출 중... 컨텍스트 길이: {len(context)}')

    raw_answer = ask_llm_with_retry(question, context)
    if DEBUG_MODE:
        print(f'[DEBUG] LLM 원본 응답: {raw_answer[:150]}...')

    answer = _normalize_answer(raw_answer)
    if answer == NOT_FOUND_MESSAGE:
        if DEBUG_MODE:
            print('[DEBUG] 정규화 후 NOT_FOUND')
        return answer, []

    # 출처 인용이 있으면 인용된 청크만 사용, 없으면 모든 검색 청크 사용
    if _contains_source_citation(answer):
        used_chunks = _filter_chunks_by_citations(answer, chunks)
        if DEBUG_MODE:
            print(f'[DEBUG] 인용된 청크: {len(used_chunks)}개')
        if not used_chunks:
            if DEBUG_MODE:
                print('[DEBUG] 인용 필터링 후 청크 없음, 모든 검색 청크 사용')
            used_chunks = chunks
    else:
        if DEBUG_MODE:
            print(f'[DEBUG] 출처 인용 없음, 모든 검색 청크 {len(chunks)}개 사용')
        used_chunks = chunks

    return answer, used_chunks


def print_result(question: str, answer: str, chunks: List[dict[str, Any]]) -> None:
    print('\n=== Question ===')
    print(question)

    print('\n=== Answer ===')
    print(answer if answer else '모델 응답이 비어 있습니다.')

    print('\n=== Evidence ===')
    if not chunks:
        print('근거 없음')
        return

    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get('metadata', {})
        chunk_index = metadata.get('chunk_index', '?') if isinstance(metadata, dict) else '?'
        snippet = str(chunk.get('content', '')).replace('\n', ' ').strip()
        snippet = snippet[:220] + ('...' if len(snippet) > 220 else '')
        distance = chunk.get('distance')
        distance_str = f'{distance:.4f}' if isinstance(distance, float) else 'N/A'

        print(
            f'[S{i}] file={chunk.get("file_path", "unknown")} '
            f'chunk_index={chunk_index} distance={distance_str}'
        )
        print(f'     snippet: {snippet}')


def run_interactive(top_k: int, max_distance: float | None) -> None:
    global DEBUG_MODE
    if DEBUG_MODE:
        print('[DEBUG] 디버그 모드 활성화')
    print('RAG QA started. Type your question, or type "exit" to quit.')
    print('(Tip: Type "debug" to toggle debug mode)')
    while True:
        question = input('\nQuestion> ').strip()
        if question.lower() in {'exit', 'quit'}:
            print('Bye.')
            return
        if question.lower() == 'debug':
            DEBUG_MODE = not DEBUG_MODE
            print(f'Debug mode: {"ON" if DEBUG_MODE else "OFF"}')
            continue
        if not question:
            continue

        try:
            answer, chunks = answer_question(question, top_k=top_k, max_distance=max_distance)
            print_result(question, answer, chunks)
        except Exception as exc:
            print(f'Error: {exc}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Document retrieval + grounded QA')
    parser.add_argument('--question', type=str, default='', help='Single question to ask')
    parser.add_argument('--top-k', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument(
        '--max-distance',
        type=float,
        default=_parse_max_distance(MAX_DISTANCE_ENV),
        help='Optional max cosine distance threshold for retrieval',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output',
    )
    parser.add_argument(
        '--require-citation',
        action='store_true',
        help='Require citations in LLM response',
    )
    global DEBUG_MODE, REQUIRE_CITATION
    args = parser.parse_args()
    if args.debug:
        DEBUG_MODE = True
    if args.require_citation:
        REQUIRE_CITATION = True

    if args.question:
        answer, chunks = answer_question(args.question, top_k=args.top_k, max_distance=args.max_distance)
        print_result(args.question, answer, chunks)
        return

    run_interactive(top_k=args.top_k, max_distance=args.max_distance)


if __name__ == '__main__':
    main()
