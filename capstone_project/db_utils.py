# db_utils.py
import psycopg2
from psycopg2.extras import execute_values


def _strip_nul(value):
    """PostgreSQL TEXT에 저장 불가능한 NUL 문자를 제거합니다."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def format_pgvector(vector):
    """pgvector가 받을 수 있는 문자열 리터럴로 임베딩을 변환합니다."""
    if vector is None:
        return None
    if isinstance(vector, str):
        return vector
    return "[" + ",".join(str(float(v)) for v in vector) + "]"


def fetch_file_tracker(cursor, file_path):
    cursor.execute(
        "SELECT last_modified, file_size, file_hash, status FROM file_tracker WHERE file_path = %s",
        (file_path,),
    )
    return cursor.fetchone()


def fetch_completed_file_by_hash(cursor, file_hash, exclude_file_path=None):
    if exclude_file_path:
        cursor.execute(
            """
            SELECT file_path
            FROM file_tracker
            WHERE file_hash = %s AND status = 'COMPLETED' AND file_path <> %s
            LIMIT 1
            """,
            (file_hash, exclude_file_path),
        )
    else:
        cursor.execute(
            """
            SELECT file_path
            FROM file_tracker
            WHERE file_hash = %s AND status = 'COMPLETED'
            LIMIT 1
            """,
            (file_hash,),
        )
    return cursor.fetchone()


def update_file_metadata(cursor, file_path, file_size, last_modified):
    cursor.execute(
        """
        UPDATE file_tracker
        SET last_modified = %s, file_size = %s, updated_at = NOW()
        WHERE file_path = %s
        """,
        (last_modified, file_size, file_path),
    )


def mark_file_completed_without_indexing(
    cursor,
    file_path,
    file_name,
    extension,
    file_size,
    last_modified,
    file_hash,
):
    cursor.execute(
        """
        INSERT INTO file_tracker (
            file_path, file_name, extension, file_size, last_modified,
            file_hash, status, error_message, last_indexed_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, 'COMPLETED', NULL, NOW(), NOW())
        ON CONFLICT (file_path) DO UPDATE SET
            file_size = EXCLUDED.file_size,
            last_modified = EXCLUDED.last_modified,
            file_hash = EXCLUDED.file_hash,
            status = 'COMPLETED',
            error_message = NULL,
            last_indexed_at = NOW(),
            updated_at = NOW()
        """,
        (file_path, file_name, extension, file_size, last_modified, file_hash),
    )


def mark_file_processing(
    cursor,
    file_path,
    file_name,
    extension,
    file_size,
    last_modified,
    file_hash,
):
    cursor.execute(
        """
        INSERT INTO file_tracker (
            file_path, file_name, extension, file_size, last_modified,
            file_hash, status, error_message, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, 'PROCESSING', NULL, NOW())
        ON CONFLICT (file_path) DO UPDATE SET
            file_size = EXCLUDED.file_size,
            last_modified = EXCLUDED.last_modified,
            file_hash = EXCLUDED.file_hash,
            status = 'PROCESSING',
            error_message = NULL,
            updated_at = NOW()
        """,
        (file_path, file_name, extension, file_size, last_modified, file_hash),
    )


def delete_document_chunks(cursor, file_path):
    cursor.execute("DELETE FROM document_chunks WHERE file_path = %s", (file_path,))


def delete_indexed_data_for_file(cursor, file_path):
    # document_children는 document_parents FK cascade로 함께 삭제됩니다.
    cursor.execute("DELETE FROM document_parents WHERE file_path = %s", (file_path,))
    cursor.execute("DELETE FROM document_chunks WHERE file_path = %s", (file_path,))


def insert_document_chunks(cursor, bulk_data, page_size=100):
    """
    가공이 끝난 정형 데이터를 받아 PostgreSQL document_chunks 테이블에 벌크 인서트를 수행합니다.
    커밋/롤백은 호출자가 관리합니다.
    """
    if not bulk_data:
        print("[!] DB에 주입할 데이터가 비어있습니다. 작업을 스킵합니다.")
        return

    insert_query = """
        INSERT INTO document_chunks (
            file_path, chunk_index, content, embedding, page_number, section_title, fts_vector
        )
        VALUES %s
    """

    sanitized_bulk_data = [
        (
            file_path,
            chunk_index,
            _strip_nul(content),
            embedding,
            page_number,
            _strip_nul(section_title),
            _strip_nul(fts_source),
        )
        for file_path, chunk_index, content, embedding, page_number, section_title, fts_source in bulk_data
    ]

    execute_values(
        cursor,
        insert_query,
        sanitized_bulk_data,
        template="(%s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))",
        page_size=page_size,
    )


def mark_file_completed(cursor, file_path):
    cursor.execute(
        """
        UPDATE file_tracker
        SET status = 'COMPLETED', error_message = NULL, last_indexed_at = NOW()
        WHERE file_path = %s
        """,
        (file_path,),
    )


def mark_file_failed(cursor, file_path, error_message):
    cursor.execute(
        """
        UPDATE file_tracker
        SET status = 'FAILED', error_message = %s, updated_at = NOW()
        WHERE file_path = %s
        """,
        (_strip_nul(error_message), file_path),
    )


def insert_document_chunks_with_config(db_config: dict, bulk_data: list, page_size: int = 100):
    """
    별도 DB 설정으로 새 커넥션을 열어 document_chunks를 벌크 인서트합니다.
    """
    if not bulk_data:
        print("[!] DB에 주입할 데이터가 비어있습니다. 작업을 스킵합니다.")
        return

    conn = None
    try:
        # 데이터베이스 연결 커넥션 생성
        conn = psycopg2.connect(**db_config)
        with conn.cursor() as cursor:
            insert_document_chunks(cursor, bulk_data, page_size=page_size)
            conn.commit()
            print(f"[+] [Data Layer] {len(bulk_data)}개의 청크 데이터가 성공적으로 적재(Commit)되었습니다.")
            
    except Exception as e:
        if conn:
            conn.rollback()
            print("[Data Layer] 데이터베이스 트랜잭션 처리 중 에러가 발생하여 롤백(Rollback)되었습니다.")
        raise e
        
    finally:
        if conn:
            conn.close()


def insert_parent_child_chunks(cursor, parent_data: list, child_data: list, page_size: int = 100):
    """부모 데이터를 먼저 넣은 뒤, 호출자 트랜잭션 안에서 자식 데이터를 벌크 인서트합니다."""
    if not parent_data or not child_data:
        raise ValueError("부모/자식 청크 데이터가 비어 있어 DB 적재를 중단합니다.")
    
    parent_query = """
        INSERT INTO document_parents (parent_id, file_path, page_number, section_title, content, image_paths)
        VALUES %s
    """
    child_query = """
        INSERT INTO document_children (child_id, parent_id, content, embedding, fts_vector)
        VALUES %s
    """

    normalized_parent_data = [
        (
            parent_id,
            file_path,
            page_number,
            _strip_nul(section_title),
            _strip_nul(content),
            [_strip_nul(path) for path in (image_paths or [])],
        )
        for parent_id, file_path, page_number, section_title, content, image_paths in parent_data
    ]

    normalized_child_data = [
        (child_id, parent_id, _strip_nul(content), format_pgvector(embedding), _strip_nul(fts_source))
        for child_id, parent_id, content, embedding, fts_source in child_data
    ]

    execute_values(cursor, parent_query, normalized_parent_data, page_size=page_size)
    execute_values(
        cursor,
        child_query,
        normalized_child_data,
        template="(%s, %s, %s, %s::vector, to_tsvector('simple', %s))",
        page_size=page_size,
    )
    print(f"[+] [DB 적재 완료] 부모 {len(parent_data)}개 / 자식 {len(child_data)}개 적재 성공.")
