# pipeline.py
import os
import hashlib
from utils import get_file_metadata, calculate_file_hash, get_harrier_embedding
from parsers import AdvancedDocumentParser
from chunker import build_parent_child_packets
from db_utils import (
    delete_indexed_data_for_file,
    fetch_completed_file_by_hash,
    fetch_file_tracker,
    insert_parent_child_chunks,
    mark_file_completed_without_indexing,
    update_file_metadata,
)

def process_single_file(conn, file_path):
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()[1:]
    
    # 예외 블록 변수 초기화 (초기 단계 실패 시 except 절에서 안전하게 참조하기 위함)
    current_size = 0
    current_mtime = 0
    current_hash = None
    
    # 💡 [개선] 함수 전체를 try-except로 감싸 초기 파일 IO나 해시 계산 오류까지 완벽 차단합니다.
    try:
        cursor = conn.cursor()
        
        # 1. 현재 물리 파일의 메타데이터 조회
        current_mtime, current_size = get_file_metadata(file_path)
        
        db_record = fetch_file_tracker(cursor, file_path)
        if db_record:
            db_mtime, db_size, db_hash, db_status = db_record
            
            # 💡 과거에 'COMPLETED'로 성공했던 파일만 스킵 검증을 진행합니다.
            if db_status == 'COMPLETED':
                if current_mtime == db_mtime and current_size == db_size:
                    print(f"[Skip] {file_name} : 기존에 성공적으로 완료된 파일이며 메타데이터 변동 없음.")
                    return

                current_hash = calculate_file_hash(file_path)
                if current_hash == db_hash:
                    print(f"[Update Status] {file_name} : 파일 내용(해시)은 같으므로 메타데이터 및 상태 정보만 동기화.")
                    update_file_metadata(cursor, file_path, current_size, current_mtime)
                    conn.commit()
                    return
                    
            # 💡 성공(COMPLETED) 상태가 아닌 경우 처리 방침 로그 출력
            elif db_status == 'PROCESSING':
                print(f"[*] [Retry Zombie] {file_name} : 이전 작업이 중간에 비정상 종료(PROCESSING)된 자산입니다. 재수집을 시작합니다.")
                
            elif db_status == 'FAILED':
                print(f"[*] [Retry Failed] {file_name} : 이전에 처리에 실패(FAILED)한 기록이 있어 다시 시도합니다.")

        if current_hash is None:
            current_hash = calculate_file_hash(file_path)

        duplicate_record = fetch_completed_file_by_hash(cursor, current_hash, exclude_file_path=file_path)
        if duplicate_record:
            duplicate_file_path = duplicate_record[0]
            delete_indexed_data_for_file(cursor, file_path)
            mark_file_completed_without_indexing(
                cursor,
                file_path,
                file_name,
                ext,
                current_size,
                current_mtime,
                current_hash,
            )
            conn.commit()
            print(f"[Skip Duplicate] {file_name} : 동일 해시 데이터가 이미 처리되어 스킵합니다. 원본: {duplicate_file_path}")
            return
        
        # 스킵 처리가 되지 않은 대상들은 정밀 파싱 단계로 진입
        print(f"[*] [Ingestion Start] {file_name} 심층 파싱 및 벡터화 파이프라인 가동...")
        
        # 작업 시작 전 즉시 테이블 상태를 'PROCESSING'으로 밀어 넣고 에러 메세지 초기화
        cursor.execute("""
            INSERT INTO file_tracker (file_path, file_name, extension, file_size, last_modified, file_hash, status, error_message, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'PROCESSING', NULL, NOW())
            ON CONFLICT (file_path) DO UPDATE SET 
                file_size = EXCLUDED.file_size, 
                last_modified = EXCLUDED.last_modified, 
                file_hash = EXCLUDED.file_hash,
                status = 'PROCESSING',
                error_message = NULL,
                updated_at = NOW();
        """, (file_path, file_name, ext, current_size, current_mtime, current_hash))
        conn.commit()

        # 구조 기반 문서 레이아웃 파서 구동
        parser = AdvancedDocumentParser(file_path, current_hash)
        raw_pages = parser.parse()
        
        # 파서 데이터 스키마 보정 (file_path 주입)
        for p in raw_pages:
            p["file_path"] = file_path
        
        # Step 2. 부모-자식 패킷 분리 및 구조화
        parent_data, child_raw_list = build_parent_child_packets(raw_pages)
        
        # [데이터 무결성 방어] 재시도 파일의 기존 부모/자식 청크를 같은 트랜잭션 안에서 일괄 삭제
        delete_indexed_data_for_file(cursor, file_path)
        
        # Step 3. 자식 청크에 대해 하리어 임베딩 벡터 동적 추출
        child_final_data = []
        for c in child_raw_list:
            embedding_vector = get_harrier_embedding(c["content"])
            if not embedding_vector:
                raise Exception("LiteLLM harrier 임베딩 벡터 생성 실패 (Gateway 혹은 모델 확인 필요)")
            child_final_data.append((
                c["child_id"],
                c["parent_id"],
                c["content"],
                embedding_vector,
                c["fts_source"]
            ))
            
        # Step 4. 관계형 벌크 데이터베이스 주입
        insert_parent_child_chunks(cursor, parent_data, child_final_data)
        
        # ⭐️ 모든 과정이 에러 없이 완전히 끝났을 때만 최종 'COMPLETED' 마크 획득
        cursor.execute("""
            UPDATE file_tracker 
            SET status = 'COMPLETED', error_message = NULL, last_indexed_at = NOW() 
            WHERE file_path = %s
        """, (file_path,))
        conn.commit()
        print(f"[+] [Success] {file_name} : {len(child_final_data)}개 청크 파싱 및 벡터화 적재 완료 최종 확정.")

    except Exception as e:
        # 1. 💡 가장 먼저 트랜잭션을 롤백하여 깨진 SQL 상태나 어설픈 적재 상태를 초기화합니다.
        conn.rollback()
        error_msg = str(e).replace("\x00", "")
        print(f"[-] [Error Failed] {file_name} 파이프라인 처리 중단: {error_msg}")

        # file_tracker.file_hash는 NOT NULL 제약이므로 실패 로그에서도 안전한 해시값을 확보합니다.
        failed_hash = current_hash
        if not failed_hash:
            try:
                failed_hash = calculate_file_hash(file_path)
            except Exception:
                failed_hash = hashlib.md5(file_path.encode("utf-8", errors="ignore")).hexdigest()
        
        # 2. 💡 격리된 깨끗한 서브 커서를 열어 UPSERT(ON CONFLICT) 기법으로 상태를 안전하게 기록합니다.
        # 이렇게 하면 최초 파일 오픈 실패나 메타데이터 에러 단계에서 죽더라도 file_tracker 기록이 무조건 보장됩니다.
        try:
            with conn.cursor() as err_cursor:
                err_cursor.execute("""
                    INSERT INTO file_tracker (
                        file_path, file_name, extension, file_size, last_modified,
                        file_hash, status, error_message, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, 'FAILED', %s, NOW())
                    ON CONFLICT (file_path) DO UPDATE SET 
                        file_hash = EXCLUDED.file_hash,
                        status = 'FAILED',
                        error_message = EXCLUDED.error_message,
                        updated_at = NOW();
                """, (file_path, file_name, ext, current_size, current_mtime, failed_hash, error_msg))
                conn.commit()
                print(f"[*] [Status Logged] {file_name} 의 FAILED 상태가 데이터베이스에 안전하게 기록되었습니다.")
        except Exception as db_err:
            conn.rollback()
            print(f"[-] [Critical DB Error] 파일 트래커 상태 마킹 도중 최종 실패: {db_err}")
            
        # 3. 💡 예외를 상위 루프로 던지지 않고 정상 리턴(None)시킴으로써, 상위 루프가 무중단으로 다음 파일을 진행하도록 유도합니다.
        return
