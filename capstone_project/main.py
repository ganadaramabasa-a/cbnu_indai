# main.py
import os
import psycopg2
from config import DB_CONFIG, SHARED_FOLDER
from pipeline import process_single_file

def run_pipeline(shared_folder_path):
    print("====================================================")
    print("[*] Enterprise RAG 자산 수집 파이프라인 엔진 가동 시작")
    print("====================================================")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[-] Database Connection Error: {e}")
        return

    # 마운트된 전체 스토리지 순회 디스커버리
    for root, _, files in os.walk(shared_folder_path):
        for file in files:
            # Office 제품군 열람 임시 가상 스왑 파일(~$...) 및 OS 숨김 파일 자동 차단
            if file.startswith("~$") or file.startswith("."):
                continue
                
            file_path = os.path.abspath(os.path.join(root, file))
            process_single_file(conn, file_path)
            
    conn.close()
    print("====================================================")
    print("[+] 모든 공유 스토리지 문서 데이터 자산 동기화가 종료되었습니다.")
    print("====================================================")

if __name__ == "__main__":
    run_pipeline(SHARED_FOLDER)
