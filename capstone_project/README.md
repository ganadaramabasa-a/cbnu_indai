# Git Push Summary

## 변경 내용

- `doc_embed.py`
  - PDF 변환 파이프라인에 OCR 온/오프 옵션 추가
  - `ENABLE_OCR` 환경 변수를 사용해 OCR 처리 여부를 제어하도록 수정
  - OCR 비활성 시 텍스트 레이어만으로 문서 변환 수행

- `rag_qa.py`
  - 문서 기반 QA 검색 및 응답 생성 스크립트 작성
  - PostgreSQL에서 `files` 및 `document_chunks` 테이블을 사용해 유사도 검색 수행
  - `gemma-4-26b-a4b-it` 모델을 호출하도록 구성
  - 디버그 모드(`--debug`) 및 거리 임계값(`--max-distance`) 옵션 추가
  - 출처 인용(`--require-citation`) 옵션 추가
  - 잘못된 f-string 포맷 오류 수정
  - `debug` 명령어로 대화형 모드에서도 디버그 토글 가능

## 실행 방법

- 대화형 QA 실행

  ```bash
  python rag_qa.py
  ```

- 단일 질문 실행

  ```bash
  python rag_qa.py --question "질문 내용"
  ```

- 디버그 모드 실행

  ```bash
  python rag_qa.py --question "질문 내용" --debug
  ```

- OCR 끄고 문서 변환
  ```bash
  set ENABLE_OCR=false
  python doc_embed.py
  ```

## 주의 사항

- PostgreSQL 연결 정보는 `rag_qa.py`와 `doc_embed.py` 내에 기본값으로 설정되어 있음
- API 주소 및 키는 환경변수로 대체 가능
- `rag_qa.py`는 `document_chunks.embedding` 컬럼을 이용한 벡터 검색을 전제로 함
