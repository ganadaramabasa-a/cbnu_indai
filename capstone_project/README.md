rag_pipeline/
│
├── config.py # 1. 환경 설정 및 API 주소 관리
├── utils.py # 2. 해시 계산 및 AI 모델(임베딩, VLM) API 통신
├── parsers.py # 3. 문서 확장자별 레이아웃 파싱 및 이미지 추출
├── pipeline.py # 4. 하이브리드 검증(1차/2차) 및 DB 적재 로직
└── main.py # 5. 파이프라인 총괄 구동 엔트리포인트
