import io
import os
import json
import base64
import importlib
import tempfile
import gc
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, List, Literal, Type

import requests
import psycopg2
from pydantic import ConfigDict, Field

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions, PdfPipelineOptions, TableStructureV2Options
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.factories import get_ocr_factory
from docling.utils.profiling import TimeRecorder

from langchain_text_splitters import RecursiveCharacterTextSplitter


# 사용자 정의 OCR 모델이 추출한 페이지 텍스트를 convert 단계로 전달하기 위한 버퍼
OCR_PAGE_TEXT_BUFFER: List[str] = []


class RemotePaddleOcrOptions(OcrOptions):
    kind: ClassVar[Literal['remote_paddle_ocr']] = 'remote_paddle_ocr'
    lang: List[str] = ['ko', 'en']
    model: str = os.getenv('OCR_MODEL_NAME', 'paddleocr-vl')
    api_key: str = os.getenv('OCR_API_KEY', 'EMPTY')
    prompt: str = os.getenv(
        'OCR_PROMPT',
        'Extract every visible text exactly as written from this image. Return plain text only. Do not output markdown, HTML comments, or image placeholders.',
    )
    max_tokens: int = Field(default=2048, gt=0)
    timeout: float = Field(default=180.0, gt=0)
    scale: float = Field(default=1.0, gt=0)
    max_image_size: int = Field(default=1600, ge=512)
    jpeg_quality: int = Field(default=85, ge=40, le=95)
    retry_scales: List[float] = [1.0, 0.75, 0.5]
    full_page_fallback: bool = True
    model_config = ConfigDict(extra='forbid')


class RemotePaddleOcrModel(BaseOcrModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: RemotePaddleOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: RemotePaddleOcrOptions

    def _prepare_image_bytes(self, image: Any, resize_ratio: float = 1.0) -> bytes:
        prepared = image.convert('RGB')

        if resize_ratio < 1.0:
            w, h = prepared.size
            new_w = max(1, int(w * resize_ratio))
            new_h = max(1, int(h * resize_ratio))
            prepared = prepared.resize((new_w, new_h))

        w, h = prepared.size
        max_side = self.options.max_image_size
        longest = max(w, h)
        if longest > max_side:
            shrink = max_side / float(longest)
            prepared = prepared.resize((max(1, int(w * shrink)), max(1, int(h * shrink))))

        with io.BytesIO() as buffer:
            prepared.save(buffer, format='JPEG', quality=self.options.jpeg_quality, optimize=True)
            return buffer.getvalue()

    def _request_ocr(self, image_bytes: bytes) -> Any:
        image_b64 = base64.b64encode(image_bytes).decode('ascii')
        lang_hint = f" Target languages: {', '.join(self.options.lang)}." if self.options.lang else ''
        user_prompt = f"{self.options.prompt}{lang_hint}".strip()

        headers = {'Content-Type': 'application/json'}
        if self.options.api_key and self.options.api_key.upper() != 'EMPTY':
            headers['Authorization'] = f'Bearer {self.options.api_key}'

        payload = {
            'model': self.options.model,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
                        {'type': 'text', 'text': user_prompt},
                    ],
                }
            ],
            'max_tokens': self.options.max_tokens,
        }

        response = requests.post(
            self.options.api_url,
            headers=headers,
            json=payload,
            timeout=self.options.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _extract_text_from_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, dict):
            text = content.get('text')
            if isinstance(text, str) and text.strip():
                return text.strip()

            nested_content = content.get('content')
            if isinstance(nested_content, str):
                return nested_content.strip()

            return ''

        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    part_text = part.strip()
                    if part_text:
                        text_parts.append(part_text)
                    continue

                if not isinstance(part, dict):
                    continue

                part_text = part.get('text')
                if isinstance(part_text, str) and part_text.strip():
                    text_parts.append(part_text.strip())
                    continue

                nested_content = part.get('content')
                if isinstance(nested_content, str) and nested_content.strip():
                    text_parts.append(nested_content.strip())

            return '\n'.join(text_parts).strip()

        return ''

    def _extract_openai_text(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            return ''

        if 'role' in payload or 'content' in payload:
            text = self._extract_text_from_content(payload.get('content'))
            if text:
                return text

        message = payload.get('message')
        if isinstance(message, dict):
            text = self._extract_text_from_content(message.get('content'))
            if text:
                return text

        choices = payload.get('choices')
        if not isinstance(choices, list) or not choices:
            return ''

        first_choice = choices[0] if isinstance(choices[0], dict) else None
        if not isinstance(first_choice, dict):
            return ''

        message = first_choice.get('message')
        if isinstance(message, dict):
            text = self._extract_text_from_content(message.get('content'))
            if text:
                return text

        delta = first_choice.get('delta')
        if isinstance(delta, dict):
            text = self._extract_text_from_content(delta.get('content'))
            if text:
                return text

        return ''

    def _extract_items(self, payload: Any) -> List[dict[str, Any]]:
        if isinstance(payload, dict):
            if isinstance(payload.get('data'), list):
                return payload['data']
            if isinstance(payload.get('result'), list):
                return payload['result']
            if isinstance(payload.get('results'), list):
                return payload['results']
        if isinstance(payload, list):
            return payload
        return []

    def _full_rect_cell(self, text: str, ocr_rect: BoundingBox) -> TextCell:
        return TextCell(
            index=0,
            text=text,
            orig=text,
            confidence=1.0,
            from_ocr=True,
            rect=BoundingRectangle.from_bounding_box(ocr_rect),
        )

    def _to_text_cells(self, payload: Any, ocr_rect: BoundingBox) -> List[TextCell]:
        openai_text = self._extract_openai_text(payload)
        if openai_text:
            return [self._full_rect_cell(openai_text, ocr_rect)]

        if isinstance(payload, dict) and isinstance(payload.get('text'), str):
            text = payload['text'].strip()
            return [self._full_rect_cell(text, ocr_rect)] if text else []

        cells: List[TextCell] = []
        for idx, item in enumerate(self._extract_items(payload)):
            if not isinstance(item, dict):
                continue

            text = str(item.get('text', '')).strip()
            if not text:
                continue

            score = float(item.get('score', item.get('confidence', 1.0)))
            box = item.get('box') or item.get('bbox')

            if isinstance(box, list) and len(box) >= 4 and all(isinstance(v, (int, float)) for v in box[:4]):
                x0, y0, x1, y1 = box[:4]
            elif isinstance(box, list) and len(box) >= 4 and all(isinstance(p, list) and len(p) >= 2 for p in box[:4]):
                xs = [float(p[0]) for p in box[:4]]
                ys = [float(p[1]) for p in box[:4]]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0

            bbox = BoundingBox.from_tuple(
                coord=(
                    (x0 / self.options.scale) + ocr_rect.l,
                    (y0 / self.options.scale) + ocr_rect.t,
                    (x1 / self.options.scale) + ocr_rect.l,
                    (y1 / self.options.scale) + ocr_rect.t,
                ),
                origin=CoordOrigin.TOPLEFT,
            )

            cells.append(
                TextCell(
                    index=idx,
                    text=text,
                    orig=text,
                    confidence=score,
                    from_ocr=True,
                    rect=BoundingRectangle.from_bounding_box(bbox),
                )
            )

        return cells

    def _build_full_page_rect(self, page: Page) -> BoundingBox | None:
        try:
            assert page._backend is not None
            page_image = page._backend.get_page_image(scale=self.options.scale)
            width, height = page_image.size
            return BoundingBox.from_tuple(
                coord=(0.0, 0.0, width / self.options.scale, height / self.options.scale),
                origin=CoordOrigin.TOPLEFT,
            )
        except Exception:
            return None

    def _line_set(self, cells: List[TextCell]) -> set[str]:
        lines: set[str] = set()
        for cell in cells:
            for line in cell.text.splitlines():
                cleaned = line.strip()
                if cleaned:
                    lines.add(cleaned)
        return lines

    def _collect_page_ocr_text(self, cells: List[TextCell]) -> str:
        if not cells:
            return ''

        seen: set[str] = set()
        lines: List[str] = []
        for cell in cells:
            for line in cell.text.splitlines():
                cleaned = line.strip()
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    lines.append(cleaned)

        return '\n'.join(lines).strip()

    def _merge_missing_lines(
        self,
        existing_cells: List[TextCell],
        fallback_cells: List[TextCell],
        fallback_rect: BoundingBox,
    ) -> List[TextCell]:
        if not fallback_cells:
            return []

        existing_lines = self._line_set(existing_cells)
        merged_text_parts: List[str] = []

        for cell in fallback_cells:
            missing_lines: List[str] = []
            for line in cell.text.splitlines():
                cleaned = line.strip()
                if cleaned and cleaned not in existing_lines:
                    missing_lines.append(cleaned)
            if missing_lines:
                merged_text_parts.append('\n'.join(missing_lines))

        merged_text = '\n'.join(merged_text_parts).strip()
        if not merged_text:
            return []

        # 전체 페이지 OCR은 위치 정밀도가 떨어질 수 있어 단일 셀로 추가한다.
        return [self._full_rect_cell(merged_text, fallback_rect)]

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, 'ocr'):
                ocr_rects = self.get_ocr_rects(page)
                all_ocr_cells: List[TextCell] = []

                for ocr_rect in ocr_rects:
                    if ocr_rect.area() == 0:
                        continue

                    try:
                        image = page._backend.get_page_image(scale=self.options.scale, cropbox=ocr_rect)

                        last_error: Exception | None = None
                        payload: Any = None
                        for retry_ratio in self.options.retry_scales:
                            try:
                                image_bytes = self._prepare_image_bytes(image, resize_ratio=retry_ratio)
                                payload = self._request_ocr(image_bytes)
                                last_error = None
                                break
                            except Exception as exc:  # retry smaller image on OCR/preprocess OOM
                                last_error = exc

                        if last_error is not None:
                            raise last_error

                        all_ocr_cells.extend(self._to_text_cells(payload, ocr_rect))

                    except Exception as exc:
                        page_no = getattr(page, 'page_no', '?')
                        print(f'[경고] OCR 실패 - page={page_no}, rect={ocr_rect}, error={exc}')
                        continue

                if self.options.full_page_fallback:
                    full_page_rect = self._build_full_page_rect(page)
                    if full_page_rect is not None:
                        try:
                            image = page._backend.get_page_image(scale=self.options.scale)

                            last_error: Exception | None = None
                            payload: Any = None
                            for retry_ratio in self.options.retry_scales:
                                try:
                                    image_bytes = self._prepare_image_bytes(image, resize_ratio=retry_ratio)
                                    payload = self._request_ocr(image_bytes)
                                    last_error = None
                                    break
                                except Exception as exc:
                                    last_error = exc

                            if last_error is not None:
                                raise last_error

                            full_page_cells = self._to_text_cells(payload, full_page_rect)
                            merged_cells = self._merge_missing_lines(all_ocr_cells, full_page_cells, full_page_rect)
                            all_ocr_cells.extend(merged_cells)
                            if merged_cells:
                                page_no = getattr(page, 'page_no', '?')
                                added_chars = sum(len(c.text) for c in merged_cells)
                                print(f'[정보] 전체 페이지 OCR 보강 반영 - page={page_no}, added_chars={added_chars}')
                        except Exception as exc:
                            page_no = getattr(page, 'page_no', '?')
                            print(f'[경고] 전체 페이지 OCR 보강 실패 - page={page_no}, error={exc}')

                page_ocr_text = self._collect_page_ocr_text(all_ocr_cells)
                if page_ocr_text:
                    OCR_PAGE_TEXT_BUFFER.append(page_ocr_text)

                self.post_process_cells(all_ocr_cells, page)

            yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return RemotePaddleOcrOptions


def register_remote_paddle_ocr() -> None:
    for allow_external_plugins in (False, True):
        factory = get_ocr_factory(allow_external_plugins=allow_external_plugins)
        if RemotePaddleOcrOptions not in factory.classes:
            factory.register(
                RemotePaddleOcrModel,
                plugin_name='local',
                plugin_module_name=__name__,
            )

# =======================
# 환경 설정 및 DB 연결 정보
# =======================
DB_CONFIG = {    
    'dbname': 'lhw_llm',
}

# CONNECTION_STRING = f'postgresql+psycopg2://{DB_CONFIG["DB_USER"]}:{DB_CONFIG["DB_PASSWORD"]}@{DB_CONFIG["DB_HOST"]}:{DB_CONFIG["DB_PORT"]}/{DB_CONFIG["DB_NAME"]}'

# 임베딩 모델
EMBED_MODEL_NAME = 'bge-m3'


# 파일 정보 등록
# =======================
def register_file_to_db(file_path: str) -> str:
    file_name = os.path.basename(file_path)
    
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # uuid 제외하고 파일명과 경로만 저장하고 RETURNING id 추가
            sql = 'INSERT INTO files (file_name, file_path) VALUES (%s, %s) RETURNING id'
            cur.execute(sql, (file_name, file_path))
            
            # 새로 생성된 파일 ID(UUID) 가져오기
            file_id = cur.fetchone()[0]
            conn.commit()
    
    return file_id


# =======================
# 임베딩 API 호출
# =======================
def get_embedding(text: str) -> List[float]:
    response = requests.post(
        EMBED_API_URL, json={'model': EMBED_MODEL_NAME, 'input': text}
    )
    
    return response.json()['data'][0]['embedding']


def build_converter(enable_ocr: bool = True, enable_table: bool = True, ocr_scale: float = 2.0) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = enable_ocr
    pipeline_options.do_table_structure = enable_table

    if enable_ocr:
        pipeline_options.ocr_options = RemotePaddleOcrOptions(scale=ocr_scale)

    if enable_table:
        pipeline_options.table_structure_options = TableStructureV2Options()

    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.XLSX,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )


def convert_pdf_page_by_page(path: str) -> str:
    try:
        pypdf_module = importlib.import_module('pypdf')
        PdfReader = getattr(pypdf_module, 'PdfReader')
        PdfWriter = getattr(pypdf_module, 'PdfWriter')
    except Exception:
        print('[경고] page-by-page 폴백을 위해 pypdf가 필요합니다. pip install pypdf 후 재시도하세요.')
        return ''

    markdown_parts: List[str] = []
    skipped_pages: List[int] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        reader = PdfReader(path)
        total_pages = len(reader.pages)
        print(f'[정보] PDF 안전 모드 시작: 페이지 단위 처리 ({total_pages} pages)')

        for page_idx in range(total_pages):
            page_no = page_idx + 1
            single_page_path = os.path.join(tmpdir, f'page_{page_no}.pdf')

            writer = PdfWriter()
            writer.add_page(reader.pages[page_idx])
            with open(single_page_path, 'wb') as f:
                writer.write(f)

            try:
                # 페이지마다 새 컨버터를 생성해 메모리 누적을 줄인다.
                page_converter = build_converter(enable_ocr=False, enable_table=False, ocr_scale=2.0)

                buffer_start_idx = len(OCR_PAGE_TEXT_BUFFER)
                page_doc = page_converter.convert(single_page_path)
                page_md = page_doc.document.export_to_markdown().strip()

                # 사용자 정의 OCR 모델이 수집한 페이지 텍스트를 우선 병합한다.
                ocr_text = '\n'.join(OCR_PAGE_TEXT_BUFFER[buffer_start_idx:]).strip()
                del OCR_PAGE_TEXT_BUFFER[buffer_start_idx:]
                if ocr_text and ocr_text not in page_md:
                    page_md = f'{page_md}\n\n{ocr_text}'.strip() if page_md else ocr_text

                if page_md:
                    markdown_parts.append(page_md)
            except Exception as exc:
                try:
                    # OCR을 끄고 한 번 더 시도해 텍스트 레이어라도 확보한다.
                    text_converter = build_converter(enable_ocr=False, enable_table=False, ocr_scale=1.0)
                    page_doc = text_converter.convert(single_page_path)
                    page_md = page_doc.document.export_to_markdown().strip()
                    if page_md:
                        markdown_parts.append(page_md)
                    else:
                        skipped_pages.append(page_no)
                        print(f'[경고] 페이지 단위 변환 실패 - page={page_no}, error={exc}')
                except Exception as retry_exc:
                    skipped_pages.append(page_no)
                    print(f'[경고] 페이지 단위 변환 실패 - page={page_no}, error={retry_exc}')
            finally:
                gc.collect()

    if skipped_pages:
        print(f'[경고] 변환에서 제외된 페이지: {skipped_pages}')

    return '\n\n'.join(markdown_parts)


def convert_with_fallback(path: str) -> str:
    if path.lower().endswith('.pdf'):
        markdown_text = convert_pdf_page_by_page(path)
        if markdown_text.strip():
            return markdown_text

    attempts = [
        ('full', True, True, 2.0),
        ('no_table', True, False, 1.5),
        ('low_ocr', True, False, 1.0),
        ('text_only', False, False, 1.0),
    ]

    last_error: Exception | None = None
    for name, enable_ocr, enable_table, ocr_scale in attempts:
        try:
            print(f'[정보] 변환 시도: {name} (ocr={enable_ocr}, table={enable_table}, scale={ocr_scale})')
            converter = build_converter(enable_ocr=enable_ocr, enable_table=enable_table, ocr_scale=ocr_scale)
            documents = converter.convert(path)
            markdown_text = documents.document.export_to_markdown()
            if markdown_text.strip():
                return markdown_text
        except Exception as exc:
            last_error = exc
            print(f'[경고] 변환 시도 실패: {name} - {exc}')

    if last_error is not None:
        raise last_error

    raise RuntimeError(f'문서 변환 실패: {path}')


def sanitize_markdown_for_embedding(markdown_text: str) -> str:
    cleaned = markdown_text

    # Docling 이미지 플레이스홀더 제거
    cleaned = re.sub(r'<!--\s*image\s*-->', ' ', cleaned, flags=re.IGNORECASE)
    # 일반 HTML 주석 제거
    cleaned = re.sub(r'<!--.*?-->', ' ', cleaned, flags=re.DOTALL)
    # 마크다운 이미지 문법 제거
    cleaned = re.sub(r'!\[[^\]]*\]\([^\)]*\)', ' ', cleaned)

    # 줄 단위 공백 정리 후 빈 줄 압축
    lines = [line.strip() for line in cleaned.splitlines()]
    cleaned = '\n'.join(lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)

    return cleaned.strip()


# =======================
# 메인 파이프라인(Docling + 내부 API 임베딩)
# =======================
def process_file(file_path: List[str]):
    register_remote_paddle_ocr()
    
    
    # 텍스트 분할(Chunking) 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
            
    for path in file_path:
        try:
            print(f'\n[시작] 파일 처리 중: {path}')
            
            # # DB에 파일 정보 등록
            # file_id = register_file_to_db(path)
            
            # Docling을 사용하여 문서 변환(메모리 부족 시 단계별 폴백)
            markdown_text = convert_with_fallback(path)

            # 임베딩에는 이미지 플레이스홀더를 제거한 텍스트를 사용한다.
            embedding_text = sanitize_markdown_for_embedding(markdown_text)
            if not embedding_text:
                print(f'[경고] 정제 후 텍스트가 비어 원본 마크다운으로 대체: {path}')
                embedding_text = markdown_text
            
            # Chunk 분할
            chunks = text_splitter.split_text(embedding_text)
            print(f'[정보] 총 {len(chunks)}개의 청크로 분할됨. 임베딩 및 저장 시작')
            
            # 모든 청크의 임베딩 미리 생성 (API 호출)
            prepared_chunks = []
            for idx, chunk in enumerate(chunks):
                vector = get_embedding(chunk)
                metadata = {'chunk_index': idx, 'file_path': path}
                prepared_chunks.append((chunk, vector, json.dumps(metadata)))
            
            # DB 저장
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    # files 테이블 저장
                    file_name = os.path.basename(path)
                    file_type = path.split('.')[-1]
                    
                    sql_file = 'INSERT INTO files (file_name, file_type, file_path) VALUES (%s, %s, %s) RETURNING id'
                    cur.execute(sql_file, (file_name, file_type, path))
                    file_id = cur.fetchone()[0]
                        
                    # document_chunks 테이블 저장
                    sql_chunk = 'INSERT INTO document_chunks (file_id, content, embedding, metadata) VALUES (%s, %s, %s, %s)'
                    chunk_data_for_db = [(file_id, c, v, m) for c, v, m in prepared_chunks]
                    cur.executemany(sql_chunk, chunk_data_for_db)
                        
                    conn.commit()
                    
            print(f'[완료] 파일 처리 완료: {path} (ID: {file_id})')
            
        except Exception as e:
            print(f'[오류] 파일 처리 중 오류 발생: {path} - {str(e)}')
            
            
            
# 실행
files = ['C:\\Users\\lhw12\\OneDrive - dlit.co.kr\\2026\\LLM 문서 검색\\[결과보고서] 충남 친환경 모빌리티 AI 융합 지원-(재)충남연구원.pdf']
# files = ['/mnt/e/OneDrive - dlit.co.kr/2026/LLM 문서 검색/[결과보고서] 충남 친환경 모빌리티 AI 융합 지원-(재)충남연구원.pdf']
# files = ['/mnt/e/OneDrive - dlit.co.kr/2026/LLM 문서 검색/ICHI 프로그램 수정 내역.pdf']
process_file(files)
