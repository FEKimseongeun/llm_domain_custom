from llama_index.core import SimpleDirectoryReader, Settings as LlamaSettings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import os
import re
import json
from collections import defaultdict

# 프롬프트 임포트
from .prompts import CATEGORY_PROMPTS, NO_CONTEXT_RESPONSES
from ..config import TECHNICAL_GLOSSARY

logger = logging.getLogger(__name__)


class SimpleDocumentStore:
    """간단한 문서 저장 및 키워드 검색"""

    def __init__(self, category: str, storage_path: str):
        self.category = category
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.documents = []
        self.keyword_index = defaultdict(set)
        self.technical_term_index = defaultdict(set)  # 기술 용어 전용 인덱스

        # 기존 데이터 로드
        self._load_data()

    def _get_storage_file(self):
        return self.storage_path / f"{self.category}_documents.json"

    def _load_data(self):
        """저장된 문서 데이터 로드"""
        storage_file = self._get_storage_file()
        if storage_file.exists():
            try:
                with open(storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])

                    # 키워드 인덱스 재구성
                    for i, doc in enumerate(self.documents):
                        self._index_document(doc['text'], i)

                logger.info(f"Loaded {len(self.documents)} documents for {self.category}")
            except Exception as e:
                logger.error(f"Error loading data for {self.category}: {e}")

    def _save_data(self):
        """문서 데이터 저장"""
        storage_file = self._get_storage_file()
        try:
            with open(storage_file, 'w', encoding='utf-8') as f:
                json.dump({'documents': self.documents}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving data for {self.category}: {e}")

    def _extract_technical_terms(self, text: str) -> List[str]:
        """카테고리별 기술 용어 추출"""
        terms = []
        text_lower = text.lower()

        # 카테고리별 기술 용어 검색
        if self.category in TECHNICAL_GLOSSARY:
            for term in TECHNICAL_GLOSSARY[self.category]:
                if term.lower() in text_lower:
                    terms.append(term.lower())

        return terms

    def _index_document(self, text: str, doc_id: int):
        """문서를 키워드로 인덱싱 (개선된 버전)"""
        # 일반 영어 단어 추출 (3글자 이상)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        for word in set(words):
            self.keyword_index[word].add(doc_id)

        # 기술 용어 인덱싱
        technical_terms = self._extract_technical_terms(text)
        for term in technical_terms:
            self.technical_term_index[term].add(doc_id)

        # 숫자가 포함된 용어도 인덱싱 (예: B31.3, API650)
        alphanumeric = re.findall(r'\b[A-Z]+[\d]+[\.\d]*\b', text.upper())
        for term in set(alphanumeric):
            self.keyword_index[term.lower()].add(doc_id)

    def add_document(self, text: str, metadata: dict):
        """문서 추가"""
        doc_id = len(self.documents)
        self.documents.append({
            'text': text,
            'metadata': metadata
        })

        # 키워드 인덱싱
        self._index_document(text, doc_id)

        # 저장
        self._save_data()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """개선된 키워드 기반 검색"""
        if not self.documents:
            return []

        query_lower = query.lower()

        # 쿼리에서 일반 단어와 기술 용어 추출
        query_words = re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)
        technical_terms = self._extract_technical_terms(query)

        if not query_words and not technical_terms:
            return []

        # 문서별 점수 계산 (기술 용어에 더 높은 가중치)
        doc_scores = defaultdict(float)

        # 일반 단어 매칭 (가중치 1.0)
        for word in query_words:
            for doc_id in self.keyword_index.get(word, set()):
                doc_scores[doc_id] += 1.0

        # 기술 용어 매칭 (가중치 2.0)
        for term in technical_terms:
            for doc_id in self.technical_term_index.get(term, set()):
                doc_scores[doc_id] += 2.0

        # 점수 순 정렬
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 상위 결과 반환
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if score > 0:
                doc = self.documents[doc_id]
                results.append({
                    'text': doc['text'],
                    'metadata': doc['metadata'],
                    'score': score / (
                                len(query_words) + len(technical_terms) * 2) if query_words or technical_terms else 0
                })

        return results

    def get_document_count(self) -> int:
        return len(self.documents)


class IndustrialRAGEngine:
    """산업용 문서 RAG 엔진 - 개선된 버전"""

    def __init__(self, config):
        self.config = config
        # 5개 카테고리로 확장
        self.categories = ["bim", "process", "piping", "procurement", "mechanical"]
        self.document_stores: Dict[str, SimpleDocumentStore] = {}

        # 필요한 디렉토리 생성
        self._create_directories()

        # 모델 설정
        self._setup_models()

        # 문서 저장소 초기화
        self._initialize_stores()

    def _create_directories(self):
        """필요한 디렉토리 생성"""
        for category in self.categories:
            category_path = Path(self.config.documents_path) / category
            category_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {category_path}")

    def _setup_models(self):
        """Llama LLM 설정"""
        logger.info("Setting up Llama LLM...")

        try:
            llm = Ollama(
                model=self.config.ollama_model,
                base_url=self.config.ollama_host,
                temperature=0.1,  # 더 일관된 답변을 위해 낮춤
                request_timeout=300.0,
                context_window=4096,  # 컨텍스트 윈도우 명시
                num_predict=1024  # 답변 길이 제한
            )

            # LLM 연결 테스트
            test_response = llm.complete("Hello")
            logger.info("Llama LLM connection successful")

        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.config.ollama_host}. "
                f"Make sure 'ollama serve' is running and model '{self.config.ollama_model}' is downloaded."
            )

        # 글로벌 설정
        LlamaSettings.llm = llm
        LlamaSettings.node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        logger.info("LLM setup completed")

    def _initialize_stores(self):
        """카테고리별 문서 저장소 초기화"""
        logger.info("Initializing document stores...")

        storage_path = Path("./document_storage")
        for category in self.categories:
            self.document_stores[category] = SimpleDocumentStore(category, storage_path)

        logger.info("Document stores initialization completed")

    async def add_documents(self, category: str, file_paths: List[str]) -> Dict[str, Any]:
        """카테고리에 문서 추가"""
        if category not in self.categories:
            raise ValueError(f"Invalid category. Must be one of: {self.categories}")

        try:
            logger.info(f"Adding {len(file_paths)} documents to {category}")

            # 문서 로드
            documents = SimpleDirectoryReader(
                input_files=file_paths,
                recursive=True
            ).load_data()

            # 문서를 청크로 분할
            node_parser = LlamaSettings.node_parser

            # 문서 저장소에 추가
            store = self.document_stores[category]
            for doc in documents:
                # 문서를 청크로 분할
                chunks = node_parser.split_text(doc.text)

                for i, chunk in enumerate(chunks):
                    metadata = {
                        "category": category,
                        "source": str(doc.metadata.get("file_path", "unknown")),
                        "file_name": os.path.basename(str(doc.metadata.get("file_path", "unknown"))),
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                    store.add_document(chunk, metadata)

            logger.info(f"Successfully added {len(documents)} documents to {category}")
            return {
                "status": "success",
                "documents_added": len(documents),
                "category": category
            }

        except Exception as e:
            logger.error(f"Error adding documents to {category}: {e}")
            raise

    def _prepare_context(self, search_results: List[Dict], max_context_length: int = 3000) -> str:
        """컨텍스트 준비 및 최적화"""
        if not search_results:
            return ""

        context_parts = []
        current_length = 0

        for i, result in enumerate(search_results):
            text = result['text']
            metadata = result['metadata']

            # 컨텍스트 길이 제한
            if current_length + len(text) > max_context_length:
                remaining = max_context_length - current_length
                if remaining > 100:  # 최소 100자는 포함
                    text = text[:remaining] + "..."
                else:
                    break

            # 소스 정보와 함께 컨텍스트 구성
            context_part = f"[Source: {metadata.get('file_name', 'unknown')}]\n{text}"
            context_parts.append(context_part)
            current_length += len(text)

        return "\n\n---\n\n".join(context_parts)

    async def query(self, category: str, question: str, top_k: int = 5) -> Dict[str, Any]:
        """개선된 카테고리별 질의응답"""
        if category not in self.categories:
            raise ValueError(f"Invalid category. Must be one of: {self.categories}")

        try:
            logger.info(f"Querying {category} with question: {question[:50]}...")

            # 키워드 검색으로 관련 문서 찾기
            store = self.document_stores[category]
            search_results = store.search(question, top_k)

            if not search_results:
                # 카테고리별 맞춤 응답
                no_context_response = NO_CONTEXT_RESPONSES.get(
                    category,
                    f"Sorry, I couldn't find relevant documents in the {category} category."
                )

                return {
                    "answer": no_context_response,
                    "sources": [],
                    "category": category,
                    "question": question
                }

            # 컨텍스트 준비
            context = self._prepare_context(search_results)

            # 카테고리별 특화 프롬프트 선택
            prompt_template = CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["process"])

            # 프롬프트 생성
            prompt = prompt_template.format(
                context=context,
                question=question
            )

            # LLM으로 답변 생성
            response = LlamaSettings.llm.complete(prompt)

            # 소스 정보 구성
            sources = []
            for result in search_results:
                sources.append({
                    "source": result['metadata'].get('file_name', 'unknown'),
                    "score": float(result['score']),
                    "content_preview": result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                })

            result = {
                "answer": str(response),
                "sources": sources,
                "category": category,
                "question": question
            }

            logger.info(f"Successfully queried {category}")
            return result

        except Exception as e:
            logger.error(f"Error querying {category}: {e}")
            raise

    def get_category_stats(self) -> Dict[str, Any]:
        """각 카테고리별 문서 통계"""
        stats = {}
        for category in self.categories:
            try:
                store = self.document_stores[category]
                doc_count = store.get_document_count()

                stats[category] = {
                    "document_count": doc_count,
                    "status": "active"
                }
            except Exception as e:
                logger.error(f"Error getting stats for {category}: {e}")
                stats[category] = {
                    "document_count": 0,
                    "status": "error"
                }

        return stats

    def check_health(self) -> Dict[str, Any]:
        """시스템 헬스 체크"""
        try:
            # LLM 연결 테스트
            llm_status = "healthy"
            try:
                test_response = LlamaSettings.llm.complete("Test")
                if not str(test_response).strip():
                    llm_status = "warning"
            except Exception as e:
                logger.error(f"LLM health check failed: {e}")
                llm_status = "unhealthy"

            return {
                "status": llm_status,
                "llm_status": llm_status,
                "search_type": "keyword_based_enhanced",  # 개선된 키워드 검색
                "categories": list(self.categories),
                "stores_loaded": len(self.document_stores),
                "ollama_host": self.config.ollama_host,
                "ollama_model": self.config.ollama_model
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }