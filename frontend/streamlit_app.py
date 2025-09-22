import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import json
import re
from collections import defaultdict
from typing import List, Dict, Any
import tempfile
import os

# 설정
st.set_page_config(
    page_title="Industrial RAG Assistant",
    page_icon="🏭",
    layout="wide"
)


@st.cache_resource
def load_llm_model():
    """경량 LLM 모델 로드 (캐시됨)"""
    try:
        # 매우 가벼운 모델 (Streamlit Cloud 호환)
        model_name = "distilgpt2"  # 약 300MB만 사용

        with st.spinner("Loading AI model... (first time only)"):
            # 모델과 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            # 패딩 토큰 설정
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 텍스트 생성 파이프라인
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,  # GPU 사용 가능하면 사용
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            return generator
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


@st.cache_resource
def load_simple_model():
    """더 간단한 모델 (fallback)"""
    try:
        # 매우 가벼운 모델
        return pipeline("text-generation", model="distilgpt2", device=-1)
    except:
        return None


class SimpleDocumentStore:
    """간단한 문서 저장 및 검색"""

    def __init__(self, category: str):
        self.category = category
        self.documents = []
        self.keyword_index = defaultdict(set)

        # 세션 상태에서 로드
        if f"docs_{category}" in st.session_state:
            self.documents = st.session_state[f"docs_{category}"]
            self._rebuild_index()

    def _rebuild_index(self):
        """키워드 인덱스 재구성"""
        self.keyword_index = defaultdict(set)
        for i, doc in enumerate(self.documents):
            self._index_document(doc['text'], i)

    def _index_document(self, text: str, doc_id: int):
        """문서를 키워드로 인덱싱"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        for word in set(words):
            self.keyword_index[word].add(doc_id)

    def add_document(self, text: str, metadata: dict):
        """문서 추가"""
        doc_id = len(self.documents)
        self.documents.append({
            'text': text,
            'metadata': metadata
        })

        # 키워드 인덱싱
        self._index_document(text, doc_id)

        # 세션 상태에 저장
        st.session_state[f"docs_{self.category}"] = self.documents

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """키워드 기반 검색"""
        if not self.documents:
            return []

        query_words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        if not query_words:
            return []

        # 문서별 점수 계산
        doc_scores = defaultdict(int)
        for word in query_words:
            for doc_id in self.keyword_index.get(word, set()):
                doc_scores[doc_id] += 1

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
                    'score': score / len(query_words)
                })

        return results

    def get_document_count(self) -> int:
        return len(self.documents)


def extract_text_from_file(uploaded_file) -> str:
    """업로드된 파일에서 텍스트 추출"""
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
            except:
                return "PDF processing not available. Please upload as text file."
        else:
            # 기본적으로 텍스트로 읽기 시도
            return str(uploaded_file.read(), "utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


def generate_answer_with_local_llm(question: str, context: str, category: str, generator) -> str:
    """로컬 LLM으로 답변 생성"""
    if not generator:
        return "AI model not available. Please check the setup."

    # 프롬프트 구성 (짧게 유지)
    prompt = f"""Context: {context[:500]}...

Question: {question}

Answer based on the context above:"""

    try:
        # 텍스트 생성
        result = generator(
            prompt,
            max_length=len(prompt.split()) + 100,  # 입력 + 100 토큰
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )

        # 생성된 텍스트에서 답변 부분만 추출
        generated_text = result[0]['generated_text']
        answer = generated_text[len(prompt):].strip()

        if not answer:
            return "Based on the documents, I found relevant information but couldn't generate a complete answer. Please try rephrasing your question."

        return answer

    except Exception as e:
        return f"Error generating answer: {e}"


def generate_simple_answer(question: str, context: str) -> str:
    """간단한 규칙 기반 답변 (fallback)"""
    # 키워드 매칭으로 간단한 답변 생성
    sentences = context.split('.')[:3]  # 처음 3문장
    answer = ". ".join(sentences) + "."
    return f"Based on the documents: {answer}"


def main():
    st.title("🏭 Industrial Document Assistant")
    st.markdown("**Free AI-powered document search** (no API costs!)")

    # AI 모델 로드
    with st.sidebar:
        st.header("🤖 AI Model Status")

        # 모델 선택
        model_option = st.selectbox(
            "AI Model:",
            ["Local AI (Free)", "Simple Search Only"]
        )

        if model_option == "Local AI (Free)":
            generator = load_llm_model()
            if not generator:
                generator = load_simple_model()
                if generator:
                    st.warning("Using basic model")
                else:
                    st.error("AI model failed to load")
        else:
            generator = None
            st.info("Using keyword search only")

        st.markdown("---")

        # 카테고리 선택
        categories = {
            "procurement": "Procurement/Contract",
            "piping": "Piping",
            "process": "Process",
            "mechanical": "Mechanical"
        }

        selected_category = st.selectbox(
            "Document Category:",
            options=list(categories.keys()),
            format_func=lambda x: categories[x]
        )

        st.markdown("---")

        # 문서 업로드
        st.header("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            type=['txt', 'md'],
            help="Supported: TXT, MD files"
        )

        if uploaded_files and st.button("Process Documents"):
            store = SimpleDocumentStore(selected_category)

            progress_bar = st.progress(0)
            processed = 0

            for i, uploaded_file in enumerate(uploaded_files):
                text = extract_text_from_file(uploaded_file)
                if text.strip() and len(text) > 10:
                    metadata = {
                        "category": selected_category,
                        "file_name": uploaded_file.name,
                        "file_type": uploaded_file.type
                    }
                    store.add_document(text, metadata)
                    processed += 1

                progress_bar.progress((i + 1) / len(uploaded_files))

            st.success(f"✅ Processed {processed} documents!")
            st.rerun()

        # 통계 표시
        st.header("📊 Document Stats")
        for category, name in categories.items():
            if f"docs_{category}" in st.session_state:
                count = len(st.session_state[f"docs_{category}"])
                if count > 0:
                    st.metric(name, count)

    # 메인 채팅 인터페이스
    st.header("💬 Chat with Documents")

    # 도움말
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
        **This app runs completely free!**

        - **No API costs**: Uses local AI models
        - **Private**: Your documents stay on your device
        - **Simple**: Upload documents and ask questions

        **Tips:**
        - Upload text files for best results
        - Ask specific questions about your documents
        - The AI model loads once and stays cached
        """)

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📄 Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**{i}. {source['file_name']}** (Score: {source['score']:.3f})")
                            st.markdown(f"*{source['preview']}*")

    # 사용자 입력
    if prompt := st.chat_input("Ask about your documents..."):
        # 문서 확인
        store = SimpleDocumentStore(selected_category)
        if store.get_document_count() == 0:
            st.error(f"No documents in {categories[selected_category]} category. Please upload documents first.")
            return

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 어시스턴트 응답
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                # 문서 검색
                search_results = store.search(prompt, top_k=3)

                if not search_results:
                    response = f"No relevant documents found in {categories[selected_category]} category."
                    sources = []
                else:
                    # 컨텍스트 구성
                    context_texts = []
                    sources = []

                    for result in search_results:
                        text = result['text'][:600]  # 토큰 제한으로 짧게
                        context_texts.append(text)
                        sources.append({
                            'file_name': result['metadata']['file_name'],
                            'score': result['score'],
                            'preview': text[:200] + "..." if len(text) > 200 else text
                        })

                    context = "\n\n---\n\n".join(context_texts)

                    # AI 모델로 답변 생성
                    if generator and model_option == "Local AI (Free)":
                        response = generate_answer_with_local_llm(prompt, context, categories[selected_category],
                                                                  generator)
                    else:
                        response = generate_simple_answer(prompt, context)

                st.markdown(response)

                # 소스 표시
                if sources:
                    with st.expander("📄 Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**{i}. {source['file_name']}** (Score: {source['score']:.3f})")
                            st.markdown(f"*{source['preview']}*")

                # 메시지 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })

    # 채팅 초기화
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()