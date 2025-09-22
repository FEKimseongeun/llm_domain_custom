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
def load_llm_model(model_choice: str):
    """개선된 LLM 모델 로드 (CPU 최적화)"""

    # CPU 친화적인 모델들
    model_configs = {
        "deepseek-r1-distill": {
            "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "size": "~3GB",
            "description": "DeepSeek-R1 Distilled - 최고 성능",
            "type": "chat"
        },
        "flan-t5-small": {
            "name": "google/flan-t5-small",
            "size": "~80MB",
            "description": "Google T5 - 빠르고 효율적",
            "type": "text2text"
        },
        "flan-t5-base": {
            "name": "google/flan-t5-base",
            "size": "~250MB",
            "description": "Google T5 - 균형잡힌 성능",
            "type": "text2text"
        },
        "distilgpt2": {
            "name": "distilgpt2",
            "size": "~82MB",
            "description": "DistilGPT2 - 초경량",
            "type": "causal"
        },
        "gpt2": {
            "name": "gpt2",
            "size": "~124MB",
            "description": "GPT2 - 표준 성능",
            "type": "causal"
        },
        "microsoft/DialoGPT-small": {
            "name": "microsoft/DialoGPT-small",
            "size": "~117MB",
            "description": "대화 특화 모델",
            "type": "causal"
        }
    }

    if model_choice not in model_configs:
        model_choice = "deepseek-r1-distill"

    config = model_configs[model_choice]

    try:
        with st.spinner(f"Loading {config['description']} ({config['size']})..."):
            model_name = config["name"]
            model_type = config["type"]

            # DeepSeek-R1-Distill 모델 (특별 처리)
            if model_choice == "deepseek-r1-distill":
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )

                # 패딩 토큰 설정
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # 커스텀 제너레이터 객체 생성
                class DeepSeekGenerator:
                    def __init__(self, model, tokenizer):
                        self.model = model
                        self.tokenizer = tokenizer

                    def generate_answer(self, question, context):
                        messages = [
                            {"role": "user",
                             "content": f"Based on the following context, please answer the question:\n\nContext: {context}\n\nQuestion: {question}"}
                        ]

                        inputs = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt"
                        ).to(self.model.device)

                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=200,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )

                        # 답변 부분만 추출
                        response = self.tokenizer.decode(
                            outputs[0][inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True
                        )
                        return response.strip()

                generator = DeepSeekGenerator(model, tokenizer)

            # T5 모델들은 text2text-generation 사용
            elif model_type == "text2text":
                generator = pipeline(
                    "text2text-generation",
                    model=model_name,
                    device=-1,  # CPU 사용
                    torch_dtype=torch.float32,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
            else:
                # 기타 GPT 계열 모델들
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )

                # 패딩 토큰 설정
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU 사용
                    torch_dtype=torch.float32
                )

            st.success(f"✅ Loaded: {config['description']}")
            return generator, config

    except Exception as e:
        st.error(f"Failed to load {model_choice}: {e}")
        # 폴백 - 가장 가벼운 모델
        try:
            st.info("Trying fallback model...")
            generator = pipeline("text-generation", model="distilgpt2", device=-1)
            return generator, model_configs["distilgpt2"]
        except:
            return None, None


class SimpleDocumentStore:
    """간단한 문서 저장 및 검색 (향상된 검색)"""

    def __init__(self, category: str):
        self.category = category
        self.documents = []
        self.keyword_index = defaultdict(set)
        self.phrase_index = defaultdict(set)

        # 세션 상태에서 로드
        if f"docs_{category}" in st.session_state:
            self.documents = st.session_state[f"docs_{category}"]
            self._rebuild_index()

    def _rebuild_index(self):
        """향상된 인덱싱"""
        self.keyword_index = defaultdict(set)
        self.phrase_index = defaultdict(set)

        for i, doc in enumerate(self.documents):
            self._index_document(doc['text'], i)

    def _index_document(self, text: str, doc_id: int):
        """개선된 문서 인덱싱"""
        text_lower = text.lower()

        # 단어 인덱싱
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text_lower)
        for word in set(words):
            self.keyword_index[word].add(doc_id)

        # 구문 인덱싱 (2-3 단어 조합)
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i + 1]}"
            self.phrase_index[phrase].add(doc_id)

    def add_document(self, text: str, metadata: dict):
        """문서 추가"""
        doc_id = len(self.documents)
        self.documents.append({
            'text': text,
            'metadata': metadata
        })

        # 인덱싱
        self._index_document(text, doc_id)

        # 세션 상태에 저장
        st.session_state[f"docs_{self.category}"] = self.documents

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """향상된 검색"""
        if not self.documents:
            return []

        query_lower = query.lower()
        query_words = re.findall(r'\b[a-zA-Z]{2,}\b', query_lower)

        if not query_words:
            return []

        # 문서별 점수 계산
        doc_scores = defaultdict(float)

        # 단어 매칭
        for word in query_words:
            for doc_id in self.keyword_index.get(word, set()):
                doc_scores[doc_id] += 1.0

        # 구문 매칭 (더 높은 점수)
        for i in range(len(query_words) - 1):
            phrase = f"{query_words[i]} {query_words[i + 1]}"
            for doc_id in self.phrase_index.get(phrase, set()):
                doc_scores[doc_id] += 2.0

        # 정확한 구문 매칭 (최고 점수)
        for doc_id, doc in enumerate(self.documents):
            if query_lower in doc['text'].lower():
                doc_scores[doc_id] += 5.0

        # 점수 정규화
        max_score = len(query_words) + 2.0 * (len(query_words) - 1) + 5.0
        for doc_id in doc_scores:
            doc_scores[doc_id] = min(doc_scores[doc_id] / max_score, 1.0)

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
                    'score': score
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


def generate_answer_with_local_llm(question: str, context: str, category: str, generator, model_config) -> str:
    """개선된 로컬 LLM 답변 생성"""
    if not generator:
        return "AI model not available. Please check the setup."

    # 컨텍스트 길이 제한 (모델에 따라 조정)
    if "deepseek" in model_config["name"].lower():
        max_context = 2000  # DeepSeek는 더 긴 컨텍스트 지원
    elif "small" in model_config["name"]:
        max_context = 800
    else:
        max_context = 1200

    context = context[:max_context]

    try:
        # DeepSeek-R1-Distill 모델
        if hasattr(generator, 'generate_answer'):
            answer = generator.generate_answer(question, context)

        # T5 모델용 프롬프트 (text2text)
        elif model_config["type"] == "text2text":
            prompt = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"

            result = generator(
                prompt,
                max_length=150,
                temperature=0.7,
                do_sample=True,
                early_stopping=True
            )

            answer = result[0]['generated_text'].strip()

        else:
            # GPT 계열 모델용 프롬프트
            prompt = f"""Based on the following industrial documents, please answer the question:

Context: {context}

Question: {question}

Answer:"""

            result = generator(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=generator.tokenizer.eos_token_id,
                early_stopping=True
            )

            # 생성된 텍스트에서 답변 부분만 추출
            generated_text = result[0]['generated_text']
            answer = generated_text[len(prompt):].strip()

        # 답변 정제
        if not answer or len(answer) < 10:
            return generate_fallback_answer(question, context)

        # 답변 길이 제한 (DeepSeek는 예외)
        if not hasattr(generator, 'generate_answer'):
            sentences = answer.split('.')
            if len(sentences) > 3:
                answer = '. '.join(sentences[:3]) + '.'

        return answer

    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return generate_fallback_answer(question, context)


def generate_fallback_answer(question: str, context: str) -> str:
    """폴백 답변 생성 (규칙 기반)"""
    # 컨텍스트에서 관련 문장 추출
    sentences = context.split('.')
    relevant_sentences = []

    question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))

    for sentence in sentences[:5]:  # 처음 5문장만 확인
        sentence_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower()))
        if question_words.intersection(sentence_words):
            relevant_sentences.append(sentence.strip())

    if relevant_sentences:
        answer = '. '.join(relevant_sentences[:2]) + '.'
        return f"Based on the documents: {answer}"
    else:
        return "I found some relevant documents, but couldn't extract a specific answer. Please try rephrasing your question or check the source documents."


def main():
    st.title("🏭 Industrial Document Assistant")
    st.markdown("**Latest DeepSeek AI + Advanced document search** (CPU optimized!)")

    # 사이드바 - AI 모델 설정
    with st.sidebar:
        st.header("🤖 AI Model Settings")

        # 모델 선택
        model_options = {
            "deepseek-r1-distill": "🚀 DeepSeek-R1 Distilled (최고성능)",
            "flan-t5-small": "⚡ Google T5 Small (빠름)",
            "flan-t5-base": "🎯 Google T5 Base (고성능)",
            "distilgpt2": "💨 DistilGPT2 (초경량)",
            "gpt2": "📝 GPT2 (표준)",
            "microsoft/DialoGPT-small": "💬 DialoGPT (대화특화)"
        }

        selected_model = st.selectbox(
            "Choose AI Model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0  # DeepSeek가 기본값
        )

        # 모델 로드
        if st.button("Load Model") or "current_model" not in st.session_state:
            generator, model_config = load_llm_model(selected_model)
            st.session_state.current_model = selected_model
            st.session_state.generator = generator
            st.session_state.model_config = model_config

        # 현재 모델 상태 표시
        if "generator" in st.session_state and st.session_state.generator:
            st.success(f"✅ {st.session_state.model_config['description']}")
            st.info(f"Size: {st.session_state.model_config['size']}")
        else:
            st.warning("❌ No model loaded")

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
            type=['txt', 'md', 'pdf'],
            help="Supported: TXT, MD, PDF files"
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
        **Enhanced Features:**

        - **Multiple AI Models**: Choose from various CPU-optimized models
        - **Improved Search**: Better keyword and phrase matching
        - **Smart Fallbacks**: Always provides useful responses
        - **No API Costs**: Runs completely free on your device

        **Model Recommendations:**
        - **🚀 DeepSeek-R1 Distilled**: Latest AI, best quality (3GB)
        - **⚡ T5-Small**: Best balance of speed and quality (80MB)
        - **🎯 T5-Base**: Higher quality but slower (250MB)
        - **💨 DistilGPT2**: Fastest, lightweight option (82MB)

        **Tips:**
        - Upload documents in your preferred category first
        - Ask specific questions about technical details
        - The AI provides answers based on your documents
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

        # 모델 확인
        if "generator" not in st.session_state or not st.session_state.generator:
            st.error("No AI model loaded. Please load a model first.")
            return

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 어시스턴트 응답
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating answer..."):
                # 문서 검색
                search_results = store.search(prompt, top_k=3)

                if not search_results:
                    response = f"No relevant documents found in {categories[selected_category]} category. Please try different keywords or upload more documents."
                    sources = []
                else:
                    # 컨텍스트 구성
                    context_texts = []
                    sources = []

                    for result in search_results:
                        text = result['text'][:800]  # 적절한 길이로 제한
                        context_texts.append(text)
                        sources.append({
                            'file_name': result['metadata']['file_name'],
                            'score': result['score'],
                            'preview': text[:200] + "..." if len(text) > 200 else text
                        })

                    context = "\n\n---\n\n".join(context_texts)

                    # AI로 답변 생성
                    response = generate_answer_with_local_llm(
                        prompt,
                        context,
                        categories[selected_category],
                        st.session_state.generator,
                        st.session_state.model_config
                    )

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
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Reset All"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()


if __name__ == "__main__":
    main()