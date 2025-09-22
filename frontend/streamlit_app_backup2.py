import streamlit as st
import httpx
import os
from typing import Dict, Any, List

# 설정
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Industrial RAG Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_connection():
    """백엔드 연결 확인"""
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
        return response.status_code == 200
    except:
        return False


def load_categories():
    """카테고리 정보 로드"""
    try:
        response = httpx.get(f"{BACKEND_URL}/api/v1/chat/categories", timeout=10.0)
        if response.status_code == 200:
            return response.json()["categories"]
        else:
            st.error(f"Failed to load categories: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Cannot connect to backend: {e}")
        return []


def upload_documents(category: str, files: List) -> Dict[str, Any]:
    """문서 업로드"""
    try:
        files_data = []
        for uploaded_file in files:
            files_data.append(("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))

        response = httpx.post(
            f"{BACKEND_URL}/api/v1/documents/upload",
            data={"category": category},
            files=files_data,
            timeout=60.0
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload failed: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def query_documents(category: str, question: str, top_k: int = 5) -> Dict[str, Any]:
    """문서 질의"""
    try:
        response = httpx.post(
            f"{BACKEND_URL}/api/v1/chat",
            json={
                "category": category,
                "question": question,
                "top_k": top_k
            },
            timeout=120.0
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Query failed: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def main():
    """메인 애플리케이션"""

    st.title("🏭 EPC 전용 Document Assistant")
    st.markdown("Ask questions about your technical documents")

    # 백엔드 연결 확인
    if not check_backend_connection():
        st.error("⚠️ Cannot connect to backend server. Please make sure the server is running.")
        st.info("Run the backend with: `python -m app.main` in the backend directory")
        return

    # 사이드바 설정
    with st.sidebar:
        st.header("📋 Document Categories")

        # 카테고리 로드
        categories = load_categories()
        if not categories:
            st.error("No categories available")
            return

        # 카테고리 선택
        category_options = {}
        for cat in categories:
            status_emoji = "✅" if cat["is_available"] else "❌"
            category_options[cat["id"]] = f"{status_emoji} {cat['name']} ({cat['document_count']} docs)"

        selected_category = st.selectbox(
            "Select category:",
            options=list(category_options.keys()),
            format_func=lambda x: category_options[x]
        )

        # 선택된 카테고리 정보
        selected_cat_info = next(cat for cat in categories if cat["id"] == selected_category)
        st.info(f"**{selected_cat_info['name']}**\n\n{selected_cat_info['description']}")

        # 문서 업로드 섹션
        st.header("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload technical documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'doc', 'txt', 'md'],
            help="Supported formats: PDF, DOCX, DOC, TXT, MD"
        )

        if uploaded_files and st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                result = upload_documents(selected_category, uploaded_files)

                if "error" in result:
                    st.error(f"Upload failed: {result['error']}")
                else:
                    st.success(f"✅ Successfully uploaded {result['documents_added']} documents!")
                    st.rerun()

        # 시스템 통계
        st.header("📊 System Stats")
        try:
            stats_response = httpx.get(f"{BACKEND_URL}/api/v1/stats", timeout=10.0)
            if stats_response.status_code == 200:
                stats = stats_response.json()

                st.metric("Total Documents", stats["total_documents"])

                for category, data in stats["categories"].items():
                    if data["document_count"] > 0:
                        st.metric(
                            category.capitalize(),
                            data["document_count"]
                        )
            else:
                st.warning("Stats unavailable")
        except:
            st.warning("Could not load stats")

    # 메인 채팅 인터페이스
    st.header("💬 Chat with Documents")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 소스 정보 표시
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📄 Source Documents", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>{i}. {source['source']}</strong> (Score: {source['score']:.3f})<br>
                                <small>{source['content_preview']}</small>
                            </div>
                            """, unsafe_allow_html=True)

    # 사용자 입력
    if prompt := st.chat_input("Ask about your technical documents..."):
        # 카테고리에 문서가 있는지 확인
        if not selected_cat_info["is_available"]:
            st.error(f"No documents available in {selected_cat_info['name']} category. Please upload documents first.")
            return

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 어시스턴트 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = query_documents(selected_category, prompt, top_k=5)

                if "error" in result:
                    error_msg = f"Sorry, I encountered an error: {result['error']}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    # 답변 표시
                    st.markdown(result["answer"])

                    # 소스 문서 표시
                    if result["sources"]:
                        with st.expander("📄 Source Documents", expanded=False):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>{i}. {source['source']}</strong> (Score: {source['score']:.3f})<br>
                                    <small>{source['content_preview']}</small>
                                </div>
                                """, unsafe_allow_html=True)

                    # 메시지 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })

    # 채팅 초기화 버튼
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()