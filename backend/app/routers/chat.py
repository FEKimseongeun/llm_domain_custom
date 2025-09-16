from fastapi import APIRouter, HTTPException, Depends
import logging
from datetime import datetime

from ..models.schemas import QueryRequest, QueryResponse
from ..config import DOCUMENT_CATEGORIES

logger = logging.getLogger(__name__)

router = APIRouter()


def get_rag_engine():
    """RAG 엔진 의존성 주입"""
    from ..main import rag_engine
    if not rag_engine:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. Please check server logs."
        )
    return rag_engine


@router.post("/chat", response_model=QueryResponse)
async def chat_with_documents(
        request: QueryRequest,
        rag_engine=Depends(get_rag_engine)
):
    """문서 기반 채팅 엔드포인트"""
    try:
        logger.info(f"Chat request - Category: {request.category}, Question: {request.question[:50]}...")

        # 카테고리 유효성 검사
        if request.category not in DOCUMENT_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category '{request.category}'. Must be one of: {list(DOCUMENT_CATEGORIES.keys())}"
            )

        # 질문 길이 검사
        if len(request.question.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Question must be at least 3 characters long"
            )

        # RAG 엔진으로 질의
        result = await rag_engine.query(
            category=request.category,
            question=request.question.strip(),
            top_k=request.top_k
        )

        logger.info(f"Chat successful - Category: {request.category}, Sources: {len(result.get('sources', []))}")
        return QueryResponse(**result)

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error - Category: {request.category}, Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing your question. Please try again."
        )


@router.get("/chat/categories")
async def get_chat_categories():
    """채팅에서 사용 가능한 카테고리 목록"""
    try:
        rag_engine = get_rag_engine()
        stats = rag_engine.get_category_stats()

        categories = []
        for cat_id, cat_info in DOCUMENT_CATEGORIES.items():
            category_stats = stats.get(cat_id, {"document_count": 0, "status": "unknown"})
            categories.append({
                "id": cat_id,
                "name": cat_info["name"],
                "description": cat_info["description"],
                "document_count": category_stats["document_count"],
                "is_available": category_stats["document_count"] > 0
            })

        return {
            "categories": categories,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get categories")