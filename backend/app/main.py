from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import os

from .config import settings, DOCUMENT_CATEGORIES
from .core.rag_engine import IndustrialRAGEngine
from .models.schemas import HealthCheck
from .routers import chat, documents

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RAG 엔진 글로벌 인스턴스
rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행되는 라이프사이클 관리"""
    global rag_engine

    # 시작 시
    logger.info("Starting Industrial RAG System...")
    logger.info(f"Ollama host: {settings.ollama_host}")
    logger.info(f"Ollama model: {settings.ollama_model}")

    try:
        # RAG 엔진 초기화
        rag_engine = IndustrialRAGEngine(settings)
        logger.info("RAG Engine initialized successfully")

        # 헬스 체크
        health = rag_engine.check_health()
        logger.info(f"System health: {health['status']}")

        if health['status'] == 'unhealthy':
            logger.error("System is unhealthy - check Ollama connection")
            logger.error(f"Make sure 'ollama serve' is running and model '{settings.ollama_model}' is downloaded")

    except Exception as e:
        logger.error(f"Failed to initialize RAG Engine: {e}")
        logger.error("Troubleshooting:")
        logger.error("1. Make sure 'ollama serve' is running")
        logger.error(f"2. Make sure model '{settings.ollama_model}' is downloaded: ollama pull {settings.ollama_model}")
        logger.error(f"3. Check if Ollama is accessible at: {settings.ollama_host}")
        raise

    yield

    # 종료 시
    logger.info("Shutting down Industrial RAG System...")


# FastAPI 앱 생성
app = FastAPI(
    title="Industrial Technical Document RAG System",
    description="영어 기술 문서 전용 RAG 시스템 - Llama 로컬 LLM 사용",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Industrial RAG System API - Llama LLM",
        "version": "1.0.0",
        "status": "running",
        "llm_type": "Llama (via Ollama)",
        "ollama_host": settings.ollama_host,
        "ollama_model": settings.ollama_model,
        "timestamp": datetime.now().isoformat(),
        "categories": list(DOCUMENT_CATEGORIES.keys()),
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        if not rag_engine:
            raise HTTPException(
                status_code=503,
                detail="RAG engine not initialized. Check server logs."
            )

        health = rag_engine.check_health()
        health["timestamp"] = datetime.now().isoformat()

        if health["status"] == "unhealthy":
            raise HTTPException(
                status_code=503,
                detail=f"System unhealthy: {health.get('error', 'Unknown error')}"
            )
        elif health["status"] == "warning":
            logger.warning("System running with warnings")

        return HealthCheck(**health)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@app.get("/categories")
async def get_categories():
    """사용 가능한 카테고리 목록"""
    try:
        categories_info = []
        stats = rag_engine.get_category_stats() if rag_engine else {}

        for cat_id, cat_info in DOCUMENT_CATEGORIES.items():
            category_stats = stats.get(cat_id, {"document_count": 0, "status": "unknown"})
            categories_info.append({
                "id": cat_id,
                "name": cat_info["name"],
                "description": cat_info["description"],
                "document_count": category_stats["document_count"],
                "status": category_stats["status"]
            })

        return {
            "categories": categories_info,
            "total_categories": len(categories_info),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get categories")


@app.get("/stats")
async def get_system_stats():
    """시스템 전체 통계"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")

        stats = rag_engine.get_category_stats()
        total_docs = sum(cat["document_count"] for cat in stats.values())

        return {
            "categories": stats,
            "total_documents": total_docs,
            "system_status": "healthy" if rag_engine else "unhealthy",
            "llm_type": "Llama (Ollama)",
            "ollama_model": settings.ollama_model,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system stats")


# 개발용 서버 실행
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting development server...")
    logger.info(f"Ollama should be running at: {settings.ollama_host}")
    logger.info(f"Expected model: {settings.ollama_model}")

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )