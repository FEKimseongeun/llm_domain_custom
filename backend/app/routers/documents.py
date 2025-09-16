from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List
import logging
import os
import aiofiles
from pathlib import Path
import uuid
from datetime import datetime

from ..models.schemas import DocumentUploadResponse, CategoryEnum
from ..config import settings, DOCUMENT_CATEGORIES

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


# 허용되는 파일 확장자
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.md'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def validate_file(file: UploadFile) -> None:
    """파일 유효성 검사"""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_ext}' not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_documents(
        category: CategoryEnum = Form(...),
        files: List[UploadFile] = File(...),
        rag_engine=Depends(get_rag_engine)
):
    """문서 업로드 엔드포인트"""
    try:
        logger.info(f"Document upload request - Category: {category}, Files: {len(files)}")

        # 카테고리 유효성 검사
        if category not in DOCUMENT_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category '{category}'. Must be one of: {list(DOCUMENT_CATEGORIES.keys())}"
            )

        # 파일 개수 제한
        if len(files) > 10:
            raise HTTPException(
                status_code=400,
                detail="Cannot upload more than 10 files at once"
            )

        if not files or (len(files) == 1 and not files[0].filename):
            raise HTTPException(
                status_code=400,
                detail="No files provided"
            )

        # 카테고리 폴더 경로
        category_path = Path(settings.documents_path) / category
        category_path.mkdir(parents=True, exist_ok=True)

        uploaded_files = []
        file_paths = []

        # 각 파일 처리
        for file in files:
            # 파일 유효성 검사
            validate_file(file)

            # 고유한 파일명 생성
            file_ext = Path(file.filename).suffix
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = category_path / unique_filename

            try:
                # 파일 저장
                async with aiofiles.open(file_path, 'wb') as f:
                    content = await file.read()

                    # 파일 크기 검사
                    if len(content) > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File '{file.filename}' is too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB"
                        )

                    await f.write(content)

                uploaded_files.append(file.filename)
                file_paths.append(str(file_path))
                logger.info(f"Saved file: {file_path}")

            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {e}")
                # 이미 저장된 파일들 정리
                for saved_path in file_paths:
                    try:
                        os.remove(saved_path)
                    except:
                        pass
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save file '{file.filename}': {str(e)}"
                )

        # RAG 엔진에 문서 추가
        try:
            result = await rag_engine.add_documents(category, file_paths)

            logger.info(f"Successfully uploaded and processed {len(uploaded_files)} files to {category}")

            return DocumentUploadResponse(
                status="success",
                documents_added=result["documents_added"],
                category=category,
                file_names=uploaded_files
            )

        except Exception as e:
            logger.error(f"Error adding documents to RAG engine: {e}")
            # 저장된 파일들 정리
            for file_path in file_paths:
                try:
                    os.remove(file_path)
                except:
                    pass
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process documents: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in document upload: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during file upload"
        )


@router.get("/documents/stats")
async def get_document_stats(rag_engine=Depends(get_rag_engine)):
    """문서 통계 조회"""
    try:
        stats = rag_engine.get_category_stats()

        # 파일 시스템에서 실제 파일 수도 확인
        filesystem_stats = {}
        for category in DOCUMENT_CATEGORIES.keys():
            category_path = Path(settings.documents_path) / category
            if category_path.exists():
                file_count = len([f for f in category_path.iterdir() if f.is_file()])
                filesystem_stats[category] = file_count
            else:
                filesystem_stats[category] = 0

        return {
            "document_stats": stats,
            "filesystem_stats": filesystem_stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document statistics")


@router.get("/documents/list/{category}")
async def list_documents(category: CategoryEnum):
    """카테고리별 문서 목록 조회"""
    try:
        if category not in DOCUMENT_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category '{category}'"
            )

        category_path = Path(settings.documents_path) / category

        if not category_path.exists():
            return {
                "category": category,
                "documents": [],
                "count": 0
            }

        documents = []
        for file_path in category_path.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                documents.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        # 수정 시간 기준 정렬
        documents.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "category": category,
            "documents": documents,
            "count": len(documents)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents for {category}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")