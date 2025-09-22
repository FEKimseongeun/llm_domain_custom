from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class CategoryEnum(str, Enum):
    """문서 카테고리 열거형 - 5개로 확장"""
    BIM = "bim"
    PROCESS = "process"
    PIPING = "piping"
    PROCUREMENT = "procurement"
    MECHANICAL = "mechanical"
class QueryRequest(BaseModel):
    """질의 요청 모델"""
    category: CategoryEnum = Field(..., description="Document category")
    question: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")

class SourceInfo(BaseModel):
    """소스 문서 정보"""
    source: str = Field(..., description="Source document name")
    score: float = Field(..., description="Relevance score")
    content_preview: str = Field(..., description="Content preview")

class QueryResponse(BaseModel):
    """질의 응답 모델"""
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(..., description="Source documents")
    category: str = Field(..., description="Document category")
    question: str = Field(..., description="Original question")

class DocumentUploadResponse(BaseModel):
    """문서 업로드 응답 모델"""
    status: str = Field(..., description="Upload status")
    documents_added: int = Field(..., description="Number of documents added")
    category: str = Field(..., description="Category")
    file_names: List[str] = Field(default=[], description="Uploaded file names")

class CategoryInfo(BaseModel):
    """카테고리 정보 모델"""
    id: str = Field(..., description="Category ID")
    name: str = Field(..., description="Category name")
    description: str = Field(..., description="Category description")
    document_count: int = Field(default=0, description="Number of documents")
    status: str = Field(default="active", description="Category status")

class HealthCheck(BaseModel):
    """헬스 체크 모델"""
    status: str = Field(..., description="Overall system status")
    llm_status: str = Field(..., description="LLM status")
    search_type: str = Field(..., description="Search method type")
    categories: List[str] = Field(..., description="Available categories")
    stores_loaded: int = Field(..., description="Number of loaded stores")
    timestamp: Optional[str] = Field(None, description="Check timestamp")