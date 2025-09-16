from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정 관리"""

    # API 설정
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LLM 설정
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # 문서 설정
    documents_path: str = "./data"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # 로깅
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


# 전역 설정 인스턴스
settings = Settings()

# 문서 카테고리 정의
DOCUMENT_CATEGORIES = {
    "procurement": {
        "name": "Procurement/Contract",
        "description": "Purchase orders, contracts, specifications",
        "folder": "procurement"
    },
    "piping": {
        "name": "Piping",
        "description": "Piping designs, isometrics, specs",
        "folder": "piping"
    },
    "process": {
        "name": "Process",
        "description": "P&ID drawings, process flow diagrams",
        "folder": "process"
    },
    "mechanical": {
        "name": "Mechanical",
        "description": "Equipment datasheets, vendor prints",
        "folder": "mechanical"
    }
}