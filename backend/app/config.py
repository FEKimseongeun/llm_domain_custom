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

# 문서 카테고리 정의 - 5개로 확장
DOCUMENT_CATEGORIES = {
    "bim": {
        "name": "3D/BIM",
        "description": "3D models, BIM coordination, clash detection reports, LOD specifications",
        "folder": "bim"
    },
    "process": {
        "name": "Process",
        "description": "P&ID drawings, PFDs, heat & material balance, HAZOP studies",
        "folder": "process"
    },
    "piping": {
        "name": "Piping",
        "description": "Piping specs, isometrics, stress analysis, support drawings",
        "folder": "piping"
    },
    "procurement": {
        "name": "Procurement/Contract",
        "description": "Purchase orders, contracts, vendor lists, bidding documents",
        "folder": "procurement"
    },
    "mechanical": {
        "name": "Mechanical",
        "description": "Equipment datasheets, pump curves, compressor specs, maintenance manuals",
        "folder": "mechanical"
    }
}

# EPC 기술 용어 사전
TECHNICAL_GLOSSARY = {
    "bim": ["LOD", "clash", "navisworks", "revit", "coordination", "3D model", "digital twin",
            "federated model", "IFC", "BCF", "COBie", "E3D", "S3D", "Smart3D", "Everything3D", "HEXAGON", "AVEVA"],

    "process": ["P&ID", "PFD", "HAZOP", "SIL", "heat balance", "mass balance", "flow rate",
                "pressure drop", "control valve", "PSV", "instrumentation", "DCS", "ESD"],

    "piping": ["schedule", "ASME B31.3", "isometric", "stress analysis", "pipe support",
               "flange", "welding", "NPS", "pipe spec", "branch connection", "tie-in"],

    "procurement": ["RFQ", "RFP", "bid evaluation", "vendor", "INCOTERMS", "L/C",
                    "performance bond", "warranty", "expediting", "inspection", "FAT", "SAT"],

    "mechanical": ["API", "ASME", "vibration", "bearing", "seal", "impeller", "NPSH",
                   "pump curve", "compressor", "heat exchanger", "vessel", "rotating equipment"]
}