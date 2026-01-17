from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM API Keys
    anthropic_api_key: str
    openai_api_key: str
    
    # LLM Models
    claude_model: str = "claude-sonnet-4-6"
    claude_judge_model: str = "claude-haiku-4-5-20251001"
    embedding_model: str = "text-embedding-3-small"
    
    # App settings
    app_name: str = "AI Underwriting Copilot"
    debug: bool = True

    # ChromaDB settings
    chroma_persist_dir: str = "./chromadb"
    chroma_collection_name: str = "underwriting_docs"
    top_k_results: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
