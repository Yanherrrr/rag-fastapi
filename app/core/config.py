"""Application configuration using Pydantic Settings"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Mistral AI Configuration
    mistral_api_key: str = "CF2DvjIoshzasO0mtBkPj44fo2nXDwPk"
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Search Configuration
    top_k_results: int = 5
    similarity_threshold: float = 0.6
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    
    # Storage Configuration
    vector_store_path: str = "data/vector_store.pkl"
    upload_dir: str = "uploads"
    
    # LLM Configuration
    llm_model: str = "mistral-medium"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def vector_store_full_path(self) -> Path:
        """Get full path to vector store file"""
        return Path(self.vector_store_path)
    
    @property
    def upload_dir_path(self) -> Path:
        """Get full path to upload directory"""
        return Path(self.upload_dir)
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        self.vector_store_full_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()

