"""Pydantic schemas for request/response models"""

from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


# ============= Enums =============

class Intent(str, Enum):
    """Query intent types"""
    SEARCH_KNOWLEDGE_BASE = "search"
    GREETING = "greeting"
    CHITCHAT = "chitchat"
    CLARIFICATION = "clarification"
    GOODBYE = "goodbye"


# ============= Base Models =============

class PageContent(BaseModel):
    """Represents content from a single PDF page"""
    page_number: int
    text: str
    source_file: str


class Chunk(BaseModel):
    """Represents a text chunk with metadata"""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Search result with chunk and score"""
    chunk: Chunk
    score: float
    rank: int


# ============= Request Models =============

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = Field(default=True)


# ============= Response Models =============

class SourceInfo(BaseModel):
    """Information about a source chunk"""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    similarity_score: float


class ResponseMetadata(BaseModel):
    """Metadata about response processing time"""
    search_time_ms: float
    llm_time_ms: float
    total_time_ms: float


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    status: str = "success"
    query: str
    intent: str
    answer: str
    sources: List[SourceInfo] = Field(default_factory=list)
    has_sufficient_evidence: bool
    metadata: ResponseMetadata


class FileInfo(BaseModel):
    """Information about a processed file"""
    filename: str
    chunks: int
    pages: int


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint"""
    status: str = "success"
    files_processed: int
    total_chunks: int
    processing_time_seconds: float
    files: List[FileInfo]


class StatusResponse(BaseModel):
    """Response model for status endpoint"""
    status: str
    statistics: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = "error"
    message: str
    detail: Optional[str] = None

