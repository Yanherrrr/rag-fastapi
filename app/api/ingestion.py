"""API endpoints for document ingestion"""

from typing import List
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.schemas import IngestionResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(
    files: List[UploadFile] = File(...)
):
    """
    Upload and ingest PDF files into the knowledge base.
    
    Args:
        files: List of PDF files to process
        
    Returns:
        IngestionResponse with processing statistics
    """
    # TODO: Implement in Phase 2
    logger.info(f"Ingestion endpoint called with {len(files)} files")
    
    raise HTTPException(
        status_code=501,
        detail="Ingestion endpoint not yet implemented. Coming in Phase 2!"
    )

