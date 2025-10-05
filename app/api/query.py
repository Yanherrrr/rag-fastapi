"""API endpoints for querying the knowledge base"""

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base and get AI-generated answers.
    
    Args:
        request: Query request with user question
        
    Returns:
        QueryResponse with answer and sources
    """
    # TODO: Implement in Phase 4
    logger.info(f"Query endpoint called with: {request.query}")
    
    raise HTTPException(
        status_code=501,
        detail="Query endpoint not yet implemented. Coming in Phase 4!"
    )

