"""API endpoints for querying the knowledge base"""

import logging
import time
from typing import List

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    QueryRequest, 
    QueryResponse, 
    SourceInfo, 
    ResponseMetadata,
    Intent,
    SearchResult
)
from app.core.intent import detect_intent, get_simple_response, is_conversational
from app.core.hybrid_search import hybrid_search_with_fallback
from app.core.reranking import rerank_results
from app.core.search import has_sufficient_evidence
from app.core.llm import generate_answer
from app.core.safety import check_query_safety
from app.storage.vector_store import get_vector_store
from app.core.keyword_search import get_bm25_index

logger = logging.getLogger(__name__)

router = APIRouter()


def convert_to_source_info(results: List[SearchResult]) -> List[SourceInfo]:
    """Convert SearchResult objects to SourceInfo for response"""
    return [
        SourceInfo(
            chunk_id=result.chunk.chunk_id,
            text=result.chunk.text,
            source_file=result.chunk.source_file,
            page_number=result.chunk.page_number,
            similarity_score=round(result.score, 4)
        )
        for result in results
    ]


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base and get AI-generated answers.
    
    The pipeline:
    1. Detect user intent (greeting, chitchat, search query, etc.)
    2. For conversational intents, return simple responses
    3. For search queries:
       - Perform hybrid search (semantic + keyword)
       - Rerank results using cross-encoder and MMR
       - Generate answer using Mistral AI
       - Return comprehensive response with sources
    
    Args:
        request: Query request with user question
        
    Returns:
        QueryResponse with answer, sources, and metadata
    """
    start_time = time.time()
    logger.info(f"Query received: {request.query[:100]}...")
    
    try:
        # Step 0: Safety check (before intent detection to save costs)
        safety_check = check_query_safety(request.query)
        
        if not safety_check.is_safe:
            logger.warning(f"Query refused: {safety_check.category} - {request.query[:50]}...")
            total_time = (time.time() - start_time) * 1000
            
            return QueryResponse(
                status="success",
                query=request.query,
                intent="refused",
                answer=safety_check.refusal_message,
                sources=[],
                has_sufficient_evidence=False,
                metadata=ResponseMetadata(
                    search_time_ms=0.0,
                    llm_time_ms=0.0,
                    total_time_ms=round(total_time, 2)
                )
            )
        
        logger.info(f"Safety check passed in {safety_check.check_time_ms:.2f}ms")
        
        # Step 1: Detect intent
        intent = detect_intent(request.query)
        logger.info(f"Detected intent: {intent}")
        
        # Step 2: Handle conversational intents
        if is_conversational(intent):
            simple_response = get_simple_response(intent)
            total_time = (time.time() - start_time) * 1000
            
            return QueryResponse(
                status="success",
                query=request.query,
                intent=intent.value,
                answer=simple_response,
                sources=[],
                has_sufficient_evidence=True,  # Simple responses are always sufficient
                metadata=ResponseMetadata(
                    search_time_ms=0.0,
                    llm_time_ms=0.0,
                    total_time_ms=round(total_time, 2)
                )
            )
        
        # Step 3: Search knowledge base
        search_start = time.time()
        
        # Get vector store and BM25 index
        vector_store = get_vector_store()
        bm25_index = get_bm25_index()
        
        # Check if knowledge base is empty
        if len(vector_store) == 0:
            logger.warning("Knowledge base is empty")
            total_time = (time.time() - start_time) * 1000
            return QueryResponse(
                status="success",
                query=request.query,
                intent=intent.value,
                answer="I don't have any documents in my knowledge base yet. Please upload some documents first so I can help answer your questions.",
                sources=[],
                has_sufficient_evidence=False,
                metadata=ResponseMetadata(
                    search_time_ms=0.0,
                    llm_time_ms=0.0,
                    total_time_ms=round(total_time, 2)
                )
            )
        
        # Perform hybrid search with fallback
        search_results, search_method = hybrid_search_with_fallback(
            query=request.query,
            vector_store=vector_store,
            bm25_index=bm25_index,
            top_k=request.top_k * 2,  # Get more for reranking
            semantic_weight=0.6,
            fusion_method="rrf"
        )
        
        logger.info(f"Search method used: {search_method}")
        
        if not search_results:
            logger.warning("No search results found")
            search_time = (time.time() - search_start) * 1000
            total_time = (time.time() - start_time) * 1000
            
            return QueryResponse(
                status="success",
                query=request.query,
                intent=intent.value,
                answer="I couldn't find any relevant information in my knowledge base to answer your question. Could you try rephrasing your question or asking something else?",
                sources=[],
                has_sufficient_evidence=False,
                metadata=ResponseMetadata(
                    search_time_ms=round(search_time, 2),
                    llm_time_ms=0.0,
                    total_time_ms=round(total_time, 2)
                )
            )
        
        # Step 4: Rerank results
        # Try re-ranking, but fall back gracefully if models aren't available
        try:
            reranked_results = rerank_results(
                query=request.query,
                results=search_results,
                methods=["cross_encoder", "mmr"],
                mmr_lambda=0.7,  # Balance relevance (0.7) with diversity (0.3)
                top_k=request.top_k
            )
        except Exception as e:
            logger.warning(f"Re-ranking failed ({str(e)}), using search results as-is")
            # Just take top_k from search results
            reranked_results = search_results[:request.top_k]
        
        # Check evidence quality
        sufficient_evidence = has_sufficient_evidence(reranked_results)
        
        search_time = (time.time() - search_start) * 1000
        logger.info(f"Search completed in {search_time:.2f}ms, found {len(reranked_results)} results")
        
        # Step 5: Generate answer using Mistral AI
        llm_start = time.time()
        
        try:
            context_chunks = [result.chunk for result in reranked_results]
            answer_result = generate_answer(
                question=request.query,
                context_chunks=context_chunks,
                include_source_numbers=request.include_sources
            )
            
            llm_time = (time.time() - llm_start) * 1000
            logger.info(f"LLM generation completed in {llm_time:.2f}ms")
            
            # Check if LLM returned an error
            if answer_result.get("status") == "error":
                logger.error(f"LLM error: {answer_result['answer']}")
                # Still return search results even if LLM fails
                answer_text = f"I found relevant information in the documents, but encountered an error generating a detailed answer: {answer_result.get('answer', 'Unknown error')}"
            else:
                answer_text = answer_result.get("answer", "No answer generated")
        except Exception as e:
            llm_time = (time.time() - llm_start) * 1000
            logger.error(f"LLM generation failed: {e}")
            answer_text = f"I found relevant information in the documents, but couldn't generate an answer. Error: {str(e)}"
        
        # Step 6: Prepare response
        sources = convert_to_source_info(reranked_results) if request.include_sources else []
        total_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            status="success",
            query=request.query,
            intent=intent.value,
            answer=answer_text,
            sources=sources,
            has_sufficient_evidence=sufficient_evidence,
            metadata=ResponseMetadata(
                search_time_ms=round(search_time, 2),
                llm_time_ms=round(llm_time, 2),
                total_time_ms=round(total_time, 2)
            )
        )
        
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

