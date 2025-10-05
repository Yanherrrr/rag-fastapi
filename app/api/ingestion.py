"""API endpoints for document ingestion"""

import os
import time
import logging
from typing import List
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.schemas import IngestionResponse, FileInfo, ErrorResponse
from app.core.chunking import extract_text_from_pdf, chunk_pages
from app.core.embeddings import generate_embeddings
from app.storage.vector_store import get_vector_store
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=IngestionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def ingest_documents(
    files: List[UploadFile] = File(...)
):
    """
    Upload and ingest PDF files into the knowledge base.
    
    This endpoint:
    1. Accepts one or more PDF files
    2. Extracts text from each PDF
    3. Chunks the text with overlap
    4. Generates embeddings for each chunk
    5. Stores embeddings in the vector database
    6. Returns processing statistics
    
    Args:
        files: List of PDF files to process
        
    Returns:
        IngestionResponse with processing statistics
        
    Raises:
        HTTPException: If no files provided or processing fails
    """
    start_time = time.time()
    
    # Validate input
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided. Please upload at least one PDF file."
        )
    
    # Validate file types
    pdf_files = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Skipping non-PDF file: {file.filename}")
            continue
        pdf_files.append(file)
    
    if not pdf_files:
        raise HTTPException(
            status_code=400,
            detail="No PDF files found. Please upload PDF files only."
        )
    
    logger.info(f"Starting ingestion of {len(pdf_files)} PDF files")
    
    # Statistics
    total_chunks = 0
    file_info_list = []
    temp_files = []
    
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        
        # Process each PDF
        all_chunks = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing file: {pdf_file.filename}")
                
                # Save uploaded file temporarily
                temp_path = settings.upload_dir_path / pdf_file.filename
                temp_files.append(temp_path)
                
                with open(temp_path, 'wb') as f:
                    content = await pdf_file.read()
                    f.write(content)
                
                # Extract text from PDF
                pages = extract_text_from_pdf(str(temp_path))
                
                if not pages:
                    logger.warning(f"No text extracted from {pdf_file.filename}")
                    file_info_list.append(
                        FileInfo(
                            filename=pdf_file.filename,
                            chunks=0,
                            pages=0
                        )
                    )
                    continue
                
                # Chunk the text
                chunks = chunk_pages(
                    pages=pages,
                    chunk_size=settings.chunk_size,
                    overlap=settings.chunk_overlap
                )
                
                if not chunks:
                    logger.warning(f"No chunks created from {pdf_file.filename}")
                    file_info_list.append(
                        FileInfo(
                            filename=pdf_file.filename,
                            chunks=0,
                            pages=len(pages)
                        )
                    )
                    continue
                
                # Collect chunks for batch embedding generation
                all_chunks.extend(chunks)
                
                # Store file info
                file_info_list.append(
                    FileInfo(
                        filename=pdf_file.filename,
                        chunks=len(chunks),
                        pages=len(pages)
                    )
                )
                
                total_chunks += len(chunks)
                logger.info(f"Created {len(chunks)} chunks from {pdf_file.filename}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.filename}: {e}")
                # Continue with other files even if one fails
                file_info_list.append(
                    FileInfo(
                        filename=pdf_file.filename,
                        chunks=0,
                        pages=0
                    )
                )
                continue
        
        # Generate embeddings for all chunks (batch processing)
        if all_chunks:
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            
            try:
                # Extract text from chunks
                texts = [chunk.text for chunk in all_chunks]
                
                # Generate embeddings
                embeddings = generate_embeddings(texts, batch_size=32)
                
                logger.info(f"Generated embeddings shape: {embeddings.shape}")
                
                # Add to vector store
                vector_store.add_documents(all_chunks, embeddings)
                
                # Save vector store
                vector_store.save()
                
                logger.info(f"Successfully stored {len(all_chunks)} chunks in vector database")
                
            except Exception as e:
                logger.error(f"Error generating embeddings or storing: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate embeddings or store in database: {str(e)}"
                )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(
            f"Ingestion complete: {len(pdf_files)} files, "
            f"{total_chunks} chunks, {processing_time:.2f}s"
        )
        
        return IngestionResponse(
            status="success",
            files_processed=len(pdf_files),
            total_chunks=total_chunks,
            processing_time_seconds=round(processing_time, 2),
            files=file_info_list
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
        
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}")


@router.delete("/clear")
async def clear_vector_store():
    """
    Clear all documents from the vector store.
    
    ⚠️ WARNING: This will delete all ingested documents permanently!
    
    Returns:
        Success message
    """
    try:
        vector_store = get_vector_store()
        vector_store.clear()
        vector_store.save()
        
        logger.warning("Vector store cleared by user request")
        
        return {
            "status": "success",
            "message": "Vector store cleared successfully",
            "total_chunks_deleted": 0
        }
        
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear vector store: {str(e)}"
        )
