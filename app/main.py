"""Main FastAPI application"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.schemas import StatusResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG FastAPI",
    description="Retrieval-Augmented Generation Pipeline with Mistral AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Startup & Shutdown Events =============

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting RAG FastAPI application...")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Vector store path: {settings.vector_store_path}")
    logger.info(f"Upload directory: {settings.upload_dir}")
    
    # Ensure directories exist
    settings.ensure_directories()
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down RAG FastAPI application...")


# ============= Root Endpoints =============

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {
        "message": "RAG FastAPI Backend",
        "status": "running",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get system status and statistics"""
    try:
        from app.storage.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        stats = vector_store.get_stats()
        
        return StatusResponse(
            status="ready",
            statistics={
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "embedding_dimension": settings.embedding_dimension,
                "vector_store_size_mb": stats.get("store_size_mb", 0.0),
                "memory_usage_mb": stats.get("memory_usage_mb", 0.0)
            }
        )
    except Exception as e:
        logger.warning(f"Error getting vector store stats: {e}")
        return StatusResponse(
            status="ready",
            statistics={
                "total_documents": 0,
                "total_chunks": 0,
                "embedding_dimension": settings.embedding_dimension,
                "vector_store_size_mb": 0.0
            }
        )


# ============= API Routes =============

# Import API routers
from app.api import ingestion, query

# Include ingestion router
app.include_router(
    ingestion.router,
    prefix="/api",
    tags=["Ingestion"],
    responses={404: {"description": "Not found"}}
)

# Query router will be added in Phase 4
# app.include_router(query.router, prefix="/api", tags=["query"])


# ============= Static Files =============

# Mount static files directory (will be created in Phase 6)
static_path = Path("frontend/static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# ============= Main Entry Point =============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )

