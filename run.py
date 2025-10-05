"""Convenience script to run the application"""

import uvicorn

from app.core.config import settings

if __name__ == "__main__":
    print(f"""
    ╔════════════════════════════════════════╗
    ║     RAG FastAPI Application            ║
    ║     Starting server...                 ║
    ╚════════════════════════════════════════╝
    
    🌐 Server: http://{settings.host}:{settings.port}
    📚 API Docs: http://{settings.host}:{settings.port}/docs
    🔧 Debug Mode: {settings.debug}
    """)
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )

