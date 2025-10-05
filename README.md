# rag-fastapi

A Retrieval-Augmented Generation (RAG) pipeline with FastAPI and Mistral AI, built from scratch without external RAG or search libraries.

## 🚀 Project Status

**Phase 1: Complete ✅** - Project setup and infrastructure

### What's Working:
- ✅ Complete project structure
- ✅ Configuration management with Pydantic
- ✅ FastAPI application skeleton
- ✅ API endpoint stubs
- ✅ Development environment ready

### Coming Next:
- 🔨 Phase 2: Data ingestion pipeline
- 🔨 Phase 3: Search implementation
- 🔨 Phase 4: Query processing & LLM
- 🔨 Phase 5: Bonus features
- 🔨 Phase 6: UI development

## 📋 Features (Planned)

- **PDF Ingestion**: Upload and process PDF documents
- **Smart Chunking**: Intelligent text splitting with overlap
- **Hybrid Search**: Semantic (embeddings) + Keyword (BM25)
- **Custom Vector Store**: No external vector database needed
- **Intent Detection**: Smart query routing
- **Answer Generation**: Powered by Mistral AI
- **Source Citations**: Transparent answer sourcing
- **Vanilla JS UI**: Clean, simple chat interface

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-fastapi
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings (Mistral API key is pre-configured)
```

## 🚀 Running the Application

### Development Server

```bash
python run.py
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access Points
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
rag-fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── ingestion.py     # Document upload endpoints
│   │   └── query.py         # Query endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration management
│   │   ├── chunking.py      # Text chunking (Coming in Phase 2)
│   │   ├── embeddings.py    # Embedding generation (Phase 2)
│   │   ├── search.py        # Hybrid search (Phase 3)
│   │   ├── ranking.py       # Result re-ranking (Phase 3)
│   │   └── llm.py           # Mistral AI client (Phase 4)
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   └── storage/
│       ├── __init__.py
│       └── vector_store.py  # Custom vector storage (Phase 2)
├── frontend/
│   ├── index.html           # Web UI (Phase 6)
│   └── static/
│       ├── style.css
│       └── app.js
├── data/                    # Vector store storage
├── uploads/                 # Temporary PDF uploads
├── tests/                   # Test suite
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── .gitignore
├── plan.md                 # Detailed implementation plan
├── task.md                 # Original task description
├── run.py                  # Convenience run script
└── README.md
```

## 🔧 Configuration

Edit `.env` file to configure:

```env
# API Keys
MISTRAL_API_KEY=your_api_key_here

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Search
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.6
SEMANTIC_WEIGHT=0.7
KEYWORD_WEIGHT=0.3

# Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=mistral-medium
```

## 📚 API Endpoints

### Status
- `GET /health` - Health check
- `GET /api/status` - System statistics

### Ingestion (Coming in Phase 2)
- `POST /api/ingest` - Upload PDF files

### Query (Coming in Phase 4)
- `POST /api/query` - Query the knowledge base

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/
```

## 📖 Documentation

For detailed implementation plan, see [plan.md](plan.md).

## 🎯 Design Decisions

### Why No External Libraries for RAG/Search?
Per project requirements, we implement core RAG components from scratch:
- **Custom Vector Store**: Numpy-based with pickle serialization
- **BM25 Implementation**: From-scratch keyword search
- **Hybrid Search**: Manual combination of semantic + keyword

### Tech Stack Choices
- **FastAPI**: Modern, fast, automatic API docs
- **Sentence Transformers**: Quality embeddings, runs locally
- **Mistral AI**: Powerful LLM with good API
- **Vanilla JS**: Simple, no build process needed

## 📝 License

MIT

## 👤 Author

Built as a demonstration project for RAG pipeline implementation.

---

**Note**: This is Phase 1. Core functionality (ingestion, search, query) coming in subsequent phases.
