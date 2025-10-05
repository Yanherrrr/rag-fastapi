# RAG Pipeline - Product Requirements Document (PRD)

## Project Overview

**Project Name**: RAG-FastAPI - From-Scratch Retrieval-Augmented Generation System

**Objective**: Build a complete RAG pipeline that processes PDF documents and answers user questions using Mistral AI, implementing all core components (search, ranking, embeddings storage) from scratch without external RAG or search libraries.

**Tech Stack**:
- Backend: FastAPI + Python 3.9+
- Frontend: Vanilla JavaScript + HTML/CSS
- LLM: Mistral AI API
- Storage: Custom numpy-based vector store (no external vector DB)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                  (Vanilla JS + HTML/CSS)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                            │
├─────────────────────────────────────────────────────────────────┤
│  POST /api/ingest    │  POST /api/query    │  GET /api/status  │
└──────────┬────────────┴──────────┬──────────┴────────────────────┘
           │                       │
    ┌──────▼──────┐         ┌──────▼──────┐
    │  INGESTION  │         │    QUERY    │
    │  PIPELINE   │         │  PIPELINE   │
    └──────┬──────┘         └──────┬──────┘
           │                       │
           ▼                       ▼
    ┌─────────────────────────────────────┐
    │      CUSTOM VECTOR STORE            │
    │   (numpy arrays + JSON metadata)    │
    └─────────────────────────────────────┘
                     │
                     ▼
              ┌────────────┐
              │ Mistral AI │
              │    API     │
              └────────────┘
```

---

## Core Components

### 1. Ingestion Pipeline
- PDF text extraction
- Intelligent chunking with overlap
- Embedding generation
- Vector storage

### 2. Query Pipeline
- Intent detection
- Query transformation
- Hybrid search (semantic + keyword)
- Result re-ranking
- LLM generation with prompt engineering

### 3. Storage Layer
- Custom vector store implementation
- Metadata management
- Efficient similarity search

### 4. User Interface
- File upload interface
- Chat interface
- Source citation display

---

## Detailed Implementation Phases

## **PHASE 1: Project Setup & Infrastructure** (Est: 1-2 hours)

### Step 1.1: Initialize Project Structure
- [ ] Create directory structure
- [ ] Initialize git repository
- [ ] Create `.gitignore`
- [ ] Set up virtual environment

**Files to Create**:
```
rag-fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   └── query.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── search.py
│   │   ├── ranking.py
│   │   └── llm.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── storage/
│       ├── __init__.py
│       └── vector_store.py
├── frontend/
│   ├── index.html
│   └── static/
│       ├── style.css
│       └── app.js
├── data/
├── uploads/
├── tests/
│   ├── __init__.py
│   ├── test_chunking.py
│   └── test_search.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── plan.md
```

### Step 1.2: Create requirements.txt
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0

# PDF Processing
PyPDF2==3.0.1

# Embeddings & NLP
sentence-transformers==2.2.2
numpy==1.24.3
scikit-learn==1.3.2

# LLM API
mistralai==0.1.2
httpx==0.25.0

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
```

### Step 1.3: Configuration Setup
- [ ] Create `.env.example` with Mistral API key
- [ ] Create `app/core/config.py` with Pydantic Settings
- [ ] Set up logging configuration

**Key Configuration Variables**:
```python
MISTRAL_API_KEY: str
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50
TOP_K_RESULTS: int = 5
SIMILARITY_THRESHOLD: float = 0.6
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH: str = "data/vector_store.pkl"
```

---

## **PHASE 2: Data Ingestion Pipeline** (Est: 3-4 hours)

### Step 2.1: PDF Text Extraction (`app/core/chunking.py`)
- [ ] Implement PDF reader using PyPDF2
- [ ] Extract text page-by-page
- [ ] Preserve metadata (filename, page numbers)
- [ ] Handle extraction errors gracefully

**Key Functions**:
```python
def extract_text_from_pdf(file_path: str) -> List[PageContent]
def clean_text(text: str) -> str
```

**Considerations**:
- Handle corrupted PDFs
- Remove headers/footers (repeated text across pages)
- Preserve important whitespace
- Handle multi-column layouts (best effort)

### Step 2.2: Text Chunking Algorithm
- [ ] Implement fixed-size chunking with overlap
- [ ] Ensure chunks don't break mid-sentence
- [ ] Add chunk metadata (source file, page, chunk_id)
- [ ] Optimize chunk size for retrieval quality

**Key Functions**:
```python
def chunk_text(
    text: str, 
    chunk_size: int, 
    overlap: int,
    metadata: dict
) -> List[Chunk]

def split_into_sentences(text: str) -> List[str]
```

**Algorithm**:
1. Split text into sentences
2. Combine sentences until reaching chunk_size
3. Add overlap from previous chunk
4. Preserve metadata throughout

**Edge Cases**:
- Very short documents (< chunk_size)
- Very long sentences (> chunk_size)
- Empty pages

### Step 2.3: Embedding Generation (`app/core/embeddings.py`)
- [ ] Initialize sentence-transformers model
- [ ] Batch embedding generation
- [ ] Handle large documents efficiently
- [ ] Add caching mechanism

**Key Functions**:
```python
class EmbeddingGenerator:
    def __init__(self, model_name: str)
    def generate_embeddings(self, texts: List[str]) -> np.ndarray
    def generate_single_embedding(self, text: str) -> np.ndarray
```

**Optimization**:
- Batch process for efficiency
- Use GPU if available
- Normalize embeddings for cosine similarity

### Step 2.4: Custom Vector Store (`app/storage/vector_store.py`)
- [ ] Design data structure for vectors + metadata
- [ ] Implement save/load functionality
- [ ] Add document management (add, delete, list)
- [ ] Ensure thread-safety for concurrent access

**Key Classes & Methods**:
```python
class VectorStore:
    def __init__(self, store_path: str)
    def add_documents(self, chunks: List[Chunk], embeddings: np.ndarray)
    def save(self)
    def load(self)
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]
    def get_stats(self) -> dict
    def clear(self)
```

**Storage Format**:
```python
{
    "embeddings": np.ndarray,  # Shape: (n_chunks, embedding_dim)
    "metadata": [
        {
            "chunk_id": str,
            "text": str,
            "source_file": str,
            "page_number": int,
            "chunk_index": int
        },
        ...
    ],
    "version": "1.0"
}
```

### Step 2.5: Ingestion API Endpoint (`app/api/ingestion.py`)
- [ ] Create POST `/api/ingest` endpoint
- [ ] Handle multipart file upload
- [ ] Process PDFs asynchronously (or sync for simplicity)
- [ ] Return ingestion statistics

**Endpoint Specification**:
```python
POST /api/ingest
Content-Type: multipart/form-data

Request:
  files: List[UploadFile]

Response (200):
{
    "status": "success",
    "files_processed": 3,
    "total_chunks": 150,
    "processing_time_seconds": 12.5
}

Response (400):
{
    "status": "error",
    "message": "No PDF files provided"
}
```

---

## **PHASE 3: Search Implementation** (Est: 4-5 hours)

### Step 3.1: Semantic Search (`app/core/search.py`)
- [ ] Implement cosine similarity search
- [ ] Use numpy for efficient computation
- [ ] Return top-k results with scores

**Key Functions**:
```python
def cosine_similarity_search(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int
) -> Tuple[np.ndarray, np.ndarray]  # (indices, scores)
```

**Implementation**:
```python
# Cosine similarity = dot product of normalized vectors
scores = np.dot(query_embedding, doc_embeddings.T) / (
    np.linalg.norm(query_embedding) * 
    np.linalg.norm(doc_embeddings, axis=1)
)
top_k_indices = np.argsort(scores)[::-1][:top_k]
```

### Step 3.2: Keyword Search (BM25 from Scratch)
- [ ] Implement BM25 algorithm
- [ ] Build inverted index for documents
- [ ] Calculate IDF scores
- [ ] Score documents based on term frequency

**Key Classes**:
```python
class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75)
    def tokenize(self, text: str) -> List[str]
    def build_index(self)
    def score(self, query: str, document_idx: int) -> float
    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]
```

**BM25 Formula**:
```
score(D, Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1×(1-b+b×|D|/avgdl))

where:
- f(qi,D) = frequency of term qi in document D
- |D| = length of document D
- avgdl = average document length
- k1, b = tuning parameters
- IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
```

### Step 3.3: Hybrid Search Strategy
- [ ] Combine semantic and keyword search
- [ ] Implement score normalization
- [ ] Use weighted combination or RRF

**Key Functions**:
```python
def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    vector_store: VectorStore,
    bm25_index: BM25,
    top_k: int,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[SearchResult]
```

**Combination Strategies**:

**Option A: Weighted Sum**
```python
final_score = semantic_weight × semantic_score + keyword_weight × keyword_score
```

**Option B: Reciprocal Rank Fusion (RRF)**
```python
RRF_score = Σ 1 / (k + rank_i)  # k = 60 typically
```

### Step 3.4: Result Re-ranking (`app/core/ranking.py`)
- [ ] Implement diversity-based re-ranking
- [ ] Remove duplicate or near-duplicate chunks
- [ ] Boost results with better metadata

**Key Functions**:
```python
def rerank_results(
    results: List[SearchResult],
    diversity_weight: float = 0.2
) -> List[SearchResult]

def remove_duplicates(
    results: List[SearchResult],
    similarity_threshold: float = 0.95
) -> List[SearchResult]
```

---

## **PHASE 4: Query Processing & LLM Integration** (Est: 3-4 hours)

### Step 4.1: Intent Detection
- [ ] Implement simple intent classifier
- [ ] Detect non-search intents (greetings, chitchat)
- [ ] Use keyword matching + optional LLM fallback

**Key Functions**:
```python
class IntentDetector:
    def detect(self, query: str) -> Intent
    
class Intent(Enum):
    SEARCH_KNOWLEDGE_BASE = "search"
    GREETING = "greeting"
    CHITCHAT = "chitchat"
    CLARIFICATION = "clarification"
```

**Intent Detection Rules**:
```python
GREETING_PATTERNS = ["hello", "hi", "hey", "good morning", "good afternoon"]
CHITCHAT_PATTERNS = ["how are you", "what's up", "thank you", "thanks"]
GOODBYE_PATTERNS = ["bye", "goodbye", "see you", "exit"]

# If no pattern matches and query is short (< 5 words), use LLM to classify
```

### Step 4.2: Query Transformation
- [ ] Implement query expansion
- [ ] Rephrase ambiguous queries
- [ ] Extract key entities

**Key Functions**:
```python
def transform_query(query: str) -> str
def expand_with_synonyms(query: str) -> str
def extract_keywords(query: str) -> List[str]
```

**Techniques**:
1. Convert questions to statements
2. Expand acronyms
3. Add context from conversation history (optional)

### Step 4.3: Mistral AI Integration (`app/core/llm.py`)
- [ ] Initialize Mistral client
- [ ] Implement prompt templates
- [ ] Handle API errors and retries
- [ ] Implement streaming (optional)

**Key Classes**:
```python
class MistralClient:
    def __init__(self, api_key: str)
    def generate_answer(
        self, 
        query: str, 
        context_chunks: List[str],
        temperature: float = 0.7
    ) -> str
    def detect_intent(self, query: str) -> str
```

**Prompt Template**:
```python
ANSWER_PROMPT = """You are a helpful AI assistant. Answer the user's question based ONLY on the provided context. If the context doesn't contain enough information to answer the question, respond with "I don't have sufficient information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
```

### Step 4.4: Query API Endpoint (`app/api/query.py`)
- [ ] Create POST `/api/query` endpoint
- [ ] Orchestrate full query pipeline
- [ ] Return answer with sources and metadata

**Endpoint Specification**:
```python
POST /api/query
Content-Type: application/json

Request:
{
    "query": str,
    "top_k": int = 5,
    "include_sources": bool = true
}

Response (200):
{
    "status": "success",
    "query": str,
    "intent": str,
    "answer": str,
    "sources": [
        {
            "text": str,
            "source_file": str,
            "page_number": int,
            "similarity_score": float
        },
        ...
    ],
    "has_sufficient_evidence": bool,
    "processing_time_seconds": float
}

Response (200 - No search needed):
{
    "status": "success",
    "query": "hello",
    "intent": "greeting",
    "answer": "Hello! How can I help you today?",
    "sources": [],
    "has_sufficient_evidence": true,
    "processing_time_seconds": 0.1
}
```

---

## **PHASE 5: Bonus Features** (Est: 2-3 hours)

### Step 5.1: Citation Requirements
- [ ] Implement similarity threshold checking
- [ ] Refuse to answer if confidence is low
- [ ] Return "insufficient evidence" message

**Implementation**:
```python
def check_sufficient_evidence(
    search_results: List[SearchResult],
    threshold: float = 0.6
) -> bool:
    if not search_results:
        return False
    return search_results[0].score >= threshold
```

### Step 5.2: Hallucination Filter
- [ ] Extract sentences from generated answer
- [ ] Check each sentence against source chunks
- [ ] Flag unsupported claims

**Key Functions**:
```python
def detect_hallucinations(
    answer: str,
    source_chunks: List[str]
) -> List[str]  # Returns list of unsupported sentences

def sentence_entailment_check(
    sentence: str,
    context: str
) -> float  # Returns confidence score
```

### Step 5.3: Query Refusal Policies
- [ ] Detect PII in queries
- [ ] Detect medical/legal questions
- [ ] Return appropriate disclaimers

**Patterns to Detect**:
```python
PII_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
    r'\b\d{16}\b',  # Credit card
    r'\b[\w\.-]+@[\w\.-]+\.\w+\b'  # Email
]

MEDICAL_KEYWORDS = ["diagnose", "prescription", "medication", "treatment"]
LEGAL_KEYWORDS = ["legal advice", "sue", "lawsuit", "contract"]
```

### Step 5.4: Answer Shaping
- [ ] Detect if answer should be list/table
- [ ] Switch prompt templates based on intent
- [ ] Format structured outputs

---

## **PHASE 6: User Interface** (Est: 2-3 hours)

### Step 6.1: HTML Structure (`frontend/index.html`)
- [ ] Create chat container
- [ ] Add file upload section
- [ ] Add message display area
- [ ] Add input form

**UI Components**:
```html
1. Header with title
2. Upload section:
   - File input (accept PDF only)
   - Upload button
   - Status indicator
3. Chat container:
   - Messages area (scrollable)
   - User messages (right-aligned)
   - Bot messages (left-aligned)
   - Source citations (expandable)
4. Input section:
   - Text input
   - Send button
   - Character counter
```

### Step 6.2: Styling (`frontend/static/style.css`)
- [ ] Create modern, clean design
- [ ] Responsive layout
- [ ] Message bubbles styling
- [ ] Loading indicators

**Design Guidelines**:
- Color scheme: Blue/white with accents
- Font: System fonts (sans-serif)
- Mobile-responsive (media queries)
- Smooth animations for messages

### Step 6.3: JavaScript Logic (`frontend/static/app.js`)
- [ ] Implement file upload function
- [ ] Implement query submission
- [ ] Handle API responses
- [ ] Display messages and sources

**Key Functions**:
```javascript
async function uploadFiles()
async function sendQuery()
function displayUserMessage(text)
function displayBotMessage(data)
function displaySources(sources)
function showLoading()
function hideLoading()
```

### Step 6.4: FastAPI Static File Serving
- [ ] Mount static file directory
- [ ] Serve index.html at root
- [ ] Add proper CORS headers

---

## **PHASE 7: Testing & Quality Assurance** (Est: 2-3 hours)

### Step 7.1: Unit Tests
- [ ] Test chunking algorithm
- [ ] Test embedding generation
- [ ] Test search functions
- [ ] Test BM25 implementation

**Test Files**:
```python
tests/test_chunking.py
tests/test_search.py
tests/test_ranking.py
tests/test_vector_store.py
```

### Step 7.2: Integration Tests
- [ ] Test ingestion endpoint
- [ ] Test query endpoint
- [ ] Test error handling

### Step 7.3: Manual Testing
- [ ] Upload sample PDFs
- [ ] Test various query types
- [ ] Test edge cases
- [ ] Test UI on different browsers

**Test Scenarios**:
1. Upload single PDF
2. Upload multiple PDFs
3. Query with good context
4. Query with no context
5. Greeting messages
6. Long queries
7. Empty queries
8. Special characters

---

## **PHASE 8: Documentation & Deployment** (Est: 2 hours)

### Step 8.1: README.md
- [ ] System overview and architecture diagram
- [ ] Installation instructions
- [ ] Usage examples
- [ ] API documentation
- [ ] Design decisions and trade-offs

**README Sections**:
1. Project Overview
2. Architecture
3. Installation
4. Running the Application
5. API Endpoints
6. Design Decisions
7. Chunking Strategy Considerations
8. Search Strategy (Semantic + Keyword)
9. Future Improvements
10. License

### Step 8.2: Code Documentation
- [ ] Add docstrings to all functions
- [ ] Add inline comments for complex logic
- [ ] Document configuration options

### Step 8.3: Create Demo Data
- [ ] Add sample PDFs for testing
- [ ] Create sample queries document

### Step 8.4: Git Commit History
- [ ] Make logical, atomic commits
- [ ] Write clear commit messages
- [ ] Tag important milestones

**Commit Strategy**:
```
feat: Add PDF ingestion pipeline
feat: Implement custom vector store
feat: Add semantic search
feat: Implement BM25 keyword search
feat: Add hybrid search with RRF
feat: Integrate Mistral AI
feat: Create query API endpoint
feat: Add vanilla JS frontend
feat: Implement intent detection
feat: Add citation requirements
docs: Complete README with architecture
test: Add unit tests for core components
```

---

## API Specifications (Complete)

### 1. Ingestion Endpoint
```
POST /api/ingest
Content-Type: multipart/form-data

Request:
  files: List[UploadFile]  # One or more PDF files

Response (200):
{
    "status": "success",
    "files_processed": int,
    "total_chunks": int,
    "processing_time_seconds": float,
    "files": [
        {
            "filename": str,
            "chunks": int,
            "pages": int
        }
    ]
}

Response (400):
{
    "status": "error",
    "message": str
}
```

### 2. Query Endpoint
```
POST /api/query
Content-Type: application/json

Request:
{
    "query": str,
    "top_k": int = 5,
    "include_sources": bool = true
}

Response (200):
{
    "status": "success",
    "query": str,
    "intent": str,
    "answer": str,
    "sources": [
        {
            "chunk_id": str,
            "text": str,
            "source_file": str,
            "page_number": int,
            "similarity_score": float
        }
    ],
    "has_sufficient_evidence": bool,
    "metadata": {
        "search_time_ms": float,
        "llm_time_ms": float,
        "total_time_ms": float
    }
}
```

### 3. Status Endpoint
```
GET /api/status

Response (200):
{
    "status": "ready",
    "statistics": {
        "total_documents": int,
        "total_chunks": int,
        "embedding_dimension": int,
        "vector_store_size_mb": float
    }
}
```

### 4. Clear Endpoint (Optional)
```
DELETE /api/clear

Response (200):
{
    "status": "success",
    "message": "Vector store cleared"
}
```

---

## Data Models (Pydantic Schemas)

```python
# app/models/schemas.py

class PageContent(BaseModel):
    page_number: int
    text: str
    source_file: str

class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    metadata: dict = {}

class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    include_sources: bool = True

class QueryResponse(BaseModel):
    status: str
    query: str
    intent: str
    answer: str
    sources: List[SourceInfo]
    has_sufficient_evidence: bool
    metadata: ResponseMetadata

class SourceInfo(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    similarity_score: float

class ResponseMetadata(BaseModel):
    search_time_ms: float
    llm_time_ms: float
    total_time_ms: float

class IngestionResponse(BaseModel):
    status: str
    files_processed: int
    total_chunks: int
    processing_time_seconds: float
    files: List[FileInfo]

class FileInfo(BaseModel):
    filename: str
    chunks: int
    pages: int
```

---

## Design Decisions & Trade-offs

### 1. Chunking Strategy
**Decision**: Fixed-size chunking (512 tokens) with 50-token overlap, sentence-aware splitting

**Rationale**:
- Balances context preservation with retrieval precision
- Overlap ensures important information isn't lost at boundaries
- Sentence-awareness prevents breaking mid-thought

**Trade-offs**:
- May split related content across chunks
- Alternative: Semantic chunking (more complex, better quality)

### 2. Embedding Model
**Decision**: sentence-transformers/all-MiniLM-L6-v2

**Rationale**:
- Fast inference (CPU-friendly)
- Good quality for retrieval tasks
- Small model size (~80MB)
- 384-dimensional embeddings

**Trade-offs**:
- Lower quality than larger models (e.g., all-mpnet-base-v2)
- Alternative: Use Mistral embeddings API (more expensive, higher quality)

### 3. Vector Storage
**Decision**: Custom numpy-based storage with pickle serialization

**Rationale**:
- No external dependencies
- Simple implementation
- Fast for small-to-medium datasets (<100k chunks)
- Meets "no external vector DB" requirement

**Trade-offs**:
- Not scalable to millions of documents
- No distributed search
- Full index loaded into memory

### 4. Hybrid Search Weights
**Decision**: 70% semantic, 30% keyword (BM25)

**Rationale**:
- Semantic search better for conceptual matches
- Keyword search catches exact terms/names
- Empirically good balance

**Trade-offs**:
- May need tuning per domain
- Alternative: RRF (no weight tuning needed)

### 5. Similarity Threshold
**Decision**: 0.6 minimum similarity for "sufficient evidence"

**Rationale**:
- Prevents low-confidence answers
- Reduces hallucination risk
- Conservative but safe

**Trade-offs**:
- May reject valid answers
- Tune based on use case

---

## Success Criteria

### Functional Requirements
- ✅ Upload and process PDF files
- ✅ Extract and chunk text intelligently
- ✅ Search knowledge base with hybrid approach
- ✅ Generate accurate answers using Mistral AI
- ✅ Cite sources for answers
- ✅ Handle non-search queries (greetings)
- ✅ Refuse low-confidence answers

### Performance Requirements
- Ingestion: < 5 seconds per PDF (average document)
- Query: < 2 seconds end-to-end
- Search: < 200ms for semantic + keyword search
- Support: Up to 100 documents, 10k chunks

### Quality Requirements
- Retrieval accuracy: Relevant chunks in top-5 > 80% of time
- Answer quality: Grounded in sources, minimal hallucination
- Code quality: Clean, documented, follows PEP 8
- Test coverage: > 70% for core components

---

## Timeline Estimate

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1 | Project setup | 1-2 hours |
| Phase 2 | Ingestion pipeline | 3-4 hours |
| Phase 3 | Search implementation | 4-5 hours |
| Phase 4 | Query & LLM integration | 3-4 hours |
| Phase 5 | Bonus features | 2-3 hours |
| Phase 6 | UI development | 2-3 hours |
| Phase 7 | Testing | 2-3 hours |
| Phase 8 | Documentation | 2 hours |
| **TOTAL** | | **19-26 hours** |

---

## Future Enhancements (Out of Scope)

1. **Conversation Memory**: Track chat history for context
2. **Multi-modal Support**: Images, tables from PDFs
3. **Streaming Responses**: Real-time LLM output
4. **User Authentication**: Multi-user support
5. **Advanced Re-ranking**: Cross-encoder models
6. **Query Analytics**: Track popular queries
7. **Document Management**: Delete/update individual documents
8. **Semantic Caching**: Cache similar queries
9. **Feedback Loop**: User ratings to improve retrieval
10. **Production Deployment**: Docker, cloud hosting

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mistral API key invalid | High | Test early, provide fallback instructions |
| Poor retrieval quality | High | Implement hybrid search, tunable thresholds |
| Large PDF processing slow | Medium | Add async processing, progress indicators |
| Memory issues with large docs | Medium | Implement chunked loading, pagination |
| UI/UX confusion | Low | Clear instructions, example queries |

---

## Conclusion

This PRD provides a comprehensive roadmap for building a production-quality RAG system from scratch. The phased approach ensures steady progress while maintaining code quality and system reliability.

**Next Steps**:
1. Review and approve this plan
2. Set up development environment
3. Begin Phase 1 implementation
4. Regular check-ins after each phase

**Questions? Ready to start implementation!**

