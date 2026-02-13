# System Architecture

Technical overview of the Enterprise RAG System architecture.

## Overview

The Enterprise RAG System is built on a modular architecture with clear separation of concerns:

```
┌─────────────┐
│   User UI   │ (Streamlit / CLI)
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────┐
│           RAG Chain (rag.py)                 │
│  ┌────────────┐  ┌────────┐  ┌───────────┐ │
│  │  Retriever │→ │ Prompt │→ │ Groq LLM  │ │
│  └────────────┘  └────────┘  └───────────┘ │
└──────┬───────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│      Retrieval Engine (retrieval.py)         │
│  ┌──────────────────────────────────────┐   │
│  │   FAISS Vector Store (vectorizer.py) │   │
│  └──────────────────────────────────────┘   │
└──────┬───────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│    Ingestion Pipeline (ingestion.py)         │
│  ┌──────┐  ┌─────────┐  ┌──────────────┐   │
│  │ Load │→ │ Clean   │→ │ Chunk        │   │
│  └──────┘  └─────────┘  └──────────────┘   │
└──────┬───────────────────────────────────────┘
       │
┌──────▼──────┐
│  Documents  │ (Markdown files)
└─────────────┘
```

## Core Components

### 1. Ingestion Pipeline (`src/ingestion.py`)

**Purpose:** Load, clean, and chunk documents

**Components:**
- `DocumentLoader`: Loads markdown files
- `TextCleaner`: Sanitizes text
- `TextSplitter`: Splits into chunks

**Flow:**
```python
Raw File → Load → Clean → Split → Chunks
```

**Key Features:**
- Preserves metadata (source, title)
- Configurable chunk size and overlap
- Handles multiple file formats

### 2. Vector Store (`src/vectorizer.py`)

**Purpose:** Convert text to embeddings and store in FAISS

**Components:**
- `EmbeddingModel`: HuggingFace wrapper
- `VectorStoreManager`: FAISS index manager

**Flow:**
```python
Chunks → Embed → Index → FAISS Store
```

**Key Features:**
- Local embeddings (no API calls)
- Fast similarity search
- Metadata preservation

### 3. Retrieval Engine (`src/retrieval.py`)

**Purpose:** Find relevant chunks for queries

**Components:**
- `Retriever`: Semantic search wrapper

**Flow:**
```python
Query → Embed → Search → Top-k Chunks
```

**Key Features:**
- Configurable k (number of results)
- Similarity scoring
- Logging support

### 4. RAG Chain (`src/rag.py`)

**Purpose:** Generate answers using retrieved context

**Components:**
- `RAGChain`: Orchestrates retrieval + generation

**Flow:**
```python
Query → Retrieve → Format Prompt → LLM → Answer
```

**Key Features:**
- Strict grounding
- Refusal logic
- Source attribution

### 5. Prompts (`src/prompts.py`)

**Purpose:** Define system behavior

**Components:**
- `RAG_SYSTEM_PROMPT`: System instructions
- `get_rag_prompt_template()`: Prompt builder

**Key Features:**
- Enforces grounding
- Defines refusal behavior
- Configurable

## Data Flow

### Indexing Flow (Startup)

```
1. Load Earthquake Events
   ├─ Read CSV or TXT files from /data
   ├─ Support for multiple encodings (UTF‑8, UTF‑8‑BOM, CP1252, ISO‑8859‑1)
   ├─ Parse INGV earthquake fields (EventID, Time, Magnitude, Location, etc.)
   └─ Convert each event into a LangChain Document with metadata

2. Clean Text
   ├─ Normalize whitespace
   ├─ Collapse repeated newlines
   ├─ Remove artifacts and malformed characters
   └─ Standardize fields for consistent RAG processing

3. Split into Chunks
   ├─ Chunk size: 400–500 characters
   ├─ Overlap: 40–50 characters
   ├─ Maintain event_id metadata
   └─ Ensure geophysical information stays in coherent pieces

4. Generate Embeddings
   ├─ Model: sentence-transformers/all-MiniLM-L6-v2
   ├─ Embedding dimension: 384
   ├─ Local inference (no API required)
   └─ Produces vector representations of each earthquake chunk

5. Build FAISS Index
   ├─ Index type: Flat L2 (exact vector similarity)
   ├─ Store embeddings + event metadata
   ├─ Support incremental updates (add_documents)
   └─ Ready for fast similarity search over earthquake events
```

### Query Flow (Runtime)

```
1. User Query
   ├─ The user asks something about earthquakes:
   │    • "Quali terremoti hanno magnitudo maggiore di 6?"
   │    • "Ci sono eventi profondi nel Tirreno Meridionale?"
   │    • "Terremoti recenti vicino alla Sicilia o Calabria?"
   └─ Query text is passed to the RAG pipeline

2. Query Embedding
   ├─ The natural-language query is encoded into a vector
   ├─ Embedding model: sentence-transformers/all-MiniLM-L6-v2
   ├─ Vector dimension: 384
   └─ This ensures the query uses the same vector space as the earthquake chunks

3. Vector Search (FAISS Similarity Search)
   ├─ FAISS computes similarity between:
   │    • query vector
   │    • all earthquake event vectors
   ├─ Index type: Flat L2 (exact search)
   ├─ Top‑k results (default k = 8)
   ├─ For each match, FAISS returns:
   │    • similarity score
   │    • associated document chunk
   │    • metadata (event_id, location, magnitude, time, etc.)
   └─ These chunks represent the most relevant seismic events

4. Context Construction & Prompt Formatting
   ├─ SYSTEM section:
   │      • RAG rules
   │      • Use ONLY the provided context
   │      • If info is missing → respond: "Non lo so in base ai documenti forniti."
   ├─ CONTEXT section:
   │      • Earthquake chunks retrieved by FAISS
   │      • Each chunk contains:
   │           EventID
   │           Date/Time
   │           Magnitude + Type
   │           Depth
   │           Latitude / Longitude
   │           Locality
   └─ USER section:
          The original question

5. LLM Response Generation
   ├─ Model: Groq Llama 3.3 70B
   ├─ Temperature: 0 → deterministic and reproducible
   ├─ The model reads ONLY:
   │      • user question
   │      • retrieved context
   └─ Produces a concise, grounded answer

6. Final Output
   ├─ Answer text (grounded in INGV earthquake data)
   ├─ List of source documents used (EventID)
   └─ Optional logs:
          • chunk score
          • snippet preview
          • metadata
```

## Technology Stack

### Core Framework
- **LangChain**: RAG orchestration
- **Python**: 3.9+

### LLM & Embeddings
- **Groq**: LLM API (Llama 3.3 70B)
- **HuggingFace**: Embeddings (sentence-transformers)

### Vector Database
- **FAISS**: Similarity search

### UI
- **Streamlit**: Web interface

### Testing
- **Pytest**: Unit & integration tests

## Design Decisions

### Why Local Embeddings?

**Pros:**
- No API calls = faster
- No cost for embeddings
- Privacy (data stays local)

**Cons:**
- Requires RAM (~2GB)
- Initial download (~90MB)

**Decision:** Benefits outweigh costs for most users

### Why FAISS?

**Pros:**
- Fast similarity search
- Works locally
- No external dependencies

**Cons:**
- In-memory only
- No persistence (rebuilds on restart)

**Decision:** Speed and simplicity > persistence

### Why Groq?

**Pros:**
- Very fast inference
- Free tier available
- Good model quality

**Cons:**
- Requires API key
- Rate limits

**Decision:** Speed and quality justify API dependency

### Why Streamlit?

**Pros:**
- Easy to build UI
- Python-native
- Good for demos

**Cons:**
- Not production-grade for high traffic
- Limited customization

**Decision:** Perfect for MVP and demos

## Performance Characteristics

### Indexing Performance

| Earthquake Events | Chunks Generated | Index Build Time | Memory Usage |
|-------------------|------------------|------------------|--------------|
| 5 events          | ~120 chunks      | ~8–10 seconds    | ~450–500 MB  |
| 10 events         | ~250 chunks      | ~15–20 seconds   | ~900 MB–1 GB |
| 50 events         | ~1200 chunks     | ~90–120 seconds  | ~2–3 GB      |

### Query Performance

| Component | Time       | Notes |
|-----------|------------|-------|
| Embedding | ~40–60 ms  | Local |
| Retrieval | ~8–15 ms   | FAISS |
| LLM       | ~1.8–2.5 s | llama‑3.3‑70B‑versatile |
| **Total** | **~2-3s** | From question → final answer |

### Scalability

**Current Limits:**
   - Up to ~100 INGV event files
   - Up to ~10,000 chunks
   - Up to ~4 GB RAM

**Future Improvements:**
   - daily or hourly ingestion of events
   - long-term accumulation (FAISS append mode)
   - interactive expert queries

**Future Scalability Improvements**


**Persistent Vector Store**

   - FAISS write/read from disk
   - Or migration to ChromaDB / Pinecone



**Batch Indexing**

   - Parallel ingestion of INGV datasets



**Distributed Search**

   - Multi-node FAISS
   - Sharded RAG pipelines for large catalogs

## Extension Points

**Security Considerations**

  - Keys stored in .env
  - .env added to .gitignore
  - Keys never logged, printed, or exposed in the UI
  - Only the query is sent to Groq (not the whole dataset)

**Data Privacy**

   - All INGV event documents stay local
   - Embeddings are computed locally
   - Only the user question (no document content) is sent to Groq

**Input Validation**

   - Earthquake files validated for field structure
   - Reject invalid file types (only CSV/TXT INGV formats)
   - Size limits prevent ingestion abuse
   - Sanitization removes escape sequences, malformed characters


### Custom Embeddings

Edit `src/vectorizer.py`:
```python
class EmbeddingModel:
    def __init__(self, model_name: str = "your-model"):
        # Use different model
        pass
```

### Different Vector Store

Edit `src/vectorizer.py`:
```python
from langchain_chroma import Chroma

class VectorStoreManager:
    def create_index(self, documents):
        self.vector_store = Chroma.from_documents(...)
```

### Custom LLM

Edit `src/rag.py`:
```python
from langchain_openai import ChatOpenAI

class RAGChain:
    def __init__(self, retriever):
        self.llm = ChatOpenAI(model="gpt-4")
```

## Future Enhancements

### Planned Features

Future Enhancements
1. Persistent FAISS Index

Save/load index from disk
Support incremental updates (new earthquake events)

2. Advanced Retrieval

Hybrid (semantic + keyword search)
Cross-encoder re-ranking
Semantic clustering of events

3. Multi‑Modal Expansion

Earthquake PDFs (maps, reports)
Satellite images + seismic heatmaps
Automatic extraction from tables and bulletins

4. Production Hardening

API endpoints
Authentication
Rate limiting
Observability & monitoring dashboards