# JKR FMI — Facility Management Intelligence

AI-powered chatbot for **Jabatan Kerja Raya (JKR) Malaysia**, specializing in the *Technical Specification for Building Facility Management & Maintenance* (CPAB.BPPA.FMM.TS(01).2025).

Built by [Pendakwah Teknologi](https://pendakwah.tech). Runs on NVIDIA GX10 Grace Blackwell.

---

## What It Does

JKR FMI answers technical questions about government building facility management — maintenance procedures, contractor obligations, HSE requirements, utility management, and more — grounded in the official JKR specification document.

Users can ask in Bahasa Melayu or English. The system retrieves relevant document sections, searches the web for current information when needed, and generates answers with precise section/page references.

---

## Architecture

```
User Query
    |
    v
[1] Query Classification (internal / external / hybrid)
    |
    v
[2] Query Expansion (LLM generates 3 search variants)
    |
    v
[3] Hybrid Retrieval (Vector + BM25 + Reciprocal Rank Fusion)
    |               \
    v                v
[4] Web Search      [5] Cross-Encoder Reranking
    (Tavily/Brave)       (GPU, top 7)
    |               /
    v              v
[6] LLM Generation (Qwen3.5-122B, streaming SSE)
    |
    v
[7] Self-Evaluation (3-dimension quality scoring)
    |
    v
[8] Follow-up Suggestions (3 contextual next questions)
```

### Components

| Component | Technology | Details |
|-----------|-----------|---------|
| **Main LLM** | Qwen3.5-122B | Via MTAI API, 4-key round-robin, streaming |
| **Fast LLM** | Qwen3.5-27B | Query expansion, self-eval, follow-ups |
| **Embeddings** | Mesolitica Mistral 191M | Malay-optimized, 1024-dim, GPU-accelerated |
| **Reranker** | ms-marco-MiniLM-L-6-v2 | Cross-encoder on GPU |
| **Vector DB** | ChromaDB | HNSW index (M=32, ef=200), persistent |
| **Web Search** | Tavily + Brave | Dual-provider fallback, JKR-focused |
| **STT** | Whisper large-v3 | Faster-Whisper, float16 on GPU |
| **TTS** | MMS-TTS Malay | Meta's multilingual speech, GPU |
| **Cache** | Redis | Shared across workers, 10min TTL |
| **Backend** | FastAPI + Uvicorn | 4 workers, uvloop, async |
| **Gateway** | Nginx | SSL, rate limiting, SSE proxy, gzip |

### Knowledge Base

- **Document**: JKR Technical Specification for Building FM&M (85 pages, 7 sections)
- **Chunks**: 515 semantic chunks (600 chars, 120 overlap)
- **Metadata**: Filename, chunk index, page number(s), document type
- **Sections**: A (General) through G (Technical Advisory)

---

## Project Structure

```
jkr-fmi/
  backend/
    app.py              # FastAPI application (endpoints, streaming, voice, admin)
    providers.py         # RAG pipeline (retrieval, reranking, generation, web search)
    agency_config.py     # System prompt, keywords, agency metadata
  frontend/
    index.html           # Chat interface (single-page app)
    architecture.html    # System architecture documentation page
    admin.html           # Log & audit trail dashboard
  configs/
    backend.env.template # Environment variables template (copy to backend.env)
    jkr-ai.service       # systemd service unit
    jkr-ai.conf          # Nginx site configuration
  knowledge/
    full_doc.pdf         # JKR FMM technical specification
  logo/
    jkr_logo.png         # JKR logo
  scripts/
    setup.sh             # Automated setup (venv, deps, models, systemd, nginx)
    ingest.sh            # Document ingestion into ChromaDB
  requirements.txt       # Python dependencies
```

---

## Setup

### Prerequisites

- Ubuntu 22.04+ (tested on 24.04 aarch64)
- Python 3.11+
- NVIDIA GPU with CUDA 12+ (optional, falls back to CPU)
- Redis server
- Nginx (for production deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/pendakwahteknologi/jkr-fmi.git
cd jkr-fmi

# Run automated setup
bash scripts/setup.sh
```

The setup script will:
1. Detect GPU and platform capabilities
2. Create `/opt/jkr-ai/` directory structure
3. Copy backend, frontend, and knowledge files
4. Create Python virtual environment with all dependencies
5. Install GPU-accelerated PyTorch (if GPU detected)
6. Pre-download embedding and reranker models
7. Install systemd service and nginx config

### Configuration

```bash
# Edit the environment file with your API keys
nano /opt/jkr-ai/backend.env
```

Required keys:

```env
# MTAI API (required — main LLM provider)
MTAI_API_KEYS=sk-your-key-1,sk-your-key-2

# Web Search (optional — enables internet search)
TAVILY_API_KEY=tvly-your-key
BRAVE_API_KEY=your-brave-key
```

### Document Ingestion

```bash
# Ingest documents into ChromaDB (GPU-accelerated)
bash scripts/ingest.sh
```

This clears ChromaDB, extracts text from all PDFs/DOCX/TXT/MD files in `/opt/jkr-ai/knowledge/` and `/opt/jkr-ai/documents/`, chunks them with page tracking, embeds on GPU, and stores in ChromaDB.

### Start

```bash
# Start the service
sudo systemctl start jkr-ai

# Enable auto-start on boot
sudo systemctl enable jkr-ai

# Check health
curl http://localhost:8003/api/health
```

---

## API Endpoints

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Synchronous chat (returns full response) |
| `POST` | `/api/chat/stream` | Streaming chat (SSE events: sources, chunks, done) |

**Request body:**
```json
{
  "messages": [
    {"role": "user", "content": "Apakah keperluan penyelenggaraan mekanikal?"}
  ]
}
```

**Response includes:** reply, retrieval sources (with page numbers), query type, self-evaluation scores, follow-up suggestions.

### Voice

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/voice/transcribe/local` | Speech-to-text (base64 audio) |
| `POST` | `/api/voice/synthesize/local` | Text-to-speech (returns WAV) |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health + document count |
| `GET` | `/api/mode` | Current mode, features, model info |
| `GET` | `/api/cache/stats` | Redis cache statistics |
| `POST` | `/api/cache/clear` | Clear response cache |

### Feedback

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/feedback` | Submit rating (1-5) and comment |
| `GET` | `/api/feedback/stats` | Rating distribution and average |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/admin/conversations` | Full conversation audit log |
| `GET` | `/api/admin/feedback` | All feedback with IP tracking |
| `GET` | `/api/admin/summary` | Aggregate stats (IPs, response times, cache hits) |

---

## Frontend Pages

| Page | URL | Description |
|------|-----|-------------|
| **Chat** | `/` | Main chat interface with voice input, markdown rendering, source references |
| **Architecture** | `/architecture.html` | Full system documentation with pipeline flow and component details |
| **Admin** | `/admin.html` | Audit trail dashboard — conversations, feedback, IP tracking, stats |

---

## How It Works

### Query Classification

Every query is classified as:
- **internal** — Matches JKR document keywords (55+ terms). RAG only.
- **external** — Matches current-affairs keywords. RAG + web search.
- **hybrid** — Ambiguous. RAG + web search.

### Retrieval Pipeline

1. **Query Expansion**: The fast LLM (27B) rewrites the question into 3 variants for broader recall
2. **Vector Search**: All 4 queries encoded with Mesolitica embeddings, searched against ChromaDB HNSW index (top 15 per variant, deduplicated)
3. **BM25 Search**: Parallel keyword search using BM25Okapi scoring
4. **RRF Fusion**: Vector and BM25 results combined using Reciprocal Rank Fusion (k=60)
5. **Cross-Encoder Reranking**: Top candidates scored by cross-encoder, filtered to top 7

### Web Search

For external/hybrid queries, the system searches the web using Tavily (primary) or Brave (fallback). Queries are automatically prefixed with JKR context. URLs from web results are extracted and provided to the LLM as an explicit allowlist — the LLM cannot fabricate URLs.

### Response Quality

- **Chain-of-Thought**: Internal reasoning structure (not shown to user)
- **Source Grounding**: Answers must cite document sections and page numbers
- **Self-Evaluation**: Each answer scored on relevance, accuracy, and completeness (1-5 scale)
- **URL Verification**: Only URLs from verified web sources are permitted in answers

### Audit Trail

Every conversation is logged with:
- Full query and response text
- Client IP (from X-Real-IP header)
- User-Agent string
- Query classification type
- Number of sources retrieved
- Response time in milliseconds
- Cache hit/miss status
- Timestamp

---

## Deployment Notes

### systemd Service

- 4 Uvicorn workers with uvloop
- Memory limit: 8GB (high watermark: 6GB)
- CPU quota: 60%
- Security hardening: NoNewPrivileges, ProtectSystem=strict, ProtectHome=read-only
- Logs to journald (`journalctl -u jkr-ai -f`)

### Nginx

- Rate limiting: 20 req/s per IP, burst 10
- SSE streaming: proxy_buffering off, 300s timeout
- Upstream keepalive: 32 connections, 1h lifetime
- Security headers: X-Frame-Options, X-Content-Type-Options, X-XSS-Protection
- Static asset caching: 30 days, immutable
- Blocks access to .env, .git, and common exploit paths

### Redis Cache

- Shared across all 4 workers
- Cache key includes conversation history hash (prevents cross-user collisions)
- TTL: 600 seconds (configurable via `CACHE_TTL_SECONDS`)
- In-memory fallback if Redis unavailable

---

## Adding Documents

Place PDF, DOCX, TXT, or MD files in `/opt/jkr-ai/knowledge/` or `/opt/jkr-ai/documents/`, then re-ingest:

```bash
bash scripts/ingest.sh
sudo systemctl restart jkr-ai
```

PDF files are extracted with page markers — each chunk in ChromaDB stores which page(s) it came from, enabling precise page references in answers.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MTAI_API_BASE_URL` | `https://api.mtai.com.my/v1` | LLM API base URL |
| `MTAI_API_KEYS` | *(required)* | Comma-separated API keys |
| `MTAI_MODEL` | `Qwen/Qwen3.5-122B` | Main LLM model |
| `EMBEDDING_DEVICE` | `cuda` | Embedding compute device |
| `CROSS_ENCODER_DEVICE` | `cuda` | Reranker compute device |
| `RETRIEVAL_TOP_K` | `15` | Vector search results per query |
| `EMBEDDING_BATCH_SIZE` | `256` | GPU embedding batch size |
| `CACHE_TTL_SECONDS` | `600` | Redis cache TTL |
| `TAVILY_API_KEY` | *(optional)* | Tavily web search API key |
| `BRAVE_API_KEY` | *(optional)* | Brave Search API key |
| `CHROMA_PERSIST_DIR` | `/opt/jkr-ai/chroma_db` | ChromaDB storage path |

---

## License

Proprietary. Copyright Pendakwah Teknologi.
