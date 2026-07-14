# Fashion Retrieval System

> Multimodal fashion image retrieval using Vision-Language Models, FashionCLIP, structured metadata, and Qdrant vector search.

## Architecture

```
User Query
  ├─ FashionCLIP Embedding      → visual semantic similarity
  ├─ BGE Embedding              → caption semantic similarity
  └─ Structured Parser (Qwen)  → per-attribute metadata filter
          ↓
  Qdrant Hybrid Search (named vectors + metadata filter)
          ↓
  Cross-Encoder Reranking
          ↓
  Ranked Results
```

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Copy and fill in environment variables
cp .env.example .env

# 3. Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# 4. Build the index
uv run build-index --data-dir data/images

# 5. Search
uv run search --query "blue shirt in a park"
```

## Project Structure

```
fashion-retrieval-system/
├── configs/
│   ├── config.yaml          # Qdrant, retrieval params, batch sizes
│   └── models.yaml          # All model names/paths
├── src/
│   ├── ingestion/           # Dataset loading + image preprocessing
│   ├── indexing/            # Caption, metadata, embedding generation
│   ├── services/            # Model wrappers (FashionCLIP, BGE, Qwen, CrossEncoder)
│   ├── retrieval/           # Hybrid retrieval + reranking
│   ├── database/            # Qdrant client
│   ├── engine/              # Pipeline orchestration
│   ├── schemas/             # Canonical Pydantic metadata schema
│   └── utils/               # Config, logging, helpers
├── scripts/
│   ├── build_index.py       # Offline indexing entry point
│   └── search_cli.py        # Online retrieval entry point
└── tests/
```

## Models

| Role | Model |
|---|---|
| Vision-Language (captions + metadata) | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Structured Query Parser | `Qwen/Qwen2.5-1.5B-Instruct` |
| Image + Text Embedding | `patrickjohncyh/fashion-clip` |
| Caption Embedding | `BAAI/bge-base-en-v1.5` |
| Semantic Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

## GPU / CUDA

```bash
# Install PyTorch with CUDA 12.1 wheels
uv sync --extra-index-url https://download.pytorch.org/whl/cu121
```
