# Fashion Retrieval System

> Multimodal fashion image retrieval using Vision-Language Models, FashionCLIP, structured metadata, and Qdrant vector search.

## Architecture

```
Offline Indexing Pipeline
  Image Dataset → Preprocessing → Qwen2.5-VL / Florence-2
                        ├─ Natural Language Caption → BGE Text Encoder → caption_embedding
                        ├─ Caption → Qwen/Qwen2.5-1.5B-Instruct → structured metadata
                        └─ FashionCLIP Image Encoder → fashionclip_embedding
          ↓
  Qdrant Collection (named vectors + structured payload)

Online Retrieval Pipeline
  User Query → Preprocessing →
      ├─ FashionCLIP Text Encoder → query embedding
      ├─ BGE Text Encoder → caption query embedding
                        └─ Qwen/Qwen2.5-1.5B-Instruct → structured query metadata
          ↓
  Qdrant Hybrid Retrieval (named-vector search + payload filtering)
          ↓
  Candidate Image IDs → Cross-Encoder Reranking → Qdrant Lookup
          ↓
  Final Ranked Results
```

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Copy and fill in environment variables
# PowerShell
Copy-Item .env.example .env
# or on macOS / Linux
# cp .env.example .env

# 3. Build the index (uses embedded local Qdrant at .qdrant by default)
# Drop your dataset images into data/images/ first (nested subfolders are fine)
uv run build-index --image-dir data/images --skip-existing

# 4. Search
uv run search --query "blue shirt in a park"
```

Optional: use Qdrant server mode instead of embedded local mode by setting
`QDRANT_LOCAL_PATH=` in `.env` or `qdrant.local_path: null` in
`configs/config.yaml`, then configuring `QDRANT_HOST` / `QDRANT_API_KEY`
as needed.

Common environment variables:

- `QDRANT_LOCAL_PATH` to disable embedded local storage and use server mode
- `QDRANT_HOST` and `QDRANT_API_KEY` for Qdrant server / cloud connections
- `HF_TOKEN` for Hugging Face model downloads

## Project Structure

```
fashion-retrieval-system/
├── configs/
│   ├── config.yaml          # Qdrant, retrieval params, batch sizes
│   └── models.yaml          # All model names/paths
├── src/
│   ├── embeddings/          # FashionCLIP and BGE encoders
│   ├── indexing/            # Image loading and offline indexing orchestration
│   ├── retrieval/           # Query parsing, hybrid retrieval, reranking
│   ├── utils/               # Config, logging, helpers
│   ├── vlm/                 # Captioning and structured metadata extraction
│   ├── qdrant_store.py      # Qdrant client wrapper
│   └── schemas.py           # Canonical Pydantic schema
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
| Semantic Reranker | `BAAI/bge-reranker-v2-m3` |

## GPU / CUDA

CUDA 12.4 wheels are pre-configured in `pyproject.toml` — just run:

```bash
uv sync
```

This installs `torch+cu124` automatically. No extra flags needed.

## Resumable Indexing

The indexer writes JSONL checkpoints to `data/.index_staging/` after each
image. If a run is interrupted, simply re-run the same command and it will
pick up where it left off — already-captioned images are skipped.

The `--skip-existing` flag makes reruns faster when the collection already
contains most of the dataset.

To force a clean re-index from scratch:

```bash
Remove-Item -Recurse -Force data/.index_staging   # Windows (PowerShell)
# or: rm -rf data/.index_staging                  # Linux / macOS
uv run build-index --image-dir data/images
```
