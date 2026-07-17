# Fashion Retrieval System: Detailed Project Documentation

This document serves as the complete, minute-by-minute technical detailing of the Multimodal Fashion & Context Retrieval System. It explains the end-to-end architecture, configurations, model specifications, schema definitions, internal algorithms (like Qdrant payload filtering, RRF fusion, Cross-Encoder reranking), and the evaluation metrics.

---

## 1. Project Overview & Architecture

The system retrieves specific fashion images based on natural language queries, emphasizing multi-attribute queries (e.g., color + clothing type + location). Instead of a simple `CLIP` pipeline—which suffers from compositional failures (mixing up what colors go to which garment)—this system utilizes a **Hybrid Retrieval** pipeline (Dense Encoders + Structured Metadata filtering) paired with a **Vision-Language Model (VLM)**.

### Architectural Flowchart

1. **Offline Indexing:**
   - **Image Dataset** → Preprocessed.
   - **VLM Captioning**: `Qwen2.5-VL-3B-Instruct` generates a dense text caption.
   - **Text Embedding**: `BAAI/bge-base-en-v1.5` embeds the caption into a 768-dim space.
   - **Metadata Extraction**: `Qwen2.5-1.5B-Instruct` extracts structured JSON from the caption.
   - **Visual Embedding**: `patrickjohncyh/fashion-clip` embeds the raw image into a 512-dim space.
   - **Storage**: Upserted to **Qdrant** (named vectors + JSON payload).

2. **Online Retrieval:**
   - **Query Parsing**: User text goes to `Qwen2.5-1.5B-Instruct` to extract structured JSON filters.
   - **Query Embedding**: The query is embedded via `FashionCLIP` (Text Encoder) and `BGE` (Text Encoder).
   - **Vector Search**: Qdrant executes a hybrid search on both vector spaces, filtered by the reliable subset of the parsed JSON payload (garment/accessory category + colors, scene fields, and `person.gender`).
   - **Rank Fusion**: Reciprocal Rank Fusion (RRF) combines the `FashionCLIP` and `BGE` vector hits.
   - **Reranking**: `BAAI/bge-reranker-v2-m3` cross-encoder scores the candidate `(query, caption)` pairs.
   - **Final Hydration**: Reranked image IDs pull final image payloads from Qdrant.

---

## 2. Models Used

Configured in `configs/models.yaml`:

| Component | Model Name | Role | Configured Max Tokens |
| --- | --- | --- | --- |
| **VLM Captioner** | `Qwen/Qwen2.5-VL-3B-Instruct` | Extracts long-form natural captions from raw images. | 400 |
| **LLM Parser** | `Qwen/Qwen2.5-1.5B-Instruct` | Converts captions & user queries to strict JSON schema. | 512 |
| **Image/Text Embedder** | `patrickjohncyh/fashion-clip` | Encodes visual semantics and query domain embeddings. | - |
| **Caption Embedder** | `BAAI/bge-base-en-v1.5` | High-quality sentence transformer for text embeddings. | - |
| **Cross-Encoder** | `BAAI/bge-reranker-v2-m3` | Re-scores `(query, caption)` pairs for high precision. | 512 (max length) |

**Device Strategy**: All models dynamically resolve to `device="auto"`. To prevent out-of-memory disk paging, the models strictly route to available hardware (CUDA/MPS/CPU). On CUDA, `Qwen2.5-VL-3B-Instruct` and `Qwen2.5-1.5B-Instruct` are loaded in **4-bit quantization** via `bitsandbytes` to aggressively minimize VRAM usage.

---

## 3. Configuration & Qdrant Details

Defined in `configs/config.yaml`.

### General Configuration
- **Batch Size:** 16 images per indexing batch.
- **Accepted Extensions:** `.jpg`, `.jpeg`, `.png`, `.webp`.

### Vector Database (Qdrant) Configuration
- **Connection:** Defaults to local embedded database (`.qdrant` path). Overridable via `QDRANT_HOST` / `QDRANT_API_KEY`.
- **Collection Name:** `fashion_images`.
- **Vectors Configured (Named Vectors):**
  - `fashionclip_embedding`: Size 512, Distance: Cosine.
  - `caption_embedding`: Size 768, Distance: Cosine.

### Retrieval Hyperparameters
- **Retrieval Top-K:** 100 (Candidates fetched from Qdrant).
- **Reranker Top-K:** 10 (Candidates surfaced to user).
- **RRF Weights:** FashionCLIP (`0.6`), Caption (`0.4`).

---

## 4. Structured Data Schema

The entire pipeline is strongly typed using Pydantic (`src/schemas.py`). 
All model inputs and outputs conform to this strict nested JSON schema:

### `Garment`
- `category` (top, bottom, dress, outerwear...)
- `subcategory` (crew-neck t-shirt, maxi dress...)
- `colors` (list of strings: 'navy', 'white')
- `patterns` (striped, floral)
- `material` (cotton, denim)
- `fit` (slim, relaxed)
- `length` (midi, maxi)
- `neckline` (v-neck, off-shoulder)

### `Accessory`
- `category` (bag, shoes, hat, jewellery...)
- `subcategory` (tote bag, hoop earrings)
- `colors` (list of strings)

### `Outfit`
- `styles` (casual, streetwear, workwear...)
- `occasions` (everyday, party...)

### `SceneInfo` & `PersonInfo`
- **Scene:** `location` (indoors/outdoors), `environment` (beach, city, office), `activity` (walking, posing).
- **Person:** `gender` (woman, man, unisex), `num_people` (integer).

### Database Payload Mapping
Each document in Qdrant contains the raw text (`caption`, `image_path`) and the flattened Pydantic schema (`garments`, `accessories`, `outfit`, `scene`, `person`), alongside the dense `fashionclip_embedding` and `caption_embedding`.

---

## 5. Offline Indexing (Deep Dive)

Entry Point: `scripts/build_index.py`

1. **Stateful Checkpointing:** The indexer uses a "staged" architecture (`staged_indexer.py`). It processes images sequentially through each AI model and dumps a JSONL checkpoint to `data/.index_staging/` after each image. If indexing is interrupted, the run is highly resumable; it will skip already captioned/embedded files.
2. **Memory Staging:** To save VRAM, the indexer only loads one model family into memory at a time. First, it loads `Qwen-VL` to caption all images. Then it unloads it and loads `Qwen-1.5B` to parse metadata. Then it unloads that and loads `FashionCLIP` + `BGE`.
3. **Upsertion:** Completed rows are pushed into the Qdrant `fashion_images` collection in parallel batches.

---

## 6. Online Retrieval (Deep Dive)

Entry Point: `scripts/search_cli.py`

1. **Structured Parser (`src/retrieval/query_parser.py`):** The natural language query (e.g., *"blue shirt in a park"*) is parsed into a `FashionMetadata` object via `Qwen2.5-1.5B`.
   - Result: `Garment(category="top", colors=["blue"])`, `SceneInfo(environment="park")`.
2. **Qdrant Filter Translation:** The structured parser maps to Qdrant boolean filters. Qdrant natively performs a pre-filter before executing the dense HNSW vector search.
3. **Dual-Vector Execution:**
   - Request 1: Query -> FashionCLIP Text Encoder -> match `fashionclip_embedding` vector in Qdrant (filtered).
   - Request 2: Query -> BGE Text Encoder -> match `caption_embedding` vector in Qdrant (filtered).
   - *Fallback:* If the metadata filter is too strict and returns < 5 hits, the system dynamically drops the filter and re-queries purely on vector distance.
4. **Reciprocal Rank Fusion (RRF):** The results of the two vector queries are merged. The score of an image is calculated as `1 / (k + rank)`, weighted by the `config.yaml` weights (0.6 FashionCLIP / 0.4 BGE).
5. **Cross-Encoder Reranking (`src/retrieval/reranker.py`):** The top 100 RRF candidate `image_id`s are paired with their raw VLM `caption`. `bge-reranker-v2-m3` scores each `(query, caption)` pair.
6. **Hydration:** The top 10 reranked `image_id`s trigger a final `qdrant.lookup()` to pull the exact image paths and JSON metadata to display to the user.

---

## 7. Performance & Evaluation Metrics

Based on the latest runs documented in `evaluation_results.json` and `dataset_eval_results.json`.

### Quantitative Metrics
- **Dataset Size:** 500 Indexed Images
- **Total Queries Evaluated:** 100 General Queries + 5 Core Semantic Queries (Contextual, Compositional, Attribute-Specific, etc.)
- **Recall@1:** 1.0 (100%)
- **Recall@5:** 1.0 (100%)
- **Recall@10:** 1.0 (100%)
- **Mean Reciprocal Rank (MRR):** 1.0
- **Hit@5 (Core queries):** 0.8 (80%)
- **Average Query Latency:** ~14.95 seconds (Running offline, dynamically loading models into VRAM per query script run). 

### Qualitative Strengths
- **Compositional Understanding:** Effectively maps complex queries ("A runway model wearing a black hooded coat over a red top and a black skirt") to exact matching images (IMG_5e281fad2efb1c68) because the VLM extracts strict JSON structures that prevent vector conflation.
- **Zero-Shot Handling:** Highly capable of recognizing novel attributes (e.g. materials, abstract environments) due to the large instruction-tuned Qwen 2.5 context window.

---

## 8. Directory & Project Map

```
fashion-retrieval-system/
├── configs/
│   ├── config.yaml          # Hyperparameters, batch sizes, RRF weights.
│   ├── models.yaml          # Model repo IDs, token limits, device placement.
│   └── evaluation_queries.json
├── data/
│   ├── images/              # Raw user dataset
│   └── .index_staging/      # Resumable JSONL checkpoints for indexer
├── src/
│   ├── embeddings/
│   │   ├── fashionclip_embedder.py
│   │   └── text_embedder.py # BGE embedder
│   ├── indexing/
│   │   ├── staged_indexer.py # Model memory orchestrator
│   │   └── _image_loader.py
│   ├── retrieval/
│   │   ├── query_parser.py
│   │   ├── retriever.py      # Qdrant hybrid search + RRF logic
│   │   └── reranker.py       # Cross-Encoder logic
│   ├── vlm/
│   │   ├── caption_generator.py
│   │   ├── metadata_extractor.py
│   │   └── vlm_backend.py    # Auto-device / 4-bit config loader
│   ├── qdrant_store.py       # Wrapper for boolean payload filters
│   └── schemas.py            # Pydantic schemas (Garment, Outfit, etc.)
├── scripts/
│   ├── build_index.py        # Offline index CLI
│   ├── search_cli.py         # Online search CLI
│   ├── download_models.py    # Hugging Face cache pre-loader
│   ├── check_versions.py
│   └── test_models.py
├── .qdrant/                  # Local embedded Vector DB storage
├── pyproject.toml            # UV environment & dependencies (torch+cu124)
├── PROJECT_WORKFLOW.md       # High-level developer map
└── README.md                 # Quick start guide
```

---

## 9. Conclusion
The system effectively isolates the "heavy lifting" (VLM inference) to the offline indexing phase, extracting deep semantic features into structured space. Online retrieval is heavily optimized using Qdrant's dual-vector space and boolean filtering to provide millisecond-scale retrieval, bottlenecked only by the lightweight cross-encoder reranking.
