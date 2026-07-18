# How to Run the Fashion Retrieval System

This guide covers the entire end-to-end process for setting up the environment, indexing your dataset, searching, and running evaluations.

---

## 1. Prerequisites & Setup

This project uses `uv` for lightning-fast dependency management.

1. **Install Dependencies:**
   ```bash
   uv sync
   ```
2. **Activate the Virtual Environment:**
   ```bash
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```
3. **Environment Variables:**
   Copy the example environment file and fill in your details (especially if you need a HuggingFace token).
   ```bash
   # Windows (PowerShell)
   Copy-Item .env.example .env
   # macOS/Linux
   cp .env.example .env
   ```

---

## 2. (Optional) Download Models

You can pre-download all the HuggingFace models (FashionCLIP, BGE Text Embedder, Qwen Query Parser, Cross-Encoders) locally before you run anything. This prevents slow downloads from interrupting your indexing or searching later.

```bash
uv run python scripts/download_models.py
```
*(Tip: Add `--skip-vl` if you want to skip downloading the large Vision-Language model if you're only testing text search).*

---

## 3. Indexing the Dataset

Before you can search, the system needs to process your images, generate captions, extract structured metadata, and compute embeddings.

1. Place all your fashion images inside the `data/images/` folder (subfolders are fine).
2. Run the indexer:
   ```bash
   uv run build-index --image-dir data/images
   ```
   *This command will process all images, store their vectors and metadata locally in an embedded Qdrant database (at `.qdrant/`), and write a backup of the captions to `data/.index_staging/`.*

---

## 4. Searching for Images

Use the Search CLI to query your indexed dataset.

**Interactive Mode (Recommended)**
Launch the search REPL to continuously search without reloading models:
```bash
uv run search
```
To automatically open the result images on your computer as you search:
```bash
uv run search --open-images
```

**Single Query Mode**
If you want to run one search command directly from the terminal:
```bash
uv run search --query "a model in a red dress"
```
To automatically open the result images from a single query:
```bash
uv run search --query "a model in a red dress" --open-images
```

---

## 5. Evaluating the System

The system includes a unified evaluation script to measure how accurately it finds matching images.

**A. Evaluate a specific Golden Dataset**
Evaluates rigorous metrics (Hit@K, MRR) against a curated JSON list of grounded queries.
```bash
uv run evaluate golden-set --dataset configs/golden_dataset.json
```

**B. Self-Retrieval Recall Test**
Takes all the captions the system generated during indexing, feeds them back in as search queries, and checks if it can find the original image.
```bash
uv run evaluate self-retrieval --num-samples 100
```

**C. Inspect Queries Visually**
Runs queries and prints a detailed breakdown of the top results, optionally copying the images to an output folder so you can look at them side-by-side.
```bash
uv run evaluate inspect-queries --query-file configs/evaluation_queries.json --copy-images
```
