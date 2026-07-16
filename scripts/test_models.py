import sys
from PIL import Image
from rich.console import Console
from rich import print as rprint
import torch

from src.utils.config_loader import get_model_settings
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.fashionclip_embedder import FashionCLIPEmbedder
from src.retrieval.reranker import Reranker
from src.retrieval.query_parser import QueryParser
from src.schemas import FashionMetadata, RetrievalResult

console = Console()

def test_models():
    console.rule("[bold cyan]Testing Downloaded Models[/bold cyan]")
    
    try:
        cfg = get_model_settings()
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        sys.exit(1)

    # 1. Text Embedder (BGE)
    console.print("\n[bold]1. Testing Text Embedder (BGE)[/bold]")
    try:
        text_emb = TextEmbedder(cfg.text_embedding)
        vec = text_emb.encode(["A red summer dress"])[0]
        console.print(f"[green][OK] Success! Output shape: {len(vec)}[/green]")
    except Exception as e:
        console.print(f"[red][FAIL] Failed: {e}[/red]")

    # 2. FashionCLIP Embedder
    console.print("\n[bold]2. Testing FashionCLIP Embedder[/bold]")
    try:
        clip_emb = FashionCLIPEmbedder(cfg.fashionclip)
        # Create a dummy image
        dummy_img = Image.new('RGB', (224, 224), color='red')
        img_vec = clip_emb.encode_images([dummy_img])[0]
        text_vec = clip_emb.encode_texts(["A red summer dress"])[0]
        console.print(f"[green][OK] Success! Image vector shape: {len(img_vec)}, Text vector shape: {len(text_vec)}[/green]")
    except Exception as e:
        console.print(f"[red][FAIL] Failed: {e}[/red]")

    # 3. Cross-Encoder Reranker
    console.print("\n[bold]3. Testing Cross-Encoder Reranker[/bold]")
    try:
        from src.utils.config_loader import get_app_settings
        settings = get_app_settings()
        reranker = Reranker(cfg.cross_encoder, settings)
        candidates = [
            RetrievalResult(
                image_id="IMG_TEST_1",
                image_path="/tmp/test1.jpg",
                caption="A beautiful red evening gown.",
                metadata=FashionMetadata(),
            ),
            RetrievalResult(
                image_id="IMG_TEST_2",
                image_path="/tmp/test2.jpg",
                caption="A casual blue denim jacket.",
                metadata=FashionMetadata(),
            ),
        ]
        ranked = reranker.rerank("red dress", candidates)
        scores = [r.score for r in ranked]
        console.print(f"[green][OK] Success! Scores: {scores}[/green]")
    except Exception as e:
        console.print(f"[red][FAIL] Failed: {e}[/red]")

    # 4. Qwen2.5-1.5B Query Parser
    console.print("\n[bold]4. Testing Query Parser (Qwen2.5-1.5B)[/bold]")
    try:
        qp = QueryParser(cfg.query_parser)
        meta = qp.parse("I'm looking for a casual blue denim jacket for winter.")
        console.print(f"[green][OK] Success! Extracted Metadata:[/green]")
        rprint(meta.model_dump())
    except Exception as e:
        console.print(f"[red][FAIL] Failed: {e}[/red]")

    console.rule("[bold cyan]Tests Completed[/bold cyan]")

if __name__ == "__main__":
    test_models()
