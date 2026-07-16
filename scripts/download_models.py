"""
scripts/download_models.py
──────────────────────────
Pre-downloads all HuggingFace models used by the fashion retrieval system
into the local HF cache so components can be tested fully offline.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --skip-vl   # skip the large 3B VLM
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable

import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Load .env from project root (two levels up from this script)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# Apply HF_HOME early so the cache lands on the right drive
# before any huggingface_hub / transformers code runs.
_hf_home = os.getenv("HF_HOME", "").strip()
if _hf_home:
    os.environ["HF_HOME"] = _hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(_hf_home) / "hub")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

console = Console()


@dataclass
class ModelSpec:
    name: str          # display name
    model_id: str      # HuggingFace hub ID
    loader: str        # which loader to use
    size_gb: float     # approximate download size


MODELS: list[ModelSpec] = [
    ModelSpec(
        name="FashionCLIP",
        model_id="patrickjohncyh/fashion-clip",
        loader="clip_transformers",
        size_gb=0.6,
    ),
    ModelSpec(
        name="BGE Text Embedder",
        model_id="BAAI/bge-base-en-v1.5",
        loader="sentence_transformers",
        size_gb=0.4,
    ),
    ModelSpec(
        name="Cross-Encoder Reranker",
        model_id="BAAI/bge-reranker-v2-m3",
        loader="cross_encoder",
        size_gb=0.6,
    ),
    ModelSpec(
        name="Qwen2.5-1.5B Query Parser",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        loader="transformers_auto",
        size_gb=3.1,
    ),
    ModelSpec(
        name="Qwen2.5-VL-3B Vision-Language",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        loader="transformers_vl",
        size_gb=6.5,
    ),
]


# ── Loader helpers ────────────────────────────────────────────────────────────

def _load_clip_transformers(model_id: str) -> None:
    """Download FashionCLIP via HF transformers (CLIP checkpoint).

    Matches the loader actually used in src/embeddings/fashionclip_embedder.py —
    patrickjohncyh/fashion-clip is a standard transformers CLIP checkpoint, not
    an open_clip hub model.
    """
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    console.log(f"  Downloading tokenizer & weights for [cyan]{model_id}[/]")
    CLIPProcessor.from_pretrained(model_id)
    CLIPModel.from_pretrained(model_id)


def _load_sentence_transformers(model_id: str) -> None:
    from sentence_transformers import SentenceTransformer  # type: ignore
    console.log(f"  Downloading [cyan]{model_id}[/] via sentence-transformers")
    SentenceTransformer(model_id)


def _load_transformers_auto(model_id: str) -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    console.log(f"  Downloading tokenizer for [cyan]{model_id}[/]")
    AutoTokenizer.from_pretrained(model_id)
    console.log(f"  Downloading model weights for [cyan]{model_id}[/] (this may take a while...)")
    AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")


def _load_transformers_vl(model_id: str) -> None:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore
    console.log(f"  Downloading processor for [cyan]{model_id}[/]")
    AutoProcessor.from_pretrained(model_id)
    console.log(f"  Downloading model weights for [cyan]{model_id}[/] (~6.5 GB, this will take a while...)")
    Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")


def _load_cross_encoder(model_id: str) -> None:
    from sentence_transformers import CrossEncoder  # type: ignore
    console.log(f"  Downloading [cyan]{model_id}[/] via CrossEncoder")
    CrossEncoder(model_id, max_length=512)


LOADERS: dict[str, Callable[[str], None]] = {
    "clip_transformers": _load_clip_transformers,
    "sentence_transformers": _load_sentence_transformers,
    "cross_encoder": _load_cross_encoder,
    "transformers_auto": _load_transformers_auto,
    "transformers_vl": _load_transformers_vl,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download all models locally.")
    parser.add_argument(
        "--skip-vl",
        action="store_true",
        help="Skip the large Qwen2.5-VL-3B vision-language model (~6.5 GB)",
    )
    parser.add_argument(
        "--only",
        metavar="NAME",
        help="Download only the model whose name contains this substring (case-insensitive)",
    )
    args = parser.parse_args()

    # ── HuggingFace login ─────────────────────────────────────────────────────
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        try:
            from huggingface_hub import login  # type: ignore
            login(token=hf_token, add_to_git_credential=False)
            console.print("[green]HuggingFace login successful.[/green]\n")
        except Exception as exc:
            console.print(f"[yellow]Warning: HF login failed ({exc}). Proceeding without auth.[/yellow]\n")
    else:
        console.print("[yellow]Warning: HF_TOKEN not set in .env — proceeding without auth (may hit rate limits).[/yellow]\n")

    models_to_download = list(MODELS)
    if args.skip_vl:
        models_to_download = [m for m in models_to_download if m.loader != "transformers_vl"]
    if args.only:
        models_to_download = [
            m for m in models_to_download if args.only.lower() in m.name.lower()
        ]

    # Print plan
    table = Table(title="Models to download", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("HuggingFace ID")
    table.add_column("~Size", justify="right")
    for m in models_to_download:
        table.add_row(m.name, m.model_id, f"{m.size_gb} GB")
    console.print(table)

    total_gb = sum(m.size_gb for m in models_to_download)
    console.print(f"\n[yellow]Total estimated download: {total_gb:.1f} GB[/yellow]\n")

    results: list[tuple[str, str, str]] = []

    for spec in models_to_download:
        console.rule(f"[bold green]{spec.name}[/bold green]")
        start = time.time()
        try:
            LOADERS[spec.loader](spec.model_id)
            elapsed = time.time() - start
            results.append((spec.name, "OK", f"{elapsed:.0f}s"))
            console.print(f"  [green]Done in {elapsed:.0f}s[/green]\n")
        except Exception as exc:
            elapsed = time.time() - start
            results.append((spec.name, "FAILED", str(exc)[:80]))
            console.print(f"  [red]FAILED: {exc}[/red]\n")

    # Summary
    console.rule("[bold]Download Summary[/bold]")
    summary = Table(show_lines=True)
    summary.add_column("Model", style="bold")
    summary.add_column("Status")
    summary.add_column("Detail")
    for name, status, detail in results:
        color = "green" if status == "OK" else "red"
        summary.add_row(name, f"[{color}]{status}[/{color}]", detail)
    console.print(summary)

    failed = [r for r in results if r[1] == "FAILED"]
    if failed:
        console.print(f"\n[red]{len(failed)} model(s) failed to download.[/red]")
        sys.exit(1)
    else:
        console.print("\n[bold green]All models downloaded successfully![/bold green]")


if __name__ == "__main__":
    main()
