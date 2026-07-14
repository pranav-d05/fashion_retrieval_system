"""Smoke test: verify all core modules are importable without ML deps."""
import sys

errors = []

def check(label, fn):
    try:
        fn()
        print(f"PASS  {label}")
    except Exception as e:
        print(f"FAIL  {label}: {e}")
        errors.append(label)

def test_schemas():
    from src.schemas import (
        FashionMetadata, Garment, IndexedImage,
        RetrievalResult, SceneInfo, PersonInfo
    )
    m = FashionMetadata(
        garments=[Garment(category="dress", color="red")],
        scene=SceneInfo(environment="beach"),
        person=PersonInfo(gender="woman", num_people=1),
    )
    assert m.garments[0].category == "dress"
    assert m.scene.environment == "beach"

def test_utils():
    from src.utils.config_loader import get_app_settings, get_model_settings
    from src.utils.helpers import iter_images, chunk_list, generate_image_id, timer
    from src.utils.logging_config import get_logger, setup_logging
    s = get_app_settings()
    assert s.qdrant.host == "localhost"
    ms = get_model_settings()
    assert "Qwen" in ms.vision_language_model.model_name

def test_metadata_extractor_parsing():
    from src.vlm.metadata_extractor import _parse_metadata, _strip_code_fences
    raw = '{"garments": [{"category": "dress", "color": "red"}], "scene": {}, "person": {}}'
    m = _parse_metadata(raw)
    assert m.garments[0].category == "dress"
    # Test code fence stripping
    fenced = "```json\n{}\n```"
    assert _strip_code_fences(fenced) == "{}"

def test_query_parser_parsing():
    from src.retrieval.query_parser import _parse_metadata, _strip_code_fences
    raw = '{"garments": [{"category": "jacket"}], "scene": {}, "person": {}}'
    m = _parse_metadata(raw)
    assert m.garments[0].category == "jacket"
    # Test fallback on bad JSON
    m2 = _parse_metadata("not valid json at all")
    from src.schemas import FashionMetadata
    assert isinstance(m2, FashionMetadata)

def test_rrf():
    from src.retrieval.retriever import _reciprocal_rank_fusion
    # Test with empty lists
    result = _reciprocal_rank_fusion([], [])
    assert result == []

def test_qdrant_helpers():
    from src.qdrant_store import _image_id_to_int
    val = _image_id_to_int("IMG_0a1b2c3d4e5f6789")
    assert isinstance(val, int)
    assert val >= 0

def test_image_loader_import():
    from src.indexing._image_loader import load_image
    # Just verify it imports (no file needed)

def test_build_index_argparse():
    from scripts.build_index import _parse_args
    from pathlib import Path
    args = _parse_args(["--image-dir", "./data"])
    assert args.image_dir == Path("./data")

def test_search_cli_argparse():
    from scripts.search_cli import _parse_args
    args = _parse_args(["--query", "blue dress", "--top-k", "5"])
    assert args.query == "blue dress"
    assert args.top_k == 5

check("schemas", test_schemas)
check("utils (config_loader + helpers + logging)", test_utils)
check("vlm.metadata_extractor (parsing + fallback)", test_metadata_extractor_parsing)
check("retrieval.query_parser (parsing + fallback)", test_query_parser_parsing)
check("retrieval.retriever (_reciprocal_rank_fusion)", test_rrf)
check("qdrant_store (_image_id_to_int)", test_qdrant_helpers)
check("indexing._image_loader (import)", test_image_loader_import)
check("scripts.build_index (argparse)", test_build_index_argparse)
check("scripts.search_cli (argparse)", test_search_cli_argparse)

print()
if errors:
    print(f"FAILED: {len(errors)} checks failed: {errors}")
    sys.exit(1)
else:
    print(f"All {9} smoke checks passed.")
