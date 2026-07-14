"""
Unit tests for src/utils/helpers.py

Tests cover:
  - generate_image_id: stability, uniqueness, format
  - iter_images: finds images, filters extensions, handles missing dir
  - chunk_list: correct chunking, handles empty list
  - timer: context manager runs without error
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.utils.helpers import chunk_list, generate_image_id, iter_images, timer


class TestGenerateImageId:
    def test_format(self, tmp_path: Path):
        img = tmp_path / "test.jpg"
        img.touch()
        id_ = generate_image_id(img)
        assert id_.startswith("IMG_")
        assert len(id_) == 4 + 16  # "IMG_" + 16 hex chars

    def test_stability(self, tmp_path: Path):
        img = tmp_path / "stable.jpg"
        img.touch()
        assert generate_image_id(img) == generate_image_id(img)

    def test_uniqueness(self, tmp_path: Path):
        img1 = tmp_path / "a.jpg"
        img2 = tmp_path / "b.jpg"
        img1.touch()
        img2.touch()
        assert generate_image_id(img1) != generate_image_id(img2)


class TestIterImages:
    def test_finds_images(self, tmp_path: Path):
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.jpeg").touch()
        (tmp_path / "c.png").touch()
        (tmp_path / "d.txt").touch()  # should be ignored

        found = list(iter_images(tmp_path))
        names = [p.name for p in found]
        assert "a.jpg" in names
        assert "b.jpeg" in names
        assert "c.png" in names
        assert "d.txt" not in names

    def test_recursive(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.jpg").touch()

        found = list(iter_images(tmp_path))
        assert any("nested.jpg" == p.name for p in found)

    def test_sorted_output(self, tmp_path: Path):
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (tmp_path / name).touch()

        found = list(iter_images(tmp_path))
        names = [p.name for p in found]
        assert names == sorted(names)

    def test_custom_extensions(self, tmp_path: Path):
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.webp").touch()
        (tmp_path / "c.png").touch()

        # Only allow .webp
        found = list(iter_images(tmp_path, extensions=[".webp"]))
        assert all(p.suffix == ".webp" for p in found)
        assert len(found) == 1

    def test_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(iter_images(tmp_path / "nonexistent"))

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        found = list(iter_images(tmp_path))
        assert found == []


class TestChunkList:
    def test_even_split(self):
        items = list(range(6))
        chunks = list(chunk_list(items, 2))
        assert chunks == [[0, 1], [2, 3], [4, 5]]

    def test_uneven_split(self):
        items = list(range(5))
        chunks = list(chunk_list(items, 2))
        assert chunks == [[0, 1], [2, 3], [4]]

    def test_chunk_larger_than_list(self):
        items = [1, 2]
        chunks = list(chunk_list(items, 10))
        assert chunks == [[1, 2]]

    def test_empty_list(self):
        chunks = list(chunk_list([], 4))
        assert chunks == []

    def test_chunk_size_one(self):
        items = [10, 20, 30]
        chunks = list(chunk_list(items, 1))
        assert chunks == [[10], [20], [30]]


class TestTimer:
    def test_runs_without_error(self):
        with timer("test_operation"):
            x = 1 + 1
        assert x == 2

    def test_propagates_exception(self):
        with pytest.raises(ValueError, match="test error"):
            with timer("failing_operation"):
                raise ValueError("test error")
