"""HF cache completeness — partial metadata-only downloads must not count as cached."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.model_cache import hub_dir_for, is_model_cached, weight_bytes_in_hub  # noqa: E402

REPO = "Qwen/Qwen3.6-27B-FP8"


class ModelCacheTest(unittest.TestCase):
    def test_empty_cache_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(is_model_cached(REPO, tmp))

    def test_config_only_partial_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub = hub_dir_for(REPO, tmp)
            snap = os.path.join(hub, "snapshots", "abc123")
            os.makedirs(snap)
            with open(os.path.join(snap, "config.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            self.assertLess(weight_bytes_in_hub(hub), 1_000_000)
            self.assertFalse(is_model_cached(REPO, tmp))

    def test_weights_above_threshold_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub = hub_dir_for(REPO, tmp)
            snap = os.path.join(hub, "snapshots", "abc123")
            os.makedirs(snap)
            with open(os.path.join(snap, "config.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            # 1.1 GB fake shard
            with open(os.path.join(snap, "model.safetensors"), "wb") as f:
                f.write(b"\0" * (1_100_000_000))
            self.assertTrue(is_model_cached(REPO, tmp, min_weight_bytes=1_000_000_000))

    def test_small_shard_still_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub = hub_dir_for(REPO, tmp)
            snap = os.path.join(hub, "snapshots", "abc123")
            os.makedirs(snap)
            with open(os.path.join(snap, "config.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            with open(os.path.join(snap, "model.safetensors"), "wb") as f:
                f.write(b"\0" * (28 * 1024 * 1024))  # 28M like user's stuck state
            self.assertFalse(is_model_cached(REPO, tmp))


if __name__ == "__main__":
    unittest.main()
