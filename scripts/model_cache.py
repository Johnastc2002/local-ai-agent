#!/usr/bin/env python3
"""Verify HuggingFace hub cache has complete model weights (not partial metadata)."""

from __future__ import annotations

import argparse
import os
import sys


def hub_dir_for(repo_id: str, cache_dir: str) -> str:
    slug = "models--" + repo_id.replace("/", "--")
    return os.path.join(cache_dir, slug)


def weight_bytes_in_hub(hub: str) -> int:
    total = 0
    if not os.path.isdir(hub):
        return 0
    for root, _, files in os.walk(hub):
        for name in files:
            if name.endswith((".safetensors", ".bin", ".gguf")):
                total += os.path.getsize(os.path.join(root, name))
    return total


def has_config_in_hub(hub: str, repo_id: str, cache_dir: str) -> bool:
    if os.path.isdir(hub):
        for root, _, files in os.walk(hub):
            if "config.json" in files:
                return True
    try:
        from huggingface_hub import try_to_load_from_cache

        return bool(try_to_load_from_cache(repo_id, "config.json", cache_dir=cache_dir))
    except ImportError:
        return False


def is_model_cached(
    repo_id: str,
    cache_dir: str,
    *,
    min_weight_bytes: int = 1_000_000_000,
) -> bool:
    """True only if config exists and weight files sum to min_weight_bytes+."""
    hub = hub_dir_for(repo_id, cache_dir)
    if not has_config_in_hub(hub, repo_id, cache_dir):
        return False
    return weight_bytes_in_hub(hub) >= min_weight_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check HF model cache completeness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("is-cached", help="exit 0 if complete cache")
    check.add_argument("repo_id")
    check.add_argument("cache_dir")
    check.add_argument("--min-gb", type=float, default=1.0)

    info = sub.add_parser("info", help="print cache stats")
    info.add_argument("repo_id")
    info.add_argument("cache_dir")

    args = parser.parse_args(argv)

    if args.cmd == "is-cached":
        min_bytes = int(args.min_gb * 1_000_000_000)
        return 0 if is_model_cached(args.repo_id, args.cache_dir, min_weight_bytes=min_bytes) else 1

    hub = hub_dir_for(args.repo_id, args.cache_dir)
    weights = weight_bytes_in_hub(hub)
    print(f"hub={hub}")
    print(f"weight_bytes={weights}")
    print(f"cached={is_model_cached(args.repo_id, args.cache_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
