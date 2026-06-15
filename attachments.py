#!/usr/bin/env python3
"""Build OpenAI-compatible multimodal user content from attached files."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml",
    ".kt", ".java", ".gradle", ".xml", ".html", ".css", ".scss", ".sql", ".sh",
    ".toml", ".ini", ".properties", ".rs", ".go", ".rb", ".swift", ".c", ".h",
    ".cpp", ".hpp", ".vue", ".svelte",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}


def _mime(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS or _mime(path).startswith("image/")


def is_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if is_image(path):
        return False
    try:
        sample = path.read_bytes()[:4096]
        sample.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def load_seed_images(paths: list[Path]) -> list[dict]:
    """Seed images for Python virtual filesystem (ICR-style)."""
    seeds: list[dict] = []
    for i, path in enumerate(paths):
        if not path.exists() or not is_image(path):
            continue
        mime = _mime(path)
        seeds.append({
            "name": path.name,
            "mimeType": mime,
            "base64": base64.standard_b64encode(path.read_bytes()).decode("ascii"),
        })
    return seeds


def build_initial_user_content(task: str, attach_paths: list[Path]) -> str | list[dict]:
    """
    OpenAI chat message content: string or list of {type, text|image_url} parts.
    Matches how hosted APIs accept multimodal input.
    """
    parts: list[dict] = [{"type": "text", "text": f"Initial User Request:\n{task}"}]

    for path in attach_paths:
        if not path.exists():
            raise FileNotFoundError(f"Attachment not found: {path}")
        resolved = path.resolve()

        if is_image(resolved):
            mime = _mime(resolved)
            b64 = base64.standard_b64encode(resolved.read_bytes()).decode("ascii")
            parts.append({
                "type": "text",
                "text": f"\n\nAttached image: {resolved.name}",
            })
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        elif is_text(resolved):
            try:
                body = resolved.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                body = resolved.read_text(encoding="latin-1")
            if len(body) > 120_000:
                body = body[:120_000] + "\n\n[... truncated for context limit ...]"
            parts.append({
                "type": "text",
                "text": f"\n\n--- Attached file: {resolved} ---\n{body}\n--- end {resolved.name} ---",
            })
        else:
            parts.append({
                "type": "text",
                "text": f"\n\n[Skipped binary attachment: {resolved}]",
            })

    return parts if len(parts) > 1 else parts[0]["text"]
