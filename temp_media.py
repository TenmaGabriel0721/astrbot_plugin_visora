"""Lifecycle helpers for generated temporary media files."""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from collections.abc import Iterable

from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

OWNED_PREFIXES = ("visora_image_", "visora_video_")


def _is_managed_temp_path(path: str) -> bool:
    try:
        real_path = os.path.realpath(path)
        temp_dir = os.path.realpath(get_astrbot_temp_path())
        return real_path == temp_dir or real_path.startswith(temp_dir + os.sep)
    except Exception:
        return False


def _is_owned_temp_path(path: str) -> bool:
    return _is_managed_temp_path(path) and os.path.basename(path).startswith(
        OWNED_PREFIXES
    )


def _image_suffix(data: bytes) -> str:
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


async def save_generated_image(data: bytes, task_id: str | None) -> str:
    temp_dir = get_astrbot_temp_path()
    os.makedirs(temp_dir, exist_ok=True)
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id or "image")
    path = os.path.abspath(
        os.path.join(
            temp_dir,
            f"visora_image_{safe_task_id}_{uuid.uuid4().hex}{_image_suffix(data)}",
        )
    )
    try:
        with open(path, "wb") as file:
            file.write(data)
        return path
    except BaseException:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        raise


async def remove_temp_file(path: str | None) -> None:
    if not path or not _is_owned_temp_path(path):
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.warning(f"[Visora] Failed to remove temporary media: {path}: {exc}")


async def remove_downloaded_temp_file(path: str | None) -> None:
    """Remove a file returned directly by AstrBot's download helper."""
    if not path or not _is_managed_temp_path(path):
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.warning(
            f"[Visora] Failed to remove a downloaded temporary file: {path}: {exc}"
        )


async def remove_temp_files(paths: Iterable[str]) -> None:
    unique_paths = list(dict.fromkeys(path for path in paths if path))
    if unique_paths:
        await asyncio.gather(
            *(remove_temp_file(path) for path in unique_paths),
            return_exceptions=True,
        )


async def cleanup_stale_media(max_age_seconds: int = 24 * 60 * 60) -> int:
    """Remove abandoned plugin-owned media left by crashes or forced exits."""
    temp_dir = get_astrbot_temp_path()
    if not os.path.isdir(temp_dir):
        return 0
    cutoff = time.time() - max(60, int(max_age_seconds))
    stale_paths: list[str] = []
    try:
        for entry in os.scandir(temp_dir):
            if not entry.is_file(follow_symlinks=False):
                continue
            if not entry.name.startswith(OWNED_PREFIXES):
                continue
            try:
                if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                    stale_paths.append(entry.path)
            except FileNotFoundError:
                continue
    except Exception as exc:
        logger.warning(f"[Visora] Failed to scan stale temporary media: {exc}")
        return 0

    await remove_temp_files(stale_paths)
    if stale_paths:
        logger.info(f"[Visora] Removed {len(stale_paths)} stale temporary media files")
    return len(stale_paths)
