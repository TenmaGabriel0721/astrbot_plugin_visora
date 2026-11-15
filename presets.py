"""Typed image presets shared by command handlers and the settings page."""

from __future__ import annotations

import json
import re
from typing import Any

MAX_PRESETS = 256
PRESET_MODES = {"txt2img": "文生图", "img2img": "图生图"}
PRESET_ASPECT_RATIOS = (
    "inherit",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
    "auto",
)
PRESET_RESOLUTIONS = ("inherit", "1K", "2K", "4K")


def normalize_presets(raw: Any, *, strict: bool = False) -> list[dict[str, str]]:
    """Normalize typed presets and legacy ``name:prompt`` entries.

    Legacy presets default to image-to-image, preserving the reference-based
    behavior of the original templates. Explicit legacy JSON modes are honored.

    Args:
        raw: Configuration entries or a settings-page submission.
        strict: Reject invalid fields and duplicate names when saving settings.

    Returns:
        Presets with a task type, name, prompt, aspect ratio, and resolution.

    Raises:
        ValueError: A strict submission contains invalid or ambiguous entries.
    """
    if not isinstance(raw, list) or len(raw) > MAX_PRESETS:
        if strict:
            raise ValueError(f"预设必须是列表，且最多包含 {MAX_PRESETS} 项")
        return []
    result: dict[str, dict[str, str]] = {}
    for item in raw:
        try:
            if isinstance(item, str) and ":" in item:
                name, content = item.split(":", 1)
                data: dict[str, Any] = {"name": name, "prompt": content}
                if content.strip().startswith("{"):
                    try:
                        details = json.loads(content)
                    except json.JSONDecodeError:
                        details = None
                    if isinstance(details, dict):
                        data.update(details)
                        data["name"] = name
            elif isinstance(item, dict):
                data = item
            else:
                raise ValueError("预设必须是对象，或旧版的“名称:提示词”格式")

            name = data.get("name", "")
            prompt = data.get("prompt", "")
            mode = data.get("__template_key") or data.get("mode") or "img2img"
            ratio = data.get("aspect_ratio") or "inherit"
            resolution = data.get("resolution") or "inherit"
            if (
                not isinstance(name, str)
                or not name.strip()
                or len(name) > 80
                or re.search(r"[:：\r\n]", name)
            ):
                raise ValueError("预设名称须为 1 至 80 个字符，且不能包含冒号或换行")
            name = name.strip()
            if not isinstance(prompt, str) or len(prompt) > 32768:
                raise ValueError(f"预设“{name}”的提示词必须是文本，且最多 32768 个字符")
            if strict and not prompt.strip():
                raise ValueError(f"请填写预设“{name}”的提示词")
            if mode not in PRESET_MODES:
                raise ValueError(f"请选择预设“{name}”的文生图或图生图分类")
            if ratio not in PRESET_ASPECT_RATIOS:
                raise ValueError(f"预设“{name}”的画面比例无效")
            if resolution not in PRESET_RESOLUTIONS:
                raise ValueError(f"预设“{name}”的分辨率须为 1K、2K、4K 或跟随默认")
            key = name.casefold()
            if strict and key in result:
                raise ValueError(f"预设名称重复：{name}；两种分类的名称也不能相同")
            result[key] = {
                "__template_key": mode,
                "name": name,
                "prompt": prompt.strip(),
                "aspect_ratio": ratio,
                "resolution": resolution,
            }
        except (TypeError, ValueError):
            if strict:
                raise
    return list(result.values())
