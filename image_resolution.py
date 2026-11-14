from __future__ import annotations

import re

IMAGE_RESOLUTIONS = ("1K", "2K", "4K")
CUSTOM_RESOLUTION_OPTION = "自定义"
_CUSTOM_RESOLUTION_RE = re.compile(r"^\s*(\d+)\s*[xX×]\s*(\d+)\s*$")


def normalize_image_resolution(value: object, fallback: str | None = None) -> str:
    """Normalize a preset or exact WIDTHxHEIGHT image resolution."""
    raw_value = str(value or "").strip()
    preset = raw_value.upper()
    if preset in IMAGE_RESOLUTIONS:
        return preset

    match = _CUSTOM_RESOLUTION_RE.fullmatch(raw_value)
    if match:
        width, height = (int(part) for part in match.groups())
        if width > 0 and height > 0:
            return f"{width}x{height}"

    if fallback is not None and not raw_value:
        return normalize_image_resolution(fallback)
    raise ValueError(
        f"无效的图片分辨率 {raw_value!r}，请使用 1K、2K、4K 或 WIDTHxHEIGHT（如 1920x1080）"
    )


def parse_exact_resolution(value: object) -> tuple[int, int] | None:
    """Return exact dimensions when value is WIDTHxHEIGHT, otherwise None."""
    normalized = normalize_image_resolution(value)
    if normalized in IMAGE_RESOLUTIONS:
        return None
    width, height = normalized.split("x", 1)
    return int(width), int(height)


def resolve_configured_resolution(
    selection: object,
    custom_resolution: object,
    fallback: str = "2K",
) -> str:
    """Resolve the configured preset/custom selector to a request value."""
    raw_selection = str(selection or "").strip()
    if not raw_selection:
        return normalize_image_resolution(fallback)
    if raw_selection == CUSTOM_RESOLUTION_OPTION:
        try:
            return normalize_image_resolution(custom_resolution)
        except ValueError as exc:
            raise ValueError(f"自定义分辨率配置无效：{exc}") from exc
    return normalize_image_resolution(raw_selection)
