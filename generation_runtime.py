"""Request-scoped generator factories.

Each generation request receives its own mutable generator instance. This keeps
provider/model/key rotation and video polling state isolated across concurrent
tasks while preserving the plugin's per-request provider refresh behavior.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .image_generator import ImageGenerator
from .video_generator import VideoGenerator


def create_image_generator(plugin: Any, mode: str) -> ImageGenerator:
    """Create one request-scoped generator for the selected image mode.

    Args:
        plugin: Loaded Visora plugin instance.
        mode: Image generation mode: txt2img, img2img, persona, or admin.

    Returns:
        An isolated generator with its mode-specific primary and fallback routes.

    Raises:
        ValueError: If ``mode`` is not a known image generation mode.
    """
    if mode == "img2img":
        provider_id = plugin.img2img_provider_full_id
        api_keys = plugin.img2img_api_keys
        base_url = plugin.img2img_base_url
        model = plugin.img2img_model
        api_request_format = plugin.img2img_api_request_format
        fallback_models = plugin.img2img_fallback_models
    elif mode == "persona":
        provider_id = plugin.persona_provider_full_id
        api_keys = plugin.persona_api_keys
        base_url = plugin.persona_base_url
        model = plugin.persona_model
        api_request_format = plugin.persona_api_request_format
        fallback_models = plugin.persona_fallback_models
    elif mode == "admin":
        provider_id = plugin.admin_provider_full_id
        api_keys = plugin.admin_api_keys
        base_url = plugin.admin_base_url
        model = plugin.admin_model
        api_request_format = plugin.admin_api_request_format
        fallback_models = plugin.admin_fallback_models
    elif mode == "txt2img":
        provider_id = plugin.default_provider_full_id
        api_keys = plugin.api_keys
        base_url = plugin.base_url
        model = plugin.model
        api_request_format = plugin.api_request_format
        fallback_models = plugin.fallback_models
    else:
        raise ValueError(f"未知生图模式: {mode!r}")

    return ImageGenerator(
        api_keys=list(api_keys or []),
        base_url=base_url,
        model=model,
        provider_id=provider_id,
        timeout=plugin.timeout,
        task_timeout=getattr(plugin, "task_timeout", 600),
        max_retry_attempts=plugin.max_retry_attempts,
        proxy=plugin.proxy,
        safety_settings=plugin.safety_settings,
        fallback_models=deepcopy(fallback_models),
        api_request_format=api_request_format,
    )


def create_video_generator(plugin: Any) -> VideoGenerator:
    return VideoGenerator(
        api_keys=list(plugin.video_api_keys or []),
        base_url=plugin.video_base_url,
        model=plugin.video_model,
        timeout=plugin.video_timeout,
        max_retry_attempts=plugin.max_retry_attempts,
        proxy=plugin.proxy,
        api_request_formats=list(plugin.video_api_request_formats or []),
        task_query_format=plugin.video_task_query_format,
        poll_interval=plugin.video_poll_interval,
        max_video_size_mb=plugin.max_video_size_mb,
        duration_field=plugin.video_duration_field,
        max_unknown_polls=plugin.video_max_unknown_polls,
        payload_mode=plugin.video_payload_mode,
        frame_rate=plugin.video_frame_rate,
        extra_payload=deepcopy(plugin.video_extra_payload),
        query_model_name=plugin.video_query_model_name,
    )
