from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from astrbot_plugin_visora.generation_runtime import (
    create_image_generator,
    create_video_generator,
)
from astrbot_plugin_visora.temp_media import (
    cleanup_stale_media,
    remove_downloaded_temp_file,
    remove_temp_file,
    save_generated_image,
)


def _plugin_settings() -> SimpleNamespace:
    return SimpleNamespace(
        default_provider_full_id="provider/image-model",
        api_keys=["image-key"],
        base_url="https://example.com/v1",
        model="image-model",
        api_request_format="v1/chat/completions",
        img2img_provider_full_id="provider/edit-model",
        img2img_api_keys=["edit-key"],
        img2img_base_url="https://example.com/v1",
        img2img_model="edit-model",
        img2img_api_request_format="v1/images/edits",
        persona_provider_full_id="provider/persona-model",
        persona_api_keys=["persona-key"],
        persona_base_url="https://example.com/v1",
        persona_model="persona-model",
        persona_api_request_format="v1/images/edits",
        fallback_models=[{"model": "fallback", "api_keys": ["fallback-key"]}],
        img2img_fallback_models=[
            {
                "model": "edit-fallback",
                "api_keys": ["edit-fallback-key"],
                "api_request_format": "v1/images/edits",
            }
        ],
        persona_fallback_models=[
            {
                "model": "persona-fallback",
                "api_keys": ["persona-fallback-key"],
                "api_request_format": "v1/images/edits",
            }
        ],
        admin_provider_full_id="provider/admin-model",
        admin_api_keys=["admin-key"],
        admin_base_url="https://example.com/v1",
        admin_model="admin-model",
        admin_api_request_format="v1/images/generations",
        admin_fallback_models=[
            {
                "model": "admin-fallback",
                "api_keys": ["admin-fallback-key"],
                "api_request_format": "v1/images/generations",
            }
        ],
        timeout=30,
        max_retry_attempts=2,
        proxy=None,
        safety_settings="BLOCK_NONE",
        video_api_keys=["video-key"],
        video_base_url="https://video.example.com/v1",
        video_model="video-model",
        video_timeout=60,
        video_api_request_formats=["v1/videos"],
        video_task_query_format="v1/videos/{task_id}",
        video_poll_interval=2,
        max_video_size_mb=50,
        video_duration_field="all",
        video_max_unknown_polls=4,
        video_payload_mode="auto",
        video_frame_rate=24,
        video_extra_payload={"nested": {"value": 1}},
        video_query_model_name=True,
    )


def test_request_generators_do_not_share_mutable_state():
    plugin = _plugin_settings()
    first = create_image_generator(plugin, "txt2img")
    second = create_image_generator(plugin, "txt2img")
    first.api_keys.append("mutated")
    first.fallback_models[0]["api_keys"].append("mutated")

    assert second.api_keys == ["image-key"]
    assert second.fallback_models[0]["api_keys"] == ["fallback-key"]

    edit = create_image_generator(plugin, "img2img")
    persona = create_image_generator(plugin, "persona")
    assert edit.provider_id == "provider/edit-model"
    assert edit.api_request_format == "v1/images/edits"
    assert edit.fallback_models[0]["model"] == "edit-fallback"
    assert persona.provider_id == "provider/persona-model"
    assert persona.api_request_format == "v1/images/edits"
    assert persona.fallback_models[0]["model"] == "persona-fallback"

    first_video = create_video_generator(plugin)
    second_video = create_video_generator(plugin)
    first_video.extra_payload["nested"]["value"] = 2
    assert second_video.extra_payload["nested"]["value"] == 1


def test_create_image_generator_admin_mode_uses_admin_route():
    plugin = _plugin_settings()
    admin = create_image_generator(plugin, "admin")

    assert admin.provider_id == "provider/admin-model"
    assert admin.api_keys == ["admin-key"]
    assert admin.model == "admin-model"
    assert admin.api_request_format == "v1/images/generations"
    assert admin.safety_settings == "BLOCK_NONE"
    assert admin.fallback_models[0]["model"] == "admin-fallback"


def test_create_image_generator_unknown_mode_raises():
    plugin = _plugin_settings()
    with pytest.raises(ValueError, match="未知生图模式"):
        create_image_generator(plugin, "unknown-mode")


def test_create_image_generators_share_global_safety_settings():
    plugin = _plugin_settings()
    for mode in ("txt2img", "img2img", "persona", "admin"):
        generator = create_image_generator(plugin, mode)
        assert generator.safety_settings == plugin.safety_settings


def test_owned_temp_media_lifecycle():
    async def run():
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            tmp_path = Path(directory)
            with patch(
                "astrbot_plugin_visora.temp_media.get_astrbot_temp_path",
                return_value=str(tmp_path),
            ):
                image_path = await save_generated_image(
                    b"\x89PNG\r\n\x1a\ncontent", "task"
                )
                assert os.path.exists(image_path)
                await remove_temp_file(image_path)
                assert not os.path.exists(image_path)

                generic_path = tmp_path / "io_temp_img_other.png"
                generic_path.write_bytes(b"data")
                await remove_temp_file(str(generic_path))
                assert generic_path.exists()
                await remove_downloaded_temp_file(str(generic_path))
                assert not generic_path.exists()

                stale_owned = tmp_path / "visora_video_stale.mp4"
                stale_owned.write_bytes(b"video")
                stale_generic = tmp_path / "io_temp_img_stale.png"
                stale_generic.write_bytes(b"image")
                old_time = time.time() - 7200
                os.utime(stale_owned, (old_time, old_time))
                os.utime(stale_generic, (old_time, old_time))

                removed = await cleanup_stale_media(max_age_seconds=3600)
                assert removed == 1
                assert not stale_owned.exists()
                assert stale_generic.exists()

    asyncio.run(run())
