from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_visora.image_generator import ImageGenerator


@pytest.mark.asyncio
async def test_one_deadline_bounds_the_entire_model_fallback_chain():
    generator = ImageGenerator(
        api_keys=["key"],
        base_url="https://images.example",
        model="first",
        max_retry_attempts=1,
        task_timeout=0.09,
        fallback_models=[
            {
                "model": "second",
                "api_keys": ["key"],
                "base_url": "https://images.example",
                "api_request_format": "openai_images",
            },
            {
                "model": "third",
                "api_keys": ["key"],
                "base_url": "https://images.example",
                "api_request_format": "openai_images",
            },
        ],
    )
    calls = []

    async def attempt(*args, **kwargs):
        calls.append(kwargs.get("model"))
        await asyncio.sleep(0.06)
        return None, "API 错误 (500): unavailable"

    generator._try_generate_with_current_key = AsyncMock(side_effect=attempt)
    started = time.monotonic()
    images, error = await generator.generate_image("draw", image_size="4K")
    assert images is None
    assert "总时限" in error
    assert calls == ["first", "second"]
    assert time.monotonic() - started < 0.5
    assert generator.result_metadata == {}
