from __future__ import annotations

import base64
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_visora.image_generator import ImageGenerator


@pytest.mark.asyncio
async def test_openai_data_item_prefers_url_over_b64_json():
    generator = ImageGenerator(
        api_keys=["test-key"],
        base_url="https://images.example/v1",
        model="image-model",
        api_request_format="v1/images/generations",
        max_retry_attempts=1,
    )
    url = "https://cdn.example/generated.png"
    url_image = b"url-image"
    fallback_image = b"base64-image"
    generator._download_image_source = AsyncMock(return_value=url_image)

    images = await generator._extract_image_from_response(
        {
            "data": [
                {
                    "url": url,
                    "b64_json": base64.b64encode(fallback_image).decode(),
                }
            ]
        }
    )

    assert images == [url_image]
    generator._download_image_source.assert_awaited_once_with(url)


@pytest.mark.asyncio
async def test_openai_data_item_falls_back_to_b64_json_when_url_download_fails():
    generator = ImageGenerator(
        api_keys=["test-key"],
        base_url="https://images.example/v1",
        model="image-model",
        api_request_format="v1/images/generations",
        max_retry_attempts=1,
    )
    url = "https://cdn.example/generated.png"
    fallback_image = b"base64-image"
    generator._download_image_source = AsyncMock(return_value=None)

    images = await generator._extract_image_from_response(
        {
            "data": [
                {
                    "url": url,
                    "b64_json": base64.b64encode(fallback_image).decode(),
                }
            ]
        }
    )

    assert images == [fallback_image]
    generator._download_image_source.assert_awaited_once_with(url)
