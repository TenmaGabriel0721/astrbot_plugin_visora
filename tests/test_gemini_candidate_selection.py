from __future__ import annotations

import base64
from io import BytesIO

import pytest
from astrbot_plugin_visora.image_generator import ImageGenerator
from PIL import Image


def _png_bytes(width: int, height: int, color: int) -> bytes:
    image = Image.new("1", (width, height), color=color)
    output = BytesIO()
    image.save(output, format="PNG", compress_level=1)
    return output.getvalue()


def _candidate(image_data: bytes) -> dict:
    return {
        "content": {
            "parts": [
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64.b64encode(image_data).decode("ascii"),
                    }
                }
            ]
        }
    }


def _generator() -> ImageGenerator:
    return ImageGenerator(
        api_keys=["test-key"],
        base_url="https://images.example/v1",
        model="visora-image-model",
        api_request_format="gemini",
        max_retry_attempts=1,
    )


@pytest.mark.asyncio
async def test_extract_gemini_candidates_returns_only_largest_image():
    preview = _png_bytes(1024, 1024, 0)
    full_resolution = _png_bytes(3072, 3072, 1)
    response = {
        "candidates": [
            _candidate(preview),
            _candidate(full_resolution),
        ]
    }

    result = await _generator()._extract_image_from_response(response)

    assert result == [full_resolution]


@pytest.mark.asyncio
async def test_extract_gemini_candidates_uses_last_image_when_pixels_tie():
    first = _png_bytes(1024, 1024, 0)
    last = _png_bytes(1024, 1024, 1)
    response = {
        "candidates": [
            _candidate(first),
            _candidate(last),
        ]
    }

    result = await _generator()._extract_image_from_response(response)

    assert result == [last]
