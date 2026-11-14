from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from astrbot_plugin_visora.image_generator import ImageGenerator


def _form_size(form: aiohttp.FormData) -> str:
    for field_options, _, value in form._fields:
        if field_options["name"] == "size":
            return str(value)
    raise AssertionError("size field missing")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_format", "images_data"),
    [
        ("v1/images/generations", []),
        ("v1/images/edits", [(b"source", "image/png")]),
    ],
)
async def test_relay_4k_invalid_size_retries_once_with_2k(api_format, images_data):
    generator = ImageGenerator(
        api_keys=["test-key"],
        base_url="https://relay.example/v1",
        model="gpt-image-2",
        api_request_format=api_format,
        max_retry_attempts=1,
    )
    generator._get_session = Mock(return_value=object())
    generator._make_api_request = AsyncMock(
        side_effect=[
            (None, "API 错误 (400): invalid size"),
            {"data": [{"b64_json": "ignored"}]},
        ]
    )
    generator._extract_image_from_response = AsyncMock(return_value=[b"result"])
    generator._log_result_image_metadata = Mock(return_value=[(2048, 2048)])

    images, error = await generator._try_generate_with_current_key(
        "draw this",
        images_data,
        "1:1",
        "4K",
        model="gpt-image-2",
        base_url="https://relay.example/v1",
        api_key="test-key",
        api_format=api_format,
    )

    assert images == [b"result"]
    assert error is None
    assert generator.result_metadata["requested_resolution"] == "4K"
    assert generator.result_metadata["effective_resolution"] == "2K"
    assert generator.result_metadata["resolution_downgraded"] is True
    assert generator.result_metadata["dimensions"] == [(2048, 2048)]
    assert generator._make_api_request.await_count == 2
    first_payload = generator._make_api_request.await_args_list[0].args[1]
    second_payload = generator._make_api_request.await_args_list[1].args[1]
    if isinstance(first_payload, aiohttp.FormData):
        assert _form_size(first_payload) == "2880x2880"
        assert _form_size(second_payload) == "2048x2048"
    else:
        assert first_payload["size"] == "2880x2880"
        assert second_payload["size"] == "2048x2048"


@pytest.mark.asyncio
async def test_exact_resolution_does_not_retry_with_2k():
    generator = ImageGenerator(
        api_keys=["test-key"],
        base_url="https://relay.example/v1",
        model="gpt-image-2",
        api_request_format="v1/images/generations",
        max_retry_attempts=1,
    )
    generator._get_session = Mock(return_value=object())
    generator._make_api_request = AsyncMock(
        return_value=(None, "API 错误 (400): invalid size")
    )

    images, error = await generator._try_generate_with_current_key(
        "draw this",
        [],
        "1:1",
        "1920x1088",
        model="gpt-image-2",
        base_url="https://relay.example/v1",
        api_key="test-key",
        api_format="v1/images/generations",
    )

    assert images is None
    assert "invalid size" in error
    generator._make_api_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_z_image_turbo_4k_error_skips_2k_downgrade_retry():
    generator = ImageGenerator(
        api_keys=["test-key"],
        base_url="https://relay.example/v1",
        model="Z-Image-Turbo",
        api_request_format="v1/images/generations",
        max_retry_attempts=1,
    )
    generator._get_session = Mock(return_value=object())
    generator._make_api_request = AsyncMock(
        return_value=(None, "API 错误 (400): invalid size")
    )

    images, error = await generator._try_generate_with_current_key(
        "draw this",
        [],
        "1:1",
        "4K",
        model="Z-Image-Turbo",
        base_url="https://relay.example/v1",
        api_key="test-key",
        api_format="v1/images/generations",
    )

    assert images is None
    assert "invalid size" in error
    generator._make_api_request.assert_awaited_once()
    first_payload = generator._make_api_request.await_args_list[0].args[1]
    assert "size" not in first_payload
