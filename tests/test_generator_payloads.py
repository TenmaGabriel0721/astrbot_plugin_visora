from __future__ import annotations

from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from astrbot_plugin_visora.image_generator import ImageGenerator


def make_generator(
    *,
    request_format=None,
    provider_id=None,
    fallback_models=None,
) -> ImageGenerator:
    return ImageGenerator(
        api_keys=["test-key"],
        base_url="https://images.example/v1",
        model="image-model",
        api_request_format=request_format or "v1/chat/completions",
        provider_id=provider_id,
        fallback_models=fallback_models,
        max_retry_attempts=1,
    )


@pytest.mark.parametrize(
    ("base_url", "format_path", "expected"),
    [
        (
            "https://images.example/v1",
            "v1beta/models/{model}:generateContent",
            "https://images.example/v1beta/models/image-model:generateContent",
        ),
        (
            "https://images.example/v1",
            "v1/images/generations",
            "https://images.example/v1/images/generations",
        ),
        (
            "https://images.example/api/v1",
            "v2/images/generations",
            "https://images.example/api/v2/images/generations",
        ),
        (
            "https://images.example/v1beta",
            "v1beta/models/{model}:generateContent",
            "https://images.example/v1beta/models/image-model:generateContent",
        ),
        (
            "https://images.example/api",
            "v1beta/models/{model}:generateContent",
            "https://images.example/api/v1beta/models/image-model:generateContent",
        ),
        (
            "https://images.example/v1",
            "images/generations",
            "https://images.example/v1/images/generations",
        ),
    ],
)
def test_build_endpoint_handles_versioned_and_relative_request_paths(
    base_url, format_path, expected
):
    generator = make_generator()

    assert generator._build_endpoint(base_url, format_path, "image-model") == expected


def test_build_endpoint_preserves_base_query_parameters():
    generator = make_generator()

    assert (
        generator._build_endpoint(
            "https://images.example/v1?tenant=demo",
            "v1beta/models/{model}:generateContent",
            "image-model",
        )
        == "https://images.example/v1beta/models/image-model:generateContent?tenant=demo"
    )


GPT_IMAGE_2_4K_EXPECTED_SIZES = [
    ("auto", "2880x2880"),
    ("1:1", "2880x2880"),
    ("2:3", "2336x3504"),
    ("3:2", "3504x2336"),
    ("3:4", "2448x3264"),
    ("4:3", "3264x2448"),
    ("4:5", "2560x3200"),
    ("5:4", "3200x2560"),
    ("9:16", "2160x3840"),
    ("16:9", "3840x2160"),
    ("21:9", "3808x1632"),
]

SUPPORTED_IMAGE_ASPECT_RATIOS = (
    "auto",
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
)


@pytest.mark.parametrize(("ratio", "expected"), GPT_IMAGE_2_4K_EXPECTED_SIZES)
def test_gpt_image_2_uses_documented_4k_limits_on_relays(ratio, expected):
    generator = make_generator()

    assert generator._map_openai_size(ratio, "4K", model="gpt-image-2") == expected


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        ("1:1", "2880x2880"),
        ("16:9", "3840x2160"),
        ("9:16", "2160x3840"),
    ],
)
def test_official_gpt_image_2_keeps_documented_4k_limits(ratio, expected):
    generator = make_generator()

    assert (
        generator._map_openai_size(
            ratio,
            "4K",
            model="gpt-image-2",
            base_url="https://api.openai.com/v1",
        )
        == expected
    )


@pytest.mark.parametrize("ratio", SUPPORTED_IMAGE_ASPECT_RATIOS)
@pytest.mark.parametrize("resolution", ("1K", "2K", "4K"))
def test_gpt_image_2_size_stays_within_documented_limits(ratio, resolution):
    generator = make_generator()

    size = generator._map_openai_size(ratio, resolution, model="gpt-image-2")
    width, height = (int(value) for value in size.split("x", 1))

    assert max(width, height) <= generator.GPT_IMAGE_2_MAX_EDGE
    assert (
        generator.GPT_IMAGE_2_MIN_PIXELS
        <= width * height
        <= generator.GPT_IMAGE_2_MAX_PIXELS
    )
    assert width % 16 == 0
    assert height % 16 == 0


@pytest.mark.parametrize(
    ("ratio", "resolution", "expected"),
    [
        ("1:1", "4K", "4096x4096"),
        ("5:4", "2K", "2048x1638"),
        ("4:5", "2K", "1638x2048"),
        ("16:9", "4K", "4096x2304"),
        ("9:16", "1K", "576x1024"),
    ],
)
def test_other_openai_models_keep_legacy_arbitrary_size_mapping(
    ratio, resolution, expected
):
    generator = make_generator()

    assert (
        generator._map_openai_size(ratio, resolution, model="other-image-model")
        == expected
    )


def test_exact_resolution_overrides_aspect_ratio():
    generator = make_generator()

    assert (
        generator._map_openai_size("1:1", "1920 X 1088", model="gpt-image-2")
        == "1920x1088"
    )


def test_gpt_image_2_rejects_oversized_exact_resolution():
    generator = make_generator()

    with pytest.raises(ValueError, match="最长边不能超过 3840"):
        generator._map_openai_size("auto", "4096x2160", model="gpt-image-2")


def test_other_models_keep_large_exact_resolution():
    generator = make_generator()

    assert (
        generator._map_openai_size("1:1", "4096×2160", model="other-image-model")
        == "4096x2160"
    )


def test_chat_payload_includes_image_generation_compatibility_fields():
    generator = make_generator()

    payload = generator._build_request_payload(
        "draw this",
        [(b"jpeg-data", "image/jpeg")],
        "9:16",
        "2k",
        api_format="v1/chat/completions",
    )

    assert payload["aspect_ratio"] == "9:16"
    assert payload["image_size"] == "2K"
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "draw this"}
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_gemini_chat_payload_includes_newapi_google_image_config():
    generator = make_generator()

    payload = generator._build_request_payload(
        "draw this",
        [],
        "9:16",
        "4k",
        model="gemini-3.1-flash-image",
        api_format="v1/chat/completions",
    )

    assert payload["model"] == "gemini-3.1-flash-image"
    assert payload["extra_body"] == {
        "google": {
            "image_config": {
                "aspect_ratio": "9:16",
                "image_size": "4K",
            }
        }
    }


def test_chat_payload_omits_unspecified_compatibility_fields():
    generator = make_generator()

    payload = generator._build_request_payload(
        "draw this",
        [],
        "auto",
        None,
        api_format="v1/chat/completions",
    )

    assert "aspect_ratio" not in payload
    assert "image_size" not in payload
    assert "extra_body" not in payload


def test_gemini_3_payload_includes_configured_image_size():
    generator = make_generator(request_format="gemini")

    payload = generator._build_request_payload(
        "draw this",
        [(b"jpeg-data", "image/jpeg")],
        "9:16",
        "4K",
        model="gemini-3.1-flash-image",
        api_format="gemini",
    )

    assert payload["generationConfig"]["imageConfig"] == {
        "aspectRatio": "9:16",
        "imageSize": "4K",
    }


def test_chat_compatibility_field_rejection_is_detected():
    generator = make_generator()

    assert generator._is_unsupported_param_error(
        "Unknown parameter: 'aspect_ratio'.", "aspect_ratio"
    )
    assert generator._is_unsupported_param_error(
        "image_size: Extra inputs are not permitted", "image_size"
    )


def test_generations_payload_rejects_reference_images_instead_of_custom_fields():
    generator = make_generator()

    with pytest.raises(ValueError, match="不支持参考图"):
        generator._build_request_payload(
            "edit this",
            [(b"image", "image/png")],
            "1:1",
            "1K",
            api_format="v1/images/generations",
        )


def test_gpt_image_2_generations_payload_uses_relay_compatible_size():
    generator = make_generator(request_format="v1/images/generations")

    payload = generator._build_request_payload(
        "draw this",
        [],
        "16:9",
        "4K",
        model="gpt-image-2",
        api_format="v1/images/generations",
    )

    assert payload["model"] == "gpt-image-2"
    assert payload["size"] == "3840x2160"
    assert "response_format" not in payload


@pytest.mark.parametrize(
    ("model", "ignores_size"),
    [
        ("Z-Image-Turbo", True),
        ("z-image-turbo", True),
        ("画图/Z-Image-Turbo", True),
        ("relay/z-image-turbo-2.0", True),
        ("gpt-image-2", False),
        ("gemini-3.1-flash-image", False),
        ("", False),
        (None, False),
    ],
)
def test_model_ignores_size_param_only_matches_z_image_turbo(model, ignores_size):
    generator = make_generator()

    assert generator._model_ignores_size_param(model) is ignores_size


def test_z_image_turbo_generations_payload_omits_size():
    generator = make_generator(request_format="v1/images/generations")

    payload = generator._build_request_payload(
        "draw this",
        [],
        "auto",
        "4K",
        model="Z-Image-Turbo",
        api_format="v1/images/generations",
    )

    assert payload["model"] == "Z-Image-Turbo"
    assert "size" not in payload
    assert payload["prompt"] == "draw this"


def test_non_matching_model_generations_payload_keeps_size():
    generator = make_generator(request_format="v1/images/generations")

    payload = generator._build_request_payload(
        "draw this",
        [],
        "auto",
        "4K",
        model="some-other-image-model",
        api_format="v1/images/generations",
    )

    assert payload["model"] == "some-other-image-model"
    assert "size" in payload


@pytest.mark.asyncio
async def test_generate_skips_generations_with_reference_and_tries_next_model():
    generator = make_generator(
        request_format="v1/images/generations",
        provider_id="provider/main",
        fallback_models=[
            {
                "provider_id": "provider/fallback",
                "model": "reference-image-model",
                "api_keys": ["fallback-key"],
                "base_url": "https://fallback.example/v1",
                "api_request_format": "v1/chat/completions",
            }
        ],
    )
    generator._convert_image_format = AsyncMock(
        return_value=(b"converted", "image/jpeg")
    )
    generator._try_generate_with_current_key = AsyncMock(
        return_value=([b"result"], None)
    )

    result, error = await generator.generate_image(
        "edit this",
        [(b"source", "image/png")],
        "5:4",
        "2K",
    )

    assert result == [b"result"]
    assert error is None
    generator._try_generate_with_current_key.assert_awaited_once()
    request = generator._try_generate_with_current_key.await_args.kwargs
    assert request["model"] == "reference-image-model"
    assert request["base_url"] == "https://fallback.example/v1"
    assert request["api_format"] == "v1/chat/completions"


@pytest.mark.asyncio
async def test_each_model_uses_only_its_bound_request_format():
    generator = make_generator(
        request_format="v1/images/generations",
        provider_id="provider/main",
        fallback_models=[
            {
                "provider_id": "provider/fallback",
                "model": "fallback-model",
                "api_keys": ["fallback-key"],
                "base_url": "https://fallback.example/v1",
                "api_request_format": "v1/chat/completions",
            },
        ],
    )
    generator._try_generate_with_current_key = AsyncMock(
        side_effect=[(None, "primary failed"), ([b"result"], None)]
    )

    result, error = await generator.generate_image("draw this")

    assert result == [b"result"]
    assert error is None
    requests = generator._try_generate_with_current_key.await_args_list
    assert [(call.kwargs["model"], call.kwargs["api_format"]) for call in requests] == [
        ("image-model", "v1/images/generations"),
        ("fallback-model", "v1/chat/completions"),
    ]


@pytest.mark.asyncio
async def test_primary_failure_preserves_converted_references_for_fallback():
    generator = make_generator(
        request_format="gemini",
        provider_id="provider/main",
        fallback_models=[
            {
                "provider_id": "provider/fallback",
                "model": "fallback-model",
                "api_keys": ["fallback-key"],
                "base_url": "https://fallback.example/v1",
                "api_request_format": "v1/chat/completions",
            }
        ],
    )
    generator._convert_image_format = AsyncMock(
        return_value=(b"converted-reference", "image/jpeg")
    )
    received_references = []
    received_route_lists = []

    async def try_route(prompt, route_images, *args, **kwargs):
        received_references.append(list(route_images))
        received_route_lists.append(route_images)
        route_images.clear()
        if len(received_references) == 1:
            return None, "primary unavailable"
        return [b"result"], None

    generator._try_generate_with_current_key = AsyncMock(side_effect=try_route)

    result, error = await generator.generate_image(
        "edit this",
        [(b"source", "image/png")],
        "5:4",
        "2K",
    )

    assert result == [b"result"]
    assert error is None
    assert received_references == [
        [(b"converted-reference", "image/jpeg")],
        [(b"converted-reference", "image/jpeg")],
    ]
    assert received_route_lists[0] is not received_route_lists[1]
    assert generator.route_failures == ["主路由：primary unavailable"]
    generator._convert_image_format.assert_awaited_once_with(b"source", "image/png")


@pytest.mark.asyncio
async def test_all_route_failures_are_preserved_in_summary():
    generator = make_generator(
        request_format="gemini",
        provider_id="provider/main",
        fallback_models=[
            {
                "provider_id": "provider/fallback",
                "model": "fallback-model",
                "api_keys": ["fallback-key"],
                "base_url": "https://fallback.example/v1",
                "api_request_format": "v1/chat/completions",
            },
            {
                "provider_id": "provider/fallback-2",
                "model": "fallback-model-2",
                "api_keys": ["fallback-key-2"],
                "base_url": "https://fallback-2.example/v1",
                "api_request_format": "v1/images/generations",
            },
        ],
    )
    generator._try_generate_with_current_key = AsyncMock(
        side_effect=[
            (None, "API 错误 (502): primary unavailable"),
            (None, "生成超时 (超过 30 秒)"),
            (None, "API 请求频率超限"),
        ]
    )

    result, error = await generator.generate_image("draw this")

    assert result is None
    assert error is not None
    assert generator.route_failures == [
        "主路由：API 错误 (502): primary unavailable",
        "回退路由 1：生成超时 (超过 30 秒)",
        "回退路由 2：API 请求频率超限",
    ]
    assert (
        error.index("primary unavailable")
        < error.index("生成超时")
        < error.index("API 请求频率超限")
    )
    assert all(failure in error for failure in generator.route_failures)


@pytest.mark.asyncio
async def test_key_rotation_and_retries_keep_the_model_bound_format():
    generator = ImageGenerator(
        api_keys=["key-a", "key-b"],
        base_url="https://images.example/v1",
        model="image-model",
        provider_id="provider/main",
        api_request_format="v1/images/generations",
        max_retry_attempts=2,
    )
    generator._try_generate_with_current_key = AsyncMock(return_value=(None, "failed"))

    with patch(
        "astrbot_plugin_visora.image_generator.asyncio.sleep",
        AsyncMock(),
    ):
        result, error = await generator.generate_image("draw this")

    assert result is None
    assert "failed" in error
    requests = generator._try_generate_with_current_key.await_args_list
    assert len(requests) == 4
    assert {call.kwargs["api_format"] for call in requests} == {"v1/images/generations"}
    assert [call.args[0] for call in requests] == ["draw this"] * 4
    assert [call.kwargs["api_key"] for call in requests] == [
        "key-a",
        "key-b",
        "key-a",
        "key-b",
    ]


@pytest.mark.asyncio
async def test_fallback_route_can_run_when_primary_route_has_no_key():
    generator = ImageGenerator(
        api_keys=[],
        base_url="https://images.example/v1",
        model="image-model",
        provider_id="provider/main",
        api_request_format="v1/images/generations",
        fallback_models=[
            {
                "provider_id": "provider/fallback",
                "model": "fallback-model",
                "api_keys": ["fallback-key"],
                "base_url": "https://fallback.example/v1",
                "api_request_format": "v1/chat/completions",
            }
        ],
        max_retry_attempts=1,
    )
    generator._try_generate_with_current_key = AsyncMock(
        return_value=([b"result"], None)
    )

    result, error = await generator.generate_image("draw this")

    assert result == [b"result"]
    assert error is None
    generator._try_generate_with_current_key.assert_awaited_once()
    request = generator._try_generate_with_current_key.await_args.kwargs
    assert request["model"] == "fallback-model"
    assert request["api_key"] == "fallback-key"
    assert request["api_format"] == "v1/chat/completions"


@pytest.mark.asyncio
async def test_distinct_provider_ids_are_not_deduplicated_by_model_and_base_url():
    shared_route = {
        "model": "shared-model",
        "api_keys": ["shared-key"],
        "base_url": "https://shared.example/v1",
        "api_request_format": "v1/chat/completions",
    }
    generator = make_generator(
        provider_id="provider/main",
        fallback_models=[
            {**shared_route, "provider_id": "provider/first"},
            {**shared_route, "provider_id": "provider/first"},
            {**shared_route, "provider_id": "provider/second"},
        ],
    )
    generator._try_generate_with_current_key = AsyncMock(
        side_effect=[
            (None, "primary failed"),
            (None, "first provider failed"),
            ([b"result"], None),
        ]
    )

    result, error = await generator.generate_image("draw this")

    assert result == [b"result"]
    assert error is None
    requests = generator._try_generate_with_current_key.await_args_list
    assert [call.kwargs["model"] for call in requests] == [
        "image-model",
        "shared-model",
        "shared-model",
    ]
    assert [call.kwargs["api_key"] for call in requests] == [
        "test-key",
        "shared-key",
        "shared-key",
    ]


def _form_field_names(form: aiohttp.FormData) -> list[str]:
    return [field_options["name"] for field_options, _, _ in form._fields]


def _form_field_value(form: aiohttp.FormData, name: str):
    values = [
        value
        for field_options, _, value in form._fields
        if field_options["name"] == name
    ]
    assert len(values) == 1
    return values[0]


def test_gpt_image_2_edits_payload_uses_relay_compatible_size():
    generator = make_generator(request_format="v1/images/edits")

    form = generator._build_request_payload(
        "edit",
        [(b"one", "image/jpeg")],
        "1:1",
        "4K",
        model="gpt-image-2",
        api_format="v1/images/edits",
    )

    assert isinstance(form, aiohttp.FormData)
    assert _form_field_value(form, "model") == "gpt-image-2"
    assert _form_field_value(form, "size") == "2880x2880"
    assert "response_format" not in _form_field_names(form)


def test_edits_uses_plain_image_field_for_single_reference():
    generator = make_generator()

    form = generator._build_request_payload(
        "edit",
        [(b"one", "image/jpeg")],
        "1:1",
        "1K",
        api_format="v1/images/edits",
    )

    assert isinstance(form, aiohttp.FormData)
    assert _form_field_names(form).count("image") == 1
    assert "image[]" not in _form_field_names(form)


def test_edits_uses_consistent_array_fields_for_multiple_references():
    generator = make_generator()

    form = generator._build_request_payload(
        "edit",
        [(b"one", "image/jpeg"), (b"two", "image/png")],
        "5:4",
        "2K",
        api_format="v1/images/edits",
    )

    names = _form_field_names(form)
    assert names.count("image[]") == 2
    assert "image" not in names


@pytest.mark.parametrize(
    "api_format",
    (
        "gemini",
        "v1/chat/completions",
        "v1/images/generations",
        "v1/images/edits",
    ),
)
@pytest.mark.parametrize("prompt", (None, "", "   ", "\n\t"))
def test_payload_builder_rejects_empty_prompt(api_format, prompt):
    generator = make_generator(request_format=api_format)
    images = [(b"image", "image/png")] if api_format == "v1/images/edits" else []

    with pytest.raises(ValueError, match="提示词不能为空"):
        generator._build_request_payload(
            prompt,
            images,
            "1:1",
            "1K",
            api_format=api_format,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", (None, "", "   ", "\n\t"))
async def test_generate_rejects_empty_prompt_before_image_conversion_or_routes(prompt):
    generator = make_generator(
        request_format="gemini",
        fallback_models=[
            {
                "model": "fallback-model",
                "api_keys": ["fallback-key"],
                "base_url": "https://fallback.example/v1",
                "api_request_format": "v1/images/generations",
            }
        ],
    )
    generator._convert_image_format = AsyncMock()
    generator._try_generate_with_current_key = AsyncMock()

    result, error = await generator.generate_image(
        prompt,
        [(b"source", "image/png")],
    )

    assert result is None
    assert "提示词不能为空" in error
    generator._convert_image_format.assert_not_awaited()
    generator._try_generate_with_current_key.assert_not_awaited()


def test_native_and_openai_payloads_preserve_non_empty_prompt():
    generator = make_generator()
    expected = "雨夜白猫"

    native_payload = generator._build_request_payload(
        expected,
        [],
        "1:1",
        "2K",
        model="gemini-3.1-flash-image",
        api_format="gemini",
    )
    generations_payload = generator._build_request_payload(
        expected,
        [],
        "1:1",
        "2K",
        model="gpt-image-2",
        api_format="v1/images/generations",
    )
    chat_payload = generator._build_request_payload(
        expected,
        [],
        "1:1",
        "2K",
        api_format="v1/chat/completions",
    )
    edits_payload = generator._build_request_payload(
        expected,
        [(b"image", "image/png")],
        "1:1",
        "2K",
        api_format="v1/images/edits",
    )

    assert native_payload["contents"][0]["parts"][0] == {"text": expected}
    assert generations_payload["prompt"] == expected
    assert chat_payload["messages"][0]["content"][0] == {
        "type": "text",
        "text": expected,
    }
    assert _form_field_value(edits_payload, "prompt") == expected


@pytest.mark.asyncio
async def test_primary_retries_and_fallback_keep_normalized_prompt():
    generator = make_generator(
        request_format="gemini",
        fallback_models=[
            {
                "provider_id": "provider/fallback",
                "model": "gpt-image-2",
                "api_keys": ["fallback-key"],
                "base_url": "https://fallback.example/v1",
                "api_request_format": "v1/images/generations",
            }
        ],
    )
    generator._try_generate_with_current_key = AsyncMock(
        side_effect=[(None, "primary failed"), ([b"result"], None)]
    )

    result, error = await generator.generate_image("  雨夜白猫  ")

    assert result == [b"result"]
    assert error is None
    requests = generator._try_generate_with_current_key.await_args_list
    assert [call.args[0] for call in requests] == ["雨夜白猫", "雨夜白猫"]
    assert [call.kwargs["api_format"] for call in requests] == [
        "v1beta/models/{model}:generateContent",
        "v1/images/generations",
    ]
