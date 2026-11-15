from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import astrbot.api.message_components as Comp
import pytest
from astrbot_plugin_visora.main import VisoraPlugin
from astrbot_plugin_visora.presets import normalize_presets
from astrbot_plugin_visora.reference_images import ReferenceSelection


def make_preset(mode="img2img", **overrides):
    return {
        "__template_key": mode,
        "name": "海报",
        "prompt": "minimalist poster",
        "aspect_ratio": "3:4",
        "resolution": "4K",
        **overrides,
    }


def make_event(content="~海报", *, raw_content=None):
    raw = content if raw_content is None else raw_content
    return SimpleNamespace(
        message_str=content,
        message_obj=SimpleNamespace(message_str=raw, message=[Comp.Plain(text=raw)]),
        get_sender_id=lambda: "user-1",
        get_self_id=lambda: "bot-1",
        unified_msg_origin="official:GroupMessage:group-1",
        plain_result=lambda text: text,
        stop_event=Mock(),
        is_at_or_wake_command=True,
    )


def make_plugin(*presets):
    pending = []
    plugin = VisoraPlugin.__new__(VisoraPlugin)
    config = {"enable_economy": False, "show_generating_notice": False}
    plugin._config_get = Mock(
        side_effect=lambda key, default=None: config.get(key, default)
    )
    plugin.default_aspect_ratio = "1:1"
    plugin.default_resolution = "2K"
    plugin.presets = {item["name"]: item for item in presets}
    plugin.command_alias = ["生图"]
    plugin.video_command_alias = ["生视频"]
    plugin.max_reference_images_per_request = 4
    plugin._check_rate_limit = Mock(return_value=True)
    plugin.data_manager = SimpleNamespace(get_points=AsyncMock(return_value=100))
    plugin._collect_reference_selection = AsyncMock(return_value=ReferenceSelection())
    plugin._send_wait_message = AsyncMock()
    plugin._download_image = AsyncMock()
    plugin._download_reference_source_groups = AsyncMock(
        return_value=[(b"reference", "image/png")]
    )
    plugin._generate_and_send_image_async = AsyncMock()
    plugin.create_background_task = Mock(side_effect=pending.append)
    plugin.context = SimpleNamespace(
        get_config=Mock(return_value={"wake_prefix": ["~", "～"]}),
        send_message=AsyncMock(),
    )
    return plugin, pending


def test_legacy_templates_preserve_colons_multiline_and_json_settings():
    presets = normalize_presets(
        [
            "改图:first line\nhttps://example.com:8443/reference",
            '海报:{"mode":"txt2img","prompt":"poster","aspect_ratio":"16:9","resolution":"2K"}',
        ],
        strict=True,
    )
    assert presets[0]["__template_key"] == "img2img"
    assert presets[0]["prompt"] == "first line\nhttps://example.com:8443/reference"
    assert presets[1] == make_preset(
        "txt2img", prompt="poster", aspect_ratio="16:9", resolution="2K"
    )


@pytest.mark.parametrize(
    "items",
    [
        [make_preset(prompt=" ")],
        [make_preset("video")],
        [make_preset(aspect_ratio="square")],
        [make_preset(resolution="8K")],
        [make_preset(name="bad:name")],
        [make_preset(name="Poster"), make_preset("txt2img", name="poster")],
    ],
)
def test_invalid_or_ambiguous_presets_cannot_be_saved(items):
    with pytest.raises(ValueError):
        normalize_presets(items, strict=True)


@pytest.mark.asyncio
async def test_image_preset_without_picture_stops_before_balance_notice_or_task():
    plugin, pending = make_plugin(make_preset())
    event = make_event()
    results = [r async for r in plugin._core_generate(event, "海报")]
    assert len(results) == 1 and "请同时发送图片或引用" in results[0]
    plugin._collect_reference_selection.assert_awaited_once_with(
        event, include_avatars=False
    )
    plugin._check_rate_limit.assert_not_called()
    plugin.data_manager.get_points.assert_not_awaited()
    plugin._download_image.assert_not_awaited()
    plugin._send_wait_message.assert_not_awaited()
    plugin._generate_and_send_image_async.assert_not_awaited()
    assert pending == []


@pytest.mark.asyncio
async def test_image_preset_uses_uploaded_reference_and_appends_extra_requirements():
    plugin, pending = make_plugin(make_preset())
    plugin._collect_reference_selection.return_value = ReferenceSelection(
        sources=("https://example.com/photo.png",),
        source_type="message",
        requested=True,
        reference_count=1,
    )
    assert [r async for r in plugin._core_generate(make_event(), "海报 蓝色背景")] == []
    assert len(pending) == 1
    await pending.pop()
    call = plugin._generate_and_send_image_async.await_args.kwargs
    assert call["mode"] == "img2img"
    assert call["prompt"] == "minimalist poster 蓝色背景"
    assert call["images_data"] == [(b"reference", "image/png")]
    assert (call["aspect_ratio"], call["resolution"]) == ("3:4", "4K")


@pytest.mark.asyncio
async def test_unreadable_image_preset_never_falls_back_to_text_generation():
    plugin, pending = make_plugin(make_preset())
    plugin._collect_reference_selection.return_value = ReferenceSelection(
        source_type="message", requested=True, reference_count=1
    )
    assert [r async for r in plugin._core_generate(make_event(), "海报")] == []
    await pending.pop()
    plugin._generate_and_send_image_async.assert_not_awaited()
    plugin.context.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_text_preset_ignores_pictures_and_uses_its_text_route():
    plugin, pending = make_plugin(make_preset("txt2img"))
    assert [
        r
        async for r in plugin._core_generate(
            make_event(), "海报 一只猫", image_urls=["https://example.com/photo.png"]
        )
    ] == []
    await pending.pop()
    call = plugin._generate_and_send_image_async.await_args.kwargs
    assert call["mode"] == "txt2img"
    assert call["images_data"] is None
    assert call["prompt"] == "minimalist poster 一只猫"
    plugin._collect_reference_selection.assert_not_awaited()
    plugin._download_reference_source_groups.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("require_prefix", [True, False])
async def test_bot_mention_or_private_wakeup_cannot_trigger_bare_preset(require_prefix):
    plugin, _ = make_plugin(make_preset())
    plugin._config_get = Mock(return_value=require_prefix)
    plugin._core_generate = AsyncMock()
    event = make_event("海报")
    assert [r async for r in plugin.on_message(event)] == []
    plugin._core_generate.assert_not_awaited()
    event.stop_event.assert_not_called()


@pytest.mark.asyncio
async def test_bare_command_alias_cannot_bypass_the_preset_prefix_requirement():
    plugin, pending = make_plugin(make_preset("txt2img"))
    plugin.command_alias = ["画图"]
    results = [r async for r in plugin.on_message(make_event("画图 海报"))]
    assert len(results) == 1 and "使用预设须带指令前缀" in results[0]
    plugin._check_rate_limit.assert_not_called()
    plugin.data_manager.get_points.assert_not_awaited()
    assert pending == []


@pytest.mark.asyncio
async def test_shortcut_recovers_the_active_chat_custom_prefix_after_wake_processing():
    plugin, _ = make_plugin(make_preset())
    plugin.context.get_config.return_value = {"wake_prefix": ["!!"]}
    calls = []

    async def generate(event, prompt, preset_name):
        calls.append((prompt, preset_name))
        yield "accepted"

    plugin._core_generate = generate
    event = make_event("海报", raw_content="!!海报 蓝色背景")
    assert [r async for r in plugin.on_message(event)] == ["accepted"]
    assert calls == [("海报 蓝色背景", "海报")]
    plugin.context.get_config.assert_called_with(event.unified_msg_origin)


@pytest.mark.asyncio
async def test_shortcut_rejects_a_prefix_not_enabled_in_the_active_chat():
    plugin, _ = make_plugin(make_preset())
    plugin._core_generate = AsyncMock()
    event = make_event("/海报")
    assert [r async for r in plugin.on_message(event)] == []
    plugin._core_generate.assert_not_awaited()
