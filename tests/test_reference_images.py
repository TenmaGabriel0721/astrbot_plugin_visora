from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import astrbot.api.message_components as Comp
import pytest
from astrbot_plugin_visora.llm_tools import VisoraImageGenerationTool
from astrbot_plugin_visora.main import VisoraPlugin
from astrbot_plugin_visora.reference_images import (
    ReferenceSelection,
    collect_reference_selection,
)


def make_event(components, *, raw_segments=None, self_id="999", sender_id="111"):
    raw = SimpleNamespace(message=raw_segments or [])
    message_obj = SimpleNamespace(
        message=components,
        raw_message=raw,
        self_id=self_id,
        sender=SimpleNamespace(user_id=sender_id),
    )
    return SimpleNamespace(
        message_obj=message_obj,
        get_self_id=lambda: self_id,
        get_sender_id=lambda: sender_id,
    )


@pytest.mark.asyncio
async def test_current_images_win_over_quote_and_avatar():
    event = make_event(
        [
            Comp.Image(file="https://img/current.jpg", url="https://img/current.jpg"),
            Comp.Reply(
                id="42",
                chain=[
                    Comp.Image(
                        file="https://img/quoted.jpg", url="https://img/quoted.jpg"
                    )
                ],
            ),
            Comp.At(qq="222"),
        ]
    )
    extractor = AsyncMock(return_value=["https://img/remote-quoted.jpg"])
    with patch(
        "astrbot_plugin_visora.reference_images.extract_quoted_message_images",
        extractor,
    ):
        selection = await collect_reference_selection(
            event,
            explicit_user_ids=["333"],
            max_images=4,
            max_avatars=4,
        )

    assert selection.source_type == "message"
    assert selection.sources == ("https://img/current.jpg",)
    assert selection.avatar_user_ids == ()
    extractor.assert_not_awaited()


@pytest.mark.asyncio
async def test_quote_uses_remote_parser_and_falls_back_to_chain():
    event = make_event(
        [
            Comp.Reply(
                id="42",
                chain=[Comp.Image(file="/tmp/stale.jpg", path="/tmp/stale.jpg")],
            )
        ]
    )
    with patch(
        "astrbot_plugin_visora.reference_images.extract_quoted_message_images",
        AsyncMock(return_value=["https://img/fresh.jpg"]),
    ):
        fresh = await collect_reference_selection(event)
    assert fresh.sources == ("/tmp/stale.jpg", "https://img/fresh.jpg")
    assert fresh.candidate_groups == (("/tmp/stale.jpg", "https://img/fresh.jpg"),)
    assert fresh.reference_count == 1

    with patch(
        "astrbot_plugin_visora.reference_images.extract_quoted_message_images",
        AsyncMock(return_value=[]),
    ):
        fallback = await collect_reference_selection(event)
    assert fallback.sources == ("/tmp/stale.jpg",)
    assert fallback.requested is True


@pytest.mark.asyncio
async def test_quote_local_jpeg_and_remote_original_are_one_logical_image():
    local_jpeg = "/tmp/quoted-converted.jpg"
    remote_original = "https://cdn.example/quoted-original.jpg"
    event = make_event(
        [
            Comp.Reply(
                id="42",
                chain=[Comp.Image(file=local_jpeg, path=local_jpeg, url=local_jpeg)],
            )
        ]
    )

    with patch(
        "astrbot_plugin_visora.reference_images.extract_quoted_message_images",
        AsyncMock(return_value=[local_jpeg, remote_original]),
    ):
        selection = await collect_reference_selection(event)

    assert selection.reference_count == 1
    assert selection.candidate_groups == ((local_jpeg, remote_original),)


@pytest.mark.asyncio
async def test_mentions_filter_bot_all_invalid_duplicates_and_apply_limit():
    event = make_event(
        [
            Comp.At(qq="all"),
            Comp.At(qq="999"),
            Comp.At(qq="abc"),
            Comp.At(qq="222"),
            Comp.At(qq=222),
            Comp.At(qq="333"),
            Comp.At(qq="444"),
        ]
    )
    selection = await collect_reference_selection(
        event,
        explicit_user_ids=["999", "bad", "222"],
        max_images=4,
        max_avatars=2,
    )

    assert selection.source_type == "avatar"
    assert selection.avatar_user_ids == ("222", "333")
    assert selection.reference_count == 2

    string_selection = await collect_reference_selection(
        make_event([]),
        explicit_user_ids="222",
        max_images=4,
        max_avatars=2,
    )
    assert string_selection.avatar_user_ids == ("222",)


@pytest.mark.asyncio
async def test_stale_component_path_keeps_raw_url_candidate():
    event = make_event(
        [Comp.Image(file="/tmp/missing.jpg", path="/tmp/missing.jpg")],
        raw_segments=[
            {
                "type": "image",
                "data": {
                    "file": "/tmp/missing.jpg",
                    "path": "/tmp/missing.jpg",
                    "url": "https://cdn.example/fresh.jpg",
                },
            }
        ],
    )
    selection = await collect_reference_selection(event, max_images=4)

    assert selection.source_type == "message"
    assert selection.sources == (
        "/tmp/missing.jpg",
        "https://cdn.example/fresh.jpg",
    )
    assert selection.candidate_groups == (
        ("/tmp/missing.jpg", "https://cdn.example/fresh.jpg"),
    )


def test_downloaded_image_content_is_deduped_and_limited():
    plugin = object.__new__(VisoraPlugin)
    plugin.max_reference_images_per_request = 2
    first = (b"same", "image/png")
    duplicate = (b"same", "image/jpeg")
    second = (b"second", "image/png")
    third = (b"third", "image/png")

    result = plugin._dedupe_reference_image_data([first, duplicate, second, third])

    assert result == [first, second]


@pytest.mark.asyncio
async def test_download_tries_fallback_candidates_then_dedupes_and_limits():
    plugin = object.__new__(VisoraPlugin)
    plugin.max_reference_images_per_request = 2
    plugin._download_image = AsyncMock(
        side_effect=[
            None,
            (b"first", "image/jpeg"),
            (b"first", "image/jpeg"),
            (b"second", "image/png"),
            (b"third", "image/png"),
        ]
    )

    result = await plugin._download_reference_images(
        ["stale", "fresh", "duplicate", "second", "third"]
    )

    assert result == [
        (b"first", "image/jpeg"),
        (b"second", "image/png"),
    ]


@pytest.mark.asyncio
async def test_candidate_group_stops_after_first_success_and_returns_one_image():
    plugin = object.__new__(VisoraPlugin)
    plugin.max_reference_images_per_request = 4
    plugin._download_image = AsyncMock(
        side_effect=[None, (b"same-logical-image", "image/jpeg")]
    )

    result = await plugin._download_reference_source_groups(
        [
            (
                "/tmp/stale.jpg",
                "https://cdn.example/image.jpg",
                "37A70A8DFFFC0557D7683D13AC14222B.jpg",
            )
        ]
    )

    assert result == [(b"same-logical-image", "image/jpeg")]
    assert [item.args[0] for item in plugin._download_image.await_args_list] == [
        "/tmp/stale.jpg",
        "https://cdn.example/image.jpg",
    ]


@pytest.mark.asyncio
async def test_llm_tool_reference_failure_does_not_charge_or_start_generation():
    event = make_event([])
    event.unified_msg_origin = "origin"
    data_manager = SimpleNamespace(
        get_points=AsyncMock(return_value=100),
        deduct_points=AsyncMock(return_value=True),
    )
    plugin = SimpleNamespace(
        _load_config=Mock(),
        _check_rate_limit=Mock(return_value=True),
        _config_get=Mock(
            side_effect=lambda key, default=None: {
                "enable_economy": True,
                "cost_per_image": 7,
                "show_generating_notice": True,
            }.get(key, default)
        ),
        data_manager=data_manager,
        points_name="星琼",
        max_requests_per_minute=3,
        default_resolution="2K",
        default_aspect_ratio="1:1",
        _collect_reference_selection=AsyncMock(
            return_value=ReferenceSelection(
                sources=("/tmp/stale.jpg",),
                source_type="quoted",
                requested=True,
                reference_count=1,
            )
        ),
        _send_wait_message=AsyncMock(return_value=123),
        _download_reference_selection=AsyncMock(return_value=[]),
        _recall_message=AsyncMock(),
        create_background_task=Mock(),
    )
    tool = VisoraImageGenerationTool(plugin=plugin)

    result = await tool.call({"event": event}, prompt="修改这张图")

    assert "未能读取" in result
    data_manager.deduct_points.assert_not_awaited()
    plugin._send_wait_message.assert_not_awaited()
    plugin.create_background_task.assert_not_called()
