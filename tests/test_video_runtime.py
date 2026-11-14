from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from astrbot_plugin_visora.llm_tools import VisoraVideoGenerationTool
from astrbot_plugin_visora.main import VisoraPlugin
from astrbot_plugin_visora.reference_images import ReferenceSelection
from astrbot_plugin_visora.video_generator import VideoGenerator


def make_event(*, sender_id: str = "111"):
    return SimpleNamespace(
        message_obj=SimpleNamespace(
            message=[], raw_message=SimpleNamespace(message=[])
        ),
        unified_msg_origin="origin",
        get_sender_id=lambda: sender_id,
        plain_result=lambda text: text,
    )


def make_video_tool_plugin(*, schedule_error: Exception | None = None):
    pending = []
    config = {
        "enable_economy": True,
        "cost_per_video": 13,
        "show_generating_notice": False,
    }

    def create_background_task(coro):
        if schedule_error is not None:
            raise schedule_error
        pending.append(coro)
        return None

    plugin = SimpleNamespace(
        _load_config=Mock(),
        enable_video_generation=True,
        _check_rate_limit=Mock(return_value=True),
        _config_get=Mock(
            side_effect=lambda key, default=None: config.get(key, default)
        ),
        data_manager=SimpleNamespace(
            get_points=AsyncMock(return_value=100),
            deduct_points=AsyncMock(return_value=True),
            add_points=AsyncMock(),
        ),
        points_name="星琼",
        max_requests_per_minute=3,
        default_video_duration=5,
        default_video_ratio="16:9",
        default_video_resolution="720p",
        _send_wait_message=AsyncMock(return_value=None),
        _recall_message=AsyncMock(),
        _collect_reference_selection=AsyncMock(),
        _download_reference_selection=AsyncMock(),
        _build_video_image_urls=AsyncMock(return_value=[]),
        _generate_and_send_video_async=AsyncMock(),
        create_background_task=Mock(side_effect=create_background_task),
    )
    return plugin, pending


@pytest.mark.asyncio
async def test_video_tool_reference_failure_does_not_charge_or_schedule():
    plugin, pending = make_video_tool_plugin()
    plugin._collect_reference_selection.return_value = ReferenceSelection(
        sources=("/tmp/stale.jpg", "https://cdn.example/fresh.jpg"),
        source_groups=(("/tmp/stale.jpg", "https://cdn.example/fresh.jpg"),),
        source_type="quoted",
        requested=True,
        reference_count=1,
    )
    plugin._download_reference_selection.return_value = []

    result = await VisoraVideoGenerationTool(plugin=plugin).call(
        {"event": make_event()},
        prompt="让图片动起来",
    )

    assert "未能读取" in result
    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.create_background_task.assert_not_called()
    assert pending == []


@pytest.mark.asyncio
async def test_video_command_reference_failure_does_not_deduct_or_downgrade():
    plugin = object.__new__(VisoraPlugin)
    plugin._config_get = Mock(
        side_effect=lambda key, default=None: {
            "enable_video_generation": True,
            "enable_economy": True,
            "cost_per_video": 13,
            "show_generating_notice": False,
        }.get(key, default)
    )
    plugin._check_rate_limit = Mock(return_value=True)
    plugin.max_requests_per_minute = 3
    plugin.max_reference_images_per_request = 4
    plugin.points_name = "星琼"
    plugin.data_manager = SimpleNamespace(
        get_points=AsyncMock(return_value=100),
        deduct_points=AsyncMock(return_value=True),
    )
    selection = ReferenceSelection(
        sources=("/tmp/stale.jpg",),
        source_groups=(("/tmp/stale.jpg",),),
        source_type="message",
        requested=True,
        reference_count=1,
    )
    plugin._collect_reference_selection = AsyncMock(return_value=selection)
    plugin._download_reference_selection = AsyncMock(return_value=[])
    plugin._generate_and_send_video_async = AsyncMock()

    results = [
        item
        async for item in plugin._core_generate_video(
            make_event(),
            "让这张图动起来",
        )
    ]

    assert len(results) == 1
    assert "未能读取" in results[0]
    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin._generate_and_send_video_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_video_tool_uses_selection_and_explicit_precharge_snapshot():
    plugin, pending = make_video_tool_plugin()
    selection = ReferenceSelection(
        sources=("https://cdn.example/avatar.jpg",),
        source_groups=(("https://cdn.example/avatar.jpg",),),
        source_type="avatar",
        requested=True,
        reference_count=1,
        avatar_user_ids=("222",),
    )
    plugin._collect_reference_selection.return_value = selection
    plugin._download_reference_selection.return_value = [(b"image", "image/jpeg")]
    plugin._build_video_image_urls.return_value = ["https://cdn.example/avatar.jpg"]

    result = await VisoraVideoGenerationTool(plugin=plugin).call(
        {"event": make_event()},
        prompt="挥手",
        user_ids=["999", "222"],
    )

    assert result == "同意"
    plugin._send_wait_message.assert_not_awaited()
    plugin._collect_reference_selection.assert_awaited_once_with(
        plugin._collect_reference_selection.await_args.args[0],
        explicit_user_ids=["999", "222"],
    )
    plugin.data_manager.deduct_points.assert_awaited_once_with("111", 13)
    assert len(pending) == 1
    await pending.pop()

    call = plugin._generate_and_send_video_async.await_args.kwargs
    assert call["images_data"] == [(b"image", "image/jpeg")]
    assert call["image_urls"] == ["https://cdn.example/avatar.jpg"]
    assert call["precharged_amount"] == 13
    assert call["reference_requested"] is True
    assert "skip_deduction" not in call
    assert "avatar_user_ids" not in call


@pytest.mark.asyncio
async def test_video_tool_background_creation_failure_refunds_precharge():
    plugin, pending = make_video_tool_plugin(schedule_error=RuntimeError("boom"))
    plugin._collect_reference_selection.return_value = ReferenceSelection()
    plugin._download_reference_selection.return_value = []

    result = await VisoraVideoGenerationTool(plugin=plugin).call(
        {"event": make_event()},
        prompt="雨夜街景",
    )

    assert "启动失败" in result
    assert "已返还" in result
    plugin.data_manager.add_points.assert_awaited_once_with("111", 13)
    assert pending == []


def make_video_runtime_plugin(*, slots: int = 1):
    plugin = object.__new__(VisoraPlugin)
    plugin._config_get = Mock(
        side_effect=lambda key, default=None: {
            "enable_economy": True,
            "cost_per_video": 13,
            "show_generation_stats": False,
        }.get(key, default)
    )
    plugin.data_manager = SimpleNamespace(
        deduct_points=AsyncMock(return_value=True),
        add_points=AsyncMock(),
        get_points=AsyncMock(return_value=100),
    )
    plugin.context = SimpleNamespace(send_message=AsyncMock())
    plugin.points_name = "星琼"
    plugin._generation_semaphore = asyncio.Semaphore(slots)
    plugin._load_config = Mock()
    plugin.video_model = "video-model"
    plugin.default_video_duration = 5
    plugin.default_video_ratio = "16:9"
    plugin.default_video_resolution = "720p"
    return plugin


@pytest.mark.asyncio
async def test_precharged_video_cancellation_refunds_once_and_reraises():
    plugin = make_video_runtime_plugin()
    generator = SimpleNamespace(
        api_keys=["key"],
        generate_video=AsyncMock(side_effect=asyncio.CancelledError()),
        close_session=AsyncMock(),
    )

    with patch(
        "astrbot_plugin_visora.main.create_video_generator",
        return_value=generator,
    ):
        with pytest.raises(asyncio.CancelledError):
            await plugin._generate_and_send_video_async(
                prompt="test",
                unified_msg_origin="origin",
                user_id="111",
                task_id="video-cancel",
                precharged_amount=13,
            )

    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.data_manager.add_points.assert_awaited_once_with("111", 13)
    generator.close_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_precharged_video_waiting_cancellation_refunds_once():
    plugin = make_video_runtime_plugin(slots=0)
    task = asyncio.create_task(
        plugin._generate_and_send_video_async(
            prompt="test",
            unified_msg_origin="origin",
            user_id="111",
            task_id="video-wait-cancel",
            precharged_amount=13,
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    plugin.data_manager.add_points.assert_awaited_once_with("111", 13)


def test_openai_chat_video_payload_uses_only_one_reference_representation():
    generator = VideoGenerator(
        api_keys=["key"],
        base_url="https://example.com/v1",
        model="video-model",
    )
    payload = generator._build_openai_chat_payload(
        "test",
        [(b"image", "image/jpeg")],
        ["https://cdn.example/same-image.jpg"],
        5,
        "16:9",
        "720p",
        "video-model",
    )

    image_parts = payload["messages"][0]["content"][1:]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", (None, "", "   ", "\n\t"))
async def test_video_command_rejects_empty_prompt_without_side_effects(prompt):
    plugin = object.__new__(VisoraPlugin)
    plugin._config_get = Mock(
        side_effect=lambda key, default=None: {
            "enable_video_generation": True,
        }.get(key, default)
    )
    plugin._check_rate_limit = Mock()
    plugin.data_manager = SimpleNamespace(get_points=AsyncMock())
    plugin._collect_reference_selection = AsyncMock()
    plugin._download_reference_selection = AsyncMock()
    plugin._build_video_image_urls = AsyncMock()
    plugin._send_wait_message = AsyncMock()
    plugin._generate_and_send_video_async = AsyncMock()
    event = make_event()

    results = [item async for item in plugin._core_generate_video(event, prompt)]

    assert len(results) == 1
    assert "请提供视频生成提示词" in results[0]
    assert plugin._config_get.call_args_list == [
        (("enable_video_generation", True),),
    ]
    plugin._check_rate_limit.assert_not_called()
    plugin.data_manager.get_points.assert_not_awaited()
    plugin._collect_reference_selection.assert_not_awaited()
    plugin._download_reference_selection.assert_not_awaited()
    plugin._build_video_image_urls.assert_not_awaited()
    plugin._send_wait_message.assert_not_awaited()
    plugin._generate_and_send_video_async.assert_not_awaited()
