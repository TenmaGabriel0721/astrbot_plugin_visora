from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot_plugin_visora.background_tasks import (
    ChargeRejectedError,
    QueueFullError,
    TaskSpec,
    build_scope_key,
)
from astrbot_plugin_visora.llm_tools import (
    SILENT_IMAGE_TOOL_FAILURE,
    VisoraImageGenerationTool,
    VisoraImageTaskStatusTool,
)
from astrbot_plugin_visora.reference_images import ReferenceSelection


def make_event(
    *,
    user_id: str = "user-1",
    umo: str = "platform:GroupMessage:group-1",
    message_id: str = "message-1",
) -> SimpleNamespace:
    extras = {}
    message_obj = SimpleNamespace(
        message=[],
        raw_message=SimpleNamespace(message=[]),
        message_id=message_id,
        self_id="bot-1",
        group_id="group-1",
        sender=SimpleNamespace(user_id=user_id, nickname="Alice"),
    )
    return SimpleNamespace(
        message_obj=message_obj,
        message_str="请生成一张图",
        unified_msg_origin=umo,
        get_platform_id=lambda: "platform",
        get_message_type=lambda: SimpleNamespace(value="GroupMessage"),
        get_session_id=lambda: "group-1",
        get_group_id=lambda: "group-1",
        get_sender_id=lambda: user_id,
        get_sender_name=lambda: "Alice",
        get_self_id=lambda: "bot-1",
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=lambda key, value: extras.__setitem__(key, value),
    )


def make_manager(
    *,
    task_id: str = "task-1",
    state: str = "queued",
    created: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        accepting=True,
        start=AsyncMock(),
        accept=AsyncMock(
            return_value=SimpleNamespace(
                task_id=task_id,
                state=state,
                created=created,
            )
        ),
        get=AsyncMock(),
        format_scope_status=AsyncMock(return_value="当前生图任务：task-1 排队中"),
    )


def make_plugin(
    manager: SimpleNamespace,
    *,
    conversation_id: str = "cid-1",
    selection: ReferenceSelection | None = None,
    frozen_images: list[tuple[bytes, str]] | None = None,
) -> SimpleNamespace:
    config = {
        "enable_economy": True,
        "cost_per_image": 7,
        "default_resolution": "2K",
        "custom_resolution": "1920x1080",
        "show_generating_notice": False,
    }
    return SimpleNamespace(
        model="image-model",
        default_resolution="2K",
        default_aspect_ratio="1:1",
        points_name="星琼",
        max_requests_per_minute=3,
        _background_runtime_id="runtime-1",
        _background_session_epochs={},
        _register_background_acceptance=Mock(),
        _release_background_acceptance=Mock(),
        background_task_manager=manager,
        context=SimpleNamespace(
            conversation_manager=SimpleNamespace(
                get_curr_conversation_id=AsyncMock(return_value=conversation_id)
            )
        ),
        data_manager=SimpleNamespace(
            get_points=AsyncMock(return_value=100),
            deduct_points=AsyncMock(),
            add_points=AsyncMock(),
        ),
        _load_config=Mock(),
        _check_rate_limit=Mock(return_value=True),
        _config_get=Mock(
            side_effect=lambda key, default=None: config.get(key, default)
        ),
        _collect_reference_selection=AsyncMock(
            return_value=selection or ReferenceSelection()
        ),
        _download_reference_selection=AsyncMock(return_value=list(frozen_images or [])),
        _recall_message=AsyncMock(),
        create_background_task=Mock(),
    )


@pytest.mark.asyncio
async def test_general_image_tool_admits_task_spec_with_frozen_image_bytes():
    manager = make_manager(task_id="task-general")
    selection = ReferenceSelection(
        sources=("https://img/reference.png",),
        source_type="avatar",
        requested=True,
        reference_count=1,
        avatar_user_ids=("user-2",),
    )
    frozen_images = [(b"frozen-reference", "image/png")]
    plugin = make_plugin(
        manager,
        selection=selection,
        frozen_images=frozen_images,
    )
    event = make_event()

    result = await VisoraImageGenerationTool(plugin=plugin).call(
        {"event": event},
        prompt="雨夜街道上的白猫",
        user_ids=["user-2"],
        aspect_ratio="16:9",
    )

    assert "task-general" in result
    manager.start.assert_not_awaited()
    manager.accept.assert_awaited_once()
    spec, input_images = manager.accept.await_args.args
    assert isinstance(spec, TaskSpec)
    assert spec.scope_key == build_scope_key(
        event.unified_msg_origin,
        "user-1",
        "cid-1",
    )
    assert spec.umo == event.unified_msg_origin
    assert spec.user_id == "user-1"
    assert spec.conversation_id == "cid-1"
    assert spec.prompt == "雨夜街道上的白猫"
    assert spec.mode == "img2img"
    assert spec.aspect_ratio == "16:9"
    assert spec.resolution == "2K"
    assert spec.charge_amount == 7
    assert spec.source_message_id == "message-1"
    assert spec.request_fingerprint
    assert spec.metadata == {
        "reply_mode": "combined",
        "ack_key": spec.request_fingerprint,
        "user_prompt": "请生成一张图",
        "effective_prompt_summary": "雨夜街道上的白猫",
        "reference_count": 1,
        "avatar_user_ids": ["user-2"],
        "delivery_target": {
            "platform_id": "platform",
            "message_type": "GroupMessage",
            "session_id": "group-1",
            "group_id": "group-1",
            "self_id": "bot-1",
            "sender_name": "Alice",
            "runtime_id": "runtime-1",
            "session_epoch": 0,
        },
    }
    assert event.get_extra("_visora_image_ack_task_id") == "task-general"
    assert event.get_extra("_visora_image_ack_key") == spec.request_fingerprint
    plugin._register_background_acceptance.assert_called_once_with(
        spec.request_fingerprint
    )
    assert input_images == frozen_images
    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.create_background_task.assert_not_called()


@pytest.mark.asyncio
async def test_general_image_tool_bounds_persisted_completion_prompt_metadata():
    manager = make_manager(task_id="task-bounded")
    plugin = make_plugin(manager)
    event = make_event()
    event.message_str = "请\n" + ("画" * 700) + "\x00"
    prompt = "细节 " * 500

    result = await VisoraImageGenerationTool(plugin=plugin).call(
        {"event": event},
        prompt=prompt,
    )

    assert "task-bounded" in result
    spec = manager.accept.await_args.args[0]
    assert len(spec.metadata["user_prompt"]) <= 512
    assert len(spec.metadata["effective_prompt_summary"]) <= 1200
    assert "\n" not in spec.metadata["user_prompt"]
    assert "\x00" not in spec.metadata["user_prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (QueueFullError("full"), "队列已满"),
        (ChargeRejectedError("charge rejected"), "星琼不足"),
    ],
)
async def test_general_image_tool_reports_admission_rejection(error, expected):
    manager = make_manager()
    manager.accept.side_effect = error
    plugin = make_plugin(manager)

    result = await VisoraImageGenerationTool(plugin=plugin).call(
        {"event": make_event()},
        prompt="一只橘猫",
    )

    assert expected in result
    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.create_background_task.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["config", "references", "admission"])
async def test_general_image_tool_keeps_internal_failures_out_of_the_reply(stage):
    manager = make_manager()
    plugin = make_plugin(manager)
    diagnostic = "API error (502), private diagnostic sentinel"
    target = {
        "config": plugin._load_config,
        "references": plugin._collect_reference_selection,
        "admission": manager.accept,
    }[stage]
    target.side_effect = RuntimeError(diagnostic)

    result = await VisoraImageGenerationTool(plugin=plugin).call(
        {"event": make_event()}, prompt="一只橘猫"
    )

    assert result == SILENT_IMAGE_TOOL_FAILURE
    assert diagnostic not in result
    plugin.data_manager.deduct_points.assert_not_awaited()
    if stage == "admission":
        plugin._release_background_acceptance.assert_called_once()


@pytest.mark.asyncio
async def test_duplicate_admission_returns_the_same_task_id():
    manager = make_manager(task_id="task-dedup")
    manager.accept.side_effect = [
        SimpleNamespace(task_id="task-dedup", state="queued", created=True),
        SimpleNamespace(task_id="task-dedup", state="queued", created=False),
    ]
    plugin = make_plugin(manager)
    tool = VisoraImageGenerationTool(plugin=plugin)
    event = make_event(message_id="stable-message")

    first = await tool.call({"event": event}, prompt="湖边日落")
    duplicate = await tool.call({"event": event}, prompt="湖边日落")

    assert "task-dedup" in first
    assert "task-dedup" in duplicate
    assert "已经受理" in duplicate
    first_spec = manager.accept.await_args_list[0].args[0]
    duplicate_spec = manager.accept.await_args_list[1].args[0]
    assert first_spec.request_fingerprint == duplicate_spec.request_fingerprint
    assert first_spec.scope_key == duplicate_spec.scope_key
    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.create_background_task.assert_not_called()


@pytest.mark.asyncio
async def test_status_tool_limits_task_lookup_to_current_user_umo_and_conversation():
    manager = make_manager()
    task_scope = build_scope_key(
        "platform:GroupMessage:group-1",
        "user-1",
        "cid-1",
    )
    manager.get.return_value = SimpleNamespace(
        task_id="task-private",
        scope_key=task_scope,
        umo="platform:GroupMessage:group-1",
        user_id="user-1",
        state="queued",
        delivery_state="not_started",
        error_message="",
    )
    plugin = make_plugin(manager)
    conversation_lookup = plugin.context.conversation_manager.get_curr_conversation_id
    tool = VisoraImageTaskStatusTool(plugin=plugin)

    allowed = await tool.call(
        {"event": make_event()},
        task_id="task-private",
    )
    assert "任务 task-private" in allowed
    assert "排队中" in allowed

    denied_user = await tool.call(
        {"event": make_event(user_id="user-2")},
        task_id="task-private",
    )
    assert "未找到当前用户在本会话中的该任务" in denied_user

    denied_umo = await tool.call(
        {"event": make_event(umo="platform:GroupMessage:group-2")},
        task_id="task-private",
    )
    assert "未找到当前用户在本会话中的该任务" in denied_umo

    conversation_lookup.return_value = "cid-2"
    denied_conversation = await tool.call(
        {"event": make_event()},
        task_id="task-private",
    )
    assert "未找到当前用户在本会话中的该任务" in denied_conversation


@pytest.mark.asyncio
async def test_status_tool_lists_only_the_current_scope():
    manager = make_manager()
    plugin = make_plugin(manager, conversation_id="cid-current")
    event = make_event(user_id="user-current")

    result = await VisoraImageTaskStatusTool(plugin=plugin).call({"event": event})

    assert result == "当前生图任务：task-1 排队中"
    manager.format_scope_status.assert_awaited_once_with(
        build_scope_key(
            event.unified_msg_origin,
            "user-current",
            "cid-current",
        ),
        limit=5,
    )
    manager.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_status_tool_reports_failure_without_provider_diagnostics():
    manager = make_manager()
    plugin = make_plugin(manager)
    event = make_event()
    manager.get.return_value = SimpleNamespace(
        task_id="failed-task",
        scope_key=build_scope_key(
            event.unified_msg_origin, event.get_sender_id(), "cid-1"
        ),
        umo=event.unified_msg_origin,
        user_id=event.get_sender_id(),
        state="failed",
        delivery_state="not_started",
        error_message="API error (502), private diagnostic sentinel",
    )

    result = await VisoraImageTaskStatusTool(plugin=plugin).call(
        {"event": event}, task_id="failed-task"
    )

    assert result == "任务 failed-task: 失败"
    assert "502" not in result
