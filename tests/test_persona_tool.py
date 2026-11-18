from __future__ import annotations

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import astrbot.api.message_components as Comp
import pytest
from astrbot_plugin_visora.background_tasks import (
    ChargeRejectedError,
    TaskSpec,
)
from astrbot_plugin_visora.llm_tools import (
    PERSONA_ALLOWED_ASPECT_RATIOS,
    SILENT_IMAGE_TOOL_FAILURE,
    VisoraImageGenerationTool,
    VisoraPersonaPhotoTool,
    build_persona_photo_prompt,
    deduct_points_cancellation_safe,
    get_configured_resolution,
    get_explicit_user_resolution,
)
from astrbot_plugin_visora.persona_profiles import PersonaProfileStore
from astrbot_plugin_visora.reference_images import (
    ReferenceSelection,
    collect_reference_selection,
)
from PIL import Image


def make_event(
    components=None,
    *,
    message_text="",
    sender_id="111",
    self_id="999",
):
    extras = {}
    message_obj = SimpleNamespace(
        message=list(components or []),
        raw_message=SimpleNamespace(message=[]),
        message_id="message-persona",
        self_id=self_id,
        group_id="",
        sender=SimpleNamespace(user_id=sender_id, nickname="Alice"),
    )
    return SimpleNamespace(
        message_obj=message_obj,
        message_str=message_text,
        unified_msg_origin="origin",
        get_platform_id=lambda: "platform",
        get_message_type=lambda: SimpleNamespace(value="FriendMessage"),
        get_session_id=lambda: sender_id,
        get_group_id=lambda: "",
        get_sender_id=lambda: sender_id,
        get_sender_name=lambda: "Alice",
        get_self_id=lambda: self_id,
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=lambda key, value: extras.__setitem__(key, value),
    )


def make_plugin(*, economy=True, deduct_result=True, persona_resolution="2K"):
    pending = []
    config = {
        "enable_economy": economy,
        "cost_per_image": 7,
        "show_generating_notice": False,
        "persona_photo_default_resolution": persona_resolution,
    }
    accept = AsyncMock(
        return_value=SimpleNamespace(
            task_id="task-persona",
            state="queued",
            created=True,
        )
    )
    if not deduct_result:
        accept.side_effect = ChargeRejectedError("charge rejected")
    plugin = SimpleNamespace(
        _load_config=Mock(),
        _check_rate_limit=Mock(return_value=True),
        _config_get=Mock(
            side_effect=lambda key, default=None: config.get(key, default)
        ),
        data_manager=SimpleNamespace(
            get_points=AsyncMock(return_value=100),
            deduct_points=AsyncMock(return_value=deduct_result),
            add_points=AsyncMock(),
        ),
        points_name="星琼",
        max_requests_per_minute=3,
        _background_runtime_id="runtime-1",
        _background_session_epochs={},
        _register_background_acceptance=Mock(),
        _release_background_acceptance=Mock(),
        context=SimpleNamespace(
            conversation_manager=SimpleNamespace(
                get_curr_conversation_id=AsyncMock(return_value="cid-persona")
            )
        ),
        background_task_manager=SimpleNamespace(
            accepting=True,
            start=AsyncMock(),
            accept=accept,
        ),
        _get_persona_ref_dir=Mock(return_value="/definitely/missing/persona-refs"),
        persona_profiles=SimpleNamespace(
            resolve=Mock(
                return_value={
                    "id": "bot_a",
                    "name": "Bot A",
                    "bot_id": "platform",
                    "identity_prompt": "golden hair and blue-violet eyes",
                    "style": "photo",
                    "resolution": "inherit",
                }
            ),
            load_references=AsyncMock(return_value=[(b"identity", "image/png")]),
        ),
        _download_reference_selection=AsyncMock(),
        _send_wait_message=AsyncMock(return_value=None),
        _recall_message=AsyncMock(),
        _generate_and_send_image_async=AsyncMock(),
        create_background_task=Mock(),
    )
    return plugin, pending


def make_context(event):
    return SimpleNamespace(context=SimpleNamespace(event=event))


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["config", "references", "admission"])
async def test_persona_tool_keeps_internal_failures_out_of_the_reply(stage):
    plugin, _ = make_plugin()
    diagnostic = "API error (502), private diagnostic sentinel"
    target = {
        "config": plugin._load_config,
        "references": plugin.persona_profiles.load_references,
        "admission": plugin.background_task_manager.accept,
    }[stage]
    target.side_effect = RuntimeError(diagnostic)

    result = await VisoraPersonaPhotoTool(plugin=plugin).call(
        make_context(make_event()), scene_hint="雨后街头", extra_request="休闲外套"
    )

    assert result == SILENT_IMAGE_TOOL_FAILURE
    assert diagnostic not in result
    plugin.data_manager.deduct_points.assert_not_awaited()
    if stage == "admission":
        plugin._release_background_acceptance.assert_called_once()


def test_persona_tool_schema_requires_descriptions_and_constrains_parameters():
    schema = VisoraPersonaPhotoTool().parameters

    assert set(schema["required"]) == {"scene_hint", "extra_request"}
    assert "shot_type" not in schema["properties"]
    assert set(schema["properties"]["aspect_ratio"]["enum"]) == set(
        PERSONA_ALLOWED_ASPECT_RATIOS
    )
    assert "default" not in schema["properties"]["resolution"]
    assert "人格自拍默认分辨率" in schema["properties"]["resolution"]["description"]


def test_general_image_tool_does_not_override_configured_default_resolution():
    resolution_schema = VisoraImageGenerationTool().parameters["properties"][
        "resolution"
    ]

    assert "default" not in resolution_schema
    assert "配置中的默认分辨率" in resolution_schema["description"]


def test_general_image_tool_accepts_exact_resolution():
    resolution_schema = VisoraImageGenerationTool().parameters["properties"][
        "resolution"
    ]

    assert "enum" not in resolution_schema
    assert "WIDTHxHEIGHT" in resolution_schema["description"]


def test_general_image_tool_resolves_custom_configured_resolution():
    plugin, _ = make_plugin()
    values = {
        "default_resolution": "自定义",
        "custom_resolution": "1920 X 1080",
    }
    plugin._config_get.side_effect = lambda key, default=None: values.get(key, default)

    assert get_configured_resolution(plugin, "default_resolution") == "1920x1080"


def test_user_message_exact_resolution_is_normalized():
    event = make_event(message_text="请按 4K 构图，最终输出 1920×1080")

    assert get_explicit_user_resolution(event) == "1920x1080"

    plugin, _ = make_plugin(persona_resolution="4K")
    plugin.model = "image-model"
    plugin.default_resolution = "1K"
    config_values = {
        "default_resolution": "1K",
        "persona_photo_default_resolution": "4K",
    }
    plugin._config_get.side_effect = lambda key, default=None: config_values.get(
        key, default
    )

    image_tool = VisoraImageGenerationTool(plugin=plugin)
    persona_tool = VisoraPersonaPhotoTool(plugin=plugin)

    assert image_tool.parameters["properties"]["resolution"]["default"] == "1K"
    assert persona_tool.parameters["properties"]["resolution"]["default"] == "4K"


def test_persona_prompt_preserves_user_text_and_declares_third_person_exclusion():
    prompt = build_persona_photo_prompt(
        "window-side bedroom in the morning",
        "白色碎花裙，单手拿着手机进行前置自拍，另一只手挽起头发",
        has_user_style_reference=False,
    )

    assert "SHOT TYPE - THIRD PERSON:" in prompt
    assert prompt.count("SHOT TYPE - ") == 1
    assert "DEVICE AND FRAMING EXCLUSION:" in prompt
    assert "Do not show or imply a phone" in prompt
    assert "单手拿着手机进行前置自拍" in prompt
    assert "白色碎花裙" in prompt
    assert "另一只手挽起头发" in prompt


def test_persona_prompt_separates_identity_and_user_style_reference_roles():
    prompt = build_persona_photo_prompt(
        "sunlit cafe",
        "copy the jacket and seated pose",
        has_user_style_reference=True,
    )

    assert "REFERENCE ROLE A - PERSONA IDENTITY:" in prompt
    assert "REFERENCE ROLE B - USER STYLE ONLY:" in prompt
    style_section = prompt.split("REFERENCE ROLE B - USER STYLE ONLY:", 1)[1]
    assert "face" in style_section.lower()
    assert any(word in style_section.lower() for word in ("ignore", "never", "do not"))


def test_anime_profile_does_not_inherit_bot_as_features_or_forced_photo_style():
    prompt = build_persona_photo_prompt(
        "a library",
        "white blouse",
        False,
        identity_prompt="pink hair and bright blue eyes",
        style="anime",
    )
    assert "pink hair" in prompt
    assert "anime-style" in prompt
    assert "golden-blonde" not in prompt
    assert "blue-violet" not in prompt
    assert "No anime" not in prompt


@pytest.mark.asyncio
async def test_persona_reference_selection_accepts_message_or_quote_but_not_avatar():
    message = await collect_reference_selection(
        make_event(
            [
                Comp.Image(file="https://img/one.jpg", url="https://img/one.jpg"),
                Comp.Image(file="https://img/two.jpg", url="https://img/two.jpg"),
            ]
        ),
        max_images=1,
        include_avatars=False,
    )
    assert message.source_type == "message"
    assert message.reference_count == 1

    with patch(
        "astrbot_plugin_visora.reference_images.extract_quoted_message_images",
        AsyncMock(return_value=["https://img/quoted.jpg"]),
    ):
        quoted = await collect_reference_selection(
            make_event([Comp.Reply(id="42", chain=[])]),
            max_images=1,
            include_avatars=False,
        )
    assert quoted.source_type == "quoted"
    assert quoted.sources == ("https://img/quoted.jpg",)

    avatar = await collect_reference_selection(
        make_event([Comp.At(qq="222")]),
        max_images=1,
        include_avatars=False,
    )
    assert avatar == ReferenceSelection()


@pytest.mark.asyncio
async def test_persona_reference_read_failure_does_not_charge_or_schedule():
    plugin, pending = make_plugin()
    plugin._download_reference_selection.return_value = []
    tool = VisoraPersonaPhotoTool(plugin=plugin)
    selection = ReferenceSelection(
        sources=("https://img/stale.jpg",),
        source_type="message",
        requested=True,
        reference_count=1,
    )
    event = make_event()

    with patch(
        "astrbot_plugin_visora.llm_tools.collect_reference_selection",
        AsyncMock(return_value=selection),
    ):
        result = await tool.call(
            make_context(event),
            scene_hint="bedroom at dusk with warm window light",
            extra_request="blue cardigan, seated pose, gentle smile",
            shot_type="third_person",
        )

    assert "未能读取" in result
    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.background_task_manager.accept.assert_not_awaited()
    plugin.create_background_task.assert_not_called()
    assert pending == []


@pytest.mark.asyncio
async def test_persona_reference_parser_failure_does_not_charge_or_schedule():
    plugin, pending = make_plugin()
    tool = VisoraPersonaPhotoTool(plugin=plugin)

    with patch(
        "astrbot_plugin_visora.llm_tools.collect_reference_selection",
        AsyncMock(side_effect=RuntimeError("parser failed")),
    ):
        result = await tool.call(
            make_context(make_event()),
            scene_hint="bedroom at dusk with warm window light",
            extra_request="blue cardigan, seated pose, gentle smile",
            shot_type="third_person",
        )

    assert "未能解析" in result
    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.background_task_manager.accept.assert_not_awaited()
    plugin.create_background_task.assert_not_called()
    assert pending == []


@pytest.mark.asyncio
async def test_persona_tool_passes_frozen_inputs_to_background_manager():
    plugin, pending = make_plugin()
    first = (b"first", "image/jpeg")
    plugin._download_reference_selection.return_value = [
        first,
        (b"second", "image/png"),
    ]
    tool = VisoraPersonaPhotoTool(plugin=plugin)
    selection = ReferenceSelection(
        sources=("https://img/one.jpg", "https://img/two.jpg"),
        source_type="quoted",
        requested=True,
        reference_count=1,
    )
    event = make_event(message_text="想看一张雨后街头的生活照")

    with patch(
        "astrbot_plugin_visora.llm_tools.collect_reference_selection",
        AsyncMock(return_value=selection),
    ):
        result = await tool.call(
            make_context(event),
            scene_hint="city street after rain with neon reflections",
            extra_request="black coat, walking pose, cinematic framing",
            shot_type="third_person",
            aspect_ratio="3:4",
        )

    assert "task-persona" in result
    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin._send_wait_message.assert_not_awaited()
    plugin.create_background_task.assert_not_called()
    plugin._generate_and_send_image_async.assert_not_awaited()
    assert pending == []

    spec, frozen_inputs = plugin.background_task_manager.accept.await_args.args
    assert isinstance(spec, TaskSpec)
    assert spec.mode == "persona"
    assert spec.user_id == "111"
    assert spec.umo == "origin"
    assert spec.conversation_id == "cid-persona"
    assert spec.aspect_ratio == "3:4"
    assert spec.resolution == "2K"
    assert spec.charge_amount == 7
    assert spec.metadata == {
        "reply_mode": "combined",
        "ack_key": spec.request_fingerprint,
        "user_prompt": "想看一张雨后街头的生活照",
        "effective_prompt_summary": (
            "场景：city street after rain with neon reflections；人物、穿搭与动作："
            "black coat, walking pose, cinematic framing；构图：第三人称他拍；"
            "额外风格参考：有"
        ),
        "persona_reference_count": 1,
        "persona_profile_id": "bot_a",
        "persona_profile_name": "Bot A",
        "persona_style": "photo",
        "has_user_style_reference": True,
        "delivery_target": {
            "platform_id": "platform",
            "message_type": "FriendMessage",
            "session_id": "111",
            "group_id": "",
            "self_id": "999",
            "sender_name": "Alice",
            "runtime_id": "runtime-1",
            "session_epoch": 0,
        },
    }
    assert event.get_extra("_visora_image_ack_task_id") == "task-persona"
    assert event.get_extra("_visora_image_ack_key") == spec.request_fingerprint
    plugin._register_background_acceptance.assert_called_once_with(
        spec.request_fingerprint
    )
    assert frozen_inputs == [(b"identity", "image/png"), first]


@pytest.mark.asyncio
async def test_persona_tool_uses_configured_resolution_when_argument_is_omitted():
    plugin, pending = make_plugin(persona_resolution="4K")
    tool = VisoraPersonaPhotoTool(plugin=plugin)

    result = await tool.call(
        make_context(make_event()),
        scene_hint="sunlit balcony in the morning with green plants",
        extra_request="cream sweater, relaxed standing pose, natural portrait",
    )

    assert "task-persona" in result
    spec = plugin.background_task_manager.accept.await_args.args[0]
    assert spec.resolution == "4K"
    assert pending == []


@pytest.mark.asyncio
async def test_persona_tool_ignores_llm_resolution_without_explicit_user_request():
    plugin, pending = make_plugin(persona_resolution="4K")
    tool = VisoraPersonaPhotoTool(plugin=plugin)

    result = await tool.call(
        make_context(make_event(message_text="Bot A来张自拍")),
        scene_hint="sunlit living room with a warm everyday atmosphere",
        extra_request="yellow hoodie, playful pose, natural portrait",
        resolution="2K",
    )

    assert "task-persona" in result
    spec = plugin.background_task_manager.accept.await_args.args[0]
    assert spec.resolution == "4K"
    assert pending == []


@pytest.mark.asyncio
async def test_persona_tool_allows_explicit_user_resolution_to_override_config():
    plugin, pending = make_plugin(persona_resolution="4K")
    tool = VisoraPersonaPhotoTool(plugin=plugin)

    result = await tool.call(
        make_context(make_event(message_text="Bot A来张 2K 自拍")),
        scene_hint="sunlit living room with a warm everyday atmosphere",
        extra_request="yellow hoodie, playful pose, natural portrait",
        resolution="2K",
    )

    assert "task-persona" in result
    spec = plugin.background_task_manager.accept.await_args.args[0]
    assert spec.resolution == "2K"
    assert pending == []


@pytest.mark.asyncio
async def test_persona_tool_rejects_invalid_aspect_ratio_before_charge():
    plugin, pending = make_plugin()
    tool = VisoraPersonaPhotoTool(plugin=plugin)

    result = await tool.call(
        make_context(make_event()),
        scene_hint="sunny park",
        extra_request="summer dress and waving pose",
        shot_type="selfie",
        aspect_ratio="13:7",
    )

    assert "比例" in result
    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.background_task_manager.accept.assert_not_awaited()
    plugin.create_background_task.assert_not_called()
    assert pending == []


@pytest.mark.asyncio
async def test_persona_precharge_failure_does_not_schedule_generation():
    plugin, pending = make_plugin(deduct_result=False)
    tool = VisoraPersonaPhotoTool(plugin=plugin)

    result = await tool.call(
        make_context(make_event()),
        scene_hint="quiet library with afternoon light",
        extra_request="beige sweater, reading pose, waist-up framing",
        shot_type="third_person",
    )

    assert "不足" in result
    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.background_task_manager.accept.assert_awaited_once()
    plugin.create_background_task.assert_not_called()
    assert pending == []


@pytest.mark.asyncio
async def test_precharged_generation_cancellation_refunds_once_and_reraises():
    from astrbot_plugin_visora.main import VisoraPlugin

    plugin = object.__new__(VisoraPlugin)
    plugin._config_get = Mock(
        side_effect=lambda key, default=None: {
            "enable_economy": True,
            "cost_per_image": 7,
        }.get(key, default)
    )
    plugin.data_manager = SimpleNamespace(
        deduct_points=AsyncMock(),
        add_points=AsyncMock(),
    )
    plugin.context = SimpleNamespace(send_message=AsyncMock())
    plugin.points_name = "星琼"
    plugin._generation_semaphore = asyncio.Semaphore(1)
    plugin._load_config = Mock()

    generator = SimpleNamespace(
        api_keys=["key"],
        generate_image=AsyncMock(side_effect=asyncio.CancelledError()),
        close_session=AsyncMock(),
    )
    with patch(
        "astrbot_plugin_visora.main.create_image_generator",
        return_value=generator,
    ):
        with pytest.raises(asyncio.CancelledError):
            await plugin._generate_and_send_image_async(
                prompt="test",
                unified_msg_origin="origin",
                user_id="111",
                task_id="persona-test",
                precharged_amount=7,
                mode="persona",
            )

    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.data_manager.add_points.assert_awaited_once_with("111", 7)
    generator.close_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_precharged_cancellation_while_waiting_for_slot_refunds_once():
    from astrbot_plugin_visora.main import VisoraPlugin

    plugin = object.__new__(VisoraPlugin)
    plugin._config_get = Mock(return_value=False)
    plugin.data_manager = SimpleNamespace(
        deduct_points=AsyncMock(),
        add_points=AsyncMock(),
    )
    plugin.context = SimpleNamespace(send_message=AsyncMock())
    plugin.points_name = "星琼"
    plugin._generation_semaphore = asyncio.Semaphore(0)

    task = asyncio.create_task(
        plugin._generate_and_send_image_async(
            prompt="test",
            unified_msg_origin="origin",
            user_id="111",
            task_id="persona-wait-cancel",
            precharged_amount=7,
            mode="persona",
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.data_manager.add_points.assert_awaited_once_with("111", 7)


@pytest.mark.asyncio
async def test_precharged_multiple_failure_paths_still_refund_once():
    from astrbot_plugin_visora.main import VisoraPlugin

    plugin = object.__new__(VisoraPlugin)
    plugin._config_get = Mock(return_value=False)
    plugin.data_manager = SimpleNamespace(
        deduct_points=AsyncMock(),
        add_points=AsyncMock(),
    )
    plugin.context = SimpleNamespace(
        send_message=AsyncMock(side_effect=RuntimeError("send failed"))
    )
    plugin.points_name = "星琼"
    plugin._generation_semaphore = asyncio.Semaphore(1)
    plugin._load_config = Mock()

    generator = SimpleNamespace(
        api_keys=["key"],
        generate_image=AsyncMock(return_value=([], "upstream failed")),
        close_session=AsyncMock(),
    )
    with patch(
        "astrbot_plugin_visora.main.create_image_generator",
        return_value=generator,
    ):
        with pytest.raises(RuntimeError, match="send failed"):
            await plugin._generate_and_send_image_async(
                prompt="test",
                unified_msg_origin="origin",
                user_id="111",
                task_id="persona-double-failure",
                precharged_amount=7,
                mode="persona",
            )

    plugin.data_manager.add_points.assert_awaited_once_with("111", 7)
    generator.close_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_explicit_free_charge_state_is_not_recharged_after_config_change():
    from astrbot_plugin_visora.main import VisoraPlugin

    plugin = object.__new__(VisoraPlugin)
    plugin._config_get = Mock(
        side_effect=lambda key, default=None: {
            "enable_economy": True,
            "cost_per_image": 7,
        }.get(key, default)
    )
    plugin.data_manager = SimpleNamespace(
        deduct_points=AsyncMock(return_value=True),
        add_points=AsyncMock(),
    )
    plugin.context = SimpleNamespace(send_message=AsyncMock())
    plugin.points_name = "星琼"
    plugin._generation_semaphore = asyncio.Semaphore(1)
    plugin._load_config = Mock()

    generator = SimpleNamespace(api_keys=[], close_session=AsyncMock())
    with patch(
        "astrbot_plugin_visora.main.create_image_generator",
        return_value=generator,
    ):
        await plugin._generate_and_send_image_async(
            prompt="test",
            unified_msg_origin="origin",
            user_id="111",
            task_id="persona-free-snapshot",
            precharged_amount=0,
            mode="persona",
        )

    plugin.data_manager.deduct_points.assert_not_awaited()
    plugin.data_manager.add_points.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_during_precharge_compensates_before_reraising():
    started = asyncio.Event()
    finish = asyncio.Event()
    balance = {"value": 100}

    async def deduct_points(user_id, amount):
        balance["value"] -= amount
        started.set()
        await finish.wait()
        return True

    async def add_points(user_id, amount):
        balance["value"] += amount

    manager = SimpleNamespace(
        deduct_points=deduct_points,
        add_points=add_points,
    )
    task = asyncio.create_task(deduct_points_cancellation_safe(manager, "111", 7))
    await started.wait()
    task.cancel()
    finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert balance["value"] == 100


@pytest.mark.asyncio
async def test_unbound_or_empty_profile_never_charges_or_schedules_a_selfie():
    plugin, _ = make_plugin()
    plugin.persona_profiles.resolve.return_value = None
    result = await VisoraPersonaPhotoTool(plugin=plugin).call(
        make_context(make_event())
    )
    assert "尚未绑定" in result
    plugin.data_manager.get_points.assert_not_awaited()
    plugin.background_task_manager.accept.assert_not_awaited()
    plugin.persona_profiles.resolve.return_value = {
        "id": "bot_b",
        "name": "Bot B",
        "identity_prompt": "",
        "style": "anime",
        "resolution": "2K",
    }
    plugin.persona_profiles.load_references.return_value = []
    result = await VisoraPersonaPhotoTool(plugin=plugin).call(
        make_context(make_event())
    )
    assert "Bot B" in result and "参考图" in result
    plugin.background_task_manager.accept.assert_not_awaited()
    plugin.data_manager.deduct_points.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_bots_freeze_their_own_images_style_and_resolution(tmp_path):
    plugin, _ = make_plugin(persona_resolution="4K")
    profiles = [
        {
            "id": "bot_a",
            "name": "Bot A",
            "bot_id": "bot-a",
            "identity_prompt": "golden hair and violet eyes",
            "style": "photo",
            "resolution": "inherit",
        },
        {
            "id": "bot_b",
            "name": "Bot B",
            "bot_id": "bot-b",
            "identity_prompt": "pink hair and blue eyes",
            "style": "anime",
            "resolution": "2K",
        },
    ]
    original_get = plugin._config_get.side_effect
    plugin._config_get.side_effect = lambda key, default=None: (
        profiles if key == "persona_profiles" else original_get(key, default)
    )
    store = PersonaProfileStore(plugin, tmp_path / "persistent", tmp_path / "legacy")
    plugin.persona_profiles = store
    expected = []
    for profile, color in zip(profiles, ("yellow", "pink"), strict=True):
        output = BytesIO()
        Image.new("RGB", (24, 24), color).save(output, format="PNG")
        expected.append((output.getvalue(), "image/png"))
        await store.add_references(profile["id"], [expected[-1]], max_bytes=10000)
    tool = VisoraPersonaPhotoTool(plugin=plugin)
    no_style = ReferenceSelection(
        sources=(), source_type="none", requested=False, reference_count=0
    )
    with patch(
        "astrbot_plugin_visora.llm_tools.collect_reference_selection",
        AsyncMock(return_value=no_style),
    ):
        for profile in profiles:
            event = make_event(message_text="来一张生活照")
            event.get_platform_id = lambda profile=profile: profile["bot_id"]
            event.unified_msg_origin = profile["bot_id"] + ":FriendMessage:111"
            result = await tool.call(
                make_context(event),
                scene_hint="sunlit cafe",
                extra_request="white blouse",
                profile_id="bot_a",
                resolution="4K",
            )
            assert "task-persona" in result
    first_spec, first_inputs = plugin.background_task_manager.accept.await_args_list[
        0
    ].args
    second_spec, second_inputs = plugin.background_task_manager.accept.await_args_list[
        1
    ].args
    assert first_inputs == [expected[0]] and second_inputs == [expected[1]]
    assert "golden hair" in first_spec.prompt and "pink hair" not in first_spec.prompt
    assert "pink hair" in second_spec.prompt and "golden hair" not in second_spec.prompt
    assert "anime-style" in second_spec.prompt and "No anime" not in second_spec.prompt
    assert first_spec.resolution == "4K" and second_spec.resolution == "2K"
    assert second_spec.metadata["persona_profile_id"] == "bot_b"
    await store.delete_references(
        "bot_b", [path.name for path in store.reference_files("bot_b")]
    )
    assert second_inputs == [expected[1]]
    await store.add_references("bot_b", [expected[1]], max_bytes=10000)
    event = make_event(message_text="来张 4K 自拍")
    event.get_platform_id = lambda: "bot-b"
    with patch(
        "astrbot_plugin_visora.llm_tools.collect_reference_selection",
        AsyncMock(return_value=no_style),
    ):
        await tool.call(
            make_context(event),
            scene_hint="sunlit cafe",
            extra_request="white blouse",
            resolution="2K",
        )
    assert plugin.background_task_manager.accept.await_args.args[0].resolution == "4K"
