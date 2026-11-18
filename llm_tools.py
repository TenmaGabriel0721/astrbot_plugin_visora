"""LLM tools exposed by the Visora plugin.

The tools depend on the plugin through a runtime object reference and do not
import main.py, keeping the AstrBot entrypoint free of circular imports.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time

from astrbot.api import logger
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from .background_tasks import (
    BackgroundTaskError,
    ChargeRejectedError,
    QueueFullError,
    TaskSpec,
    build_scope_key,
)
from .image_resolution import (
    normalize_image_resolution,
    resolve_configured_resolution,
)
from .reference_images import collect_reference_selection

SILENT_IMAGE_TOOL_FAILURE = (
    "本次没有可发送的图片，不要声称已经生成或发送。"
    "继续原话题自然回复，不要向用户输出工具报错、接口状态、配置故障或返还明细，"
    "不要自动重试本工具。"
)

PERSONA_ALLOWED_ASPECT_RATIOS = (
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
    "auto",
)

PERSONA_SCENE_TEMPLATES = (
    "a quiet cafe by a large window in the late morning, warm daylight falling across the table, medium-close framing, relaxed everyday atmosphere",
    "a sunlit apartment living room in the afternoon, soft curtains and a tidy bookshelf in the background, natural window light, intimate everyday atmosphere",
    "a tree-lined city street near sunset, softly blurred storefronts behind her, warm golden-hour backlight, candid medium shot, calm weekend atmosphere",
    "a cozy bedroom in the evening, warm practical lamps, clean lived-in details, gentle cinematic shadows, natural private-life atmosphere",
)
PERSONA_OUTFIT_TEMPLATES = (
    "a cream knit cardigan over a pale blue camisole with a soft pleated skirt, delicate fabric texture, relaxed posture, one hand lightly arranging her hair, gentle smile, natural everyday portrait composition",
    "an oversized white cotton shirt with high-waisted denim shorts, naturally rolled sleeves, casual seated pose, relaxed hands, bright friendly expression, clean candid composition",
    "a light beige trench coat over a simple fitted knit dress, subtle accessories, natural walking pose, one hand holding the coat lapel, soft confident expression, candid street-portrait composition",
    "a pastel lounge sweater with a flowing ankle-length skirt, soft woven material, comfortable standing pose, hands naturally at ease, quiet warm expression, intimate indoor portrait style",
)


def _bounded_task_text(value: object, *, limit: int) -> str:
    """Normalize user-visible task text before persisting it as metadata.

    Args:
        value: Source value that may contain whitespace or control characters.
        limit: Maximum number of characters to retain.

    Returns:
        A single-line bounded string safe for later completion prompts.
    """

    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value or ""))
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: max(0, limit - 3)] + "..."
    return text


def get_tool_event(context: ContextWrapper[AstrAgentContext] | dict) -> object | None:
    """Return the AstrBot event carried by a tool execution context.

    Args:
        context: Native AstrBot context wrapper or compatibility mapping.

    Returns:
        The event object when available, otherwise ``None``.
    """
    wrapped = getattr(context, "context", None)
    if wrapped is not None and hasattr(wrapped, "event"):
        return wrapped.event
    if isinstance(context, dict):
        return context.get("event")
    return None


async def get_current_conversation_id(plugin: object, event: object) -> str:
    """Read the current AstrBot conversation ID without retaining the event.

    Args:
        plugin: Active Visora plugin instance.
        event: Current message event.

    Returns:
        Current conversation ID, or an empty string when unavailable.
    """
    try:
        manager = plugin.context.conversation_manager
        return str(
            await manager.get_curr_conversation_id(event.unified_msg_origin) or ""
        )
    except Exception as exc:
        logger.debug(f"[Visora] Failed to read the current conversation ID: {exc}")
        return ""


async def deduct_points_cancellation_safe(
    data_manager: object,
    user_id: str,
    amount: int,
) -> bool:
    """Finish an in-flight deduction and compensate it before propagating cancel."""
    deduct_task = asyncio.create_task(data_manager.deduct_points(user_id, amount))
    try:
        return await asyncio.shield(deduct_task)
    except asyncio.CancelledError:
        deducted = await asyncio.shield(deduct_task)
        if deducted:
            refund_task = asyncio.create_task(data_manager.add_points(user_id, amount))
            try:
                await asyncio.shield(refund_task)
            except Exception as exc:
                logger.error(
                    f"[Visora] Failed to refund an upfront charge after cancellation: {exc}",
                    exc_info=True,
                )
        raise


def validate_persona_aspect_ratio(aspect_ratio: str | None) -> str:
    value = str(aspect_ratio or "9:16").strip()
    if value not in PERSONA_ALLOWED_ASPECT_RATIOS:
        raise ValueError(f"unsupported aspect ratio: {value}")
    return value


def get_configured_resolution(
    plugin: object | None,
    config_key: str,
    fallback: str = "2K",
) -> str:
    """Return a validated image resolution from the active plugin config."""
    selection: object = fallback
    custom_resolution: object = "1920x1080"
    if plugin is not None:
        config_get = getattr(plugin, "_config_get", None)
        if callable(config_get):
            selection = config_get(config_key, fallback)
            if config_key == "default_resolution":
                custom_resolution = config_get("custom_resolution", custom_resolution)
        elif config_key == "default_resolution":
            selection = getattr(plugin, "default_resolution", fallback)
    if config_key == "default_resolution":
        return resolve_configured_resolution(
            selection,
            custom_resolution,
            fallback,
        )
    return normalize_image_resolution(selection, fallback=fallback)


def get_explicit_user_resolution(event: object | None) -> str | None:
    """Read an explicit 1K/2K/4K request from the user's original message."""
    if event is None:
        return None

    message_text = getattr(event, "message_str", "")
    if not isinstance(message_text, str) or not message_text.strip():
        get_message_str = getattr(event, "get_message_str", None)
        if callable(get_message_str):
            try:
                message_text = get_message_str()
            except Exception:
                message_text = ""
    if not isinstance(message_text, str):
        return None

    matches: list[tuple[int, str]] = []
    for match in re.finditer(
        r"(?<![A-Za-z0-9])([124])\s*[Kk](?![A-Za-z0-9])",
        message_text,
    ):
        matches.append((match.start(), f"{match.group(1)}K"))
    for match in re.finditer(
        r"(?<![A-Za-z0-9])(\d+)\s*[xX×]\s*(\d+)(?![A-Za-z0-9])",
        message_text,
    ):
        try:
            resolution = normalize_image_resolution(match.group(0))
        except ValueError:
            continue
        matches.append((match.start(), resolution))
    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def resolve_tool_resolution(
    plugin: object | None,
    event: object | None,
    config_key: str,
    tool_value: object | None,
) -> str:
    """Use config by default and only let the user's own text override it."""
    configured = get_configured_resolution(plugin, config_key)
    explicit = get_explicit_user_resolution(event)
    supplied = ""
    if tool_value:
        try:
            supplied = normalize_image_resolution(tool_value)
        except ValueError:
            logger.info(f"[Visora] Ignored invalid tool resolution {tool_value!r}")

    if explicit:
        if supplied and supplied != explicit:
            logger.info(
                f"[Visora] The original user message explicitly requests {explicit}，"
                f"ignoring tool resolution {supplied}"
            )
        return explicit

    if supplied and supplied != configured:
        logger.info(
            f"[Visora] No explicit resolution in the user message; "
            f"ignoring tool resolution {supplied}; using configured value {configured}"
        )
    return configured


def resolve_persona_prompt_inputs(
    scene_hint: str | None,
    extra_request: str | None,
) -> tuple[str, str]:
    """Backfill malformed/legacy tool calls from varied prompt pools."""
    scene = str(scene_hint or "").strip()
    extra = str(extra_request or "").strip()
    return (
        scene or random.choice(PERSONA_SCENE_TEMPLATES),
        extra or random.choice(PERSONA_OUTFIT_TEMPLATES),
    )


def build_persona_photo_prompt(
    scene: str,
    extra: str,
    has_user_style_reference: bool,
    *,
    identity_prompt: str = "",
    style: str = "photo",
) -> str:
    """Build a portrait prompt using the receiving Bot's identity and style.

    Args:
        scene: Requested setting and lighting.
        extra: Requested clothing, pose, and expression.
        has_user_style_reference: Whether the final image is a style reference.
        identity_prompt: Identity details belonging to this Bot alone.
        style: Configured photo, anime, or follow style.

    Returns:
        A prompt preserving identity references independently of user styling.
    """
    shot_instruction = (
        "SHOT TYPE - THIRD PERSON: A natural candid personal portrait shown from an outside observer's perspective, "
        "using a believable observer perspective and relaxed everyday framing. "
        "DEVICE AND FRAMING EXCLUSION: Do not show or imply a phone, handheld device, selfie stick, "
        "front-facing selfie, mirror selfie, or visible recording equipment. Keep the result strictly third-person. "
    )

    persona_role = (
        "REFERENCE ROLE A - PERSONA IDENTITY: Persona reference images define only the identity, face, facial structure, "
        "eye shape and color, hair color, and body proportions of the current persona. "
        "Preserve the same age impression shown by the persona references. "
        "They do not define the requested outfit, background, pose, lighting, hairstyle arrangement, or composition. "
        "Do not generate a different girl. Do not blend identities. Zero facial drift. "
    )
    if identity_prompt.strip():
        persona_role += (
            f"Additional identity details for this persona: {identity_prompt.strip()}. "
        )
    if has_user_style_reference:
        user_role = (
            "REFERENCE ROLE B - USER STYLE ONLY: The final input image is a weak secondary style reference only. "
            "Use it only for the requested outfit, accessories, pose, composition, or atmosphere. "
            "If the user asks for the same outfit, reproduce the clothing design and matching style as closely as possible. "
            "Completely ignore its face, hairstyle, hair color, eye color, body identity, age, and overall persona. "
            "All earlier persona reference images have absolute priority for identity; if there is any conflict, follow role A. "
        )
    else:
        user_role = ""

    if style == "anime":
        rendering = (
            "A detailed two-dimensional anime-style personal portrait of the same persona. "
            "Style: Anime illustration, expressive clean lines, thoughtful shading and lighting, "
            "natural proportions matching the identity references. Do not convert to realistic photography. "
        )
    elif style == "follow":
        rendering = (
            "A personal portrait of the same persona. Follow the visual style explicitly requested "
            "in the scene and outfit instructions; when unspecified, preserve the identity references' style. "
            "Do not impose a photographic or anime style against the request. "
        )
    else:
        rendering = (
            "Photorealistic real-life lifestyle portrait photography of the same persona. "
            "Style: Realistic three-dimensional portrait photography, natural light, photo-realistic skin texture, "
            "soft clean makeup, real camera look, shallow depth of field, high detail. "
            "No anime, no illustration, no CG render, no plastic skin, no doll-like face. "
        )
    return (
        rendering
        + persona_role
        + user_role
        + f"Scene: {scene}. "
        + f"Outfit / pose / expression: {extra}. "
        + shot_instruction
        + "Composition: Natural daily-life portrait, clean composition, believable environment, coherent lighting. "
        + "Quality: High detail, natural body proportions, normal hands, no text, no watermark, no extra people."
    )


@pydantic_dataclass
class VisoraImageGenerationTool(FunctionTool[AstrAgentContext]):
    """
    统一图像生成工具：支持头像生图、用户图片图生图、纯文生图
    一步到位，无需多次调用
    """

    name: str = "visora_generate_image"
    description: str = (
        "使用映境生成或修改图片。根据用户意图选择头像生图、用户图片图生图或纯文生图。\n"
        "任务受理后会在后台生成并主动发送；同一请求不要重复调用，进度请使用 visora_image_task_status 查询。\n"
        "图片和后续文字是否合并发送由插件配置决定；不要向用户复述工具或模型供应商的报错。\n"
        "如果用户要生成当前人格自己的真实照片、自拍、近照、镜子照、穿搭照或生活照，改用 visora_persona_photo。\n"
        "资源仅表示本插件的生图消耗；消耗的资源为每个用户个人累积的，并非是公共资源，提示不足仅代表当前用户资源不够。\n"
        "\n"
        "支持三种场景：\n"
        "1. 头像生图：当用户要求使用自己或他人的头像时，传入 user_ids。\n"
        "2. 用户图片图生图：用户发送或引用了图片时，不传 user_ids，插件会自动检测图片。\n"
        "3. 纯文生图：仅根据文字描述生成图片。\n"
        "\n"
        "Prompt 编写建议：\n"
        "- 头像生图时：描述场景、动作、服装、背景，不要重复描述人物外貌。\n"
        "- 用户提供图生图时：描述需要改变的方向。\n"
        "- 纯文生图时：完整描述主体、环境、风格和质量要求。\n"
        "- 默认使用中文。\n"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "图片生成提示词（使用中文）。\n"
                        "\n"
                        "⚠️ 涉及人物时（提供了 user_ids）：\n"
                        "• 描述场景、动作、服装、背景\n"
                        "• 不要描述人物外貌（外貌由头像决定）\n"
                        "• 示例：'两个人开心地自拍，樱花树下，阳光明媚'\n"
                        "\n"
                        "⚠️ 纯文生图时：\n"
                        "• 完整描述所有细节，包括主体特征\n"
                        "• 示例：'一只橘色的猫，毛茸茸的，蓝色眼睛，坐在窗台'\n"
                    ),
                },
                "user_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "要使用其头像的用户 ID 列表（仅头像生图时使用）。\n"
                        "\n"
                        "⚠️ 何时提供：\n"
                        "• 用户要求使用头像时：'画我' / '把我变成xxx'\n"
                        "• 用户 @ 了某人：'画他/她'\n"
                        "\n"
                        "⚠️ 获取流程：\n"
                        "1. 识别用户消息中的人名/昵称（'我'指代发送者）\n"
                        "2. 使用 get_user_id_by_nickname 工具获取每个人的 ID\n"
                        "3. 将所有 ID 组成数组传入此参数\n"
                        "\n"
                        "示例：['123456', '789012']"
                    ),
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": (
                        "图片宽高比。根据用户需求选择：\n"
                        "- '1:1': 方形/正方形\n"
                        "- '16:9': 横屏/宽屏/电脑壁纸\n"
                        "- '9:16': 竖屏/手机壁纸\n"
                        "- '4:3': 传统横屏\n"
                        "- '3:4': 传统竖屏\n"
                        "- 'auto': 自动选择（默认）"
                    ),
                    "enum": [
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
                        "auto",
                    ],
                },
                "resolution": {
                    "type": "string",
                    "description": (
                        "可选图片分辨率。只有用户明确要求具体分辨率时才传入；"
                        "否则不要传此参数，由插件使用配置中的默认分辨率。\n"
                        "- '1K': 标准清晰度（速度快）\n"
                        "- '2K': 高清\n"
                        "- '4K': 超高清（耗时较长）\n"
                        "- 'WIDTHxHEIGHT': 精确宽高，例如 1920x1080"
                    ),
                },
            },
            "required": ["prompt"],
        }
    )

    plugin: object | None = None

    def __post_init__(self):
        """Expose the active model and configured defaults to the LLM."""
        if self.plugin and hasattr(self.plugin, "model"):
            self.description += f"\n当前默认模型: {self.plugin.model}。"
        if self.plugin:
            configured_resolution = get_configured_resolution(
                self.plugin,
                "default_resolution",
            )
            resolution_schema = self.parameters["properties"]["resolution"]
            resolution_schema["default"] = configured_resolution
            resolution_schema["description"] += (
                f"\n当前插件配置的默认值是 {configured_resolution}。"
            )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        """Admit an image task without exposing internal exceptions to the chat.

        Args:
            context: Current AstrBot tool execution context.
            **kwargs: Image generation parameters supplied by the LLM.

        Returns:
            Admission facts or instructions to continue the conversation quietly.
        """
        try:
            return await self._admit_task(context, **kwargs)
        except Exception as exc:
            logger.error(
                f"[Visora] Image tool failed before admission: {exc}", exc_info=True
            )
            return SILENT_IMAGE_TOOL_FAILURE

    async def _admit_task(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        """Validate inputs and persist the complete image generation request.

        Args:
            context: Current AstrBot tool execution context.
            **kwargs: Image generation parameters supplied by the LLM.

        Returns:
            Tool-facing admission result.
        """
        if not (prompt := kwargs.get("prompt", "")):
            return "❌ 请提供图片生成的提示词"

        plugin = self.plugin
        if not plugin:
            return SILENT_IMAGE_TOOL_FAILURE

        event = get_tool_event(context)

        if not event:
            logger.warning(
                f"[Visora] Tool call context missing event. Context type: {type(context)}"
            )
            return SILENT_IMAGE_TOOL_FAILURE

        # Let the background runtime load provider keys asynchronously.
        plugin._load_config()

        explicit_user_ids = kwargs.get("user_ids", [])

        # Normal image generation flow.
        # Check rate limits and balances before sending a wait notice.
        user_id = event.get_sender_id()
        if not plugin._check_rate_limit(user_id):
            return f"⚠️ 请求过于频繁，请稍后再试 (每分钟限 {plugin.max_requests_per_minute} 次)"

        cost = 0
        economy_enabled = plugin._config_get("enable_economy", False)
        if economy_enabled:
            cost = plugin._config_get("cost_per_image", 10)
            current_points = await plugin.data_manager.get_points(user_id)
            if current_points < cost:
                return f"⚠️ {plugin.points_name}不足！需要 {cost} {plugin.points_name}，当前拥有 {current_points} {plugin.points_name}。"

        reference_selection = await plugin._collect_reference_selection(
            event,
            explicit_user_ids=explicit_user_ids,
        )

        # The LLM supplies the acknowledgment; do not send a second wait notice.
        wait_message_id = None

        images_data = await plugin._download_reference_selection(reference_selection)
        if reference_selection.requested and not images_data:
            if wait_message_id:
                try:
                    await plugin._recall_message(event, wait_message_id)
                except Exception:
                    pass
            return "⚠️ 未能读取你提供的参考图，本次未扣除资源。请重新发送图片、重新引用，或确认 @ 的用户头像可访问。"

        if images_data:
            logger.info(
                f"[Visora] From {reference_selection.source_type} received "
                f"{len(images_data)} deduplicated reference images"
            )
        else:
            logger.info("[Visora] No reference images; using text-to-image generation")

        res = resolve_tool_resolution(
            plugin,
            event,
            "default_resolution",
            kwargs.get("resolution"),
        )
        ar = kwargs.get("aspect_ratio") or plugin.default_aspect_ratio
        img_count = len(images_data) if images_data else 0
        has_user_ids = "是" if reference_selection.avatar_user_ids else "否"
        mode_arg = "img2img" if images_data else "txt2img"
        conversation_id = await get_current_conversation_id(plugin, event)
        message_obj = getattr(event, "message_obj", None)
        source_message_id = str(
            getattr(message_obj, "message_id", "")
            or getattr(event, "created_at", "")
            or id(event)
        )
        raw_user_prompt = str(getattr(event, "message_str", "") or "")
        get_message_str = getattr(event, "get_message_str", None)
        if not raw_user_prompt and callable(get_message_str):
            raw_user_prompt = str(get_message_str() or "")
        original_user_prompt = _bounded_task_text(
            raw_user_prompt or prompt,
            limit=512,
        )
        effective_prompt_summary = _bounded_task_text(prompt, limit=1200)
        umo_parts = str(event.unified_msg_origin or "").split(":", 2)
        get_platform_id = getattr(event, "get_platform_id", None)
        get_message_type = getattr(event, "get_message_type", None)
        get_session_id = getattr(event, "get_session_id", None)
        get_group_id = getattr(event, "get_group_id", None)
        get_sender_name = getattr(event, "get_sender_name", None)
        message_type_obj = get_message_type() if callable(get_message_type) else None
        message_type = str(
            getattr(message_type_obj, "value", message_type_obj)
            or (umo_parts[1] if len(umo_parts) == 3 else "FriendMessage")
        )
        session_id = str(
            (get_session_id() if callable(get_session_id) else "")
            or (umo_parts[2] if len(umo_parts) == 3 else user_id)
        )
        group_id = str(
            (get_group_id() if callable(get_group_id) else "")
            or (session_id if message_type == "GroupMessage" else "")
        )
        delivery_target = {
            "platform_id": str(
                (get_platform_id() if callable(get_platform_id) else "")
                or (umo_parts[0] if len(umo_parts) == 3 else "")
            ),
            "message_type": message_type,
            "session_id": session_id,
            "group_id": group_id,
            "self_id": str(event.get_self_id() or ""),
            "sender_name": str(
                (get_sender_name() if callable(get_sender_name) else "")
                or getattr(getattr(message_obj, "sender", None), "nickname", "")
                or ""
            ),
            "runtime_id": str(getattr(plugin, "_background_runtime_id", "") or ""),
            "session_epoch": int(
                getattr(plugin, "_background_session_epochs", {}).get(
                    event.unified_msg_origin,
                    0,
                )
            ),
        }
        fingerprint_payload = {
            "source_message_id": source_message_id,
            "tool": self.name,
            "prompt": prompt,
            "user_ids": list(reference_selection.avatar_user_ids),
            "aspect_ratio": ar,
            "resolution": res,
            "input_sha256": [
                hashlib.sha256(image_bytes).hexdigest()
                for image_bytes, _ in images_data
            ],
        }
        request_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manager = getattr(plugin, "background_task_manager", None)
        if manager is None:
            logger.error("[Visora] Image task manager is unavailable")
            return SILENT_IMAGE_TOOL_FAILURE
        register_ack = getattr(plugin, "_register_background_acceptance", None)
        release_ack = getattr(plugin, "_release_background_acceptance", None)
        ack_registered = callable(register_ack)
        if ack_registered:
            register_ack(request_fingerprint)
        try:
            if not manager.accepting:
                await manager.start()
            admission = await manager.accept(
                TaskSpec(
                    scope_key=build_scope_key(
                        event.unified_msg_origin,
                        user_id,
                        conversation_id,
                    ),
                    umo=event.unified_msg_origin,
                    user_id=user_id,
                    prompt=prompt,
                    mode=mode_arg,
                    aspect_ratio=ar,
                    resolution=res,
                    source_message_id=source_message_id,
                    conversation_id=conversation_id,
                    request_fingerprint=request_fingerprint,
                    charge_amount=cost if economy_enabled else 0,
                    metadata={
                        "reply_mode": (
                            "combined"
                            if plugin._config_get("combine_image_reply", True)
                            else "separate"
                        ),
                        "ack_key": request_fingerprint,
                        "user_prompt": original_user_prompt,
                        "effective_prompt_summary": effective_prompt_summary,
                        "reference_count": img_count,
                        "avatar_user_ids": list(reference_selection.avatar_user_ids),
                        "delivery_target": delivery_target,
                    },
                ),
                images_data,
            )
        except QueueFullError:
            if ack_registered and callable(release_ack):
                release_ack(request_fingerprint)
            return "⚠️ 当前后台生图队列已满，请稍后再试；本次未扣除资源。"
        except ChargeRejectedError:
            if ack_registered and callable(release_ack):
                release_ack(request_fingerprint)
            return f"⚠️ {plugin.points_name}不足！需要 {cost} {plugin.points_name}。"
        except BackgroundTaskError as exc:
            if ack_registered and callable(release_ack):
                release_ack(request_fingerprint)
            logger.error(f"[Visora] Failed to persist a background image task: {exc}")
            return SILENT_IMAGE_TOOL_FAILURE
        except Exception as exc:
            if ack_registered and callable(release_ack):
                release_ack(request_fingerprint)
            logger.error(
                f"[Visora] Failed to admit a background image task: {exc}",
                exc_info=True,
            )
            return SILENT_IMAGE_TOOL_FAILURE

        logger.info(
            f"[Visora] Background task [{admission.task_id}] - prompt: {prompt} | "
            f"references: {img_count} images | resolution: {res} | aspect ratio: {ar} | "
            f"avatar references: {has_user_ids}"
        )
        set_extra = getattr(event, "set_extra", None)
        if callable(set_extra):
            set_extra("_visora_image_ack_task_id", admission.task_id)
            set_extra("_visora_image_ack_key", request_fingerprint)
        if not admission.created:
            return (
                f"ℹ️ 该生图请求已经受理，任务 ID: {admission.task_id}，"
                f"当前状态: {admission.state}。不要重复调用本工具。"
            )
        return (
            f"✅ 已接收后台生图任务，任务 ID: {admission.task_id}，"
            "你可以继续聊天；完成后会主动发送图片。不要重复调用本工具。"
        )


@pydantic_dataclass
class VisoraImageTaskStatusTool(FunctionTool[AstrAgentContext]):
    """Read-only status tool for durable image generation tasks."""

    name: str = "visora_image_task_status"
    description: str = (
        "查询当前用户在当前会话中的映境后台生图或人格写真任务状态。"
        "当用户询问生图进度、是否完成、任务失败原因或提供了任务 ID 时调用。"
        "本工具只读，不会重新生成、重发或扣除资源。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "可选任务 ID；不填写时返回当前会话最近的任务。",
                }
            },
        }
    )
    plugin: object | None = None

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        plugin = self.plugin
        if not plugin:
            return "⚠️ 插件未正确初始化"
        event = get_tool_event(context)
        if not event:
            return "⚠️ 无法获取当前消息上下文"
        manager = getattr(plugin, "background_task_manager", None)
        if manager is None:
            return "⚠️ 后台生图任务管理器未初始化"
        try:
            if not manager.accepting:
                await manager.start()
            user_id = event.get_sender_id()
            conversation_id = await get_current_conversation_id(plugin, event)
            scope_key = build_scope_key(
                event.unified_msg_origin,
                user_id,
                conversation_id,
            )
            requested_task_id = str(kwargs.get("task_id") or "").strip()
            if not requested_task_id:
                return await manager.format_scope_status(scope_key, limit=5)

            task = await manager.get(requested_task_id)
            if (
                task is None
                or task.scope_key != scope_key
                or task.umo != event.unified_msg_origin
                or task.user_id != user_id
            ):
                return "⚠️ 未找到当前用户在本会话中的该任务。"
            state_labels = {
                "preparing": "准备输入",
                "queued": "排队中",
                "running": "生成中",
                "generated": "已生成，等待发送",
                "sending": "发送中",
                "completed": "已完成",
                "failed": "失败",
                "cancelled": "已取消",
                "interrupted": "已中断",
                "unknown": "发送结果未知",
            }
            text = f"任务 {task.task_id}: {state_labels.get(task.state, task.state)}"
            if task.delivery_state == "unknown":
                text += "。图片可能已发送，为避免重复不会自动重发"
            return text
        except BackgroundTaskError as exc:
            logger.warning(f"[Visora] Failed to query background task status: {exc}")
            return "⚠️ 暂时无法读取后台生图任务状态。"
        except Exception as exc:
            logger.error(
                f"[Visora] Background task status query raised an error: {exc}",
                exc_info=True,
            )
            return "⚠️ 暂时无法读取后台生图任务状态。"


@pydantic_dataclass
class VisoraVideoGenerationTool(FunctionTool[AstrAgentContext]):
    """视频生成工具：用于文生视频、图生视频和让图片动起来"""

    name: str = "visora_generate_video"
    description: str = (
        "使用配置的 NewAPI/中转站视频模型生成视频。仅当用户明确要求生成视频、动画、动态画面、镜头运动、短片，"
        "或要求让当前/引用图片动起来时调用。普通静态图片请求继续使用 visora_generate_image。"
        "资源仅表示本插件的生图消耗；消耗的资源为每个用户个人累积的，并非是公共资源，提示不足仅代表当前用户资源不够。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "视频生成提示词，描述主体、动作、镜头运动、场景、风格和氛围。",
                },
                "user_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选。需要使用头像作为参考图时填写用户 ID 列表。",
                },
                "duration": {
                    "type": "integer",
                    "description": "可选。视频时长（秒），不确定时留空使用插件默认值。",
                },
                "ratio": {
                    "type": "string",
                    "description": "可选。视频比例，如 16:9、9:16、1:1，不确定时留空。",
                    "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "auto", "adaptive"],
                },
                "resolution": {
                    "type": "string",
                    "description": "可选。视频分辨率，如 720p、1080p，不确定时留空。",
                    "enum": ["480p", "720p", "1080p"],
                },
            },
            "required": ["prompt"],
        }
    )
    plugin: object | None = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        prompt = str(kwargs.get("prompt", "") or "").strip()
        if not prompt:
            return "❌ 请提供视频生成的提示词"
        plugin = self.plugin
        if not plugin:
            return "⚠️ 插件未正确初始化 (Plugin instance missing)"
        event = None
        if hasattr(context, "context") and isinstance(
            context.context, AstrAgentContext
        ):
            event = context.context.event
        elif isinstance(context, dict):
            event = context.get("event")
        if not event:
            return "⚠️ 无法获取当前消息上下文"

        plugin._load_config()
        if not getattr(plugin, "enable_video_generation", True):
            return "⚠️ 视频生成功能未启用"

        user_id = event.get_sender_id()
        if not plugin._check_rate_limit(user_id):
            return f"⚠️ 请求过于频繁，请稍后再试 (每分钟限 {plugin.max_requests_per_minute} 次)"

        # Reject insufficient balances before sending a wait notice.
        cost = 0
        economy_enabled = plugin._config_get("enable_economy", False)
        if economy_enabled:
            cost = plugin._config_get("cost_per_video", 30)
            current_points = await plugin.data_manager.get_points(user_id)
            if current_points < cost:
                return f"⚠️ {plugin.points_name}不足！需要 {cost} {plugin.points_name}，当前拥有 {current_points} {plugin.points_name}。"

        # The LLM supplies the acknowledgment; do not send a second wait notice.
        wait_message_id = None

        async def _recall_wait_message():
            if wait_message_id:
                try:
                    await plugin._recall_message(event, wait_message_id)
                except Exception:
                    pass

        # Use ReferenceSelection in message, quote, then avatar priority order.
        try:
            selection = await plugin._collect_reference_selection(
                event,
                explicit_user_ids=kwargs.get("user_ids", []),
            )
        except asyncio.CancelledError:
            await _recall_wait_message()
            raise
        except Exception as exc:
            logger.error(
                f"[Visora][Video Tool] Failed to parse references: {exc}",
                exc_info=True,
            )
            await _recall_wait_message()
            return "⚠️ 未能解析参考图，本次未扣除资源，请重试。"

        try:
            images_data = await plugin._download_reference_selection(selection)
            if selection.requested and not images_data:
                await _recall_wait_message()
                return (
                    "⚠️ 未能读取你提供的参考图，本次未扣除资源。"
                    "请重新发送图片、重新引用，或确认 @ 的用户头像可访问。"
                )
            public_image_urls = await plugin._build_video_image_urls(
                images_data,
                selection.candidate_groups,
                "视频工具",
            )
        except asyncio.CancelledError:
            await _recall_wait_message()
            raise
        except Exception as exc:
            logger.error(
                f"[Visora][Video Tool] Failed to read references: {exc}",
                exc_info=True,
            )
            await _recall_wait_message()
            return "⚠️ 未能读取参考图，本次未扣除资源，请重试。"

        # Charge points.
        deducted = False
        if economy_enabled:
            try:
                deducted_ok = await deduct_points_cancellation_safe(
                    plugin.data_manager,
                    user_id,
                    cost,
                )
            except asyncio.CancelledError:
                await _recall_wait_message()
                raise
            if not deducted_ok:
                await _recall_wait_message()
                return f"⚠️ {plugin.points_name}不足！需要 {cost} {plugin.points_name}。"
            deducted = True

        task_id = hashlib.md5(f"video{time.time()}{user_id}".encode()).hexdigest()[:8]

        # Wrap the notice result so it can be recalled.
        async def _video_generate_with_recall():
            try:
                await plugin._generate_and_send_video_async(
                    prompt=prompt,
                    images_data=images_data or None,
                    image_urls=public_image_urls or None,
                    unified_msg_origin=event.unified_msg_origin,
                    user_id=user_id,
                    duration=kwargs.get("duration") or plugin.default_video_duration,
                    ratio=kwargs.get("ratio") or plugin.default_video_ratio,
                    resolution=kwargs.get("resolution")
                    or plugin.default_video_resolution,
                    task_id=task_id,
                    precharged_amount=(cost if deducted else 0),
                    reference_requested=selection.requested,
                )
            finally:
                await _recall_wait_message()

        generation_coro = _video_generate_with_recall()
        try:
            plugin.create_background_task(generation_coro)
            # Give the newly created task one event-loop turn so its cancellation
            # guard is active before this precharged tool call returns.
            await asyncio.sleep(0)
        except Exception as exc:
            generation_coro.close()
            if deducted:
                refund_task = asyncio.create_task(
                    plugin.data_manager.add_points(user_id, cost)
                )
                await asyncio.shield(refund_task)
            await _recall_wait_message()
            logger.error(
                f"[Visora][Video Tool] Failed to create a background task: {exc}"
            )
            return "❌ 生视频任务启动失败" + (
                f"，{plugin.points_name}已返还" if deducted else ""
            )
        return "同意"


@pydantic_dataclass
class VisoraPersonaPhotoTool(FunctionTool[AstrAgentContext]):
    name: str = "visora_persona_photo"
    description: str = (
        "用于生成当前人格的自拍、近照、写真、生活照、穿搭照或动漫插画。"
        "参考图、人物身份和画风自动读取当前 Bot 的独立自拍资料，不接受指定另一个 Bot 的身份。"
        "当用户表达想看“你”的照片、自拍、日常照、穿搭照、近照、今天穿什么等意图时使用本工具，但成片统一使用第三人称他拍视角。"
        "调用本工具时必须把 scene_hint 和 extra_request 写得具体，不能只写十几个字。"
        "scene_hint 应包含具体场景、时间、背景细节、光线、镜头距离和氛围。"
        "extra_request 应包含服装、颜色、材质、姿势、手部动作、表情、构图和照片风格。"
        "如果用户没有指定场景或服装，你需要根据当前时间和日常生活随机设计一个自然但不重复的场景与穿搭。"
        "当前 Bot 的人物参考图只用于锁定人物身份、脸部特征、发色、瞳色和体型，不能照抄其服装、背景、动作和构图。"
        "用户随消息或引用提供的额外图片只用于参考服装、姿势、构图和氛围，必须忽略其中人物的脸与身份。"
        "画风遵循当前 Bot 的独立资料：真人摄影、动漫插画或跟随本次要求；不要擅自改成另一种画风。"
        "不要把本描述当作回复用户的固定话术；工具返回任务已受理后，应按当前对话语气自然简短地告知用户正在后台生成，且不要重复调用。"
        "图片和后续文字是否合并发送由插件配置决定；不要向用户复述工具或模型供应商的报错。"
        "资源仅表示本插件的生图消耗；消耗的资源为每个用户个人累积的，并非是公共资源，提示不足仅代表当前用户资源不够。"
    )
    plugin: object = Field(default=None, exclude=True)
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "scene_hint": {
                    "type": "string",
                    "description": (
                        "详细场景提示词。必须包含地点、时间、背景细节、光线、镜头距离、氛围。"
                        "不要只写短句。"
                    ),
                },
                "extra_request": {
                    "type": "string",
                    "description": (
                        "详细人物与穿搭提示词。必须包含服装、颜色、材质、姿势、手部动作、表情、构图和风格。"
                        "不要照抄人格参考图的衣服、背景、动作和构图；如果用户附带或引用图片，"
                        "只参考那张图的服装、姿势、构图和氛围，忽略其中人物的脸与身份。"
                        "不要描述手机、前置自拍或对镜自拍，统一按第三人称他拍构图。"
                    ),
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": list(PERSONA_ALLOWED_ASPECT_RATIOS),
                    "default": "9:16",
                    "description": "图片比例，自拍/竖图优先使用 9:16 或 3:4；不确定可使用 auto。",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["1K", "2K", "4K"],
                    "description": (
                        "可选生成分辨率。只有用户明确要求 1K、2K 或 4K 时才传入；"
                        "否则不要传此参数，由插件使用“人格自拍默认分辨率”配置。"
                    ),
                },
            },
            "required": ["scene_hint", "extra_request"],
        }
    )

    def __post_init__(self):
        if not self.plugin:
            return
        configured_resolution = get_configured_resolution(
            self.plugin,
            "persona_photo_default_resolution",
        )
        resolution_schema = self.parameters["properties"]["resolution"]
        resolution_schema["default"] = configured_resolution
        resolution_schema["description"] += (
            f" 当前插件配置的默认值是 {configured_resolution}。"
        )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        """Admit a persona task without exposing internal exceptions to the chat.

        Args:
            context: Current AstrBot tool execution context.
            **kwargs: Persona generation parameters supplied by the LLM.

        Returns:
            Admission facts or instructions to continue the conversation quietly.
        """
        try:
            return await self._admit_task(context, **kwargs)
        except Exception as exc:
            logger.error(
                f"[Visora][Persona] Image tool failed before admission: {exc}",
                exc_info=True,
            )
            return SILENT_IMAGE_TOOL_FAILURE

    async def _admit_task(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        """Resolve the Bot identity and persist the complete persona request.

        Args:
            context: Current AstrBot tool execution context.
            **kwargs: Persona generation parameters supplied by the LLM.

        Returns:
            Tool-facing admission result.
        """
        plugin = self.plugin
        if not plugin:
            return SILENT_IMAGE_TOOL_FAILURE
        event = get_tool_event(context)
        if not event:
            return SILENT_IMAGE_TOOL_FAILURE

        # Let the background persona runtime load provider keys asynchronously.
        plugin._load_config()

        profile_store = getattr(plugin, "persona_profiles", None)
        if profile_store is None:
            return "⚠️ 自拍资料尚未初始化，请稍后再试。"
        try:
            profile = profile_store.resolve(event)
        except ValueError as exc:
            logger.warning(f"[Visora][Persona] Invalid Bot profile: {exc}")
            return "⚠️ 自拍资料配置有误，请管理员在 映境 Visora 管理页检查 Bot 绑定。"
        if profile is None:
            return "⚠️ 当前 Bot 尚未绑定自拍资料，请管理员在 映境 Visora 管理页为它绑定独立资料并上传参考图。"

        user_id = event.get_sender_id()
        if not plugin._check_rate_limit(user_id):
            return f"⚠️ 请求过于频繁，请稍后再试 (每分钟限 {plugin.max_requests_per_minute} 次)"

        # Reject insufficient balances before sending a wait notice.
        economy_enabled = plugin._config_get("enable_economy", False)
        cost = plugin._config_get("cost_per_image", 10) if economy_enabled else 0
        if economy_enabled:
            current_points = await plugin.data_manager.get_points(user_id)
            if current_points < cost:
                return f"⚠️ {plugin.points_name}不足！需要 {cost} {plugin.points_name}，当前拥有 {current_points} {plugin.points_name}。"

        scene, extra = resolve_persona_prompt_inputs(
            kwargs.get("scene_hint"), kwargs.get("extra_request")
        )
        try:
            aspect_ratio = validate_persona_aspect_ratio(
                kwargs.get("aspect_ratio", "9:16")
            )
        except ValueError:
            return "⚠️ 不支持的图片比例，请使用工具列出的标准比例。"

        # The LLM supplies the acknowledgment; do not send a second wait notice.
        wait_message_id = None

        max_references = max(
            1, int(getattr(plugin, "max_reference_images_per_request", 4))
        )
        try:
            images_data = await profile_store.load_references(
                profile,
                limit=max_references,
                max_bytes=int(getattr(plugin, "max_image_size_mb", 10) * 1024 * 1024),
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                f"[Visora][Persona] Failed to read identity references: {exc}"
            )
            images_data = []
        if not images_data:
            return (
                f"⚠️ {profile['name']}还没有可用的自拍参考图，请管理员在管理页上传，"
                "或向这个 Bot 发送图片并使用“自拍参考图 上传”。本次未扣除资源。"
            )

        # Additional persona references accept only message or quoted images, not avatars.
        # An @ mention must not introduce another user's avatar into the persona identity.
        user_images: list[tuple[bytes, str]] = []
        try:
            selection = await collect_reference_selection(
                event,
                max_images=1,
                max_avatars=1,
                include_avatars=False,
            )
        except Exception as exc:
            logger.warning(
                f"[Visora][Persona] Failed to parse additional references: {exc}"
            )
            if wait_message_id:
                try:
                    await plugin._recall_message(event, wait_message_id)
                except Exception:
                    pass
            return (
                "⚠️ 未能解析你提供的参考图，本次未扣除资源。请重新发送或重新引用图片。"
            )

        if selection.source_type in ("message", "quoted"):
            try:
                downloaded_images = await plugin._download_reference_selection(
                    selection,
                    limit=1,
                )
            except Exception as exc:
                logger.warning(
                    f"[Visora][Persona] Failed to read additional references: {exc}"
                )
                downloaded_images = []
            user_images = list(downloaded_images)[:1]
            if selection.requested and not user_images:
                if wait_message_id:
                    try:
                        await plugin._recall_message(event, wait_message_id)
                    except Exception:
                        pass
                return "⚠️ 未能读取你提供的参考图，本次未扣除资源。请重新发送或重新引用图片。"

        # Identity references come first; append at most one user style reference last.
        if user_images:
            if max_references < 2:
                return "⚠️ 同时使用人物和穿搭参考图需要至少 2 张参考图额度，请管理员调整参考图数量上限。本次未扣除资源。"
            images_data = images_data[: max_references - 1]
        images_data.extend(user_images)

        p = build_persona_photo_prompt(
            scene,
            extra,
            has_user_style_reference=bool(user_images),
            identity_prompt=profile["identity_prompt"],
            style=profile["style"],
        )
        persona_resolution = resolve_tool_resolution(
            plugin,
            event,
            "persona_photo_default_resolution",
            kwargs.get("resolution"),
        )
        if (
            not get_explicit_user_resolution(event)
            and profile["resolution"] != "inherit"
        ):
            persona_resolution = profile["resolution"]
        conversation_id = await get_current_conversation_id(plugin, event)
        message_obj = getattr(event, "message_obj", None)
        source_message_id = str(
            getattr(message_obj, "message_id", "")
            or getattr(event, "created_at", "")
            or id(event)
        )
        public_prompt_summary = (
            f"场景：{scene}；人物、穿搭与动作：{extra}；构图：第三人称他拍；"
            f"额外风格参考：{'有' if user_images else '无'}"
        )
        raw_user_prompt = str(getattr(event, "message_str", "") or "")
        get_message_str = getattr(event, "get_message_str", None)
        if not raw_user_prompt and callable(get_message_str):
            raw_user_prompt = str(get_message_str() or "")
        original_user_prompt = _bounded_task_text(
            raw_user_prompt or public_prompt_summary,
            limit=512,
        )
        effective_prompt_summary = _bounded_task_text(
            public_prompt_summary,
            limit=1200,
        )
        umo_parts = str(event.unified_msg_origin or "").split(":", 2)
        get_platform_id = getattr(event, "get_platform_id", None)
        get_message_type = getattr(event, "get_message_type", None)
        get_session_id = getattr(event, "get_session_id", None)
        get_group_id = getattr(event, "get_group_id", None)
        get_sender_name = getattr(event, "get_sender_name", None)
        message_type_obj = get_message_type() if callable(get_message_type) else None
        message_type = str(
            getattr(message_type_obj, "value", message_type_obj)
            or (umo_parts[1] if len(umo_parts) == 3 else "FriendMessage")
        )
        session_id = str(
            (get_session_id() if callable(get_session_id) else "")
            or (umo_parts[2] if len(umo_parts) == 3 else user_id)
        )
        group_id = str(
            (get_group_id() if callable(get_group_id) else "")
            or (session_id if message_type == "GroupMessage" else "")
        )
        delivery_target = {
            "platform_id": str(
                (get_platform_id() if callable(get_platform_id) else "")
                or (umo_parts[0] if len(umo_parts) == 3 else "")
            ),
            "message_type": message_type,
            "session_id": session_id,
            "group_id": group_id,
            "self_id": str(event.get_self_id() or ""),
            "sender_name": str(
                (get_sender_name() if callable(get_sender_name) else "")
                or getattr(getattr(message_obj, "sender", None), "nickname", "")
                or ""
            ),
            "runtime_id": str(getattr(plugin, "_background_runtime_id", "") or ""),
            "session_epoch": int(
                getattr(plugin, "_background_session_epochs", {}).get(
                    event.unified_msg_origin,
                    0,
                )
            ),
        }
        fingerprint_payload = {
            "source_message_id": source_message_id,
            "tool": self.name,
            "persona_profile_id": profile["id"],
            "persona_identity_prompt": profile["identity_prompt"],
            "persona_style": profile["style"],
            "scene": scene,
            "extra": extra,
            "aspect_ratio": aspect_ratio,
            "resolution": persona_resolution,
            "input_sha256": [
                hashlib.sha256(image_bytes).hexdigest()
                for image_bytes, _ in images_data
            ],
        }
        request_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manager = getattr(plugin, "background_task_manager", None)
        if manager is None:
            logger.error("[Visora][Persona] Image task manager is unavailable")
            return SILENT_IMAGE_TOOL_FAILURE
        register_ack = getattr(plugin, "_register_background_acceptance", None)
        release_ack = getattr(plugin, "_release_background_acceptance", None)
        ack_registered = callable(register_ack)
        if ack_registered:
            register_ack(request_fingerprint)
        try:
            if not manager.accepting:
                await manager.start()
            admission = await manager.accept(
                TaskSpec(
                    scope_key=build_scope_key(
                        event.unified_msg_origin,
                        user_id,
                        conversation_id,
                    ),
                    umo=event.unified_msg_origin,
                    user_id=user_id,
                    prompt=p,
                    mode="persona",
                    aspect_ratio=aspect_ratio,
                    resolution=persona_resolution,
                    source_message_id=source_message_id,
                    conversation_id=conversation_id,
                    request_fingerprint=request_fingerprint,
                    charge_amount=cost if economy_enabled else 0,
                    metadata={
                        "reply_mode": (
                            "combined"
                            if plugin._config_get("combine_image_reply", True)
                            else "separate"
                        ),
                        "ack_key": request_fingerprint,
                        "user_prompt": original_user_prompt,
                        "effective_prompt_summary": effective_prompt_summary,
                        "persona_reference_count": len(images_data) - len(user_images),
                        "persona_profile_id": profile["id"],
                        "persona_profile_name": profile["name"],
                        "persona_style": profile["style"],
                        "has_user_style_reference": bool(user_images),
                        "delivery_target": delivery_target,
                    },
                ),
                images_data,
            )
        except QueueFullError:
            if ack_registered and callable(release_ack):
                release_ack(request_fingerprint)
            return "⚠️ 当前后台生图队列已满，请稍后再试；本次未扣除资源。"
        except ChargeRejectedError:
            if ack_registered and callable(release_ack):
                release_ack(request_fingerprint)
            return f"⚠️ {plugin.points_name}不足！需要 {cost} {plugin.points_name}。"
        except BackgroundTaskError as exc:
            if ack_registered and callable(release_ack):
                release_ack(request_fingerprint)
            logger.error(
                f"[Visora][Persona] Failed to persist a background task: {exc}"
            )
            return SILENT_IMAGE_TOOL_FAILURE
        except Exception as exc:
            if ack_registered and callable(release_ack):
                release_ack(request_fingerprint)
            logger.error(
                f"[Visora][Persona] Failed to admit a background task: {exc}",
                exc_info=True,
            )
            return SILENT_IMAGE_TOOL_FAILURE

        set_extra = getattr(event, "set_extra", None)
        if callable(set_extra):
            set_extra("_visora_image_ack_task_id", admission.task_id)
            set_extra("_visora_image_ack_key", request_fingerprint)
        if not admission.created:
            return (
                f"ℹ️ 该人格写真请求已经受理，任务 ID: {admission.task_id}，"
                f"当前状态: {admission.state}。不要重复调用本工具。"
            )
        return (
            f"✅ 已接收后台人格写真任务，任务 ID: {admission.task_id}，"
            "你可以继续聊天；完成后会主动发送图片。不要重复调用本工具。"
        )
