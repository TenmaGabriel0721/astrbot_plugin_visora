"""Dashboard settings for model order, presets, and Bot identity references."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from astrbot.api import logger
from astrbot.api.web import request
from PIL import Image, ImageOps

from .persona_profiles import MAX_IDENTITY_REFERENCES, PersonaProfileStore
from .presets import MAX_PRESETS, normalize_presets

ROUTE_MODES = ("txt2img", "img2img", "persona", "admin")
REQUEST_FORMATS = (
    ("gemini", "Gemini 原生", "v1beta/models/{model}:generateContent"),
    ("openai_chat", "OpenAI Chat 生图", "v1/chat/completions"),
    ("openai_images", "OpenAI 图片生成", "v1/images/generations"),
    ("openai_edits", "OpenAI 图片编辑", "v1/images/edits"),
)


class SettingsPageApi:
    """Expose authenticated settings through the AstrBot plugin page bridge."""

    def __init__(self, plugin: Any) -> None:
        """Create a settings editor using the loaded plugin instance.

        Args:
            plugin: Plugin providing configuration, provider, and identity APIs.
        """
        self.plugin = plugin
        self._lock = asyncio.Lock()

    def register_routes(self) -> None:
        """Register settings and reference endpoints in the plugin namespace."""
        for path, handler, methods in (
            ("/settings", self.get_settings, ["GET"]),
            ("/settings", self.save_settings, ["POST"]),
            ("/references/<profile_id>", self.get_references, ["GET"]),
            ("/references/<profile_id>", self.upload_reference, ["POST"]),
            ("/references/<profile_id>/delete", self.delete_references, ["POST"]),
        ):
            self.plugin.context.register_web_api(
                f"/astrbot_plugin_visora/page{path}",
                handler,
                methods,
                "Visora plugin settings",
            )

    def normalize_routes(self, raw: Any) -> list[dict[str, Any]]:
        """Validate route entries and derive priorities from their list order.

        Args:
            raw: Routes already arranged in the desired per-mode order.

        Returns:
            Canonical route entries with contiguous per-mode priorities.

        Raises:
            ValueError: A route contains an invalid mode, provider, or endpoint.
        """
        if not isinstance(raw, list) or len(raw) > 128:
            raise ValueError("模型列表必须是列表，且最多包含 128 项")
        aliases = {
            "text": "txt2img",
            "text2img": "txt2img",
            "image": "img2img",
            "image2image": "img2img",
        }
        seen: set[tuple[str, str]] = set()
        ranks = dict.fromkeys(ROUTE_MODES, 0)
        result = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("模型路由必须是对象")
            mode = (
                str(item.get("__template_key") or item.get("mode") or "")
                .strip()
                .lower()
            )
            mode = aliases.get(mode, mode)
            provider_id = str(
                item.get("provider_id") or item.get("model_provider_id") or ""
            ).strip()
            endpoint = str(
                item.get("request_format") or item.get("api_request_format") or ""
            ).strip()
            if mode not in ROUTE_MODES or not provider_id or len(provider_id) > 256:
                raise ValueError("模型路由缺少有效的任务类型或供应商")
            if not endpoint or len(endpoint) > 512:
                raise ValueError("请选择请求协议，或填写自定义请求路径")
            endpoint = self.plugin._api_format_to_endpoint(endpoint).lstrip("/")
            parsed = urlparse(endpoint)
            if (
                parsed.scheme
                or parsed.netloc
                or parsed.query
                or parsed.fragment
                or any(segment in {".", ".."} for segment in endpoint.split("/"))
                or re.search(r"[\s,\\]", endpoint)
                or re.findall(r"\{([^}]+)\}", endpoint) not in ([], ["model"])
            ):
                raise ValueError("自定义请求路径必须是相对路径，仅支持 {model} 占位符")
            if (mode, provider_id) in seen:
                raise ValueError("同一任务类型不能重复添加同一个模型供应商")
            seen.add((mode, provider_id))
            result.append(
                {
                    "__template_key": mode,
                    "provider_id": provider_id,
                    "request_format": endpoint,
                    "enabled": self.plugin._parse_route_enabled(item.get("enabled")),
                    "priority": ranks[mode],
                }
            )
            ranks[mode] += 1
        return result

    def snapshot(self) -> dict[str, Any]:
        """Collect public settings, provider choices, and a conflict revision.

        Returns:
            A detached configuration payload containing no provider secrets.
        """
        managed = {
            "routes": self.plugin._config_get("image_model_routes", []),
            "profiles": self.plugin._config_get("persona_profiles", []),
            "task_timeout": self.plugin._config_get("task_timeout", 600),
            "video_provider_id": self.plugin._config_get("video_provider_id", ""),
            "presets": self.plugin._config_get("presets", []),
            "combine_image_reply": self.plugin._config_get("combine_image_reply", True),
        }
        revision = hashlib.sha256(
            json.dumps(
                managed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        ordered = sorted(
            managed["routes"],
            key=lambda entry: (
                self.plugin._parse_route_priority(entry.get("priority"))
                if isinstance(entry, dict)
                else 0
            ),
        )
        routes = self.normalize_routes(ordered)
        profiles = self.plugin.persona_profiles.profiles()
        providers = []
        for provider in self.plugin.context.get_all_providers():
            config = getattr(provider, "provider_config", {}) or {}
            provider_id = str(config.get("id") or getattr(provider, "id", "") or "")
            if not provider_id:
                continue
            model = str(config.get("model") or provider_id.rsplit("/", 1)[-1])
            providers.append({"id": provider_id, "model": model})
        providers.sort(key=lambda item: item["id"].casefold())
        get_config = getattr(self.plugin.context, "get_config", None)
        platform_config = (
            get_config().get("platform", []) if callable(get_config) else []
        )
        names = {
            profile["bot_id"]: profile["name"]
            for profile in profiles
            if profile["bot_id"]
        }
        bots = [
            {
                "id": str(item.get("id") or ""),
                "type": str(item.get("type") or ""),
                "enabled": self.plugin._parse_route_enabled(item.get("enable")),
                "name": names.get(
                    str(item.get("id") or ""),
                    str(item.get("name") or item.get("id") or "Bot"),
                ),
            }
            for item in platform_config
            if isinstance(item, dict) and item.get("id")
        ]
        return {
            "routes": routes,
            "profiles": [
                {
                    **profile,
                    "reference_count": len(
                        self.plugin.persona_profiles.reference_files(profile["id"])
                    ),
                }
                for profile in profiles
            ],
            "providers": providers,
            "bots": bots,
            "formats": [
                {"value": value, "label": label, "endpoint": endpoint}
                for value, label, endpoint in REQUEST_FORMATS
            ],
            "task_timeout": getattr(self.plugin, "task_timeout", 600),
            "video_provider_id": managed["video_provider_id"],
            "presets": normalize_presets(managed["presets"]),
            "max_preset_count": MAX_PRESETS,
            "max_reference_count": MAX_IDENTITY_REFERENCES,
            "max_reference_bytes": int(
                min(10, self.plugin.max_image_size_mb) * 1024 * 1024
            ),
            "revision": revision,
            "combine_image_reply": managed["combine_image_reply"],
            "config_name": Path(
                getattr(self.plugin.config, "config_path", "当前配置")
            ).name,
        }

    async def get_settings(self) -> dict[str, Any]:
        """Read settings without exposing provider connection values.

        Returns:
            Public settings or a sanitized error response.
        """
        try:
            async with self._lock:
                return {"status": "ok", "data": self.snapshot()}
        except Exception as exc:
            logger.warning(f"[Visora] Failed to read dashboard settings: {exc}")
            return {
                "status": "error",
                "message": self.plugin._format_user_error_detail(exc),
            }

    async def save_settings(self) -> dict[str, Any]:
        """Save list order and independent identities without restarting Bots.

        Returns:
            Saved public settings or an error preserving the previous config.
        """
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("设置内容必须是对象")
            async with self._lock:
                current = self.snapshot()
                if payload.get("revision") != current["revision"]:
                    raise ValueError(
                        "设置已在别处更新，请重新加载后再保存，避免覆盖新的配置"
                    )
                routes = self.normalize_routes(payload.get("routes"))
                profiles = PersonaProfileStore.normalize_profiles(
                    payload.get("profiles")
                )
                presets = normalize_presets(
                    payload.get("presets", current["presets"]), strict=True
                )
                combine_image_reply = payload.get(
                    "combine_image_reply", current["combine_image_reply"]
                )
                if not isinstance(combine_image_reply, bool):
                    raise ValueError("图文合并发送必须是开启或关闭")
                allowed = {item["id"] for item in current["providers"]} | {
                    item["provider_id"] for item in current["routes"]
                }
                if any(item["provider_id"] not in allowed for item in routes):
                    raise ValueError("供应商不存在，请刷新供应商列表后重新选择")
                task_timeout = payload.get("task_timeout", current["task_timeout"])
                if (
                    isinstance(task_timeout, bool)
                    or not isinstance(task_timeout, int)
                    or not 30 <= task_timeout <= 3600
                ):
                    raise ValueError("生图总时限必须是 30 至 3600 秒的整数")
                video_provider_id = str(
                    payload.get("video_provider_id", current["video_provider_id"]) or ""
                ).strip()
                if video_provider_id and video_provider_id not in allowed | {
                    current["video_provider_id"]
                }:
                    raise ValueError("视频供应商不存在，请刷新列表后重新选择")
                previous = copy.deepcopy(dict(self.plugin.config))
                try:
                    for key, value in (
                        ("image_model_routes", routes),
                        ("persona_profiles", profiles),
                        ("task_timeout", task_timeout),
                        ("video_provider_id", video_provider_id),
                        ("presets", presets),
                        ("combine_image_reply", combine_image_reply),
                    ):
                        self.plugin._config_set(key, value)
                    self.plugin._load_config()
                    save_task = asyncio.create_task(
                        asyncio.to_thread(self.plugin.config.save_config)
                    )
                    try:
                        await asyncio.shield(save_task)
                    except asyncio.CancelledError:
                        await asyncio.shield(save_task)
                        raise
                except Exception:
                    self.plugin.config.clear()
                    self.plugin.config.update(previous)
                    self.plugin._load_config()
                    raise
                return {"status": "ok", "data": self.snapshot()}
        except Exception as exc:
            logger.warning(f"[Visora] Failed to save dashboard settings: {exc}")
            return {
                "status": "error",
                "message": self.plugin._format_user_error_detail(exc),
            }

    async def get_references(self, profile_id: str) -> dict[str, Any]:
        """Return small previews for one configured Bot's reference library.

        Args:
            profile_id: Profile selected in the settings page.

        Returns:
            Preview data URLs and file names, or a sanitized error response.
        """
        try:
            store = self.plugin.persona_profiles
            store.get(profile_id)
            max_bytes = int(min(10, self.plugin.max_image_size_mb) * 1024 * 1024)

            def previews() -> list[dict[str, Any]]:
                result = []
                for path in store.reference_files(profile_id)[:MAX_IDENTITY_REFERENCES]:
                    preview = ""
                    try:
                        if path.stat().st_size > max_bytes:
                            raise ValueError("Reference exceeds the file size limit")
                        with Image.open(path) as source:
                            thumbnail = ImageOps.exif_transpose(source)
                            thumbnail.thumbnail((320, 320))
                            buffer = BytesIO()
                            thumbnail.convert("RGB").save(
                                buffer, format="JPEG", quality=82
                            )
                        preview = "data:image/jpeg;base64," + base64.b64encode(
                            buffer.getvalue()
                        ).decode("ascii")
                    except (OSError, ValueError, Image.DecompressionBombError):
                        pass
                    result.append({"name": path.name, "data_url": preview})
                return result

            return {
                "status": "ok",
                "data": {"references": await asyncio.to_thread(previews)},
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": self.plugin._format_user_error_detail(exc),
            }

    async def upload_reference(self, profile_id: str) -> dict[str, Any]:
        """Validate and append one image from the page's native upload bridge.

        Args:
            profile_id: Profile receiving the uploaded identity reference.

        Returns:
            Added file names and updated counts, or a sanitized error response.
        """
        try:
            async with self._lock:
                files = await request.files()
                uploaded = files.get("file")
                if uploaded is None:
                    raise ValueError("请选择参考图")
                max_bytes = int(min(10, self.plugin.max_image_size_mb) * 1024 * 1024)
                data = await uploaded.read(max_bytes + 1)
                names = await self.plugin.persona_profiles.add_references(
                    profile_id,
                    [(data, uploaded.content_type or "")],
                    max_bytes=max_bytes,
                )
                counts = {
                    profile["id"]: profile["reference_count"]
                    for profile in self.snapshot()["profiles"]
                }
                return {
                    "status": "ok",
                    "data": {"added": names, "reference_count": counts[profile_id]},
                }
        except Exception as exc:
            return {
                "status": "error",
                "message": self.plugin._format_user_error_detail(exc),
            }

    async def delete_references(self, profile_id: str) -> dict[str, Any]:
        """Remove explicitly selected files from the current profile alone.

        Args:
            profile_id: Profile owning the selected images.

        Returns:
            Deleted file count or a sanitized error response.
        """
        try:
            async with self._lock:
                payload = await request.json()
                names = payload.get("names") if isinstance(payload, dict) else None
                if not isinstance(names, list) or any(
                    not isinstance(name, str) for name in names
                ):
                    raise ValueError("请选择要删除的参考图")
                count = await self.plugin.persona_profiles.delete_references(
                    profile_id, names
                )
                return {"status": "ok", "data": {"deleted": count}}
        except Exception as exc:
            return {
                "status": "error",
                "message": self.plugin._format_user_error_detail(exc),
            }
