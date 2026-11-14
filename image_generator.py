# SPDX-License-Identifier: AGPL-3.0-only
"""Multi-provider image generation for Visora.

Modified for Visora on 2026-11-15. Upstream provenance is retained in NOTICE.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import re
import socket
from fractions import Fraction
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import aiohttp
from astrbot.api import logger
from PIL import Image

from .image_resolution import normalize_image_resolution, parse_exact_resolution

# Models that reject the OpenAI-compatible size parameter with HTTP 400.
# Match case-insensitively, including model path prefixes or version suffixes.
_OPENAI_SIZE_IGNORING_MODELS = ("z-image-turbo",)


class ImageGenerator:
    """Generate images through Gemini and OpenAI-compatible APIs."""

    MAX_RESULT_IMAGE_BYTES = 20 * 1024 * 1024
    GPT_IMAGE_2_MIN_PIXELS = 655_360
    GPT_IMAGE_2_MAX_PIXELS = 8_294_400
    GPT_IMAGE_2_MAX_EDGE = 3_840
    FOUR_K_WARNING_MIN_LONG_EDGE = 2_800

    def __init__(
        self,
        api_keys: list[str],
        base_url: str,
        model: str,
        fallback_models: list[str | dict[str, Any]] | None = None,
        timeout: int = 300,
        max_retry_attempts: int = 3,
        proxy: str | None = None,
        safety_settings: str = "BLOCK_MEDIUM_AND_ABOVE",
        api_request_formats: list[str] | str | None = None,
        provider_id: str | None = None,
        api_request_format: str | None = None,
        task_timeout: float | None = 600,
    ):
        self.api_keys = api_keys if api_keys else []
        self.current_key_index = 0
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider_id = str(provider_id or "").strip()
        self.fallback_models = fallback_models or []
        self.timeout = timeout
        self.task_timeout = task_timeout
        self.result_metadata: dict[str, Any] = {}
        self.max_retry_attempts = max(1, min(max_retry_attempts, 10))
        self.proxy = proxy
        self.safety_settings = safety_settings
        legacy_formats = self._normalize_api_formats(api_request_formats)
        self.api_request_format = self._normalize_api_formats(
            api_request_format,
            legacy_formats,
        )[0]
        self.api_request_formats = [self.api_request_format]
        self.api_format = self.api_request_format
        self.route_failures: list[str] = []
        self._session: aiohttp.ClientSession | None = None

    @staticmethod
    def _api_format_to_endpoint(api_format: str | None) -> str:
        raw_value = str(api_format or "gemini").strip()
        value = raw_value.lower()
        aliases = {
            "gemini": "v1beta/models/{model}:generateContent",
            "openai_chat": "v1/chat/completions",
            "openai_images": "v1/images/generations",
            "openai_edits": "v1/images/edits",
        }
        return aliases.get(value, raw_value or aliases["gemini"])

    @classmethod
    def _normalize_api_formats(cls, value, default=None) -> list[str]:
        if value is None or value == "":
            value = default
        if value is None or value == "":
            value = ["v1beta/models/{model}:generateContent"]
        if isinstance(value, str):
            raw_items = [
                item.strip() for item in re.split(r"[,\n]", value) if item.strip()
            ]
        elif isinstance(value, (list, tuple)):
            raw_items = []
            for item in value:
                if isinstance(item, str):
                    raw_items.extend(
                        part.strip()
                        for part in re.split(r"[,\n]", item)
                        if part.strip()
                    )
                elif item is not None:
                    raw_items.append(str(item).strip())
        else:
            raw_items = [str(value).strip()]
        normalized = []
        seen = set()
        for item in raw_items:
            endpoint = cls._api_format_to_endpoint(item)
            if endpoint and endpoint not in seen:
                seen.add(endpoint)
                normalized.append(endpoint)
        return normalized or ["v1beta/models/{model}:generateContent"]

    @staticmethod
    def _classify_api_format(format_path: str) -> str:
        value = str(format_path or "").strip().lower()
        if value in ("gemini",) or "generatecontent" in value:
            return "gemini_generate_content"
        if value in ("openai_chat",) or "chat/completions" in value:
            return "openai_chat_completions"
        if value in ("openai_images",) or "images/generations" in value:
            return "openai_images_generations"
        if value in ("openai_edits",) or "images/edits" in value:
            return "openai_images_edits"
        return "unknown"

    @staticmethod
    def _normalize_api_format(api_format: str | None) -> str:
        return ImageGenerator._api_format_to_endpoint(api_format)

    def _get_current_api_key(self) -> str:
        """获取当前使用的 API Key"""
        if not self.api_keys:
            return ""
        return self.api_keys[self.current_key_index % len(self.api_keys)]

    def _rotate_api_key(self):
        """切换到下一个 API Key"""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.info(
                f"[Visora] Switched to the next API key (index: {self.current_key_index})"
            )

    def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        """关闭 aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @staticmethod
    def _is_private_or_loopback_ip(ip: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
            return (
                addr.is_private
                or addr.is_loopback
                or addr.is_link_local
                or addr.is_multicast
                or addr.is_reserved
                or addr.is_unspecified
            )
        except ValueError:
            return True

    async def _is_safe_external_image_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                return False
            hostname = parsed.hostname.strip().lower()
            if hostname in ("localhost",) or hostname.endswith(".localhost"):
                return False
            try:
                infos = await asyncio.to_thread(
                    socket.getaddrinfo, hostname, None, type=socket.SOCK_STREAM
                )
            except socket.gaierror:
                return False
            return not any(
                self._is_private_or_loopback_ip(info[4][0]) for info in infos
            )
        except Exception as e:
            logger.warning(f"[Visora] Result image URL validation failed: {e}")
            return False

    def _build_endpoint(self, base_url: str, format_path: str, model: str) -> str:
        # A provider's ``api_base`` is commonly configured with an API version
        # suffix (for example ``https://relay.example/v1``), while the route
        # format may carry a *different* version (the native Gemini route is
        # ``v1beta/models/...``).  Blindly concatenating those values produces
        # invalid paths such as ``/v1/v1beta/...``.  Parse the base URL so that
        # an explicit version in the route replaces the base URL's trailing
        # version segment; unversioned routes still remain relative to the
        # configured base.
        base = (base_url or self.base_url or "").strip()
        path = self._api_format_to_endpoint(format_path).strip().lstrip("/")
        path = path.replace("{model}", model)

        parsed_base = urlparse(base)
        base_path = parsed_base.path.rstrip("/")
        # Treat any leading ``v<digits>...`` path component as an explicit API
        # version (v1, v1beta, v1beta1, v2-preview, ...).  The broad suffix is
        # intentional so newly introduced version labels do not duplicate a
        # provider's existing version prefix.
        route_version = re.match(r"^(v\d+[A-Za-z0-9._-]*)(?:/|$)", path)
        base_version = re.search(r"/(v\d+[A-Za-z0-9._-]*)$", base_path)
        if route_version and base_version:
            base_path = base_path[: base_version.start()]

        joined_path = f"{base_path}/{path}" if base_path else f"/{path}"
        if parsed_base.scheme or parsed_base.netloc:
            return parsed_base._replace(path=joined_path, params="").geturl()

        # Keep the historical behavior for a non-URL base (useful in tests and
        # for callers that deliberately pass a relative prefix).
        base_prefix = base_path.rstrip("/")
        return f"{base_prefix}/{path}" if base_prefix else f"/{path}"

    def _map_openai_size(
        self,
        aspect_ratio: str,
        image_size: str | None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> str:
        """Map a preset or exact resolution to an OpenAI size string."""
        resolution = normalize_image_resolution(image_size, fallback="1K")
        exact_size = parse_exact_resolution(resolution)
        request_model = str(model or self.model or "").strip().lower()
        is_gpt_image_2 = request_model.rsplit("/", 1)[-1] == "gpt-image-2"

        if exact_size is not None:
            width, height = exact_size
            if is_gpt_image_2:
                self._validate_gpt_image_2_size(width, height)
            return f"{width}x{height}"

        raw_ratio = str(aspect_ratio or "1:1").strip()
        if raw_ratio.lower() == "auto":
            raw_ratio = "1:1"

        try:
            width_part, height_part = raw_ratio.split(":", 1)
            ratio_width = Fraction(width_part.strip())
            ratio_height = Fraction(height_part.strip())
            if ratio_width <= 0 or ratio_height <= 0:
                raise ValueError
        except (TypeError, ValueError, ZeroDivisionError):
            ratio_width = ratio_height = Fraction(1, 1)

        if is_gpt_image_2:
            reduced_ratio = ratio_width / ratio_height
            ratio_numerator = reduced_ratio.numerator
            ratio_denominator = reduced_ratio.denominator
            if max(ratio_numerator, ratio_denominator) > 3 * min(
                ratio_numerator, ratio_denominator
            ):
                raise ValueError("gpt-image-2 的图片宽高比不能超过 3:1")

            target_long_edge = {
                "1K": 1024,
                "2K": 2048,
                "4K": self.GPT_IMAGE_2_MAX_EDGE,
            }.get(resolution, 1024)
            unit_width = 16 * ratio_numerator
            unit_height = 16 * ratio_denominator
            max_scale = self.GPT_IMAGE_2_MAX_EDGE // max(unit_width, unit_height)
            candidates: list[tuple[int, int]] = []
            for scale in range(1, max_scale + 1):
                width = unit_width * scale
                height = unit_height * scale
                pixel_count = width * height
                if not (
                    self.GPT_IMAGE_2_MIN_PIXELS
                    <= pixel_count
                    <= self.GPT_IMAGE_2_MAX_PIXELS
                ):
                    continue
                candidates.append((width, height))

            if not candidates:
                raise ValueError("无法为 gpt-image-2 生成符合官方限制的图片尺寸")

            width, height = min(
                candidates,
                key=lambda size: (
                    abs(max(size) - target_long_edge),
                    max(size),
                    size[0] * size[1],
                ),
            )
            return f"{width}x{height}"

        long_edges = {"1K": 1024, "2K": 2048, "4K": 4096}
        long_edge = long_edges.get(resolution, 1024)
        ratio_width_float = float(ratio_width)
        ratio_height_float = float(ratio_height)

        if ratio_width_float >= ratio_height_float:
            width = long_edge
            height = max(1, round(long_edge * ratio_height_float / ratio_width_float))
        else:
            width = max(1, round(long_edge * ratio_width_float / ratio_height_float))
            height = long_edge
        return f"{width}x{height}"

    def _validate_gpt_image_2_size(self, width: int, height: int) -> None:
        if max(width, height) > self.GPT_IMAGE_2_MAX_EDGE:
            raise ValueError(
                f"gpt-image-2 不支持 {width}x{height}：最长边不能超过 "
                f"{self.GPT_IMAGE_2_MAX_EDGE}px"
            )
        pixel_count = width * height
        if pixel_count > self.GPT_IMAGE_2_MAX_PIXELS:
            raise ValueError(
                f"gpt-image-2 不支持 {width}x{height}：总像素不能超过 "
                f"{self.GPT_IMAGE_2_MAX_PIXELS:,}"
            )
        if pixel_count < self.GPT_IMAGE_2_MIN_PIXELS:
            raise ValueError(
                f"gpt-image-2 不支持 {width}x{height}：总像素不能少于 "
                f"{self.GPT_IMAGE_2_MIN_PIXELS:,}"
            )
        if max(width, height) > 3 * min(width, height):
            raise ValueError("gpt-image-2 的图片宽高比不能超过 3:1")

    @staticmethod
    def _is_invalid_size_error(error_message: str | None) -> bool:
        text = str(error_message or "").lower()
        return "size" in text and any(
            marker in text
            for marker in (
                "invalid",
                "unsupported",
                "not supported",
                "allowed",
                "must be",
                "not one of",
            )
        )

    @staticmethod
    def _model_ignores_size_param(model: str | None) -> bool:
        """这些模型完全不接受 size 参数，构建请求时需省略该字段。"""
        name = str(model or "").rsplit("/", 1)[-1].strip().lower()
        return any(marker in name for marker in _OPENAI_SIZE_IGNORING_MODELS)

    @staticmethod
    def _payload_output_size(payload: dict | aiohttp.FormData) -> str:
        if isinstance(payload, aiohttp.FormData):
            for field_options, _, value in payload._fields:
                if field_options.get("name") == "size":
                    return str(value)
            return "未指定"

        if payload.get("size"):
            return str(payload["size"])
        generation_config = payload.get("generationConfig")
        if isinstance(generation_config, dict):
            image_config = generation_config.get("imageConfig")
            if isinstance(image_config, dict) and image_config.get("imageSize"):
                return str(image_config["imageSize"])
        return "未指定"

    def _select_largest_result_image(
        self,
        images: list[bytes],
        task_id: str | None = None,
    ) -> bytes:
        """Choose the largest decoded candidate; use the last one for ties."""
        prefix = f"[{task_id}] " if task_id else ""
        scored: list[tuple[int, int, int, int]] = []
        for index, image_data in enumerate(images):
            width = height = 0
            try:
                with Image.open(BytesIO(image_data)) as image:
                    width, height = image.size
            except Exception:
                pass
            scored.append((width * height, index, width, height))

        _, selected_index, width, height = max(scored)
        if len(images) > 1:
            logger.info(
                f"[Visora] {prefix}Response contains {len(images)} candidates; "
                f"selected the largest image at index {selected_index + 1} images ({width}x{height})"
            )
        return images[selected_index]

    def _log_result_image_metadata(
        self,
        images: list[bytes],
        requested_size: str | None,
        model: str,
        task_id: str | None = None,
    ) -> list[tuple[int, int]]:
        """Log output metadata and return actual decoded dimensions.

        Args:
            images: Returned image files.
            requested_size: Requested resolution preset or exact dimensions.
            model: Successful model name.
            task_id: Optional diagnostic task ID.

        Returns:
            Dimensions of images whose metadata can be decoded.
        """
        prefix = f"[{task_id}] " if task_id else ""
        normalized_size = str(requested_size or "未指定").strip().upper()
        dimensions = []
        for index, image_data in enumerate(images, 1):
            try:
                with Image.open(BytesIO(image_data)) as image:
                    width, height = image.size
                    image_format = image.format or "unknown"
            except Exception as exc:
                logger.warning(
                    f"[Visora] {prefix}Returned image #{index} failed to inspect image dimensions: {exc}"
                )
                continue

            byte_count = len(image_data)
            dimensions.append((width, height))
            logger.info(
                f"[Visora] {prefix}Returned image #{index}: "
                f"model='{model}', requested resolution='{normalized_size}', "
                f"actual pixels={width}x{height}, format={image_format}, "
                f"bytes={byte_count} ({byte_count / 1024 / 1024:.2f} MiB)"
            )
            if (
                normalized_size == "4K"
                and max(width, height) < self.FOUR_K_WARNING_MIN_LONG_EDGE
            ):
                logger.warning(
                    f"[Visora] {prefix}Requested 4K, but returned image #{index} "
                    f"actual value is only {width}x{height}; the upstream model or relay may have reduced the resolution"
                )
        return dimensions

    def _build_openai_data_url(self, image_data: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _looks_like_url(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        value = value.strip()
        return (
            value.startswith("http://")
            or value.startswith("https://")
            or value.startswith("data:image/")
        )

    def _looks_like_base64(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        s = value.strip().replace("\n", "").replace("\r", "")
        if len(s) < 256:
            return False
        return re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", s) is not None

    def _normalize_to_image_source(
        self, value, default_mime: str = "image/png"
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            if s.startswith("data:image/"):
                return s
            if self._looks_like_url(s):
                return s
            if self._looks_like_base64(s):
                s = s.replace(chr(10), "").replace(chr(13), "").strip()
                return f"data:{default_mime};base64,{s}"
            return None
        if isinstance(value, dict):
            for key in ("url", "image_url", "file_url", "link", "src", "image"):
                if key in value:
                    nested = value[key]
                    if key == "image_url" and isinstance(nested, dict):
                        nested = nested.get("url")
                    normalized = self._normalize_to_image_source(nested, default_mime)
                    if normalized:
                        return normalized
            for key in ("b64_json", "base64", "image_base64", "data", "image_data"):
                if key in value and isinstance(value[key], str):
                    normalized = self._normalize_to_image_source(
                        value[key], default_mime
                    )
                    if normalized:
                        return normalized
        return None

    def _find_image_source_deep(self, node) -> str | None:
        normalized = self._normalize_to_image_source(node)
        if normalized:
            return normalized
        if isinstance(node, dict):
            for key, value in node.items():
                if key in (
                    "url",
                    "image_url",
                    "file_url",
                    "link",
                    "src",
                    "image",
                    "b64_json",
                    "base64",
                    "image_base64",
                    "data",
                    "image_data",
                ):
                    normalized = self._normalize_to_image_source(value)
                    if normalized:
                        return normalized
                deep = self._find_image_source_deep(value)
                if deep:
                    return deep
        elif isinstance(node, list):
            for item in node:
                deep = self._find_image_source_deep(item)
                if deep:
                    return deep
        return None

    async def _download_image_source(self, source: str) -> bytes | None:
        try:
            if not source:
                return None
            if source.startswith("data:image/"):
                m = re.match(r"^data:image/[\w\-+.]+;base64,(.*)$", source, re.S)
                if not m:
                    return None
                raw_b64 = (
                    m.group(1).replace("\n", "").replace("\r", "").replace(" ", "")
                )
                if len(raw_b64) > self.MAX_RESULT_IMAGE_BYTES * 4 // 3 + 4:
                    logger.warning(
                        "[Visora] Data URL result image exceeds the size limit"
                    )
                    return None
                data = base64.b64decode(raw_b64)
                if len(data) > self.MAX_RESULT_IMAGE_BYTES:
                    logger.warning(
                        "[Visora] Data URL result image exceeds the size limit"
                    )
                    return None
                return data
            if source.startswith("http://") or source.startswith("https://"):
                if not await self._is_safe_external_image_url(source):
                    logger.warning(
                        f"[Visora] Rejected an unsafe result image URL: {source[:120]}"
                    )
                    return None
                session = self._get_session()
                async with session.get(
                    source,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    proxy=self.proxy,
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"[Visora] Failed to download result image: HTTP {resp.status} | {source[:120]}"
                        )
                        return None
                    data = await resp.read()
                    if len(data) > self.MAX_RESULT_IMAGE_BYTES:
                        logger.warning("[Visora] Result image exceeds the size limit")
                        return None
                    return data
        except (binascii.Error, ValueError) as e:
            logger.warning(f"[Visora] Failed to decode Base64 result image: {e}")
        except Exception as e:
            logger.warning(f"[Visora] Result image download failed: {e}")
        return None

    def _sync_convert_image_format(
        self, image_data: bytes, mime_type: str
    ) -> tuple[bytes, str]:
        """同步的图片格式转换逻辑（在线程池中执行）"""
        try:
            img = Image.open(BytesIO(image_data))
            # Flatten transparent images.
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                elif img.mode == "LA":
                    img = img.convert("RGBA")
                # In RGBA mode, use the fourth channel as the alpha mask.
                background.paste(img, mask=img.split()[3])
                img = background
            # Convert to JPEG.
            output = BytesIO()
            img.save(output, format="JPEG", quality=95)
            logger.debug("[Visora] Image format conversion succeeded")
            return output.getvalue(), "image/jpeg"
        except Exception as e:
            logger.error(f"[Visora] Image format conversion failed: {e}")
            return image_data, mime_type

    def _detect_mime_type(self, data: bytes) -> str:
        """检测图片 MIME 类型"""
        mime = "application/octet-stream"
        if data.startswith(b"\xff\xd8"):
            mime = "image/jpeg"
        elif data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            mime = "image/gif"
        elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            mime = "image/webp"
        elif len(data) > 12 and data[4:8] == b"ftyp":
            brand = data[8:12]
            if brand in (b"heic", b"heix", b"heim", b"heis"):
                mime = "image/heic"
            elif brand in (b"mif1", b"msf1", b"heif"):
                mime = "image/heif"
        logger.debug(f"[Visora] Detect MIME: {mime}")
        return mime

    async def _convert_image_format(
        self, image_data: bytes, mime_type: str
    ) -> tuple[bytes, str]:
        """转换不支持的图片格式为 JPEG"""
        real_mime = self._detect_mime_type(image_data)
        supported_formats = [
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/heic",
            "image/heif",
        ]
        if real_mime in supported_formats:
            return image_data, real_mime
        logger.info(f"[Visora] Converting image format: {mime_type} -> image/jpeg")
        return await asyncio.to_thread(
            self._sync_convert_image_format, image_data, mime_type
        )

    async def generate_image(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        image_size: str | None = None,
        task_id: str | None = None,
    ) -> tuple[list[bytes] | None, str | None]:
        """Bound conversion, model fallback, and downloads by one task deadline.

        Args:
            prompt: Image generation or editing prompt.
            images_data: Optional immutable identity or editing references.
            aspect_ratio: Requested aspect ratio.
            image_size: Resolution preset or exact dimensions.
            task_id: Optional diagnostic task ID.

        Returns:
            Image bytes and no error, or an error describing the failed task.
        """
        self.result_metadata = {}
        try:
            return await asyncio.wait_for(
                self._generate_image_routes(
                    prompt, images_data, aspect_ratio, image_size, task_id
                ),
                timeout=self.task_timeout,
            )
        except TimeoutError:
            self.result_metadata = {}
            return None, f"生图任务超过总时限 {self.task_timeout:g} 秒，已停止后续回退"

    async def _generate_image_routes(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        image_size: str | None = None,
        task_id: str | None = None,
    ) -> tuple[list[bytes] | None, str | None]:
        """Generate images through ordered, single-format model routes.

        Args:
            prompt: Image generation or editing prompt.
            images_data: Optional reference images and MIME types.
            aspect_ratio: Requested output aspect ratio.
            image_size: Optional 1K, 2K, or 4K output size.
            task_id: Optional identifier included in diagnostic logs.

        Returns:
            Generated image bytes and no error, or no images and an error message.
        """
        self.route_failures = []
        prompt = str(prompt or "").strip()
        if not prompt:
            return None, "图片生成提示词不能为空"

        prefix = f"[{task_id}] " if task_id else ""

        attempts = [
            {
                "provider_id": self.provider_id,
                "model": self.model,
                "api_keys": self.api_keys,
                "base_url": self.base_url,
                "api_request_format": self.api_request_format,
            }
        ]
        seen_provider_ids = {self.provider_id} if self.provider_id else set()

        for fb in self.fallback_models:
            if isinstance(fb, dict):
                fallback_provider_id = str(fb.get("provider_id") or "").strip()
                if fallback_provider_id and fallback_provider_id in seen_provider_ids:
                    continue
                if fallback_provider_id:
                    seen_provider_ids.add(fallback_provider_id)
                fallback_format = fb.get("api_request_format")
                if not fallback_format:
                    fallback_format = self._normalize_api_formats(
                        fb.get("api_request_formats"),
                        [self.api_request_format],
                    )[0]
                attempts.append(
                    {
                        **fb,
                        "provider_id": fallback_provider_id,
                        "api_request_format": self._normalize_api_format(
                            fallback_format
                        ),
                    }
                )
            else:
                if fb == self.model:
                    continue
                attempts.append(
                    {
                        "provider_id": "",
                        "model": fb,
                        "api_keys": self.api_keys,
                        "base_url": self.base_url,
                        "api_request_format": self.api_request_format,
                    }
                )

        converted_images_list: list[tuple[bytes, str]] = []
        if images_data:
            for image_data, mime_type in images_data:
                converted_data, converted_mime = await self._convert_image_format(
                    image_data, mime_type
                )
                converted_images_list.append((converted_data, converted_mime))

        # Keep a request-owned immutable snapshot. Each route receives a fresh
        # list so a failed provider attempt cannot consume or clear references
        # before the next fallback route runs.
        converted_images = tuple(converted_images_list)

        last_error = "未知错误"

        for route_index, attempt in enumerate(attempts):
            attempt_model = attempt.get("model", self.model)
            attempt_api_keys = attempt.get("api_keys", self.api_keys) or []
            attempt_base_url = (
                attempt.get("base_url", self.base_url) or self.base_url
            ).rstrip("/")
            attempt_format = self._normalize_api_formats(
                attempt.get("api_request_format"),
                [self.api_request_format],
            )[0]

            route_error: str | None = None
            if not attempt_api_keys:
                route_error = f"模型 {attempt_model} 的提供商未配置 API Key"
                logger.warning(f"[Visora] {prefix}{route_error}")
            else:
                format_kind = self._classify_api_format(attempt_format)
                logger.info(
                    f"[Visora] {prefix}Trying model: {attempt_model} "
                    f"(request path: {attempt_format}, type: {format_kind}, "
                    f"references: {len(converted_images)} images, "
                    f"provider base URL: {attempt_base_url})"
                )
                if format_kind == "unknown":
                    route_error = f"不支持的 API 请求路径: {attempt_format}"
                    logger.warning(f"[Visora] {prefix}{route_error}")
                elif format_kind == "openai_images_generations" and converted_images:
                    route_error = (
                        "当前 v1/images/generations 请求格式不支持参考图，已跳过该模型"
                    )
                    logger.info(f"[Visora] {prefix}{route_error}")
                else:
                    key_index = self.current_key_index % len(attempt_api_keys)
                    for retry_attempt in range(self.max_retry_attempts):
                        for _ in range(len(attempt_api_keys)):
                            api_key = attempt_api_keys[
                                key_index % len(attempt_api_keys)
                            ]
                            result = await self._try_generate_with_current_key(
                                prompt,
                                list(converted_images),
                                aspect_ratio,
                                image_size,
                                task_id,
                                model=attempt_model,
                                base_url=attempt_base_url,
                                api_key=api_key,
                                api_format=attempt_format,
                            )
                            if result[0] is not None:
                                self.result_metadata.update(
                                    provider_id=attempt.get("provider_id") or "",
                                    model=attempt_model,
                                    request_format=attempt_format,
                                )
                                self.current_key_index = key_index % len(
                                    attempt_api_keys
                                )
                                return result
                            route_error = result[1] or "生成失败"
                            key_index = (key_index + 1) % len(attempt_api_keys)
                            if len(attempt_api_keys) > 1:
                                logger.info(
                                    "[Visora] Switched to the next API key "
                                    f"(local index: {key_index})"
                                )
                        if retry_attempt < self.max_retry_attempts - 1:
                            await asyncio.sleep(min(2**retry_attempt, 10))
                    logger.warning(
                        f"[Visora] {prefix}Model {attempt_model} "
                        f"Request path {attempt_format} attempt failed: {route_error}"
                    )

            last_error = route_error or "生成失败"
            short_error = re.sub(r"\s+", " ", str(last_error)).strip()
            short_error = re.sub(
                r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
                r"\1[REDACTED]",
                short_error,
            )
            short_error = re.sub(
                r"(?i)(api[\s_-]?key|access[\s_-]?token|secret)(\s*[:=]\s*)[^\s,;]+",
                r"\1\2[REDACTED]",
                short_error,
            )
            if len(short_error) > 160:
                short_error = short_error[:157] + "..."
            route_label = "主路由" if route_index == 0 else f"回退路由 {route_index}"
            self.route_failures.append(f"{route_label}：{short_error or '生成失败'}")

        failure_summary = "；".join(self.route_failures)
        return None, f"所有生图路由均失败：{failure_summary or last_error}"

    async def _try_generate_with_current_key(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]],
        aspect_ratio: str,
        image_size: str | None,
        task_id: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_format: str | None = None,
    ) -> tuple[list[bytes] | None, str | None]:
        """使用当前 API Key 尝试生成图片"""
        prefix = f"[{task_id}] " if task_id else ""
        self.result_metadata = {}
        effective_size = image_size
        resolution_downgraded = False
        try:
            request_model = model or self.model
            request_api_format = self._normalize_api_format(
                api_format or self.api_format
            )
            format_kind = self._classify_api_format(request_api_format)
            if format_kind == "unknown":
                return None, f"不支持的 API 请求路径: {request_api_format}"
            if format_kind == "openai_images_edits" and not images_data:
                return None, "当前 v1/images/edits 请求格式需要参考图"
            if format_kind == "openai_images_generations" and images_data:
                return None, "当前 v1/images/generations 请求格式不支持参考图"
            payload = self._build_request_payload(
                prompt,
                images_data,
                aspect_ratio,
                image_size,
                request_model,
                request_api_format,
                base_url or self.base_url,
            )
            logger.info(
                f"[Visora] {prefix}Output request parameters: "
                f"model='{request_model}', resolution='{str(image_size or 'unspecified').upper()}', "
                f"aspect ratio='{aspect_ratio}', size='{self._payload_output_size(payload)}'"
            )
            session = self._get_session()
            result = await self._make_api_request(
                session,
                payload,
                task_id,
                request_model,
                base_url or self.base_url,
                api_key or self._get_current_api_key(),
                request_api_format,
            )

            if (
                isinstance(result, tuple)
                and normalize_image_resolution(image_size, fallback="1K") == "4K"
                and format_kind in ("openai_images_generations", "openai_images_edits")
                and self._is_invalid_size_error(result[1])
                and not self._model_ignores_size_param(request_model)
            ):
                logger.warning(
                    f"[Visora] {prefix}The current endpoint rejected 4K size; retrying once at 2K"
                )
                fallback_payload = self._build_request_payload(
                    prompt,
                    images_data,
                    aspect_ratio,
                    "2K",
                    request_model,
                    request_api_format,
                    base_url or self.base_url,
                )
                effective_size = "2K"
                resolution_downgraded = True
                logger.info(
                    f"[Visora] {prefix}Downgrade retry with size='"
                    f"{self._payload_output_size(fallback_payload)}'"
                )
                result = await self._make_api_request(
                    session,
                    fallback_payload,
                    task_id,
                    request_model,
                    base_url or self.base_url,
                    api_key or self._get_current_api_key(),
                    request_api_format,
                )

            # _make_api_request returns either response data or (None, error_message).
            if isinstance(result, tuple):
                return result  # Return the error tuple directly.

            response_data = result
            if response_data is None:
                return None, "API 请求失败，未返回有效响应"
            result_image_data = await self._extract_image_from_response(
                response_data, task_id
            )
            if result_image_data:
                dimensions = self._log_result_image_metadata(
                    result_image_data,
                    image_size,
                    request_model,
                    task_id,
                )
                self.result_metadata = {
                    "model": request_model,
                    "requested_resolution": image_size or "",
                    "effective_resolution": effective_size or "",
                    "resolution_downgraded": resolution_downgraded,
                    "dimensions": dimensions,
                }
                return result_image_data, None
            return None, "响应中未找到图片数据，可能是模型拒绝了请求"
        except TimeoutError:
            return None, f"生成超时 (超过 {self.timeout} 秒)"
        except aiohttp.ClientError as e:
            error_msg = str(e)
            logger.error(f"[Visora] {prefix}Network request failed: {error_msg}")
            return None, f"网络请求失败: {error_msg}"
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[Visora] {prefix}Generation failed: {error_msg}")
            return None, f"生成失败: {error_msg}"

    def _build_request_payload(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]],
        aspect_ratio: str,
        image_size: str | None,
        model: str | None = None,
        api_format: str | None = None,
        base_url: str | None = None,
    ) -> dict | aiohttp.FormData:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("图片生成提示词不能为空")

        request_model = model or self.model
        request_api_format = self._normalize_api_format(api_format or self.api_format)
        format_kind = self._classify_api_format(request_api_format)

        if format_kind == "openai_images_generations":
            if images_data:
                raise ValueError("当前 v1/images/generations 请求格式不支持参考图")
            payload: dict[str, Any] = {
                "model": request_model,
                "prompt": prompt,
                "n": 1,
            }
            if not self._model_ignores_size_param(request_model):
                payload["size"] = self._map_openai_size(
                    aspect_ratio,
                    image_size,
                    request_model,
                    base_url,
                )
            return payload

        if format_kind == "openai_images_edits":
            form = aiohttp.FormData()
            form.add_field("model", request_model)
            form.add_field("prompt", prompt)
            form.add_field(
                "size",
                self._map_openai_size(
                    aspect_ratio,
                    image_size,
                    request_model,
                    base_url,
                ),
            )
            edit_images = images_data or []
            field_name = "image" if len(edit_images) == 1 else "image[]"
            for idx, (image_data, mime_type) in enumerate(edit_images):
                ext = "jpg" if mime_type == "image/jpeg" else mime_type.split("/")[-1]
                form.add_field(
                    field_name,
                    image_data,
                    filename=f"image_{idx}.{ext}",
                    content_type=mime_type,
                )
            return form

        if format_kind == "openai_chat_completions":
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for image_data, mime_type in images_data or []:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._build_openai_data_url(image_data, mime_type)
                        },
                    }
                )
            payload: dict[str, Any] = {
                "model": request_model,
                "messages": [{"role": "user", "content": content}],
            }
            if aspect_ratio and aspect_ratio != "auto":
                payload["aspect_ratio"] = aspect_ratio
            normalized_size = (
                normalize_image_resolution(image_size) if image_size else None
            )
            if normalized_size:
                payload["image_size"] = normalized_size
            if "gemini" in request_model.lower():
                google_image_config: dict[str, str] = {}
                if aspect_ratio and aspect_ratio != "auto":
                    google_image_config["aspect_ratio"] = aspect_ratio
                if normalized_size:
                    google_image_config["image_size"] = normalized_size
                if google_image_config:
                    # NewAPI converts this OpenAI Chat extension back into
                    # Gemini's generationConfig.imageConfig when the eventual
                    # upstream channel is a native Gemini channel.  Keeping it
                    # alongside the common top-level compatibility fields also
                    # lets OpenAI-compatible relays forward the exact 4K/ratio
                    # request without relying on a lossy Gemini -> Chat rewrite.
                    payload["extra_body"] = {
                        "google": {"image_config": google_image_config}
                    }
            return payload

        generation_config = {"responseModalities": ["IMAGE"]}
        image_config = {}
        if aspect_ratio and aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        if image_size and "gemini-3" in request_model.lower():
            image_config["imageSize"] = normalize_image_resolution(image_size)
        if image_config:
            generation_config["imageConfig"] = image_config
        safety_settings = []
        if self.safety_settings:
            for category in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
                "HARM_CATEGORY_CIVIC_INTEGRITY",
            ]:
                safety_settings.append(
                    {"category": category, "threshold": self.safety_settings}
                )
        parts = [{"text": prompt}]
        if images_data:
            for image_data, mime_type in images_data:
                encoded_data = base64.b64encode(image_data).decode("utf-8")
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_data,
                        }
                    }
                )
        return {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
            "safetySettings": safety_settings,
        }

    @staticmethod
    def _is_unsupported_param_error(error_message: str, param_name: str) -> bool:
        text = (error_message or "").lower()
        param = param_name.lower()
        return param in text and (
            "unsupportedparamserror" in text
            or "not supported" in text
            or "unsupported" in text
            or "unknown parameter" in text
            or "unrecognized" in text
            or "extra inputs are not permitted" in text
            or "extra_forbidden" in text
            or "drop_params" in text
        )

    async def _make_api_request(
        self,
        session: aiohttp.ClientSession,
        payload: dict | aiohttp.FormData,
        task_id: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_format: str | None = None,
    ) -> dict | tuple[None, str] | None:
        prefix = f"[{task_id}] " if task_id else ""
        request_model = model or self.model
        request_api_format = self._normalize_api_format(api_format or self.api_format)
        format_kind = self._classify_api_format(request_api_format)
        request_base_url = (base_url or self.base_url).rstrip("/")
        url = self._build_endpoint(request_base_url, request_api_format, request_model)
        headers = {}
        if format_kind == "gemini_generate_content":
            headers["Content-Type"] = "application/json"
            headers["x-goog-api-key"] = api_key or self._get_current_api_key()
        else:
            headers["Authorization"] = (
                f"Bearer {api_key or self._get_current_api_key()}"
            )
            if not isinstance(payload, aiohttp.FormData):
                headers["Content-Type"] = "application/json"

        request_kwargs = {
            "headers": headers,
            "timeout": aiohttp.ClientTimeout(total=self.timeout),
            "proxy": self.proxy,
        }
        if isinstance(payload, aiohttp.FormData):
            request_kwargs["data"] = payload
        else:
            request_kwargs["json"] = payload

        async with session.post(url, **request_kwargs) as response:
            logger.debug(f"[Visora] {prefix}API Response Status: {response.status}")
            if response.status != 200:
                error_text = await response.text()
                # Truncate long error details before logging.
                error_preview = (
                    error_text[:200] + "..." if len(error_text) > 200 else error_text
                )
                logger.error(
                    f"[Visora] {prefix}API error: {response.status} - {error_preview}"
                )
                # Return a structured error with details.
                # Try parsing a JSON error response.
                try:
                    error_json = await response.json() if error_text else {}
                    error_message = error_json.get("error", {}).get(
                        "message", error_text
                    )
                except Exception:
                    error_message = error_text

                # Some strict OpenAI-compatible chat relays reject extension
                # fields. Gracefully retry without those compatibility fields.
                retry_fields: list[str] = []
                retry_description = ""
                if response.status == 400 and isinstance(payload, dict):
                    if format_kind == "openai_chat_completions" and any(
                        field in payload
                        and self._is_unsupported_param_error(error_message, field)
                        for field in ("aspect_ratio", "image_size")
                    ):
                        retry_fields = ["aspect_ratio", "image_size"]
                        retry_description = "aspect_ratio/image_size 兼容字段"

                if retry_fields:
                    retry_payload = dict(payload)
                    for field in retry_fields:
                        retry_payload.pop(field, None)
                    logger.warning(
                        f"[Visora] {prefix}The current endpoint does not support {retry_description}; "
                        "removed the fields and retried automatically"
                    )
                    retry_kwargs = dict(request_kwargs)
                    retry_kwargs.pop("json", None)
                    retry_kwargs["json"] = retry_payload
                    async with session.post(url, **retry_kwargs) as retry_response:
                        logger.debug(
                            f"[Visora] {prefix}API Retry Response Status: {retry_response.status}"
                        )
                        if retry_response.status == 200:
                            try:
                                return await retry_response.json()
                            except Exception as e:
                                logger.error(
                                    f"[Visora] {prefix}Failed to parse retry response JSON: {e}"
                                )
                                return None, f"解析响应失败: {str(e)}"
                        retry_error_text = await retry_response.text()
                        retry_preview = (
                            retry_error_text[:200] + "..."
                            if len(retry_error_text) > 200
                            else retry_error_text
                        )
                        logger.error(
                            f"[Visora] {prefix}API retry error: {retry_response.status} - {retry_preview}"
                        )
                        try:
                            retry_error_json = (
                                await retry_response.json() if retry_error_text else {}
                            )
                            retry_error_message = retry_error_json.get("error", {}).get(
                                "message", retry_error_text
                            )
                        except Exception:
                            retry_error_message = retry_error_text
                        short_retry_error = (
                            retry_error_message[:100]
                            if len(retry_error_message) > 100
                            else retry_error_message
                        )
                        return (
                            None,
                            f"API 错误 ({retry_response.status}): {short_retry_error}",
                        )

                if response.status == 401:
                    return None, "API Key 无效或已过期"
                elif response.status == 403:
                    return None, "API 访问被拒绝，请检查权限"
                elif response.status == 429:
                    return None, "API 请求频率超限，请稍后再试"
                else:
                    # Limit the error excerpt to 100 characters.
                    short_error = (
                        error_message[:100]
                        if len(error_message) > 100
                        else error_message
                    )
                    return None, f"API 错误 ({response.status}): {short_error}"
            try:
                return await response.json()
            except Exception as e:
                logger.error(f"[Visora] {prefix}Failed to parse response JSON: {e}")
                return None, f"解析响应失败: {str(e)}"

    async def _extract_image_from_response(
        self, response: dict, task_id: str | None = None
    ) -> list[bytes] | None:
        """从多种 API 响应中提取图片数据（兼容 Gemini/OpenAI/流式中转）"""
        prefix = f"[{task_id}] " if task_id else ""
        images: list[bytes] = []

        async def append_source(source: str | None):
            if not source:
                return
            data = await self._download_image_source(source)
            if data:
                images.append(data)

        def extract_image_url_from_text(text: str) -> str | None:
            if not text:
                return None
            md_match = re.search(r"!\[[^\]]*\]\(([^)]+)\)", text)
            if md_match:
                return md_match.group(1).strip()
            data_match = re.search(
                r"(data:image/[\w\-+.]+;base64,[A-Za-z0-9+/=\s]+)",
                text,
                re.S,
            )
            if data_match:
                return data_match.group(1).replace(" ", "").strip()
            url_match = re.search(r"(https?://\S+)", text)
            if url_match:
                return url_match.group(1).strip()
            return None

        try:
            if not isinstance(response, dict):
                logger.warning(
                    f"[Visora] {prefix}Response is not a dict: {type(response)}"
                )
                return None
            candidates = response.get("candidates", [])
            if isinstance(candidates, list) and candidates:
                for candidate_index, candidate in enumerate(candidates, 1):
                    if not isinstance(candidate, dict):
                        continue
                    finish_reason = candidate.get("finishReason")
                    if finish_reason and finish_reason not in (
                        "STOP",
                        "MAX_TOKENS",
                    ):
                        logger.warning(
                            f"[Visora] {prefix}Candidate #{candidate_index} "
                            f"generation was interrupted: {finish_reason}"
                        )
                    content = candidate.get("content", {})
                    parts = (
                        content.get("parts", []) if isinstance(content, dict) else []
                    )
                    if not isinstance(parts, list):
                        continue
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        inline_data = part.get("inline_data") or part.get("inlineData")
                        if isinstance(inline_data, dict):
                            data = inline_data.get("data")
                            if data:
                                try:
                                    images.append(base64.b64decode(data))
                                except Exception as e:
                                    logger.warning(
                                        f"[Visora] {prefix}Failed to decode inlineData: {e}"
                                    )
                        text_content = part.get("text")
                        if isinstance(text_content, str):
                            source = extract_image_url_from_text(text_content)
                            if source:
                                await append_source(source)
                if images:
                    return [self._select_largest_result_image(images, task_id)]
            data_list = response.get("data")
            if isinstance(data_list, list) and data_list:
                for item in data_list:
                    if not isinstance(item, dict):
                        continue
                    if item.get("url"):
                        image_count_before_download = len(images)
                        await append_source(item["url"])
                        if len(images) > image_count_before_download:
                            continue
                    if item.get("b64_json"):
                        try:
                            images.append(base64.b64decode(item["b64_json"]))
                        except Exception as e:
                            logger.warning(
                                f"[Visora] {prefix}Failed to decode b64_json: {e}"
                            )
                if images:
                    return images
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                collected_text_parts: list[str] = []
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            collected_text_parts.append(content)
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict):
                                    if isinstance(block.get("text"), str):
                                        collected_text_parts.append(block["text"])
                                    image_url = block.get("image_url")
                                    if isinstance(image_url, str):
                                        await append_source(image_url)
                                    elif isinstance(image_url, dict) and isinstance(
                                        image_url.get("url"), str
                                    ):
                                        await append_source(image_url["url"])
                        msg_images = message.get("images")
                        if isinstance(msg_images, list):
                            for img in msg_images:
                                if isinstance(img, str):
                                    await append_source(img)
                                elif isinstance(img, dict):
                                    if isinstance(img.get("url"), str):
                                        await append_source(img["url"])
                                    elif isinstance(img.get("image_url"), str):
                                        await append_source(img["image_url"])
                                    elif isinstance(
                                        img.get("image_url"), dict
                                    ) and isinstance(img["image_url"].get("url"), str):
                                        await append_source(img["image_url"]["url"])
                    delta = choice.get("delta", {})
                    if isinstance(delta, dict):
                        delta_content = delta.get("content")
                        if isinstance(delta_content, str):
                            collected_text_parts.append(delta_content)
                merged_text = "\n".join(collected_text_parts).strip()
                if merged_text:
                    source = extract_image_url_from_text(merged_text)
                    if source:
                        await append_source(source)
                if images:
                    return images
            for key in ("content", "text", "output"):
                value = response.get(key)
                if isinstance(value, str):
                    source = extract_image_url_from_text(value)
                    if source:
                        await append_source(source)
            if images:
                return images
            logger.warning(
                f"[Visora] {prefix}No image found in the response; top-level fields: {list(response.keys())[:20]}"
            )
            return None
        except Exception as e:
            logger.error(f"[Visora] {prefix}Failed to parse response: {e}")
            return None
