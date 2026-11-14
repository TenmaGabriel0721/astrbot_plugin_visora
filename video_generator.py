"""
通用视频生成模块

面向 NewAPI / OpenAI 兼容中转站的视频生成接口，支持：
- 同步返回视频 URL / base64
- 异步返回 task_id 后轮询
- 可配置提交路径和任务查询路径
- Agnes Video v2.0 的 video_id 轮询与专用参数
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import time
import uuid
from copy import deepcopy
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiofiles
import aiohttp
from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path


class VideoGenerator:
    """NewAPI/OpenAI/Agnes 兼容视频生成器"""

    MAX_RESULT_VIDEO_BYTES = 100 * 1024 * 1024

    def __init__(
        self,
        api_keys: list[str],
        base_url: str,
        model: str,
        timeout: int = 600,
        max_retry_attempts: int = 3,
        proxy: str | None = None,
        api_request_formats: list[str] | str | None = None,
        task_query_format: str | None = None,
        poll_interval: int = 5,
        max_video_size_mb: int = 100,
        duration_field: str = "all",
        max_unknown_polls: int = 6,
        payload_mode: str = "auto",
        frame_rate: int = 24,
        extra_payload: dict[str, Any] | str | None = None,
        query_model_name: bool = True,
        agnes_query_base_url: str = "https://apihub.agnes-ai.com",
    ):
        self.api_keys = api_keys if api_keys else []
        self.current_key_index = 0
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.timeout = int(timeout or 600)
        self.max_retry_attempts = max(1, min(int(max_retry_attempts or 3), 10))
        self.proxy = proxy
        self.api_request_formats = self._normalize_api_formats(api_request_formats)
        self.api_format = self.api_request_formats[0]
        self.task_query_format = (task_query_format or "").strip()
        self.poll_interval = max(1, int(poll_interval or 5))
        self.max_video_size_mb = max(1, int(max_video_size_mb or 100))
        self.duration_field = str(duration_field or "all").strip().lower()
        self.max_unknown_polls = max(1, int(max_unknown_polls or 6))
        self.payload_mode = str(payload_mode or "auto").strip().lower()
        self.frame_rate = max(1, min(int(frame_rate or 24), 60))
        self.extra_payload = self._normalize_extra_payload(extra_payload)
        self.query_model_name = bool(query_model_name)
        self.agnes_query_base_url = (
            agnes_query_base_url or "https://apihub.agnes-ai.com"
        ).rstrip("/")
        self.MAX_RESULT_VIDEO_BYTES = self.max_video_size_mb * 1024 * 1024
        self._session: aiohttp.ClientSession | None = None

    @staticmethod
    def _api_format_to_endpoint(api_format: str | None) -> str:
        value = str(api_format or "openai_videos").strip()
        aliases = {
            "agnes": "v1/videos",
            "agnes_videos": "v1/videos",
            "openai_videos_submit": "v1/videos",
            "openai_videos": "v1/videos/generations",
            "videos": "v1/videos/generations",
            "video": "v1/videos/generations",
            "newapi_videos": "v1/videos/generations",
            "openai_responses": "v1/responses",
        }
        return aliases.get(value.lower(), value or aliases["openai_videos"])

    @classmethod
    def _normalize_api_formats(cls, value, default=None) -> list[str]:
        if value is None or value == "":
            value = default
        if value is None or value == "":
            value = ["v1/videos/generations"]
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
        return normalized or ["v1/videos/generations"]

    @staticmethod
    def _normalize_extra_payload(value: dict[str, Any] | str | None) -> dict[str, Any]:
        if not value:
            return {}
        if isinstance(value, dict):
            return deepcopy(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                logger.warning(
                    "[Visora][Video] video_extra_payload is not a valid JSON object; ignored"
                )
        return {}

    def _get_current_api_key(self) -> str:
        if not self.api_keys:
            return ""
        return self.api_keys[self.current_key_index % len(self.api_keys)]

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _build_endpoint(
        self, base_url: str, format_path: str, model: str | None = None
    ) -> str:
        base = (base_url or self.base_url).rstrip("/")
        path = self._api_format_to_endpoint(format_path).strip().lstrip("/")
        path = path.replace("{model}", model or self.model)
        if base.endswith("/v1") and path.startswith("v1/"):
            path = path[3:]
        elif base.endswith("/v1beta") and path.startswith("v1beta/"):
            path = path[7:]
        return f"{base}/{path}"

    def _build_data_url(self, data: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _classify_video_endpoint(self, api_format: str, endpoint: str) -> str:
        mode = (self.payload_mode or "auto").lower()
        if mode in ("agnes", "agnes_videos"):
            return "agnes_videos"
        if mode in ("flat", "custom_flat"):
            return "custom_flat"
        if mode in ("openai_videos", "openai_chat", "openai_videos_generations"):
            return mode

        model = (self.model or "").lower()
        path = urlparse(endpoint).path.lower().rstrip("/")
        fmt = str(api_format or "").lower().strip()
        if (
            "agnes-video" in model
            or fmt in ("agnes", "agnes_videos")
            or path.endswith("/v1/videos")
        ):
            return "agnes_videos"
        if "chat/completions" in path:
            return "openai_chat"
        if "videos/generations" in path:
            return "openai_videos_generations"
        if path.endswith("/videos"):
            return "openai_videos"
        return "custom_flat"

    def _merge_payload(
        self, payload: dict[str, Any], extra: dict[str, Any]
    ) -> dict[str, Any]:
        merged = deepcopy(payload)
        for key, value in (extra or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_payload(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    def _build_payload(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]] | None,
        duration: int,
        ratio: str,
        resolution: str | None,
        model: str,
    ) -> dict[str, Any]:
        duration_value = int(duration or 5)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "ratio": ratio,
            "aspect_ratio": ratio,
        }
        if self.duration_field in ("all", "duration"):
            payload["duration"] = duration_value
        if self.duration_field in ("all", "duration_seconds"):
            payload["duration_seconds"] = duration_value
        if self.duration_field in ("all", "seconds"):
            payload["seconds"] = str(duration_value)
        if resolution:
            payload["resolution"] = resolution
            payload["quality"] = resolution

        if images_data:
            data_urls = [self._build_data_url(data, mime) for data, mime in images_data]
            # Accept common field names used by relay services.
            payload["image"] = data_urls[0]
            payload["image_url"] = data_urls[0]
            payload["images"] = data_urls
            payload["input_image"] = data_urls[0]

        return payload

    def _build_request_payload(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]] | None,
        image_urls: list[str] | None,
        duration: int,
        ratio: str,
        resolution: str | None,
        model: str,
        endpoint_kind: str,
    ) -> dict[str, Any]:
        if endpoint_kind == "agnes_videos":
            return self._build_agnes_payload(
                prompt, image_urls or [], duration, ratio, resolution, model
            )
        if endpoint_kind == "openai_chat":
            return self._build_openai_chat_payload(
                prompt, images_data, image_urls, duration, ratio, resolution, model
            )
        return self._build_payload(
            prompt, images_data, duration, ratio, resolution, model
        )

    @staticmethod
    def _is_legacy_agnes_v2(model: str) -> bool:
        m = (model or "").lower()
        return "v2.0" in m or ("2.0" in m and "2.5" not in m)

    def _build_agnes_payload(
        self,
        prompt: str,
        image_urls: list[str],
        duration: int,
        ratio: str,
        resolution: str | None,
        model: str,
    ) -> dict[str, Any]:
        m = (model or self.model or "").lower()
        if not self._is_legacy_agnes_v2(m):
            try:
                sec = max(4, min(int(duration or 5), 12))
            except (TypeError, ValueError):
                sec = 5

            ratio_val = str(ratio or "16:9").strip().lower()
            valid_ratios = {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9"}
            aspect_ratio = ratio_val if ratio_val in valid_ratios else "16:9"

            res_val = str(resolution or "").strip().upper()
            if "flash" in m:
                size = "720P"
            elif "1080" in res_val:
                size = "1080P"
            elif "2K" in res_val:
                size = "2K"
            elif "1K" in res_val or "1024" in res_val:
                size = "1K"
            else:
                size = "720P"

            valid_images = [u for u in (image_urls or []) if u]
            mode = "reference" if valid_images else "text"

            payload: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "mode": mode,
                "seconds": str(sec),
                "size": size,
                "aspect_ratio": aspect_ratio,
                "n": 1,
            }
            if mode == "reference":
                max_imgs = 5 if ("flash" in m) else 8
                payload["images"] = valid_images[:max_imgs]

            if self.extra_payload:
                payload = self._merge_payload(payload, self.extra_payload)
            return payload

        width, height = self._map_agnes_size(ratio, resolution)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": self._duration_to_num_frames(duration, self.frame_rate),
            "frame_rate": self.frame_rate,
        }
        if image_urls:
            if len(image_urls) == 1:
                payload["image"] = image_urls[0]
            else:
                payload["extra_body"] = {"image": image_urls}
        if self.extra_payload:
            payload = self._merge_payload(payload, self.extra_payload)
        return payload

    def _build_openai_chat_payload(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]] | None,
        image_urls: list[str] | None,
        duration: int,
        ratio: str,
        resolution: str | None,
        model: str,
    ) -> dict[str, Any]:
        text = f"{prompt}\n\n视频参数：时长 {int(duration or 5)} 秒，比例 {ratio}，分辨率 {resolution or '默认'}。"
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        # A logical reference image may have both a public CDN URL and locally
        # downloaded bytes. OpenAI chat accepts either representation, so prefer
        # the validated bytes and never submit both copies of the same image.
        if images_data:
            for data, mime in images_data:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._build_data_url(data, mime)},
                    }
                )
        else:
            for url in image_urls or []:
                content.append({"type": "image_url", "image_url": {"url": url}})
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }
        if self.extra_payload:
            payload = self._merge_payload(payload, self.extra_payload)
        return payload

    def _map_agnes_size(self, ratio: str, resolution: str | None) -> tuple[int, int]:
        ratio_value = str(ratio or "16:9").strip().lower()
        resolution_value = str(resolution or "720p").strip().lower()
        landscape = {
            "480p": (768, 512),
            "720p": (1152, 768),
            "1080p": (1536, 1024),
        }.get(resolution_value, (1152, 768))
        width, height = landscape
        if ratio_value == "9:16":
            return height, width
        if ratio_value == "1:1":
            side = min(width, height)
            return side, side
        if ratio_value == "4:3":
            return 1024, 768
        if ratio_value == "3:4":
            return 768, 1024
        return width, height

    @staticmethod
    def _duration_to_num_frames(duration: int, frame_rate: int) -> int:
        try:
            seconds = max(1, int(duration or 5))
        except (TypeError, ValueError):
            seconds = 5
        frame_rate = max(1, min(int(frame_rate or 24), 60))
        if frame_rate == 24:
            if seconds <= 3:
                return 81
            if seconds <= 5:
                return 121
            if seconds <= 10:
                return 241
            return 441
        raw = min(seconds * frame_rate, 441)
        candidates = list(range(81, 442, 8))
        return min(candidates, key=lambda item: abs(item - raw))

    def _is_agnes_mode(self, endpoint_kind: str | None = None) -> bool:
        if endpoint_kind == "agnes_videos":
            return True
        mode = (self.payload_mode or "auto").lower()
        return (
            mode in ("agnes", "agnes_videos")
            or "agnes-video" in (self.model or "").lower()
        )

    @staticmethod
    def _is_agnes_submit_endpoint(endpoint: str) -> bool:
        path = urlparse(endpoint).path.lower().rstrip("/")
        return path.endswith("/v1/videos") or path.endswith("/videos")

    async def generate_video(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]] | None = None,
        image_urls: list[str] | None = None,
        duration: int = 5,
        ratio: str = "16:9",
        resolution: str | None = None,
        task_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        """生成视频并返回本地 mp4 路径。"""
        if not self.api_keys:
            return None, "未配置 API Key"
        if not self.base_url:
            return None, "未配置视频 API Base URL"
        if not self.model:
            return None, "未配置视频模型"

        prefix = f"[{task_id}] " if task_id else ""
        last_error = "未知错误"
        formats = self._normalize_api_formats(self.api_request_formats)
        key_index = self.current_key_index % len(self.api_keys)

        for api_format in formats:
            endpoint = self._build_endpoint(self.base_url, api_format, self.model)
            endpoint_kind = self._classify_video_endpoint(api_format, endpoint)
            logger.info(
                f"[Visora][Video] {prefix}Trying video generation: model={self.model}, "
                f"endpoint={endpoint}, mode={endpoint_kind}"
            )
            if endpoint_kind == "agnes_videos" and not self._is_agnes_submit_endpoint(
                endpoint
            ):
                last_error = (
                    f"Agnes 模式只支持 v1/videos 提交路径，已跳过不兼容路径: {endpoint}"
                )
                logger.warning(f"[Visora][Video] {prefix}{last_error}")
                continue
            if endpoint_kind == "agnes_videos" and images_data and not image_urls:
                last_error = (
                    "Agnes 图生视频需要公网图片 URL；当前图片无法生成公网直链。"
                    "请在插件配置中填写 video_public_base_url（AstrBot 面板公网地址）后重试。"
                )
                logger.warning(f"[Visora][Video] {prefix}{last_error}")
                continue
            for retry_attempt in range(self.max_retry_attempts):
                for _ in range(len(self.api_keys)):
                    api_key = self.api_keys[key_index % len(self.api_keys)]
                    (
                        local_path,
                        error,
                        remote_task_started,
                    ) = await self._try_generate_with_key(
                        endpoint=endpoint,
                        endpoint_kind=endpoint_kind,
                        api_key=api_key,
                        prompt=prompt,
                        images_data=images_data or [],
                        image_urls=image_urls or [],
                        duration=duration,
                        ratio=ratio,
                        resolution=resolution,
                        task_id=task_id,
                    )
                    if local_path:
                        logger.info(
                            f"[Visora][Video] {prefix}Video generation completed; local file: {local_path}"
                        )
                        self.current_key_index = key_index % len(self.api_keys)
                        return local_path, None
                    last_error = error or "生成失败"
                    if remote_task_started:
                        # After task_id/video_id is returned, rotating keys or retrying submission
                        # would create another remote task; retry or report errors for the original task only.
                        self.current_key_index = key_index % len(self.api_keys)
                        logger.warning(
                            f"[Visora][Video] {prefix}Remote video task post-processing failed; "
                            f"no resubmission to avoid duplicate tasks: {last_error}"
                        )
                        return None, last_error
                    key_index = (key_index + 1) % len(self.api_keys)
                    if len(self.api_keys) > 1:
                        logger.info(
                            f"[Visora][Video] Switched to the next API key (local index: {key_index})"
                        )
                if retry_attempt < self.max_retry_attempts - 1:
                    await asyncio.sleep(min(2**retry_attempt, 10))
            logger.warning(
                f"[Visora][Video] {prefix}Request path {api_format} attempt failed: {last_error}"
            )

        return None, f"所有视频生成尝试均失败。最后一次错误: {last_error}"

    async def _try_generate_with_key(
        self,
        endpoint: str,
        endpoint_kind: str,
        api_key: str,
        prompt: str,
        images_data: list[tuple[bytes, str]],
        image_urls: list[str],
        duration: int,
        ratio: str,
        resolution: str | None,
        task_id: str | None,
    ) -> tuple[str | None, str | None, bool]:
        """尝试提交一次视频请求。

        第三个返回值表示服务端是否已经接受请求并返回远程 task_id/video_id。
        调用方必须在该值为 True 时停止所有 POST 重试，避免创建重复任务。
        """
        prefix = f"[{task_id}] " if task_id else ""
        remote_task_started = False
        try:
            payload = self._build_request_payload(
                prompt,
                images_data,
                image_urls,
                duration,
                ratio,
                resolution,
                self.model,
                endpoint_kind,
            )
            logger.info(
                f"[Visora][Video] {prefix}Video request parameters: mode={endpoint_kind}, "
                f"duration={duration}, ratio={ratio}, resolution={resolution}, model={payload.get('model')}"
            )
            response_data, error = await self._make_json_request(
                "POST", endpoint, api_key, json_payload=payload
            )
            if error:
                return None, error, False
            if response_data is None:
                return None, "视频 API 未返回有效响应", False

            task_ref = self._extract_task_ref(response_data)
            video_source = self._find_video_source_deep(response_data)
            if video_source:
                if task_ref:
                    remote_task_started = True
                    (
                        local_path,
                        download_error,
                    ) = await self._download_video_source_with_retry(
                        video_source, task_id
                    )
                    return local_path, download_error, True
                local_path, download_error = await self._download_video_source(
                    video_source, task_id
                )
                return local_path, download_error, False

            if task_ref:
                remote_task_started = True
                task_ref["submit_endpoint"] = endpoint
                task_ref["endpoint_kind"] = endpoint_kind
                logger.info(
                    f"[Visora][Video] {prefix}Video task submitted: "
                    f"task_id={task_ref.get('task_id')}, video_id={task_ref.get('video_id')}"
                )
                poll_data, poll_error = await self._poll_task(
                    task_ref, api_key, task_id
                )
                if poll_error:
                    return None, poll_error, True
                video_source = self._find_video_source_deep(poll_data)
                if video_source:
                    (
                        local_path,
                        download_error,
                    ) = await self._download_video_source_with_retry(
                        video_source, task_id
                    )
                    return local_path, download_error, True
                return None, "任务完成响应中未找到视频地址", True

            return None, "响应中未找到视频 URL、base64 或 task_id", False
        except TimeoutError:
            return None, f"视频生成超时 (超过 {self.timeout} 秒)", remote_task_started
        except aiohttp.ClientError as e:
            logger.error(f"[Visora][Video] {prefix}Network request failed: {e}")
            return None, f"网络请求失败: {e}", remote_task_started
        except Exception as e:
            logger.error(
                f"[Visora][Video] {prefix}Generation failed: {e}", exc_info=True
            )
            return None, f"生成失败: {e}", remote_task_started

    async def _make_json_request(
        self,
        method: str,
        url: str,
        api_key: str,
        json_payload: dict[str, Any] | None = None,
    ) -> tuple[Any | None, str | None]:
        session = self._get_session()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            async with session.request(
                method,
                url,
                headers=headers,
                json=json_payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                proxy=self.proxy,
            ) as resp:
                text = await resp.text()
                if resp.status not in (200, 201, 202):
                    err = self._format_http_error(resp.status, text)
                    logger.warning(f"[Visora][Video] API request failed: {err}")
                    return None, err
                try:
                    return await resp.json(content_type=None), None
                except Exception:
                    return None, f"API 返回非 JSON 响应: {text[:300]}"
        except Exception as e:
            return None, str(e)

    def _format_http_error(self, status: int, text: str) -> str:
        detail = text[:500] if text else ""
        if status in (401, 403):
            return f"认证失败或无权限 (HTTP {status}): {detail}"
        if status == 429:
            return f"请求过于频繁或额度不足 (HTTP 429): {detail}"
        if status >= 500:
            return f"服务端错误 (HTTP {status}): {detail}"
        return f"HTTP {status}: {detail}"

    def _extract_task_ref(self, response_data: Any) -> dict[str, str] | None:
        if not isinstance(response_data, dict):
            return None
        if response_data.get("object") == "video":
            task_id = (
                response_data.get("id")
                or response_data.get("task_id")
                or response_data.get("taskId")
            )
        else:
            task_id = (
                response_data.get("task_id")
                or response_data.get("taskId")
                or response_data.get("id")
            )
        video_id = response_data.get("video_id") or response_data.get("videoId")
        task_id = (
            task_id
            or response_data.get("job_id")
            or response_data.get("jobId")
            or response_data.get("request_id")
        )
        status_url = (
            response_data.get("status_url")
            or response_data.get("statusUrl")
            or response_data.get("polling_url")
            or response_data.get("poll_url")
            or response_data.get("pollUrl")
            or response_data.get("location")
        )

        data = response_data.get("data")
        if isinstance(data, dict):
            if not task_id:
                if data.get("object") == "video":
                    task_id = (
                        data.get("id") or data.get("task_id") or data.get("taskId")
                    )
                else:
                    task_id = (
                        data.get("task_id") or data.get("taskId") or data.get("id")
                    )
                task_id = (
                    task_id
                    or data.get("job_id")
                    or data.get("jobId")
                    or data.get("request_id")
                )
            video_id = video_id or data.get("video_id") or data.get("videoId")
            status_url = (
                status_url
                or data.get("status_url")
                or data.get("statusUrl")
                or data.get("polling_url")
                or data.get("poll_url")
                or data.get("pollUrl")
                or data.get("location")
            )
        if task_id or video_id:
            return {
                "task_id": str(task_id or video_id),
                "video_id": str(video_id or ""),
                "status_url": str(status_url or ""),
            }
        return None

    async def _poll_task(
        self,
        task_ref: dict[str, str],
        api_key: str,
        task_id: str | None,
    ) -> tuple[Any | None, str | None]:
        task_ref_id = task_ref.get("task_id", "")
        video_id = task_ref.get("video_id", "")
        endpoint_kind = task_ref.get("endpoint_kind", "")
        status_url = task_ref.get("status_url", "")
        if not status_url:
            if self.task_query_format:
                query_path = self.task_query_format.replace(
                    "{task_id}", task_ref_id
                ).replace("{video_id}", video_id or task_ref_id)
                status_url = self._build_endpoint(self.base_url, query_path, self.model)
            elif endpoint_kind == "agnes_videos" and (video_id or task_ref_id):
                status_url = self._build_agnes_status_url(video_id or task_ref_id)
                logger.info(
                    f"[Visora][Video] Querying the task through the official Agnes endpoint: {status_url}"
                )
            else:
                submit_endpoint = task_ref.get("submit_endpoint", "")
                if submit_endpoint:
                    status_url = f"{submit_endpoint.rstrip('/')}/{task_ref_id}"
                    logger.info(
                        f"[Visora][Video] No video_task_query_format; polling the submission endpoint with task_id: {status_url}"
                    )
                else:
                    return (
                        None,
                        f"任务 {task_ref_id or video_id} 需要轮询，但未配置 video_task_query_format",
                    )

        deadline = time.monotonic() + self.timeout
        unknown_count = 0
        consecutive_request_errors = 0
        seen_task_ids = {item for item in (task_ref_id, video_id) if item}
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll_interval)
            response_data, error = await self._make_json_request(
                "GET", status_url, api_key
            )
            if error:
                consecutive_request_errors += 1
                logger.warning(
                    f"[Visora][Video] [{task_id or task_ref_id or video_id}] "
                    f"polling request failed ({consecutive_request_errors}/{self.max_retry_attempts}); "
                    f"retrying the same task: {error}"
                )
                if consecutive_request_errors >= self.max_retry_attempts:
                    return None, (
                        f"视频任务轮询连续失败 {consecutive_request_errors} 次 "
                        f"(task_id={task_ref_id or video_id}): {error}"
                    )
                continue
            consecutive_request_errors = 0
            status = self._extract_status(response_data)
            video_source = self._find_video_source_deep(response_data)
            if video_source:
                return response_data, None
            if status in (
                "succeeded",
                "success",
                "completed",
                "complete",
                "done",
                "finished",
            ):
                return response_data, None
            if status in (
                "failed",
                "failure",
                "error",
                "cancelled",
                "canceled",
                "expired",
                "rejected",
            ):
                return None, self._extract_error_message(
                    response_data
                ) or f"视频任务失败: {status}"

            if endpoint_kind == "agnes_videos" and isinstance(response_data, dict):
                response_video_id = response_data.get("video_id") or response_data.get(
                    "videoId"
                )
                if response_video_id and str(response_video_id) not in seen_task_ids:
                    seen_task_ids.add(str(response_video_id))
                    status_url = self._build_agnes_status_url(str(response_video_id))
                    logger.info(
                        f"[Visora][Video] Agnes polling returned video_id; switching to the official query: {status_url}"
                    )
                    unknown_count = 0
                    continue

            if not status or status in ("unknown", "none", "null"):
                unknown_count += 1
                if unknown_count == 1 or unknown_count >= self.max_unknown_polls:
                    logger.info(
                        f"[Visora][Video] [{task_id or task_ref_id or video_id}] polling response status is {status or 'unknown'}; continuing to wait: "
                        f"{json.dumps(response_data, ensure_ascii=False)[:800]}"
                    )
                if unknown_count >= self.max_unknown_polls:
                    return None, (
                        f"视频任务连续 {unknown_count} 次返回 {status or '未识别'} 状态，"
                        "可能是任务查询路径不正确或中转站未返回有效状态"
                    )
            else:
                unknown_count = 0
            logger.debug(
                f"[Visora][Video] [{task_id or task_ref_id or video_id}] task status: {status or 'unknown'}"
            )
        return None, f"视频任务轮询超时 (task_id={task_ref_id or video_id})"

    async def _download_video_source_with_retry(
        self,
        source: str,
        task_id: str | None,
    ) -> tuple[str | None, str | None]:
        """有限重试同一个远程任务返回的成品地址，不触发新的视频提交。"""
        last_error = "视频下载失败"
        for attempt in range(self.max_retry_attempts):
            local_path, error = await self._download_video_source(source, task_id)
            if local_path:
                return local_path, None
            last_error = error or last_error
            if attempt < self.max_retry_attempts - 1:
                logger.warning(
                    f"[Visora][Video] [{task_id or 'video'}] result download failed "
                    f"({attempt + 1}/{self.max_retry_attempts}); retrying the same URL: {last_error}"
                )
                await asyncio.sleep(min(2**attempt, 10))
        return None, (
            f"远程视频任务已完成，但成品下载连续失败 {self.max_retry_attempts} 次: "
            f"{last_error}"
        )

    def _build_agnes_status_url(self, video_id: str) -> str:
        parsed = urlparse(self.agnes_query_base_url or "https://apihub.agnes-ai.com")
        base_path = parsed.path.rstrip("/")
        if base_path.endswith("/v1"):
            base_path = base_path[:-3]
        path = f"{base_path}/agnesapi" if base_path else "/agnesapi"
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["video_id"] = video_id
        if self.query_model_name and self.model:
            query["model_name"] = self.model
        return urlunparse(
            (parsed.scheme, parsed.netloc, path, "", urlencode(query), "")
        )

    def _extract_status(self, node: Any) -> str:
        if isinstance(node, dict):
            for key in (
                "status",
                "state",
                "task_status",
                "taskStatus",
                "status_code",
                "statusCode",
                "phase",
            ):
                if key in node and node[key] is not None:
                    return str(node[key]).lower()
            data = node.get("data")
            if isinstance(data, dict):
                return self._extract_status(data)
            if isinstance(data, list):
                for item in data:
                    status = self._extract_status(item)
                    if status:
                        return status
        if isinstance(node, list):
            for item in node:
                status = self._extract_status(item)
                if status:
                    return status
        return ""

    def _extract_error_message(self, node: Any) -> str | None:
        if isinstance(node, dict):
            for key in ("error", "message", "msg", "reason"):
                val = node.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if isinstance(val, dict):
                    nested = self._extract_error_message(val)
                    if nested:
                        return nested
            data = node.get("data")
            if isinstance(data, dict):
                return self._extract_error_message(data)
        return None

    def _find_video_source_deep(self, node: Any) -> str | None:
        source = self._normalize_to_video_source(node)
        if source:
            return source
        if isinstance(node, dict):
            for key, value in node.items():
                key_lower = key.lower()
                if key_lower in (
                    "status_url",
                    "statusurl",
                    "polling_url",
                    "poll_url",
                    "pollurl",
                    "location",
                ):
                    continue
                if key_lower in (
                    "url",
                    "video_url",
                    "videourl",
                    "video",
                    "file_url",
                    "fileurl",
                    "output_url",
                    "outputurl",
                    "download_url",
                    "downloadurl",
                    "result_url",
                    "resulturl",
                    "remixed_from_video_id",
                    "b64_json",
                    "base64",
                    "video_base64",
                    "data",
                    "output",
                    "outputs",
                    "artifacts",
                    "file",
                ):
                    source = self._normalize_to_video_source(value)
                    if source:
                        return source
                deep = self._find_video_source_deep(value)
                if deep:
                    return deep
        elif isinstance(node, list):
            for item in node:
                deep = self._find_video_source_deep(item)
                if deep:
                    return deep
        return None

    def _normalize_to_video_source(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            if s.startswith(("http://", "https://", "data:video/")):
                return s
            url_match = re.search(
                r"https?://[^\s'\"<>]+\.(?:mp4|mov|webm|m4v)(?:\?[^\s'\"<>]*)?", s, re.I
            )
            if url_match:
                return url_match.group(0)
            if self._looks_like_base64(s):
                return f"data:video/mp4;base64,{s.replace(chr(10), '').replace(chr(13), '').strip()}"
            return None
        if isinstance(value, dict):
            for key in (
                "url",
                "video_url",
                "videoUrl",
                "file_url",
                "fileUrl",
                "download_url",
                "downloadUrl",
                "output_url",
                "outputUrl",
                "result_url",
                "resultUrl",
                "remixed_from_video_id",
                "file",
            ):
                nested = value.get(key)
                source = self._normalize_to_video_source(nested)
                if source:
                    return source
            for key in ("b64_json", "base64", "video_base64"):
                nested = value.get(key)
                if isinstance(nested, str):
                    source = self._normalize_to_video_source(nested)
                    if source:
                        return source
        return None

    def _looks_like_base64(self, value: str) -> bool:
        s = value.strip().replace("\n", "").replace("\r", "")
        if len(s) < 512:
            return False
        return re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", s) is not None

    async def _download_video_source(
        self, source: str, task_id: str | None
    ) -> tuple[str | None, str | None]:
        try:
            if source.startswith("data:video/"):
                match = re.match(r"^data:video/[\w\-+.]+;base64,(.*)$", source, re.S)
                if not match:
                    return None, "视频 data URL 格式无效"
                raw_b64 = (
                    match.group(1).replace("\n", "").replace("\r", "").replace(" ", "")
                )
                if len(raw_b64) > self.MAX_RESULT_VIDEO_BYTES * 4 // 3 + 4:
                    return None, f"视频超过大小限制 ({self.max_video_size_mb}MB)"
                data = base64.b64decode(raw_b64)
                if len(data) > self.MAX_RESULT_VIDEO_BYTES:
                    return None, f"视频超过大小限制 ({self.max_video_size_mb}MB)"
                return await self._write_video_bytes(data, task_id), None

            if source.startswith(("http://", "https://")):
                return await self._download_video_url(source, task_id)
            return None, "不支持的视频来源格式"
        except (binascii.Error, ValueError) as e:
            return None, f"视频 Base64 解码失败: {e}"
        except Exception as e:
            logger.error(f"[Visora][Video] Video download failed: {e}", exc_info=True)
            return None, f"视频下载失败: {e}"

    async def _download_video_url(
        self, url: str, task_id: str | None
    ) -> tuple[str | None, str | None]:
        parsed = urlparse(url)
        suffix = os.path.splitext(parsed.path)[1].lower()
        if suffix not in (".mp4", ".mov", ".webm", ".m4v"):
            suffix = ".mp4"
        path = self._build_temp_video_path(task_id, suffix)
        session = self._get_session()
        total = 0
        completed = False
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                proxy=self.proxy,
            ) as resp:
                if resp.status != 200:
                    return None, f"下载视频失败: HTTP {resp.status}"
                async with aiofiles.open(path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1024 * 256):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > self.MAX_RESULT_VIDEO_BYTES:
                            return (
                                None,
                                f"视频超过大小限制 ({self.max_video_size_mb}MB)",
                            )
                        await f.write(chunk)
            if total <= 0:
                return None, "下载到的视频文件为空"
            completed = True
            return path, None
        finally:
            if not completed and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as exc:
                    logger.warning(
                        f"[Visora][Video] Failed to remove an incomplete video file: {path}: {exc}"
                    )

    async def _write_video_bytes(self, data: bytes, task_id: str | None) -> str:
        path = self._build_temp_video_path(task_id, ".mp4")
        try:
            async with aiofiles.open(path, "wb") as f:
                await f.write(data)
            return path
        except BaseException:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
            raise

    def _build_temp_video_path(self, task_id: str | None, suffix: str) -> str:
        temp_dir = get_astrbot_temp_path()
        os.makedirs(temp_dir, exist_ok=True)
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id or "video")
        return os.path.abspath(
            os.path.join(
                temp_dir, f"visora_video_{safe_task_id}_{uuid.uuid4().hex}{suffix}"
            )
        )
