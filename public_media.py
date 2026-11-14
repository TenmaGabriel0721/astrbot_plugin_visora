"""为外部视频 API（如 Agnes）提供参考图的公网直链。

QQ 图床在海外经常被拦、本地图又没有外网地址，而 Agnes 这类接口只接受
公网图片 URL。这里把已下载的图片字节转存为临时文件，并通过 AstrBot
面板的免鉴权路由 ``/api/v1/plugins/astrbot_plugin_visora/media/<token>``
提供直链。与核心 ``file_token_service`` 的单次弹出不同，token 在 TTL 内
可被多次访问，避免对端重试时 404。
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from starlette.responses import FileResponse, Response

from .temp_media import _image_suffix

MEDIA_ROUTE_PATH = "/api/v1/plugins/astrbot_plugin_visora/media"
_QUART_ROUTE_RULE = f"{MEDIA_ROUTE_PATH}/<token>"
_CONVERTED_ROUTE_PATH = f"{MEDIA_ROUTE_PATH}/{{token}}"
_STATIC_CATCHALL_PATH = "/{static_path:path}"
_REGISTRY_STATE_KEY = "visora_image_public_media_registry"
DEFAULT_TTL_SECONDS = 3600


class PublicMediaRegistry:
    """token -> (本地文件路径, 过期时间)。TTL 内可重复访问。"""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._entries: dict[str, tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    async def register(self, data: bytes) -> str | None:
        """把图片字节转存到临时文件并注册 token。"""
        if not data:
            return None
        temp_dir = get_astrbot_temp_path()
        os.makedirs(temp_dir, exist_ok=True)
        token = uuid.uuid4().hex
        path = os.path.abspath(
            os.path.join(
                temp_dir,
                f"visora_video_public_{token}{_image_suffix(data)}",
            )
        )
        try:
            await asyncio.to_thread(self._write_file, path, data)
        except Exception as exc:
            logger.warning(
                f"[Visora][Video] Failed to publish a reference image: {exc}"
            )
            return None
        async with self._lock:
            self._cleanup_expired_locked()
            self._entries[token] = (path, time.time() + self.ttl_seconds)
        return token

    @staticmethod
    def _write_file(path: str, data: bytes) -> None:
        with open(path, "wb") as file:
            file.write(data)

    def _cleanup_expired_locked(self) -> None:
        now = time.time()
        expired = [t for t, (_, expire) in self._entries.items() if expire < now]
        for token in expired:
            path, _ = self._entries.pop(token)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(
                    f"[Visora][Video] Failed to remove an expired public reference image: {path}: {exc}"
                )

    def resolve(self, token: str) -> str | None:
        """返回 token 对应的本地文件路径；过期或文件缺失时返回 None。"""
        entry = self._entries.get(str(token or ""))
        if not entry:
            return None
        path, expire = entry
        if expire < time.time():
            return None
        if not os.path.exists(path):
            return None
        return path


_registry = PublicMediaRegistry()


def get_registry() -> PublicMediaRegistry:
    return _registry


def build_media_url(base_url: str, token: str) -> str:
    return f"{str(base_url).rstrip('/')}{MEDIA_ROUTE_PATH}/{token}"


def _insert_route_before_catchall(routes: list, new_route) -> bool:
    """把新路由插到会吞掉所有路径的 catch-all 之前，返回是否找到插入点。

    fastapi 0.139+ 的 include_router 是懒加载的：被包含路由（含静态文件
    catch-all ``/{static_path:path}``）不在顶层列表里，而是包在
    ``_IncludedRouter`` 包装对象中，且包装对象对任意路径都可能 FULL match。
    因此只要顶层列表里存在 ``_IncludedRouter``，就必须把新路由插到第一个
    包装对象之前；旧版扁平路由布局则插到 catch-all 路由之前。
    """
    first_wrapper_index = next(
        (
            index
            for index, route in enumerate(routes)
            if type(route).__name__ == "_IncludedRouter"
        ),
        None,
    )
    if first_wrapper_index is not None:
        routes.insert(first_wrapper_index, new_route)
        return True
    for index, route in enumerate(routes):
        if getattr(route, "path", None) == _STATIC_CATCHALL_PATH:
            routes.insert(index, new_route)
            return True
    return False


def register_route_on_app(adapter, registry: PublicMediaRegistry) -> bool:
    """在给定的面板应用适配器上注册（幂等）公网媒体路由。"""
    fastapi_app = getattr(adapter, "_app", None)
    routes = getattr(getattr(fastapi_app, "router", None), "routes", None)
    if fastapi_app is None or routes is None:
        logger.warning(
            "[Visora][Video] Dashboard application is unavailable; cannot register the public media route"
        )
        return False

    # The dashboard survives reloads; keep the media registry on app.state.
    # Route handlers always read the current registry.
    setattr(fastapi_app.state, _REGISTRY_STATE_KEY, registry)

    for route in routes:
        if getattr(route, "path", None) == _CONVERTED_ROUTE_PATH:
            return True

    state_key = _REGISTRY_STATE_KEY

    async def serve_media(token: str):
        current_registry = getattr(fastapi_app.state, state_key, None)
        path = current_registry.resolve(token) if current_registry else None
        if not path:
            return Response(status_code=404)
        return FileResponse(path)

    try:
        adapter.add_url_rule(
            _QUART_ROUTE_RULE,
            serve_media,
            methods=["GET"],
            endpoint="visora_image_public_media",
        )
    except Exception as exc:
        logger.warning(
            f"[Visora][Video] Failed to register the public media route: {exc}"
        )
        return False

    # add_url_rule appends routes after the static catch-all, which would shadow them.
    # Move the media route before that catch-all.
    try:
        new_route = routes.pop()
        if getattr(new_route, "path", None) != _CONVERTED_ROUTE_PATH:
            routes.append(new_route)
            return False
        if not _insert_route_before_catchall(routes, new_route):
            logger.warning(
                "[Visora][Video] No safe position for the public media route; "
                "the static catch-all may shadow it"
            )
            return False
    except Exception as exc:
        logger.warning(
            f"[Visora][Video] Failed to reorder the public media route: {exc}"
        )
        return False

    logger.info(
        f"[Visora][Video] Registered public media route: {_CONVERTED_ROUTE_PATH}"
    )
    return True


def ensure_public_media_route(
    registry: PublicMediaRegistry, quiet: bool = False
) -> bool:
    """在 AstrBot 面板应用上注册（幂等）公网媒体路由。

    面板应用在核心初始化之后才创建，插件加载时全局 APP 还是 None，
    因此必须在每次使用时延迟读取。quiet=True 时不记录“尚未就绪”告警，
    供启动期轮询使用。
    """
    try:
        from astrbot.dashboard import server as dashboard_server
    except Exception as exc:
        logger.warning(f"[Visora][Video] Failed to import the dashboard module: {exc}")
        return False

    adapter = getattr(dashboard_server, "APP", None)
    if adapter is None:
        if not quiet:
            logger.warning(
                "[Visora][Video] Dashboard application is not ready; cannot register the public media route"
            )
        return False
    return register_route_on_app(adapter, registry)
