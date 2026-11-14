from __future__ import annotations

import asyncio
import os
import time

from astrbot_plugin_visora.public_media import (
    MEDIA_ROUTE_PATH,
    PublicMediaRegistry,
    build_media_url,
    register_route_on_app,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 16


def _run(coro):
    return asyncio.run(coro)


def test_register_and_resolve_is_multi_use():
    async def run():
        registry = PublicMediaRegistry()
        token = await registry.register(PNG_BYTES)
        assert token
        path = registry.resolve(token)
        assert path and os.path.exists(path)
        # 与核心 file_token_service 不同：多次访问不失效。
        assert registry.resolve(token) == path
        assert registry.resolve(token) == path

    _run(run())


def test_resolve_unknown_and_expired_tokens():
    async def run():
        registry = PublicMediaRegistry()
        assert registry.resolve("no-such-token") is None
        token = await registry.register(PNG_BYTES)
        assert token
        # 人为把过期时间改到过去，模拟 TTL 结束。
        path, _ = registry._entries[token]
        registry._entries[token] = (path, time.time() - 1)
        assert registry.resolve(token) is None

    _run(run())


def test_register_rejects_empty_data():
    async def run():
        registry = PublicMediaRegistry()
        assert await registry.register(b"") is None

    _run(run())


def test_build_media_url():
    assert (
        build_media_url("https://astr.example.com/", "abc")
        == f"https://astr.example.com{MEDIA_ROUTE_PATH}/abc"
    )


def test_register_route_on_app_is_idempotent_and_ordered():
    async def run():
        from astrbot.dashboard.asgi_runtime import FastAPIAppAdapter
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        # 复刻面板应用构建时的路由顺序：普通路由在前，静态 catch-all 最后。
        fastapi_app = FastAPI(openapi_url=None)
        adapter = FastAPIAppAdapter(fastapi_app)

        @fastapi_app.get("/api/other")
        async def other():
            return {"ok": True}

        fastapi_app.add_api_route(
            "/{static_path:path}",
            lambda static_path: {"static": static_path},
            methods=["GET"],
            include_in_schema=False,
        )

        registry = PublicMediaRegistry()
        assert register_route_on_app(adapter, registry) is True
        # 幂等：重复注册不会产生重复路由。
        assert register_route_on_app(adapter, registry) is True
        route_paths = [getattr(r, "path", "") for r in fastapi_app.router.routes]
        assert route_paths.count(f"{MEDIA_ROUTE_PATH}/{{token}}") == 1
        # 新路由必须排在静态 catch-all 之前，否则永远不会被匹配。
        assert route_paths.index(f"{MEDIA_ROUTE_PATH}/{{token}}") < route_paths.index(
            "/{static_path:path}"
        )

        token = await registry.register(PNG_BYTES)
        assert token

        client = TestClient(fastapi_app)
        resp = client.get(f"{MEDIA_ROUTE_PATH}/{token}")
        assert resp.status_code == 200
        assert resp.content == PNG_BYTES
        # 404 不会触发异常，未注册 token 返回 404。
        assert client.get(f"{MEDIA_ROUTE_PATH}/nope").status_code == 404
        # 静态 catch-all 仍然工作。
        assert client.get("/api/other").status_code == 200

        # 插件重载场景：新 registry 挂到 app.state 后，旧路由读取到新注册表。
        registry2 = PublicMediaRegistry()
        assert register_route_on_app(adapter, registry2) is True
        assert client.get(f"{MEDIA_ROUTE_PATH}/{token}").status_code == 404
        token2 = await registry2.register(b"\xff\xd8" + b"1" * 8)
        resp2 = client.get(f"{MEDIA_ROUTE_PATH}/{token2}")
        assert resp2.status_code == 200

    _run(run())


def test_register_route_on_app_with_lazy_included_routers():
    """fastapi 0.139+ 懒加载 include_router 布局：catch-all 包在 _IncludedRouter 里。"""

    async def run():
        from astrbot.dashboard.asgi_runtime import FastAPIAppAdapter
        from fastapi import APIRouter, FastAPI
        from fastapi.testclient import TestClient

        fastapi_app = FastAPI(openapi_url=None)
        adapter = FastAPIAppAdapter(fastapi_app)

        api_router = APIRouter()

        @api_router.get("/api/v1/other")
        async def other():
            return {"ok": True}

        static_router = APIRouter()

        @static_router.get("/{static_path:path}")
        async def static_catchall(static_path: str):
            return {"static": static_path}

        # include_router 产生 _IncludedRouter 包装对象，catch-all 不在顶层列表里。
        fastapi_app.include_router(api_router)
        fastapi_app.include_router(static_router)
        route_types = [type(r).__name__ for r in fastapi_app.router.routes]
        assert "_IncludedRouter" in route_types

        registry = PublicMediaRegistry()
        assert register_route_on_app(adapter, registry) is True

        token = await registry.register(PNG_BYTES)
        client = TestClient(fastapi_app)
        resp = client.get(f"{MEDIA_ROUTE_PATH}/{token}")
        assert resp.status_code == 200
        assert resp.content == PNG_BYTES
        # 既有路由不受影响。
        assert client.get("/api/v1/other").status_code == 200
        assert client.get("/some/static/path").status_code == 200

    _run(run())
