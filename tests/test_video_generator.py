from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from astrbot_plugin_visora.video_generator import VideoGenerator


def _generator(**overrides) -> VideoGenerator:
    settings = {
        "api_keys": ["key-a", "key-b"],
        "base_url": "https://video.example.com/v1",
        "model": "video-model",
        "timeout": 30,
        "max_retry_attempts": 3,
        "api_request_formats": ["v1/videos", "v1/videos/generations"],
        "task_query_format": "v1/videos/{task_id}",
        "poll_interval": 1,
        "max_unknown_polls": 2,
    }
    settings.update(overrides)
    return VideoGenerator(**settings)


def test_remote_task_transient_poll_error_reuses_same_task_without_resubmit():
    async def run():
        generator = _generator()
        methods: list[str] = []
        get_count = 0

        async def request(method, url, api_key, json_payload=None):
            nonlocal get_count
            methods.append(method)
            if method == "POST":
                return {"task_id": "remote-task-1"}, None
            get_count += 1
            if get_count == 1:
                return None, "temporary gateway failure"
            return {
                "status": "completed",
                "video_url": "https://cdn.example.com/result.mp4",
            }, None

        async def download(source, task_id):
            return "/tmp/result.mp4", None

        generator._make_json_request = request
        generator._download_video_source = download
        with patch(
            "astrbot_plugin_visora.video_generator.asyncio.sleep",
            new=AsyncMock(),
        ):
            path, error = await generator.generate_video("test")

        assert path == "/tmp/result.mp4"
        assert error is None
        assert methods.count("POST") == 1
        assert methods.count("GET") == 2

    asyncio.run(run())


def test_remote_task_unknown_status_failure_does_not_resubmit():
    async def run():
        generator = _generator()
        methods: list[str] = []

        async def request(method, url, api_key, json_payload=None):
            methods.append(method)
            if method == "POST":
                return {"task_id": "remote-task-2"}, None
            return {"status": "unknown"}, None

        generator._make_json_request = request
        with patch(
            "astrbot_plugin_visora.video_generator.asyncio.sleep",
            new=AsyncMock(),
        ):
            path, error = await generator.generate_video("test")

        assert path is None
        assert "连续 2 次" in (error or "")
        assert methods.count("POST") == 1
        assert methods.count("GET") == 2

    asyncio.run(run())


def test_remote_task_download_failure_retries_download_not_submission():
    async def run():
        generator = _generator()
        methods: list[str] = []
        download_count = 0

        async def request(method, url, api_key, json_payload=None):
            methods.append(method)
            if method == "POST":
                return {"task_id": "remote-task-3"}, None
            return {
                "status": "completed",
                "video_url": "https://cdn.example.com/result.mp4",
            }, None

        async def download(source, task_id):
            nonlocal download_count
            download_count += 1
            return None, "temporary download failure"

        generator._make_json_request = request
        generator._download_video_source = download
        with patch(
            "astrbot_plugin_visora.video_generator.asyncio.sleep",
            new=AsyncMock(),
        ):
            path, error = await generator.generate_video("test")

        assert path is None
        assert "成品下载连续失败 3 次" in (error or "")
        assert methods.count("POST") == 1
        assert methods.count("GET") == 1
        assert download_count == 3

    asyncio.run(run())


def test_completed_submit_response_with_task_id_never_resubmits_on_download_failure():
    async def run():
        generator = _generator()
        methods: list[str] = []
        download_count = 0

        async def request(method, url, api_key, json_payload=None):
            methods.append(method)
            return {
                "task_id": "remote-task-4",
                "status": "completed",
                "video_url": "https://cdn.example.com/result.mp4",
            }, None

        async def download(source, task_id):
            nonlocal download_count
            download_count += 1
            return None, "temporary download failure"

        generator._make_json_request = request
        generator._download_video_source = download
        with patch(
            "astrbot_plugin_visora.video_generator.asyncio.sleep",
            new=AsyncMock(),
        ):
            path, error = await generator.generate_video("test")

        assert path is None
        assert "成品下载连续失败 3 次" in (error or "")
        assert methods == ["POST"]
        assert download_count == 3

    asyncio.run(run())
