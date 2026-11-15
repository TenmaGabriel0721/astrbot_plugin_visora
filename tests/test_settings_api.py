from __future__ import annotations

import copy
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from astrbot_plugin_visora.main import VisoraPlugin
from astrbot_plugin_visora.persona_profiles import PersonaProfileStore
from astrbot_plugin_visora.presets import normalize_presets
from astrbot_plugin_visora.settings_api import SettingsPageApi
from PIL import Image


class Config(dict):
    def __init__(self, data, path):
        super().__init__(data)
        self.config_path = str(path)
        self.save_config = Mock(
            side_effect=lambda: path.write_text(json.dumps(self, ensure_ascii=False))
        )
        self.save_config()


def make_api(tmp_path):
    providers = [
        SimpleNamespace(
            id=name,
            provider_config={
                "id": name,
                "model": name.split("/")[-1],
                "key": ["secret-api-key"],
                "api_base": "https://secret-provider.example",
            },
        )
        for name in ("images/gpt-image", "images/gemini-image")
    ]
    provider_map = {provider.id: provider for provider in providers}
    plugin = VisoraPlugin.__new__(VisoraPlugin)
    plugin.config = Config(
        {
            "personality": {
                "image_model_routes": [
                    {
                        "__template_key": "txt2img",
                        "provider_id": "images/gpt-image",
                        "request_format": "openai_images",
                        "priority": 10,
                        "enabled": True,
                    },
                    {
                        "__template_key": "txt2img",
                        "provider_id": "images/gemini-image",
                        "request_format": "gemini",
                        "priority": "0",
                        "enabled": "true",
                    },
                ]
            },
            "persona": {
                "persona_profiles": [
                    {
                        "id": "bot_a",
                        "name": "Bot A",
                        "bot_id": "bot-a",
                        "identity_prompt": "",
                        "style": "photo",
                        "resolution": "inherit",
                    },
                    {
                        "id": "bot_b",
                        "name": "Bot B",
                        "bot_id": "bot-b",
                        "identity_prompt": "",
                        "style": "anime",
                        "resolution": "2K",
                    },
                ]
            },
            "comfort": {"enable_economy": True, "initial_points": 333},
            "wake_interaction": {"presets": ["custom:do not overwrite"]},
        },
        tmp_path / "config.json",
    )
    plugin.context = SimpleNamespace(
        get_all_providers=lambda: providers,
        get_provider_by_id=lambda provider_id: provider_map.get(provider_id),
        get_config=lambda: {
            "platform": [
                {"id": "bot-a", "type": "aiocqhttp"},
                {"id": "bot-b", "type": "qq_official"},
            ]
        },
        register_web_api=Mock(),
    )
    plugin.persona_profiles = PersonaProfileStore(
        plugin, tmp_path / "persistent", tmp_path / "legacy"
    )
    plugin._load_config()
    return SettingsPageApi(plugin)


@pytest.mark.asyncio
async def test_page_presets_can_be_created_edited_and_deleted_without_restart(tmp_path):
    api = make_api(tmp_path)
    preset = {
        "__template_key": "txt2img",
        "name": "旅行海报",
        "prompt": "minimalist travel poster",
        "aspect_ratio": "3:4",
        "resolution": "4K",
    }
    payload = api.snapshot()
    payload["presets"].append(preset)
    for action in ("create", "edit", "delete"):
        if action == "edit":
            payload["presets"][-1].update(
                __template_key="img2img", prompt="convert the reference to a poster"
            )
        elif action == "delete":
            payload["presets"].pop()
        with patch(
            "astrbot_plugin_visora.settings_api.request",
            SimpleNamespace(json=AsyncMock(return_value=payload)),
        ):
            result = await api.save_settings()
        assert result["status"] == "ok"
        payload = result["data"]
        if action == "delete":
            assert "旅行海报" not in api.plugin.presets
        else:
            saved = api.plugin.presets["旅行海报"]
            assert saved["__template_key"] == (
                "txt2img" if action == "create" else "img2img"
            )
            assert saved["aspect_ratio"] == "3:4"
            assert saved["resolution"] == "4K"


@pytest.mark.asyncio
async def test_invalid_preset_submission_preserves_all_settings_and_runtime(tmp_path):
    api = make_api(tmp_path)
    original = copy.deepcopy(dict(api.plugin.config))
    previous = copy.deepcopy(api.plugin.presets)
    payload = api.snapshot()
    payload["presets"][0]["prompt"] = " "
    payload["routes"].reverse()
    with patch(
        "astrbot_plugin_visora.settings_api.request",
        SimpleNamespace(json=AsyncMock(return_value=payload)),
    ):
        result = await api.save_settings()
    assert result["status"] == "error"
    assert "提示词" in result["message"]
    assert dict(api.plugin.config) == original
    assert api.plugin.presets == previous


@pytest.mark.asyncio
async def test_preset_command_edit_invalidates_an_open_page_revision(tmp_path):
    api = make_api(tmp_path)
    payload = api.snapshot()
    api.plugin._config_set("presets", ["another:updated from a command"])
    with patch(
        "astrbot_plugin_visora.settings_api.request",
        SimpleNamespace(json=AsyncMock(return_value=payload)),
    ):
        result = await api.save_settings()
    assert result["status"] == "error"
    assert "别处更新" in result["message"]
    assert api.plugin._config_get("presets") == ["another:updated from a command"]


def test_migrated_flat_keys_cannot_override_new_grouped_settings_on_reload(tmp_path):
    api = make_api(tmp_path)
    plugin = api.plugin
    plugin.config["task_timeout"] = 100
    plugin._migrate_legacy_flat_config()
    assert "task_timeout" not in plugin.config
    assert plugin._config_get("task_timeout") == 100
    plugin._config_set("task_timeout", 200)
    plugin._migrate_legacy_flat_config()
    assert plugin._config_get("task_timeout") == 200


@pytest.mark.asyncio
async def test_settings_expose_provider_choices_without_secrets_and_preserve_current_priority(
    tmp_path,
):
    api = make_api(tmp_path)
    result = await api.get_settings()
    assert result["status"] == "ok"
    data = result["data"]
    assert [item["provider_id"] for item in data["routes"]] == [
        "images/gemini-image",
        "images/gpt-image",
    ]
    assert [item["name"] for item in data["bots"]] == ["Bot A", "Bot B"]
    assert "secret-api-key" not in json.dumps(data)
    assert "secret-provider" not in json.dumps(data)


@pytest.mark.asyncio
async def test_one_list_swap_updates_both_priorities_and_runtime_without_touching_other_settings(
    tmp_path,
):
    api = make_api(tmp_path)
    payload = api.snapshot()
    payload["routes"].reverse()
    original = copy.deepcopy(dict(api.plugin.config))
    with patch(
        "astrbot_plugin_visora.settings_api.request",
        SimpleNamespace(json=AsyncMock(return_value=payload)),
    ):
        result = await api.save_settings()
    assert result["status"] == "ok"
    saved = json.loads(Path(api.plugin.config.config_path).read_text())
    assert [
        item["provider_id"] for item in saved["personality"]["image_model_routes"]
    ] == ["images/gpt-image", "images/gemini-image"]
    assert [
        item["priority"] for item in saved["personality"]["image_model_routes"]
    ] == [0, 1]
    assert api.plugin.default_provider_full_id == "images/gpt-image"
    assert saved["comfort"] == original["comfort"]
    assert normalize_presets(saved["wake_interaction"]["presets"]) == normalize_presets(
        original["wake_interaction"]["presets"]
    )


@pytest.mark.asyncio
async def test_stale_revision_and_failed_save_preserve_existing_configuration(tmp_path):
    api = make_api(tmp_path)
    original = copy.deepcopy(dict(api.plugin.config))
    payload = api.snapshot()
    payload["revision"] = "stale"
    with patch(
        "astrbot_plugin_visora.settings_api.request",
        SimpleNamespace(json=AsyncMock(return_value=payload)),
    ):
        result = await api.save_settings()
    assert result["status"] == "error"
    assert "别处更新" in result["message"]
    assert dict(api.plugin.config) == original
    payload = api.snapshot()
    payload["routes"].reverse()
    api.plugin.config.save_config.side_effect = OSError("disk full")
    with patch(
        "astrbot_plugin_visora.settings_api.request",
        SimpleNamespace(json=AsyncMock(return_value=payload)),
    ):
        result = await api.save_settings()
    assert result["status"] == "error"
    assert dict(api.plugin.config) == original
    assert api.plugin.default_provider_full_id == "images/gemini-image"
    assert json.loads(Path(api.plugin.config.config_path).read_text()) == original


@pytest.mark.asyncio
async def test_duplicate_bindings_or_routes_do_not_save_partial_configuration(tmp_path):
    api = make_api(tmp_path)
    original = copy.deepcopy(dict(api.plugin.config))
    payload = api.snapshot()
    payload["profiles"][1]["bot_id"] = payload["profiles"][0]["bot_id"]
    with patch(
        "astrbot_plugin_visora.settings_api.request",
        SimpleNamespace(json=AsyncMock(return_value=payload)),
    ):
        result = await api.save_settings()
    assert result["status"] == "error"
    assert dict(api.plugin.config) == original
    payload = api.snapshot()
    payload["routes"].append(copy.deepcopy(payload["routes"][0]))
    with patch(
        "astrbot_plugin_visora.settings_api.request",
        SimpleNamespace(json=AsyncMock(return_value=payload)),
    ):
        result = await api.save_settings()
    assert result["status"] == "error"
    assert dict(api.plugin.config) == original


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://another.example/image",
        "../images",
        "images?key=secret",
        "v1/images,gemini",
        "models/{other}:generateContent",
    ],
)
def test_invalid_custom_paths_are_rejected(tmp_path, endpoint):
    api = make_api(tmp_path)
    route = api.snapshot()["routes"][0]
    route["request_format"] = endpoint
    with pytest.raises(ValueError):
        api.normalize_routes([route])


@pytest.mark.asyncio
async def test_native_upload_and_thumbnail_preview_are_scoped_to_one_profile(tmp_path):
    api = make_api(tmp_path)
    output = BytesIO()
    Image.new("RGB", (40, 60), "pink").save(output, format="PNG")
    uploaded = SimpleNamespace(
        content_type="image/jpeg", read=AsyncMock(return_value=output.getvalue())
    )
    with patch(
        "astrbot_plugin_visora.settings_api.request",
        SimpleNamespace(files=AsyncMock(return_value={"file": uploaded})),
    ):
        result = await api.upload_reference("bot_b")
    assert result["status"] == "ok"
    assert not api.plugin.persona_profiles.reference_files("bot_a")
    preview = await api.get_references("bot_b")
    assert preview["data"]["references"][0]["data_url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert (await api.get_references("bot_a"))["data"]["references"] == []
    name = result["data"]["added"][0]
    with patch(
        "astrbot_plugin_visora.settings_api.request",
        SimpleNamespace(json=AsyncMock(return_value={"names": [name]})),
    ):
        refused = await api.delete_references("bot_a")
        removed = await api.delete_references("bot_b")
    assert refused["status"] == "error"
    assert removed["status"] == "ok"
    assert not api.plugin.persona_profiles.reference_files("bot_b")
