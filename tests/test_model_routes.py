from __future__ import annotations

from types import SimpleNamespace

from astrbot_plugin_visora.main import VisoraPlugin


class FakeProvider:
    def __init__(
        self,
        provider_id: str,
        model: str,
        key: str,
        *,
        api_base: str = "https://images.example/v1",
        **extra,
    ) -> None:
        self.id = provider_id
        self.api_base = api_base
        self.provider_config = {
            "id": provider_id,
            "model": model,
            "key": [key],
            "api_base": api_base,
            **extra,
        }

    def get_keys(self) -> list[str]:
        return list(self.provider_config["key"])


def make_plugin(config: dict, providers: list[FakeProvider]) -> VisoraPlugin:
    plugin = VisoraPlugin.__new__(VisoraPlugin)
    provider_map = {provider.provider_config["id"]: provider for provider in providers}
    plugin.config = config
    plugin.context = SimpleNamespace(
        get_provider_by_id=lambda provider_id: provider_map.get(provider_id),
        get_all_providers=lambda: list(providers),
    )
    return plugin


def test_explicit_routes_bind_one_format_per_model_and_mode():
    providers = [
        FakeProvider("channel-a/shared-image", "shared-image", "key-a"),
        FakeProvider("channel-b/shared-image", "shared-image", "key-b"),
    ]
    config = {
        "personality": {
            "image_model_routes": [
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-a/shared-image",
                    "request_format": "openai_images",
                },
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-b/shared-image",
                    "request_format": "openai_chat",
                },
                {
                    "__template_key": "img2img",
                    "provider_id": "channel-a/shared-image",
                    "request_format": "openai_edits",
                },
                {
                    "__template_key": "persona",
                    "provider_id": "channel-a/shared-image",
                    "request_format": "openai_chat",
                },
            ]
        },
    }
    plugin = make_plugin(config, providers)

    plugin._load_config()

    assert plugin.default_provider_full_id == "channel-a/shared-image"
    assert plugin.api_request_format == "v1/images/generations"
    assert [route["provider_id"] for route in plugin.fallback_models] == [
        "channel-b/shared-image"
    ]
    assert plugin.fallback_models[0]["api_request_format"] == "v1/chat/completions"
    assert plugin.fallback_models[0]["model"] == plugin.model == "shared-image"
    assert plugin.fallback_models[0]["base_url"] == plugin.base_url

    assert plugin.img2img_provider_full_id == "channel-a/shared-image"
    assert plugin.img2img_api_request_format == "v1/images/edits"
    assert plugin.img2img_fallback_models == []
    assert plugin.persona_provider_full_id == "channel-a/shared-image"
    assert plugin.persona_api_request_format == "v1/chat/completions"
    assert plugin.persona_fallback_models == []

    plugin._load_config()
    assert len(plugin.image_routes["txt2img"]) == 2
    assert len(plugin.image_routes["img2img"]) == 1
    assert len(plugin.image_routes["persona"]) == 1


def test_unconfigured_modes_remain_disabled():
    providers = [FakeProvider("channel-a/text-image", "text-image", "key-a")]
    config = {
        "personality": {
            "image_model_routes": [
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-a/text-image",
                    "request_format": "openai_images",
                }
            ]
        }
    }
    plugin = make_plugin(config, providers)

    plugin._load_config()

    assert plugin.default_provider_full_id == "channel-a/text-image"
    assert plugin.image_routes["img2img"] == []
    assert plugin.img2img_provider_full_id == ""
    assert plugin.img2img_api_keys == []
    assert plugin.image_routes["persona"] == []
    assert plugin.persona_provider_full_id == ""
    assert plugin.persona_api_keys == []
    assert plugin.image_routes["admin"] == []
    assert plugin.admin_provider_full_id == ""
    assert plugin.admin_api_keys == []


def test_routes_priority_orders_and_enabled_filters():
    providers = [
        FakeProvider("channel-a/model-a", "model-a", "key-a"),
        FakeProvider("channel-b/model-b", "model-b", "key-b"),
        FakeProvider("channel-c/model-c", "model-c", "key-c"),
        FakeProvider("channel-d/model-d", "model-d", "key-d"),
    ]
    config = {
        "personality": {
            "image_model_routes": [
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-a/model-a",
                    "request_format": "openai_images",
                    "priority": 10,
                },
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-b/model-b",
                    "request_format": "openai_images",
                    "priority": 0,
                    "enabled": False,
                },
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-c/model-c",
                    "request_format": "openai_chat",
                    "priority": 5,
                },
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-d/model-d",
                    "request_format": "openai_images",
                    "priority": 1,
                },
            ]
        }
    }
    plugin = make_plugin(config, providers)

    plugin._load_config()

    # 数值越小越优先；enabled=False 的 channel-b 被过滤掉
    assert [route["provider_id"] for route in plugin.image_routes["txt2img"]] == [
        "channel-d/model-d",
        "channel-c/model-c",
        "channel-a/model-a",
    ]
    assert plugin.default_provider_full_id == "channel-d/model-d"
    assert [route["provider_id"] for route in plugin.fallback_models] == [
        "channel-c/model-c",
        "channel-a/model-a",
    ]


def test_routes_same_priority_keep_configuration_order():
    providers = [
        FakeProvider("channel-a/model-a", "model-a", "key-a"),
        FakeProvider("channel-b/model-b", "model-b", "key-b"),
        FakeProvider("channel-c/model-c", "model-c", "key-c"),
    ]
    config = {
        "personality": {
            "image_model_routes": [
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-a/model-a",
                    "request_format": "openai_images",
                },
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-b/model-b",
                    "request_format": "openai_images",
                },
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-c/model-c",
                    "request_format": "openai_images",
                },
            ]
        }
    }
    plugin = make_plugin(config, providers)

    plugin._load_config()

    # 旧配置（无 priority）全部 priority=0，稳定排序保持原数组顺序
    assert [route["provider_id"] for route in plugin.image_routes["txt2img"]] == [
        "channel-a/model-a",
        "channel-b/model-b",
        "channel-c/model-c",
    ]
    assert plugin.default_provider_full_id == "channel-a/model-a"


def test_routes_dedup_keeps_highest_priority_entry():
    providers = [FakeProvider("channel-a/model-a", "model-a", "key-a")]
    config = {
        "personality": {
            "image_model_routes": [
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-a/model-a",
                    "request_format": "openai_images",
                    "priority": 10,
                },
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-a/model-a",
                    "request_format": "openai_chat",
                    "priority": 1,
                },
            ]
        }
    }
    plugin = make_plugin(config, providers)

    plugin._load_config()

    # 同一个 provider 在同类路由中只保留高优先级（priority=1）那一项
    assert len(plugin.image_routes["txt2img"]) == 1
    assert plugin.image_routes["txt2img"][0]["api_request_format"] == (
        "v1/chat/completions"
    )
    assert plugin.default_provider_full_id == "channel-a/model-a"


def test_routes_string_enabled_trap_values_filter_out():
    providers = [
        FakeProvider("channel-a/model-a", "model-a", "key-a"),
        FakeProvider("channel-b/model-b", "model-b", "key-b"),
        FakeProvider("channel-c/model-c", "model-c", "key-c"),
    ]
    config = {
        "personality": {
            "image_model_routes": [
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-a/model-a",
                    "request_format": "openai_images",
                    "enabled": "false",
                },
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-b/model-b",
                    "request_format": "openai_images",
                    "enabled": 0,
                },
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-c/model-c",
                    "request_format": "openai_images",
                },
            ]
        }
    }
    plugin = make_plugin(config, providers)

    plugin._load_config()

    # 字符串 "false" 与数字 0 都判为禁用，缺省 enabled 视为启用
    assert [route["provider_id"] for route in plugin.image_routes["txt2img"]] == [
        "channel-c/model-c"
    ]
    assert plugin.default_provider_full_id == "channel-c/model-c"


def test_admin_route_isolated_from_other_modes():
    providers = [
        FakeProvider("channel-a/text-image", "text-image", "key-a"),
        FakeProvider("channel-a/admin-image", "admin-image", "key-admin"),
    ]
    config = {
        "personality": {
            "image_model_routes": [
                {
                    "__template_key": "txt2img",
                    "provider_id": "channel-a/text-image",
                    "request_format": "openai_images",
                },
                {
                    "__template_key": "admin",
                    "provider_id": "channel-a/admin-image",
                    "request_format": "openai_chat",
                },
            ]
        }
    }
    plugin = make_plugin(config, providers)

    plugin._load_config()

    # admin 路由独立，不影响文生图路由
    assert plugin.default_provider_full_id == "channel-a/text-image"
    assert plugin.image_routes["txt2img"] != []
    assert plugin.image_routes["admin"] != []
    assert plugin.admin_provider_full_id == "channel-a/admin-image"
    assert plugin.admin_api_keys == ["key-admin"]
    assert plugin.admin_api_request_format == "v1/chat/completions"
    # 未配置 admin 时不得回退到 txt2img：这里显式断言已配置 admin 与 txt2img 不同
    assert plugin.admin_provider_full_id != plugin.default_provider_full_id


def test_route_parse_enabled_and_priority_helpers():
    plugin = VisoraPlugin.__new__(VisoraPlugin)
    assert plugin._parse_route_enabled(None) is True
    assert plugin._parse_route_enabled(True) is True
    assert plugin._parse_route_enabled(False) is False
    assert plugin._parse_route_enabled(1) is True
    assert plugin._parse_route_enabled(0) is False
    assert plugin._parse_route_enabled("true") is True
    assert plugin._parse_route_enabled("false") is False
    assert plugin._parse_route_enabled("off") is False
    assert plugin._parse_route_enabled("yes") is True

    assert plugin._parse_route_priority(None) == 0
    assert plugin._parse_route_priority("3") == 3
    assert plugin._parse_route_priority("abc") == 0
    assert plugin._parse_route_priority(7.9) == 7
