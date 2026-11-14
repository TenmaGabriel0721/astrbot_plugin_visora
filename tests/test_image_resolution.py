from __future__ import annotations

import json
from pathlib import Path

from astrbot_plugin_visora.image_resolution import (
    normalize_image_resolution,
    resolve_configured_resolution,
)


def test_normalize_exact_resolution_variants():
    assert normalize_image_resolution(" 1920 X 1080 ") == "1920x1080"
    assert normalize_image_resolution("1920×1080") == "1920x1080"


def test_resolve_custom_configuration():
    assert resolve_configured_resolution("自定义", "1920x1080") == "1920x1080"


def test_schema_exposes_custom_resolution_conditionally():
    schema_path = Path(__file__).parents[1] / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    items = schema["tool_decoration"]["items"]

    assert items["default_resolution"]["default"] == "2K"
    assert items["default_resolution"]["options"] == [
        "1K",
        "2K",
        "4K",
        "自定义",
    ]
    assert items["custom_resolution"]["condition"] == {"default_resolution": "自定义"}
