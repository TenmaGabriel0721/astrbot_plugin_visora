from __future__ import annotations

import asyncio
import copy
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from astrbot_plugin_visora.persona_profiles import (
    DEFAULT_PERSONA_IDENTITY,
    PersonaProfileStore,
)
from PIL import Image


def make_store(tmp_path, *, profiles=None):
    config = {
        "persona_profiles": profiles
        or [
            {
                "id": "legacy",
                "name": "Bot A",
                "bot_id": "bot-a",
                "style": "photo",
                "resolution": "inherit",
                "identity_prompt": DEFAULT_PERSONA_IDENTITY,
            },
            {
                "id": "bot_b",
                "name": "Bot B",
                "bot_id": "bot-b",
                "style": "anime",
                "resolution": "2K",
                "identity_prompt": "pink hair and blue eyes",
            },
        ]
    }
    plugin = SimpleNamespace(
        _config_get=lambda key, default=None: copy.deepcopy(config.get(key, default)),
        _config_set=lambda key, value: config.__setitem__(key, value),
        config=SimpleNamespace(save_config=Mock()),
    )
    return PersonaProfileStore(
        plugin, tmp_path / "persistent", tmp_path / "legacy"
    ), config


def image_bytes(color="red"):
    output = BytesIO()
    Image.new("RGB", (24, 36), color).save(output, format="PNG")
    return output.getvalue()


def test_identity_resolves_by_platform_instance_and_never_another_bots_self_id(
    tmp_path,
):
    store, _ = make_store(tmp_path)
    bot_a = SimpleNamespace(
        get_platform_id=lambda: "bot-a", get_self_id=lambda: "bot-b"
    )
    bot_b = SimpleNamespace(
        get_platform_id=lambda: "bot-b", get_self_id=lambda: "bot-a"
    )
    unknown = SimpleNamespace(
        get_platform_id=lambda: "other", get_self_id=lambda: "bot-a"
    )
    assert store.resolve(bot_a)["name"] == "Bot A"
    assert store.resolve(bot_b)["name"] == "Bot B"
    assert store.resolve(unknown) is None
    assert (
        store.resolve(SimpleNamespace(unified_msg_origin="bot-b:FriendMessage:u"))["id"]
        == "bot_b"
    )


@pytest.mark.asyncio
async def test_libraries_remain_independent_and_delete_cannot_reach_another_profile(
    tmp_path,
):
    store, _ = make_store(tmp_path)
    red, blue = image_bytes("red"), image_bytes("blue")
    bot_a_names = await store.add_references(
        "legacy", [(red, "image/jpeg")], max_bytes=10000
    )
    bot_b_names = await store.add_references(
        "bot_b", [(blue, "image/png")], max_bytes=10000
    )
    assert await store.load_references(
        store.get("legacy"), limit=4, max_bytes=10000
    ) == [(red, "image/png")]
    assert await store.load_references(
        store.get("bot_b"), limit=4, max_bytes=10000
    ) == [(blue, "image/png")]
    with pytest.raises(ValueError, match="不属于"):
        await store.delete_references("bot_b", bot_a_names)
    await store.delete_references("bot_b", bot_b_names)
    assert store.reference_files("legacy")
    assert not store.reference_files("bot_b")


@pytest.mark.asyncio
async def test_legacy_migration_preserves_ownership_newer_files_and_source_images(
    tmp_path,
):
    store, config = make_store(tmp_path)
    source = store.legacy_dir
    source.mkdir()
    old = image_bytes("red")
    newer = image_bytes("blue")
    (source / "face.png").write_bytes(old)
    (source / "second.png").write_bytes(old)
    destination = store.reference_dir("legacy")
    destination.mkdir(parents=True)
    (destination / "face.png").write_bytes(newer)
    await store.migrate_legacy()
    assert config["persona_profiles"][0]["name"] == "Bot A"
    assert config["persona_profiles"][0]["bot_id"] == "bot-a"
    assert (destination / "face.png").read_bytes() == newer
    assert (destination / "second.png").read_bytes() == old
    assert (source / "face.png").read_bytes() == old
    assert not store.reference_files("bot_b")
    (destination / "second.png").unlink()
    await store.migrate_legacy()
    assert not (destination / "second.png").exists()


@pytest.mark.asyncio
async def test_invalid_batch_and_concurrent_uploads_cannot_exceed_four_references(
    tmp_path,
):
    store, _ = make_store(tmp_path)
    valid = image_bytes()
    with pytest.raises(ValueError, match="有效"):
        await store.add_references(
            "legacy", [(valid, "image/png"), (b"broken", "image/png")], max_bytes=10000
        )
    assert not store.reference_files("legacy")
    results = await asyncio.gather(
        *(
            store.add_references("legacy", [(valid, "image/png")], max_bytes=10000)
            for _ in range(5)
        ),
        return_exceptions=True,
    )
    assert len(store.reference_files("legacy")) == 4
    assert sum(isinstance(result, ValueError) for result in results) == 1


@pytest.mark.asyncio
async def test_corrupt_oversized_and_symlink_files_are_skipped_before_selecting_valid_images(
    tmp_path,
):
    store, _ = make_store(tmp_path)
    directory = store.reference_dir("legacy")
    directory.mkdir(parents=True)
    (directory / "0-corrupt.png").write_bytes(b"broken")
    (directory / "1-oversized.png").write_bytes(b"x" * 10001)
    valid = image_bytes()
    (directory / "2-valid.png").write_bytes(valid)
    (directory / "3-link.png").symlink_to(directory / "2-valid.png")
    assert await store.load_references(
        store.get("legacy"), limit=1, max_bytes=10000
    ) == [(valid, "image/png")]


def test_duplicate_bot_bindings_and_symlink_profile_directories_are_rejected(tmp_path):
    store, config = make_store(tmp_path)
    config["persona_profiles"][1]["bot_id"] = "bot-a"
    with pytest.raises(ValueError, match="每个 Bot"):
        store.profiles()
    store.root.mkdir(parents=True)
    target = store.reference_dir("bot_b")
    target.mkdir()
    (store.root / "legacy").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="路径"):
        store.reference_dir("legacy")
    with pytest.raises(ValueError, match="ID"):
        store.reference_dir("../outside")


def test_generated_profile_ids_are_stable_before_configuration_is_persisted():
    raw = [{"name": "Bot", "bot_id": "bot"}]
    assert PersonaProfileStore.normalize_profiles(
        raw
    ) == PersonaProfileStore.normalize_profiles(raw)
