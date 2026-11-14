from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_visora.data_manager import DataManager


def make_manager(data_file, points=10):
    data_file.write_text(
        json.dumps(
            {"users": {"u": {"points": points, "last_checkin": None}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plugin = SimpleNamespace(
        _config_get=lambda key, default=None: default,
        get_points_name=lambda: "积分",
    )
    manager = DataManager(plugin)
    manager.data_file = str(data_file)
    manager.lock_file = f"{data_file}.lock"
    manager.data = manager._load_data()
    return manager


@pytest.mark.asyncio
async def test_points_update_is_persisted_atomically(tmp_path):
    data_file = tmp_path / "data.json"
    manager = make_manager(data_file)

    assert await asyncio.wait_for(manager.deduct_points("u", 3), timeout=3)
    assert await asyncio.wait_for(manager.get_points("u"), timeout=3) == 7
    saved = json.loads(data_file.read_text(encoding="utf-8"))
    assert saved["users"]["u"]["points"] == 7


@pytest.mark.asyncio
async def test_save_failure_rolls_back_memory_and_propagates(tmp_path):
    data_file = tmp_path / "data.json"
    manager = make_manager(data_file)
    manager._save_data_unlocked = AsyncMock(side_effect=OSError("disk full"))

    with pytest.raises(OSError, match="disk full"):
        await manager.add_points("u", 5)

    assert manager.data["users"]["u"]["points"] == 10
    saved = json.loads(data_file.read_text(encoding="utf-8"))
    assert saved["users"]["u"]["points"] == 10


@pytest.mark.asyncio
async def test_two_instances_do_not_lose_updates(tmp_path):
    data_file = tmp_path / "data.json"
    first = make_manager(data_file, points=0)
    second = make_manager(data_file, points=0)

    await asyncio.gather(
        first.add_points("u", 1),
        second.add_points("u", 1),
    )

    saved = json.loads(data_file.read_text(encoding="utf-8"))
    assert saved["users"]["u"]["points"] == 2


@pytest.mark.asyncio
async def test_task_charge_and_refund_are_idempotent(tmp_path):
    data_file = tmp_path / "data.json"
    manager = make_manager(data_file, points=10)

    assert await manager.charge_task("task-1", "u", 3)
    assert await manager.charge_task("task-1", "u", 3)
    assert await manager.get_points("u") == 7

    charge = await manager.get_task_charge("task-1")
    assert charge == {
        "user_id": "u",
        "amount": 3,
        "charged": True,
        "refunded": False,
    }
    charge["amount"] = 99
    assert (await manager.get_task_charge("task-1"))["amount"] == 3

    assert await manager.refund_task("task-1")
    assert await manager.refund_task("task-1")
    assert await manager.get_points("u") == 10
    assert (await manager.get_task_charge("task-1"))["refunded"] is True


@pytest.mark.asyncio
async def test_concurrent_task_charge_and_refund_change_balance_once(tmp_path):
    data_file = tmp_path / "data.json"
    first = make_manager(data_file, points=10)
    second = make_manager(data_file, points=10)

    charged = await asyncio.gather(
        first.charge_task("task-concurrent", "u", 4),
        second.charge_task("task-concurrent", "u", 4),
    )
    assert charged == [True, True]
    assert await first.get_points("u") == 6

    refunded = await asyncio.gather(
        first.refund_task("task-concurrent"),
        second.refund_task("task-concurrent"),
    )
    assert refunded == [True, True]
    assert await second.get_points("u") == 10


@pytest.mark.asyncio
async def test_task_charge_rejects_conflicting_parameters(tmp_path):
    data_file = tmp_path / "data.json"
    manager = make_manager(data_file, points=10)

    assert await manager.charge_task("task-conflict", "u", 3)

    with pytest.raises(ValueError, match="conflicting charge parameters"):
        await manager.charge_task("task-conflict", "u", 4)
    with pytest.raises(ValueError, match="conflicting charge parameters"):
        await manager.charge_task("task-conflict", "other", 3)

    assert await manager.get_points("u") == 7


@pytest.mark.asyncio
async def test_insufficient_task_charge_is_recorded_and_can_be_retried(tmp_path):
    data_file = tmp_path / "data.json"
    manager = make_manager(data_file, points=2)

    assert not await manager.charge_task("task-retry", "u", 3)
    assert await manager.get_task_charge("task-retry") == {
        "user_id": "u",
        "amount": 3,
        "charged": False,
        "refunded": False,
    }
    assert not await manager.refund_task("task-retry")

    await manager.add_points("u", 2)
    assert await manager.charge_task("task-retry", "u", 3)
    assert await manager.get_points("u") == 1


@pytest.mark.asyncio
async def test_legacy_economy_migrates_once_and_never_overwrites_persistent_balances(
    tmp_path,
):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "users": {"u": {"points": 50, "last_checkin": "2026-11-14"}},
                "task_charges": {},
            }
        )
    )
    plugin = SimpleNamespace(
        plugin_data_dir=tmp_path / "persistent",
        _config_get=lambda key, default=None: default,
        get_points_name=lambda: "积分",
    )
    first = DataManager(plugin)
    first.legacy_data_file = legacy
    await first.initialize()
    assert await first.get_points("u") == 50
    assert await first.deduct_points("u", 7)
    second = DataManager(plugin)
    second.legacy_data_file = legacy
    await second.initialize()
    assert await second.get_points("u") == 43
    assert json.loads(legacy.read_text())["users"]["u"]["points"] == 50
    assert first.data_file.parent == plugin.plugin_data_dir


@pytest.mark.asyncio
async def test_invalid_legacy_ledger_is_not_replaced_with_an_empty_balance(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text("broken json")
    plugin = SimpleNamespace(plugin_data_dir=tmp_path / "persistent")
    manager = DataManager(plugin)
    manager.legacy_data_file = legacy
    with pytest.raises(ValueError):
        await manager.initialize()
    assert not manager.data_file.exists()
    assert legacy.read_text() == "broken json"
