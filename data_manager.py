"""Persistent economy data management for the Visora plugin."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.star import StarTools
from filelock import FileLock, Timeout


class DataManager:
    def __init__(self, plugin: Any):
        self.plugin = plugin
        data_dir = Path(
            getattr(plugin, "plugin_data_dir", None)
            or StarTools.get_data_dir("astrbot_plugin_visora")
        )
        self.data_file = data_dir / "economy.json"
        self.legacy_data_file = Path(__file__).parent / "data.json"
        self.lock_file = f"{self.data_file}.lock"
        self._lock = asyncio.Lock()
        self._initialized = False
        self.data = {"users": {}, "task_charges": {}}

    async def initialize(self) -> None:
        """Import the legacy ledger once without overwriting newer balances.

        Raises:
            ValueError: The legacy ledger is not a valid data mapping.
            OSError: Persistent storage cannot be read or written.
        """
        async with self._lock:
            if self._initialized:
                return

            def migrate() -> None:
                destination = Path(self.data_file)
                destination.parent.mkdir(parents=True, exist_ok=True)
                legacy = Path(self.legacy_data_file)
                with FileLock(self.lock_file, thread_local=False):
                    if destination.exists() or not legacy.is_file():
                        return
                    with FileLock(f"{legacy}.lock", thread_local=False):
                        payload = json.loads(legacy.read_text(encoding="utf-8-sig"))
                        if not isinstance(payload, dict) or not isinstance(
                            payload.get("users", {}), dict
                        ):
                            raise ValueError("Invalid legacy economy ledger")
                        temporary = destination.with_name(
                            f".{destination.name}.{uuid.uuid4().hex}.tmp"
                        )
                        try:
                            with temporary.open("x", encoding="utf-8") as output:
                                temporary.chmod(0o600)
                                json.dump(payload, output, ensure_ascii=False, indent=2)
                                output.flush()
                                os.fsync(output.fileno())
                            temporary.replace(destination)
                        finally:
                            temporary.unlink(missing_ok=True)

            await asyncio.to_thread(migrate)
            self.data = await asyncio.to_thread(self._load_data, True)
            self._initialized = True

    def _load_data(self, strict: bool = False) -> dict:
        if Path(self.data_file).exists():
            try:
                with open(self.data_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[Visora] Failed to load data: {e}")
                if strict:
                    raise
        return {"users": {}, "task_charges": {}}

    @asynccontextmanager
    async def _acquire_file_lock(self):
        lock = FileLock(self.lock_file, thread_local=False)
        while True:
            try:
                lock.acquire(timeout=0)
                break
            except Timeout:
                await asyncio.sleep(0.05)
        try:
            yield
        finally:
            lock.release()

    async def _run_transaction(
        self,
        operation: Callable[[], tuple[Any, bool]],
    ) -> Any:
        """Run a read-modify-write transaction against the latest disk state."""
        await self.initialize()
        async with self._lock:
            async with self._acquire_file_lock():
                self.data = await asyncio.to_thread(self._load_data, True)
                snapshot = await asyncio.to_thread(copy.deepcopy, self.data)
                try:
                    result, changed = operation()
                    if changed:
                        await self._save_data_unlocked()
                    return result
                except BaseException:
                    self.data = snapshot
                    raise

    async def _save_data_unlocked(self):
        """Finish an atomic disk write before releasing the transaction lock."""
        destination = Path(self.data_file)
        payload = self.data

        def write() -> None:
            temporary = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temporary.open("x", encoding="utf-8") as output:
                    temporary.chmod(0o600)
                    json.dump(payload, output, indent=2, ensure_ascii=False)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(destination)
            except OSError as exc:
                logger.error(f"[Visora] Failed to save data: {exc}")
                raise
            finally:
                temporary.unlink(missing_ok=True)

        write_task = asyncio.create_task(asyncio.to_thread(write))
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            await asyncio.shield(write_task)
            raise

    def _get_user_data_unlocked(self, user_id: str) -> tuple[dict, bool]:
        created = False
        users = self.data.setdefault("users", {})
        if user_id not in users:
            users[user_id] = {
                "points": self.plugin._config_get("initial_points", 100),
                "last_checkin": None,
            }
            created = True
        return users[user_id], created

    def _get_task_charges_unlocked(self) -> dict:
        """Return the mutable task charge ledger for the active transaction.

        Returns:
            The top-level mapping keyed by task ID.

        Raises:
            ValueError: If the persisted ledger has an invalid structure.
        """
        task_charges = self.data.setdefault("task_charges", {})
        if not isinstance(task_charges, dict):
            raise ValueError("task_charges must be a mapping")
        return task_charges

    async def get_user_data(self, user_id: str) -> dict:
        def operation():
            user_data, created = self._get_user_data_unlocked(user_id)
            return dict(user_data), created

        return await self._run_transaction(operation)

    async def get_points(self, user_id: str) -> int:
        def operation():
            user_data, created = self._get_user_data_unlocked(user_id)
            return user_data["points"], created

        return await self._run_transaction(operation)

    async def add_points(self, user_id: str, amount: int):
        def operation():
            user_data, _ = self._get_user_data_unlocked(user_id)
            user_data["points"] += amount
            return user_data["points"], True

        return await self._run_transaction(operation)

    async def deduct_points(self, user_id: str, amount: int) -> bool:
        def operation():
            user_data, _ = self._get_user_data_unlocked(user_id)
            if user_data["points"] >= amount:
                user_data["points"] -= amount
                return True, True
            return False, False

        return await self._run_transaction(operation)

    async def charge_task(self, task_id: str, user_id: str, amount: int) -> bool:
        """Charge points once for a persistent background task.

        Repeating a successful charge with the same task ID, user ID, and amount
        returns ``True`` without changing the balance again. An insufficient
        attempt is recorded and may be retried with the same parameters after the
        user receives more points.

        Args:
            task_id: Stable identifier of the background task.
            user_id: User whose balance should be charged.
            amount: Non-negative number of points to deduct.

        Returns:
            ``True`` if the task is charged or was already charged; otherwise
            ``False`` when the current balance is insufficient.

        Raises:
            ValueError: If identifiers or amount are invalid, or an existing task
                uses different charge parameters.
        """
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("amount must be a non-negative integer")
        task_id = task_id.strip()
        user_id = user_id.strip()

        def operation():
            task_charges = self._get_task_charges_unlocked()
            record = task_charges.get(task_id)
            record_created = record is None
            if record is not None:
                if not isinstance(record, dict):
                    raise ValueError(f"invalid charge record for task {task_id}")
                if record.get("user_id") != user_id or record.get("amount") != amount:
                    raise ValueError(
                        f"conflicting charge parameters for task {task_id}"
                    )
                if record.get("charged") is True:
                    return True, False

            user_data, user_created = self._get_user_data_unlocked(user_id)
            if record is None:
                record = {
                    "user_id": user_id,
                    "amount": amount,
                    "charged": False,
                    "refunded": False,
                }
                task_charges[task_id] = record

            if user_data["points"] < amount:
                return False, record_created or user_created

            user_data["points"] -= amount
            record["charged"] = True
            record["refunded"] = False
            return True, True

        return await self._run_transaction(operation)

    async def refund_task(self, task_id: str) -> bool:
        """Refund a successfully charged task at most once.

        Args:
            task_id: Stable identifier of the background task.

        Returns:
            ``True`` when the task is already refunded or this call applies the
            refund; otherwise ``False`` for unknown or uncharged tasks.

        Raises:
            ValueError: If the task ID or persisted charge record is invalid.
        """
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        task_id = task_id.strip()

        def operation():
            record = self._get_task_charges_unlocked().get(task_id)
            if record is None:
                return False, False
            if not isinstance(record, dict):
                raise ValueError(f"invalid charge record for task {task_id}")
            if record.get("charged") is not True:
                return False, False
            if record.get("refunded") is True:
                return True, False

            user_id = record.get("user_id")
            amount = record.get("amount")
            if not isinstance(user_id, str) or not user_id:
                raise ValueError(f"invalid user_id in charge record for task {task_id}")
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise ValueError(f"invalid amount in charge record for task {task_id}")

            user_data, _ = self._get_user_data_unlocked(user_id)
            user_data["points"] += amount
            record["refunded"] = True
            return True, True

        return await self._run_transaction(operation)

    async def get_task_charge(self, task_id: str) -> dict | None:
        """Return a detached snapshot of one task charge record.

        Args:
            task_id: Stable identifier of the background task.

        Returns:
            A copy of the charge record, or ``None`` if the task is unknown.

        Raises:
            ValueError: If the task ID or persisted charge record is invalid.
        """
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        task_id = task_id.strip()

        def operation():
            record = self._get_task_charges_unlocked().get(task_id)
            if record is None:
                return None, False
            if not isinstance(record, dict):
                raise ValueError(f"invalid charge record for task {task_id}")
            return copy.deepcopy(record), False

        return await self._run_transaction(operation)

    async def checkin(self, user_id: str) -> tuple[bool, int, str]:
        def operation():
            user_data, _ = self._get_user_data_unlocked(user_id)
            last_checkin = user_data.get("last_checkin")
            today = date.today().isoformat()

            if last_checkin == today:
                return (False, 0, "今天已经签到过了哦~"), False

            min_points = self.plugin._config_get("checkin_points_min", 10)
            max_points = self.plugin._config_get("checkin_points_max", 50)
            points = random.randint(min_points, max_points)

            user_data["points"] += points
            user_data["last_checkin"] = today

            points_name = self.plugin.get_points_name()
            return (
                True,
                points,
                f"签到成功！获得 {points} {points_name}，当前{points_name}: {user_data['points']}",
            ), True

        return await self._run_transaction(operation)
