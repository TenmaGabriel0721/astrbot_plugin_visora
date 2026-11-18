"""Durable background image task execution for the Visora plugin.

The manager in this module intentionally has no AstrBot event dependency. An
accepted task stores only serializable routing data and immutable input files;
provider, delivery, economy, and notification behavior is supplied through
process-wide callbacks by the plugin entrypoint.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROMPT_LIMIT = 32_768
INPUT_FILE_LIMIT = 20 * 1024 * 1024
INPUT_TASK_LIMIT = 64 * 1024 * 1024

ACTIVE_STATES = {
    "preparing",
    "queued",
    "running",
    "generated",
    "sending",
}
TERMINAL_STATES = {
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "unknown",
}
ALL_STATES = ACTIVE_STATES | TERMINAL_STATES
DELIVERY_STATES = {
    "not_started",
    "attempting",
    "confirmed",
    "rejected",
    "unknown",
}
NOTIFICATION_STATES = {
    "not_needed",
    "pending",
    "sending",
    "sent",
    "failed",
    "unknown",
    "unavailable",
}
CHARGE_STATES = {"none", "pending", "charged", "rejected"}
REFUND_STATES = {"none", "pending", "refunded", "failed"}

_STATE_LABELS = {
    "preparing": "准备输入",
    "queued": "排队中",
    "running": "生成中",
    "generated": "已生成，等待发送",
    "sending": "发送中",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "interrupted": "已中断",
    "unknown": "发送结果未知",
}


class BackgroundTaskError(RuntimeError):
    """Base error for durable background task operations."""


class QueueFullError(BackgroundTaskError):
    """Raised when the bounded task capacity is exhausted."""


class ChargeRejectedError(BackgroundTaskError):
    """Raised when the configured charge callback rejects a task."""


class DeliveryRejectedError(BackgroundTaskError):
    """Indicate that the transport definitively rejected an image message."""


class DeliveryUnknownError(BackgroundTaskError):
    """Indicate that image delivery may have succeeded and must not be retried."""


def build_scope_key(umo: str, user_id: str, conversation_id: str = "") -> str:
    """Build the stable privacy-preserving key used by every task hook.

    Args:
        umo: AstrBot unified message origin.
        user_id: Requesting user ID.
        conversation_id: Optional AstrBot conversation ID.

    Returns:
        A stable SHA-256 scope key.
    """

    payload = json.dumps(
        [str(umo or ""), str(user_id or ""), str(conversation_id or "")],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Serializable input required to accept one background image task."""

    scope_key: str
    umo: str
    user_id: str
    prompt: str
    mode: str = "txt2img"
    aspect_ratio: str = "1:1"
    resolution: str = "2K"
    preset_name: str = ""
    source_message_id: str = ""
    conversation_id: str = ""
    request_fingerprint: str = ""
    charge_amount: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    """Immutable public view of a persisted background image task."""

    task_id: str
    scope_key: str
    umo: str
    user_id: str
    prompt: str
    mode: str
    aspect_ratio: str
    resolution: str
    preset_name: str
    source_message_id: str
    conversation_id: str
    request_fingerprint: str
    metadata: dict[str, Any]
    state: str
    delivery_state: str
    charge_amount: int
    charge_state: str
    refund_state: str
    cancel_requested: bool
    image_count: int
    elapsed_seconds: float
    send_attempt_id: str
    terminal_reason: str
    error_code: str
    error_message: str
    notification_state: str
    notification_error: str
    created_at_ms: int
    queued_at_ms: int
    started_at_ms: int
    generated_at_ms: int
    finished_at_ms: int
    updated_at_ms: int
    revision: int

    @property
    def is_terminal(self) -> bool:
        """Return whether this snapshot is in an immutable terminal state."""

        return self.state in TERMINAL_STATES


@dataclass(slots=True)
class GeneratedImageResult:
    """Provider output handed to the delivery callback."""

    images: list[bytes]
    elapsed_seconds: float = 0.0
    route_failures: list[str] = field(default_factory=list)
    result_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedDelivery:
    """Pre-send payload whose local resources require deterministic cleanup."""

    payload: Any
    cleanup_runner: Callable[[], Awaitable[None]] | None = None


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """Result of accepting a new task or finding a recent duplicate."""

    task: TaskSnapshot
    created: bool

    @property
    def task_id(self) -> str:
        """Return the accepted or deduplicated task ID."""

        return self.task.task_id

    @property
    def state(self) -> str:
        """Return the task state at admission time."""

        return self.task.state


ProviderRunner = Callable[
    [TaskSnapshot, list[tuple[bytes, str]]], Awaitable[GeneratedImageResult]
]
DeliveryPrepareRunner = Callable[
    [TaskSnapshot, GeneratedImageResult], Awaitable[PreparedDelivery]
]
DeliveryRunner = Callable[[TaskSnapshot, Any], Awaitable[None]]
ChargeRunner = Callable[[str, str, int], Awaitable[bool]]
RefundRunner = Callable[[str], Awaitable[bool]]
NotificationRunner = Callable[[TaskSnapshot, str], Awaitable[bool | None]]


class BackgroundImageTaskManager:
    """Persist, schedule, execute, and terminate background image tasks.

    Args:
        data_dir: Caller-provided plugin data directory.
        provider_runner: Global callback that generates image bytes.
        delivery_prepare_runner: Optional callback for local pre-send work.
        delivery_runner: Global callback that sends generated images.
        charge_runner: Optional idempotent callback keyed by task ID.
        refund_runner: Optional idempotent callback keyed by task ID.
        notify_runner: Optional callback for one terminal user-facing reply.
        max_running: Number of fixed provider workers.
        max_queued: Maximum number of tasks waiting behind active workers.
        dedupe_seconds: Retention window for explicit request fingerprints.
        terminal_ttl_seconds: Retention period for terminal database records.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        provider_runner: ProviderRunner,
        delivery_runner: DeliveryRunner,
        delivery_prepare_runner: DeliveryPrepareRunner | None = None,
        charge_runner: ChargeRunner | None = None,
        refund_runner: RefundRunner | None = None,
        notify_runner: NotificationRunner | None = None,
        max_running: int = 2,
        max_queued: int = 16,
        dedupe_seconds: int = 600,
        terminal_ttl_seconds: int = 7 * 24 * 60 * 60,
    ) -> None:
        self.base_dir = Path(data_dir) / "background_tasks"
        self.db_path = self.base_dir / "background_tasks.sqlite3"
        self.provider_runner = provider_runner
        self.delivery_runner = delivery_runner
        self.delivery_prepare_runner = delivery_prepare_runner
        self.charge_runner = charge_runner
        self.refund_runner = refund_runner
        self.notify_runner = notify_runner
        self.max_running = max(1, min(10, int(max_running)))
        self.max_queued = max(1, min(256, int(max_queued)))
        self.dedupe_seconds = max(0, int(dedupe_seconds))
        self.terminal_ttl_seconds = max(3600, int(terminal_ttl_seconds))

        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.max_queued)
        self._lifecycle_lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()
        self._admission_lock = asyncio.Lock()
        self._refund_lock = asyncio.Lock()
        self._notification_lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._execution_tasks: dict[str, asyncio.Task[None]] = {}
        self._refund_tasks: dict[str, asyncio.Task[bool]] = {}
        self._notification_tasks: dict[str, asyncio.Task[None]] = {}
        self._started = False
        self._accepting = False
        self._closing = False
        self._closed = False

    @staticmethod
    def now_ms() -> int:
        """Return the current Unix timestamp in milliseconds."""

        return int(time.time() * 1000)

    @property
    def accepting(self) -> bool:
        """Return whether the manager currently accepts new tasks."""

        return self._accepting and not self._closing

    async def start(self) -> list[TaskSnapshot]:
        """Serialize startup and return any tasks recovered from a prior run.

        Returns:
            Recovered terminal task snapshots from the startup that performed
            initialization, or an empty list when already started.

        Raises:
            BackgroundTaskError: The manager is closed or storage initialization
                fails.
        """

        async with self._lifecycle_lock:
            return await self._start_locked()

    async def _start_locked(self) -> list[TaskSnapshot]:
        """Initialize storage, recover abandoned tasks, and start fixed workers.

        Returns:
            Recovered terminal task snapshots.

        Raises:
            BackgroundTaskError: The database cannot be initialized safely.
        """

        if self._started:
            return []
        if self._closing or self._closed:
            raise BackgroundTaskError("A closed task manager cannot be restarted")

        self._prepare_storage_sync()
        async with self._db_lock:
            recovered = self._recover_abandoned_sync()

        settlement_attempted: set[str] = set()
        for snapshot in recovered:
            if snapshot.state == "interrupted":
                await self._reconcile_charge_and_refund(snapshot.task_id)
                settlement_attempted.add(snapshot.task_id)
            await self._cleanup_task_files(snapshot.task_id)

        refundable_tasks = self._list_refundable_tasks_sync()
        for snapshot in refundable_tasks:
            if snapshot.task_id in settlement_attempted:
                continue
            if snapshot.charge_state == "pending":
                await self._reconcile_charge_and_refund(snapshot.task_id)
            else:
                await self._refund_once(snapshot.task_id)
            await self._cleanup_task_files(snapshot.task_id)

        pending_notifications = await self._list_pending_notifications()
        for snapshot in pending_notifications:
            await self._notify_once(
                snapshot.task_id,
                wait=snapshot.state != "completed",
            )

        self._workers = [
            asyncio.create_task(
                self._worker_loop(index),
                name=f"visora-image-worker-{index}",
            )
            for index in range(self.max_running)
        ]
        for worker in self._workers:
            worker.add_done_callback(self._consume_task_exception)
        self._started = True
        self._accepting = True
        return [await self.get(item.task_id) or item for item in recovered]

    async def accept(
        self,
        spec: TaskSpec,
        input_images: Sequence[bytes | tuple[bytes, str]] = (),
    ) -> AdmissionResult:
        """Persist, charge, spool, and enqueue one image task.

        The database reservation is created before any charge. The tool-facing
        caller should only report acceptance after this method returns.

        Args:
            spec: Serializable task parameters and delivery routing fields.
            input_images: Image bytes, optionally paired with a MIME type.

        Returns:
            The accepted task, or a recent task with the same explicit
            fingerprint.

        Raises:
            QueueFullError: The configured active and queued capacity is full.
            ChargeRejectedError: The global charge callback rejected the task.
            BackgroundTaskError: Input validation, storage, or lifecycle failed.
            asyncio.CancelledError: The accepting pipeline turn was cancelled.
        """

        if not self.accepting:
            raise BackgroundTaskError("Background image tasks are not accepting work")
        normalized_spec = self._validate_spec(spec)
        normalized_inputs = self._normalize_inputs(input_images)
        task_id = self._new_task_id()

        async with self._db_lock:
            snapshot, created = self._reserve_task_sync(task_id, normalized_spec)
        if not created:
            return AdmissionResult(snapshot, False)

        charged = False
        try:
            manifest = self._spool_inputs_sync(task_id, normalized_inputs)
            await self._update_columns(
                task_id,
                input_manifest_json=json.dumps(
                    manifest,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )

            if normalized_spec.charge_amount > 0:
                if self.charge_runner is None or self.refund_runner is None:
                    raise BackgroundTaskError(
                        "Charged tasks require global charge and refund callbacks"
                    )
                await self._update_columns(task_id, charge_state="pending")
                charged = bool(
                    await self.charge_runner(
                        task_id,
                        normalized_spec.user_id,
                        normalized_spec.charge_amount,
                    )
                )
                if not charged:
                    await self._update_columns(task_id, charge_state="rejected")
                    await self._finish_terminal(
                        task_id,
                        expected={"preparing"},
                        state="failed",
                        terminal_reason="charge_rejected",
                        error_code="charge_rejected",
                        error_message="Task charge was rejected.",
                        notify=False,
                    )
                    await self._cleanup_task_files(task_id)
                    raise ChargeRejectedError("Task charge was rejected")
                await self._update_columns(task_id, charge_state="charged")

            async with self._admission_lock:
                current = await self.get(task_id)
                if current is None:
                    raise BackgroundTaskError("Accepted task record disappeared")
                if self._closing or current.cancel_requested or current.is_terminal:
                    terminal_state = "interrupted" if self._closing else "cancelled"
                    if not current.is_terminal:
                        await self._finish_terminal(
                            task_id,
                            expected={"preparing"},
                            state=terminal_state,
                            terminal_reason=terminal_state,
                            error_code=terminal_state,
                            error_message="Task stopped before entering the queue.",
                            notify=self._closing,
                        )
                    if charged:
                        await self._refund_once(task_id)
                    await self._cleanup_task_files(task_id)
                    if self._closing:
                        raise BackgroundTaskError("Task manager is closing")
                    raise asyncio.CancelledError

                queued = await self._transition(
                    task_id,
                    expected={"preparing"},
                    state="queued",
                    queued_at_ms=self.now_ms(),
                )
                if queued is None:
                    raise BackgroundTaskError("Task could not enter the queue")
                await self._queue.put(task_id)

            accepted = await self.get(task_id)
            if accepted is None:
                raise BackgroundTaskError("Accepted task record disappeared")
            return AdmissionResult(accepted, True)
        except asyncio.CancelledError:
            await self._settle_aborted_accept(task_id, charged, cancelled=True)
            raise
        except ChargeRejectedError:
            raise
        except BaseException:
            await self._settle_aborted_accept(task_id, charged, cancelled=False)
            raise

    async def get(self, task_id: str) -> TaskSnapshot | None:
        """Return one task snapshot by ID.

        Args:
            task_id: Durable task identifier.

        Returns:
            The task snapshot, or ``None`` when absent.
        """

        return self._get_sync(str(task_id))

    async def list_for_scope(
        self,
        scope_key: str,
        *,
        limit: int = 20,
        active_only: bool = False,
    ) -> list[TaskSnapshot]:
        """List newest tasks belonging to one stable scope key.

        Args:
            scope_key: Value returned by :func:`build_scope_key`.
            limit: Maximum number of rows returned.
            active_only: Only return nonterminal tasks when true.

        Returns:
            Newest-first task snapshots.
        """

        return self._list_for_scope_sync(
            str(scope_key),
            max(1, min(100, int(limit))),
            bool(active_only),
        )

    async def cancel_scope(
        self,
        *,
        scope_key: str | None = None,
        umo: str | None = None,
        user_id: str | None = None,
        reason: str = "cancelled",
    ) -> int:
        """Cancel matching queued or executing tasks without replaying delivery.

        A caller may use the exact scope key, or the broader ``umo`` and
        ``user_id`` pair after a conversation reset made the old CID unavailable.
        A task already inside its platform send await is cancelled into the
        ``unknown`` terminal state because remote acceptance cannot be disproved.

        Args:
            scope_key: Optional exact stable scope key.
            umo: Optional unified message origin filter.
            user_id: Optional requester filter.
            reason: Persisted cancellation reason.

        Returns:
            Number of matching active tasks marked for cancellation.

        Raises:
            ValueError: No selection filter was provided.
        """

        if not scope_key and not (umo and user_id):
            raise ValueError(
                "cancel_scope requires scope_key or the umo and user_id pair"
            )
        snapshots = self._list_matching_active_sync(
            scope_key,
            umo,
            user_id,
        )
        cancelled = 0
        for snapshot in snapshots:
            updated = await self._update_columns(
                snapshot.task_id,
                cancel_requested=1,
                terminal_reason=self._sanitize_text(reason, 300),
            )
            if not updated:
                continue

            while True:
                current = await self.get(snapshot.task_id)
                if current is None or current.is_terminal:
                    break

                execution = self._execution_tasks.get(snapshot.task_id)
                if current.state == "sending":
                    changed = await self._mark_delivery_unknown(
                        snapshot.task_id,
                        "cancelled_during_send",
                        "Delivery was cancelled after transport started.",
                    )
                    if not changed:
                        continue
                    if execution is not None and not execution.done():
                        execution.cancel()
                    await self._cleanup_task_files(snapshot.task_id)
                    await self._notify_once(snapshot.task_id, wait=False)
                    cancelled += 1
                    break

                terminal = await self._finish_terminal(
                    snapshot.task_id,
                    expected={current.state},
                    state="cancelled",
                    terminal_reason=reason,
                    error_code="cancelled",
                    error_message=(
                        "Task was cancelled before delivery."
                        if current.state in {"preparing", "queued"}
                        else "Task was cancelled."
                    ),
                    notify=False,
                )
                if terminal is None:
                    continue
                if execution is not None and not execution.done():
                    execution.cancel()
                await self._refund_once(snapshot.task_id)
                await self._cleanup_task_files(snapshot.task_id)
                cancelled += 1
                break
        return cancelled

    async def format_scope_status(self, scope_key: str, *, limit: int = 5) -> str:
        """Format concise real task facts for an LLM status tool or hook.

        Args:
            scope_key: Stable task scope key.
            limit: Maximum number of recent tasks included.

        Returns:
            Human-readable Chinese status text without fabricated percentages.
        """

        tasks = await self.list_for_scope(scope_key, limit=limit)
        if not tasks:
            return "当前没有生图任务记录。"
        now = self.now_ms()
        lines = ["当前生图任务："]
        for task in tasks:
            end = task.finished_at_ms or now
            elapsed = max(0, end - (task.started_at_ms or task.created_at_ms)) // 1000
            prompt = re.sub(r"\s+", " ", task.prompt).strip()
            if len(prompt) > 80:
                prompt = prompt[:77] + "..."
            detail = (
                f"- {task.task_id}: {_STATE_LABELS.get(task.state, task.state)}，"
                f"已用时 {elapsed}s，提示词：{prompt or '（空）'}"
            )
            if task.delivery_state == "unknown":
                detail += "；发送结果不确定，未自动重发"
            lines.append(detail)
        return "\n".join(lines)

    async def close(self, *, grace_seconds: float = 10.0) -> None:
        """Serialize shutdown and durably settle all unfinished work.

        Args:
            grace_seconds: Maximum wait for execution and worker cancellation.
        """

        async with self._lifecycle_lock:
            if self._closed:
                return
            self._accepting = False
            self._closing = True
            try:
                await self._close_locked(grace_seconds=grace_seconds)
            except BaseException:
                self._closing = False
                raise
            self._closed = True

    async def _close_locked(self, *, grace_seconds: float = 10.0) -> None:
        """Stop intake and durably settle all unfinished work.

        Args:
            grace_seconds: Maximum wait for execution and worker cancellation.
        """

        async with self._admission_lock:
            active = self._list_matching_active_sync(
                None,
                None,
                None,
                True,
            )
            for snapshot in active:
                await self._update_columns(
                    snapshot.task_id,
                    cancel_requested=1,
                    terminal_reason="shutdown",
                )

        for execution in list(self._execution_tasks.values()):
            if not execution.done():
                execution.cancel()
        if self._execution_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *list(self._execution_tasks.values()),
                        return_exceptions=True,
                    ),
                    timeout=max(1.0, float(grace_seconds)),
                )
            except TimeoutError:
                pass

        for worker in self._workers:
            if not worker.done():
                worker.cancel()
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=max(1.0, float(grace_seconds)),
                )
            except TimeoutError:
                pass

        notification_tasks = list(self._notification_tasks.values())
        for task in notification_tasks:
            if not task.done():
                task.cancel()
        if notification_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*notification_tasks, return_exceptions=True),
                    timeout=max(1.0, float(grace_seconds)),
                )
            except TimeoutError:
                logger.warning(
                    "Timed out while stopping background image notifications"
                )

        async with self._db_lock:
            recovered = self._recover_abandoned_sync()
        for snapshot in recovered:
            if snapshot.state == "interrupted":
                await self._reconcile_charge_and_refund(snapshot.task_id)
            await self._cleanup_task_files(snapshot.task_id)

        pending_refunds = list(self._refund_tasks.values())
        if pending_refunds:
            await asyncio.gather(*pending_refunds, return_exceptions=True)

        self._workers.clear()
        self._execution_tasks.clear()
        self._refund_tasks.clear()
        self._notification_tasks.clear()
        self._started = False

    async def _worker_loop(self, index: int) -> None:
        """Run one fixed worker that owns at most one provider call at a time.

        Args:
            index: Stable worker index used only for diagnostics.
        """

        del index
        while True:
            task_id = await self._queue.get()
            try:
                snapshot = await self.get(task_id)
                if snapshot is None or snapshot.state != "queued":
                    continue
                execution = asyncio.create_task(
                    self._execute_task(task_id),
                    name=f"visora-image-execution-{task_id}",
                )
                self._execution_tasks[task_id] = execution
                try:
                    await execution
                except asyncio.CancelledError:
                    worker = asyncio.current_task()
                    if worker is not None and worker.cancelling():
                        raise
                except BaseException as exc:
                    logger.exception(
                        "Background image execution escaped its settlement: task_id=%s",
                        task_id,
                    )
                    try:
                        await self._settle_escaped_execution(task_id, exc)
                    except BaseException:
                        logger.exception(
                            "Failed to settle escaped background image execution: "
                            "task_id=%s",
                            task_id,
                        )
                finally:
                    if self._execution_tasks.get(task_id) is execution:
                        self._execution_tasks.pop(task_id, None)
            finally:
                self._queue.task_done()

    async def _execute_task(self, task_id: str) -> None:
        """Execute one persisted task through provider and delivery phases.

        Args:
            task_id: Durable task ID dequeued by a fixed worker.
        """

        try:
            snapshot = await self._transition(
                task_id,
                expected={"queued"},
                state="running",
                started_at_ms=self.now_ms(),
            )
            if snapshot is None:
                return
            if snapshot.cancel_requested:
                raise asyncio.CancelledError

            images = self._read_spooled_inputs_sync(task_id)
            generated = await self.provider_runner(snapshot, images)
            if not isinstance(generated, GeneratedImageResult):
                raise TypeError("provider_runner must return GeneratedImageResult")
            if not generated.images or any(
                not isinstance(image, bytes) or not image for image in generated.images
            ):
                raise BackgroundTaskError("Provider returned no usable image bytes")

            current = await self.get(task_id)
            if current is None or current.cancel_requested:
                raise asyncio.CancelledError
            snapshot = await self._transition(
                task_id,
                expected={"running"},
                state="generated",
                generated_at_ms=self.now_ms(),
                image_count=len(generated.images),
                elapsed_seconds=max(0.0, float(generated.elapsed_seconds)),
                metadata_json=json.dumps(
                    {
                        **snapshot.metadata,
                        "generation_result": generated.result_metadata,
                    },
                    ensure_ascii=False,
                ),
            )
            if snapshot is None:
                return
            if snapshot.cancel_requested:
                raise asyncio.CancelledError

            prepared = PreparedDelivery(payload=generated)
            try:
                if self.delivery_prepare_runner is not None:
                    prepared = await self.delivery_prepare_runner(snapshot, generated)
                    if not isinstance(prepared, PreparedDelivery):
                        raise TypeError(
                            "delivery_prepare_runner must return PreparedDelivery"
                        )

                current = await self.get(task_id)
                if current is None or current.cancel_requested:
                    raise asyncio.CancelledError

                attempt_id = f"send_{uuid.uuid4().hex}"
                snapshot = await self._begin_delivery(task_id, attempt_id)
                if snapshot is None:
                    return
                await self.delivery_runner(snapshot, prepared.payload)
                await self._confirm_delivery(task_id, attempt_id)
            finally:
                await self._cleanup_prepared_delivery(prepared, task_id)
            await self._cleanup_task_files(task_id)
            await self._notify_once(task_id, wait=False)
        except asyncio.CancelledError:
            current = await self.get(task_id)
            if current is None or current.is_terminal:
                return
            if current.state == "sending" or current.delivery_state == "attempting":
                await self._mark_delivery_unknown(
                    task_id,
                    "cancelled_during_send",
                    "Delivery was cancelled after transport started.",
                )
                await self._cleanup_task_files(task_id)
                if not self._closing:
                    await self._notify_once(task_id)
                return
            terminal_state = "interrupted" if self._closing else "cancelled"
            await self._finish_terminal(
                task_id,
                expected={current.state},
                state=terminal_state,
                terminal_reason="shutdown" if self._closing else "cancelled",
                error_code=terminal_state,
                error_message=(
                    "Task was interrupted by plugin shutdown."
                    if self._closing
                    else "Task was cancelled."
                ),
                notify=self._closing,
            )
            await self._refund_once(task_id)
            await self._cleanup_task_files(task_id)
        except DeliveryRejectedError as exc:
            current = await self.get(task_id)
            if current is not None and (
                current.state == "sending" or current.delivery_state == "attempting"
            ):
                await self._mark_delivery_rejected(task_id, exc)
            elif current is not None and not current.is_terminal:
                await self._finish_terminal(
                    task_id,
                    expected={current.state},
                    state="failed",
                    terminal_reason="delivery_rejected",
                    error_code="delivery_rejected",
                    error_message=self._sanitize_text(exc, 1000),
                    notify=True,
                )
            await self._refund_once(task_id)
            await self._cleanup_task_files(task_id)
            await self._notify_once(task_id)
        except (DeliveryUnknownError, TimeoutError, ConnectionError) as exc:
            current = await self.get(task_id)
            if current is not None and (
                current.state == "sending" or current.delivery_state == "attempting"
            ):
                await self._mark_delivery_unknown(
                    task_id,
                    "delivery_unknown",
                    self._sanitize_text(exc, 1000),
                )
            elif current is not None and not current.is_terminal:
                await self._finish_terminal(
                    task_id,
                    expected={current.state},
                    state="failed",
                    terminal_reason="provider_failed",
                    error_code="provider_failed",
                    error_message=self._sanitize_text(exc, 1000),
                    notify=True,
                )
                await self._refund_once(task_id)
            await self._cleanup_task_files(task_id)
            await self._notify_once(task_id)
        except BaseException as exc:
            current = await self.get(task_id)
            if current is not None and (
                current.state == "sending" or current.delivery_state == "attempting"
            ):
                await self._mark_delivery_unknown(
                    task_id,
                    "delivery_unknown",
                    self._sanitize_text(exc, 1000),
                )
                await self._cleanup_task_files(task_id)
                await self._notify_once(task_id)
                return
            if current is not None and not current.is_terminal:
                await self._finish_terminal(
                    task_id,
                    expected={current.state},
                    state="failed",
                    terminal_reason="provider_failed",
                    error_code="provider_failed",
                    error_message=self._sanitize_text(exc, 1000),
                    notify=True,
                )
            await self._refund_once(task_id)
            await self._cleanup_task_files(task_id)
            await self._notify_once(task_id)

    async def _settle_escaped_execution(
        self,
        task_id: str,
        error: BaseException,
    ) -> None:
        """Best-effort settlement when an execution handler itself escapes.

        Args:
            task_id: Durable task whose execution escaped unexpectedly.
            error: Escaped exception used only for a sanitized diagnostic.
        """

        current = await self.get(task_id)
        if current is None or current.is_terminal:
            return
        if current.state == "sending" or current.delivery_state == "attempting":
            await self._mark_delivery_unknown(
                task_id,
                "execution_settlement_unknown",
                self._sanitize_text(error, 1000),
            )
            await self._cleanup_task_files(task_id)
            if not self._closing:
                await self._notify_once(task_id)
            return

        terminal = await self._finish_terminal(
            task_id,
            expected={current.state},
            state="interrupted" if self._closing else "failed",
            terminal_reason="execution_settlement_failed",
            error_code="execution_settlement_failed",
            error_message=self._sanitize_text(error, 1000),
            notify=True,
        )
        if terminal is None:
            return
        await self._refund_once(task_id)
        await self._cleanup_task_files(task_id)
        if not self._closing:
            await self._notify_once(task_id)

    async def _begin_delivery(
        self,
        task_id: str,
        attempt_id: str,
    ) -> TaskSnapshot | None:
        """Persist the external side-effect boundary before platform sending.

        Args:
            task_id: Task entering its send phase.
            attempt_id: Unique transport attempt ID.

        Returns:
            Updated sending snapshot, or ``None`` if the state changed.
        """

        now = self.now_ms()
        async with self._db_lock:
            return self._begin_delivery_sync(task_id, attempt_id, now)

    @staticmethod
    async def _cleanup_prepared_delivery(
        prepared: PreparedDelivery,
        task_id: str,
    ) -> None:
        """Release local delivery resources without changing task settlement.

        Args:
            prepared: Prepared payload and optional cleanup callback.
            task_id: Durable task ID used for diagnostics.
        """

        if prepared.cleanup_runner is None:
            return
        try:
            await prepared.cleanup_runner()
        except BaseException as exc:
            logger.warning(
                "Prepared delivery cleanup failed: task_id=%s error=%s",
                task_id,
                BackgroundImageTaskManager._sanitize_text(exc, 300),
            )

    async def _confirm_delivery(self, task_id: str, attempt_id: str) -> None:
        """Atomically confirm transport acceptance and complete the task.

        Args:
            task_id: Sending task ID.
            attempt_id: Attempt previously persisted by ``_begin_delivery``.
        """

        async with self._db_lock:
            changed = self._confirm_delivery_sync(
                task_id,
                attempt_id,
                self.now_ms(),
            )
        if not changed:
            raise BackgroundTaskError("Delivery confirmation lost its task state")

    async def _mark_delivery_rejected(
        self,
        task_id: str,
        error: BaseException | str,
    ) -> None:
        """Persist a definitive transport rejection as a refundable failure.

        Args:
            task_id: Sending task ID.
            error: Sanitized transport error source.
        """

        async with self._db_lock:
            self._finish_delivery_sync(
                task_id,
                "failed",
                "rejected",
                "delivery_rejected",
                self._sanitize_text(error, 1000),
                self.now_ms(),
                True,
            )

    async def _mark_delivery_unknown(
        self,
        task_id: str,
        error_code: str,
        error_message: str,
    ) -> bool:
        """Persist an ambiguous transport result and prohibit automatic retry.

        Args:
            task_id: Sending task ID.
            error_code: Stable machine-readable reason.
            error_message: Sanitized diagnostic text.

        Returns:
            Whether the sending task was changed to the unknown terminal state.
        """

        async with self._db_lock:
            return self._finish_delivery_sync(
                task_id,
                "unknown",
                "unknown",
                error_code,
                self._sanitize_text(error_message, 1000),
                self.now_ms(),
                True,
            )

    async def _finish_terminal(
        self,
        task_id: str,
        *,
        expected: set[str],
        state: str,
        terminal_reason: str,
        error_code: str,
        error_message: str,
        notify: bool,
    ) -> TaskSnapshot | None:
        """Move a task into a non-delivery terminal state.

        Args:
            task_id: Durable task ID.
            expected: States allowed to make this transition.
            state: Terminal state to write.
            terminal_reason: User-meaningful terminal category.
            error_code: Stable internal error code.
            error_message: Sanitized diagnostic message.
            notify: Whether a deterministic notification should be queued.

        Returns:
            Updated snapshot, or ``None`` when the expected state did not match.
        """

        return await self._transition(
            task_id,
            expected=expected,
            state=state,
            terminal_reason=self._sanitize_text(terminal_reason, 300),
            error_code=self._sanitize_text(error_code, 100),
            error_message=self._sanitize_text(error_message, 1000),
            notification_state="pending" if notify else "not_needed",
            finished_at_ms=self.now_ms(),
            expires_at_ms=self.now_ms() + self.terminal_ttl_seconds * 1000,
        )

    async def _transition(
        self,
        task_id: str,
        *,
        expected: set[str],
        state: str,
        **updates: Any,
    ) -> TaskSnapshot | None:
        """Apply one compare-and-set state transition.

        Args:
            task_id: Durable task ID.
            expected: Current states accepted by the transition.
            state: New state.
            **updates: Additional whitelisted persisted columns.

        Returns:
            Updated snapshot, or ``None`` when the compare-and-set failed.
        """

        if state not in ALL_STATES:
            raise BackgroundTaskError(f"Unknown task state: {state}")
        async with self._db_lock:
            return self._transition_sync(
                task_id,
                expected,
                state,
                updates,
                self.now_ms(),
            )

    async def _update_columns(self, task_id: str, **updates: Any) -> bool:
        """Update whitelisted non-state task columns.

        Args:
            task_id: Durable task ID.
            **updates: Persisted column values.

        Returns:
            Whether one row was updated.
        """

        async with self._db_lock:
            return self._update_columns_sync(
                task_id,
                updates,
                self.now_ms(),
            )

    async def _settle_aborted_accept(
        self,
        task_id: str,
        charged: bool,
        *,
        cancelled: bool,
    ) -> None:
        """Best-effort terminal settlement for an incomplete accept call.

        Args:
            task_id: Partially accepted task ID.
            charged: Whether the caller observed a successful charge.
            cancelled: Whether pipeline cancellation caused the abort.
        """

        snapshot = await self.get(task_id)
        if snapshot is not None and not snapshot.is_terminal:
            state = (
                "interrupted"
                if self._closing
                else ("cancelled" if cancelled else "failed")
            )
            await self._finish_terminal(
                task_id,
                expected={snapshot.state},
                state=state,
                terminal_reason="accept_aborted",
                error_code="accept_aborted",
                error_message="Task acceptance did not complete.",
                notify=self._closing,
            )
        snapshot = await self.get(task_id)
        if (
            snapshot is not None
            and snapshot.charge_amount > 0
            and snapshot.charge_state in {"pending", "charged"}
        ):
            try:
                await self._reconcile_charge_and_refund(task_id)
            except BaseException as exc:
                logger.error(
                    "Failed to reconcile an aborted task charge: task_id=%s error=%s",
                    task_id,
                    self._sanitize_text(exc, 300),
                )
        elif charged:
            await self._refund_once(task_id)
        await self._cleanup_task_files(task_id)

    async def _reconcile_charge_and_refund(self, task_id: str) -> None:
        """Reconcile a crash-window charge before issuing an idempotent refund.

        Args:
            task_id: Recovered interrupted task ID.
        """

        snapshot = await self.get(task_id)
        if snapshot is None or snapshot.charge_amount <= 0:
            return
        if snapshot.charge_state == "pending" and self.charge_runner is not None:
            charged = bool(
                await self.charge_runner(
                    snapshot.task_id,
                    snapshot.user_id,
                    snapshot.charge_amount,
                )
            )
            await self._update_columns(
                task_id,
                charge_state="charged" if charged else "rejected",
            )
        await self._refund_once(task_id)

    async def _refund_once(self, task_id: str) -> bool:
        """Run the global idempotent refund callback at most once concurrently.

        Args:
            task_id: Charged task to compensate.

        Returns:
            Whether the callback reported a completed refund.
        """

        snapshot = await self.get(task_id)
        if (
            snapshot is None
            or snapshot.state not in {"failed", "cancelled", "interrupted"}
            or snapshot.charge_amount <= 0
            or snapshot.charge_state != "charged"
            or snapshot.refund_state == "refunded"
        ):
            return False
        if self.refund_runner is None:
            await self._update_columns(task_id, refund_state="failed")
            return False

        async with self._refund_lock:
            refund_task = self._refund_tasks.get(task_id)
            if refund_task is None or refund_task.done():
                refund_task = asyncio.create_task(
                    self._perform_refund(task_id),
                    name=f"visora-image-refund-{task_id}",
                )
                refund_task.add_done_callback(self._consume_task_exception)
                self._refund_tasks[task_id] = refund_task
        try:
            return await asyncio.shield(refund_task)
        finally:
            if refund_task.done() and self._refund_tasks.get(task_id) is refund_task:
                self._refund_tasks.pop(task_id, None)

    async def _perform_refund(self, task_id: str) -> bool:
        """Persist and execute one idempotent refund attempt.

        Args:
            task_id: Charged task ID.

        Returns:
            Whether the external economy callback completed the refund.
        """

        snapshot = await self.get(task_id)
        if (
            snapshot is None
            or snapshot.state not in {"failed", "cancelled", "interrupted"}
            or snapshot.refund_state == "refunded"
        ):
            return False
        await self._update_columns(task_id, refund_state="pending")
        try:
            refunded = bool(await self.refund_runner(task_id))  # type: ignore[misc]
        except BaseException:
            await self._update_columns(task_id, refund_state="failed")
            logger.exception("Background image refund failed: task_id=%s", task_id)
            return False
        await self._update_columns(
            task_id,
            refund_state="refunded" if refunded else "failed",
            expires_at_ms=self.now_ms() + self.terminal_ttl_seconds * 1000,
        )
        return refunded

    async def _notify_once(self, task_id: str, *, wait: bool = True) -> None:
        """Start one terminal notification without replaying generated images.

        Args:
            task_id: Terminal task requiring a text notification.
            wait: Whether to wait for the notification transport to settle.
        """

        snapshot = await self.get(task_id)
        if snapshot is None or snapshot.notification_state != "pending":
            return
        if self.notify_runner is None:
            await self._update_columns(task_id, notification_state="unavailable")
            return

        async with self._notification_lock:
            existing = self._notification_tasks.get(task_id)
            if existing is None or existing.done():
                task = asyncio.create_task(
                    self._perform_notification(task_id),
                    name=f"visora-image-notify-{task_id}",
                )
                task.add_done_callback(self._consume_task_exception)

                def forget_notification(completed: asyncio.Task[None]) -> None:
                    if self._notification_tasks.get(task_id) is completed:
                        self._notification_tasks.pop(task_id, None)

                task.add_done_callback(forget_notification)
                self._notification_tasks[task_id] = task
            else:
                task = existing
        if not wait:
            return
        try:
            await asyncio.shield(task)
        finally:
            if task.done() and self._notification_tasks.get(task_id) is task:
                self._notification_tasks.pop(task_id, None)

    async def _perform_notification(self, task_id: str) -> None:
        """Execute one durable terminal notification attempt.

        Args:
            task_id: Terminal task ID.
        """

        snapshot = await self.get(task_id)
        if snapshot is None or snapshot.notification_state != "pending":
            return
        await self._update_columns(
            task_id,
            notification_state="sending",
            notification_error="",
        )
        snapshot = await self.get(task_id)
        if snapshot is None:
            return
        text = self._notification_text(snapshot)
        try:
            sent = await self.notify_runner(snapshot, text)  # type: ignore[misc]
        except asyncio.CancelledError as exc:
            await self._update_columns(
                task_id,
                notification_state="unknown",
                notification_error=self._sanitize_text(exc, 500)
                or "Notification was cancelled during shutdown.",
            )
            raise
        except (DeliveryUnknownError, TimeoutError, ConnectionError) as exc:
            await self._update_columns(
                task_id,
                notification_state="unknown",
                notification_error=self._sanitize_text(exc, 500),
            )
        except BaseException as exc:
            await self._update_columns(
                task_id,
                notification_state="failed",
                notification_error=self._sanitize_text(exc, 500),
            )
            logger.warning(
                "Background image notification failed: task_id=%s error=%s",
                task_id,
                self._sanitize_text(exc, 300),
            )
        else:
            await self._update_columns(
                task_id,
                notification_state="not_needed" if sent is False else "sent",
                notification_error="",
            )

    @staticmethod
    def _notification_text(snapshot: TaskSnapshot) -> str:
        if snapshot.state == "completed" and snapshot.delivery_state == "confirmed":
            return "照片拍好啦，我已经发出来了。"
        if snapshot.state == "interrupted":
            return "刚才的生图任务因为服务重载或重启中断了，没有继续执行。"
        if snapshot.state == "cancelled":
            return "刚才的生图任务已取消，没有继续执行。"
        if snapshot.state == "unknown":
            return "图片可能已经发送，但结果无法确认；为避免重复，本次没有自动重发。"
        if snapshot.delivery_state == "rejected":
            return "图片已经生成，但消息平台明确拒绝了发送，任务已经结束。"
        if snapshot.error_code == "provider_failed":
            error_detail = re.sub(r"\s+", " ", snapshot.error_message).strip()
            if len(error_detail) > 500:
                error_detail = error_detail[:497] + "..."
            if error_detail:
                return f"刚才那次生图没有成功：{error_detail}"
        return "刚才那次生图没有成功，任务已经结束了。"

    async def _list_pending_notifications(self) -> list[TaskSnapshot]:
        return self._list_pending_notifications_sync()

    def _prepare_storage_sync(self) -> None:
        """Create and validate the private SQLite task store."""

        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.base_dir, 0o700)
        except OSError:
            pass
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                "schema_version INTEGER NOT NULL)"
            )
            row = conn.execute(
                "SELECT schema_version FROM schema_meta WHERE singleton=1"
            ).fetchone()
            if row is not None and int(row[0]) > SCHEMA_VERSION:
                conn.rollback()
                raise BackgroundTaskError(
                    f"Unsupported background task schema: {row[0]}"
                )
            conn.execute(
                "INSERT INTO schema_meta(singleton, schema_version) VALUES(1, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET schema_version=excluded.schema_version",
                (SCHEMA_VERSION,),
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS image_tasks ("
                "task_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, umo TEXT NOT NULL, "
                "user_id TEXT NOT NULL, prompt TEXT NOT NULL, mode TEXT NOT NULL, "
                "aspect_ratio TEXT NOT NULL, resolution TEXT NOT NULL, "
                "preset_name TEXT NOT NULL DEFAULT '', source_message_id TEXT NOT NULL DEFAULT '', "
                "conversation_id TEXT NOT NULL DEFAULT '', request_fingerprint TEXT NOT NULL DEFAULT '', "
                "metadata_json TEXT NOT NULL DEFAULT '{}', input_manifest_json TEXT NOT NULL DEFAULT '[]', "
                "state TEXT NOT NULL, delivery_state TEXT NOT NULL DEFAULT 'not_started', "
                "charge_amount INTEGER NOT NULL DEFAULT 0, charge_state TEXT NOT NULL DEFAULT 'none', "
                "refund_state TEXT NOT NULL DEFAULT 'none', cancel_requested INTEGER NOT NULL DEFAULT 0, "
                "image_count INTEGER NOT NULL DEFAULT 0, elapsed_seconds REAL NOT NULL DEFAULT 0, "
                "send_attempt_id TEXT NOT NULL DEFAULT '', terminal_reason TEXT NOT NULL DEFAULT '', "
                "error_code TEXT NOT NULL DEFAULT '', error_message TEXT NOT NULL DEFAULT '', "
                "notification_state TEXT NOT NULL DEFAULT 'not_needed', "
                "notification_error TEXT NOT NULL DEFAULT '', created_at_ms INTEGER NOT NULL, "
                "queued_at_ms INTEGER NOT NULL DEFAULT 0, started_at_ms INTEGER NOT NULL DEFAULT 0, "
                "generated_at_ms INTEGER NOT NULL DEFAULT 0, finished_at_ms INTEGER NOT NULL DEFAULT 0, "
                "updated_at_ms INTEGER NOT NULL, expires_at_ms INTEGER NOT NULL, "
                "revision INTEGER NOT NULL DEFAULT 1)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS delivery_attempts ("
                "attempt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES image_tasks(task_id) "
                "ON DELETE CASCADE, state TEXT NOT NULL, transport TEXT NOT NULL DEFAULT 'context', "
                "error_digest TEXT NOT NULL DEFAULT '', started_at_ms INTEGER NOT NULL, "
                "finished_at_ms INTEGER NOT NULL DEFAULT 0)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_tasks_state_updated "
                "ON image_tasks(state, updated_at_ms DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_tasks_scope_created "
                "ON image_tasks(scope_key, created_at_ms DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_tasks_fingerprint_created "
                "ON image_tasks(request_fingerprint, created_at_ms DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_tasks_scope_fingerprint_created "
                "ON image_tasks(scope_key, request_fingerprint, created_at_ms DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_tasks_expiry "
                "ON image_tasks(expires_at_ms)"
            )
            conn.commit()
            check = conn.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise BackgroundTaskError(f"SQLite quick_check failed: {check}")
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    @contextmanager
    def _connect(self):
        """Yield one short-lived configured SQLite connection.

        Yields:
            A connection that is always explicitly closed after the transaction.
        """

        conn = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
        finally:
            conn.close()

    def _reserve_task_sync(
        self,
        task_id: str,
        spec: TaskSpec,
    ) -> tuple[TaskSnapshot, bool]:
        """Atomically deduplicate, enforce capacity, and insert preparing state.

        Args:
            task_id: New candidate task ID.
            spec: Validated task specification.

        Returns:
            Snapshot plus whether a new row was created.

        Raises:
            QueueFullError: Active capacity is exhausted.
        """

        now = self.now_ms()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM image_tasks WHERE expires_at_ms<? "
                "AND notification_state NOT IN ('pending','sending') AND ("
                "state IN ('completed','unknown') OR ("
                "state IN ('failed','cancelled','interrupted') AND ("
                "charge_state IN ('none','rejected') OR refund_state='refunded'))) ",
                (now,),
            )
            if spec.request_fingerprint and self.dedupe_seconds > 0:
                duplicate = conn.execute(
                    "SELECT * FROM image_tasks WHERE scope_key=? "
                    "AND request_fingerprint=? "
                    "AND created_at_ms>=? ORDER BY created_at_ms DESC LIMIT 1",
                    (
                        spec.scope_key,
                        spec.request_fingerprint,
                        now - self.dedupe_seconds * 1000,
                    ),
                ).fetchone()
                if duplicate is not None:
                    conn.commit()
                    return self._row_to_snapshot(duplicate), False

            placeholders = ",".join("?" for _ in ACTIVE_STATES)
            active_count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM image_tasks WHERE state IN ({placeholders})",
                    tuple(sorted(ACTIVE_STATES)),
                ).fetchone()[0]
            )
            if active_count >= self.max_running + self.max_queued:
                conn.rollback()
                raise QueueFullError(
                    f"Background image queue is full ({active_count}/"
                    f"{self.max_running + self.max_queued})"
                )

            conn.execute(
                "INSERT INTO image_tasks("
                "task_id, scope_key, umo, user_id, prompt, mode, aspect_ratio, resolution, "
                "preset_name, source_message_id, conversation_id, request_fingerprint, "
                "metadata_json, state, charge_amount, charge_state, refund_state, "
                "created_at_ms, updated_at_ms, expires_at_ms) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'preparing', ?, ?, 'none', ?, ?, ?)",
                (
                    task_id,
                    spec.scope_key,
                    spec.umo,
                    spec.user_id,
                    spec.prompt,
                    spec.mode,
                    spec.aspect_ratio,
                    spec.resolution,
                    spec.preset_name,
                    spec.source_message_id,
                    spec.conversation_id,
                    spec.request_fingerprint,
                    json.dumps(
                        dict(spec.metadata),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    spec.charge_amount,
                    "none" if spec.charge_amount <= 0 else "pending",
                    now,
                    now,
                    now + self.terminal_ttl_seconds * 1000,
                ),
            )
            row = conn.execute(
                "SELECT * FROM image_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            conn.commit()
        if row is None:
            raise BackgroundTaskError("Task insertion did not return a row")
        return self._row_to_snapshot(row), True

    def _recover_abandoned_sync(self) -> list[TaskSnapshot]:
        """Conservatively terminalize tasks left active by another process run.

        Returns:
            Recovered interrupted or unknown task snapshots.
        """

        now = self.now_ms()
        recovered_ids: list[str] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM image_tasks WHERE expires_at_ms<? "
                "AND notification_state NOT IN ('pending','sending') AND ("
                "state IN ('completed','unknown') OR ("
                "state IN ('failed','cancelled','interrupted') AND ("
                "charge_state IN ('none','rejected') OR refund_state='refunded'))) ",
                (now,),
            )
            rows = conn.execute(
                "SELECT task_id, state, delivery_state FROM image_tasks WHERE state IN "
                "('preparing','queued','running','generated','sending')"
            ).fetchall()
            for row in rows:
                task_id = str(row["task_id"])
                recovered_ids.append(task_id)
                if row["state"] == "sending" or row["delivery_state"] == "attempting":
                    state = "unknown"
                    delivery_state = "unknown"
                    reason = "restart_during_send"
                    message = (
                        "Process restarted during image delivery; delivery is unknown."
                    )
                    conn.execute(
                        "UPDATE delivery_attempts SET state='unknown', finished_at_ms=? "
                        "WHERE task_id=? AND state='attempting'",
                        (now, task_id),
                    )
                else:
                    state = "interrupted"
                    delivery_state = str(row["delivery_state"] or "not_started")
                    reason = "restart_interrupted"
                    message = "Process restarted before the task reached a safe terminal state."
                conn.execute(
                    "UPDATE image_tasks SET state=?, delivery_state=?, terminal_reason=?, "
                    "error_code=?, error_message=?, notification_state='pending', "
                    "cancel_requested=1, finished_at_ms=?, updated_at_ms=?, "
                    "expires_at_ms=?, revision=revision+1 "
                    "WHERE task_id=?",
                    (
                        state,
                        delivery_state,
                        reason,
                        reason,
                        message,
                        now,
                        now,
                        now + self.terminal_ttl_seconds * 1000,
                        task_id,
                    ),
                )
            conn.execute(
                "UPDATE image_tasks SET notification_state='unknown', "
                "notification_error='Process restarted during notification.' "
                "WHERE notification_state='sending'"
            )
            conn.commit()
            snapshots = []
            for task_id in recovered_ids:
                row = conn.execute(
                    "SELECT * FROM image_tasks WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                if row is not None:
                    snapshots.append(self._row_to_snapshot(row))
        return snapshots

    def _begin_delivery_sync(
        self,
        task_id: str,
        attempt_id: str,
        now: int,
    ) -> TaskSnapshot | None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT state, cancel_requested FROM image_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if row is None or row["state"] != "generated" or row["cancel_requested"]:
                conn.rollback()
                return None
            conn.execute(
                "INSERT INTO delivery_attempts(attempt_id, task_id, state, started_at_ms) "
                "VALUES(?, ?, 'attempting', ?)",
                (attempt_id, task_id, now),
            )
            conn.execute(
                "UPDATE image_tasks SET state='sending', delivery_state='attempting', "
                "send_attempt_id=?, updated_at_ms=?, revision=revision+1 WHERE task_id=?",
                (attempt_id, now, task_id),
            )
            updated = conn.execute(
                "SELECT * FROM image_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            conn.commit()
        return self._row_to_snapshot(updated) if updated is not None else None

    def _confirm_delivery_sync(
        self,
        task_id: str,
        attempt_id: str,
        now: int,
    ) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            attempt = conn.execute(
                "SELECT state FROM delivery_attempts WHERE attempt_id=? AND task_id=?",
                (attempt_id, task_id),
            ).fetchone()
            if attempt is None or attempt["state"] != "attempting":
                conn.rollback()
                return False
            changed = conn.execute(
                "UPDATE image_tasks SET state='completed', delivery_state='confirmed', "
                "terminal_reason='', error_code='', error_message='', "
                "notification_state='pending', finished_at_ms=?, updated_at_ms=?, "
                "expires_at_ms=?, revision=revision+1 WHERE task_id=? AND state='sending' "
                "AND send_attempt_id=?",
                (
                    now,
                    now,
                    now + self.terminal_ttl_seconds * 1000,
                    task_id,
                    attempt_id,
                ),
            ).rowcount
            if changed != 1:
                conn.rollback()
                return False
            conn.execute(
                "UPDATE delivery_attempts SET state='confirmed', finished_at_ms=? "
                "WHERE attempt_id=?",
                (now, attempt_id),
            )
            conn.commit()
        return True

    def _finish_delivery_sync(
        self,
        task_id: str,
        state: str,
        delivery_state: str,
        error_code: str,
        error_message: str,
        now: int,
        notify: bool,
    ) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT state, send_attempt_id FROM image_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if row is None or row["state"] != "sending":
                conn.rollback()
                return False
            attempt_id = str(row["send_attempt_id"] or "")
            changed = conn.execute(
                "UPDATE image_tasks SET state=?, delivery_state=?, terminal_reason=?, "
                "error_code=?, error_message=?, notification_state=?, finished_at_ms=?, "
                "updated_at_ms=?, expires_at_ms=?, revision=revision+1 "
                "WHERE task_id=? AND state='sending'",
                (
                    state,
                    delivery_state,
                    error_code,
                    error_code,
                    error_message,
                    "pending" if notify else "not_needed",
                    now,
                    now,
                    now + self.terminal_ttl_seconds * 1000,
                    task_id,
                ),
            ).rowcount
            if attempt_id:
                conn.execute(
                    "UPDATE delivery_attempts SET state=?, error_digest=?, finished_at_ms=? "
                    "WHERE attempt_id=? AND state='attempting'",
                    (
                        delivery_state,
                        hashlib.sha256(error_message.encode("utf-8")).hexdigest(),
                        now,
                        attempt_id,
                    ),
                )
            conn.commit()
        return changed == 1

    def _transition_sync(
        self,
        task_id: str,
        expected: set[str],
        state: str,
        updates: dict[str, Any],
        now: int,
    ) -> TaskSnapshot | None:
        allowed = self._validated_updates(updates)
        allowed["state"] = state
        allowed["updated_at_ms"] = now
        assignments = ", ".join(f"{column}=?" for column in allowed)
        placeholders = ",".join("?" for _ in expected)
        values = [*allowed.values(), task_id, *sorted(expected)]
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                f"UPDATE image_tasks SET {assignments}, revision=revision+1 "
                f"WHERE task_id=? AND state IN ({placeholders})",
                values,
            ).rowcount
            if changed != 1:
                conn.rollback()
                return None
            row = conn.execute(
                "SELECT * FROM image_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            conn.commit()
        return self._row_to_snapshot(row) if row is not None else None

    def _update_columns_sync(
        self,
        task_id: str,
        updates: dict[str, Any],
        now: int,
    ) -> bool:
        allowed = self._validated_updates(updates)
        if not allowed:
            return False
        allowed["updated_at_ms"] = now
        assignments = ", ".join(f"{column}=?" for column in allowed)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                f"UPDATE image_tasks SET {assignments}, revision=revision+1 WHERE task_id=?",
                [*allowed.values(), task_id],
            ).rowcount
            conn.commit()
        return changed == 1

    @staticmethod
    def _validated_updates(updates: Mapping[str, Any]) -> dict[str, Any]:
        allowed_columns = {
            "metadata_json",
            "input_manifest_json",
            "delivery_state",
            "charge_state",
            "refund_state",
            "cancel_requested",
            "image_count",
            "elapsed_seconds",
            "send_attempt_id",
            "terminal_reason",
            "error_code",
            "error_message",
            "notification_state",
            "notification_error",
            "queued_at_ms",
            "started_at_ms",
            "generated_at_ms",
            "finished_at_ms",
            "expires_at_ms",
        }
        unknown = set(updates) - allowed_columns
        if unknown:
            raise BackgroundTaskError(
                f"Unsupported task update columns: {sorted(unknown)}"
            )
        return dict(updates)

    def _get_sync(self, task_id: str) -> TaskSnapshot | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM image_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
        return self._row_to_snapshot(row) if row is not None else None

    def _list_for_scope_sync(
        self,
        scope_key: str,
        limit: int,
        active_only: bool,
    ) -> list[TaskSnapshot]:
        query = "SELECT * FROM image_tasks WHERE scope_key=?"
        params: list[Any] = [scope_key]
        if active_only:
            placeholders = ",".join("?" for _ in ACTIVE_STATES)
            query += f" AND state IN ({placeholders})"
            params.extend(sorted(ACTIVE_STATES))
        query += " ORDER BY created_at_ms DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def _list_matching_active_sync(
        self,
        scope_key: str | None,
        umo: str | None,
        user_id: str | None,
        allow_all: bool = False,
    ) -> list[TaskSnapshot]:
        clauses = []
        params: list[Any] = []
        if scope_key:
            clauses.append("scope_key=?")
            params.append(scope_key)
        if umo:
            clauses.append("umo=?")
            params.append(umo)
        if user_id:
            clauses.append("user_id=?")
            params.append(user_id)
        if not clauses and not allow_all:
            return []
        placeholders = ",".join("?" for _ in ACTIVE_STATES)
        where = " AND ".join(clauses) if clauses else "1=1"
        query = (
            f"SELECT * FROM image_tasks WHERE {where} "
            f"AND state IN ({placeholders}) ORDER BY created_at_ms"
        )
        params.extend(sorted(ACTIVE_STATES))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def _list_pending_notifications_sync(self) -> list[TaskSnapshot]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM image_tasks WHERE notification_state='pending' "
                "AND state IN ('completed','failed','interrupted','unknown') "
                "ORDER BY finished_at_ms"
            ).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def _list_refundable_tasks_sync(self) -> list[TaskSnapshot]:
        """List terminal tasks whose charge settlement is incomplete.

        Returns:
            Failed, cancelled, or interrupted tasks needing charge reconciliation
            or a refund retry. Completed and delivery-unknown tasks are
            deliberately excluded.
        """

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM image_tasks WHERE state IN "
                "('failed','cancelled','interrupted') "
                "AND charge_state IN ('pending','charged') "
                "AND refund_state IN ('none','pending','failed') "
                "ORDER BY finished_at_ms, created_at_ms"
            ).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def _spool_inputs_sync(
        self,
        task_id: str,
        images: list[tuple[bytes, str]],
    ) -> list[dict[str, Any]]:
        """Atomically write immutable input bytes and produce a hash manifest.

        Args:
            task_id: Task owning the private input directory.
            images: Normalized bytes and MIME type pairs.

        Returns:
            JSON-compatible relative path, MIME, size, and SHA-256 entries.
        """

        total = sum(len(data) for data, _ in images)
        if total > INPUT_TASK_LIMIT:
            raise BackgroundTaskError(
                f"Task input total exceeds {INPUT_TASK_LIMIT} bytes"
            )
        if any(len(data) > INPUT_FILE_LIMIT for data, _ in images):
            raise BackgroundTaskError(
                f"An input image exceeds {INPUT_FILE_LIMIT} bytes"
            )
        if not images:
            return []

        input_dir = self.base_dir / task_id / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(input_dir.parent, 0o700)
            os.chmod(input_dir, 0o700)
        except OSError:
            pass

        manifest: list[dict[str, Any]] = []
        try:
            for index, (data, mime_type) in enumerate(images, start=1):
                path = input_dir / f"{index:04d}.bin"
                part = input_dir / f".{index:04d}.{uuid.uuid4().hex}.part"
                with part.open("wb") as file:
                    file.write(data)
                    file.flush()
                    os.fsync(file.fileno())
                try:
                    os.chmod(part, 0o600)
                except OSError:
                    pass
                part.replace(path)
                manifest.append(
                    {
                        "relative_path": str(path.relative_to(self.base_dir)),
                        "mime_type": mime_type,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
        except BaseException:
            shutil.rmtree(self.base_dir / task_id, ignore_errors=True)
            raise
        return manifest

    def _read_spooled_inputs_sync(
        self,
        task_id: str,
    ) -> list[tuple[bytes, str]]:
        """Read task inputs after validating path, size, and SHA-256.

        Args:
            task_id: Task whose manifest and private files are read.

        Returns:
            Original bytes and MIME type pairs in manifest order.

        Raises:
            BackgroundTaskError: A path escaped, file changed, or limit failed.
        """

        with self._connect() as conn:
            row = conn.execute(
                "SELECT input_manifest_json FROM image_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise BackgroundTaskError("Task input manifest is missing")
        manifest = json.loads(str(row["input_manifest_json"] or "[]"))
        base = self.base_dir.resolve(strict=False)
        results: list[tuple[bytes, str]] = []
        total = 0
        for item in manifest:
            path = (self.base_dir / str(item["relative_path"])).resolve(strict=False)
            try:
                path.relative_to(base)
            except ValueError as exc:
                raise BackgroundTaskError(
                    "Task input path escaped the plugin data directory"
                ) from exc
            data = path.read_bytes()
            expected_size = int(item.get("size") or 0)
            expected_hash = str(item.get("sha256") or "")
            if expected_size != len(data):
                raise BackgroundTaskError("Task input size no longer matches manifest")
            if hashlib.sha256(data).hexdigest() != expected_hash:
                raise BackgroundTaskError(
                    "Task input SHA-256 no longer matches manifest"
                )
            total += len(data)
            if len(data) > INPUT_FILE_LIMIT or total > INPUT_TASK_LIMIT:
                raise BackgroundTaskError("Task input exceeded configured byte limits")
            results.append(
                (data, str(item.get("mime_type") or "application/octet-stream"))
            )
        return results

    async def _cleanup_task_files(self, task_id: str) -> None:
        shutil.rmtree(self.base_dir / task_id, ignore_errors=True)

    @staticmethod
    def _normalize_inputs(
        images: Sequence[bytes | tuple[bytes, str]],
    ) -> list[tuple[bytes, str]]:
        normalized: list[tuple[bytes, str]] = []
        for item in images:
            if isinstance(item, bytes):
                data = item
                mime_type = "application/octet-stream"
            elif (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], bytes)
            ):
                data = item[0]
                mime_type = str(item[1] or "application/octet-stream")
            else:
                raise BackgroundTaskError(
                    "Input images must be bytes or (bytes, mime_type) tuples"
                )
            if not data:
                raise BackgroundTaskError("Empty input images are not allowed")
            normalized.append((data, mime_type))
        return normalized

    @staticmethod
    def _validate_spec(spec: TaskSpec) -> TaskSpec:
        if not isinstance(spec, TaskSpec):
            raise TypeError("spec must be a TaskSpec")
        if not spec.scope_key or not spec.umo or not spec.user_id:
            raise BackgroundTaskError("scope_key, umo, and user_id are required")
        if not spec.prompt or len(spec.prompt) > PROMPT_LIMIT:
            raise BackgroundTaskError(
                f"prompt must contain 1-{PROMPT_LIMIT} characters"
            )
        if int(spec.charge_amount) < 0:
            raise BackgroundTaskError("charge_amount cannot be negative")
        try:
            json.dumps(dict(spec.metadata), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise BackgroundTaskError(
                "Task metadata must be JSON serializable"
            ) from exc
        return TaskSpec(
            scope_key=str(spec.scope_key),
            umo=str(spec.umo),
            user_id=str(spec.user_id),
            prompt=str(spec.prompt),
            mode=str(spec.mode or "txt2img"),
            aspect_ratio=str(spec.aspect_ratio or "1:1"),
            resolution=str(spec.resolution or "2K"),
            preset_name=str(spec.preset_name or ""),
            source_message_id=str(spec.source_message_id or ""),
            conversation_id=str(spec.conversation_id or ""),
            request_fingerprint=str(spec.request_fingerprint or ""),
            charge_amount=int(spec.charge_amount),
            metadata=dict(spec.metadata),
        )

    @staticmethod
    def _new_task_id() -> str:
        return f"img_{int(time.time() * 1000):x}_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _sanitize_text(value: BaseException | str, limit: int) -> str:
        text = str(value or "")
        text = re.sub(
            r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
            r"\1[REDACTED]",
            text,
        )
        text = re.sub(
            r"(?i)(api[\s_-]?key|access[\s_-]?token|secret)(\s*[:=]\s*)[^\s,;]+",
            r"\1\2[REDACTED]",
            text,
        )
        return text[: max(0, int(limit))]

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> TaskSnapshot:
        metadata = json.loads(str(row["metadata_json"] or "{}"))
        return TaskSnapshot(
            task_id=str(row["task_id"]),
            scope_key=str(row["scope_key"]),
            umo=str(row["umo"]),
            user_id=str(row["user_id"]),
            prompt=str(row["prompt"]),
            mode=str(row["mode"]),
            aspect_ratio=str(row["aspect_ratio"]),
            resolution=str(row["resolution"]),
            preset_name=str(row["preset_name"]),
            source_message_id=str(row["source_message_id"]),
            conversation_id=str(row["conversation_id"]),
            request_fingerprint=str(row["request_fingerprint"]),
            metadata=dict(metadata),
            state=str(row["state"]),
            delivery_state=str(row["delivery_state"]),
            charge_amount=int(row["charge_amount"]),
            charge_state=str(row["charge_state"]),
            refund_state=str(row["refund_state"]),
            cancel_requested=bool(row["cancel_requested"]),
            image_count=int(row["image_count"]),
            elapsed_seconds=float(row["elapsed_seconds"]),
            send_attempt_id=str(row["send_attempt_id"]),
            terminal_reason=str(row["terminal_reason"]),
            error_code=str(row["error_code"]),
            error_message=str(row["error_message"]),
            notification_state=str(row["notification_state"]),
            notification_error=str(row["notification_error"]),
            created_at_ms=int(row["created_at_ms"]),
            queued_at_ms=int(row["queued_at_ms"]),
            started_at_ms=int(row["started_at_ms"]),
            generated_at_ms=int(row["generated_at_ms"]),
            finished_at_ms=int(row["finished_at_ms"]),
            updated_at_ms=int(row["updated_at_ms"]),
            revision=int(row["revision"]),
        )

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[Any]) -> None:
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        except BaseException:
            logger.exception("Failed to inspect a managed background task")
            return
        if error is not None:
            logger.error("Managed background image task failed: %s", error)


__all__ = [
    "AdmissionResult",
    "BackgroundImageTaskManager",
    "BackgroundTaskError",
    "ChargeRejectedError",
    "DeliveryRejectedError",
    "DeliveryUnknownError",
    "GeneratedImageResult",
    "PreparedDelivery",
    "QueueFullError",
    "TaskSnapshot",
    "TaskSpec",
    "build_scope_key",
]
