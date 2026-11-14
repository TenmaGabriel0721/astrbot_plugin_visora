from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from astrbot_plugin_visora.background_tasks import (
    BackgroundImageTaskManager,
    DeliveryRejectedError,
    DeliveryUnknownError,
    GeneratedImageResult,
    PreparedDelivery,
    QueueFullError,
    TaskSpec,
    build_scope_key,
)


async def wait_for_state(
    manager: BackgroundImageTaskManager,
    task_id: str,
    expected: set[str],
    *,
    timeout: float = 3.0,
):
    async def poll():
        while True:
            snapshot = await manager.get(task_id)
            if snapshot is not None and snapshot.state in expected:
                return snapshot
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(poll(), timeout=timeout)


async def wait_for_refund(
    manager: BackgroundImageTaskManager,
    task_id: str,
    *,
    timeout: float = 3.0,
):
    async def poll():
        while True:
            snapshot = await manager.get(task_id)
            if snapshot is not None and snapshot.refund_state in {"refunded", "failed"}:
                return snapshot
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(poll(), timeout=timeout)


def make_spec(
    suffix: str,
    *,
    charge_amount: int = 0,
    fingerprint: str = "",
) -> TaskSpec:
    umo = "platform:GroupMessage:100"
    user_id = "200"
    return TaskSpec(
        scope_key=build_scope_key(umo, user_id, "cid-1"),
        umo=umo,
        user_id=user_id,
        prompt=f"draw {suffix}",
        source_message_id=f"message-{suffix}",
        conversation_id="cid-1",
        request_fingerprint=fingerprint,
        charge_amount=charge_amount,
        metadata={"suffix": suffix},
    )


@pytest.mark.asyncio
async def test_generation_metadata_is_persisted_for_delivery_and_task_queries(tmp_path):
    result_metadata = {
        "model": "image-model",
        "requested_resolution": "4K",
        "effective_resolution": "2K",
        "resolution_downgraded": True,
        "dimensions": [[2048, 2048]],
    }

    async def provider(snapshot, images):
        return GeneratedImageResult([b"result"], result_metadata=result_metadata)

    async def deliver(snapshot, generated):
        assert snapshot.metadata["generation_result"] == result_metadata
        assert snapshot.metadata["suffix"] == "metadata"

    manager = BackgroundImageTaskManager(
        tmp_path, provider_runner=provider, delivery_runner=deliver, max_running=1
    )
    try:
        await manager.start()
        admission = await manager.accept(make_spec("metadata"), [])
        completed = await wait_for_state(manager, admission.task_id, {"completed"})
        assert completed.metadata["generation_result"] == result_metadata
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_concurrent_start_waits_until_manager_is_accepting(tmp_path: Path):
    refund_started = asyncio.Event()
    release_refund = asyncio.Event()

    async def provider(snapshot, images):
        del snapshot, images
        return GeneratedImageResult([b"generated"])

    async def deliver(snapshot, generated):
        del snapshot, generated

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        max_running=1,
        max_queued=1,
    )
    pending_refund = type(
        "PendingRefund",
        (),
        {
            "task_id": "task-refund",
            "state": "failed",
            "charge_state": "charged",
        },
    )()

    def list_refundable_tasks():
        return [pending_refund]

    manager._list_refundable_tasks_sync = list_refundable_tasks

    async def block_refund(task_id: str) -> bool:
        assert task_id == "task-refund"
        refund_started.set()
        await release_refund.wait()
        return True

    manager._refund_once = block_refund

    first_start = asyncio.create_task(manager.start())
    await asyncio.wait_for(refund_started.wait(), timeout=1)
    second_start = asyncio.create_task(manager.start())
    await asyncio.sleep(0)

    assert not manager.accepting
    assert not second_start.done()

    release_refund.set()
    await asyncio.gather(first_start, second_start)

    assert manager.accepting
    assert len(manager._workers) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_successful_task_spools_validated_inputs_and_completes(tmp_path: Path):
    provider_inputs = []
    delivered = []

    async def provider(snapshot, images):
        provider_inputs.append((snapshot, images))
        return GeneratedImageResult([b"generated"], 1.25)

    async def deliver(snapshot, generated):
        delivered.append((snapshot, generated))

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        max_running=1,
        max_queued=2,
    )
    await manager.start()
    admission = await manager.accept(
        make_spec("success"),
        [(b"input-image", "image/png")],
    )
    completed = await wait_for_state(manager, admission.task_id, {"completed"})

    assert completed.delivery_state == "confirmed"
    assert completed.image_count == 1
    assert completed.elapsed_seconds == 1.25
    assert provider_inputs[0][1] == [(b"input-image", "image/png")]
    assert delivered[0][0].state == "sending"
    assert delivered[0][1].images == [b"generated"]
    assert not (manager.base_dir / admission.task_id).exists()
    await manager.close()


@pytest.mark.asyncio
async def test_success_notification_starts_only_after_delivery_is_confirmed(
    tmp_path: Path,
):
    timeline = []
    notification_started = asyncio.Event()

    async def provider(snapshot, images):
        del snapshot, images
        return GeneratedImageResult([b"generated"], 0.5)

    async def deliver(snapshot, generated):
        del generated
        timeline.append(("image", snapshot.state, snapshot.delivery_state))

    async def notify(snapshot, text):
        persisted = await manager.get(snapshot.task_id)
        timeline.append(
            (
                "reply",
                persisted.state,
                persisted.delivery_state,
                text,
            )
        )
        notification_started.set()

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        notify_runner=notify,
        max_running=1,
    )
    await manager.start()
    admission = await manager.accept(make_spec("completion-reply"))
    await asyncio.wait_for(notification_started.wait(), timeout=2)

    async def notification_finished():
        while True:
            snapshot = await manager.get(admission.task_id)
            if snapshot is not None and snapshot.notification_state == "sent":
                return snapshot
            await asyncio.sleep(0.01)

    completed = await asyncio.wait_for(notification_finished(), timeout=2)
    assert completed.state == "completed"
    assert completed.delivery_state == "confirmed"
    assert timeline == [
        ("image", "sending", "attempting"),
        ("reply", "completed", "confirmed", "照片拍好啦，我已经发出来了。"),
    ]
    await manager.close()


@pytest.mark.asyncio
async def test_tampered_spooled_input_fails_and_refunds_once(tmp_path: Path):
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    refunds = []

    async def provider(snapshot, images):
        if snapshot.metadata["suffix"] == "blocker":
            first_started.set()
            await release_first.wait()
        return GeneratedImageResult([b"generated"])

    async def deliver(snapshot, generated):
        return None

    async def charge(task_id, user_id, amount):
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
        max_running=1,
        max_queued=2,
    )
    await manager.start()
    first = await manager.accept(make_spec("blocker"))
    await asyncio.wait_for(first_started.wait(), timeout=2)
    second = await manager.accept(
        make_spec("tampered", charge_amount=7),
        [(b"original", "image/jpeg")],
    )
    input_path = manager.base_dir / second.task_id / "inputs" / "0001.bin"
    input_path.write_bytes(b"modified")
    release_first.set()

    failed = await wait_for_state(manager, second.task_id, {"failed"})
    failed = await wait_for_refund(manager, second.task_id)
    assert failed.error_code == "provider_failed"
    assert failed.refund_state == "refunded"
    assert refunds == [second.task_id]
    await wait_for_state(manager, first.task_id, {"completed"})
    await manager.close()


@pytest.mark.asyncio
async def test_bounded_queue_rejects_third_active_task(tmp_path: Path):
    started = asyncio.Event()
    release = asyncio.Event()

    async def provider(snapshot, images):
        started.set()
        await release.wait()
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        return None

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        max_running=1,
        max_queued=1,
    )
    await manager.start()
    first = await manager.accept(make_spec("one"))
    await asyncio.wait_for(started.wait(), timeout=2)
    second = await manager.accept(make_spec("two"))

    with pytest.raises(QueueFullError):
        await manager.accept(make_spec("three"))

    release.set()
    await wait_for_state(manager, first.task_id, {"completed"})
    await wait_for_state(manager, second.task_id, {"completed"})
    await manager.close()


@pytest.mark.asyncio
async def test_delivery_unknown_is_terminal_not_retried_or_refunded(tmp_path: Path):
    deliveries = []
    refunds = []

    async def provider(snapshot, images):
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        deliveries.append(snapshot.task_id)
        raise DeliveryUnknownError("transport timeout")

    async def charge(task_id, user_id, amount):
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
    )
    await manager.start()
    result = await manager.accept(make_spec("unknown", charge_amount=5))
    unknown = await wait_for_state(manager, result.task_id, {"unknown"})

    assert unknown.delivery_state == "unknown"
    assert deliveries == [result.task_id]
    assert refunds == []
    assert unknown.refund_state == "none"
    await manager.close()


@pytest.mark.asyncio
async def test_delivery_rejection_fails_and_refunds_once(tmp_path: Path):
    refunds = []

    async def provider(snapshot, images):
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        raise DeliveryRejectedError("platform rejected payload")

    async def charge(task_id, user_id, amount):
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
    )
    await manager.start()
    result = await manager.accept(make_spec("rejected", charge_amount=3))
    failed = await wait_for_state(manager, result.task_id, {"failed"})
    failed = await wait_for_refund(manager, result.task_id)

    assert failed.delivery_state == "rejected"
    assert failed.refund_state == "refunded"
    assert refunds == [result.task_id]
    await manager.close()


@pytest.mark.asyncio
async def test_provider_failure_refunds_and_sends_deterministic_notice(tmp_path: Path):
    refunds = []
    notices = []

    async def provider(snapshot, images):
        raise RuntimeError("provider exploded")

    async def deliver(snapshot, generated):
        raise AssertionError("delivery must not run")

    async def charge(task_id, user_id, amount):
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    async def notify(snapshot, text):
        notices.append((snapshot.state, text))

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
        notify_runner=notify,
    )
    await manager.start()
    result = await manager.accept(make_spec("provider-fail", charge_amount=2))
    await wait_for_state(manager, result.task_id, {"failed"})

    async def notification_finished():
        while True:
            snapshot = await manager.get(result.task_id)
            if snapshot is not None and snapshot.notification_state == "sent":
                return snapshot
            await asyncio.sleep(0.01)

    failed = await asyncio.wait_for(notification_finished(), timeout=2)
    assert failed.refund_state == "refunded"
    assert failed.error_message == "provider exploded"
    assert refunds == [result.task_id]
    assert notices == [("failed", "刚才那次生图没有成功：provider exploded")]
    await manager.close()


@pytest.mark.asyncio
async def test_charge_success_then_exception_is_reconciled_and_refunded(tmp_path: Path):
    charge_calls = []
    refunds = []

    async def provider(snapshot, images):
        raise AssertionError("provider must not run after aborted admission")

    async def deliver(snapshot, generated):
        raise AssertionError("delivery must not run")

    async def charge(task_id, user_id, amount):
        charge_calls.append(task_id)
        if len(charge_calls) == 1:
            raise RuntimeError("wrapper failed after the durable charge")
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
    )
    await manager.start()

    with pytest.raises(RuntimeError, match="wrapper failed"):
        await manager.accept(make_spec("charge-crash", charge_amount=6))

    assert len(charge_calls) == 2
    task_id = charge_calls[0]
    failed = await manager.get(task_id)
    assert failed is not None
    assert failed.state == "failed"
    assert failed.charge_state == "charged"
    assert failed.refund_state == "refunded"
    assert refunds == [task_id]
    await manager.close()


@pytest.mark.asyncio
async def test_start_retries_terminal_pending_charge_without_orphan_workers(
    tmp_path: Path,
):
    charge_calls = []
    refunds = []

    async def unused_provider(snapshot, images):
        raise AssertionError("terminal recovery must not run the provider")

    async def unused_delivery(snapshot, generated):
        raise AssertionError("terminal recovery must not deliver")

    seed = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=unused_provider,
        delivery_runner=unused_delivery,
    )
    seed._prepare_storage_sync()
    seed._reserve_task_sync(
        "task-terminal-pending",
        make_spec("terminal-pending", charge_amount=6),
    )
    now = seed.now_ms()
    seed._transition_sync(
        "task-terminal-pending",
        {"preparing"},
        "failed",
        {
            "terminal_reason": "accept_aborted",
            "error_code": "accept_aborted",
            "error_message": "Task acceptance did not complete.",
            "finished_at_ms": now,
        },
        now,
    )

    async def charge(task_id, user_id, amount):
        del user_id, amount
        charge_calls.append(task_id)
        if len(charge_calls) == 1:
            raise RuntimeError("charge ledger temporarily unavailable")
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=unused_provider,
        delivery_runner=unused_delivery,
        charge_runner=charge,
        refund_runner=refund,
        max_running=1,
    )

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        await manager.start()

    assert not manager.accepting
    assert not manager._started
    assert manager._workers == []

    await manager.start()
    settled = await manager.get("task-terminal-pending")

    assert settled is not None
    assert settled.state == "failed"
    assert settled.charge_state == "charged"
    assert settled.refund_state == "refunded"
    assert charge_calls == ["task-terminal-pending", "task-terminal-pending"]
    assert refunds == ["task-terminal-pending"]
    await manager.close()


@pytest.mark.asyncio
async def test_failed_refund_is_retried_once_after_restart(tmp_path: Path):
    provider_calls = []
    refund_attempts = []
    successful_refunds = []

    async def provider(snapshot, images):
        provider_calls.append(snapshot.task_id)
        raise RuntimeError("provider exploded")

    async def deliver(snapshot, generated):
        raise AssertionError("delivery must not run")

    async def charge(task_id, user_id, amount):
        return True

    async def failing_refund(task_id):
        refund_attempts.append(task_id)
        raise RuntimeError("refund service unavailable")

    first = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=failing_refund,
    )
    await first.start()
    admission = await first.accept(make_spec("refund-restart", charge_amount=5))
    failed = await wait_for_refund(first, admission.task_id)

    assert failed.state == "failed"
    assert failed.charge_state == "charged"
    assert failed.refund_state == "failed"
    assert refund_attempts == [admission.task_id]
    await first.close()
    with first._connect() as conn:
        conn.execute(
            "UPDATE image_tasks SET expires_at_ms=? WHERE task_id=?",
            (first.now_ms() - 1, admission.task_id),
        )

    async def successful_refund(task_id):
        refund_attempts.append(task_id)
        successful_refunds.append(task_id)
        return True

    second = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=successful_refund,
    )
    await second.start()
    refunded = await second.get(admission.task_id)

    assert refunded is not None
    assert refunded.state == "failed"
    assert refunded.refund_state == "refunded"
    assert provider_calls == [admission.task_id]
    assert refund_attempts == [admission.task_id, admission.task_id]
    assert successful_refunds == [admission.task_id]
    await second.close()

    async def unexpected_refund(task_id):
        refund_attempts.append(task_id)
        raise AssertionError("a completed refund must not be retried")

    third = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=unexpected_refund,
    )
    await third.start()
    assert (await third.get(admission.task_id)).refund_state == "refunded"
    assert provider_calls == [admission.task_id]
    assert refund_attempts == [admission.task_id, admission.task_id]
    assert successful_refunds == [admission.task_id]
    await third.close()


@pytest.mark.asyncio
async def test_cancel_scope_by_umo_and_user_cancels_running_task(tmp_path: Path):
    started = asyncio.Event()
    refunds = []

    async def provider(snapshot, images):
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def deliver(snapshot, generated):
        raise AssertionError("delivery must not run")

    async def charge(task_id, user_id, amount):
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
        max_running=1,
    )
    await manager.start()
    result = await manager.accept(make_spec("cancel", charge_amount=4))
    await asyncio.wait_for(started.wait(), timeout=2)
    cancelled_count = await manager.cancel_scope(
        umo="platform:GroupMessage:100",
        user_id="200",
        reason="new_session",
    )
    cancelled = await wait_for_state(manager, result.task_id, {"cancelled"})
    cancelled = await wait_for_refund(manager, result.task_id)

    assert cancelled_count == 1
    assert cancelled.terminal_reason == "new_session"
    assert cancelled.refund_state == "refunded"
    assert refunds == [result.task_id]
    await manager.close()


@pytest.mark.asyncio
async def test_cancel_during_send_becomes_unknown_without_refund(tmp_path: Path):
    sending = asyncio.Event()
    refunds = []
    delivery_calls = []

    async def provider(snapshot, images):
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        delivery_calls.append(snapshot.task_id)
        sending.set()
        await asyncio.Event().wait()

    async def charge(task_id, user_id, amount):
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
    )
    await manager.start()
    result = await manager.accept(make_spec("sending-cancel", charge_amount=4))
    await asyncio.wait_for(sending.wait(), timeout=2)
    assert await manager.cancel_scope(scope_key=result.task.scope_key) == 1
    unknown = await wait_for_state(manager, result.task_id, {"unknown"})

    assert unknown.delivery_state == "unknown"
    assert delivery_calls == [result.task_id]
    assert refunds == []
    await manager.close()


@pytest.mark.asyncio
async def test_cancel_during_delivery_preparation_refunds_without_unknown(
    tmp_path: Path,
):
    preparation_started = asyncio.Event()
    delivery_calls = []
    refunds = []

    async def provider(snapshot, images):
        del snapshot, images
        return GeneratedImageResult([b"image"])

    async def prepare(snapshot, generated):
        del snapshot, generated
        preparation_started.set()
        await asyncio.Event().wait()
        return PreparedDelivery(payload="unreachable")

    async def deliver(snapshot, payload):
        delivery_calls.append((snapshot.task_id, payload))

    async def charge(task_id, user_id, amount):
        del task_id, user_id, amount
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_prepare_runner=prepare,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
    )
    await manager.start()
    result = await manager.accept(make_spec("prepare-cancel", charge_amount=4))
    await asyncio.wait_for(preparation_started.wait(), timeout=2)

    assert await manager.cancel_scope(scope_key=result.task.scope_key) == 1
    cancelled = await wait_for_refund(manager, result.task_id)

    assert cancelled.state == "cancelled"
    assert cancelled.delivery_state == "not_started"
    assert cancelled.refund_state == "refunded"
    assert delivery_calls == []
    assert refunds == [result.task_id]
    await manager.close()


@pytest.mark.asyncio
async def test_cancel_uses_current_state_and_never_refunds_a_successful_send(
    tmp_path: Path,
):
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_sending = asyncio.Event()
    release_second_send = asyncio.Event()
    delivered = []
    refunds = []

    async def provider(snapshot, images):
        del images
        if snapshot.metadata["suffix"] == "cancel-race-blocker":
            first_started.set()
            await release_first.wait()
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        del generated
        if snapshot.metadata["suffix"] == "cancel-race-target":
            second_sending.set()
            await release_second_send.wait()
        delivered.append(snapshot.task_id)

    async def charge(task_id, user_id, amount):
        del task_id, user_id, amount
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
        max_running=1,
        max_queued=2,
    )
    await manager.start()
    blocker = await manager.accept(make_spec("cancel-race-blocker"))
    await asyncio.wait_for(first_started.wait(), timeout=2)
    target_scope = build_scope_key(
        "platform:GroupMessage:100",
        "200",
        "cid-cancel-race",
    )
    target = await manager.accept(
        replace(
            make_spec("cancel-race-target", charge_amount=5),
            scope_key=target_scope,
            conversation_id="cid-cancel-race",
        )
    )

    original_update_columns = manager._update_columns

    async def delay_cancel_update(task_id: str, **updates):
        if task_id == target.task_id and updates.get("cancel_requested") == 1:
            release_first.set()
            await asyncio.wait_for(second_sending.wait(), timeout=2)
        return await original_update_columns(task_id, **updates)

    manager._update_columns = delay_cancel_update
    cancelled = await manager.cancel_scope(
        scope_key=target_scope,
        reason="race-test",
    )
    unknown = await wait_for_state(manager, target.task_id, {"unknown"})

    assert cancelled == 1
    assert unknown.delivery_state == "unknown"
    assert unknown.refund_state == "none"
    assert target.task_id not in delivered
    assert target.task_id not in refunds
    assert (await manager.get(blocker.task_id)).state == "completed"
    await manager.close()


@pytest.mark.asyncio
async def test_cancel_during_failure_settlement_keeps_worker_and_refunds(
    tmp_path: Path,
):
    failure_settlement_started = asyncio.Event()
    release_failure_settlement = asyncio.Event()
    refunds = []
    delivered = []

    async def provider(snapshot, images):
        del images
        if snapshot.metadata["suffix"] == "failure-race":
            raise RuntimeError("provider failed")
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        del generated
        delivered.append(snapshot.task_id)

    async def charge(task_id, user_id, amount):
        del task_id, user_id, amount
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
        max_running=1,
    )
    original_finish_terminal = manager._finish_terminal

    async def delay_provider_failure(*args, **kwargs):
        if kwargs.get("error_code") == "provider_failed":
            failure_settlement_started.set()
            await release_failure_settlement.wait()
        return await original_finish_terminal(*args, **kwargs)

    manager._finish_terminal = delay_provider_failure
    await manager.start()
    failed = await manager.accept(make_spec("failure-race", charge_amount=3))
    await asyncio.wait_for(failure_settlement_started.wait(), timeout=2)

    assert await manager.cancel_scope(scope_key=failed.task.scope_key) == 1
    cancelled = await wait_for_refund(manager, failed.task_id)

    assert cancelled.state == "cancelled"
    assert cancelled.refund_state == "refunded"
    assert refunds == [failed.task_id]

    follow_up = await manager.accept(make_spec("worker-still-alive"))
    await wait_for_state(manager, follow_up.task_id, {"completed"})
    assert delivered == [follow_up.task_id]
    assert all(not worker.done() for worker in manager._workers)
    await manager.close()


@pytest.mark.asyncio
async def test_worker_settles_an_escaped_failure_handler_and_keeps_running(
    tmp_path: Path,
):
    refunds = []
    delivered = []

    async def provider(snapshot, images):
        del images
        if snapshot.metadata["suffix"] == "escaped-handler":
            raise RuntimeError("provider failed")
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        del generated
        delivered.append(snapshot.task_id)

    async def charge(task_id, user_id, amount):
        del task_id, user_id, amount
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
        max_running=1,
    )
    original_finish_terminal = manager._finish_terminal
    failed_once = False

    async def fail_first_settlement(*args, **kwargs):
        nonlocal failed_once
        if kwargs.get("error_code") == "provider_failed" and not failed_once:
            failed_once = True
            raise RuntimeError("database write failed once")
        return await original_finish_terminal(*args, **kwargs)

    manager._finish_terminal = fail_first_settlement
    await manager.start()
    escaped = await manager.accept(make_spec("escaped-handler", charge_amount=2))
    settled = await wait_for_refund(manager, escaped.task_id)

    assert settled.state == "failed"
    assert settled.error_code == "execution_settlement_failed"
    assert settled.refund_state == "refunded"
    assert refunds == [escaped.task_id]

    follow_up = await manager.accept(make_spec("escaped-handler-follow-up"))
    await wait_for_state(manager, follow_up.task_id, {"completed"})
    assert delivered == [follow_up.task_id]
    assert all(not worker.done() for worker in manager._workers)
    await manager.close()


@pytest.mark.asyncio
async def test_close_leaves_interrupted_notice_for_next_start(tmp_path: Path):
    started = asyncio.Event()
    refunds = []
    notices = []

    async def provider(snapshot, images):
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def deliver(snapshot, generated):
        raise AssertionError("delivery must not run")

    async def charge(task_id, user_id, amount):
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    async def notify(snapshot, text):
        notices.append((snapshot.task_id, text))

    first = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
        notify_runner=notify,
    )
    await first.start()
    admission = await first.accept(make_spec("shutdown", charge_amount=2))
    await asyncio.wait_for(started.wait(), timeout=2)
    await first.close()

    interrupted = await first.get(admission.task_id)
    assert interrupted is not None
    assert interrupted.state == "interrupted"
    assert interrupted.refund_state == "refunded"
    assert interrupted.notification_state == "pending"
    assert notices == []

    second = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
        notify_runner=notify,
    )
    await second.start()
    notified = await second.get(admission.task_id)
    assert notified is not None
    assert notified.notification_state == "sent"
    assert len(notices) == 1
    await second.close()


@pytest.mark.asyncio
async def test_close_cancels_and_awaits_inflight_notifications(tmp_path: Path):
    notification_started = asyncio.Event()
    notification_cancelled = asyncio.Event()

    async def provider(snapshot, images):
        del snapshot, images
        raise RuntimeError("provider failed")

    async def deliver(snapshot, generated):
        raise AssertionError("failed task must not deliver")

    async def notify(snapshot, text):
        del snapshot, text
        notification_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            notification_cancelled.set()

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        notify_runner=notify,
    )
    await manager.start()
    result = await manager.accept(make_spec("notification-shutdown"))
    await asyncio.wait_for(notification_started.wait(), timeout=2)

    await manager.close(grace_seconds=0.1)
    snapshot = await manager.get(result.task_id)

    assert notification_cancelled.is_set()
    assert manager._notification_tasks == {}
    assert snapshot is not None
    assert snapshot.notification_state == "unknown"


@pytest.mark.asyncio
async def test_close_can_retry_after_partial_shutdown_failure(tmp_path: Path):
    async def provider(snapshot, images):
        del snapshot, images
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        del snapshot, generated

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
    )
    await manager.start()
    original_recover = manager._recover_abandoned_sync
    failed_once = False

    def fail_first_recovery():
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("database temporarily unavailable")
        return original_recover()

    manager._recover_abandoned_sync = fail_first_recovery

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        await manager.close()

    assert not manager._closing
    assert not manager._closed

    await manager.close()

    assert manager._closed
    assert manager._workers == []


@pytest.mark.asyncio
async def test_close_during_delivery_is_unknown_and_not_refunded(tmp_path: Path):
    sending = asyncio.Event()
    refunds = []

    async def provider(snapshot, images):
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        sending.set()
        await asyncio.Event().wait()

    async def charge(task_id, user_id, amount):
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
    )
    await manager.start()
    admission = await manager.accept(make_spec("shutdown-send", charge_amount=2))
    await asyncio.wait_for(sending.wait(), timeout=2)
    await manager.close()

    unknown = await manager.get(admission.task_id)
    assert unknown is not None
    assert unknown.state == "unknown"
    assert unknown.delivery_state == "unknown"
    assert unknown.refund_state == "none"
    assert refunds == []


@pytest.mark.asyncio
async def test_start_recovers_running_as_interrupted_and_sending_as_unknown(
    tmp_path: Path,
):
    async def unused_provider(snapshot, images):
        raise AssertionError("recovered tasks must not call the provider")

    async def unused_delivery(snapshot, generated):
        raise AssertionError("recovered tasks must not be delivered")

    seed = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=unused_provider,
        delivery_runner=unused_delivery,
    )
    seed._prepare_storage_sync()
    running_spec = make_spec("recovery-running")
    sending_spec = make_spec("recovery-sending")
    seed._reserve_task_sync("task-running", running_spec)
    seed._reserve_task_sync("task-sending", sending_spec)
    now = seed.now_ms()
    seed._transition_sync(
        "task-running",
        {"preparing"},
        "running",
        {"started_at_ms": now},
        now,
    )
    for old_state, new_state in (
        ("preparing", "queued"),
        ("queued", "running"),
        ("running", "generated"),
    ):
        seed._transition_sync(
            "task-sending",
            {old_state},
            new_state,
            {},
            now,
        )
    seed._begin_delivery_sync(
        "task-sending",
        "attempt-old",
        now,
    )

    notices = []

    async def notify(snapshot, text):
        notices.append((snapshot.task_id, snapshot.state, text))

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=unused_provider,
        delivery_runner=unused_delivery,
        notify_runner=notify,
    )
    recovered = await manager.start()
    states = {snapshot.task_id: snapshot.state for snapshot in recovered}

    assert states == {"task-running": "interrupted", "task-sending": "unknown"}
    assert (await manager.get("task-sending")).delivery_state == "unknown"
    assert {item[:2] for item in notices} == {
        ("task-running", "interrupted"),
        ("task-sending", "unknown"),
    }
    await manager.close()


@pytest.mark.asyncio
async def test_start_resumes_pending_completion_reply_without_replaying_image(
    tmp_path: Path,
):
    async def unused_provider(snapshot, images):
        raise AssertionError("completed task must not call the provider")

    async def unused_delivery(snapshot, generated):
        raise AssertionError("completed task must not replay image delivery")

    seed = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=unused_provider,
        delivery_runner=unused_delivery,
    )
    seed._prepare_storage_sync()
    seed._reserve_task_sync("task-completed", make_spec("recovery-completed"))
    now = seed.now_ms()
    for old_state, new_state in (
        ("preparing", "queued"),
        ("queued", "running"),
        ("running", "generated"),
    ):
        seed._transition_sync(
            "task-completed",
            {old_state},
            new_state,
            {},
            now,
        )
    seed._begin_delivery_sync("task-completed", "attempt-completed", now)
    assert seed._confirm_delivery_sync(
        "task-completed",
        "attempt-completed",
        now,
    )

    notified = asyncio.Event()
    notices = []

    async def notify(snapshot, text):
        notices.append((snapshot.task_id, snapshot.state, text))
        notified.set()

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=unused_provider,
        delivery_runner=unused_delivery,
        notify_runner=notify,
    )
    await manager.start()
    await asyncio.wait_for(notified.wait(), timeout=2)

    async def notification_finished():
        while True:
            snapshot = await manager.get("task-completed")
            if snapshot is not None and snapshot.notification_state == "sent":
                return snapshot
            await asyncio.sleep(0.01)

    completed = await asyncio.wait_for(notification_finished(), timeout=2)
    assert completed.state == "completed"
    assert completed.delivery_state == "confirmed"
    assert notices == [("task-completed", "completed", "照片拍好啦，我已经发出来了。")]
    await manager.close()


@pytest.mark.asyncio
async def test_expired_active_task_is_recovered_before_terminal_gc(tmp_path: Path):
    refunds = []

    async def unused_provider(snapshot, images):
        raise AssertionError("expired recovery must not run the provider")

    async def unused_delivery(snapshot, generated):
        raise AssertionError("expired recovery must not deliver")

    async def charge(task_id, user_id, amount):
        del task_id, user_id, amount
        return True

    async def refund(task_id):
        refunds.append(task_id)
        return True

    seed = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=unused_provider,
        delivery_runner=unused_delivery,
    )
    seed._prepare_storage_sync()
    seed._reserve_task_sync(
        "task-expired-active",
        make_spec("expired-active", charge_amount=4),
    )
    with seed._connect() as conn:
        conn.execute(
            "UPDATE image_tasks SET state='running', charge_state='charged', "
            "started_at_ms=?, expires_at_ms=? WHERE task_id=?",
            (seed.now_ms() - 1000, seed.now_ms() - 1, "task-expired-active"),
        )

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=unused_provider,
        delivery_runner=unused_delivery,
        charge_runner=charge,
        refund_runner=refund,
    )
    recovered = await manager.start()
    interrupted = await manager.get("task-expired-active")

    assert [item.task_id for item in recovered] == ["task-expired-active"]
    assert interrupted is not None
    assert interrupted.state == "interrupted"
    assert interrupted.refund_state == "refunded"
    assert refunds == ["task-expired-active"]
    await manager.close()


@pytest.mark.asyncio
async def test_explicit_fingerprint_deduplicates_without_second_charge(tmp_path: Path):
    release = asyncio.Event()
    charges = []

    async def provider(snapshot, images):
        await release.wait()
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        return None

    async def charge(task_id, user_id, amount):
        charges.append(task_id)
        return True

    async def refund(task_id):
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
    )
    await manager.start()
    spec = make_spec("dedupe", charge_amount=1, fingerprint="same-request")
    first = await manager.accept(spec)
    second = await manager.accept(spec)

    assert first.created is True
    assert second.created is False
    assert second.task_id == first.task_id
    assert charges == [first.task_id]
    release.set()
    await wait_for_state(manager, first.task_id, {"completed"})
    await manager.close()


@pytest.mark.asyncio
async def test_fingerprint_deduplication_is_scoped_to_requester_context(tmp_path: Path):
    release = asyncio.Event()
    charges = []

    async def provider(snapshot, images):
        del snapshot, images
        await release.wait()
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        del snapshot, generated

    async def charge(task_id, user_id, amount):
        del amount
        charges.append((task_id, user_id))
        return True

    async def refund(task_id):
        del task_id
        return True

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
        charge_runner=charge,
        refund_runner=refund,
        max_running=2,
    )
    await manager.start()
    first_spec = make_spec(
        "dedupe-scope-one",
        charge_amount=1,
        fingerprint="shared-fingerprint",
    )
    second_spec = replace(
        first_spec,
        scope_key=build_scope_key("platform:GroupMessage:other", "other", "cid-2"),
        umo="platform:GroupMessage:other",
        user_id="other",
        conversation_id="cid-2",
    )
    first = await manager.accept(first_spec)
    second = await manager.accept(second_spec)

    assert first.created is True
    assert second.created is True
    assert second.task_id != first.task_id
    assert charges == [(first.task_id, "200"), (second.task_id, "other")]
    release.set()
    await wait_for_state(manager, first.task_id, {"completed"})
    await wait_for_state(manager, second.task_id, {"completed"})
    await manager.close()


@pytest.mark.asyncio
async def test_scope_status_and_scope_key_are_stable(tmp_path: Path):
    async def provider(snapshot, images):
        return GeneratedImageResult([b"image"])

    async def deliver(snapshot, generated):
        return None

    manager = BackgroundImageTaskManager(
        tmp_path,
        provider_runner=provider,
        delivery_runner=deliver,
    )
    await manager.start()
    spec = make_spec("status")
    result = await manager.accept(spec)
    await wait_for_state(manager, result.task_id, {"completed"})

    assert (
        build_scope_key(spec.umo, spec.user_id, spec.conversation_id) == spec.scope_key
    )
    status = await manager.format_scope_status(spec.scope_key)
    assert result.task_id in status
    assert "已完成" in status
    await manager.close()
