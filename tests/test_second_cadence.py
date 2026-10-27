"""Per-second observation must preserve closed-candle execution safeguards."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import multi_agent_orchestrator as runtime
from test_execution_state import make_system as make_execution_system
from test_native_refresh import frame, make_system as make_native_system


def test_same_closed_snapshot_rechecks_both_gates_each_second(
        monkeypatch, tmp_path, caplog):
    system, bars, clock = make_execution_system(monkeypatch, tmp_path)
    system.observe_only = True
    original = runtime.evaluate_both
    evaluate = Mock(wraps=original)
    monkeypatch.setattr(runtime, "evaluate_both", evaluate)

    async def check():
        await system._evaluate_and_trade(1.105)
        clock.now += 1.0
        await system._evaluate_and_trade(1.106)

    with caplog.at_level(logging.INFO, logger="multi_agent_orchestrator"):
        asyncio.run(check())

    assert evaluate.call_count == 2
    assert all(call.args == (bars[30], bars[60]) for call in evaluate.call_args_list)
    assert sum("OnlyGate]" in record.message for record in caplog.records) == 4
    system.supervisor.safe_execute_trade.assert_not_awaited()


def test_each_second_recheck_never_retries_a_rejected_submission(
        monkeypatch, tmp_path):
    system, _, clock = make_execution_system(monkeypatch, tmp_path)
    system.supervisor.safe_execute_trade.return_value = False

    async def check():
        for second in (121.0, 122.0, 123.0, 125.0):
            clock.now = second
            await system._evaluate_and_trade(1.105)

    asyncio.run(check())

    system.supervisor.safe_execute_trade.assert_awaited_once()
    system.risk.record_trade.assert_not_called()
    assert system._last_attempted_entry == 60.0


@pytest.mark.parametrize("recheck_at, submissions", [(123.0, 1), (126.0, 0)])
def test_payout_recovery_only_allows_entry_inside_existing_confirmation_window(
        monkeypatch, tmp_path, recheck_at, submissions):
    system, _, clock = make_execution_system(monkeypatch, tmp_path)
    system.bot.get_asset_payout.side_effect = [None, .85]

    async def check():
        await system._evaluate_and_trade(1.105)
        assert system._last_attempted_entry is None
        clock.now = recheck_at
        await system._evaluate_and_trade(1.105)

    asyncio.run(check())

    assert system.supervisor.safe_execute_trade.await_count == submissions
    assert system.bot.get_asset_payout.await_count == 1 + submissions


def test_secondly_check_does_not_queue_behind_an_inflight_order(
        monkeypatch, tmp_path):
    system, _, _ = make_execution_system(monkeypatch, tmp_path)

    async def check():
        async with system._execution_lock:
            await asyncio.wait_for(system._evaluate_and_trade(1.105), timeout=.2)

    asyncio.run(check())

    system.bot.get_asset_payout.assert_not_awaited()
    system.supervisor.safe_execute_trade.assert_not_awaited()


def test_pending_native_request_leaves_sampled_ticks_running_without_overlap(monkeypatch):
    system = make_native_system()
    system.agent1.seed_from_dataframe(frame(30, 3510), timeframe=30, now=3600)
    system._seed_sources[60] = "broker_browser_sample"
    system._native_refresh_task = None
    clock = SimpleNamespace(now=3601.0)
    monkeypatch.setattr(runtime.time, "time", lambda: clock.now)

    async def check():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def fetch(*args):
            entered.set()
            await release.wait()
            return frame(30, 3570)

        system.bot = SimpleNamespace(get_native_candles=AsyncMock(side_effect=fetch))
        try:
            system._quote = {"observed_at": clock.now}
            assert await asyncio.wait_for(
                system._update_closed_candles(1.11, wait_for_native=False), .2) == set()
            await asyncio.wait_for(entered.wait(), .2)
            pending = system._native_refresh_task
            assert pending is not None and not pending.done()
            assert system.agent1.builders[60]._last_tick_ts == 3601.0

            clock.now = 3602.0
            system._quote = {"observed_at": clock.now}
            assert await asyncio.wait_for(
                system._update_closed_candles(1.12, wait_for_native=False), .2) == set()
            assert system._native_refresh_task is pending
            assert system.bot.get_native_candles.await_count == 1
            assert system.agent1.builders[60]._last_tick_ts == 3602.0
            assert system.agent1.builders[60]._current["close"] == 1.12

            release.set()
            await asyncio.wait_for(pending, 1.0)
            clock.now = 3603.0
            system._quote = {"observed_at": clock.now}
            closed = await system._update_closed_candles(1.13, wait_for_native=False)
            assert closed == {30}
            assert system.agent1.builders[30].latest_closed_ts == 3570
            assert system.agent1.builders[30].get_closes()[-1] == 1.15
            assert system.bot.get_native_candles.await_count == 1
        finally:
            release.set()
            task = system._native_refresh_task
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def configure_run_loop(system, monkeypatch):
    """Keep the real loop and scheduler while removing broker/setup side effects."""
    system.max_runtime = 0
    system.max_candle_closes = 0
    system.closed_events = 0
    system._feed_missing = False
    system._native_refresh_task = None
    system._decision_tasks = set()
    system.candle_monitor = SimpleNamespace(
        add_tick=Mock(), snapshot=Mock(return_value={}), invalidate=Mock())
    system.agent1.invalidate = Mock()
    system.setup = AsyncMock(return_value=True)
    system._shutdown = AsyncMock()
    system._evaluate_and_trade = AsyncMock()
    system._update_closed_candles = AsyncMock(return_value=set())
    real_sleep = asyncio.sleep
    sleeps = []

    async def next_poll(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(runtime.asyncio, "sleep", next_poll)
    return sleeps, real_sleep


def test_run_checks_each_valid_quote_when_no_strategy_candle_has_closed(
        monkeypatch, tmp_path):
    system, _, clock = make_execution_system(monkeypatch, tmp_path)
    sleeps, _ = configure_run_loop(system, monkeypatch)
    quotes = iter([(121.0, 1.105), (122.0, 1.106), (123.0, 1.107)])

    def next_quote():
        observed_at, price = next(quotes)
        clock.now = observed_at
        system._quote = {"observed_at": observed_at, "price": price}
        if observed_at == 123.0:
            system.running = False
        return price

    system._get_price = AsyncMock(side_effect=next_quote)

    async def check():
        await system.run(lock_acquired=True)
        await asyncio.gather(*system._decision_tasks)

    asyncio.run(check())

    assert system._get_price.await_count == 3
    assert system.candle_monitor.add_tick.call_count == 3
    assert system._evaluate_and_trade.await_count == 3
    assert [call.args[0] for call in system._evaluate_and_trade.await_args_list] == [
        1.105, 1.106, 1.107]
    assert all(call.kwargs == {"wait_for_native": False}
               for call in system._update_closed_candles.await_args_list)
    assert len(sleeps) == 3
    assert all(0 < delay <= 1.0 for delay in sleeps)
    assert system.closed_events == 0
    system._shutdown.assert_awaited_once()


def test_quote_outage_cancels_native_refresh_before_invalidating_histories(
        monkeypatch, tmp_path):
    system, _, _ = make_execution_system(monkeypatch, tmp_path)
    _, real_sleep = configure_run_loop(system, monkeypatch)
    polls = 0

    def missing_quote():
        nonlocal polls
        polls += 1
        system._quote = None
        if polls == 2:
            system.running = False
        return None

    system._get_price = AsyncMock(side_effect=missing_quote)

    async def check():
        async def pending_refresh():
            await asyncio.Event().wait()

        pending = asyncio.create_task(pending_refresh())
        system._native_refresh_task = pending
        await real_sleep(0)

        def invalidate(reason):
            assert reason == "quote_unavailable"
            assert pending.cancelled()

        system.agent1.invalidate.side_effect = invalidate
        await system.run(lock_acquired=True)
        assert pending.cancelled()
        assert system._native_refresh_task is None

    asyncio.run(check())

    system.agent1.invalidate.assert_called_once_with("quote_unavailable")
    system.candle_monitor.invalidate.assert_called_once_with("quote_unavailable")
    system._evaluate_and_trade.assert_not_awaited()
    system._update_closed_candles.assert_not_awaited()
    assert system._last_evaluation_key is None


def test_payout_response_from_before_feed_outage_cannot_submit_after_recovery(
        monkeypatch, tmp_path):
    system, bars, clock = make_execution_system(monkeypatch, tmp_path)
    system._feed_generation = 0
    system._feed_missing = False

    async def check():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_payout(*args):
            entered.set()
            await release.wait()
            return .85

        system.bot.get_asset_payout = AsyncMock(side_effect=delayed_payout)
        decision = asyncio.create_task(system._evaluate_and_trade(1.105))
        try:
            await asyncio.wait_for(entered.wait(), .2)
            original = {period: list(candles) for period, candles in bars.items()}
            system._feed_generation += 1
            system._feed_missing = True
            for candles in bars.values():
                candles.clear()
            # Recovered current data must not make the old in-flight decision valid.
            clock.now = 122.0
            for period, candles in bars.items():
                candles.extend(original[period])
            system._feed_missing = False
            release.set()
            await asyncio.wait_for(decision, .2)

            system.supervisor.safe_execute_trade.assert_not_awaited()
            assert system._last_attempted_entry is None
            assert not system._order_state_path.exists()

            # A fresh evaluation of recovered candles still uses the same five-second window.
            clock.now = 123.0
            await system._evaluate_and_trade(1.105)
            system.supervisor.safe_execute_trade.assert_awaited_once()
            assert system.bot.get_asset_payout.await_count == 2
            assert system._last_attempted_entry == 60.0
        finally:
            release.set()
            await asyncio.gather(decision, return_exceptions=True)

    asyncio.run(check())
