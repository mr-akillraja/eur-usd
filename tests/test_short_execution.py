"""Short signals share the existing order, risk, and persistence safeguards."""

import asyncio
import json
import math
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock

import pytest

import multi_agent_orchestrator as runtime
from agents.agent_pipeline_supervisor import PipelineSupervisorAgent
from models.predictor import Signal
from strategies.closed_candle import evaluate_both
from strategies.short_candle import evaluate_short
from test_browser_execution import execution_bot
from test_execution_state import make_system as make_execution_system
from test_short_strategy import buy_fixture, mirror


def make_short_system(monkeypatch, tmp_path, *, direction="BUY", mode="hybrid"):
    system, classic, clock = make_execution_system(monkeypatch, tmp_path)
    system.entry_mode = mode
    system._last_attempted_short_entry = None
    system._feed_generation = 0
    system._feed_missing = False
    short = {
        period: [dict(open=1.1, high=1.11, low=1.09, close=1.105,
                      ts=math.floor(clock.now / period) * period - period)]
        for period in (5, 10, 15)
    }
    system.candle_monitor = SimpleNamespace(closed_snapshots=lambda: {
        period: [dict(candle) for candle in candles]
        for period, candles in short.items()
    })
    # Real classic gates reject these deliberately insufficient histories.
    monkeypatch.setattr(runtime, "evaluate_both", evaluate_both)
    evaluate = Mock(return_value={
        name: {"passed": name == direction, "entry_close": 1.105}
        for name in ("BUY", "SELL")
    })
    monkeypatch.setattr(runtime, "evaluate_short", evaluate)
    return system, classic, short, clock, evaluate


def real_supervisor(system, monkeypatch):
    """Keep actual supervisor/risk approval; replace only the browser order call."""
    monkeypatch.setattr(runtime.settings, "TRADE_AMOUNT", system.trade_amount)
    monkeypatch.setattr(runtime.settings, "MAX_TRADE_LOSS_USD", 2000.0)
    monkeypatch.setattr(runtime.settings, "FTT_DURATION", 60)
    monkeypatch.setattr(runtime.settings, "TRAINING_MODE", False)
    monkeypatch.setattr(runtime.settings, "CONFIDENCE_THRESHOLD", .60)
    system.bot.place_ftt_trade = AsyncMock(return_value=True)
    system.supervisor = PipelineSupervisorAgent(
        system.bot, system.risk, system.agent1, SimpleNamespace())
    system.supervisor._current_asset = system._current_asset
    return system.supervisor


@pytest.mark.parametrize("direction", ["BUY", "SELL"])
@pytest.mark.parametrize("mode", ["hybrid", "short"])
def test_real_short_setup_reaches_broker_when_classic_gates_fail(
        monkeypatch, tmp_path, direction, mode):
    system, classic, _, clock, _ = make_short_system(
        monkeypatch, tmp_path, direction=direction, mode=mode)
    short, clock.now = buy_fixture(now=2401.0)
    if direction == "SELL":
        short = mirror(short)
    system.candle_monitor = SimpleNamespace(closed_snapshots=lambda: short)
    monkeypatch.setattr(runtime, "evaluate_short", evaluate_short)
    real_supervisor(system, monkeypatch)
    assert not any(result["passed"] for result in evaluate_both(
        classic[30], classic[60]).values())
    assert evaluate_short(short, clock.now)[direction]["passed"]

    async def check():
        await system._evaluate_and_trade(short[5][-1]["close"])
        clock.now += 1
        await system._evaluate_and_trade(short[5][-1]["close"])

    asyncio.run(check())
    signal = Signal.BUY if direction == "BUY" else Signal.SELL
    system.bot.place_ftt_trade.assert_awaited_once_with(
        signal, 60, asset=system._current_asset, entry_deadline=2405.0,
        entry_guard=ANY)
    guard = system.bot.place_ftt_trade.await_args.kwargs["entry_guard"]
    assert callable(guard) and guard()
    system._feed_missing = True
    assert not guard()
    system._feed_missing = False
    assert guard()
    system._feed_generation += 1
    assert not guard()
    assert system._accepted_count == len(system.active_trades) == 1
    assert system.active_trades[0]["signal"] == direction
    assert system.active_trades[0]["state_key"] == "SHORT_CANDLE_GATE"
    persisted = json.loads(system._order_state_path.read_text())
    assert persisted["status"] == "accepted"
    assert persisted["order"]["id"] == "broker-123"
    record = system.risk.record_trade.call_args.args[0]
    assert record.signal == direction
    assert record.state_key == "SHORT_CANDLE_GATE"
    assert record.amount == system.bot.last_order["amount"]


def test_classic_mode_never_uses_short_signals(monkeypatch, tmp_path):
    system, _, _, _, evaluate = make_short_system(monkeypatch, tmp_path, mode="classic")
    asyncio.run(system._evaluate_and_trade(1.105))
    evaluate.assert_not_called()
    system.supervisor.safe_execute_trade.assert_not_awaited()


@pytest.mark.parametrize("feed_state", ["healthy", "missing", "recovered"])
def test_real_short_browser_order_checks_orchestrator_feed_before_click(
        monkeypatch, tmp_path, feed_state):
    system, _, _, clock, _ = make_short_system(monkeypatch, tmp_path)
    short, clock.now = buy_fixture(now=2401.0)
    system.candle_monitor = SimpleNamespace(closed_snapshots=lambda: short)
    monkeypatch.setattr(runtime, "evaluate_short", evaluate_short)
    real_supervisor(system, monkeypatch)
    browser, button = execution_bot()
    system.bot = system.supervisor.bot = browser
    browser._order_rows.side_effect = [[], [{
        "id": "confirmed-short-order", "asset": system._current_asset,
        "amount": system.trade_amount, "direction": "BUY", "duration": 60,
    }]]

    async def prepare_amount(*args):
        if feed_state != "healthy":
            system._feed_generation += 1
            # A reconnect clears the missing flag but cannot validate the old signal.
            system._feed_missing = feed_state == "missing"
        return True

    browser.set_amount = AsyncMock(side_effect=prepare_amount)
    asyncio.run(system._evaluate_and_trade(short[5][-1]["close"]))
    browser.set_amount.assert_awaited_once_with(system.trade_amount)
    if feed_state == "healthy":
        button.click.assert_awaited_once()
        assert system.active_trades[0]["id"] == "confirmed-short-order"
        assert system.active_trades[0]["state_key"] == "SHORT_CANDLE_GATE"
        system.risk.record_trade.assert_called_once()
    else:
        button.click.assert_not_awaited()
        assert not system.active_trades
        assert not system._execution_uncertain
        assert json.loads(system._order_state_path.read_text()) is None
        system.risk.record_trade.assert_not_called()


def test_short_mode_does_not_submit_a_passing_classic_signal(monkeypatch, tmp_path):
    system, _, _, _, evaluate = make_short_system(monkeypatch, tmp_path, mode="short")
    monkeypatch.setattr(runtime, "evaluate_both", lambda *args: {
        "BUY": {"passed": True}, "SELL": {"passed": False}})
    evaluate.return_value = {"BUY": {"passed": False}, "SELL": {"passed": False}}
    asyncio.run(system._evaluate_and_trade(1.105))
    evaluate.assert_called_once()
    system.supervisor.safe_execute_trade.assert_not_awaited()


def test_fresh_opposing_strategies_do_not_submit(monkeypatch, tmp_path):
    system, _, _, _, _ = make_short_system(monkeypatch, tmp_path, direction="SELL")
    monkeypatch.setattr(runtime, "evaluate_both", lambda *args: {
        "BUY": {"passed": True}, "SELL": {"passed": False}})
    asyncio.run(system._evaluate_and_trade(1.105))
    system.bot.get_asset_payout.assert_not_awaited()
    system.supervisor.safe_execute_trade.assert_not_awaited()
    assert system._last_attempted_entry is None
    assert system._last_attempted_short_entry is None


def test_matching_strategies_prefer_the_original_entry(monkeypatch, tmp_path):
    system, _, _, _, _ = make_short_system(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime, "evaluate_both", lambda *args: {
        "BUY": {"passed": True}, "SELL": {"passed": False}})
    system.supervisor.safe_execute_trade.return_value = True
    asyncio.run(system._evaluate_and_trade(1.105))
    system.supervisor.safe_execute_trade.assert_awaited_once()
    assert system.active_trades[0]["state_key"] == "CLOSED_CANDLE_GATE"
    assert system._last_attempted_entry == 60.0
    assert system._last_attempted_short_entry is None


def test_stale_classic_direction_cannot_veto_a_new_short_entry(monkeypatch, tmp_path):
    system, _, short, clock, _ = make_short_system(monkeypatch, tmp_path, direction="SELL")
    monkeypatch.setattr(runtime, "evaluate_both", lambda *args: {
        "BUY": {"passed": True}, "SELL": {"passed": False}})
    clock.now = 127.0
    for period, candles in short.items():
        candles[-1]["ts"] = math.floor(clock.now / period) * period - period
    asyncio.run(system._evaluate_and_trade(1.105))
    system.supervisor.safe_execute_trade.assert_awaited_once_with(
        Signal.SELL, .95, "ftt", entry_deadline=130.0, entry_guard=ANY)
    assert system._last_attempted_short_entry == 120.0
    assert system._last_attempted_entry is None


def test_secondly_short_checks_do_not_retry_a_rejected_order(monkeypatch, tmp_path):
    system, _, _, clock, evaluate = make_short_system(monkeypatch, tmp_path)

    async def check():
        for second in (121.0, 122.0, 123.0, 124.0):
            clock.now = second
            await system._evaluate_and_trade(1.105)

    asyncio.run(check())
    assert evaluate.call_count == 4
    system.supervisor.safe_execute_trade.assert_awaited_once()
    assert system._last_attempted_short_entry == 115.0
    assert system._last_attempted_entry is None
    assert json.loads(system._order_state_path.read_text()) is None
    system.risk.record_trade.assert_not_called()


def test_next_short_candle_can_submit_after_known_rejection(monkeypatch, tmp_path):
    system, _, short, clock, _ = make_short_system(monkeypatch, tmp_path)

    async def check():
        await system._evaluate_and_trade(1.105)
        clock.now = 126.0
        short[5].append(dict(short[5][-1], ts=120.0))
        await system._evaluate_and_trade(1.106)

    asyncio.run(check())
    assert system.supervisor.safe_execute_trade.await_count == 2
    assert system._last_attempted_short_entry == 120.0
    assert system._last_attempted_entry is None


@pytest.mark.parametrize("pending", ["accepted", "uncertain"])
def test_pending_order_blocks_short_entries(monkeypatch, tmp_path, pending):
    system, _, _, _, _ = make_short_system(monkeypatch, tmp_path)
    if pending == "accepted":
        system.active_trades = [dict(system.bot.last_order)]
    else:
        system._execution_uncertain = True
    asyncio.run(system._evaluate_and_trade(1.105))
    system.bot.get_asset_payout.assert_not_awaited()
    system.supervisor.safe_execute_trade.assert_not_awaited()


@pytest.mark.parametrize("blocked", ["circuit", "loss_halt", "daily_count", "loss_budget"])
def test_short_entries_obey_existing_daily_controls(monkeypatch, tmp_path, blocked):
    system, _, _, _, _ = make_short_system(monkeypatch, tmp_path)
    if blocked == "circuit":
        system._check_circuit_breaker = lambda: True
    elif blocked == "loss_halt":
        system._check_consecutive_loss_halt = lambda: (True, "loss halt")
    elif blocked == "daily_count":
        system._trade_capacity = lambda: (0, 10, 2000.0)
    else:
        system._trade_capacity = lambda: (20, 0, 100.0)
    asyncio.run(system._evaluate_and_trade(1.105))
    system.bot.get_asset_payout.assert_not_awaited()
    system.supervisor.safe_execute_trade.assert_not_awaited()


@pytest.mark.parametrize("payout", [None, .79, 1.01])
def test_short_entries_require_current_acceptable_payout(monkeypatch, tmp_path, payout):
    system, _, _, _, _ = make_short_system(monkeypatch, tmp_path)
    system.bot.get_asset_payout.return_value = payout
    asyncio.run(system._evaluate_and_trade(1.105))
    system.supervisor.safe_execute_trade.assert_not_awaited()
    assert system._last_attempted_short_entry is None


@pytest.mark.parametrize("start, finish", [
    (121.0, 125.0), (121.0, 126.0), (129.0, 130.0), (134.0, 135.0),
])
def test_short_snapshot_expiring_during_payout_is_never_submitted(
        monkeypatch, tmp_path, start, finish):
    system, _, short, clock, _ = make_short_system(monkeypatch, tmp_path)
    clock.now = start
    for period, candles in short.items():
        candles[-1]["ts"] = math.floor(start / period) * period - period

    async def delayed_payout(*args):
        clock.now = finish
        return .85

    system.bot.get_asset_payout = AsyncMock(side_effect=delayed_payout)
    asyncio.run(system._evaluate_and_trade(1.105))
    system.bot.get_asset_payout.assert_awaited_once()
    system.supervisor.safe_execute_trade.assert_not_awaited()
    assert system._last_attempted_short_entry is None
    assert not system._order_state_path.exists()


@pytest.mark.parametrize("period", [5, 10, 15])
def test_executor_rechecks_freshness_of_every_short_timeframe(monkeypatch, tmp_path, period):
    system, _, short, _, _ = make_short_system(monkeypatch, tmp_path)
    short[period][-1]["ts"] -= period
    asyncio.run(system._evaluate_and_trade(1.105))
    system.bot.get_asset_payout.assert_not_awaited()
    system.supervisor.safe_execute_trade.assert_not_awaited()


def test_short_signal_cannot_survive_feed_outage_during_payout(monkeypatch, tmp_path):
    system, _, _, _, _ = make_short_system(monkeypatch, tmp_path)

    async def reconnected_payout(*args):
        system._feed_generation += 1
        return .85

    system.bot.get_asset_payout = AsyncMock(side_effect=reconnected_payout)
    asyncio.run(system._evaluate_and_trade(1.105))
    system.supervisor.safe_execute_trade.assert_not_awaited()
    assert system._last_attempted_short_entry is None


def test_short_submission_persists_before_the_only_order_attempt(monkeypatch, tmp_path):
    system, _, _, _, _ = make_short_system(monkeypatch, tmp_path)

    async def inspect_submission(*args, **kwargs):
        assert kwargs["entry_deadline"] == 125.0
        persisted = json.loads(system._order_state_path.read_text())
        assert persisted["status"] == "submitting"
        assert persisted["entry_ts"] == 115.0
        assert persisted["entry_period_seconds"] == 5
        assert persisted["strategy"] == "short"
        assert persisted["duration"] == 60
        assert persisted["amount"] == 200.0
        return True

    system.supervisor.safe_execute_trade = AsyncMock(side_effect=inspect_submission)
    asyncio.run(system._evaluate_and_trade(1.105))
    persisted = json.loads(system._order_state_path.read_text())
    assert persisted["status"] == "accepted"
    assert persisted["order"]["state_key"] == "SHORT_CANDLE_GATE"
    assert persisted["order"]["amount"] == 123.0
    assert system.risk.record_trade.call_args.args[0].amount == 123.0
    assert system._accepted_count == 1


@pytest.mark.parametrize("blocked", ["per_trade", "daily_budget", "supervisor_circuit"])
def test_short_signal_cannot_bypass_real_supervisor_approval(monkeypatch, tmp_path, blocked):
    system, _, _, _, _ = make_short_system(monkeypatch, tmp_path)
    supervisor = real_supervisor(system, monkeypatch)
    if blocked == "per_trade":
        monkeypatch.setattr(runtime.settings, "MAX_TRADE_LOSS_USD", 199.0)
    elif blocked == "daily_budget":
        system.risk.max_daily_loss = 199.0
    else:
        for _ in range(5):
            supervisor.cb.record(False)
    asyncio.run(system._evaluate_and_trade(1.105))
    system.bot.place_ftt_trade.assert_not_awaited()
    assert not system.active_trades
    system.risk.record_trade.assert_not_called()


def test_short_observe_only_does_not_submit(monkeypatch, tmp_path):
    system, _, _, _, _ = make_short_system(monkeypatch, tmp_path)
    system.observe_only = True
    asyncio.run(system._evaluate_and_trade(1.105))
    system.bot.get_asset_payout.assert_not_awaited()
    system.supervisor.safe_execute_trade.assert_not_awaited()


def test_short_execution_requires_process_lock(monkeypatch, tmp_path):
    system, _, _, _, _ = make_short_system(monkeypatch, tmp_path)
    system._process_lock_held = False
    with pytest.raises(RuntimeError, match="process lock"):
        asyncio.run(system._evaluate_and_trade(1.105))
    system.supervisor.safe_execute_trade.assert_not_awaited()


def test_inflight_order_does_not_queue_short_candidates(monkeypatch, tmp_path):
    system, _, _, _, _ = make_short_system(monkeypatch, tmp_path)

    async def check():
        async with system._execution_lock:
            await asyncio.wait_for(system._evaluate_and_trade(1.105), timeout=.2)

    asyncio.run(check())
    system.bot.get_asset_payout.assert_not_awaited()
    system.supervisor.safe_execute_trade.assert_not_awaited()
