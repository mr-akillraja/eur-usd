"""Recover historical exposure using broker evidence, with isolated journals."""

import asyncio
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import multi_agent_orchestrator as runtime
from bot import risk as risk_module
from bot.risk import RiskManager, TradeRecord
from config import settings


@pytest.fixture
def recovery_system(monkeypatch, tmp_path):
    clock = SimpleNamespace(wall=datetime(2026, 10, 5, 12).timestamp(), monotonic=1000.0)

    class ClockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(clock.wall, tz)

    monkeypatch.setattr(runtime, "datetime", ClockDatetime)
    monkeypatch.setattr(risk_module, "datetime", ClockDatetime)
    monkeypatch.setattr(runtime, "time", SimpleNamespace(
        time=lambda: clock.wall, monotonic=lambda: clock.monotonic))
    monkeypatch.setattr(risk_module, "time", SimpleNamespace(time=lambda: clock.wall))
    monkeypatch.setattr(settings, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "JOURNAL_ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setattr(settings, "RESET_TODAY_LOGS_ON_STARTUP", False)
    monkeypatch.setattr(settings, "TRAINING_MODE", False)

    system = runtime.MultiAgentTradingSystem.__new__(runtime.MultiAgentTradingSystem)
    system.risk = RiskManager(max_daily_trades=50, max_daily_loss=2000)
    system.bot = SimpleNamespace(
        get_flat_ftt_account_evidence=AsyncMock(return_value=None),
        get_order_result=AsyncMock(return_value=None),
    )
    system.supervisor = SimpleNamespace(record_outcome=Mock())
    system._execution_lock = asyncio.Lock()
    system._execution_uncertain = False
    system._flat_reconciled_date = None
    system._next_flat_reconciliation_at = 0.0
    system.active_trades = []
    system._order_state_path = tmp_path / "execution_state.json"
    system.trade_amount = 200.0
    system.max_trades_per_day = 50
    system.daily_loss_limit = 2000.0
    return system, clock


def proof(clock, **overrides):
    return dict({
        "source": "platform_active_order_list", "demo": True,
        "no_active_ftt": True, "observed_at": clock.wall,
    }, **overrides)


def pending(system, clock, *, days=0, seconds=0, amount=200, mode="ftt"):
    timestamp = (datetime.fromtimestamp(clock.wall)
                 + timedelta(days=days, seconds=seconds)).isoformat()
    record = TradeRecord(timestamp=timestamp, asset="GBP/USD OTC", mode=mode,
                         signal="BUY", confidence=.95, amount=amount,
                         state_key="SHORT_CANDLE_GATE")
    system.risk.record_trade(record)
    return record


def journal_rows(system):
    with Path(system.risk.journal_path).open(newline="") as handle:
        return {row["timestamp"]: row for row in csv.DictReader(handle)}


def advance(clock, seconds):
    clock.wall += seconds
    clock.monotonic += seconds


def one_settlement_cycle(system, monkeypatch):
    async def stop_after_cycle(*args):
        raise asyncio.CancelledError

    monkeypatch.setattr(runtime.asyncio, "sleep", stop_after_cycle)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(system._settlement_loop())


def test_restored_order_settlement_releases_only_proven_historical_exposure(
        recovery_system, monkeypatch):
    system, clock = recovery_system
    historical = pending(system, clock, days=-2, amount=3000)
    today_pending = pending(system, clock, seconds=-120, amount=25)
    restored = pending(system, clock, seconds=-60)
    system.active_trades = [{
        "id": "restored-order", "timestamp": restored.timestamp,
        "signal": "BUY", "amount": 200, "state_key": "SHORT_CANDLE_GATE",
    }]
    system._persist_order_state({"status": "accepted", "order": system.active_trades[0]})
    path = Path(system.risk.journal_path)
    original_rows = journal_rows(system)
    preserved_lines = [line for line in path.read_bytes().splitlines()
                       if line.startswith((historical.timestamp.encode(), today_pending.timestamp.encode()))]
    assert system.risk.daily_snapshot()["halted"]

    # A restored active order prevents startup from claiming a flat account.
    asyncio.run(system._reconcile_flat_account())
    system.bot.get_flat_ftt_account_evidence.assert_not_awaited()

    system.bot.get_order_result.return_value = {"result": "WIN", "pnl": 170.0, "source": "platform"}
    system.bot.get_flat_ftt_account_evidence.return_value = proof(clock)
    one_settlement_cycle(system, monkeypatch)

    system.bot.get_flat_ftt_account_evidence.assert_awaited_once()
    snapshot = system.risk.daily_snapshot()
    assert snapshot["pending_exposure"] == 25
    assert snapshot["historical_unconfirmed_ftt_count"] == 1
    assert snapshot["accepted_count"] == 2
    assert snapshot["settled_count"] == 1
    assert snapshot["net_pnl"] == 170
    assert not snapshot["halted"]
    assert not system.active_trades
    assert json.loads(system._order_state_path.read_text()) is None
    rows = journal_rows(system)
    assert rows[historical.timestamp] == original_rows[historical.timestamp]
    assert rows[today_pending.timestamp] == original_rows[today_pending.timestamp]
    assert all(line in path.read_bytes().splitlines() for line in preserved_lines)
    assert rows[restored.timestamp]["result"] == "WIN"
    system.supervisor.record_outcome.assert_called_once_with(True)


@pytest.mark.parametrize("bad_proof", [
    "missing", "exception", "stale", "future", "nonfinite",
    "not_demo", "not_flat", "wrong_source",
])
def test_missing_or_invalid_broker_proof_never_releases_exposure(recovery_system, bad_proof):
    system, clock = recovery_system
    pending(system, clock, days=-1, amount=3000)
    path = Path(system.risk.journal_path)
    original = path.read_bytes()
    overrides = {
        "stale": {"observed_at": clock.wall - 10},
        "future": {"observed_at": clock.wall + 10},
        "nonfinite": {"observed_at": float("nan")},
        "not_demo": {"demo": False},
        "not_flat": {"no_active_ftt": False},
        "wrong_source": {"source": "inferred_from_expiry"},
    }
    if bad_proof == "exception":
        system.bot.get_flat_ftt_account_evidence.side_effect = RuntimeError("panel unavailable")
    elif bad_proof != "missing":
        system.bot.get_flat_ftt_account_evidence.return_value = proof(clock, **overrides[bad_proof])

    asyncio.run(system._reconcile_flat_account())
    system.bot.get_flat_ftt_account_evidence.assert_awaited_once()
    assert system.risk.daily_snapshot()["pending_exposure"] == 3000
    assert system.risk.daily_snapshot()["halted"]
    assert path.read_bytes() == original
    # Missing and rejected evidence share the retry delay rather than polling the UI each tick.
    asyncio.run(system._reconcile_flat_account())
    assert system.bot.get_flat_ftt_account_evidence.await_count == 1
    advance(clock, 30)
    system.bot.get_flat_ftt_account_evidence.side_effect = None
    system.bot.get_flat_ftt_account_evidence.return_value = proof(clock)
    asyncio.run(system._reconcile_flat_account())
    assert system.bot.get_flat_ftt_account_evidence.await_count == 2
    assert system.risk.daily_snapshot()["pending_exposure"] == 0
    assert path.read_bytes() == original


@pytest.mark.parametrize("blocked_by", ["active_order", "uncertain_click", "submitting"])
def test_reconciliation_skips_unresolved_or_submitting_execution(recovery_system, blocked_by):
    system, clock = recovery_system
    pending(system, clock, days=-1, amount=3000)
    system.bot.get_flat_ftt_account_evidence.return_value = proof(clock)

    async def check():
        if blocked_by == "active_order":
            system.active_trades = [{"id": "still-open"}]
        elif blocked_by == "uncertain_click":
            system._execution_uncertain = True
        else:
            await system._execution_lock.acquire()
        try:
            await system._reconcile_flat_account()
        finally:
            if blocked_by == "submitting":
                system._execution_lock.release()

    asyncio.run(check())
    system.bot.get_flat_ftt_account_evidence.assert_not_awaited()
    assert system.risk.daily_snapshot()["pending_exposure"] == 3000


def test_pending_exposure_ignores_stale_pending_rows(recovery_system):
    system, clock = recovery_system
    pending(system, clock, days=-1, amount=3000)
    pending(system, clock, seconds=-120, amount=25)
    pending(system, clock, seconds=-60, amount=40)

    assert system.risk.daily_snapshot()["pending_exposure"] == 65


def test_missing_startup_proof_retries_under_lock_and_caches_success_until_rollover(recovery_system):
    system, clock = recovery_system
    pending(system, clock, days=-1, amount=3000)
    pending(system, clock, seconds=-60, amount=25)
    original = Path(system.risk.journal_path).read_bytes()
    asyncio.run(system._reconcile_flat_account())
    advance(clock, 29)
    asyncio.run(system._reconcile_flat_account())
    assert system.bot.get_flat_ftt_account_evidence.await_count == 1

    async def fresh_proof():
        assert system._execution_lock.locked()
        return proof(clock)

    confirm = system.risk.confirm_flat_ftt_account

    def confirm_under_lock(value):
        assert system._execution_lock.locked()
        return confirm(value)

    system.risk.confirm_flat_ftt_account = Mock(side_effect=confirm_under_lock)
    system.bot.get_flat_ftt_account_evidence.side_effect = fresh_proof
    advance(clock, 1)
    asyncio.run(system._reconcile_flat_account())
    assert system.bot.get_flat_ftt_account_evidence.await_count == 2
    assert system.risk.daily_snapshot()["pending_exposure"] == 25
    assert not system._execution_lock.locked()
    advance(clock, 60)
    asyncio.run(system._reconcile_flat_account())
    assert system.bot.get_flat_ftt_account_evidence.await_count == 2

    advance(clock, 86400)
    asyncio.run(system._reconcile_flat_account())
    assert system.bot.get_flat_ftt_account_evidence.await_count == 3
    assert system.risk.confirm_flat_ftt_account.call_count == 2
    assert system.risk.daily_snapshot()["pending_exposure"] == 0
    assert Path(system.risk.journal_path).read_bytes() == original
