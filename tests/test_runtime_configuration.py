"""Runtime configuration reaches durable risk controls without browser orders."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from bot.risk import TradeRecord
from config import settings
import multi_agent_orchestrator as runtime


@pytest.fixture
def isolated_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "JOURNAL_ARCHIVE_DIR", str(tmp_path / "archive"))
    for name, value in {
        "RESET_TODAY_LOGS_ON_STARTUP": False,
        "TRAINING_MODE": False,
        "TRADE_AMOUNT": 125,
        "MAX_DAILY_TRADES": 73,
        "MAX_DAILY_LOSS_USD": 1750,
        "MAX_TRADE_LOSS_USD": 500,
        "MAX_CONCURRENT_OPEN_TRADES": 2,
        "CONFIDENCE_THRESHOLD": 0.50,
    }.items():
        monkeypatch.setattr(settings, name, value)
    for name in ("OlymtradeBot", "AssetSelectorAgent", "RLFeedbackAgent",
                 "RegimeDetector", "ChartAnalyzer", "MultiTimeframeConsensusAnalyzer"):
        monkeypatch.setattr(runtime, name, lambda *args, **kwargs: SimpleNamespace())


def assert_limits(system, amount, trades, loss):
    assert system.trade_amount == amount
    assert system.max_trades_per_day == system.risk.max_daily_trades == trades
    assert system.daily_loss_limit == system.risk.max_daily_loss == loss
    snapshot = system.risk.daily_snapshot()
    assert snapshot["max_daily_trades"] == trades
    assert snapshot["max_daily_loss"] == loss


def approve(system, *, confidence=0.95):
    return system.risk.approve(SimpleNamespace(value="BUY"), confidence, "ftt",
                               trade_amount=system.trade_amount)


def record(system, index, *, result="DRAW", amount=200, pnl=0):
    at = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamp = (at + timedelta(microseconds=index)).isoformat()
    row = TradeRecord(timestamp=timestamp, asset=settings.ASSET, mode="ftt",
                      signal="BUY", confidence=0.95, amount=amount,
                      result=result, pnl=pnl, state_key="CLOSED_CANDLE_GATE",
                      settled_at=timestamp if result != "PENDING" else "")
    assert system.risk.record_trade(row)
    return row


def test_constructor_resolves_current_settings_for_each_instance(isolated_runtime, monkeypatch):
    first = runtime.MultiAgentTradingSystem()
    assert_limits(first, 125, 73, 1750)

    monkeypatch.setattr(settings, "TRADE_AMOUNT", 275)
    monkeypatch.setattr(settings, "MAX_DAILY_TRADES", 61)
    monkeypatch.setattr(settings, "MAX_DAILY_LOSS_USD", 3100)
    second = runtime.MultiAgentTradingSystem()
    assert_limits(second, 275, 61, 3100)
    assert_limits(first, 125, 73, 1750)


def test_hybrid_mode_keeps_pattern_gate_enabled_by_default(isolated_runtime):
    hybrid = runtime.MultiAgentTradingSystem(entry_mode="hybrid")
    assert hybrid.enable_pattern_gate is True

    classic = runtime.MultiAgentTradingSystem(entry_mode="classic")
    assert classic.enable_pattern_gate is False


def test_cli_omitted_values_preserve_configured_limits(isolated_runtime):
    args = runtime.make_argument_parser().parse_args([])
    assert args.trade_amount is args.max_daily_trades is args.max_daily_loss is None
    system = runtime.MultiAgentTradingSystem(
        trade_amount=args.trade_amount, max_trades_per_day=args.max_daily_trades,
        daily_loss_limit=args.max_daily_loss)
    assert_limits(system, 125, 73, 1750)


@pytest.mark.parametrize("argv, expected", [
    (["--trade-amount", "50"], (50, 73, 1750)),
    (["--max-daily-trades", "4"], (125, 4, 1750)),
    (["--max-daily-loss", "300"], (125, 73, 300)),
    (["--trade-amount", "50", "--max-daily-trades", "4", "--max-daily-loss", "300"],
     (50, 4, 300)),
])
def test_explicit_cli_values_override_only_their_settings(isolated_runtime, argv, expected):
    args = runtime.make_argument_parser().parse_args(argv)
    system = runtime.MultiAgentTradingSystem(
        trade_amount=args.trade_amount, max_trades_per_day=args.max_daily_trades,
        daily_loss_limit=args.max_daily_loss)
    assert_limits(system, *expected)


def test_configured_fifty_trade_cap_allows_twenty_first_but_stops_at_fifty(
        isolated_runtime, monkeypatch):
    monkeypatch.setattr(settings, "TRADE_AMOUNT", 200)
    monkeypatch.setattr(settings, "MAX_DAILY_TRADES", 50)
    monkeypatch.setattr(settings, "MAX_DAILY_LOSS_USD", 2000)
    initial = runtime.MultiAgentTradingSystem()
    for index in range(20):
        record(initial, index)

    system = runtime.MultiAgentTradingSystem()
    assert system._trades_today == 20
    assert system._trade_capacity() == (30, 10, 2000)
    assert not system._check_circuit_breaker()
    assert approve(system) == (True, "Approved")
    twenty_first = record(system, 20, result="PENDING")
    assert system.risk.daily_snapshot()["accepted_count"] == 21
    assert system.risk.update_pnl_and_journal(twenty_first.timestamp, "DRAW", 0)
    for index in range(21, 50):
        assert approve(system) == (True, "Approved")
        record(system, index)

    restarted = runtime.MultiAgentTradingSystem()
    assert restarted._trades_today == 50
    assert restarted._trade_capacity()[0] == 0
    assert restarted._check_circuit_breaker()
    accepted, reason = approve(restarted)
    assert not accepted
    assert "Max daily trades reached (50)" in reason


def test_configured_daily_loss_still_blocks_after_restart(isolated_runtime, monkeypatch):
    monkeypatch.setattr(settings, "MAX_DAILY_LOSS_USD", 400)
    system = runtime.MultiAgentTradingSystem()
    record(system, 0, result="LOSS", pnl=-200)
    record(system, 1, result="LOSS", pnl=-200)

    restarted = runtime.MultiAgentTradingSystem()
    assert restarted._trade_capacity()[2] == 0
    assert restarted._check_circuit_breaker()
    accepted, reason = approve(restarted)
    assert not accepted
    assert "Daily loss budget exhausted" in reason


def test_pending_exposure_reserves_configured_daily_loss_budget(isolated_runtime, monkeypatch):
    monkeypatch.setattr(settings, "MAX_DAILY_LOSS_USD", 300)
    system = runtime.MultiAgentTradingSystem()
    assert approve(system)[0]
    record(system, 0, result="PENDING", amount=200)

    restarted = runtime.MultiAgentTradingSystem()
    assert restarted._trade_capacity()[2] == 100
    accepted, reason = approve(restarted)
    assert not accepted
    assert "pending_exposure=$200.00" in reason
    assert "breach daily loss cap" in reason


def test_configured_stake_cannot_bypass_per_trade_limit(isolated_runtime, monkeypatch):
    monkeypatch.setattr(settings, "MAX_TRADE_LOSS_USD", 100)
    system = runtime.MultiAgentTradingSystem()
    accepted, reason = approve(system)
    assert not accepted
    assert "MAX_TRADE_LOSS_USD $100.00" in reason


def test_configured_trade_cap_does_not_bypass_concurrent_order_limit(isolated_runtime):
    system = runtime.MultiAgentTradingSystem()
    record(system, 0, result="PENDING")
    record(system, 1, result="PENDING")
    accepted, reason = approve(system)
    assert not accepted
    assert "Max concurrent open trades reached (2)" in reason


def test_configured_confidence_threshold_still_filters_entries(isolated_runtime):
    system = runtime.MultiAgentTradingSystem()
    assert approve(system, confidence=0.50) == (True, "Approved")
    accepted, reason = approve(system, confidence=0.49)
    assert not accepted
    assert "threshold 50.00%" in reason
