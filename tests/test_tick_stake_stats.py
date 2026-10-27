import math

from agents.candle_monitor import CandleMonitor
from multi_agent_orchestrator import MultiAgentTradingSystem
from stats import wilson_interval


def test_tick_aggregator_tracks_tick_counts_and_thin_flags():
    monitor = CandleMonitor(periods=(5, 10, 15, 20, 30, 60), max_tick_gap=5.0)
    for ts, price in enumerate([1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10]):
        monitor.add_tick(price, ts, source="broker_native_ws")
    snapshot = monitor.snapshot(now=10)
    first = snapshot["5s"]["history"][-1]
    assert first["tick_count"] == 5
    assert first["thin"] is False
    assert first["source"] == "broker_native_ws"

    partial = CandleMonitor(periods=(5,), max_tick_gap=5.0)
    partial.add_tick(1.00, 0, source="broker_native_ws")
    partial.add_tick(1.01, 1, source="broker_native_ws")
    state = partial.snapshot(now=1)["5s"]
    assert state["forming"]["tick_count"] == 2
    assert state["forming"]["thin"] is True


def test_volatility_adjusted_stake_is_clamped_and_rounded():
    assert MultiAgentTradingSystem._stake_from_range(100.0, 1.0, 1.0, min_stake=1.0, step=1.0) == 1.0
    assert MultiAgentTradingSystem._stake_from_range(100.0, 4.0, 1.0, min_stake=1.0, step=1.0) == 1.0
    assert MultiAgentTradingSystem._stake_from_range(100.0, 0.2, 1.0, min_stake=1.0, step=1.0) == 1.0
    assert MultiAgentTradingSystem._stake_from_range(10.0, 2.0, 1.0, min_stake=5.0, step=5.0) == 5.0


def test_wilson_interval_is_conservative_for_binomial_process():
    lower, upper = wilson_interval(55, 100)
    assert math.isfinite(lower)
    assert math.isfinite(upper)
    assert 0.45 < lower < 0.55
    assert lower < 0.549
    assert upper > 0.55
