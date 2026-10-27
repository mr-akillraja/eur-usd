"""Observation-only short candles retain honest closure and freshness semantics."""

import json

import pytest

from agents.candle_monitor import CandleMonitor
from multi_agent_orchestrator import MultiAgentTradingSystem


def test_short_candles_align_and_close_only_on_a_new_quote():
    monitor = CandleMonitor()
    events = {}
    for ts in range(31):
        periods = monitor.add_tick(1 + ts / 1000, ts)
        if periods:
            events[ts] = periods
    assert events == {5: {5}, 10: {5, 10}, 15: {5, 15},
                      20: {5, 10}, 25: {5}, 30: {5, 10, 15}}
    snapshot = monitor.snapshot(now=30)
    for period, count in [(5, 6), (10, 3), (15, 2)]:
        state = snapshot[f"{period}s"]
        assert state["closed_count"] == count
        assert state["ready"] is True
        assert state["latest_closed"]["open_ts"] == 30 - period
        assert state["latest_closed"]["close_ts"] == 30
        assert state["latest_closed"]["close"] == 1.029
        assert state["latest_closed"]["closed"] is True
        assert state["forming"]["open_ts"] == 30
        assert state["forming"]["close"] == 1.03
        assert state["forming"]["closed"] is False
        assert state["last_close_change"] == pytest.approx(period / 1000)
    # Time passing cannot fabricate a closing quote or additional candles.
    later = monitor.snapshot(now=300)
    assert later["5s"]["closed_count"] == 6
    assert later["5s"]["forming"] == snapshot["5s"]["forming"]
    assert later["5s"]["ready"] is False
    assert later["5s"]["fresh"] is False
    assert later["5s"]["status"] == "stale"


def test_forming_movement_and_snapshots_are_detached_from_history():
    monitor = CandleMonitor()
    monitor.add_tick(1.2, 0)
    monitor.add_tick(1.3, 1)
    rising = monitor.snapshot(now=1)["5s"]
    assert rising["observed_direction"] == "up"
    assert rising["change_from_open"] == pytest.approx(0.1)
    assert rising["ready"] is False
    rising["forming"]["open"] = 100
    monitor.add_tick(1.1, 2)
    falling = monitor.snapshot(now=2)["5s"]
    assert falling["observed_direction"] == "down"
    assert falling["forming"]["open"] == 1.2
    assert falling["forming"]["high"] == 1.3
    monitor.add_tick(1.2, 3)
    assert monitor.snapshot(now=3)["5s"]["observed_direction"] == "flat"
    monitor.add_tick(1.2, 5)
    closed = monitor.snapshot(now=5)["5s"]
    closed["latest_closed"]["close"] = 100
    assert monitor.snapshot(now=5)["5s"]["latest_closed"]["close"] == 1.2


def test_startup_partial_candles_are_identified_and_discarded():
    monitor = CandleMonitor()
    monitor.add_tick(1.1, 2.25)
    assert all(not state["forming"]["complete_from_open"]
               for state in monitor.snapshot(now=2.25).values())
    for ts in range(3, 16):
        monitor.add_tick(1.1, ts)
    state = monitor.snapshot(now=15)
    assert state["5s"]["closed_count"] == 2
    assert state["10s"]["closed_count"] == 0
    assert state["15s"]["closed_count"] == 0
    assert all(item["forming"]["complete_from_open"] for item in state.values())
    for ts in range(16, 31):
        monitor.add_tick(1.1, ts)
    assert monitor.snapshot(now=30)["15s"]["latest_closed"]["open_ts"] == 15


def test_gap_discards_history_without_filling_missing_candles():
    monitor = CandleMonitor()
    for ts in range(31):
        monitor.add_tick(1.1, ts)
    assert monitor.add_tick(1.3, 42) == set()
    after_gap = monitor.snapshot(now=42)
    assert all(state["closed_count"] == 0 for state in after_gap.values())
    assert all(not state["forming"]["complete_from_open"]
               for state in after_gap.values())
    for ts in range(43, 61):
        monitor.add_tick(1.3, ts)
    assert monitor.snapshot(now=60)["15s"]["closed_count"] == 1
    assert monitor.snapshot(now=60)["15s"]["latest_closed"]["open_ts"] == 45


@pytest.mark.parametrize("price, ts", [
    (99, 0), (99, 1), (99, 0.5), (0, 2), (-1, 2), (float("nan"), 2),
    (float("inf"), 2), (1.1, float("nan")), (1.1, float("inf")), (None, 2),
])
def test_invalid_duplicate_and_out_of_order_ticks_do_not_mutate_state(price, ts):
    monitor = CandleMonitor()
    monitor.add_tick(1.1, 0)
    monitor.add_tick(1.2, 1)
    before = monitor.snapshot(now=1)
    assert monitor.add_tick(price, ts) == set()
    assert monitor.snapshot(now=1) == before


def test_invalidation_resets_all_observation_and_freshness_state():
    monitor = CandleMonitor()
    for ts in range(31):
        monitor.add_tick(1.1, ts)
    monitor.invalidate("asset_changed")
    for state in monitor.snapshot(now=31).values():
        assert state["closed_count"] == 0
        assert state["forming"] is None
        assert state["latest_closed"] is None
        assert state["last_tick_ts"] is None
        assert state["quote_age_seconds"] is None
        assert state["ready"] is False
        assert state["status"] == "no_quotes"
        assert state["invalidated_reason"] == "asset_changed"


def test_history_is_bounded_and_snapshots_are_strict_json_safe():
    monitor = CandleMonitor(max_candles=2)
    json.dumps(monitor.snapshot(now=0), allow_nan=False)
    for ts in range(61):
        monitor.add_tick(1.1, ts)
    snapshot = monitor.snapshot(now=61)
    assert all(state["closed_count"] == 2 for state in snapshot.values())
    json.dumps(snapshot, allow_nan=False)
    with pytest.raises(ValueError, match="finite"):
        monitor.snapshot(now=float("nan"))


def test_strategy_snapshots_exclude_forming_bars_and_are_detached():
    monitor = CandleMonitor()
    for ts in range(32):
        monitor.add_tick(1.1 + ts * .0001, ts)
    snapshots = monitor.closed_snapshots()
    assert {period: len(bars) for period, bars in snapshots.items()} == {5: 6, 10: 3, 15: 2}
    for period, bars in snapshots.items():
        assert bars[-1]["ts"] == 30 - period
        assert bars[-1]["close"] == pytest.approx(1.1029)
        bars[-1]["close"] = 99
    assert all(bars[-1]["close"] == pytest.approx(1.1029)
               for bars in monitor.closed_snapshots().values())


def test_monitor_can_restore_historical_warmup_without_drop_to_zero():
    monitor = CandleMonitor()
    for ts in range(31):
        monitor.add_tick(1.1 + ts * 0.0005, ts)
    snapshot = monitor.snapshot(now=30)
    restored = CandleMonitor()
    restored.restore_from_snapshot(snapshot)
    assert restored.snapshot(now=30)["10s"]["closed_count"] == 3
    assert restored.snapshot(now=30)["15s"]["closed_count"] == 2
    assert restored.snapshot(now=30)["5s"]["status"] == "ready"


def test_monitor_restore_keeps_the_active_forming_bar_for_continuity():
    monitor = CandleMonitor()
    for ts in range(1, 12):
        monitor.add_tick(1.1 + ts * 0.0005, ts)

    snapshot = monitor.snapshot(now=11)
    restored = CandleMonitor()
    restored.restore_from_snapshot(snapshot)

    restored_state = restored.snapshot(now=11)["5s"]
    assert restored_state["forming"] is not None
    assert restored_state["forming"]["ts"] == snapshot["5s"]["forming"]["ts"]
    assert restored_state["forming"]["close"] == pytest.approx(snapshot["5s"]["forming"]["close"])
    assert restored_state["forming"]["open_ts"] == snapshot["5s"]["forming"]["open_ts"]


def test_bootstrap_allows_live_history_to_fill_missing_warmup():
    missing = MultiAgentTradingSystem._missing_history_for_bootstrap({5: 0, 10: 0, 15: 0})
    assert missing == {5: 10, 10: 15, 15: 15}
    assert MultiAgentTradingSystem._should_block_startup(missing) is False

    partial = MultiAgentTradingSystem._missing_history_for_bootstrap({5: 6, 10: 8, 15: 0})
    assert partial == {5: 4, 10: 7, 15: 15}
    assert MultiAgentTradingSystem._should_block_startup(partial) is False


def test_restore_closed_history_does_not_crash_when_logging_is_enabled(tmp_path):
    monitor = CandleMonitor(periods=(5, 10, 15))
    for ts in range(31):
        monitor.add_tick(1.1 + ts * 0.0005, ts)

    db_path = tmp_path / "candles.sqlite3"
    monitor.persist_closed_history(path=str(db_path), asset="EUR/USD")

    restored = CandleMonitor(periods=(5, 10, 15))
    count = restored.restore_closed_history(path=str(db_path), asset="EUR/USD")

    assert count > 0
    assert restored.snapshot(now=30)["5s"]["closed_count"] == 6
