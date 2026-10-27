"""Regression tests for provenance, candle finalization, and real-data warmup."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agents.agent_multi_timeframe import CandleBuilder, MultiTimeframeAgent, _rsi
from data import fetcher


def history(count=180, period=60, start=0):
    frame = pd.DataFrame({"open": np.full(count, 1.1), "high": np.full(count, 1.2),
                          "low": np.full(count, 1.0), "close": np.full(count, 1.15)},
                         index=pd.to_datetime(np.arange(count) * period + start, unit="s", utc=True))
    frame.attrs["timeframe_seconds"] = period
    return frame


def test_m1_seed_never_populates_30s_and_preserves_timestamps(caplog):
    agent = MultiTimeframeAgent("EUR/USD OTC")
    frame = history()
    with caplog.at_level("INFO"):
        agent.seed_from_dataframe(frame, source="verified_test_history", now=10800)
    assert len(agent.builders[30]) == 0
    assert len(agent.builders[60]) == 70
    assert agent.builders[60].latest_closed_ts == frame.index[-1].timestamp()
    assert len(agent.builders[300]) == 36
    assert "timeframe=30s source=broker_browser_sample closed_count=0" in caplog.text
    assert "timeframe=60s source=verified_test_history closed_count=70" in caplog.text


def test_native_30s_seed_requires_matching_resolution():
    agent = MultiTimeframeAgent("EUR/USD OTC")
    with pytest.raises(ValueError, match="resolution"):
        agent.seed_from_dataframe(history(), timeframe=30)
    frame = history(50, period=30)
    agent.seed_from_dataframe(frame, timeframe=30, source="broker_native_fixture", now=1500)
    assert len(agent.builders[30]) == 50
    assert len(agent.builders[60]) == 0
    assert agent.builders[30].latest_closed_ts == 1470


def test_seed_rejects_untimestamped_and_unclosed_bars():
    agent = MultiTimeframeAgent("test")
    with pytest.raises(ValueError, match="DatetimeIndex"):
        agent.seed_from_dataframe(history().reset_index(drop=True))
    agent.seed_from_dataframe(history(3), now=125)
    assert agent.builders[60].get_closes().size == 2
    assert agent.builders[60].latest_closed_ts == 60
    assert len(agent.builders[300]) == 0


def test_five_minute_seed_requires_complete_aligned_minutes():
    agent = MultiTimeframeAgent("test")
    agent.seed_from_dataframe(history(12, start=60), now=780)
    bars = agent.builders[300].closed_snapshot()
    assert [bar["ts"] for bar in bars] == [300]


def test_seed_uses_only_contiguous_tail_after_missing_bar():
    agent = MultiTimeframeAgent("test")
    frame = history(10).drop(pd.Timestamp(300, unit="s", tz="UTC"))
    agent.seed_from_dataframe(frame, now=600)
    assert [bar["ts"] for bar in agent.builders[60].closed_snapshot()] == [360, 420, 480, 540]


def test_forming_candle_is_invisible_and_closed_snapshot_is_detached():
    builder = CandleBuilder(30)
    for ts in range(30):
        assert builder.add_tick(1 + ts / 1000, ts) is False
    assert len(builder) == 0
    assert builder.get_closes().size == 0
    assert builder.add_tick(9, 30) is True
    assert len(builder) == 1
    assert builder.get_opens().tolist() == [1]
    assert builder.get_closes().tolist() == [1.029]
    assert builder.get_highs().tolist() == [1.029]
    assert builder.get_lows().tolist() == [1]
    snapshot = builder.closed_snapshot()
    snapshot[0]["close"] = 100
    assert builder.get_closes().tolist() == [1.029]


def test_startup_partial_bar_is_dropped():
    builder = CandleBuilder(30)
    for ts in range(17, 60):
        assert builder.add_tick(1.1, ts) is False
    assert builder.add_tick(1.2, 60) is True
    assert [bar["ts"] for bar in builder.closed_snapshot()] == [30]


def test_quote_gap_clears_history_and_requires_a_new_complete_bar():
    builder = CandleBuilder(30)
    for ts in range(61):
        builder.add_tick(1.1, ts)
    assert len(builder) == 2
    assert builder.add_tick(1.2, 70) is False
    assert len(builder) == 0
    for ts in range(71, 120):
        assert builder.add_tick(1.2, ts) is False
    assert builder.add_tick(1.3, 120) is True
    assert builder.latest_closed_ts == 90


def test_invalid_and_out_of_order_quotes_cannot_mutate_a_bar():
    builder = CandleBuilder(30)
    builder.add_tick(1.1, 0)
    builder.add_tick(1.2, 1)
    for price, ts in [(99, 0.5), (99, 1), (float("nan"), 2), (-2, 2), (2, float("inf"))]:
        assert builder.add_tick(price, ts) is False
    for ts in range(2, 31):
        builder.add_tick(1.1, ts)
    assert builder.get_highs().tolist() == [1.2]


def test_source_handoff_between_native_and_fallback_keeps_active_bar():
    builder = CandleBuilder(30)
    builder.add_tick(1.0, 0, source="broker_native_ws")
    builder.add_tick(1.01, 1, source="fallback")
    assert builder._current is not None
    assert builder._current["tick_count"] == 2
    builder.add_tick(1.02, 2, source="broker_native_ws")
    assert builder._current is not None
    assert builder._current["tick_count"] == 3
    assert builder.source == "broker_native_ws"


def test_live_warmup_requires_fifty_real_completed_30s_intervals():
    agent = MultiTimeframeAgent("EUR/USD OTC")
    closed_events = []
    for ts in range(1501):
        periods = agent.add_tick(1.1, ts)
        if 30 in periods:
            closed_events.append(ts)
        if ts == 1499:
            assert len(agent.builders[30]) == 49
    assert closed_events == list(range(30, 1501, 30))
    assert len(agent.builders[30]) == 50
    assert len(agent.builders[60]) == 25
    assert agent.builders[30].closed_snapshot()[-1]["source"] == "broker_browser_sample"


def test_compatibility_bootstrap_does_not_invent_history():
    agent = MultiTimeframeAgent("test")
    agent.bootstrap(1.1)
    assert all(len(builder) == 0 for builder in agent.builders.values())


def test_rsi_averages_every_change_and_uses_wilder_smoothing():
    closes = np.r_[1.0, 1.0 + np.cumsum([1] * 13 + [-1])]
    assert _rsi(closes) == pytest.approx(100 * 13 / 14)
    # One subsequent loss smooths both previous averages with Wilder's 1/14.
    assert _rsi(np.r_[closes, closes[-1] - 1]) == pytest.approx(100 * 169 / 196)
    assert _rsi(np.arange(15, dtype=float)) == 100
    assert _rsi(np.arange(15, 0, -1, dtype=float)) == 0
    assert _rsi(np.ones(15)) == 50
    assert np.isnan(_rsi(np.ones(14)))


def test_yahoo_rejects_otc_and_non_native_intervals_without_network(monkeypatch):
    def fail_if_requested(*args, **kwargs):
        raise AssertionError("unsupported data must not trigger a network request")
    monkeypatch.setattr(fetcher, "yf", SimpleNamespace(Ticker=fail_if_requested), raising=False)
    monkeypatch.setattr(fetcher, "_YF_AVAILABLE", True)
    for asset, interval in [("EUR/USD OTC", "M1"), ("EURUSD-OTC", 60), ("EURUSD", 30), ("EURUSD", "30s")]:
        with pytest.raises(fetcher.MarketDataUnavailable):
            fetcher.fetch_candles(asset, interval)


def test_empty_yahoo_response_fails_without_synthetic_fallback(monkeypatch):
    monkeypatch.setattr(fetcher, "yf", SimpleNamespace(Ticker=lambda _: SimpleNamespace(history=lambda **kw: pd.DataFrame())), raising=False)
    monkeypatch.setattr(fetcher, "_YF_AVAILABLE", True)
    fetcher._cache.clear()
    with pytest.raises(fetcher.MarketDataUnavailable, match="no 1m history"):
        fetcher.fetch_candles("EURUSD", "M1")


def test_yahoo_errors_fail_without_synthetic_fallback(monkeypatch):
    def fail(**kwargs):
        raise ConnectionError("test outage")
    monkeypatch.setattr(fetcher, "yf", SimpleNamespace(Ticker=lambda _: SimpleNamespace(history=fail)), raising=False)
    monkeypatch.setattr(fetcher, "_YF_AVAILABLE", True)
    fetcher._cache.clear()
    with pytest.raises(fetcher.MarketDataUnavailable, match="test outage"):
        fetcher.fetch_candles("EURUSD", "M1")
