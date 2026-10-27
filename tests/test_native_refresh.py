"""Native history refresh integration: no polling ticks masquerade as new bars."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd

import multi_agent_orchestrator as runtime
from agents.agent_multi_timeframe import MultiTimeframeAgent


def frame(period, last):
    ts = list(range(last - period * 59, last + period, period))
    df = pd.DataFrame({"open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15},
                      index=pd.to_datetime(ts, unit="s", utc=True))
    df.attrs.update(source="broker_native_ws", timeframe_seconds=period)
    return df


def make_system():
    s = runtime.MultiAgentTradingSystem.__new__(runtime.MultiAgentTradingSystem)
    s.agent1 = MultiTimeframeAgent("EUR/USD OTC")
    s._current_asset = "EUR/USD OTC"
    s._seed_sources = {30: "broker_native_ws", 60: "broker_native_ws"}
    s._quote = {"observed_at": 3600.0}
    return s


def test_refresh_native_periods_only_when_their_candles_close(monkeypatch):
    s = make_system()
    s.agent1.seed_from_dataframe(frame(30, 3510), timeframe=30, now=3600)
    s.agent1.seed_from_dataframe(frame(60, 3480), timeframe=60, now=3600)
    monkeypatch.setattr(runtime.time, "time", lambda: 3541.0)
    s.bot = SimpleNamespace(get_native_candles=AsyncMock())
    assert asyncio.run(s._update_closed_candles(99.0)) == set()
    s.bot.get_native_candles.assert_not_awaited()
    monkeypatch.setattr(runtime.time, "time", lambda: 3601.0)
    s.bot.get_native_candles.side_effect = [frame(30, 3570), frame(60, 3540)]
    assert asyncio.run(s._update_closed_candles(99.0)) == {30, 60}
    assert s.agent1.builders[30].get_closes()[-1] == 1.15
    assert s.agent1.builders[60].get_closes()[-1] == 1.15
    assert s.agent1.builders[30].latest_closed_ts == 3570
    assert s.agent1.builders[60].latest_closed_ts == 3540
    s.bot.get_native_candles.reset_mock()
    assert asyncio.run(s._update_closed_candles(999.0)) == set()
    s.bot.get_native_candles.assert_not_awaited()


def test_failed_native_refresh_does_not_create_candle_from_title(monkeypatch):
    s = make_system()
    s.agent1.seed_from_dataframe(frame(30, 3510), timeframe=30, now=3600)
    s.agent1.seed_from_dataframe(frame(60, 3480), timeframe=60, now=3600)
    monkeypatch.setattr(runtime.time, "time", lambda: 3601.0)
    s.bot = SimpleNamespace(get_native_candles=AsyncMock(return_value=None))
    assert asyncio.run(s._update_closed_candles(99.0)) == set()
    assert s.agent1.builders[30].latest_closed_ts == 3510
    assert s.agent1.builders[60].latest_closed_ts == 3480
