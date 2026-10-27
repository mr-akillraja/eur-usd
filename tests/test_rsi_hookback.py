import multi_agent_orchestrator as runtime


def make_system():
    system = runtime.MultiAgentTradingSystem.__new__(runtime.MultiAgentTradingSystem)
    system._rsi_buy_armed = True
    system._rsi_sell_armed = True
    system._last_rsi_signal = None
    return system


def test_rsi_hookback_fires_once_until_reset():
    system = make_system()

    first = system._evaluate_rsi_hookback({"ts": 100, "close": 1.0}, 34.0, 36.0)
    assert first["signal"] == "BUY"
    assert system._rsi_buy_armed is False

    second = system._evaluate_rsi_hookback({"ts": 200, "close": 1.0}, 36.0, 38.0)
    assert second["signal"] is None
    assert system._rsi_buy_armed is False

    rearmed = system._evaluate_rsi_hookback({"ts": 300, "close": 1.0}, 48.0, 52.0)
    assert rearmed["signal"] is None
    assert system._rsi_buy_armed is True


def test_rsi_hookback_sell_uses_high_to_low_cross_and_reset():
    system = make_system()

    first = system._evaluate_rsi_hookback({"ts": 100, "close": 1.0}, 70.0, 62.0)
    assert first["signal"] == "SELL"
    assert system._rsi_sell_armed is False

    second = system._evaluate_rsi_hookback({"ts": 200, "close": 1.0}, 62.0, 60.0)
    assert second["signal"] is None
    assert system._rsi_sell_armed is False

    rearmed = system._evaluate_rsi_hookback({"ts": 300, "close": 1.0}, 52.0, 48.0)
    assert rearmed["signal"] is None
    assert system._rsi_sell_armed is True


def test_rsi_wilder_series_uses_closed_bar_history_only():
    values = [100.0, 101.0, 102.0, 101.5, 100.5, 101.0, 101.5, 102.0, 101.0, 99.5]
    series = runtime.MultiAgentTradingSystem._compute_rsi_wilder(values, period=3)
    assert len(series) == len(values)
    assert series[-1] >= 0.0 and series[-1] <= 100.0
    assert series[0] is None
