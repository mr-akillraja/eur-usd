from strategies.pattern_gate import evaluate_pattern_gate


def make_candle(ts, open_price, close_price, high, low):
    return {
        "ts": ts,
        "open": open_price,
        "close": close_price,
        "high": high,
        "low": low,
    }


def build_bull_flag():
    bars = []
    price = 100.0
    ts = 1000
    for index in range(30):
        if index < 12:
            price += 0.6
        elif index < 18:
            price += 0.2
        else:
            price += 0.1
        open_price = price - 0.3
        close_price = price + 0.25
        high = max(open_price, close_price) + 0.5
        low = min(open_price, close_price) - 0.5
        bars.append(make_candle(ts + index, open_price, close_price, high, low))
    return bars


def build_flat_noise():
    bars = []
    price = 100.0
    ts = 1000
    for index in range(30):
        drift = 0.02 * (index % 5 - 2)
        open_price = price + drift
        close_price = price + drift * 0.5
        high = max(open_price, close_price) + 0.1
        low = min(open_price, close_price) - 0.1
        bars.append(make_candle(ts + index, open_price, close_price, high, low))
    return bars


def test_pattern_gate_detects_bull_flag_breakout():
    trend = build_bull_flag()
    entry = trend[-15:]
    result = evaluate_pattern_gate(trend, entry)
    assert result["BUY"]["passed"] is True
    assert result["BUY"]["score"] >= 5
    assert result["BUY"]["pattern"] in {"bull_flag", "breakout_retest", "ascending_triangle"}


def test_pattern_gate_rejects_flat_noise():
    trend = build_flat_noise()
    entry = trend[-15:]
    result = evaluate_pattern_gate(trend, entry)
    assert result["BUY"]["score"] < 5
    assert result["SELL"]["score"] < 5


def test_pattern_gate_detects_lower_high_breakdown_sell():
    trend = []
    price = 106.0
    for index in range(28):
        price = price - 0.12 if index % 3 else price - 0.04
        trend.append(make_candle(1000 + index, price, price - 0.04, price + 0.6, price - 0.8))
    entry = trend[-15:]
    for i in range(len(entry) - 1):
        entry[i]["high"] = entry[i]["high"] + 0.2
    result = evaluate_pattern_gate(trend, entry)
    assert result["SELL"]["passed"] is True
    assert result["SELL"]["pattern"] in {"lower_high_breakdown", "double_top", "breakout_retest", "descending_triangle"}


def test_pattern_gate_rejects_spike_candles_and_requires_closed_1m_bar():
    trend = []
    price = 100.0
    for index in range(25):
        price += 0.08 if index % 2 else -0.04
        trend.append(make_candle(1000 + index, price, price + 0.05, price + 0.12, price - 0.10))

    entry = trend[-12:]
    last = entry[-1]
    last["high"] = last["high"] + 1.20
    last["low"] = last["low"] - 1.20
    last["close"] = last["open"] + 0.30
    last["closed"] = False
    last["complete_from_open"] = False

    result = evaluate_pattern_gate(trend, entry)
    assert result["BUY"]["passed"] is False
    assert result["BUY"]["filter"] in {"skip_spike", "wait_for_close"}
