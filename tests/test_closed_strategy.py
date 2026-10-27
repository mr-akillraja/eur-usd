"""Regression checks use explicit fixtures, never broker/live data."""
import copy
import json
import math

import pytest

from strategies.closed_candle import confirmed_swings, ema, evaluate_both, rsi_series


def candles(closes, period=60):
    return [dict(open=c, high=c + .15, low=c - .15, close=c,
                 ts=2400 - len(closes) * period + index * period)
            for index, c in enumerate(closes)]


def buy_fixture():
    trend = candles([101 + index * .05 for index in range(60)], 30)
    entry = candles([100 + 5 * math.sin(index * math.pi / 3) for index in range(30)]
                    + [101, 102, 101.8, 102.5, 103, 102.7, 102.9, 103.5, 103.2, 104])
    entry[35]["low"] = 101.8
    return trend, entry


def mirror(bars):
    return [dict(open=210 - c["open"], high=210 - c["low"], low=210 - c["high"],
                 close=210 - c["close"], ts=c["ts"]) for c in bars]


def test_complete_buy_and_sell_setups_pass_with_original_bands():
    trend, entry = buy_fixture()
    result = evaluate_both(trend, entry)
    assert result["BUY"]["passed"], result["BUY"]["failed_rules"]
    assert not result["SELL"]["passed"]
    assert 50 <= result["BUY"]["rsi"] <= 65
    result = evaluate_both(mirror(trend), mirror(entry))
    assert result["SELL"]["passed"], result["SELL"]["failed_rules"]
    assert not result["BUY"]["passed"]
    assert 35 <= result["SELL"]["rsi"] <= 50


def test_wick_beyond_ema21_does_not_claim_a_close_beyond_ema21():
    trend, entry = buy_fixture()
    entry[35]["low"] = 100.0
    result = evaluate_both(trend, entry)["BUY"]
    assert entry[35]["low"] < result["pullback_closes_ema21"][0]["ema21"]
    assert result["passed"], result["failed_rules"]
    result = evaluate_both(mirror(trend), mirror(entry))["SELL"]
    assert result["passed"], result["failed_rules"]


def test_real_pullback_close_beyond_ema21_rejects_both_directions():
    trend, entry = buy_fixture()
    entry[-3].update(open=99.0, high=99.1, low=98.9, close=99.0)
    assert "1m_pullback_close_vs_ema21" in evaluate_both(trend, entry)["BUY"]["failed_rules"]
    result = evaluate_both(mirror(trend), mirror(entry))["SELL"]
    assert "1m_pullback_close_vs_ema21" in result["failed_rules"]


def test_pullback_uses_each_candles_own_ema21():
    trend, _ = buy_fixture()
    entry = candles([100 + index * .1 for index in range(39)] + [130.0])
    result = evaluate_both(trend, entry)["BUY"]
    assert entry[-5]["close"] < result["ema21"]
    assert "1m_pullback_close_vs_ema21" not in result["failed_rules"]


def test_ema_three_bars_ago_is_three_intervals_not_two():
    trend, entry = buy_fixture()
    result = evaluate_both(trend, entry)["BUY"]
    values = [c["close"] for c in trend]
    assert result["ema20_3ago"] == ema(values, 20)[-4]
    assert result["ema50_3ago"] == ema(values, 50)[-4]
    assert result["ema20_3ago"] != ema(values, 20)[-3]


def test_warmup_waits_for_mature_ema50_at_both_slope_endpoints():
    trend, entry = buy_fixture()
    result = evaluate_both(trend[:52], entry)
    assert all(gate["status"] == "insufficient_data" for gate in result.values())
    assert result["BUY"]["ema50_3ago"] is None
    assert evaluate_both(trend[:53], entry)["BUY"]["status"] != "insufficient_data"
    assert evaluate_both(trend, entry[:20])["SELL"]["status"] == "insufficient_data"
    assert evaluate_both([], [])["BUY"]["rsi"] is None


def test_wilder_rsi_counts_zero_losses_and_zero_gains_in_period():
    prices = [100 + index for index in range(14)] + [112]
    assert rsi_series(prices)[-1] == pytest.approx(100 * 13 / 14)
    assert rsi_series([100] * 15)[-1] == 50
    assert rsi_series(list(range(20)))[-1] == 100
    assert rsi_series(list(range(20, 0, -1)))[-1] == 0


def test_pivot_requires_two_completed_right_bars():
    bars = candles([10, 11, 13, 11, 10])
    assert not confirmed_swings(bars[:-1])[0]
    highs, _ = confirmed_swings(bars)
    assert len(highs) == 1 and highs[0]["index"] == 2
    assert highs[0]["ts"] == bars[2]["ts"]


def test_confirmation_range_compares_with_previous_ten_only():
    trend, entry = buy_fixture()
    previous_average = sum(c["high"] - c["low"] for c in entry[-11:-1]) / 10
    entry[-1]["high"] = entry[-1]["close"] + 2.0
    result = evaluate_both(trend, entry)["BUY"]
    assert result["avg_prev_10_range"] == pytest.approx(previous_average)
    assert "1m_confirmation_range" in result["failed_rules"]


def test_obstacle_uses_distance_and_preserves_different_side_thresholds():
    trend, entry = buy_fixture()
    trend[20]["high"] = 104.4
    result = evaluate_both(trend, entry)["BUY"]
    assert result["nearest_resistance"] == 104.4
    assert result["resistance_distance"] == pytest.approx(.4)
    assert "30s_obstacle_clearance" in result["failed_rules"]  # .4 < 1.5 * .3
    result = evaluate_both(mirror(trend), mirror(entry))["SELL"]
    assert result["support_distance"] == pytest.approx(.4)
    assert result["passed"]  # .4 > existing SELL clearance of 1.0 * .3
    trend[20]["high"] = 104.6
    assert evaluate_both(trend, entry)["BUY"]["passed"]


def test_sell_retains_existing_descending_candle_fallback():
    trend = candles([110 - index * .05 for index in range(60)], 30)
    entry = candles([110 - index * .5 for index in range(40)])
    result = evaluate_both(trend, entry)["SELL"]
    assert not result["entry_swing_highs"] and not result["entry_swing_lows"]
    assert "1m_confirmed_pullback_swings_or_descending_highs" not in result["failed_rules"]
    assert "1m_confirmed_breakout_swing_or_descending_lows" not in result["failed_rules"]
    assert result["failed_rules"] == ["1m_rsi14_band", "1m_rsi14_direction"]


def test_diagnostics_are_complete_even_when_first_rule_fails_and_json_safe():
    trend, entry = buy_fixture()
    original = copy.deepcopy((trend, entry))
    result = evaluate_both(trend, entry)
    sell = result["SELL"]
    assert sell["reason"] == "30s_close_vs_ema20"
    assert sell["ema9"] is not None and sell["rsi"] is not None
    assert len(sell["failed_rules"]) > 1
    assert sell["trend_ts"] == result["BUY"]["trend_ts"]
    assert sell["entry_ts"] == result["BUY"]["entry_ts"]
    json.dumps(result, allow_nan=False)
    json.dumps(evaluate_both([], []), allow_nan=False)
    assert (trend, entry) == original


@pytest.mark.parametrize("bad", ["duplicate_timestamp", "nan", "invalid_range"])
def test_invalid_input_fails_closed(bad):
    trend, entry = buy_fixture()
    if bad == "duplicate_timestamp":
        entry[-1]["ts"] = entry[-2]["ts"]
    elif bad == "nan":
        entry[-1]["close"] = float("nan")
    else:
        entry[-1]["high"] = entry[-1]["low"] - 1
    result = evaluate_both(trend, entry)
    assert all(gate["status"] == "invalid_data" for gate in result.values())
    json.dumps(result, allow_nan=False)
