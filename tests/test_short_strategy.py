"""Synthetic short-candle setups; no broker connection or order is used here."""

import copy
import json
import math

import pytest

from strategies.closed_candle import ema
from strategies.short_candle import SHORT_MIN_CANDLES, evaluate_short


def candles(closes, period, now=2400.0):
    end = math.floor(now / period) * period
    return [dict(ts=end - len(closes) * period + index * period,
                 open=close, high=close + .15, low=close - .15, close=close,
                 closed=True, complete_from_open=True)
            for index, close in enumerate(closes)]


def buy_fixture(now=2400.0):
    """Return (period->bars, now); valid at the exact minimum history lengths."""
    histories = {
        5: candles([100 + (index % 2) * .8 + index * .02 for index in range(15)]
                   + [100.85], 5, now),
        10: candles([99 + index * .06 for index in range(21)], 10, now),
        15: candles([99 + index * .06 for index in range(21)], 15, now),
    }
    histories[5][-1]["open"] -= .2
    histories[5][-1]["low"] -= .2
    return histories, now


def mirror(histories):
    return {
        period: [{**bar, "open": 210 - bar["open"], "high": 210 - bar["low"],
                  "low": 210 - bar["high"], "close": 210 - bar["close"]}
                 for bar in bars]
        for period, bars in histories.items()
    }


def test_completed_buy_and_mirrored_sell_pass_without_altering_inputs():
    histories, now = buy_fixture()
    original = copy.deepcopy(histories)
    for data, direction, opposite in ((histories, "BUY", "SELL"),
                                      (mirror(histories), "SELL", "BUY")):
        results = evaluate_short(data, now)
        gate = results[direction]
        assert gate["passed"], gate["failed_rules"]
        assert gate["status"] == "pass"
        assert gate["reason"] == "short_setup_confirmed"
        assert not results[opposite]["passed"]
        assert gate["entry_ts"] == now - 5
        assert gate["confirmation_ts"] == now - 10
        assert gate["trend_ts"] == now - 15
        assert gate["entry_period_seconds"] == 5
        assert gate["confirmation_period_seconds"] == 10
        assert gate["trend_period_seconds"] == 15
        assert gate["closed_only"] is True
        assert gate["profile"] == "short"
        json.dumps(results, allow_nan=False)
    assert histories == original


def test_each_period_uses_its_own_latest_completed_boundary():
    histories, now = buy_fixture(now=2407.25)
    result = evaluate_short(histories, now)["BUY"]
    assert result["passed"]
    assert result["entry_ts"] == 2400
    assert result["confirmation_ts"] == 2390
    assert result["trend_ts"] == 2385


@pytest.mark.parametrize("period", [5, 10, 15])
def test_warmup_requires_enough_closed_history_for_each_period(period):
    histories, now = buy_fixture()
    histories[period] = histories[period][-((SHORT_MIN_CANDLES[period] - 1)):]
    results = evaluate_short(histories, now)
    assert all(gate["status"] == "insufficient_data" for gate in results.values())
    assert not any(gate["passed"] for gate in results.values())
    assert results["BUY"]["reason"] == "insufficient_closed_candles"


@pytest.mark.parametrize("period", [5, 10, 15])
def test_missing_period_waits_for_warmup(period):
    histories, now = buy_fixture()
    del histories[period]
    assert evaluate_short(histories, now)["BUY"]["status"] == "insufficient_data"
    assert evaluate_short({}, now)["SELL"]["status"] == "insufficient_data"


def test_trend_slope_uses_three_full_closed_intervals():
    histories, now = buy_fixture()
    result = evaluate_short(histories, now)["BUY"]
    expected = ema([bar["close"] for bar in histories[15]], 9)
    assert result["trend_ema9_3ago"] == expected[-4]
    assert result["trend_ema9_3ago"] != expected[-3]


@pytest.mark.parametrize("rule", [
    "15s_ema9_vs_ema21", "15s_close_vs_ema9", "15s_ema9_slope_3_candles",
    "10s_ema9_vs_ema21", "10s_close_vs_ema9", "5s_directional_body",
    "5s_close_breaks_previous_candle", "5s_rsi14_band", "5s_rsi14_direction",
    "5s_confirmation_range",
])
def test_each_entry_filter_blocks_both_directions_when_violated(rule):
    histories, now = buy_fixture()
    if rule in ("15s_ema9_vs_ema21", "10s_ema9_vs_ema21"):
        period = int(rule.split("s_")[0])
        histories[period] = mirror(histories)[period]
    elif rule in ("15s_close_vs_ema9", "10s_close_vs_ema9"):
        period = int(rule.split("s_")[0])
        histories[period][-1].update(open=90, high=90.15, low=89.85, close=90)
    elif rule == "15s_ema9_slope_3_candles":
        for bar in histories[15][-3:]:
            bar.update(open=95, high=95.15, low=94.85, close=95)
    elif rule == "5s_directional_body":
        histories[5][-1]["open"] = histories[5][-1]["close"]
    elif rule == "5s_close_breaks_previous_candle":
        histories[5][-2]["high"] = histories[5][-1]["close"]
    elif rule == "5s_rsi14_band":
        histories[5] = candles([100 + index * .1 for index in range(16)], 5, now)
    elif rule == "5s_rsi14_direction":
        histories[5][-1].update(open=100.25, high=100.3, low=100, close=100.1)
    elif rule == "5s_confirmation_range":
        histories[5][-1]["high"] += 10
    for data, direction in ((histories, "BUY"), (mirror(histories), "SELL")):
        gate = evaluate_short(data, now)[direction]
        assert not gate["passed"]
        assert rule in gate["failed_rules"]
        assert gate["status"] == "fail"


def test_range_uses_previous_ten_and_rejects_zero_range():
    histories, now = buy_fixture()
    expected = sum(bar["high"] - bar["low"] for bar in histories[5][-11:-1]) / 10
    histories[5][-1]["high"] += 10
    result = evaluate_short(histories, now)["BUY"]
    assert result["avg_prev_10_range"] == pytest.approx(expected)
    assert "5s_confirmation_range" in result["failed_rules"]
    close = histories[5][-1]["close"]
    histories[5][-1].update(open=close, high=close, low=close)
    assert "5s_confirmation_range" in evaluate_short(histories, now)["BUY"]["failed_rules"]


@pytest.mark.parametrize("period", [5, 10, 15])
@pytest.mark.parametrize("mutation,reason", [
    ("stale", "stale_closed_candles"),
    ("future", "future_or_forming_closed_candle"),
    ("unaligned", "unaligned_candle_timestamps"),
    ("gap", "noncontiguous_closed_candles"),
    ("duplicate", "noncontiguous_closed_candles"),
    ("reversed", "noncontiguous_closed_candles"),
])
def test_timestamp_errors_block_entries_even_when_other_periods_pass(period, mutation, reason):
    histories, now = buy_fixture()
    bars = histories[period]
    if mutation == "stale":
        for bar in bars:
            bar["ts"] -= period
    elif mutation == "future":
        bars[-1]["ts"] += period
    elif mutation == "unaligned":
        bars[0]["ts"] += 1
    elif mutation == "gap":
        del bars[3]
    elif mutation == "duplicate":
        bars[4]["ts"] = bars[3]["ts"]
    else:
        bars.reverse()
    results = evaluate_short(histories, now)
    assert all(not gate["passed"] and gate["reason"] == reason for gate in results.values())
    json.dumps(results, allow_nan=False)


@pytest.mark.parametrize("period", [5, 10, 15])
@pytest.mark.parametrize("flag", ["closed", "complete_from_open"])
@pytest.mark.parametrize("value", [False, None, 0, "true"])
def test_incomplete_or_ambiguous_metadata_never_allows_an_entry(period, flag, value):
    histories, now = buy_fixture()
    histories[period][0][flag] = value
    results = evaluate_short(histories, now)
    assert all(gate["reason"] == "incomplete_closed_candle" for gate in results.values())
    assert not any(gate["passed"] for gate in results.values())


def test_completion_flags_are_optional_for_already_completed_caller_histories():
    histories, now = buy_fixture()
    for bars in histories.values():
        for bar in bars:
            del bar["closed"]
            del bar["complete_from_open"]
    assert evaluate_short(histories, now)["BUY"]["passed"]


@pytest.mark.parametrize("bad", [None, True, "bad", [], {}, -1, float("nan"), float("inf")])
def test_invalid_now_fails_closed_and_is_json_safe(bad):
    histories, _ = buy_fixture()
    result = evaluate_short(histories, bad)
    assert all(gate["reason"] == "invalid_current_timestamp" for gate in result.values())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [None, True, "bad", [], 42])
def test_malformed_history_container_fails_closed(bad):
    results = evaluate_short(bad, 2400)
    assert all(gate["status"] == "invalid_data" for gate in results.values())
    histories, now = buy_fixture()
    histories[10] = bad
    results = evaluate_short(histories, now)
    assert not any(gate["passed"] for gate in results.values())
    json.dumps(results, allow_nan=False)


@pytest.mark.parametrize("bad", [None, [], "bad", {},
    {"close": float("nan")}, {"high": float("inf")}, {"low": 0},
    {"close": True}, {"open": "bad"}, {"ts": -15}, {"high": 1}, {"low": 500},
])
def test_malformed_candles_fail_closed_without_leaking_nonfinite_values(bad):
    histories, now = buy_fixture()
    if isinstance(bad, dict) and bad:
        histories[15][-1].update(bad)
    else:
        histories[15][-1] = bad
    results = evaluate_short(histories, now)
    assert all(gate["status"] == "invalid_data" for gate in results.values())
    json.dumps(results, allow_nan=False)


def test_extreme_finite_prices_fail_closed_if_indicator_math_overflows():
    histories, now = buy_fixture()
    for index, bar in enumerate(histories[5]):
        value = 1e308 if index % 2 else 1.0
        bar.update(open=value, high=value, low=value, close=value)
    results = evaluate_short(histories, now)
    assert all(gate["reason"] == "invalid_indicator_values" for gate in results.values())
    json.dumps(results, allow_nan=False)


def test_all_failures_are_available_when_first_trend_filter_fails():
    histories, now = buy_fixture()
    gate = evaluate_short(histories, now)["SELL"]
    assert gate["reason"] == "15s_ema9_vs_ema21"
    assert len(gate["failed_rules"]) > 1
    assert gate["rsi"] is not None
    assert gate["confirmation_ema21"] is not None
    assert {5: gate["entry_required"], 10: gate["confirmation_required"],
            15: gate["trend_required"]} == SHORT_MIN_CANDLES
