"""Additive pattern-aware entry scoring without replacing the existing gate logic.

This module is intentionally conservative: it only augments the current rule set
with a weighted pattern score. The default runtime behavior stays the same unless a
caller opts into the pattern gate.
"""

import math
from collections.abc import Sequence

from strategies.closed_candle import confirmed_swings, ema, rsi_series


def _mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _line_slope(points):
    if len(points) < 2:
        return 0.0
    xs = [index for index, _ in enumerate(points)]
    ys = [float(value) for value in points]
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    return 0.0 if den == 0 else num / den


def _swing_recent(candles, kind):
    points = confirmed_swings(candles)
    bucket = points[1] if kind == "low" else points[0]
    return bucket[-3:]


def _candle_range(candle):
    try:
        high = float(candle["high"])
        low = float(candle["low"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return max(0.0, high - low)


def _apply_updated_strategy_filters(trend_candles, entry_candles):
    """Additive pre-signal checks based on the updated strategy overlay.

    These checks do not replace the original logic: they simply suppress spike or
    weak-regime entries until the bar has closed and the setup is clean enough.
    """
    if not trend_candles or not entry_candles:
        return False, "insufficient_data"

    entry = list(entry_candles)
    entry_last = entry[-1]
    if entry_last.get("closed") is False or entry_last.get("complete_from_open") is False:
        return False, "wait_for_close"
    if entry_last.get("closed") is not None and entry_last.get("closed") is not True:
        return False, "wait_for_close"

    ranges = [_candle_range(item) for item in entry[-10:]]
    if len(ranges) >= 3:
        avg_range = sum(ranges[:-1]) / (len(ranges) - 1)
        if avg_range > 0 and ranges[-1] > 2.0 * avg_range:
            return False, "skip_spike"

    return True, "filters_pass"


def _detect_pattern(trend_candles, entry_candles, direction):
    if not trend_candles or not entry_candles:
        return None, 0.0, None
    trend = list(trend_candles)
    entry = list(entry_candles)
    close = float(entry[-1]["close"])
    prev_close = float(entry[-2]["close"]) if len(entry) >= 2 else close
    recent_trend = [float(item["close"]) for item in trend[-5:]]
    recent_entry = [float(item["close"]) for item in entry[-5:]]
    recent_highs = [float(item["high"]) for item in entry[-6:]]
    recent_lows = [float(item["low"]) for item in entry[-6:]]
    if max(recent_trend) - min(recent_trend) < 0.2:
        return None, 0.0, None
    if direction == "BUY":
        if len(recent_trend) >= 5 and all(recent_trend[index] > recent_trend[index - 1] for index in range(1, len(recent_trend))) and close > max(recent_entry[:-1]) and close > prev_close:
            return "bull_flag", 0.72, max(recent_entry)
        if len(recent_highs) >= 4 and all(recent_highs[index] < recent_highs[index + 1] for index in range(len(recent_highs) - 1)) and close > max(recent_highs[:-1]):
            return "higher_low_breakout", 0.68, max(recent_highs)
        if len(entry) >= 8:
            highs = [float(item["high"]) for item in entry[-8:]]
            lows = [float(item["low"]) for item in entry[-8:]]
            if close > max(highs[:-2]) and close > prev_close and min(lows[:4]) < min(lows[4:]):
                return "double_bottom", 0.60, _mean(lows[-3:])
        if len(entry) >= 6 and close > max(float(item["high"]) for item in entry[-6:]) and close > prev_close:
            return "breakout_retest", 0.66, max(float(item["high"]) for item in entry[-6:])
        if len(recent_trend) >= 5 and all(recent_trend[index] > recent_trend[index - 1] for index in range(1, len(recent_trend))) and close > max(float(item["high"]) for item in entry[-5:]):
            return "ascending_triangle", 0.68, max(float(item["high"]) for item in entry[-5:])
    else:
        if len(recent_highs) >= 4 and all(recent_highs[index] > recent_highs[index + 1] for index in range(len(recent_highs) - 1)) and close < min(recent_entry[:-1]) and close < prev_close:
            return "lower_high_breakdown", 0.75, min(recent_lows)
        if len(recent_trend) >= 5 and all(recent_trend[index] < recent_trend[index - 1] for index in range(1, len(recent_trend))) and close < min(recent_entry[:-1]) and close < prev_close:
            return "bear_flag", 0.72, min(recent_entry)
        if len(entry) >= 8:
            highs = [float(item["high"]) for item in entry[-8:]]
            lows = [float(item["low"]) for item in entry[-8:]]
            if close < min(lows[:-2]) and close < prev_close and max(highs[:4]) > max(highs[4:]):
                return "double_top", 0.60, _mean(highs[-3:])
        if len(entry) >= 6 and close < min(float(item["low"]) for item in entry[-6:]) and close < prev_close:
            return "breakout_retest", 0.66, min(float(item["low"]) for item in entry[-6:])
        if len(recent_trend) >= 5 and all(recent_trend[index] < recent_trend[index - 1] for index in range(1, len(recent_trend))) and close < min(float(item["low"]) for item in entry[-5:]):
            return "descending_triangle", 0.68, min(float(item["low"]) for item in entry[-5:])
    return None, 0.0, None


def evaluate_pattern_gate(trend_candles: Sequence[dict], entry_candles: Sequence[dict]) -> dict:
    """Return a pattern score for BUY and SELL using the current closed candles.

    The score is intentionally additive and designed as a lightweight overlay: it
    does not replace or mutate the strict gate in closed_candle.py unless the
    caller opts into the pattern path.
    """
    trend = list(trend_candles)
    entry = list(entry_candles)
    if len(trend) < 12 or len(entry) < 8:
        return {
            "BUY": {"passed": False, "score": 0, "pattern": None, "direction": "BUY", "confidence": 0.0, "trigger_price": None, "regime": "insufficient_data", "filter": "insufficient_data"},
            "SELL": {"passed": False, "score": 0, "pattern": None, "direction": "SELL", "confidence": 0.0, "trigger_price": None, "regime": "insufficient_data", "filter": "insufficient_data"},
        }

    filter_ok, filter_reason = _apply_updated_strategy_filters(trend, entry)
    if not filter_ok:
        return {
            "BUY": {"passed": False, "score": 0, "pattern": None, "direction": "BUY", "confidence": 0.0, "trigger_price": None, "regime": "filtered", "filter": filter_reason},
            "SELL": {"passed": False, "score": 0, "pattern": None, "direction": "SELL", "confidence": 0.0, "trigger_price": None, "regime": "filtered", "filter": filter_reason},
        }

    t20 = ema([float(candle["close"]) for candle in trend], 20)
    e9 = ema([float(candle["close"]) for candle in entry], 9)
    e21 = ema([float(candle["close"]) for candle in entry], 21)
    rsi = rsi_series([float(candle["close"]) for candle in entry], 14)
    current_rsi = rsi[-1] if rsi and rsi[-1] is not None else 50.0
    results = {}
    for direction, sign in (("BUY", 1), ("SELL", -1)):
        close = float(entry[-1]["close"])
        prev_close = float(entry[-2]["close"]) if len(entry) >= 2 else close
        score = 0
        pattern, confidence, trigger_price = _detect_pattern(trend, entry, direction)
        regime = "trend" if sign * (t20[-1] - t20[-4]) > 0 else "range"

        if sign * (close - e21[-1]) > 0:
            score += 1
        if sign * (close - prev_close) > 0:
            score += 1
        if sign * (e9[-1] - e21[-1]) > 0:
            score += 1
        if regime == "trend" and sign * (current_rsi - 50.0) > 0:
            score += 1
        if pattern is not None:
            score += 2
            score += int(min(2, max(0, confidence * 3)))
        if direction == "BUY" and (close > max(float(candle["high"]) for candle in entry[-5:])):
            score += 1
        if direction == "SELL" and (close < min(float(candle["low"]) for candle in entry[-5:])):
            score += 1

        score = min(score, 8)
        trend_strength = sign * (float(entry[-1]["close"]) - float(entry[0]["close"]))
        direction_match = (direction == "BUY" and trend_strength > 0) or (direction == "SELL" and trend_strength < 0)
        passed = (score >= 5 and pattern is not None) or (score >= 5 and direction_match and sign * (close - prev_close) > 0)
        results[direction] = {
            "passed": bool(passed),
            "score": score,
            "pattern": pattern,
            "direction": direction,
            "confidence": min(0.98, max(0.0, confidence or (score / 8.0))),
            "trigger_price": trigger_price,
            "regime": regime,
            "rsi": current_rsi,
        }
    return results
