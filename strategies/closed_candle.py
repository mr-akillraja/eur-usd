"""Deterministic two-direction gates over *completed* broker candles.

The caller supplies chronological OHLC snapshots and must exclude the forming
candle. No prices, candles, or indicator values are fabricated here. Both
directions use the same indicators, confirmed pivots, and candle timestamps.
"""

import math
from collections.abc import Sequence

from config import settings

TREND_MIN_CANDLES = 53  # EMA50 must also have 50 observations three bars ago.
ENTRY_MIN_CANDLES = 21


def ema(values: Sequence[float], period: int) -> list[float]:
    """First-observation seeded EMA, matching the existing strategy convention."""
    if not len(values):
        return []
    result = [float(values[0])]
    weight = 2.0 / (period + 1.0)
    for value in values[1:]:
        result.append(float(value) * weight + result[-1] * (1.0 - weight))
    return result


def rsi_series(values: Sequence[float], period: int = 14) -> list:
    """Wilder RSI with zero moves retained in each period's gain/loss average."""
    result = [None] * len(values)
    if len(values) <= period:
        return result
    changes = [float(right) - float(left) for left, right in zip(values, values[1:])]
    gain = sum(max(change, 0.0) for change in changes[:period]) / period
    loss = sum(max(-change, 0.0) for change in changes[:period]) / period

    def value():
        if loss == 0:
            return 100.0 if gain > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + gain / loss)

    result[period] = value()
    for index in range(period + 1, len(values)):
        change = changes[index - 1]
        gain = (gain * (period - 1) + max(change, 0.0)) / period
        loss = (loss * (period - 1) + max(-change, 0.0)) / period
        result[index] = value()
    return result


def confirmed_swings(candles: Sequence[dict]) -> tuple[list, list]:
    """Strict pivots with the existing two closed bars on either side."""
    highs, lows = [], []
    for index in range(2, len(candles) - 2):
        candle = candles[index]
        neighbours = list(candles[index - 2:index]) + list(candles[index + 1:index + 3])
        point = {"ts": candle["ts"], "index": index}
        if candle["high"] > max(other["high"] for other in neighbours):
            highs.append({**point, "price": float(candle["high"])})
        if candle["low"] < min(other["low"] for other in neighbours):
            lows.append({**point, "price": float(candle["low"])})
    return highs, lows


def _structure(highs, lows):
    if len(highs) < 2 or len(lows) < 2:
        return "insufficient"
    if highs[-1]["price"] > highs[-2]["price"] and lows[-1]["price"] > lows[-2]["price"]:
        return "HH+HL"
    if highs[-1]["price"] < highs[-2]["price"] and lows[-1]["price"] < lows[-2]["price"]:
        return "LH+LL"
    return "mixed"


def _valid_candles(candles):
    previous_ts = None
    for candle in candles:
        try:
            values = [float(candle[field]) for field in ("open", "high", "low", "close", "ts")]
        except (KeyError, TypeError, ValueError):
            return False
        opening, high, low, close, timestamp = values
        if not all(math.isfinite(value) for value in values):
            return False
        if low <= 0 or high < max(opening, close) or low > min(opening, close):
            return False
        if previous_ts is not None and timestamp <= previous_ts:
            return False
        previous_ts = timestamp
    return True


def evaluate_both(trend_candles: Sequence[dict], entry_candles: Sequence[dict], *, source: str | None = None) -> dict:
    """Evaluate BUY and SELL and return JSON-safe diagnostics for *all* rules.

    Insufficient or malformed input fails closed. The strategy periods, strict
    two-sided pivot lookback, BUY RSI band 50..65 / SELL 35..50, two-times
    range cap, and obstacle clearance (BUY 1.5 / SELL 1.0) are unchanged.
    Trend swing structure is diagnostic, as in the two existing strategies.
    """
    trend, entry = list(trend_candles), list(entry_candles)
    metrics = {
        "trend_count": len(trend), "entry_count": len(entry), "source": source or "unknown",
        "trend_required": TREND_MIN_CANDLES, "entry_required": ENTRY_MIN_CANDLES,
        "trend_ts": None, "entry_ts": None,
        "close": None, "ema20": None, "ema50": None,
        "ema20_3ago": None, "ema50_3ago": None,
        "structure": "insufficient", "trend_swing_highs": [], "trend_swing_lows": [],
        "entry_close": None, "entry_low": None, "entry_high": None,
        "ema9": None, "ema21": None, "entry_structure": "insufficient",
        "entry_swing_highs": [], "entry_swing_lows": [],
        "rsi": None, "rsi_previous": None, "rsi_dir": "unavailable",
        "pullback_closes_ema21": [], "confirmation_range": None,
        "avg_prev_10_range": None, "previous_high": None, "previous_low": None,
        "nearest_resistance": None, "resistance_distance": None,
        "nearest_support": None, "support_distance": None,
        "atr_1m": None, "range_ratio": None,
    }

    def blocked(reason, status):
        return {direction: {**metrics, "direction": direction, "passed": False,
                            "status": status, "reason": reason, "failed_rules": [reason]}
                for direction in ("BUY", "SELL")}

    if not _valid_candles(trend) or not _valid_candles(entry):
        return blocked("invalid_ohlc_or_timestamps", "invalid_data")

    tc = [float(candle["close"]) for candle in trend]
    ec = [float(candle["close"]) for candle in entry]
    t20, t50 = ema(tc, 20), ema(tc, 50)
    e9, e21 = ema(ec, 9), ema(ec, 21)
    trend_highs, trend_lows = confirmed_swings(trend)
    entry_highs, entry_lows = confirmed_swings(entry)
    rsi_values = rsi_series(ec)
    metrics.update({
        "structure": _structure(trend_highs, trend_lows),
        "trend_swing_highs": trend_highs[-2:], "trend_swing_lows": trend_lows[-2:],
        "entry_structure": _structure(entry_highs, entry_lows),
        "entry_swing_highs": entry_highs[-2:], "entry_swing_lows": entry_lows[-2:],
    })
    if trend:
        metrics.update(trend_ts=trend[-1]["ts"], close=tc[-1])
    if len(trend) >= 20:
        metrics["ema20"] = t20[-1]
    if len(trend) >= 23:
        metrics["ema20_3ago"] = t20[-4]
    if len(trend) >= 50:
        metrics["ema50"] = t50[-1]
    if len(trend) >= 53:
        metrics["ema50_3ago"] = t50[-4]
    if entry:
        metrics.update(entry_ts=entry[-1]["ts"], entry_close=ec[-1],
                       entry_low=float(entry[-1]["low"]), entry_high=float(entry[-1]["high"]),
                       confirmation_range=float(entry[-1]["high"] - entry[-1]["low"]))
        metrics["rsi"] = rsi_values[-1]
        metrics["rsi_previous"] = rsi_values[-2] if len(entry) >= 2 else None
        if metrics["rsi"] is not None and metrics["rsi_previous"] is not None:
            delta = metrics["rsi"] - metrics["rsi_previous"]
            metrics["rsi_dir"] = "rising" if delta > 0 else "falling" if delta < 0 else "flat"
        resistance = [point["price"] for point in trend_highs if point["price"] > ec[-1]]
        support = [point["price"] for point in trend_lows if point["price"] < ec[-1]]
        if resistance:
            metrics.update(nearest_resistance=min(resistance), resistance_distance=min(resistance) - ec[-1])
        if support:
            metrics.update(nearest_support=max(support), support_distance=ec[-1] - max(support))
    if len(entry) >= 2:
        metrics.update(previous_high=float(entry[-2]["high"]), previous_low=float(entry[-2]["low"]))
    if len(entry) >= 9:
        metrics["ema9"] = e9[-1]
    if len(entry) >= 21:
        metrics["ema21"] = e21[-1]
        metrics["pullback_closes_ema21"] = [
            {"ts": entry[index]["ts"], "close": ec[index], "ema21": e21[index]}
            for index in range(len(entry) - 5, len(entry))
        ]
    if len(entry) >= 11:
        metrics["avg_prev_10_range"] = sum(float(candle["high"] - candle["low"])
                                             for candle in entry[-11:-1]) / 10.0
    if len(entry) >= 14:
        tx = [max(float(candle["high"]) - float(candle["low"]), 1e-9) for candle in entry[-14:]]
        metrics["atr_1m"] = sum(tx) / len(tx)
    if metrics["avg_prev_10_range"]:
        metrics["range_ratio"] = metrics["confirmation_range"] / max(metrics["avg_prev_10_range"], 1e-9)
    if len(trend) < TREND_MIN_CANDLES or len(entry) < ENTRY_MIN_CANDLES:
        return blocked("insufficient_closed_candles", "insufficient_data")

    results = {}
    for direction, sign in (("BUY", 1), ("SELL", -1)):
        failures = []

        def check(condition, name):
            if not condition:
                failures.append(name)

        check(sign * (tc[-1] - t20[-1]) > 0, "30s_close_vs_ema20")
        check(sign * (t20[-1] - t50[-1]) > 0, "30s_ema20_vs_ema50")
        check(sign * (t20[-1] - t20[-4]) > 0 and sign * (t50[-1] - t50[-4]) > 0,
              "30s_ema_slope_3_candles")
        check(sign * (e9[-1] - e21[-1]) > 0, "1m_ema9_vs_ema21")
        check(sign * (ec[-1] - e21[-1]) > 0, "1m_close_vs_ema21")
        check(all(sign * (pair["close"] - pair["ema21"]) >= 0
                  for pair in metrics["pullback_closes_ema21"]), "1m_pullback_close_vs_ema21")
        pullback_swings = entry_lows if sign == 1 else entry_highs
        breakout_swings = entry_highs if sign == 1 else entry_lows
        if len(pullback_swings) >= 2:
            check(sign * (pullback_swings[-1]["price"] - pullback_swings[-2]["price"]) > 0,
                  "1m_pullback_swing_progression")
            extreme = metrics["entry_low"] if sign == 1 else metrics["entry_high"]
            check(sign * (extreme - pullback_swings[-2]["price"]) > 0, "1m_pullback_vs_prior_swing")
        elif sign == -1:
            check(entry[-3]["high"] > entry[-2]["high"] > entry[-1]["high"],
                  "1m_confirmed_pullback_swings_or_descending_highs")
        else:
            check(False, "1m_confirmed_pullback_swings")
        if breakout_swings:
            check(sign * (ec[-1] - breakout_swings[-1]["price"]) > 0, "1m_close_breaks_swing")
        elif sign == -1:
            check(entry[-3]["low"] > entry[-2]["low"] > entry[-1]["low"],
                  "1m_confirmed_breakout_swing_or_descending_lows")
        else:
            check(False, "1m_confirmed_breakout_swing")
        check(sign * (ec[-1] - e9[-1]) > 0, "1m_confirmation_close_vs_ema9")
        previous_extreme = metrics["previous_high"] if sign == 1 else metrics["previous_low"]
        check(sign * (ec[-1] - previous_extreme) > 0, "1m_close_breaks_previous_candle")
        check(metrics["confirmation_range"] <= 2.0 * metrics["avg_prev_10_range"], "1m_confirmation_range")
        band = (50.0, 65.0) if sign == 1 else (35.0, 50.0)
        check(band[0] <= metrics["rsi"] <= band[1], "1m_rsi14_band")
        check(metrics["rsi_dir"] == ("rising" if sign == 1 else "falling"), "1m_rsi14_direction")
        obstacle_distance = metrics["resistance_distance"] if sign == 1 else metrics["support_distance"]
        atr_distance = settings.ATR_DISTANCE_MULTIPLIER * max(metrics.get("atr_1m") or 0.0, 1e-9)
        legacy_clearance = (1.5 if sign == 1 else 1.0) * max(metrics["confirmation_range"] or 0.0, 1e-9)
        clearance = max(legacy_clearance, atr_distance)
        check(obstacle_distance is None or obstacle_distance > clearance,
              "30s_obstacle_clearance")
        results[direction] = {**metrics, "direction": direction, "passed": not failures,
                              "status": "fail" if failures else "pass",
                              "reason": failures[0] if failures else "closed_candle_setup_confirmed",
                              "failed_rules": failures}
    return results
