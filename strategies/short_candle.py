"""Separate 5s entries confirmed by completed 10s and 15s candle trends.

The original 30s/1m strategy is unchanged. This evaluator uses only completed,
contiguous candles and never fills missing intervals or uses a forming candle.
Its output is a signal, not an order: execution and risk checks remain with the
caller. Timestamps identify interval starts in epoch seconds.
"""

import math
from collections.abc import Mapping, Sequence

from strategies.closed_candle import ema, rsi_series


SHORT_MIN_CANDLES = {5: 10, 10: 15, 15: 15}


def _number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a candle number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite candle number")
    return number


def _normalize(candles, period, now):
    """Validate the entire supplied history, returning newly allocated bars."""
    if not isinstance(candles, Sequence) or isinstance(candles, (str, bytes)):
        return [], "invalid_candle_history"
    normalized = []
    expected_latest = math.floor(now / period) * period - period
    for candle in candles:
        if not isinstance(candle, Mapping):
            return [], "invalid_ohlc_or_timestamps"
        if any(flag in candle and candle[flag] is not True
               for flag in ("closed", "complete_from_open")):
            return [], "incomplete_closed_candle"
        try:
            bar = {name: _number(candle[name])
                   for name in ("ts", "open", "high", "low", "close")}
        except (KeyError, TypeError, ValueError, OverflowError):
            return [], "invalid_ohlc_or_timestamps"
        if (bar["low"] <= 0 or bar["high"] < max(bar["open"], bar["close"])
                or bar["low"] > min(bar["open"], bar["close"])):
            return [], "invalid_ohlc_or_timestamps"
        if bar["ts"] < 0 or bar["ts"] % period:
            return [], "unaligned_candle_timestamps"
        if bar["ts"] > expected_latest:
            return [], "future_or_forming_closed_candle"
        if normalized and bar["ts"] - normalized[-1]["ts"] != period:
            return [], "noncontiguous_closed_candles"
        normalized.append(bar)
    if normalized and normalized[-1]["ts"] != expected_latest:
        return normalized, "stale_closed_candles"
    return normalized, None


def evaluate_short(candles_by_period: dict[int, list[dict]], now: float, *, source: str | None = None) -> dict:
    """Evaluate mirrored BUY/SELL rules, failing closed for unusable snapshots.

    Each period must end at its most recently completed interval as of ``now``.
    Optional ``closed`` and ``complete_from_open`` flags must be true when given.
    Unspecified completion flags rely on the caller's completed-only contract;
    timestamps independently prevent accepting the current forming interval.
    """
    metrics = {
        "profile": "short", "closed_only": True, "source": source or "unknown",
        "entry_period_seconds": 5, "confirmation_period_seconds": 10,
        "trend_period_seconds": 15,
        "entry_count": 0, "entry_required": SHORT_MIN_CANDLES[5],
        "confirmation_count": 0, "confirmation_required": SHORT_MIN_CANDLES[10],
        "trend_count": 0, "trend_required": SHORT_MIN_CANDLES[15],
        "entry_ts": None, "confirmation_ts": None, "trend_ts": None,
        "entry_open": None, "entry_close": None, "entry_high": None, "entry_low": None,
        "close": None, "confirmation_close": None,
        "trend_ema9": None, "trend_ema21": None, "trend_ema9_3ago": None,
        "confirmation_ema9": None, "confirmation_ema21": None,
        "rsi": None, "rsi_previous": None, "rsi_dir": "unavailable",
        "confirmation_range": None, "avg_prev_10_range": None,
        "previous_high": None, "previous_low": None,
    }

    def blocked(reason, status="invalid_data"):
        return {
            direction: {**metrics, "direction": direction, "passed": False,
                        "status": status, "reason": reason, "failed_rules": [reason]}
            for direction in ("BUY", "SELL")
        }

    try:
        now = _number(now)
    except (TypeError, ValueError, OverflowError):
        return blocked("invalid_current_timestamp")
    if now < 0:
        return blocked("invalid_current_timestamp")
    if not isinstance(candles_by_period, Mapping):
        return blocked("invalid_candle_history")

    histories = {}
    errors = []
    for period, label in ((5, "entry"), (10, "confirmation"), (15, "trend")):
        raw = candles_by_period.get(period, [])
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            metrics[f"{label}_count"] = len(raw)
        bars, error = _normalize(raw, period, now)
        histories[period] = bars
        if bars:
            metrics[f"{label}_ts"] = bars[-1]["ts"]
        if error:
            errors.append(error)
    if errors:
        return blocked(errors[0], "stale_data" if errors[0] == "stale_closed_candles" else "invalid_data")
    if any(len(histories[period]) < count for period, count in SHORT_MIN_CANDLES.items()):
        return blocked("insufficient_closed_candles", "insufficient_data")

    entry, confirmation, trend = histories[5], histories[10], histories[15]
    ec = [bar["close"] for bar in entry]
    cc = [bar["close"] for bar in confirmation]
    tc = [bar["close"] for bar in trend]
    t9, t21 = ema(tc, 9), ema(tc, 21)
    c9, c21 = ema(cc, 9), ema(cc, 21)
    rsi_values = rsi_series(ec)
    metrics.update({
        "entry_open": entry[-1]["open"], "entry_close": ec[-1],
        "entry_high": entry[-1]["high"], "entry_low": entry[-1]["low"],
        "close": tc[-1], "confirmation_close": cc[-1],
        "trend_ema9": t9[-1], "trend_ema21": t21[-1], "trend_ema9_3ago": t9[-4],
        "confirmation_ema9": c9[-1], "confirmation_ema21": c21[-1],
        "rsi": rsi_values[-1], "rsi_previous": rsi_values[-2],
        "confirmation_range": entry[-1]["high"] - entry[-1]["low"],
        "avg_prev_10_range": sum((bar["high"] - bar["low"]) / 10.0 for bar in entry[-11:-1]),
        "previous_high": entry[-2]["high"], "previous_low": entry[-2]["low"],
    })
    # Even finite but extreme inputs must never leak NaN/Infinity into logs.
    invalid_values = [key for key, value in metrics.items()
                      if isinstance(value, float) and not math.isfinite(value)]
    if invalid_values:
        for key in invalid_values:
            metrics[key] = None
        return blocked("invalid_indicator_values")
    delta = metrics["rsi"] - metrics["rsi_previous"]
    metrics["rsi_dir"] = "rising" if delta > 0 else "falling" if delta < 0 else "flat"

    results = {}
    for direction, sign in (("BUY", 1), ("SELL", -1)):
        failures = []

        def check(condition, rule):
            if not condition:
                failures.append(rule)

        check(sign * (t9[-1] - t21[-1]) > 0, "15s_ema9_vs_ema21")
        check(sign * (tc[-1] - t9[-1]) > 0, "15s_close_vs_ema9")
        check(sign * (t9[-1] - t9[-4]) > 0, "15s_ema9_slope_3_candles")
        check(sign * (c9[-1] - c21[-1]) > 0, "10s_ema9_vs_ema21")
        check(sign * (cc[-1] - c9[-1]) > 0, "10s_close_vs_ema9")
        check(sign * (ec[-1] - entry[-1]["open"]) > 0, "5s_directional_body")
        previous_extreme = metrics["previous_high"] if sign == 1 else metrics["previous_low"]
        check(sign * (ec[-1] - previous_extreme) > 0, "5s_close_breaks_previous_candle")
        band = (50.0, 70.0) if sign == 1 else (30.0, 50.0)
        check(band[0] <= metrics["rsi"] <= band[1], "5s_rsi14_band")
        check(metrics["rsi_dir"] == ("rising" if sign == 1 else "falling"), "5s_rsi14_direction")
        check(0 < metrics["confirmation_range"] <= 2.0 * metrics["avg_prev_10_range"],
              "5s_confirmation_range")
        results[direction] = {
            **metrics, "direction": direction, "passed": not failures,
            "status": "fail" if failures else "pass",
            "reason": failures[0] if failures else "short_setup_confirmed",
            "failed_rules": failures,
        }
    return results
