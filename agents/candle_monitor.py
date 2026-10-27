"""Short sampled candles for monitoring and the separate short-entry strategy.

Only actual observed quotes are aggregated. Forming bars are explicitly marked,
startup partial bars are excluded from completed history, and sampling outages
discard history using the same continuity rules as the existing CandleBuilder.
Only detached completed snapshots are exposed to the short-entry strategy.
"""

import logging
import math
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from config import settings
from agents.agent_multi_timeframe import CandleBuilder

log = logging.getLogger("candle_monitor")


def _change(current: float, previous: float) -> Optional[float]:
    difference = current - previous
    return difference if math.isfinite(difference) else None


class CandleMonitor:
    """Build independent short candles without changing classic strategy inputs."""

    def __init__(self, periods: Iterable[int] = (5, 10, 15),
                 max_tick_gap: float = 5.0, max_candles: int = 120):
        periods = tuple(periods)
        if (not periods or len(set(periods)) != len(periods)
                or any(not isinstance(period, int) or isinstance(period, bool)
                       or period <= 0 for period in periods)):
            raise ValueError("Monitor periods must be distinct positive integers")
        if not math.isfinite(max_tick_gap) or max_tick_gap <= 0:
            raise ValueError("Maximum quote gap must be finite and positive")
        if not isinstance(max_candles, int) or max_candles <= 0:
            raise ValueError("Candle history size must be a positive integer")
        self.asset = None
        self._builders = {
            period: CandleBuilder(period, max_candles, max_tick_gap)
            for period in periods
        }
        self._last_tick_ts: Optional[float] = None
        self._invalidated_reason: Optional[str] = None
        self._last_valid_snapshot: Optional[dict] = None
        self.max_tick_gap = max_tick_gap

    def set_asset(self, asset: str):
        self.asset = str(asset)
        for builder in self._builders.values():
            if builder.source == "broker_native_ws" or builder.source == "fallback":
                builder.source = builder.source

    def add_tick(self, price: float, ts: float, source: str = "broker_browser_sample") -> Set[int]:
        """Add one real observation and return periods that completed a bar."""
        try:
            price, ts = float(price), float(ts)
        except (TypeError, ValueError, OverflowError):
            return set()
        if (not math.isfinite(price) or price <= 0 or not math.isfinite(ts)
                or (self._last_tick_ts is not None and ts <= self._last_tick_ts)):
            return set()
        if self._last_tick_ts is None and self._last_valid_snapshot is not None:
            self.restore_from_snapshot(self._last_valid_snapshot)
        closed = set()
        for period, builder in self._builders.items():
            action = builder.add_tick(price, ts, source=source)
            if action:
                closed.add(period)
        self._last_tick_ts = ts
        self._invalidated_reason = None
        self._last_valid_snapshot = self.snapshot(now=ts)
        if closed and self.asset:
            try:
                self.persist_closed_history(path=settings.CANDLE_STORE_PATH, asset=self.asset)
            except Exception:
                pass
        return closed

    def invalidate(self, reason: str = "quote_gap"):
        """Keep the last valid warm-up snapshot so reconnects do not blank short history."""
        self._last_valid_snapshot = self.snapshot(now=time.time()) if any(
            bool(builder.closed_snapshot()) for builder in self._builders.values()) else self._last_valid_snapshot
        counts = {f"{period}s": len(builder.closed_snapshot()) for period, builder in self._builders.items()}
        log.warning("[FeedReset] reason=%s counts=%s", reason, counts)
        for builder in self._builders.values():
            builder.invalidate(reason)
        self._last_tick_ts = None
        self._invalidated_reason = reason

    def restore_from_snapshot(self, snapshot: dict) -> bool:
        """Restore completed history and any in-flight short candle for continuity."""
        if not isinstance(snapshot, dict):
            return False
        restored = False
        last_tick_ts = None
        for period, builder in self._builders.items():
            period_key = f"{period}s"
            state = snapshot.get(period_key, {})
            if isinstance(state, dict):
                tick_ts = state.get("last_tick_ts")
                if tick_ts is not None:
                    last_tick_ts = tick_ts if last_tick_ts is None else max(last_tick_ts, tick_ts)
            history = state.get("history")
            if not isinstance(history, list):
                latest = state.get("latest_closed")
                if isinstance(latest, dict):
                    history = [dict(latest)]
            forming = state.get("forming")
            if not isinstance(history, list) and not isinstance(forming, dict):
                continue

            builder.candles.clear()
            builder._current = None
            builder._bar_end = 0.0
            builder._last_tick_ts = state.get("last_tick_ts")
            builder._current_complete = False
            if isinstance(history, list):
                builder.candles.extend(dict(bar) for bar in history)
            if isinstance(forming, dict):
                builder._current = {
                    "open": float(forming.get("open", history[-1]["close"] if history else 0.0)),
                    "high": float(forming.get("high", forming.get("open", 0.0))),
                    "low": float(forming.get("low", forming.get("open", 0.0))),
                    "close": float(forming.get("close", forming.get("open", 0.0))),
                    "ts": float(forming.get("ts", builder.candles[-1]["ts"] + period if builder.candles else 0.0)),
                    "source": forming.get("source", state.get("source", builder.source)),
                    "tick_count": int(forming.get("tick_count", 1)),
                    "thin": bool(forming.get("thin", False)),
                }
                builder._current_complete = bool(forming.get("complete_from_open", False))
                builder._bar_end = float(forming.get("close_ts", builder._current["ts"] + period))
                builder._last_tick_ts = forming.get("last_tick_ts", builder._last_tick_ts)
            builder.source = CandleBuilder._normalize_source(state.get("source", builder.source))
            self._invalidated_reason = None
            restored = True
        if restored:
            self._last_valid_snapshot = snapshot
            self._last_tick_ts = last_tick_ts
        return restored

    def closed_snapshots(self) -> Dict[int, list]:
        """Return detached completed OHLC; forming and startup partial bars stay out."""
        return {period: builder.closed_snapshot()
                for period, builder in self._builders.items()}

    def persist_closed_history(self, path: str = "data/candles.sqlite3", asset: str = "EUR/USD") -> int:
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        inserted = 0
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS candles (asset TEXT, timeframe INTEGER, open_ts REAL, open REAL, high REAL, low REAL, close REAL, source TEXT, tick_count INTEGER, thin INTEGER, PRIMARY KEY(asset, timeframe, open_ts))")
            for period, bars in self.closed_snapshots().items():
                for candle in bars:
                    conn.execute(
                        "INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (str(asset), int(period), float(candle.get("ts", 0.0)), float(candle.get("open", 0.0)),
                         float(candle.get("high", 0.0)), float(candle.get("low", 0.0)), float(candle.get("close", 0.0)),
                         str(candle.get("source", "fallback")), int(candle.get("tick_count", 1)), int(bool(candle.get("thin", False))))
                    )
                    inserted += 1
        return inserted

    def restore_closed_history(self, path: str = "data/candles.sqlite3", asset: str = "EUR/USD") -> int:
        db_path = Path(path)
        if not db_path.exists():
            return 0
        restored = 0
        counts = {}
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT timeframe, open_ts, open, high, low, close, source, tick_count, thin FROM candles WHERE asset=? ORDER BY timeframe, open_ts", (str(asset),)).fetchall()
            for period, ts, open_, high, low, close, source, tick_count, thin in rows:
                builder = self._builders.get(int(period))
                if builder is None:
                    continue
                builder.candles.append({
                    "open": float(open_), "high": float(high), "low": float(low), "close": float(close),
                    "ts": float(ts), "source": str(source or "fallback"), "tick_count": int(tick_count or 1), "thin": bool(thin)
                })
                restored += 1
                counts[int(period)] = counts.get(int(period), 0) + 1
        if restored:
            self._last_valid_snapshot = self.snapshot(now=time.time())
            log.info("[CandleHistory] asset=%s restored_counts=%s total=%s", asset, counts, restored)
        return restored

    def snapshot(self, now: Optional[float] = None) -> Dict[str, dict]:
        """Return JSON-safe observations; a clock alone never completes a bar."""
        now = time.time() if now is None else float(now)
        if not math.isfinite(now):
            raise ValueError("Snapshot time must be finite")
        age = (None if self._last_tick_ts is None
               else _change(now, self._last_tick_ts))
        fresh = age is not None and 0 <= age <= self.max_tick_gap
        result = {}
        for period, builder in self._builders.items():
            closed = builder.closed_snapshot()
            forming = builder.forming_snapshot()
            latest = ({**closed[-1], "closed": True,
                       "complete_from_open": True,
                       "open_ts": closed[-1]["ts"],
                       "close_ts": closed[-1]["ts"] + period}
                      if closed else None)
            direction = None
            change = None
            if forming is not None:
                direction = ("up" if forming["close"] > forming["open"] else
                             "down" if forming["close"] < forming["open"] else "flat")
                change = _change(forming["close"], forming["open"])
            ready = fresh and bool(closed)
            status = ("no_quotes" if self._last_tick_ts is None else
                      "stale" if not fresh else "ready" if ready else "warming_up")
            result[f"{period}s"] = {
                "source": builder.source,
                "closed_count": len(closed),
                "history": [dict(bar) for bar in closed],
                "latest_closed": latest,
                "forming": forming,
                "observed_direction": direction,
                "change_from_open": change,
                "last_close_change": (_change(closed[-1]["close"], closed[-2]["close"])
                                      if len(closed) >= 2 else None),
                "last_tick_ts": self._last_tick_ts,
                "quote_age_seconds": age,
                "fresh": fresh,
                "ready": ready,
                "status": status,
                "invalidated_reason": self._invalidated_reason,
                "low_information": any(bool(bar.get("low_information")) for bar in closed) or bool(forming and forming.get("low_information")),
            }
        return result
