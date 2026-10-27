"""Closed-candle multi-timeframe analysis with explicit market-data provenance."""

import logging
import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("agent1_mtf")

TF_30S  = 30
TF_1M   = 60
TF_5M   = 300
ALL_TIMEFRAMES = [TF_30S, TF_1M, TF_5M]

MIN_CANDLES = {
    TF_30S: 5,
    TF_1M:  4,
    TF_5M:  3,
}
SR_HISTORY_SIZE = 60

PREFERRED_ASSETS_ORDERED = [
    "AUDUSD-OTC", "EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC",
    "EURGBP-OTC", "USDCAD-OTC", "NZDUSD-OTC",
    "BTCUSD", "ETHUSD", "LTCUSD",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
]


# ─── Candle Builder ───────────────────────────────────────────────
class CandleBuilder:
    """Aligned OHLC of observed broker quotes, exposing only completed bars.

    The first partial bar and bars affected by quote outages are discarded;
    missing OHLC is never interpolated. This is sampled OHLC, not native broker
    history. ``max_tick_gap`` bounds sampling gaps, not market tick frequency.
    """

    def __init__(self, period: int, max_candles: int = 500,
                 max_tick_gap: float = 5.0, source: str = "broker_browser_sample"):
        if period <= 0 or max_tick_gap <= 0:
            raise ValueError("Candle period and maximum quote gap must be positive")
        self.period = period
        self.max_tick_gap = max_tick_gap
        self.min_tick_count = 2
        self.candles: deque = deque(maxlen=max_candles)
        self._current: Optional[dict] = None
        self._bar_end = 0.0
        self._last_tick_ts: Optional[float] = None
        self._current_complete = False
        self._price_window = set()
        self.source = self._normalize_source(source)

    @staticmethod
    def _normalize_source(source):
        if source is None:
            return "fallback"
        source = str(source).strip()
        lowered = source.lower()
        if "browser" in lowered or "sample" in lowered or "poll" in lowered or "1hz" in lowered:
            return "broker_browser_sample"
        if "native" in lowered or "socket" in lowered or "ws_ticks" in lowered or lowered.endswith("ws") or ("ws" in lowered and not any(token in lowered for token in ("browser", "sample", "poll", "1hz"))):
            return "broker_native_ws"
        if lowered in ("", "fallback"):
            return "fallback"
        return source

    @staticmethod
    def _source_family(source: str) -> str:
        normalized = CandleBuilder._normalize_source(source)
        if normalized in {"broker_native_ws", "broker_browser_sample", "fallback"}:
            return "market_feed"
        return normalized

    def invalidate(self, reason: str = "quote_gap"):
        if self.candles or self._current:
            log.warning("[FeedReset] timeframe=%ss source=%s reason=%s closed_count=%s",
                        self.period, self.source, reason, len(self.candles))
        self.candles.clear()
        self._current = None
        self._bar_end = 0.0
        self._last_tick_ts = None
        self._current_complete = False
        self._price_window.clear()
        self.source = "fallback"

    def add_tick(self, price: float, ts: float, source: str = "broker_browser_sample") -> bool:
        if not math.isfinite(price) or price <= 0 or not math.isfinite(ts):
            return False
        normalized_source = self._normalize_source(source)
        if self.candles or self._current:
            if self.source and self.source != normalized_source:
                if self._source_family(self.source) == self._source_family(normalized_source):
                    log.info("[Candles] timeframe=%ss source_switch=%s->%s tolerated", self.period, self.source, normalized_source)
                    self.source = normalized_source
                else:
                    log.warning("[Candles] timeframe=%ss source_switch=%s->%s", self.period, self.source, normalized_source)
                    self.invalidate("source_mismatch")
                    self.source = normalized_source
        else:
            self.source = normalized_source
        if self._last_tick_ts is not None and ts <= self._last_tick_ts:
            return False
        if self._last_tick_ts is not None and ts - self._last_tick_ts > self.max_tick_gap:
            self.invalidate("quote_gap")

        closed = False
        bar_start = math.floor(ts / self.period) * self.period
        if self._current is None:
            if self.candles and self.candles[-1]["ts"] + self.period < bar_start:
                self.candles.clear()
            self._current_complete = math.isclose(ts, bar_start, abs_tol=1e-6, rel_tol=0)
        elif ts >= self._bar_end:
            if self._current_complete and int(self._current.get("tick_count", 1)) >= self.min_tick_count:
                if self.candles and self._current["ts"] != self.candles[-1]["ts"] + self.period:
                    self.candles.clear()
                self._current["tick_count"] = int(self._current.get("tick_count", 1))
                self._current["distinct_prices"] = len(self._price_window)
                self._current["flat"] = self._current["distinct_prices"] < 3
                self._current["thin"] = self._current["tick_count"] < 3
                self._current["low_information"] = self.source in {"polled_1hz", "broker_browser_sample"} and self.period in (5, 10, 15)
                self.candles.append(dict(self._current, source=self.source, tick_count=self._current["tick_count"],
                                         thin=self._current["thin"], distinct_prices=self._current["distinct_prices"],
                                         flat=self._current["flat"], low_information=self._current["low_information"]))
                closed = True
            else:
                self._current["tick_count"] = int(self._current.get("tick_count", 1))
                self._current["thin"] = self._current["tick_count"] < self.min_tick_count
                self._current["complete_from_open"] = False
                self._current["closed"] = False
            self._current_complete = bar_start == self._bar_end
            self._current = None
            self._price_window = set()

        if self._current is None:
            self._bar_end = bar_start + self.period
            self._current = {"open": price, "high": price, "low": price,
                             "close": price, "ts": bar_start,
                             "source": self.source, "tick_count": 1, "thin": False,
                             "distinct_prices": 1, "flat": False, "low_information": False}
            self._price_window = {round(float(price), 6)}
        else:
            self._price_window.add(round(float(price), 6))
            self._current["high"] = max(self._current["high"], price)
            self._current["low"] = min(self._current["low"], price)
            self._current["close"] = price
            self._current["tick_count"] = int(self._current.get("tick_count", 1)) + 1
            self._current["thin"] = self._current["tick_count"] < 3
            self._current["distinct_prices"] = len(self._price_window)
            self._current["flat"] = self._current["distinct_prices"] < 3
            self._current["low_information"] = self.source in {"polled_1hz", "broker_browser_sample"} and self.period in (5, 10, 15)
            self._current["source"] = self.source
        self._last_tick_ts = ts
        return closed

    def closed_snapshot(self) -> List[dict]:
        return [dict(c) for c in self.candles]

    def forming_snapshot(self) -> Optional[dict]:
        """Return detached sampled OHLC for observation, never signal inputs."""
        if self._current is None:
            return None
        current = dict(self._current)
        current["tick_count"] = int(current.get("tick_count", 1))
        current["thin"] = current["tick_count"] < 3
        current["distinct_prices"] = len(self._price_window)
        current["flat"] = current["distinct_prices"] < 3
        current["low_information"] = self.source in {"polled_1hz", "broker_browser_sample"} and self.period in (5, 10, 15)
        return {
            **current,
            "closed": False,
            "complete_from_open": self._current_complete,
            "open_ts": self._current["ts"],
            "close_ts": self._bar_end,
            "last_tick_ts": self._last_tick_ts,
        }

    @property
    def latest_closed_ts(self) -> Optional[float]:
        return self.candles[-1]["ts"] if self.candles else None

    def get_opens(self) -> np.ndarray:
        return np.array([c["open"] for c in self.candles], dtype=float)

    def get_closes(self) -> np.ndarray:
        return np.array([c["close"] for c in self.candles], dtype=float)

    def get_highs(self) -> np.ndarray:
        return np.array([c["high"] for c in self.candles], dtype=float)

    def get_lows(self) -> np.ndarray:
        return np.array([c["low"] for c in self.candles], dtype=float)

    def __len__(self):
        return len(self.candles)


# ─── Indicators ───────────────────────────────────────────────────
def _ema(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    k, v = 2.0 / (p + 1), arr[0]
    for i, x in enumerate(arr):
        v = x * k + v * (1 - k); out[i] = v
    return out

def _rsi(c: np.ndarray, p: int = 14) -> float:
    """Wilder RSI: initialize with all p changes, then apply 1/p smoothing."""
    if len(c) < p + 1:
        return float("nan")
    delta = np.diff(np.asarray(c, dtype=float))
    if not np.isfinite(delta).all():
        return float("nan")
    gains = np.maximum(delta, 0)
    losses = np.maximum(-delta, 0)
    gain, loss = float(gains[:p].mean()), float(losses[:p].mean())
    for up, down in zip(gains[p:], losses[p:]):
        gain = (gain * (p - 1) + up) / p
        loss = (loss * (p - 1) + down) / p
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return float(100 - 100 / (1 + gain / loss))

def _macd(c: np.ndarray) -> Tuple[float, float, float]:
    if len(c) < 26:
        return 0.0, 0.0, 0.0
    m = _ema(c, 12) - _ema(c, 26)
    s = _ema(m[~np.isnan(m)], 9)
    macd_v = float(m[-1])
    macd_s = float(s[-1])
    hist = macd_v - macd_s
    return macd_v, macd_s, float(hist)

def _bollinger(c: np.ndarray, p: int = 20) -> Tuple[float, float, float]:
    if len(c) < p: m = c[-1]; return m, m, m
    w = c[-p:]; mid = w.mean(); std = w.std()
    return mid + 2 * std, mid, mid - 2 * std

def _stoch(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int = 14) -> float:
    if len(c) < p: return 50.0
    hi, lo = h[-p:].max(), l[-p:].min()
    return 50.0 if hi == lo else 100 * (c[-1] - lo) / (hi - lo)

def _ema_cross(c: np.ndarray) -> int:
    if len(c) < 21: return 0
    f, s = _ema(c, 9), _ema(c, 21)
    return 1 if f[-1] > s[-1] else (-1 if f[-1] < s[-1] else 0)


# ─── Support / Resistance ─────────────────────────────────────────
class SREngine:
    def __init__(self, tol: float = 0.0005):
        self.tol = tol
        self.supports: List[float]    = []
        self.resistances: List[float] = []

    def update(self, h: np.ndarray, l: np.ndarray, c: np.ndarray):
        if len(c) < 5: return
        ph, pl = [], []
        for i in range(2, len(c) - 2):
            if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
                ph.append(h[i])
            if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
                pl.append(l[i])
        self.resistances = self._cluster(ph)
        self.supports    = self._cluster(pl)

    def _cluster(self, pts: list) -> List[float]:
        if not pts: return []
        pts = sorted(pts); clusters, grp = [], [pts[0]]
        for p in pts[1:]:
            if abs(p - grp[-1]) / max(grp[-1], 1e-9) < self.tol:
                grp.append(p)
            else:
                clusters.append(float(np.mean(grp))); grp = [p]
        clusters.append(float(np.mean(grp)))
        return clusters

    def zone_bias(self, price: float) -> str:
        all_z = [(s, "support") for s in self.supports] + \
                [(r, "resistance") for r in self.resistances]
        if not all_z: return "neutral"
        closest = min(all_z, key=lambda z: abs(z[0] - price))
        if abs(closest[0] - price) / max(closest[0], 1e-9) > 0.003:
            return "neutral"
        return "bullish" if closest[1] == "support" else "bearish"


# ─── Main Agent ───────────────────────────────────────────────────
class MultiTimeframeAgent:
    def __init__(self, asset: str, max_tick_gap: float = 5.0):
        self.asset = asset
        self.max_tick_gap = max_tick_gap
        self.reset(asset)

    def reset(self, asset: str = None):
        if asset is not None:
            self.asset = asset
        self.builders: Dict[int, CandleBuilder] = {
            TF_30S: CandleBuilder(TF_30S, 200, self.max_tick_gap),
            TF_1M:  CandleBuilder(TF_1M,  70, self.max_tick_gap),
            TF_5M:  CandleBuilder(TF_5M,  40, self.max_tick_gap),
        }
        self.sr_engine  = SREngine()
        self._last_price = 0.0
        self._tick_count = 0
        self._last_tf_state: Dict = {}

    def log_seed_sources(self):
        for period, builder in self.builders.items():
            log.info("[Agent1Seed] timeframe=%ss source=%s closed_count=%s native_history=%s",
                     period, builder.source, len(builder),
                     "unavailable" if builder.source == "broker_browser_sample" else "provided")

    def seed_from_dataframe(self, df, timeframe: int = TF_1M,
                            source: str = "unspecified", now: Optional[float] = None, log_seed: bool = True):
        """Seed real timestamped OHLC; never invent shorter-timeframe candles.

        Native 30s history must declare its resolution in DataFrame metadata.
        Minute history may additionally seed complete aligned 5m bars.
        """
        import pandas as pd

        if timeframe not in self.builders:
            raise ValueError(f"Unsupported seed timeframe: {timeframe}")
        if df is None or df.empty:
            self.log_seed_sources()
            return
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("OHLC history must carry its original DatetimeIndex")
        if source == "unspecified":
            source = df.attrs.get("source", source)
        declared = df.attrs.get("timeframe_seconds")
        if declared is not None and declared != timeframe:
            raise ValueError("History resolution does not match requested seed timeframe")
        if timeframe == TF_30S and declared != TF_30S:
            raise ValueError("Native 30s history requires timeframe_seconds=30 metadata; M1 cannot seed 30s")
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns) or df.index.has_duplicates:
            raise ValueError("OHLC history has missing columns or duplicate timestamps")
        now = time.time() if now is None else now
        bars = []
        for timestamp, row in df.sort_index().iterrows():
            timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp
            ts = timestamp.timestamp()
            if ts % timeframe:
                raise ValueError("OHLC timestamps must be aligned candle opening times")
            if ts + timeframe > now:
                continue
            candle = {key: float(row[key]) for key in required}
            if (not all(math.isfinite(v) and v > 0 for v in candle.values())
                    or candle["low"] > min(candle["open"], candle["close"])
                    or candle["high"] < max(candle["open"], candle["close"])):
                raise ValueError("Invalid historical OHLC values")
            if bars and ts != bars[-1]["ts"] + timeframe:
                bars.clear()
            bars.append(dict(candle, ts=ts, source=source))
        builder = self.builders[timeframe]
        builder.candles.clear()
        builder._current = None
        builder._last_tick_ts = None
        builder._current_complete = False
        builder.candles.extend(bars)
        builder.source = source
        if timeframe == TF_1M:
            grouped = {}
            for candle in bars:
                start = math.floor(candle["ts"] / TF_5M) * TF_5M
                grouped.setdefault(start, []).append(candle)
            b5m = self.builders[TF_5M]
            b5m.candles.clear()
            b5m._current = None
            for start, group in sorted(grouped.items()):
                if [bar["ts"] for bar in group] != [start + i * TF_1M for i in range(5)]:
                    continue
                b5m.candles.append({"open": group[0]["open"], "close": group[-1]["close"],
                                    "high": max(bar["high"] for bar in group),
                                    "low": min(bar["low"] for bar in group), "ts": start,
                                    "source": f"{source}:complete_aligned_M1_aggregation"})
            b5m.source = f"{source}:complete_aligned_M1_aggregation"
        if log_seed:
            self.log_seed_sources()
        self._update_sr()

    def bootstrap(self, base_price: float, n_ticks: int = 200):
        # Compatibility: one observed price supplies no historical OHLC.
        if math.isfinite(base_price) and base_price > 0:
            self._last_price = base_price
        log.warning("[Agent1] Historical bootstrap unavailable; awaiting real closed candles")
        self.log_seed_sources()

    def invalidate(self, reason: str = "quote_gap"):
        for builder in self.builders.values():
            builder.invalidate(reason)
        self.sr_engine = SREngine()
        self._last_tf_state = {}
        self._last_price = 0.0

    def _update_sr(self):
        b1m = self.builders[TF_1M]
        if len(b1m) >= 5:
            self.sr_engine.update(b1m.get_highs(), b1m.get_lows(), b1m.get_closes())
        else:
            self.sr_engine = SREngine()

    def add_tick(self, price: float, ts: Optional[float] = None) -> set:
        ts = time.time() if ts is None else ts
        if not math.isfinite(price) or price <= 0 or not math.isfinite(ts):
            return set()
        last_ts = self.builders[TF_30S]._last_tick_ts
        if last_ts is not None and ts <= last_ts:
            return set()
        if last_ts is not None and ts - last_ts > self.max_tick_gap:
            self.invalidate("quote_gap")
        self._last_price = price
        self._tick_count += 1
        closed = {period for period, builder in self.builders.items() if builder.add_tick(price, ts)}
        self._update_sr()
        return closed

    def analyze(self) -> Tuple["Signal", float]:
        from models.predictor import Signal

        if self._last_price <= 0:
            return Signal.HOLD, 0.0

        votes_buy = votes_sell = 0
        total_conf = 0.0
        tf_details = {}
        active_tfs = 0

        for tf in ALL_TIMEFRAMES:
            b = self.builders[tf]
            if len(b) < MIN_CANDLES[tf]:
                log.debug(f"[Agent1] TF={tf}s: {len(b)}/{MIN_CANDLES[tf]} candles")
                continue
            active_tfs += 1
            vote, conf = self._analyze_tf(b.get_closes(), b.get_highs(), b.get_lows(), tf)
            tf_details[tf] = {"vote": vote, "confidence": conf}
            if vote == "BUY":
                votes_buy  += 1; total_conf += conf
            elif vote == "SELL":
                votes_sell += 1; total_conf += conf

        self._last_tf_state = tf_details

        def _emit(direction: str, votes: int, avg_raw: float) -> Tuple["Signal", float]:
            bias = self.sr_engine.zone_bias(self._last_price)
            if direction == "BUY":
                sr_adj = +0.04 if bias == "bullish" else (-0.05 if bias == "bearish" else 0.0)
            else:
                sr_adj = +0.04 if bias == "bearish" else (-0.05 if bias == "bullish" else 0.0)
            if votes >= 3:
                floor = 0.50
            elif votes == 2:
                floor = 0.45
            else:
                floor = 0.40
            conf = max(floor, min(0.97, avg_raw + sr_adj))
            sig  = Signal.BUY if direction == "BUY" else Signal.SELL
            log.info(f"[Agent1] {direction} {votes}/{active_tfs} active TFs | raw={avg_raw:.2%} "
                     f"sr={sr_adj:+.2%} floor={floor:.2%} → conf={conf:.2%} | bias={bias}")
            return sig, conf

        if votes_buy >= 2:
            return _emit("BUY",  votes_buy,  total_conf / votes_buy)
        if votes_sell >= 2:
            return _emit("SELL", votes_sell, total_conf / votes_sell)

        if active_tfs >= 1 and (votes_buy >= 1 or votes_sell >= 1):
            if votes_buy >= 1:
                return _emit("BUY",  votes_buy,  total_conf / votes_buy)
            else:
                return _emit("SELL", votes_sell, total_conf / votes_sell)

        log.debug(f"[Agent1] No consensus BUY={votes_buy} SELL={votes_sell} from {active_tfs} active TFs")
        return Signal.HOLD, 0.0

    def get_latest_multi_tf_state(self) -> Dict:
        return self._last_tf_state

    def evaluate_chart_strength(self) -> float:
        if not self._last_tf_state:
            return 0.0
        score = 0.0
        weight = 0.0
        for tf, data in self._last_tf_state.items():
            conf = data.get("confidence", 0.0)
            vote = data.get("vote")
            tf_weight = 1.0
            if tf >= TF_1M:
                tf_weight = 1.25
            elif tf >= TF_30S:
                tf_weight = 1.1
            if vote == "BUY":
                score += conf * tf_weight
            elif vote == "SELL":
                score -= conf * tf_weight
            weight += tf_weight

        if weight == 0:
            return 0.0

        bias = self.sr_engine.zone_bias(self._last_price)
        if bias == "bullish":
            score += 0.1
        elif bias == "bearish":
            score -= 0.1

        return float(score / max(weight, 1.0))

    def _analyze_tf(self, c: np.ndarray, h: np.ndarray,
                    l: np.ndarray, tf: int) -> Tuple[str, float]:
        scores = []

        rsi = _rsi(c)
        if   rsi < 30:  scores.append(+1.0)
        elif rsi > 70:  scores.append(-1.0)
        elif rsi <= 45: scores.append(+0.5)
        elif rsi >= 55: scores.append(-0.5)
        else:           scores.append(0.0)

        ml, sl, _ = _macd(c)
        if   ml > sl and ml > 0: scores.append(+1.0)
        elif ml < sl and ml < 0: scores.append(-1.0)
        elif ml > sl:            scores.append(+0.5)
        elif ml < sl:            scores.append(-0.5)
        else:                    scores.append(0.0)

        up, mid, lo = _bollinger(c)
        p = c[-1]
        if   p <= lo:  scores.append(+1.0)
        elif p >= up:  scores.append(-1.0)
        elif p < mid:  scores.append(+0.3)
        elif p > mid:  scores.append(-0.3)
        else:          scores.append(0.0)

        scores.append(_ema_cross(c) * 0.8)

        sk = _stoch(h, l, c)
        if   sk < 20: scores.append(+1.0)
        elif sk > 80: scores.append(-1.0)
        elif sk < 40: scores.append(+0.3)
        elif sk > 60: scores.append(-0.3)
        else:         scores.append(0.0)

        norm  = sum(scores) / 5.0
        abs_n = abs(norm)
        conf  = 0.45 + abs_n * 0.47

        if   norm >=  0.20: return "BUY",  conf
        elif norm <= -0.20: return "SELL", conf
        return "NEUTRAL", conf
