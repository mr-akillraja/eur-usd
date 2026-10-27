"""Unified EUR/USD OTC BUY/SELL orchestration on genuine closed candles.

30s EMA20/50 trend, 1m EMA9/21 pullback/resumption, RSI14; both
strategies share one quote stream, browser, snapshot, and execution lock.
Native 30s/1m history comes from the authenticated broker WebSocket. If a
resolution is unavailable, its builder warms up from live observations only.
One-second checks preserve these rules and optionally allow a separate
5s breakout strategy with 10s/15s confirmation, sharing the same execution guard.
"""

import asyncio
import csv
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple
import shutil
from pathlib import Path
import numpy as np
import argparse
import json
import math
from contextlib import nullcontext
from bot.process_lock import ProcessLock

sys.path.insert(0, os.path.dirname(__file__))

from config import settings
from bot.risk import RiskManager, TradeRecord
from bot.browser import OlymtradeBot
from bot.chart_indicators import configure_strategy_chart
from models.predictor import Signal
from agents.agent_multi_timeframe import MultiTimeframeAgent
from agents.candle_monitor import CandleMonitor
from agents.agent_rl_learning import RLFeedbackAgent
from agents.agent_pipeline_supervisor import PipelineSupervisorAgent
from agents.agent_regime_detector import RegimeDetector
from agents.agent_asset_selector import AssetSelectorAgent
from agents.chart_analyzer import ChartAnalyzer, AnalysisResult
from agents.multi_timeframe_consensus import MultiTimeframeConsensusAnalyzer

BEAR_REGIMES = {"BEAR_TREND", "HIGH_VOL_DOWN", "BREAKOUT_DOWN"}
BULL_REGIMES = {"BULL_TREND",  "HIGH_VOL_UP",  "BREAKOUT_UP"}

# ── Strategy constants; stake and daily limits come from config/settings.py ──
# Compatibility exports for the standalone demo execution diagnostic.
# The runtime resolves current settings in __init__, not from these aliases.
TRADE_AMOUNT        = settings.TRADE_AMOUNT
DAILY_LOSS_LIMIT    = settings.MAX_DAILY_LOSS_USD
MAX_TRADES_PER_DAY  = settings.MAX_DAILY_TRADES
MIN_CONFIDENCE      = 0.60     # 60% minimum (strategy doc: 60–65%)
MIN_CONFLUENCES     = 5        # require 5/9 indicators to agree
PAYOUT_MINIMUM      = 0.80     # skip assets below 80% payout
FTT_DURATION_S      = 60       # 1 minute trades
RSI_PERIOD = 14
RSI_LOW = 35
RSI_HIGH = 65
RSI_RESET = 50
ENTRY_WINDOW_SECONDS = 10
STAKE_PERCENT = 1.0
MAX_CONSECUTIVE_LOSSES = 3
LOSS_PAUSE_MINUTES = 30
DAILY_LOSS_LIMIT_PERCENT = 5.0
MIN_PAYOUT_PERCENT = 80


def _rotate_daily_log(log_path: str):
    if not os.path.exists(log_path):
        return
    archive_dir = settings.LOG_ARCHIVE_DIR
    os.makedirs(archive_dir, exist_ok=True)
    today     = datetime.now().date()
    file_date = datetime.fromtimestamp(os.path.getmtime(log_path)).date()
    if file_date < today:
        name = (f"multi_agent_system_"
                f"{file_date.strftime(settings.LOG_DATE_FORMAT)}.log")
        path = os.path.join(archive_dir, name)
        if os.path.exists(path):
            name = (f"multi_agent_system_"
                    f"{file_date.strftime(settings.LOG_DATE_FORMAT)}"
                    f"_{int(time.time())}.log")
            path = os.path.join(archive_dir, name)
        os.rename(log_path, path)



log = logging.getLogger("multi_agent_orchestrator")


def configure_logging():
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_file = os.path.join(settings.LOG_DIR, "multi_agent_system.log")
    _rotate_daily_log(log_file)
    run_dir = Path(settings.LOG_DIR) / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / ("unified_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, encoding="utf-8"),
                  logging.FileHandler(run_log, encoding="utf-8")])
    log.info("[RunLog] path=%s", run_log.resolve())


# ── OHLC helper (fixes RSI=2.9 seed noise bug) ───────────────────────────────

def _get_ohlc_from_builder(builder):
    """Read actual closed OHLC only; never invent highs, lows, or opens."""
    return (builder.get_opens(), builder.get_highs(),
            builder.get_lows(), builder.get_closes())


# ── Session window helpers ────────────────────────────────────────────────────

# ── Main system ───────────────────────────────────────────────────────────────

class MultiAgentTradingSystem:

    TICK_INTERVAL     = 1.0
    DECISION_INTERVAL = 5.0
    ASSET_CHECK_EVERY = settings.ASSET_SCAN_INTERVAL

    def __init__(self, force_trades: bool = False,
                 training_mode: bool = False,
                 disable_circuit_breaker: bool = False,
                 min_confidence: float = MIN_CONFIDENCE,
                 trade_amount: Optional[float] = None,
                 payout_minimum: float = PAYOUT_MINIMUM,
                 max_trades_per_day: Optional[int] = None,
                 daily_loss_limit: Optional[float] = None,
                 observe_only: bool = False, max_candle_closes: int = 0,
                 max_runtime: float = 0, chart_view: str = "entry",
                 entry_mode: str = "hybrid",
                 enable_pattern_gate: bool = False,
                 pattern_score_threshold: int = 5):
        trade_amount = settings.TRADE_AMOUNT if trade_amount is None else trade_amount
        max_trades_per_day = settings.MAX_DAILY_TRADES if max_trades_per_day is None else max_trades_per_day
        configured_daily_loss = getattr(settings, "MAX_DAILY_LOSS_USD", None)
        start_balance = float(getattr(self, "_current_balance_proxy", lambda: 0.0)() or 0.0)
        if daily_loss_limit is None:
            if configured_daily_loss is not None and float(configured_daily_loss) > 0:
                daily_loss_limit = float(configured_daily_loss)
            else:
                daily_loss_limit = max(start_balance * 0.05, 10.0) if start_balance > 0 else 10.0
        if settings.RESET_TODAY_LOGS_ON_STARTUP:
            log.warning("[Risk] startup_reset_ignored=true reason=preserve_trade_accounting")
            settings.RESET_TODAY_LOGS_ON_STARTUP = False
        if not math.isfinite(float(trade_amount)) or float(trade_amount) <= 0:
            raise ValueError("Trade amount must be finite and positive")
        if chart_view not in {"entry", "trend"}:
            raise ValueError("Chart view must be entry or trend")
        allow_modes = {"classic", "short", "hybrid", "rsi_hookback"}
        if entry_mode not in allow_modes:
            entry_mode = "hybrid"
        self.entry_mode = entry_mode
        if entry_mode == "hybrid" and not enable_pattern_gate:
            enable_pattern_gate = True
        elif entry_mode in {"classic", "short", "rsi_hookback"} and enable_pattern_gate:
            enable_pattern_gate = bool(enable_pattern_gate)
        self.chart_view = chart_view
        self.bot            = OlymtradeBot()
        self.risk           = RiskManager(max_daily_trades=max_trades_per_day,
                                          max_daily_loss=daily_loss_limit,
                                          max_concurrent_open_trades=1)
        self.agent1         = MultiTimeframeAgent(settings.ASSET)
        self.candle_monitor = CandleMonitor()
        # RSI hook-back is the single live trading path. Legacy strategy agents are intentionally not wired into the execution loop.
        self.force_trades: bool = bool(force_trades)
        self.training_mode: bool = bool(training_mode)
        self.disable_circuit_breaker: bool = bool(disable_circuit_breaker)
        self.enable_pattern_gate: bool = bool(enable_pattern_gate)
        self.pattern_score_threshold: int = int(pattern_score_threshold)
        self.min_confidence: float = float(min_confidence)
        self.trade_amount: float = float(trade_amount)
        settings.TRADE_AMOUNT = self.trade_amount
        self.payout_minimum: float = float(payout_minimum)
        self.max_trades_per_day = self.risk.max_daily_trades
        self.daily_loss_limit = self.risk.max_daily_loss

        self.supervisor     = PipelineSupervisorAgent(
            self.bot, self.risk, self.agent1, None,
            asset_selector=None,
            force_trades=self.force_trades,
            disable_circuit_breaker=self.disable_circuit_breaker,
        )

        self.observe_only = observe_only
        self.max_candle_closes = max_candle_closes
        self.max_runtime = max_runtime
        self.closed_events = 0
        self._last_evaluation_key = None
        self._last_attempted_entry = None
        self._last_attempted_short_entry = None
        self._execution_lock = asyncio.Lock()
        self._feed_missing = False
        self._feed_generation = 0
        self._process_lock_held = False
        self._execution_uncertain = False
        self._flat_reconciled_date = None
        self._next_flat_reconciliation_at = 0.0
        self._quote = None
        self._decision_tasks = set()
        self._native_refresh_task = None
        self._accepted_count = 0
        self._seed_sources = {}
        self._rsi_buy_armed = True
        self._rsi_sell_armed = True
        self._pending_rsi_entry = None
        self._last_rsi_signal = None
        self._last_logged_candle_ts = None
        self._last_handled_candle_ts = None
        self._last_compact_tick_log_ts = 0.0
        self._last_warmup_log_ts = 0.0
        self._last_warmup_log_count = None
        self._local_timestamp_warning_logged = False
        self._clock_offset_ms = 0.0
        self._next_clock_offset_log_ts = 0.0
        self._pending_signal_timing = None
        self._last_native_eval_ts = 0.0
        self._last_candle_monitor_log_ts = 0.0
        self._ws_tick_count_1m = 0
        self._fallback_tick_count_1m = 0
        self._last_1m_candle_source = "ws_ticks"
        self.active_trades: List[Dict] = []
        self._order_state_path = Path(settings.LOG_DIR) / "execution_state.json"
        if self._order_state_path.exists():
            try:
                state = json.loads(self._order_state_path.read_text())
                if state and state.get("status") == "accepted":
                    self.active_trades = [state["order"]]
                    order = state["order"]
                    self.risk.record_trade(TradeRecord(
                        timestamp=order["timestamp"], asset=order.get("asset", settings.ASSET),
                        mode="ftt", signal=order["signal"], confidence=0.95,
                        amount=order["amount"], state_key=order.get("state_key", "")))
                elif state:
                    self._execution_uncertain = True
            except (ValueError, KeyError, OSError):
                self._execution_uncertain = True
            if self.active_trades or self._execution_uncertain:
                log.warning("[Execution] pending_state_restored=true entries=paused_until_reconciled")
        self.running = False
        self._last_decision_ts:    float = 0.0
        self._last_asset_check_ts: float = 0.0
        self._current_asset: str   = settings.ASSET
        self._bg_tasks: List[asyncio.Task] = []
        self._decision_log_path = Path(settings.DECISION_LOG_PATH)

        self._loss_halt_duration_sec = 300.0
        snapshot = self._sync_daily_accounting()
        self.supervisor.cb.restore_from_outcomes(snapshot["strategy"]["recent_outcomes"])
        self._log_daily_accounting("restored")

    def _sync_daily_accounting(self):
        """Daily controls derive from durable records, including pending orders."""
        snapshot = self.risk.daily_snapshot()
        self._daily_loss = snapshot["gross_loss"]
        self._daily_profit = snapshot["gross_profit"]
        self._trades_today = snapshot["accepted_count"]
        self._settled_today = snapshot["settled_count"]
        self._wins_today = snapshot["wins"]
        self._consecutive_losses = snapshot["strategy"]["consecutive_losses"]
        last_loss = snapshot["strategy"]["last_loss_at"]
        self._loss_halt_time = datetime.fromisoformat(last_loss).timestamp() if last_loss else 0.0
        self._account_snapshot = snapshot
        return snapshot

    def _log_daily_accounting(self, event):
        snapshot = self._sync_daily_accounting()
        strategy = snapshot["strategy"]
        log.info("[DailyRisk] %s", json.dumps({
            "event": event, "date": snapshot["date"], "source": "trade_journal",
            "max_daily_trades": self.max_trades_per_day,
            "max_daily_loss": self.daily_loss_limit, "loss_basis": "gross_loss_plus_pending",
            "accepted": snapshot["accepted_count"], "settled": snapshot["settled_count"],
            "gross_loss": snapshot["gross_loss"], "net_pnl": snapshot["net_pnl"],
            "pending_exposure": snapshot["pending_exposure"],
            "historical_unconfirmed_ftt_count": snapshot["historical_unconfirmed_ftt_count"],
            "strategy_settled": strategy["settled_count"],
            "strategy_wins": strategy["wins"], "demo_tests_in_strategy_stats": False,
        }, sort_keys=True))

    def _persist_rsi_state(self):
        state = {"armed": {"BUY": bool(self._rsi_buy_armed), "SELL": bool(self._rsi_sell_armed)},
                 "last_handled_candle_ts": self._last_handled_candle_ts}
        path = Path(settings.LOG_DIR) / "rsi_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True)

    def _restore_rsi_state(self):
        path = Path(settings.LOG_DIR) / "rsi_state.json"
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (TypeError, ValueError, OSError):
            return
        if isinstance(state, dict):
            armed = state.get("armed") or {}
            self._rsi_buy_armed = bool(armed.get("BUY", self._rsi_buy_armed))
            self._rsi_sell_armed = bool(armed.get("SELL", self._rsi_sell_armed))
            handled = state.get("last_handled_candle_ts")
            if handled is not None:
                try:
                    self._last_handled_candle_ts = float(handled)
                except (TypeError, ValueError):
                    self._last_handled_candle_ts = None

    def _check_consecutive_loss_halt(self) -> Tuple[bool, str]:
        self._sync_daily_accounting()
        remaining = self._loss_halt_duration_sec - (time.time() - self._loss_halt_time)
        if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES and remaining > 0:
            reason = f"Loss halt active ({self._consecutive_losses} losses), {remaining:.0f}s left"
            log.warning("[DailyRisk] %s", reason)
            return True, reason
        return False, ""

    def _trade_capacity(self) -> Tuple[int, int, float]:
        snapshot = self._sync_daily_accounting()
        loss_budget = snapshot["loss_budget_remaining"]
        return snapshot["trades_left"], int(loss_budget // self.trade_amount), loss_budget

    def _check_circuit_breaker(self) -> bool:
        snapshot = self._sync_daily_accounting()
        blocked = snapshot["halted"] or snapshot["trades_left"] <= 0
        if blocked:
            log.warning("[DailyRisk] status=blocked accepted=%s gross_loss=%s pending_exposure=%s",
                        snapshot["accepted_count"], snapshot["gross_loss"], snapshot["pending_exposure"])
        return blocked

    def _reset_daily_counters(self):
        # RiskManager detects rollover; never discard a pending order or reset the journal.
        snapshot = self._sync_daily_accounting()
        self.supervisor.cb.restore_from_outcomes(snapshot["strategy"]["recent_outcomes"])
        self._log_daily_accounting("rollover")

    def _record_trade_outcome(self, won: bool, pnl: float):
        # Settlement was already committed by RiskManager. Do not count it twice.
        self._log_daily_accounting("settled")

    @staticmethod
    def _round_to_step(value: float, step: float) -> float:
        if not math.isfinite(value) or not math.isfinite(step) or step <= 0:
            return float(value)
        return float(math.ceil(value / step - 1e-9) * step)

    @classmethod
    def _stake_from_range(cls, balance: float, baseline_range: float, current_range: float,
                          *, min_stake: float = 1.0, step: float = 1.0) -> float:
        if not math.isfinite(balance) or balance <= 0:
            return float(min_stake)
        baseline_range = max(float(baseline_range), 1e-9)
        current_range = max(float(current_range), 1e-9)
        factor = current_range / baseline_range
        factor = min(1.0, max(0.25, factor))
        stake = balance * settings.BASE_STAKE_PCT * factor
        minimum = max(float(min_stake), 1.0)
        stake = max(stake, minimum)
        if step and step > 0:
            stake = max(minimum, cls._round_to_step(stake, step))
        return float(stake)

    def _volatility_stake(self, *, balance: Optional[float] = None) -> Tuple[float, float, float]:
        base_stake = float(settings.TRADE_AMOUNT if balance is None else balance)
        if not math.isfinite(base_stake) or base_stake <= 0:
            base_stake = float(settings.TRADE_AMOUNT)
        current = self.agent1.builders[60].closed_snapshot()
        if len(current) >= 3:
            recent_ranges = [float(candle["high"] - candle["low"]) for candle in current[-3:]]
            current_range = sum(recent_ranges) / max(len(recent_ranges), 1)
        elif current:
            current_range = float(current[-1]["high"] - current[-1]["low"])
        else:
            current_range = 0.001
        history = self.agent1.builders[60].closed_snapshot()
        if len(history) >= 100:
            all_ranges = [float(candle["high"] - candle["low"]) for candle in history[-100:]]
            baseline_range = float(np.median(all_ranges)) if all_ranges else current_range
        else:
            baseline_range = current_range
        base_stake = max(base_stake, settings.STAKE_MIN)
        factor = min(1.0, max(0.25, current_range / max(baseline_range, 1e-9)))
        final_stake = self._stake_from_range(base_stake, baseline_range, current_range,
                                            min_stake=settings.STAKE_MIN, step=settings.STAKE_STEP)
        if final_stake < settings.STAKE_MIN:
            final_stake = settings.STAKE_MIN
        return base_stake, factor, float(final_stake)

    def _log_decision_row(self, *, session: str, gate_name: str, direction: str, passed_rules: list,
                          failed_rules: list, metrics: dict, stake: Optional[float], factor: Optional[float]):
        path = Path(settings.DECISION_LOG_PATH)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = csv.writer(path.open("a", newline="", encoding="utf-8"))
            if not path.exists() or path.stat().st_size == 0:
                writer.writerow([
                    "timestamp", "session", "gate_name", "direction", "passed_rules", "failed_rules",
                    "ema9", "ema21", "ema50", "rsi", "rsi_dir", "atr_1m", "confirmation_range",
                    "range_ratio", "nearest_resistance", "nearest_support", "tick_count_5s",
                    "tick_count_10s", "tick_count_15s", "tick_count_20s", "tick_count_30s",
                    "tick_count_60s", "stake", "factor", "result", "profit", "close_price"
                ])
            tick_counts = {
                period: 0 for period in (5, 10, 15, 20, 30, 60)
            }
            for period in (5, 10, 15, 20, 30, 60):
                if isinstance(metrics.get("tick_counts"), dict):
                    tick_counts[period] = int(metrics["tick_counts"].get(period, 0) or 0)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(timespec="microseconds"), session, gate_name, direction,
                ";".join(passed_rules), ";".join(failed_rules), metrics.get("ema9"), metrics.get("ema21"),
                metrics.get("ema50"), metrics.get("rsi"), metrics.get("rsi_dir"), metrics.get("atr_1m"),
                metrics.get("confirmation_range"), metrics.get("range_ratio"), metrics.get("nearest_resistance"),
                metrics.get("nearest_support"), tick_counts[5], tick_counts[10], tick_counts[15],
                tick_counts[20], tick_counts[30], tick_counts[60], stake, factor, "", "", ""
            ])
        except Exception:
            log.warning("[DecisionLog] status=write_failed gate=%s direction=%s", gate_name, direction, exc_info=True)

    @staticmethod
    def _compute_rsi_wilder(closes: List[float], period: int = RSI_PERIOD) -> List[Optional[float]]:
        if len(closes) <= period:
            return [None for _ in closes]
        changes = [float(right) - float(left) for left, right in zip(closes, closes[1:])]
        gains = []
        losses = []
        for change in changes[:period]:
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            elif change < 0:
                gains.append(0.0)
                losses.append(-change)
            else:
                gains.append(0.0)
                losses.append(0.0)
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        def to_rsi(gain: float, loss: float) -> float:
            if loss == 0:
                return 100.0 if gain > 0 else 50.0
            return 100.0 - 100.0 / (1.0 + (gain / loss))

        result = [None] * len(closes)
        result[period] = to_rsi(avg_gain, avg_loss)
        for index in range(period + 1, len(closes)):
            change = changes[index - 1]
            avg_gain = ((avg_gain * (period - 1)) + max(change, 0.0)) / period
            avg_loss = ((avg_loss * (period - 1)) + max(-change, 0.0)) / period
            result[index] = to_rsi(avg_gain, avg_loss)
        return result

    def _evaluate_rsi_hookback(self, candle: dict, rsi_prev: Optional[float], rsi_now: Optional[float]):
        signal = None
        if rsi_prev is None or rsi_now is None:
            return {"signal": None, "armed": {"BUY": self._rsi_buy_armed, "SELL": self._rsi_sell_armed}, "rsi_prev": rsi_prev, "rsi_now": rsi_now, "time": candle.get("ts")}

        if rsi_prev <= RSI_LOW and rsi_now > RSI_LOW and self._rsi_buy_armed:
            signal = "BUY"
            self._rsi_buy_armed = False
        elif rsi_prev >= RSI_HIGH and rsi_now < RSI_HIGH and self._rsi_sell_armed:
            signal = "SELL"
            self._rsi_sell_armed = False

        if not signal:
            if not self._rsi_buy_armed and rsi_now >= RSI_RESET:
                self._rsi_buy_armed = True
            if not self._rsi_sell_armed and rsi_now <= RSI_RESET:
                self._rsi_sell_armed = True

        return {
            "signal": signal,
            "armed": {"BUY": self._rsi_buy_armed, "SELL": self._rsi_sell_armed},
            "rsi_prev": rsi_prev,
            "rsi_now": rsi_now,
            "time": candle.get("ts"),
        }

    def _check_rsi_risk_gate(self) -> str:
        snapshot = self.risk.daily_snapshot()
        strategy = snapshot.get("strategy", {})
        consecutive_losses = int(strategy.get("consecutive_losses", 0) or 0)
        if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return "consecutive_losses_pause"
        last_loss_at = strategy.get("last_loss_at")
        if last_loss_at:
            try:
                delta_seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(last_loss_at).astimezone(timezone.utc)).total_seconds()
            except ValueError:
                delta_seconds = LOSS_PAUSE_MINUTES * 60 + 1
            if delta_seconds < LOSS_PAUSE_MINUTES * 60 and consecutive_losses > 0:
                return "loss_pause"
        balance = self._current_balance_proxy()
        if balance and balance > 0:
            daily_loss_pct = (snapshot.get("gross_loss", 0.0) / balance) * 100.0
            if daily_loss_pct >= DAILY_LOSS_LIMIT_PERCENT:
                return "daily_loss_limit"
        return "ok"

    def _current_balance_proxy(self) -> float:
        bot = getattr(self, "bot", None)
        if bot is not None and hasattr(bot, "_page") and bot._page is not None:
            try:
                selectors = [
                    "[data-test=\"account-balance\"]",
                    "[data-test=\"account-balance-title\"]",
                    "text=Balance",
                ]
                for selector in selectors:
                    locator = bot._page.locator(selector).first
                    if locator is None:
                        continue
                    try:
                        text = locator.inner_text(timeout=1000)
                    except Exception:
                        continue
                    match = re.search(r"[0-9][0-9,\.]*", text or "")
                    if match:
                        candidate = float(match.group(0).replace(",", ""))
                        if math.isfinite(candidate) and candidate > 0:
                            return candidate
            except Exception:
                pass
        fallback = max(float(getattr(self, "trade_amount", settings.TRADE_AMOUNT) or settings.TRADE_AMOUNT), 100.0)
        return max(fallback * 100.0, 1000.0)

    def _append_trade_row(self, *, direction: str, rsi_prev: Optional[float], rsi_now: Optional[float],
                          entry_price: Optional[float], payout: Optional[float], stake: Optional[float],
                          result: str = "", pnl: float = 0.0):
        path = Path(settings.LOG_DIR) / "trades.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not file_exists:
                writer.writerow(["time", "direction", "rsi_prev", "rsi_now", "entry_price", "payout", "stake", "result", "pnl"])
            writer.writerow([
                datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                direction,
                "" if rsi_prev is None else f"{float(rsi_prev):.2f}",
                "" if rsi_now is None else f"{float(rsi_now):.2f}",
                "" if entry_price is None else f"{float(entry_price):.5f}",
                "" if payout is None else f"{float(payout):.2f}",
                "" if stake is None else f"{float(stake):.2f}",
                result,
                f"{float(pnl):.2f}",
            ])

    def _write_decision_log_async(self, **kwargs):
        thread = threading.Thread(target=self._log_decision_row, kwargs=kwargs, daemon=True)
        thread.start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _missing_history_for_bootstrap(counts: Optional[dict]) -> dict:
        required = {5: 10, 10: 15, 15: 15}
        if not isinstance(counts, dict):
            counts = {}
        missing = {}
        for period, needed in required.items():
            have = int(counts.get(period, 0) or 0)
            shortfall = needed - have
            if shortfall > 0:
                missing[period] = shortfall
        return missing

    @staticmethod
    def _should_block_startup(missing: dict) -> bool:
        """Short warmup gaps are bootstrap conditions, not fatal startup failures."""
        return False

    def _calculate_consensus_score(self, tf_state: dict,
                                   direction: str) -> float:
        if not tf_state:
            return 0.0
        sup   = sum(1 for d in tf_state.values()
                    if isinstance(d, dict) and d.get("vote") == direction)
        total = sum(1 for d in tf_state.values() if isinstance(d, dict))
        return sup / total if total > 0 else 0.0

    def _combine_confidences(self, a1: float, rl: float,
                              experienced: bool, consensus: float,
                              chart: float) -> float:
        # Chart is now the PRIMARY signal (EMA+Aroon strategy)
        # Agent1 MTF is supporting evidence
        if experienced:
            fused = 0.40 * a1 + 0.20 * rl + 0.40 * chart
        else:
            fused = 0.45 * a1 + 0.55 * chart
        fused *= (consensus * 0.4 + 0.6)   # softer consensus discount
        return float(min(0.97, max(0.45, fused)))

    def _offset_corrected_time(self) -> float:
        offset_ms = float(getattr(self, "_clock_offset_ms", 0.0) or 0.0)
        return time.time() + (offset_ms / 1000.0)

    def _resolve_quote_tick_ts(self, quote: Optional[dict]) -> Optional[float]:
        if not isinstance(quote, dict):
            return None
        now = time.time()
        for field in ("server_timestamp", "server_time", "broker_timestamp", "timestamp", "ts"):
            value = quote.get(field)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if 1.5e9 <= value <= 2.5e9 and abs(now - value) <= 300:
                return value
            if 1.5e12 <= value <= 2.5e12 and abs(now * 1000.0 - value) <= 300000:
                return value
        observed = quote.get("observed_at", quote.get("ts", time.time()))
        try:
            observed = float(observed)
        except (TypeError, ValueError):
            observed = time.time()
        if not math.isfinite(observed):
            observed = time.time()
        if not getattr(self, "_local_timestamp_warning_logged", False):
            log.warning("[Feed] status=local_receive_time source=%s candle_alignment=receive_clock tick_lag_threshold_seconds=2", quote.get("source", "fallback"))
            self._local_timestamp_warning_logged = True
        boundary = math.floor(observed / 60.0) * 60.0
        if observed > boundary + 2.0:
            return None
        return observed

    def _log_signal_timing(self, *, direction: str, candle_close_time: float, signal_detected_time: float, order_sent_time: float):
        detection_delay_ms = (signal_detected_time - candle_close_time) * 1000.0
        order_delay_ms = (order_sent_time - signal_detected_time) * 1000.0
        log.info("[Timing] direction=%s candle_close_time=%.3f signal_detected_time=%.3f order_sent_time=%.3f detection_delay_ms=%.0f order_delay_ms=%.0f",
                 direction, candle_close_time, signal_detected_time, order_sent_time, detection_delay_ms, order_delay_ms)

    async def _get_price(self):
        """Return only a current observation of the selected broker instrument."""
        self._quote = await self.bot.get_current_quote(self._current_asset)
        if self._quote is None:
            return None
        price = self._quote.get("price")
        observed = self._resolve_quote_tick_ts(self._quote)
        if observed is None:
            self._quote = None
            return None
        age = time.time() - observed
        if price is None or not math.isfinite(price) or price <= 0 or not 0 <= age <= 5:
            self._quote = None
            return None
        self._quote["tick_ts"] = observed
        return float(price)

    async def setup(self) -> bool:
        self._restore_rsi_state()
        log.info("[Setup] strategy=rsi_hookback rsi_period=14 low=35 high=65 reset=50 duration=60 stake_pct=1.0")
        if settings.FTT_DURATION != FTT_DURATION_S:
            log.error("[Setup] status=blocked reason=duration_mismatch configured=%s required=%s", settings.FTT_DURATION, FTT_DURATION_S)
            return False
        await self.bot.start()
        asyncio.create_task(self.bot.diagnose_ws_feed(60))
        if not await self.bot.login():
            log.error("[Setup] status=blocked reason=login_failed")
            return False
        self.supervisor._current_asset = self._current_asset
        self.candle_monitor.set_asset(self._current_asset)
        restored = self.candle_monitor.restore_closed_history(path=settings.CANDLE_STORE_PATH, asset=self._current_asset)
        counts = {period: len(self.candle_monitor._builders[period].closed_snapshot()) for period in (5, 10, 15, 20, 30, 60) if period in self.candle_monitor._builders}
        log.info("[CandleHistory] asset=%s restored=%s loaded_counts=%s source=%s", self._current_asset, restored, counts, settings.CANDLE_STORE_PATH)
        missing = self._missing_history_for_bootstrap(counts)
        if missing:
            log.warning("[StartupWarmup] status=bootstrap_pending asset=%s restored=%s missing=%s live_feed_can_fill=true",
                        self._current_asset, restored, missing)
        else:
            log.info("[StartupWarmup] status=ready asset=%s restored=%s warmup_ok=true", self._current_asset, restored)
        # Avoid repeated asset switches when the active title already proves identity.
        quote = await self.bot.get_current_quote(self._current_asset)
        if quote is None:
            if not await self.bot.select_asset(self._current_asset):
                log.error("[Setup] status=blocked reason=asset_selection_failed")
                return False
            quote = await self.bot.get_current_quote(self._current_asset)
        if quote is None:
            log.error("[Setup] status=blocked reason=selected_asset_quote_unavailable")
            return False
        if settings.TRAINING_MODE and not await self.bot.is_demo_account():
            log.error("[Setup] status=blocked reason=training_mode_requires_demo_account")
            return False
        if not self.observe_only and not await self.bot.is_demo_account():
            log.error("[Setup] status=blocked reason=demo_account_not_verified")
            return False
        await self._reconcile_flat_account()
        display = await configure_strategy_chart(self.bot._page, self.chart_view)
        log.info("[ChartDisplay] %s", json.dumps(display, sort_keys=True))
        log.info("[StrategyData] source=broker_native_ws trend_seconds=30 entry_seconds=60 "
                 "display_view=%s trend_structure_required=false", self.chart_view)
        native_periods = set()
        # No Yahoo OTC mapping, fabricated 30s bars, or randomized bootstrap.
        for period in (30, 60):
            history = await self.bot.get_native_candles(self._current_asset, period, 180)
            if history is not None and not history.empty:
                native_periods.add(period)
                source = history.attrs.get("source", "broker_native")
                self.agent1.seed_from_dataframe(history, timeframe=period, source=source, log_seed=False)
                self._seed_sources[period] = source
            else:
                self._seed_sources[period] = "broker_browser_sample"
            log.info("[Seed] %s", json.dumps({
                "asset": self._current_asset, "timeframe_seconds": period,
                "source": self._seed_sources[period],
                "native_history_available": history is not None and not history.empty,
                "closed_candles": len(self.agent1.builders[period]),
                "required_closed": 53 if period == 30 else 21,
                "m1_upsampling": False,
            }, sort_keys=True))
        if native_periods != {30, 60}:
            log.info("[DataConstraint] native_history_missing_for=%s; live browser "
                     "samples are observed quotes, not broker-native tick-complete OHLC; "
                     "warmup requires 53 closed 30s bars and 21 closed 1m bars",
                     sorted({30, 60} - native_periods))
            log.warning("[StartupData] source=polled_1hz low_information=true for 5s/10s/15s candles; no per-tick broker feed detected")
        self._log_rsi_self_check()
        await self._reconcile_pending_entries(timeout_s=60.0)
        self._last_evaluation_key = (self.agent1.builders[30].latest_closed_ts,
                                     self.agent1.builders[60].latest_closed_ts)
        self._bg_tasks.append(asyncio.create_task(self._daily_reset_monitor()))
        self._bg_tasks.append(asyncio.create_task(self._settlement_loop()))
        return True

    # ── Background tasks ──────────────────────────────────────────────────────

    def _log_rsi_self_check(self):
        candles = self.agent1.builders[60].closed_snapshot()
        closes = [float(c["close"]) for c in candles[-3:]]
        if not closes:
            log.info("[RSIStartupCheck] status=empty history=0")
            return
        rsi_values = self._compute_rsi_wilder([float(c["close"]) for c in candles], period=RSI_PERIOD)
        last = candles[-3:]
        for candle in last:
            idx = candles.index(candle)
            rsi = rsi_values[idx] if idx < len(rsi_values) else None
            ts = datetime.fromtimestamp(float(candle.get("ts", 0.0)), timezone.utc).astimezone().isoformat(timespec="seconds")
            log.info("[RSIStartupCheck] time=%s close=%.5f rsi=%s", ts, float(candle.get("close", 0.0)), "n/a" if rsi is None else f"{rsi:.2f}")

    async def _reconcile_pending_entries(self, timeout_s: float = 60.0):
        deadline = time.monotonic() + timeout_s
        journal_path = Path(settings.LOG_DIR) / "trade_journal.csv"
        entries = []
        if journal_path.exists():
            try:
                with journal_path.open("r", newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    entries = [row for row in reader if (row.get("result") or "").upper() == "PENDING"]
            except Exception as exc:
                log.warning("[Reconcile] status=paused reason=journal_unreadable details=%s", exc)
                return
        by_outcome = {"kept": 0, "settled": 0, "expired_unreconciled": 0}
        kept_amount = settled_amount = expired_amount = 0.0
        still_pending = []
        for row in entries:
            if time.monotonic() >= deadline:
                log.warning("[Reconcile] status=paused reason=timeout_elapsed timeout_seconds=%s pending=%s", timeout_s, len(entries))
                return
            ts = str(row.get("timestamp", ""))
            try:
                opened = datetime.fromisoformat(ts).timestamp()
            except ValueError:
                opened = None
            if opened is None or (time.time() - opened) <= (FTT_DURATION_S + 120):
                still_pending.append(row)
                kept_amount += float(row.get("amount", 0.0) or 0.0)
                by_outcome["kept"] += 1
                continue
            broker_match = None
            try:
                history = await self.bot._order_rows(include_history=True)
                for entry in history:
                    if entry.get("asset") and self._current_asset and entry.get("asset") != self._current_asset:
                        continue
                    if (row.get("signal") or "").upper() and str(entry.get("direction", "")).upper() != (row.get("signal") or "").upper():
                        continue
                    if abs(float(entry.get("amount", 0.0) or 0.0) - float(row.get("amount", 0.0) or 0.0)) <= 0.01:
                        broker_match = entry
                        break
            except Exception:
                broker_match = None
            if broker_match is not None:
                result = str((broker_match.get("status") or "").upper())
                if result in {"WIN", "LOSS", "DRAW"}:
                    settled_amount += float(row.get("amount", 0.0) or 0.0)
                    by_outcome["settled"] += 1
                    self.risk.update_pnl_and_journal(ts, result, float(broker_match.get("pnl", 0.0) or 0.0))
                    continue
            expired_amount += float(row.get("amount", 0.0) or 0.0)
            by_outcome["expired_unreconciled"] += 1
            row["result"] = "expired_unreconciled"
            row["pnl"] = "0.0"
            row["settled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.risk._rewrite_journal_result(ts, "expired_unreconciled", 0.0)
        if not still_pending:
            log.info("[Reconcile] pending_total=0 kept=0 settled=0 expired_unreconciled=0 kept_amount=0.00 settled_amount=0.00 expired_amount=0.00")
        else:
            log.info("[Reconcile] pending_total=%s kept=%s settled=%s expired_unreconciled=%s kept_amount=%.2f settled_amount=%.2f expired_amount=%.2f",
                     len(entries), by_outcome["kept"], by_outcome["settled"], by_outcome["expired_unreconciled"],
                     kept_amount, settled_amount, expired_amount)
        snapshot = self.risk.daily_snapshot()
        self.risk._snapshot = snapshot
        self._account_snapshot = snapshot
        if snapshot["pending_exposure"] <= 0.0:
            log.info("[Reconcile] entries resumed pending_exposure=0.00")
        else:
            log.info("[Reconcile] entries resumed pending_exposure=%.2f", snapshot["pending_exposure"])
        self._log_daily_accounting("reconciled")

    async def _daily_replay_and_save(self):
        await asyncio.sleep(30)
        while True:
            try:
                self.agent2.learn_from_logs(self.risk.journal_path)
                self.agent2.q.save(self.agent2.qtable_path)
                log.info("[DailyTrain] RL replayed and Q-table saved")
                await asyncio.sleep(24 * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"[DailyTrain] {e}")
                await asyncio.sleep(600)

    async def _daily_reset_monitor(self):
        """Reset circuit breaker and daily counters at midnight."""
        while True:
            try:
                now   = datetime.now()
                # Sleep until next midnight
                secs  = (86400 - (now.hour*3600 + now.minute*60 + now.second))
                await asyncio.sleep(secs)
                self._reset_daily_counters()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"[DailyReset] {e}")
                await asyncio.sleep(3600)

    # ── Shared data and decision loop ────────────────────────────────────────

    async def run(self, lock_acquired=False):
        with (nullcontext() if lock_acquired else ProcessLock()):
            self._process_lock_held = True
            try:
                if not await self.setup():
                    return
                self.running = True
                started = time.monotonic()
                log.info("[Run] poll_seconds=%s gate_check_seconds=%s "
                         "entry_mode=%s monitor_seconds=5,10,15 pid=%s",
                         self.TICK_INTERVAL,
                         self.TICK_INTERVAL, getattr(self, "entry_mode", "classic"), os.getpid())
                while self.running:
                    tick_start = time.monotonic()
                    if self.max_runtime and tick_start - started >= self.max_runtime:
                        log.info("[Run] status=finished reason=max_runtime closed_events=%s", self.closed_events)
                        break
                    price = await self._get_price()
                    if price is None:
                        if not self._feed_missing:
                            self._feed_missing = True
                            self._feed_generation = getattr(self, "_feed_generation", 0) + 1
                            await self._cancel_native_refresh()
                            self.agent1.invalidate("quote_unavailable")
                            self.candle_monitor.invalidate("quote_unavailable")
                            self._last_evaluation_key = None
                            log.warning("[Feed] status=unavailable entries=paused history=invalidated")
                        self._feed_missing = True
                        await asyncio.sleep(self.TICK_INTERVAL)
                        continue
                    if self._feed_missing:
                        log.info("[Feed] status=recovered warmup=restarted")
                        self._feed_missing = False
                    tick_ts = self._quote.get("tick_ts", self._quote.get("observed_at", self._quote.get("ts", time.time())))
                    source = "ws_ticks" if self._quote.get("source") == "ws_ticks" else "fallback"
                    self.candle_monitor.add_tick(price, tick_ts, source=source)
                    closed = await self._update_closed_candles(price, wait_for_native=False)
                    if closed.intersection({30, 60}):
                        if time.monotonic() - getattr(self, "_last_candle_monitor_log_ts", 0.0) >= 60.0:
                            self.closed_events += 1
                            log.info("[CandleMonitor] %s", json.dumps({
                                "asset": self._current_asset, "observed_at": self._quote["observed_at"],
                                "check_interval_seconds": self.TICK_INTERVAL, "live_price": price,
                                "used_for_entries": False,
                                "timeframes": self.candle_monitor.snapshot(time.time()),
                            }, sort_keys=True, allow_nan=False))
                            self._last_candle_monitor_log_ts = time.monotonic()
                    else:
                        now_ts = time.time()
                        if now_ts - getattr(self, "_last_compact_tick_log_ts", 0.0) >= 60.0:
                            snapshot = self.candle_monitor.snapshot(now_ts)
                            compact = {period: {"source": state.get("source"), "status": state.get("status"), "ready": state.get("ready"), "closed_count": state.get("closed_count"), "low_information": state.get("low_information")} for period, state in snapshot.items() if period in {"5s", "10s", "15s"}}
                            log.debug("[CompactTick] time=%s price=%.5f snapshot=%s",
                                      datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%H:%M:%S %Z"),
                                      price, json.dumps(compact, sort_keys=True, allow_nan=False))
                            self._last_compact_tick_log_ts = now_ts
                    native_event_ts = float(getattr(self.bot, "_last_native_candle_update", 0.0) or 0.0)
                    if native_event_ts and native_event_ts > self._last_native_eval_ts:
                        self._last_native_eval_ts = native_event_ts
                        task = asyncio.create_task(self._evaluate_and_trade(price))
                        self._decision_tasks.add(task)
                        task.add_done_callback(self._decision_finished)
                    if self.max_candle_closes and self.closed_events >= self.max_candle_closes:
                        log.info("[Run] status=finished reason=candle_close_target closed_events=%s", self.closed_events)
                        break
                    await asyncio.sleep(max(0.01, self.TICK_INTERVAL - (time.monotonic() - tick_start)))
            finally:
                await self._shutdown()
                self._process_lock_held = False

    async def _cancel_native_refresh(self):
        task = getattr(self, "_native_refresh_task", None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._native_refresh_task = None

    async def _update_closed_candles(self, price, *, wait_for_native=True):
        """Sample every quote; keep slow native replies off the polling path.

        The awaited mode is retained for callers needing a completed refresh.
        The run loop allows only one native refresh in flight at a time.
        """
        closed = set()
        for period in (30, 60):
            if self._seed_sources.get(period) != "broker_native_ws":
                tick_ts = self._quote.get("tick_ts", self._quote.get("observed_at", self._quote.get("ts", time.time())))
                source = "ws_ticks" if self._quote.get("source") == "ws_ticks" else "fallback"
                if self.agent1.builders[period].add_tick(price, tick_ts, source=source):
                    closed.add(period)
        task = getattr(self, "_native_refresh_task", None)
        if task is not None and (task.done() or wait_for_native):
            closed.update(await task)
            self._native_refresh_task = None
        if wait_for_native:
            closed.update(await self._refresh_native_candles())
        elif getattr(self, "_native_refresh_task", None) is None:
            self._native_refresh_task = asyncio.create_task(self._refresh_native_candles())
        return closed

    async def _refresh_native_candles(self):
        """Refresh native intervals only after their next actual close."""
        closed = set()
        now = time.time()
        for period in (30, 60):
            builder = self.agent1.builders[period]
            if self._seed_sources.get(period) == "broker_native_ws":
                expected = math.floor(now / period) * period - period
                previous = builder.latest_closed_ts
                if previous is not None and previous >= expected:
                    continue
                history = await self.bot.get_native_candles(self._current_asset, period, 180)
                if history is None or history.empty:
                    log.warning("[NativeHistory] status=unavailable timeframe=%s entries=waiting", period)
                    continue
                self.agent1.seed_from_dataframe(history, timeframe=period,
                                                source="broker_native_ws", log_seed=False)
                if builder.latest_closed_ts is not None and builder.latest_closed_ts != previous:
                    closed.add(period)
        return closed

    def _decision_finished(self, task):
        self._decision_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("[Decision] status=error reason=%s", task.exception())
            self._execution_uncertain = True
            self.running = False

    def _pattern_gate_candidates(self, trend, entry, current_price):
        """Extra opportunity layer that keeps the default gate intact unless enabled."""
        if not self.enable_pattern_gate:
            return []
        pattern_results = evaluate_pattern_gate(trend, entry)
        candidates = []
        for direction, gate in pattern_results.items():
            if not gate.get("passed"):
                continue
            if gate.get("score", 0) < self.pattern_score_threshold:
                continue
            payload = dict(gate, asset=self._current_asset, live_price=current_price,
                           entry_mode=self.entry_mode, strategy="pattern",
                           source="pattern_gate")
            log.info("[PatternGate] %s", json.dumps(payload, sort_keys=True, allow_nan=False))
            candidates.append(dict(direction=direction, strategy="pattern",
                                   entry_ts=entry[-1]["ts"], entry_period=60,
                                   snapshots=((trend, 30), (entry, 60))))
        return candidates

    async def _evaluate_and_trade(self, current_price: float):
        feed_generation = getattr(self, "_feed_generation", 0)
        candles = self.agent1.builders[60].closed_snapshot()
        if len(candles) < 30:
            if time.monotonic() - getattr(self, "_last_warmup_log_ts", 0.0) >= 60.0:
                log.info("[RSIHookBack] warming up (%s/30)", len(candles))
                self._last_warmup_log_ts = time.monotonic()
            self._rsi_buy_armed = True
            self._rsi_sell_armed = True
            self._last_rsi_signal = None
            return

        closes = [float(candle["close"]) for candle in candles]
        rsi_values = self._compute_rsi_wilder(closes, period=RSI_PERIOD)
        rsi_prev = rsi_values[-2] if len(rsi_values) >= 2 else None
        rsi_now = rsi_values[-1] if rsi_values else None
        candle = candles[-1]
        candle_ts = float(candle.get("ts", 0.0))
        source = str(candle.get("source", "fallback"))
        tick_count = int(candle.get("tick_count", 1) or 1)
        if rsi_prev is None or rsi_now is None:
            if getattr(self, "_last_logged_candle_ts", None) != candle_ts:
                log.info("[RSIHookBack] time=%s close=%.5f source=%s tick_count=%s status=skipped reason=insufficient_rsi_history candles=%s rsi_prev=%s rsi_now=%s",
                         datetime.fromtimestamp(float(candle.get("ts", time.time())), timezone.utc).astimezone().isoformat(timespec="seconds"),
                         float(candle.get("close", 0.0)), source, tick_count, len(candles), rsi_prev if rsi_prev is not None else "n/a", rsi_now if rsi_now is not None else "n/a")
                self._last_logged_candle_ts = candle_ts
            return
        landing = self._evaluate_rsi_hookback(candle, rsi_prev, rsi_now)
        signal = landing.get("signal")
        local_time = datetime.fromtimestamp(float(candle.get("ts", time.time())), timezone.utc).astimezone()
        if getattr(self, "_last_logged_candle_ts", None) != candle_ts:
            self._last_logged_candle_ts = candle_ts
            log.info("[RSIHookBack] time=%s close=%.5f source=%s tick_count=%s rsi_now=%.2f rsi_prev=%.2f armed=%s signal=%s reason=%s",
                     local_time.isoformat(timespec="seconds"), float(candle.get("close", 0.0)), source, tick_count, rsi_now, rsi_prev,
                     json.dumps(landing["armed"], sort_keys=True), signal or "none",
                     "no_valid_hookback" if signal is None else "triggered")

        if signal is None:
            return

        signal_detected_time = self._offset_corrected_time()
        self._pending_signal_timing = {
            "direction": signal,
            "candle_close_time": float(candle.get("ts", 0.0)) + 60.0,
            "signal_detected_time": signal_detected_time,
        }

        last_handled = self._last_handled_candle_ts
        if last_handled is not None and float(candle.get("ts", 0.0)) <= float(last_handled):
            return
        if last_handled is not None and (self._offset_corrected_time() - float(last_handled)) > (ENTRY_WINDOW_SECONDS + 60):
            self._rsi_buy_armed = True
            self._rsi_sell_armed = True
            self._last_handled_candle_ts = None
            self._persist_rsi_state()
            return

        signal_range_pips = float(candle.get("high", candle.get("close", 0.0))) - float(candle.get("low", candle.get("close", 0.0)))
        min_range_pips = float(getattr(settings, "MIN_SIGNAL_CANDLE_RANGE_PIPS", 0.5) or 0)
        if min_range_pips > 0 and signal_range_pips < min_range_pips:
            log.info("[RSIHookBack] status=skipped reason=low_range candle_ts=%s source=%s range_pips=%.4f min_range_pips=%.4f", candle_ts, source, signal_range_pips, min_range_pips)
            return

        self._last_handled_candle_ts = candle_ts
        self._persist_rsi_state()
        log.info("[RSIHookBack] status=signal direction=%s candle_ts=%s source=%s tick_count=%s prev=%.2f now=%.2f", signal, candle_ts, source, tick_count, rsi_prev, rsi_now)
        next_open = float(candle.get("ts", 0.0)) + 60.0
        self._pending_rsi_entry = {
            "direction": signal,
            "entry_ts": next_open,
            "rsi_prev": rsi_prev,
            "rsi_now": rsi_now,
            "entry_price": float(candle.get("close", 0.0)),
        }
        log.info("[RSIHookBack] queued direction=%s next_open=%s", signal, datetime.fromtimestamp(next_open, timezone.utc).astimezone().isoformat(timespec="seconds"))

        if self._pending_rsi_entry:
            pending = self._pending_rsi_entry
            now = self._offset_corrected_time()
            delta = now - float(pending["entry_ts"])
            if delta > ENTRY_WINDOW_SECONDS:
                log.info("[Execution] status=blocked reason=missed_window entry_ts=%s delta_seconds=%.1f", pending["entry_ts"], delta)
                self._pending_rsi_entry = None
            elif 0 <= delta <= ENTRY_WINDOW_SECONDS:
                strategy = "rsi_hookback"
                direction = pending["direction"]
                self._pending_rsi_entry = None
                await self._execute_candidate(direction=direction, strategy=strategy, entry_ts=float(pending["entry_ts"]), entry_period=60,
                                             snapshots=((candles, 60),), feed_generation=feed_generation)
        return

    async def _execute_candidate(self, direction, strategy, entry_ts, entry_period,
                                 snapshots, feed_generation):
        """One shared risk, persistence, and order path for both entry strategies."""
        timing_state = self._pending_signal_timing
        close_ts = entry_ts + entry_period
        if not 0 <= self._offset_corrected_time() - close_ts <= self.DECISION_INTERVAL:
            log.info("[Execution] status=blocked reason=stale_confirmation entry_ts=%s", entry_ts)
            return
        if self.observe_only:
            log.info("[Execution] status=observe_only strategy=%s direction=%s entry_ts=%s",
                     strategy, direction, entry_ts)
            return
        if not self._process_lock_held:
            raise RuntimeError("Execution requires the shared process lock")
        # A slow broker operation must not queue another order check each second.
        if self._execution_lock.locked() or getattr(self, "_feed_missing", False):
            return
        async with self._execution_lock:
            attempt_field = "_last_attempted_short_entry" if strategy == "short" else "_last_attempted_entry"
            if getattr(self, attempt_field, None) == entry_ts:
                return
            if self.active_trades or self._execution_uncertain:
                log.info("[Execution] status=blocked reason=order_unresolved")
                return
            now = time.time()
            if any(candles[-1]["ts"] != math.floor(now / period) * period - period
                   for candles, period in snapshots):
                log.info("[Execution] status=blocked reason=stale_candle_snapshot")
                return
            if self._check_circuit_breaker() or self._check_consecutive_loss_halt()[0]:
                return
            if timing_state is not None and timing_state.get("direction") == direction:
                order_sent_time = self._offset_corrected_time()
                self._log_signal_timing(
                    direction=direction,
                    candle_close_time=float(timing_state.get("candle_close_time", close_ts)),
                    signal_detected_time=float(timing_state.get("signal_detected_time", time.time())),
                    order_sent_time=order_sent_time,
                )
                self._pending_signal_timing = None
            remaining, budget, _ = self._trade_capacity()
            if remaining <= 0 or budget <= 0:
                log.info("[Execution] status=blocked reason=capacity")
                return
            payout = await self.bot.get_asset_payout(self._current_asset)
            if strategy == "rsi_hookback":
                payout_ok = payout is not None and MIN_PAYOUT_PERCENT / 100.0 <= float(payout) <= 1.0
                if not payout_ok:
                    log.info("[Execution] status=blocked reason=payout payout=%s minimum=%s%%", payout, MIN_PAYOUT_PERCENT)
                    return
                now = self._offset_corrected_time()
                if abs(now - entry_ts) > ENTRY_WINDOW_SECONDS:
                    log.info("[Execution] status=blocked reason=missed_window entry_ts=%s delta_seconds=%.1f window_seconds=%s",
                             entry_ts, abs(now - entry_ts), ENTRY_WINDOW_SECONDS)
                    return
                state_key = "RSI_HOOKBACK"
            else:
                if payout is None or not self.payout_minimum <= payout <= 1.0:
                    log.info("[Execution] status=blocked reason=payout payout=%s minimum=%s", payout, self.payout_minimum)
                    return
                if (getattr(self, "_feed_missing", False)
                        or feed_generation != getattr(self, "_feed_generation", 0)):
                    log.info("[Execution] status=blocked reason=quote_interrupted")
                    return
                now = time.time()
                if not 0 <= now - close_ts <= self.DECISION_INTERVAL:
                    log.info("[Execution] status=blocked reason=stale_confirmation entry_ts=%s", entry_ts)
                    return
                # In five seconds a new short bar may close while payout is loading.
                if strategy == "short" and any(
                        candles[-1]["ts"] != math.floor(now / period) * period - period
                        for candles, period in snapshots):
                    log.info("[Execution] status=blocked reason=stale_candle_snapshot")
                    return
                state_key = "SHORT_CANDLE_GATE" if strategy == "short" else "CLOSED_CANDLE_GATE"
            setattr(self, attempt_field, entry_ts)
            signal = Signal.BUY if direction == "BUY" else Signal.SELL
            base_stake, factor, final_stake = self._volatility_stake()
            previous_amount = settings.TRADE_AMOUNT
            settings.TRADE_AMOUNT = final_stake
            self.trade_amount = final_stake
            self.risk.settings.TRADE_AMOUNT = final_stake
            log.info("[Stake] base_stake=%s factor=%s final_stake=%s strategy=%s direction=%s", base_stake, factor, final_stake, strategy, direction)
            self._persist_order_state({"status": "submitting", "asset": self._current_asset,
                                       "direction": direction, "entry_ts": entry_ts,
                                       "entry_period_seconds": entry_period, "strategy": strategy,
                                       "amount": self.trade_amount, "duration": FTT_DURATION_S,
                                       "submitted_at": datetime.now().isoformat(timespec="microseconds")})
            execution_options = {}
            if strategy == "short":
                execution_options = {
                    "entry_deadline": close_ts + entry_period,
                    "entry_guard": lambda: (
                        not getattr(self, "_feed_missing", False)
                        and feed_generation == getattr(self, "_feed_generation", 0)),
                }
            success = await self.supervisor.safe_execute_trade(signal, 0.95, "ftt", **execution_options)
            self._execution_uncertain = getattr(self.supervisor, "execution_uncertain", False)
            if not success:
                if not self._execution_uncertain:
                    self._persist_order_state(None)
                log.warning("[Execution] status=%s direction=%s entry_ts=%s",
                            "uncertain" if self._execution_uncertain else "rejected", direction, entry_ts)
                return
            order = dict(self.bot.last_order or {})
            timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            order.update(timestamp=timestamp, signal=direction, asset=self._current_asset,
                         state_key=state_key, entry_ts=entry_ts,
                         entry_period_seconds=entry_period, strategy=strategy)
            self._persist_order_state({"status": "accepted", "order": order})
            self.risk.record_trade(TradeRecord(
                timestamp=timestamp, asset=self._current_asset, mode="ftt",
                signal=direction, confidence=0.95, amount=order.get("amount", self.trade_amount),
                state_key=order["state_key"]))
            self._write_decision_log_async(
                session="overlap", gate_name=state_key, direction=direction,
                passed_rules=["trade_submitted"], failed_rules=[], metrics={
                    "ema9": None, "ema21": None, "ema50": None, "rsi": None, "rsi_dir": "unknown",
                    "atr_1m": None, "confirmation_range": None, "range_ratio": None,
                    "nearest_resistance": None, "nearest_support": None, "tick_counts": {
                        period: max(0, sum(1 for bar in self.candle_monitor.closed_snapshots().get(period, []) if bar.get("tick_count")))
                        for period in (5, 10, 15, 20, 30, 60)
                    }
                }, stake=float(order.get("amount", final_stake)), factor=factor)
            self._append_trade_row(direction=direction, rsi_prev=getattr(self, "_last_rsi_signal", {}).get("rsi_prev") if isinstance(getattr(self, "_last_rsi_signal", None), dict) else None,
                                  rsi_now=getattr(self, "_last_rsi_signal", {}).get("rsi_now") if isinstance(getattr(self, "_last_rsi_signal", None), dict) else None,
                                  entry_price=float(order.get("price", 0.0) or 0.0), payout=float(payout) if payout is not None else None,
                                  stake=float(order.get("amount", final_stake)), result="PENDING", pnl=0.0)
            settings.TRADE_AMOUNT = previous_amount
            self._last_rsi_signal = {"direction": direction, "rsi_prev": getattr(self, "_last_rsi_signal", {}).get("rsi_prev") if isinstance(getattr(self, "_last_rsi_signal", None), dict) else None, "rsi_now": getattr(self, "_last_rsi_signal", {}).get("rsi_now") if isinstance(getattr(self, "_last_rsi_signal", None), dict) else None}
            self.active_trades.append(order)
            self._accepted_count += 1
            log.info("[Execution] %s", json.dumps(dict(order, status="accepted", direction=direction), default=str))
            log.info("[Settlement] status=pending source=platform required_order_id=%s", order.get("id"))

    def _persist_order_state(self, state):
        """Persist before clicking so restart never silently resubmits an order."""
        path = getattr(self, "_order_state_path", None)
        if path is None:  # Dependency-injected unit tests have no execution store.
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        with temp.open("w") as handle:
            json.dump(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    async def _settlement_loop(self):
        while True:
            try:
                for order in list(self.active_trades):
                    result = await self.bot.get_order_result(order)
                    if result is None:
                        continue
                    if result.get("result") not in {"WIN", "LOSS", "DRAW"}:
                        continue
                    pnl = float(result["pnl"])
                    if not math.isfinite(pnl):
                        continue
                    committed = self.risk.update_pnl_and_journal(order["timestamp"], result["result"], pnl)
                    if committed is False:
                        self.active_trades.remove(order)
                        self._persist_order_state(None)
                        log.info("[Settlement] status=already_recorded order_id=%s", order.get("id"))
                        continue
                    won = result["result"] == "WIN"
                    self._record_trade_outcome(won, pnl)
                    if (result["result"] != "DRAW"
                            and order.get("state_key") != "DEMO_EXECUTION_TEST"
                            and order["timestamp"].startswith(self._account_snapshot["date"])):
                        self.supervisor.record_outcome(won)
                    self.active_trades.remove(order)
                    self._persist_order_state(None)
                    log.info("[Settlement] %s", json.dumps(dict(result, order_id=order.get("id"))))
                await self._reconcile_flat_account()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("[Settlement] status=unconfirmed reason=%s", exc)
            await asyncio.sleep(5)

    async def _reconcile_flat_account(self):
        """Retry broker flatness after startup or settlement, retaining unknown P&L.

        An open order can prevent startup proof. Once it settles, refresh the
        session's historical-exposure proof instead of leaving entries halted.
        Only RiskManager decides which historical exposure the proof releases.
        """
        if (self.active_trades or self._execution_uncertain
                or self._execution_lock.locked()):
            return
        today = datetime.now().date().isoformat()
        if getattr(self, "_flat_reconciled_date", None) == today:
            return
        if time.monotonic() < getattr(self, "_next_flat_reconciliation_at", 0.0):
            return
        async with self._execution_lock:
            # Keep proof and exposure reconciliation separate from order entry.
            if self.active_trades or self._execution_uncertain:
                return
            self._next_flat_reconciliation_at = time.monotonic() + 30.0
            try:
                proof = await self.bot.get_flat_ftt_account_evidence()
                if proof is None:
                    log.info("[RiskReconciliation] status=waiting reason=broker_flatness_unconfirmed retry_seconds=30")
                    return
                self.risk.confirm_flat_ftt_account(proof)
                self._flat_reconciled_date = today
                self._log_daily_accounting("broker_reconciled")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("[RiskReconciliation] status=unconfirmed reason=%s retry_seconds=30", exc)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    async def _shutdown(self):
        self.running = False
        await self._cancel_native_refresh()
        if self._decision_tasks:
            await asyncio.gather(*list(self._decision_tasks), return_exceptions=True)
        for t in list(self._bg_tasks):
            t.cancel()
        await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        await self.bot.stop()
        snapshot = self._sync_daily_accounting()
        strategy = snapshot["strategy"]
        log.info("[Session] accepted_this_run=%s account_accepted_today=%s "
                 "account_settled_today=%s account_net_pnl=%s strategy_settled_today=%s "
                 "strategy_wins_today=%s",
                 self._accepted_count, snapshot["accepted_count"], snapshot["settled_count"],
                 snapshot["net_pnl"], strategy["settled_count"], strategy["wins"])



def make_argument_parser():
    """Build the CLI without starting a session or changing configuration."""
    parser = argparse.ArgumentParser(description="Unified closed-candle BUY/SELL demo bot")
    parser.add_argument("--entry-mode", choices=("classic", "short", "hybrid"), default="hybrid",
                        help="Entry strategy: original 30s/1m, 5s breakout with 10s/15s confirmation, or both (default)")
    parser.add_argument("--chart-view", choices=("entry", "trend"), default="entry",
                        help="Verified chart display: 1m EMA9/21 RSI14 or 30s EMA20/50")
    parser.add_argument("--observe-only", action="store_true", help="Collect actual quotes and evaluate without clicks")
    parser.add_argument("--max-candle-closes", type=int, default=0, help="Stop after this many real close events; 0 runs continuously")
    parser.add_argument("--max-runtime", type=float, default=0, help="Bound runtime after login in seconds")
    parser.add_argument('--force-trades', action='store_true',
                        help='Force trades by bypassing the confluence gate (testing only)')
    parser.add_argument('--training-mode', action='store_true',
                        help='Enable training mode so the risk engine uses training overrides')
    parser.add_argument('--disable-circuit-breaker', action='store_true',
                        help='Ignore local circuit-breaker limits for testing')
    parser.add_argument('--pattern-gate', action='store_true',
                        help='Enable the additive pattern-scoring overlay without replacing the existing gate logic')
    parser.add_argument('--pattern-threshold', type=int, default=5,
                        help='Minimum pattern score to allow a pattern-gate fallback trade')
    parser.add_argument('--min-confidence', type=float, default=MIN_CONFIDENCE,
                        help='Minimum final confidence required to execute a trade')
    parser.add_argument('--trade-amount', type=float, default=None,
                        help='Override TRADE_AMOUNT from config/.env for this run')
    parser.add_argument('--max-daily-trades', type=int, default=None,
                        help='Override MAX_DAILY_TRADES from config/.env')
    parser.add_argument('--max-daily-loss', type=float, default=None,
                        help='Override MAX_DAILY_LOSS_USD from config/.env')
    parser.add_argument('--confidence-threshold', type=float, default=None,
                        help='Override the risk confidence threshold for the risk manager')
    parser.add_argument('--payout-minimum', type=float, default=PAYOUT_MINIMUM,
                        help='Minimum payout percentage required to place a trade')

    return parser


if __name__ == "__main__":
    configure_logging()
    args = make_argument_parser().parse_args()
    if args.force_trades:
        args.training_mode = True
        args.disable_circuit_breaker = True
    if args.training_mode:
        settings.TRAINING_MODE = True
    if args.max_daily_trades is not None:
        settings.MAX_DAILY_TRADES = args.max_daily_trades
    if args.max_daily_loss is not None:
        settings.MAX_DAILY_LOSS_USD = args.max_daily_loss
    if args.confidence_threshold is not None:
        settings.CONFIDENCE_THRESHOLD = args.confidence_threshold
    if args.trade_amount is not None:
        settings.TRADE_AMOUNT = args.trade_amount
    enable_pattern_gate = bool(args.pattern_gate)

    try:
        with ProcessLock():
            asyncio.run(
                MultiAgentTradingSystem(
                    observe_only=args.observe_only,
                    chart_view=args.chart_view,
                    entry_mode=args.entry_mode,
                    max_candle_closes=args.max_candle_closes,
                    max_runtime=args.max_runtime,
                    force_trades=args.force_trades,
                    training_mode=args.training_mode,
                    disable_circuit_breaker=args.disable_circuit_breaker,
                    enable_pattern_gate=enable_pattern_gate,
                    pattern_score_threshold=args.pattern_threshold,
                    min_confidence=args.min_confidence,
                    trade_amount=args.trade_amount,
                    payout_minimum=args.payout_minimum,
                    max_trades_per_day=args.max_daily_trades,
                    daily_loss_limit=args.max_daily_loss,
                ).run(lock_acquired=True)
            )
    except KeyboardInterrupt:
        pass
