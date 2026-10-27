"""config/settings.py — v3"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env', override=True)

EMAIL    = os.getenv("OLYMPTRADE_EMAIL", "")
PASSWORD = os.getenv("OLYMPTRADE_PASSWORD", "")
ASSET    = os.getenv("ASSET", "EUR/USD")
ALLOW_MULTIPLE_SESSIONS = os.getenv("ALLOW_MULTIPLE_SESSIONS", "false").lower() in ("1", "true", "yes", "on")
TRADE_AMOUNT  = float(os.getenv("TRADE_AMOUNT", "200"))
TRADE_MODE    = os.getenv("TRADE_MODE", "ftt").lower()
FTT_DURATION  = int(os.getenv("FTT_DURATION", "60"))
ASSET_SCAN_INTERVAL = int(os.getenv("ASSET_SCAN_INTERVAL_SECONDS", "99999"))

# Confidence gate — raise to 0.65 once profitable
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
MODEL_TYPE           = os.getenv("MODEL_TYPE", "ensemble")
LOOKBACK_CANDLES     = int(os.getenv("LOOKBACK_CANDLES", "200"))
CANDLE_TIMEFRAME     = os.getenv("CANDLE_TIMEFRAME", "M1")

MAX_DAILY_TRADES   = int(os.getenv("MAX_DAILY_TRADES", "50"))
MAX_DAILY_LOSS_USD = float(os.getenv("MAX_DAILY_LOSS_USD", "2000"))
MAX_CONCURRENT_OPEN_TRADES = int(os.getenv("MAX_CONCURRENT_OPEN_TRADES", "25"))
TRAINING_MAX_DAILY_TRADES = int(os.getenv("TRAINING_MAX_DAILY_TRADES", "10000"))
TRAINING_MAX_DAILY_LOSS_USD = float(os.getenv("TRAINING_MAX_DAILY_LOSS_USD", "10000"))
TRAINING_MAX_CONCURRENT_OPEN_TRADES = int(os.getenv("TRAINING_MAX_CONCURRENT_OPEN_TRADES", "25"))
STOP_LOSS_PCT      = float(os.getenv("STOP_LOSS_PCT", "0.02"))
POLL_INTERVAL      = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
HEADLESS           = os.getenv("HEADLESS", "true").lower() in ("1","true","yes")
LOG_DIR            = os.getenv("LOG_DIR",   "logs")
LOG_DATE_FORMAT            = os.getenv("LOG_DATE_FORMAT", "%Y%m%d")
LOG_ARCHIVE_DIR            = os.getenv("LOG_ARCHIVE_DIR", os.path.join(LOG_DIR, "archive"))
JOURNAL_ARCHIVE_DIR        = os.getenv("JOURNAL_ARCHIVE_DIR", os.path.join(LOG_DIR, "archive"))
RESET_TODAY_LOGS_ON_STARTUP = os.getenv("RESET_TODAY_LOGS_ON_STARTUP", "false").lower() in ("1", "true", "yes")
RL_USE_IN_LIVE_TRADING     = os.getenv("RL_USE_IN_LIVE_TRADING", "false").lower() in ("1", "true", "yes")
MODEL_DIR                  = os.getenv("MODEL_DIR", "models")
TARGET_PROFIT_PER_TRADE_USD = float(os.getenv("TARGET_PROFIT_PER_TRADE_USD", "100.0"))
MAX_TRADE_LOSS_USD = float(os.getenv("MAX_TRADE_LOSS_USD", "500.0"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TRAINING_MODE = os.getenv("TRAINING_MODE", "false").lower() in ("1", "true", "yes")
LOGIN_WAIT_SECONDS = int(os.getenv("LOGIN_WAIT_SECONDS", "300"))

ALLOW_HEDGING = os.getenv("ALLOW_HEDGING", "true").lower() in ("1","true","yes")
CLOSE_OPPOSITE_ON_STRONG_SIGNAL = os.getenv("CLOSE_OPPOSITE_ON_STRONG_SIGNAL", "false").lower() in ("1","true","yes")

# Volatility-aware execution and decision logging
BASE_STAKE_PCT = float(os.getenv("BASE_STAKE_PCT", "0.01"))
STAKE_MIN = float(os.getenv("STAKE_MIN", "1.0"))
STAKE_STEP = float(os.getenv("STAKE_STEP", "1.0"))
ATR_DISTANCE_MULTIPLIER = float(os.getenv("ATR_DISTANCE_MULTIPLIER", "0.5"))
CANDLE_STORE_PATH = os.getenv("CANDLE_STORE_PATH", os.path.join("data", "candles.sqlite3"))
DECISION_LOG_PATH = os.getenv("DECISION_LOG_PATH", os.path.join("data", "decisions.csv"))
