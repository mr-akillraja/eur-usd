import asyncio
import json
import math
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.browser import OlymtradeBot, OrderUncertainError, _asset_display_name
from models.predictor import Signal
from config import settings


@pytest.mark.parametrize("asset", ["EUR/USD OTC", "EURUSD-OTC"])
def test_otc_asset_display_preserved(asset):
    assert _asset_display_name(asset) == "EUR/USD OTC"


@pytest.mark.parametrize("title, valid", [
    ("▲ 1.09556 EUR/USD OTC | Trading platform", True),
    ("▼ 1.09556 EUR/USD | Trading platform", False),
    ("▲ 847.1 BNB OTC | Trading platform", False),
    ("Loading...", False),
    ("▲ 0 EUR/USD OTC | Trading platform", False),
])
def test_quote_requires_exact_instrument(title, valid):
    b = OlymtradeBot()
    b.is_alive = AsyncMock(return_value=True)
    b._page = SimplePage(title)
    quote = asyncio.run(b.get_current_quote("EUR/USD OTC"))
    assert (quote is not None) == valid
    if quote:
        assert quote["source"] == "broker_browser_sample"
        assert quote["timestamp_kind"] == "observation_time"
        assert quote["price"] == 1.09556


class SimplePage:
    def __init__(self, title):
        self.title = AsyncMock(return_value=title)


def test_native_history_parser_preserves_resolution():
    b = OlymtradeBot()
    t = math.floor(time.time() / 30) * 30 - 30
    bar = {"t": t, "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15}
    b._receive_market_frame(json.dumps([{"e": 10, "d": [
        {"p": "EURUSD_OTC", "tf": 60, "candles": [dict(bar, t=t - t % 60)]},
        {"p": "EURUSD_OTC", "tf": 30, "candles": [bar]}]}]))
    assert len(b._native_candles[("EURUSDOTC", 30)][1]) == 1
    assert b._native_candles[("EURUSDOTC", 30)][1][0]["ts"] == t
    assert ("EURUSDOTC", 60) in b._native_candles
    assert ("EURUSD", 30) not in b._native_candles


def test_native_history_rejects_invalid_ohlc():
    b = OlymtradeBot()
    b._receive_market_frame(json.dumps([{"d": [{"p": "EURUSD_OTC", "tf": 30,
        "candles": [{"t": 30, "open": 1.1, "high": 1.0, "low": 1.2, "close": 1.15}]}]}]))
    assert b._native_candles[("EURUSDOTC", 30)][1] == []


def test_native_request_returns_only_closed_matching_broker_bars():
    b = OlymtradeBot()
    b.get_current_quote = AsyncMock(return_value={"price": 1.1})
    requests = []
    async def evaluate(script, packet):
        requests.append(packet)
        end = math.floor(time.time() / 30) * 30
        candles = [{"t": end + offset, "open": 1.1, "high": 1.2,
                    "low": 1.0, "close": 1.15} for offset in (-60, -30, 0)]
        b._receive_market_frame(json.dumps([{"uuid": packet["uuid"], "d": [
            {"p": "EURUSD_OTC", "tf": 30, "candles": candles}]}]))
        return True
    b._page = MagicMock()
    b._page.evaluate = evaluate
    frame = asyncio.run(b.get_native_candles("EUR/USD OTC", 30, 180))
    assert len(frame) == 2
    assert frame.attrs["source"] == "broker_native_ws"
    assert frame.attrs["timeframe_seconds"] == 30
    assert requests[0]["d"][0]["size"] == 30
    assert requests[0]["d"][0]["pair"] == "EURUSD_OTC"


@pytest.mark.parametrize("failure", ["demo", "quote", "amount", "duration", "payout"])
def test_order_preconditions_prevent_click(failure):
    b, button = execution_bot()
    if failure == "demo": b.is_demo_account.return_value = False
    if failure == "quote": b.get_current_quote.return_value = None
    if failure == "amount": b.set_amount.return_value = False
    if failure == "duration": b._set_ftt_duration.return_value = False
    if failure == "payout": b.get_asset_payout.return_value = None
    assert not asyncio.run(b.place_ftt_trade(Signal.BUY, 60, "EUR/USD OTC"))
    button.click.assert_not_awaited()


def execution_bot():
    b = OlymtradeBot()
    b.is_demo_account = AsyncMock(return_value=True)
    b.get_current_quote = AsyncMock(return_value={"price": 1.1})
    b.set_amount = AsyncMock(return_value=True)
    b._set_ftt_duration = AsyncMock(return_value=True)
    b.get_asset_payout = AsyncMock(return_value=0.85)
    button = MagicMock()
    button.is_enabled = AsyncMock(return_value=True)
    button.is_visible = AsyncMock(return_value=True)
    button.click = AsyncMock()
    b._page = MagicMock()
    b._page.locator.return_value = button
    b._order_rows = AsyncMock(return_value=[])
    return b, button


@pytest.mark.parametrize("signal, suffix", [(Signal.BUY, "up"), (Signal.SELL, "down")])
def test_order_needs_unique_platform_confirmation(signal, suffix):
    b, button = execution_bot()
    b._order_rows.side_effect = [[], [{"id": "confirmed-demo-order", "asset": "EUR/USD OTC", "amount": float(settings.TRADE_AMOUNT), "direction": signal.value, "duration": 60}]]
    assert asyncio.run(b.place_ftt_trade(signal, 60, "EUR/USD OTC"))
    assert b.last_order["id"] == "confirmed-demo-order"
    assert b.last_order["payout"] == 0.85
    assert b.last_order["duration"] == 60
    button.click.assert_awaited_once()
    assert any("deal-submit-button-" + suffix in call.args[0] for call in b._page.locator.call_args_list)


def test_click_without_acknowledgement_is_uncertain(monkeypatch):
    b, button = execution_bot()
    monkeypatch.setattr("bot.browser.asyncio.sleep", AsyncMock())
    with pytest.raises(OrderUncertainError):
        asyncio.run(b.place_ftt_trade(Signal.BUY, 60, "EUR/USD OTC"))
    button.click.assert_awaited_once()
    assert b.last_order is None


def test_result_requires_matching_id_and_broker_pnl():
    b = OlymtradeBot()
    b._order_rows = AsyncMock(return_value=[
        {"id": "other", "status": "WIN", "pnl": "85"},
        {"id": "ours", "status": "OPEN", "pnl": None},
    ])
    assert asyncio.run(b.get_order_result({"id": "ours"})) is None
    b._order_rows.return_value = [{"id": "ours", "status": "LOSS", "pnl": "-200"}]
    assert asyncio.run(b.get_order_result({"id": "ours"}))["pnl"] == -200
