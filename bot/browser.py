"""
bot/browser.py
Playwright-based browser controller for Olymptrade.

Handles:
  • Login with session persistence (avoids re-login every run)
  • Asset selection
  • Fixed Time Trade (FTT) — click UP/DOWN + set duration
  • Forex/CFD trade        — click BUY/SELL + set stop-loss
  • Screenshot on every action for audit trail
"""
import asyncio
import json
import uuid
import logging
import os
import re
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from playwright.async_api import async_playwright, Page, Browser, BrowserContext, TimeoutError as PWTimeout

from config import settings
from models.predictor import Signal

log = logging.getLogger("browser")


def _entry_guard_allows(entry_guard):
    """Treat invalidated or unreadable signal state as a known non-click."""
    if entry_guard is None:
        return True
    try:
        if entry_guard() is True:
            return True
    except Exception:
        pass
    log.info("[Order] rejected=signal_invalidated")
    return False


ROOT = Path(__file__).resolve().parents[1]
SESSION_FILE = os.path.join(ROOT, "data", "session.json")
SCREENSHOT_DIR = os.path.join(settings.LOG_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)


def _asset_to_url_slug(asset: Optional[str]) -> str:
    """Normalize an asset name to a simple URL slug used by the UI."""
    if not asset:
        return ""
    s = str(asset).strip()
    s = s.replace("/", "-")
    s = s.replace(" ", "-")
    return re.sub(r"[^0-9A-Za-z\-]", "", s).lower()


def _asset_display_name(asset: Optional[str]) -> str:
    """Return the user-facing asset name used by the Olymptrade asset search."""
    if not asset:
        return ""
    asset_str = str(asset).strip()
    has_otc = asset_str.upper().endswith("OTC")
    base = re.sub(r"[\s_-]*OTC$", "", asset_str, flags=re.IGNORECASE) if has_otc else asset_str
    cleaned = re.sub(r"[^A-Za-z0-9/]+", "", base).upper()

    if "/" in cleaned:
        display = cleaned.replace("/", "/")
    else:
        if len(cleaned) == 6:
            display = f"{cleaned[:3]}/{cleaned[3:6]}"
        else:
            display = cleaned

    if has_otc and not display.endswith("OTC"):
        display = f"{display} OTC"

    return display


def _normalized_asset(asset: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(asset).upper())


class OrderUncertainError(RuntimeError):
    """A click may have submitted an order; never retry it automatically."""


class OlymtradeBot:
    """
    Async Playwright controller.

    Use as an async context manager:

        async with OlymtradeBot() as bot:
            await bot.place_ftt_trade(Signal.BUY, 60)
    """

    def __init__(self):
        self._pw = None
        self._browser: Optional[Browser] = None
        self._ctx: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._logged_in = False
        self.last_order = None
        self._selected_asset = None
        self._order_lock = asyncio.Lock()
        self._native_candles = {}
        self._native_responses = {}
        self._ws_diag = {"counts": {}, "samples": {}, "has_tick_quotes": False, "source_kind": "unknown"}
        self._last_ws_quote = None
        self._last_ws_lag_ms = 0.0
        self._last_ws_lag_logged_at = 0.0
        self._last_feed_reset_at = 0.0
        self._eurusd_payload_logged = False
        self._clock_offset_ms = 0.0
        self._clock_offset_logged_at = 0.0
        self._last_native_candle_update = 0.0

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def start(self):
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=settings.HEADLESS,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )

        if os.path.exists(SESSION_FILE):
            self._ctx = await self._browser.new_context(
                storage_state=SESSION_FILE,
                viewport={"width": 1366, "height": 768},
            )
            log.info("Loaded saved browser session")
        else:
            self._ctx = await self._browser.new_context(viewport={"width": 1366, "height": 768})

        self._page = await self._ctx.new_page()
        self._page.on("websocket", lambda socket: socket.on("framereceived", self._receive_market_frame))
        await self._page.add_init_script("""(() => {
            const NativeSocket = window.WebSocket;
            window.__tgHistorySockets = [];
            window.WebSocket = class extends NativeSocket {
                constructor(...args) {
                    super(...args);
                    window.__tgHistorySockets.push(this);
                }
                send(data) {
                    try {
                        const packets = JSON.parse(data);
                        if (Array.isArray(packets) && packets.some(p => p.e === 10 && p.t === 2))
                            this.__tgHistory = true;
                    } catch (_) {}
                    return super.send(data);
                }
            };
        })();""")
        await self._page.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")

    async def stop(self):
        if self._ctx:
            try:
                await self._ctx.storage_state(path=SESSION_FILE)
                log.info("Browser session saved")
            except Exception as e:
                log.warning(f"Could not save browser session: {e}")
        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                log.warning(f"Browser close failed during shutdown: {e}")
        if self._pw:
            try:
                await self._pw.stop()
            except Exception as e:
                log.warning(f"Playwright stop failed during shutdown: {e}")

    async def is_alive(self) -> bool:
        """Health check used by the supervisor: True if the page is responsive."""
        try:
            if not self._page or self._page.is_closed():
                return False
            state = await self._page.evaluate("() => document.readyState")
            return state in ("interactive", "complete")
        except Exception:
            return False

    async def login(self) -> bool:
        """Log in with email/password, or fallback to manual Google Sign-In."""
        page = self._page

        if os.path.exists(SESSION_FILE):
            try:
                await page.goto("https://olymptrade.com/platform", wait_until="domcontentloaded", timeout=30_000)
                await page.locator('[data-test="account-balance-title"]').wait_for(state="visible", timeout=15_000)
                self._logged_in = True
                log.info("Verified saved platform session")
                return True
            except Exception:
                log.info("Saved session requires login")
        try:
            try:
                await page.goto("https://olymptrade.com", wait_until="load", timeout=30_000)
            except PWTimeout:
                log.warning("Page load timed out; falling back to commit.")
                await page.goto("https://olymptrade.com", wait_until="commit", timeout=30_000)
            await asyncio.sleep(2)
        except Exception as e:
            log.error(f"Navigation failed: {e}")
            return False

        try:
            if await page.locator(".platform-chart, .trade-panel, .chart-wrapper").count() > 0:
                log.info("Already logged in (session reused)")
                self._logged_in = True
                return True
        except Exception:
            pass

        log.info("Logging in to Olymptrade...")

        has_credentials = bool(settings.EMAIL and settings.PASSWORD and "@" in settings.EMAIL)

        if has_credentials:
            opened = False
            for selector in [
                'button[data-test="action-sign-in"]',
                'button:has-text("Sign in")',
                'text=Sign in',
                'button:has-text("Log in")',
            ]:
                try:
                    await page.click(selector, timeout=2000, force=True)
                    opened = True
                    await asyncio.sleep(0.6)
                    break
                except Exception:
                    continue

            email_selectors = ['input[type="email"]', 'input[name="email"]', 'input[placeholder*="email"]', 'input[type="text"]']
            found_email = None
            for sel in email_selectors:
                try:
                    await page.wait_for_selector(sel, state="visible", timeout=4000)
                    found_email = sel
                    break
                except Exception:
                    continue

            if not found_email:
                log.warning("Auto-login: email field not found after opening modal")
            else:
                try:
                    await page.fill(found_email, settings.EMAIL, timeout=4000)
                    pass_selectors = ['input[type="password"]', 'input[name="password"]']
                    found_pass = None
                    for ps in pass_selectors:
                        try:
                            await page.wait_for_selector(ps, state="visible", timeout=2000)
                            found_pass = ps
                            break
                        except Exception:
                            continue
                    if found_pass:
                        await page.fill(found_pass, settings.PASSWORD, timeout=3000)
                    await self._screenshot("before_login")
                    for btn in ['button[type="submit"]', '.login-btn', '.sign-in-btn', 'button:has-text("Sign in")', 'button:has-text("Log in")']:
                        try:
                            await page.click(btn, timeout=3000)
                            break
                        except Exception:
                            continue
                    await asyncio.sleep(3)
                    if await page.locator(".platform-chart, .trade-panel, .chart-wrapper").count() > 0:
                        log.info("Automatic login successful!")
                        self._logged_in = True
                        try:
                            await self._ctx.storage_state(path=SESSION_FILE)
                        except Exception:
                            pass
                        return True
                except Exception as e:
                    log.warning(f"Auto-login failed: {e}")

        log.info("MANUAL SIGN-IN if required — waiting up to 180s")
        for i in range(180):
            try:
                url = page.url.lower()
                logged = (
                    await page.locator(".platform-chart, .trade-panel, .chart-wrapper, .sidebar, .user-avatar, [data-test='deposit-button']").count() > 0
                ) or any(k in url for k in ("platform", "cabinet", "dashboard"))
                if logged:
                    log.info(f"Login detected! URL={page.url}")
                    self._logged_in = True
                    try:
                        await self._ctx.storage_state(path=SESSION_FILE)
                    except Exception:
                        pass
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
            if i % 15 == 0 and i > 0:
                log.info(f"Waiting for manual sign-in... ({180-i}s left) | {page.url}")

        log.error("Login timed out (180s)")
        await self._screenshot("login_timeout")
        return False

    async def select_asset(self, asset: str) -> bool:
        """Select the exact broker instrument and verify its active chart title."""
        if await self.get_current_quote(asset):
            return True
        code = _normalized_asset(asset)
        code = code[:-3] + "_OTC" if code.endswith("OTC") else code
        try:
            existing = self._page.locator(f'[data-test="asset-select-button-{code}/ftt"]')
            if await existing.count():
                await existing.click(timeout=3000)
            else:
                await self._page.locator('[data-test="assets-tabs-add-button"]').click(timeout=3000)
                search = self._page.locator('[data-test="asset-search-field"] input')
                await search.fill(_asset_display_name(asset).replace(" OTC", ""), timeout=3000)
                await self._page.locator(f'[data-test="asset-item-title-{code}"]').click(timeout=5000)
            for _ in range(50):
                if await self.get_current_quote(asset):
                    await self._page.keyboard.press("Escape")
                    log.info("[Browser] asset_verified=%s", asset)
                    return True
                await asyncio.sleep(0.1)
        except Exception as exc:
            log.error("[Browser] asset_selection_failed=%s reason=%s", asset, exc)
        return False

    async def _verify_chart_loaded(self, asset: str) -> bool:
        return await self.get_current_quote(asset) is not None

    async def set_amount(self, amount: float) -> bool:
        try:
            field = self._page.locator('[data-test="amount-input"] input')
            await field.fill(format(float(amount), "g"), timeout=2000)
            await field.press("Tab")
            actual = float((await field.input_value()).replace(",", "").strip())
            return math.isclose(actual, float(amount), rel_tol=0, abs_tol=0.00001)
        except Exception as exc:
            log.warning("[Order] amount_not_verified reason=%s", exc)
            return False

    async def _set_ftt_duration(self, seconds: int) -> bool:
        try:
            field = self._page.locator('[data-test="trade-time-duration-input"] input')
            actual = (await field.input_value()).strip().lower()
            if actual in self._ftt_duration_labels(seconds):
                return True
            label = f"{seconds // 60}m" if seconds % 60 == 0 else f"{seconds}s"
            await field.fill(label, timeout=2000)
            await field.press("Tab")
            return (await field.input_value()).strip().lower() in self._ftt_duration_labels(seconds)
        except Exception as exc:
            log.warning("[Order] duration_not_verified reason=%s", exc)
            return False

    def _ftt_duration_labels(self, seconds: int) -> list[str]:
        labels = [f"{seconds}s", f"{seconds} sec"]
        if seconds % 60 == 0:
            labels += [f"{seconds // 60}m", f"{seconds // 60} min"]
        return labels

    @staticmethod
    def _money(text):
        normalized = text.replace(",", "").replace("−", "-")
        match = re.search(r"([+-]?)[^0-9]*([0-9]+(?:\.[0-9]+)?)([KM]?)", normalized)
        if not match:
            return None
        value = float(match.group(2)) * ({"K": 1000, "M": 1000000}.get(match.group(3), 1))
        return -value if match.group(1) == "-" else value

    async def _order_rows(self, include_history=False, wanted_id=None):
        """Read IDs and results from the broker's expanded trade cards."""
        panel = self._page.locator('[data-test="cor-w-panel-trades"]')
        if not await panel.is_visible():
            await self._page.locator('[data-test="sidebar-btn-trading-bar"]').click(timeout=2000)
        await panel.wait_for(state="visible", timeout=3000)
        # Active and settled cards share the misleading "deal-history-card-ftt"
        # data-test on the broker UI. Membership in active-deal-list identifies
        # open positions; the card's name cannot identify its lifecycle state.
        for _ in range(30):
            if await panel.locator('[data-test="active-deal-list"], [data-test="deal-list"]').count():
                break
            if "No active" in " ".join((await panel.inner_text()).split()):
                break
            await asyncio.sleep(0.1)
        scope = panel if include_history else panel.locator('[data-test="active-deal-list"]')
        headers = scope.locator('[data-test="deal-header"]')
        rows = []
        for index in range(await headers.count()):
            header = headers.nth(index)
            card = header.locator('xpath=ancestor::*[contains(@data-test, "card")][1]')
            if not await card.count():
                continue
            text = await card.inner_text()
            match = re.search(r"Trade ID\s+(\d+)", text)
            if not match:
                await header.click(timeout=1500)
                try:
                    await card.get_by_text("Trade ID", exact=True).wait_for(timeout=1500)
                except Exception:
                    continue
                text = await card.inner_text()
                match = re.search(r"Trade ID\s+(\d+)", text)
            if not match:
                continue
            trade_id = match.group(1)
            if wanted_id is not None and trade_id != str(wanted_id):
                continue
            asset_node = card.locator('[data-test^="asset-name-"]').first
            asset = (await asset_node.inner_text()).strip()
            amount_match = re.search(r"\bAmount\s+([^\n]+)", text)
            amount = self._money(amount_match.group(1)) if amount_match else None
            icon = await card.locator('[data-test="direction-icon"]').get_attribute("data-icon") or ""
            side = "BUY" if "up" in icon else "SELL" if "down" in icon else None
            duration_match = re.search(r"\bDuration\s+(\d+)\s*(min|sec|s|m)\b", text)
            duration = None
            if duration_match:
                duration = int(duration_match.group(1)) * (60 if duration_match.group(2) in {"min", "m"} else 1)
            closed = bool(re.search(r"\bClosed\b", text))
            pnl_node = card.locator('[data-test="pnl"]').first
            pnl = self._money(await pnl_node.inner_text()) if closed and await pnl_node.count() else None
            status = "OPEN"
            if closed and pnl is not None:
                status = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "DRAW"
            rows.append({"id": trade_id, "asset": asset, "amount": amount,
                         "direction": side, "duration": duration,
                         "status": status, "pnl": pnl})
            if wanted_id is not None:
                break
        return rows

    async def get_flat_ftt_account_evidence(self):
        """Verify the broker explicitly reports no active FTT trades/orders."""
        if not await self.is_demo_account():
            return None
        panel = self._page.locator('[data-test="cor-w-panel-trades"]')
        if not await panel.is_visible():
            await self._page.locator('[data-test="sidebar-btn-trading-bar"]').click(timeout=2000)
        await panel.wait_for(state="visible", timeout=3000)
        for _ in range(30):
            text = " ".join((await panel.inner_text()).split())
            active = panel.locator('[data-test="active-deal-list"] [data-test="deal-header"]')
            if await active.count():
                return None
            if ("No active Fixed Time trades or orders on this account" in text
                    and await self.is_demo_account()):
                return {"source": "platform_active_order_list", "demo": True,
                        "no_active_ftt": True, "observed_at": time.time()}
            await asyncio.sleep(0.1)
        return None

    async def place_ftt_trade(self, signal: Signal, duration_seconds: int, asset: str = None,
                              *, entry_deadline: Optional[float] = None,
                              entry_guard: Optional[Callable[[], bool]] = None) -> bool:
        """Place a verified demo order while an optional entry signal is fresh."""
        if entry_deadline is not None:
            try:
                entry_deadline = float(entry_deadline)
            except (TypeError, ValueError, OverflowError):
                log.info("[Order] rejected=invalid_entry_deadline")
                return False
            if not math.isfinite(entry_deadline):
                log.info("[Order] rejected=invalid_entry_deadline")
                return False
            if time.time() >= entry_deadline:
                log.info("[Order] rejected=signal_expired")
                return False
        if not _entry_guard_allows(entry_guard):
            return False
        asset = asset or settings.ASSET
        async with self._order_lock:
            self.last_order = None
            if entry_deadline is not None and time.time() >= entry_deadline:
                log.info("[Order] rejected=signal_expired")
                return False
            if not _entry_guard_allows(entry_guard):
                return False
            if signal not in (Signal.BUY, Signal.SELL):
                return False
            if not await self.is_demo_account() or not await self.get_current_quote(asset):
                log.error("[Order] rejected=demo_or_asset_not_verified")
                return False
            if not await self._page.locator('[data-test="trade-form-ftt"]').is_visible():
                log.error("[Order] rejected=ftt_mode_not_verified")
                return False
            if not await self.set_amount(settings.TRADE_AMOUNT):
                return False
            if not await self._set_ftt_duration(duration_seconds):
                return False
            payout = await self.get_asset_payout(asset)
            if payout is None:
                return False
            before = {row["id"] for row in await self._order_rows()}
            button = self._page.locator('[data-test="deal-submit-button-' +
                ("up" if signal == Signal.BUY else "down") + '"]')
            if not await button.is_enabled():
                return False
            # Revalidate the account and active instrument immediately before clicking.
            quote = await self.get_current_quote(asset)
            if not await self.is_demo_account() or quote is None:
                return False
            if not _entry_guard_allows(entry_guard):
                return False
            click_timeout_ms = 3000
            if entry_deadline is not None:
                remaining = entry_deadline - time.time()
                if remaining <= 0:
                    log.info("[Order] rejected=signal_expired")
                    return False
                # A positive fractional timeout must not be rounded to zero:
                # Playwright treats zero as disabling its timeout entirely.
                click_timeout_ms = min(click_timeout_ms, remaining * 1000)
            try:
                await button.click(timeout=click_timeout_ms)
                for _ in range(50):
                    rows = [row for row in await self._order_rows() if
                            row["id"] not in before and
                            _normalized_asset(row.get("asset", "")) == _normalized_asset(asset) and
                            row.get("direction") == signal.value and
                            row.get("amount") == float(settings.TRADE_AMOUNT) and
                            row.get("duration") == duration_seconds]
                    if len(rows) == 1:
                        row = rows[0]
                        self.last_order = {"id": row["id"], "asset": asset,
                            "amount": float(settings.TRADE_AMOUNT), "duration": duration_seconds,
                            "payout": payout, "accepted_at": time.time(), "demo": True,
                            "confirmation_source": "platform_order_id"}
                        log.info("[Order] accepted_id=%s side=%s", row["id"], signal.value)
                        return True
                    await asyncio.sleep(0.1)
            except Exception as exc:
                raise OrderUncertainError("Click outcome requires broker reconciliation") from exc
            raise OrderUncertainError("No unique broker order acknowledgement after click")

    async def get_order_result(self, order):
        for row in await self._order_rows(include_history=True, wanted_id=order.get("id")):
            if row["id"] != order.get("id"):
                continue
            state = (row.get("status") or "").upper()
            if state not in {"WIN", "LOSS", "DRAW"} or row.get("pnl") is None:
                return None
            try:
                pnl = float(row["pnl"])
            except (ValueError, TypeError):
                return None
            if math.isfinite(pnl):
                return {"result": state, "pnl": pnl, "source": "platform"}
        return None

    async def place_forex_trade(self, signal: Signal, asset: str = None) -> bool:
        page = self._page
        try:
            await self._switch_trade_mode('forex')
            await self.set_amount(settings.TRADE_AMOUNT)
            await self._screenshot(f'forex_before_{signal.value.lower()}')
            if signal == Signal.BUY:
                await page.locator('button[data-test="buy-btn"], button:has-text("Buy")').first.click(timeout=5000)
                log.info(f"[Trade] Forex BUY placed | Asset: {asset}")
            else:
                await page.locator('button[data-test="sell-btn"], button:has-text("Sell")').first.click(timeout=5000)
                log.info(f"[Trade] Forex SELL placed | Asset: {asset}")
            await asyncio.sleep(0.8)
            await self._screenshot(f'forex_after_{signal.value.lower()}')
            try:
                if await page.locator('button:has-text("Confirm"), .confirm-btn').count() > 0:
                    await page.locator('button:has-text("Confirm"), .confirm-btn').first.click()
            except Exception:
                pass
            return True
        except Exception as e:
            log.error(f"[Trade] Forex trade failed: {e}")
            await self._screenshot('forex_error')
            return False

    async def _set_forex_stop_loss(self):
        page = self._page
        try:
            sl = page.locator('input[data-test="stop-loss"], .stop-loss-input')
            if await sl.count() > 0:
                await sl.first.fill(str(settings.STOP_LOSS_PCT))
        except Exception:
            pass

    async def _switch_trade_mode(self, mode: str):
        page = self._page
        try:
            if mode == 'ftt':
                sel = 'button:has-text("Fixed Time"), [data-test="ftt-tab"]'
            else:
                sel = 'button:has-text("Forex"), [data-test="forex-tab"]'
            try:
                await page.click(sel, timeout=1500)
                await asyncio.sleep(0.6)
            except Exception:
                pass
        except Exception:
            pass

    async def _screenshot(self, label: str):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"{ts}_{label}.png")
            if self._page:
                await self._page.screenshot(path=path, timeout=5000)
        except Exception:
            pass

    async def close_active_position(self, signal: Signal, entry_price: float) -> bool:
        page = self._page
        try:
            rows = ['.position-row', '.open-position', '.deal-row']
            for r in rows:
                try:
                    nodes = await page.locator(r).all()
                    for n in nodes:
                        try:
                            txt = await n.inner_text()
                            if (signal.value in txt or signal.name in txt) and str(entry_price)[:6] in txt:
                                try:
                                    btn = n.locator('button:has-text("Close"), button.close-btn')
                                    if await btn.count() > 0:
                                        await btn.first.click()
                                        await asyncio.sleep(0.5)
                                        await self._screenshot('early_close')
                                        return True
                                except Exception:
                                    continue
                        except Exception:
                            continue
                except Exception:
                    continue

            for sel in ['button:has-text("Close"), .close-position, .close-deal, button.close']:
                try:
                    await page.click(sel, timeout=1200)
                    await asyncio.sleep(0.4)
                    await self._screenshot('early_close_fallback')
                    return True
                except Exception:
                    continue
        except Exception:
            pass
        return False

    @staticmethod
    def _looks_like_epoch(value):
        try:
            num = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(num):
            return False
        now = time.time()
        if 1.5e9 <= num <= 2.5e9 and abs(now - num) <= 300:
            return True
        if 1.5e12 <= num <= 2.5e12 and abs(now * 1000.0 - num) <= 300000:
            return True
        return False

    @staticmethod
    def _mask_ws_payload(payload):
        if isinstance(payload, dict):
            masked = {}
            for key, value in payload.items():
                key_lower = str(key).lower()
                if key_lower in {"accountid", "account_id", "userid", "user_id", "user", "email", "emailflag", "balance", "account_balance", "demo_balance"}:
                    masked[key] = "***"
                elif isinstance(value, (dict, list)):
                    masked[key] = OlymtradeBot._mask_ws_payload(value)
                else:
                    masked[key] = value
            return masked
        if isinstance(payload, list):
            return [OlymtradeBot._mask_ws_payload(item) for item in payload]
        return payload

    @staticmethod
    def _has_numeric_price_value(payload):
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_lower = str(key).lower()
                if key_lower in {"price", "last", "v", "value", "close", "bid", "ask", "p"}:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(numeric) and numeric > 0:
                        return True
                if isinstance(value, (dict, list)) and OlymtradeBot._has_numeric_price_value(value):
                    return True
            return False
        if isinstance(payload, list):
            return any(OlymtradeBot._has_numeric_price_value(item) for item in payload)
        return False

    @staticmethod
    def _extract_ws_tick(packet):
        if not isinstance(packet, dict):
            return None
        kind = str(packet.get("e") or packet.get("type") or packet.get("event") or packet.get("msg") or "unknown")
        if kind == "11":
            if isinstance(packet.get("d"), list):
                if not any(OlymtradeBot._has_numeric_price_value(item) for item in packet["d"]):
                    return None
            elif not OlymtradeBot._has_numeric_price_value(packet):
                return None
        pair = packet.get("p") or packet.get("pair") or packet.get("symbol")
        if not isinstance(pair, str):
            nested = packet.get("d")
            if isinstance(nested, list) and nested and isinstance(nested[0], dict):
                pair = nested[0].get("p") or nested[0].get("pair") or nested[0].get("symbol")
        if not isinstance(pair, str):
            return None
        pair_norm = _normalized_asset(pair)
        if pair_norm != _normalized_asset("EURUSD"):
            return None
        value = None
        for key in ("price", "last", "v", "value", "close", "bid", "ask"):
            if key in packet:
                try:
                    candidate = float(packet[key])
                except (TypeError, ValueError):
                    candidate = None
                if candidate is not None and math.isfinite(candidate) and candidate > 0:
                    value = candidate
                    break
        if value is None:
            nested = packet.get("d")
            if isinstance(nested, dict):
                for key in ("price", "last", "v", "value", "close", "bid", "ask"):
                    if key in nested:
                        try:
                            candidate = float(nested[key])
                        except (TypeError, ValueError):
                            candidate = None
                        if candidate is not None and math.isfinite(candidate) and candidate > 0:
                            value = candidate
                            break
            elif isinstance(nested, list) and nested and isinstance(nested[0], dict):
                for key in ("price", "last", "v", "value", "close", "bid", "ask"):
                    if key in nested[0]:
                        try:
                            candidate = float(nested[0][key])
                        except (TypeError, ValueError):
                            candidate = None
                        if candidate is not None and math.isfinite(candidate) and candidate > 0:
                            value = candidate
                            break
        if value is None:
            return None
        ts_value = None
        for field in ("st", "server_timestamp", "server_time", "broker_timestamp", "ts", "time", "timestamp"):
            if field in packet and OlymtradeBot._looks_like_epoch(packet[field]):
                ts_value = float(packet[field])
                break
        if ts_value is None and isinstance(packet.get("d"), dict):
            d_payload = packet["d"]
            for field in ("st", "server_timestamp", "server_time", "broker_timestamp", "ts", "time", "timestamp"):
                if field in d_payload and OlymtradeBot._looks_like_epoch(d_payload[field]):
                    ts_value = float(d_payload[field])
                    break
        if ts_value is None and isinstance(packet.get("d"), list) and packet["d"] and isinstance(packet["d"][0], dict):
            d_payload = packet["d"][0]
            for field in ("t", "st", "server_timestamp", "server_time", "broker_timestamp", "ts", "time", "timestamp"):
                if field in d_payload and OlymtradeBot._looks_like_epoch(d_payload[field]):
                    ts_value = float(d_payload[field])
                    break
        if ts_value is None:
            ts_value = time.time()
        return {"pair": pair, "price": float(value), "ts": float(ts_value), "source": "ws_ticks"}

    def _record_ws_diag(self, packet):
        if not isinstance(packet, dict):
            return
        kind = str(packet.get("e") or packet.get("type") or packet.get("event") or packet.get("msg") or "unknown")
        has_priced_tick = OlymtradeBot._has_numeric_price_value(packet)
        if kind == "11" and isinstance(packet.get("d"), list):
            has_priced_tick = any(OlymtradeBot._has_numeric_price_value(item) for item in packet["d"])
        if not has_priced_tick:
            pair_values = []
            for container in (packet, packet.get("d") or []):
                if isinstance(container, dict):
                    pair_values.extend([container.get("p"), container.get("pair"), container.get("symbol")])
                elif isinstance(container, list):
                    for item in container:
                        if isinstance(item, dict):
                            pair_values.extend([item.get("p"), item.get("pair"), item.get("symbol")])
            if not any(isinstance(value, str) and _normalized_asset(value) == _normalized_asset("EURUSD") for value in pair_values):
                return
        pair_values = []
        for container in (packet, packet.get("d") or []):
            if isinstance(container, dict):
                pair_values.extend([container.get("p"), container.get("pair"), container.get("symbol")])
            elif isinstance(container, list):
                for item in container:
                    if isinstance(item, dict):
                        pair_values.extend([item.get("p"), item.get("pair"), item.get("symbol")])
        if not any(isinstance(value, str) and _normalized_asset(value) == _normalized_asset("EURUSD") for value in pair_values):
            return
        self._ws_diag["counts"][kind] = self._ws_diag["counts"].get(kind, 0) + 1
        if kind not in self._ws_diag["samples"]:
            ts_field = next((key for key in ("ts", "time", "timestamp", "server_timestamp", "server_time", "broker_timestamp", "st") if key in packet), None)
            sample = self._mask_ws_payload(dict(packet))
            if ts_field is not None:
                sample["timestamp_field"] = ts_field
            if "d" in packet and isinstance(packet["d"], list) and packet["d"]:
                sample["d_preview"] = self._mask_ws_payload(packet["d"][0])
            self._ws_diag["samples"][kind] = sample
        if has_priced_tick:
            self._ws_diag["has_tick_quotes"] = True
            self._ws_diag["source_kind"] = "per_tick_quotes"
            self._ws_diag["priced_types"] = self._ws_diag.get("priced_types", set())
            self._ws_diag["priced_types"].add(kind)

    def _receive_market_frame(self, data):
        """Capture only broker candle replies, never authentication/account frames."""
        try:
            packets = json.loads(data)
            if not isinstance(packets, list):
                return
            for packet in packets:
                recv_time = time.time()
                if isinstance(packet, dict):
                    st = packet.get("st")
                    kind = str(packet.get("e") or packet.get("type") or packet.get("event") or "unknown")
                    if st is not None and kind in {"11", "14"}:
                        try:
                            server_ts = float(st)
                            if math.isfinite(server_ts):
                                offset_ms = (server_ts - time.time() * 1000.0) if server_ts > 1e12 else (server_ts - time.time()) * 1000.0
                                self._clock_offset_ms = float(offset_ms)
                                if time.time() - self._clock_offset_logged_at >= 300:
                                    log.info("[Clock] clock_offset_ms=%.0f message_type=%s", self._clock_offset_ms, kind)
                                    self._clock_offset_logged_at = time.time()
                                lag_seconds = recv_time - (server_ts / 1000.0 if server_ts > 1e12 else server_ts)
                                self._last_ws_lag_ms = lag_seconds * 1000.0
                                if time.time() - self._last_ws_lag_logged_at >= 1.0:
                                    if lag_seconds > 1.0:
                                        log.warning("[WSFeed] recv_time=%.6f st=%.6f lag_seconds=%.3f message_type=%s", recv_time, server_ts, lag_seconds, kind)
                                    else:
                                        log.info("[WSFeed] recv_time=%.6f st=%.6f lag_seconds=%.3f message_type=%s", recv_time, server_ts, lag_seconds, kind)
                                    self._last_ws_lag_logged_at = time.time()
                        except (TypeError, ValueError):
                            pass
                    if kind == "10" and isinstance(packet.get("d"), list):
                        self._last_native_candle_update = time.time()
                self._record_ws_diag(packet)
                quote = self._extract_ws_tick(packet)
                if quote is not None:
                    self._last_ws_quote = quote
                if not isinstance(packet, dict) or not isinstance(packet.get("d"), list):
                    continue
                for item in packet["d"]:
                    if not isinstance(item, dict) or not isinstance(item.get("candles"), list):
                        continue
                    period = item.get("tf")
                    pair = item.get("p")
                    if period not in (30, 60) or not isinstance(pair, str):
                        continue
                    rows = []
                    for candle in item["candles"]:
                        row = {key: float(candle[key]) for key in ("open", "high", "low", "close")}
                        row["ts"] = float(candle["t"])
                        if not all(math.isfinite(value) for value in row.values()):
                            continue
                        if row["ts"] % period != 0 or row["low"] <= 0:
                            continue
                        if row["high"] < max(row["open"], row["close"]) or row["low"] > min(row["open"], row["close"]):
                            continue
                        rows.append(row)
                    key = (_normalized_asset(pair), period)
                    self._native_candles[key] = (time.time(), rows)
                    if packet.get("uuid"):
                        self._native_responses[packet["uuid"]] = key
        except (ValueError, TypeError, KeyError):
            return

    async def diagnose_ws_feed(self, duration_seconds: int = 60):
        """Capture WS message counts for a short window and log whether the feed is per-tick or candle-only."""
        self._ws_diag = {"counts": {}, "samples": {}, "has_tick_quotes": False, "source_kind": "unknown", "priced_types": set()}
        await asyncio.sleep(max(0, duration_seconds))
        counts = dict(sorted(self._ws_diag["counts"].items()))
        for message_type, count in counts.items():
            if message_type not in {"10", "11", "14"} and message_type not in {"candle", "quote", "payout"}:
                continue
            sample = self._ws_diag["samples"].get(message_type, {})
            ts_field = sample.get("timestamp_field") or next((key for key in ("t", "ts", "time", "timestamp", "st") if key in sample), None)
            ts_value = sample.get(ts_field) if ts_field else None
            log.info("[WSFeed] type=%s count=%s timestamp_field=%s sample_ts=%s sample=%s", message_type, count, ts_field, ts_value, json.dumps(self._mask_ws_payload(sample), sort_keys=True)[:200])
        if self._ws_diag.get("priced_types"):
            log.info("[WSFeed] price_types=%s", sorted(self._ws_diag["priced_types"]))
        if self._ws_diag["has_tick_quotes"]:
            log.info("[WSFeed] status=per_tick_quotes_detected source_kind=%s", self._ws_diag["source_kind"])
        else:
            log.info("[WSFeed] status=candle_only source_kind=%s", self._ws_diag["source_kind"] or "candle_snapshots")
        for handler in log.handlers:
            try:
                handler.flush()
            except Exception:
                pass
        return dict(self._ws_diag)

    async def get_native_candles(self, asset: str, period: int, count: int = 180):
        """Request native OHLC on the already-authenticated chart connection.

        Event 10 and pair/size/to/limit are observed chart history messages.
        The explicit broker tf must match the requested interval. M1 is never
        converted to 30s; only bars whose end time has passed are returned.
        """
        if period not in (30, 60) or await self.get_current_quote(asset) is None:
            return None
        import pandas as pd
        code = _normalized_asset(asset)
        pair = code[:-3] + "_OTC" if code.endswith("OTC") else code
        request_id = "tg-history-" + uuid.uuid4().hex
        requested_at = time.time()
        packet = {"t": 2, "e": 10, "uuid": request_id,
                  "d": [{"pair": pair, "size": period,
                         "to": int(requested_at), "limit": int(count) + 2}]}
        sent = await self._page.evaluate("""packet => {
            const socket = (window.__tgHistorySockets || []).find(s => s.__tgHistory && s.readyState === 1);
            if (!socket) return false;
            socket.send(JSON.stringify([packet]));
            return true;
        }""", packet)
        if not sent:
            log.warning("[NativeHistory] transport=unavailable timeframe=%s", period)
            return None
        key = (code, period)
        for _ in range(60):
            cached = self._native_candles.get(key)
            if cached and cached[0] >= requested_at:
                now = time.time()
                rows = [row for row in cached[1] if row["ts"] + period <= now]
                if rows:
                    frame = pd.DataFrame(rows).sort_values("ts").drop_duplicates("ts").tail(count)
                    frame.index = pd.to_datetime(frame.pop("ts"), unit="s", utc=True)
                    frame.index.name = "timestamp"
                    frame.attrs.update(source="broker_native_ws", asset=asset,
                                       timeframe_seconds=period, closed_only=True)
                    return frame
            await asyncio.sleep(0.05)
        log.warning("[NativeHistory] response=unavailable timeframe=%s", period)
        return None

    def feed_is_healthy(self) -> bool:
        """Read-only feed health check for external callers. No strategy wiring is performed here."""
        now = time.time()
        last_quote = self._last_ws_quote if isinstance(self._last_ws_quote, dict) else None
        if last_quote is None:
            return False
        try:
            last_tick_age = now - float(last_quote.get("ts", now))
        except (TypeError, ValueError):
            return False
        lag_ms = float(self._last_ws_lag_ms or 0.0)
        return bool(last_tick_age < 1.0 and lag_ms < 1000.0 and self._last_feed_reset_at <= last_quote.get("ts", now))

    async def get_current_quote(self, asset: str):
        if self._last_ws_quote is not None:
            pair_norm = _normalized_asset(self._last_ws_quote.get("pair", ""))
            if pair_norm == _normalized_asset(asset):
                age = time.time() - float(self._last_ws_quote.get("ts", time.time()))
                if 0 <= age <= 3:
                    return {"price": float(self._last_ws_quote["price"]), "asset": asset,
                            "ts": float(self._last_ws_quote["ts"]), "observed_at": time.time(),
                            "timestamp_kind": "ws_server_time", "source": "ws_ticks"}
        if not await self.is_alive():
            return None
        try:
            title = await self._page.title()
            match = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s+(.+?)(?:\s*\||$)", title)
            if not match or _normalized_asset(match.group(2)) != _normalized_asset(asset):
                return None
            price = float(match.group(1).replace(",", "."))
            if not math.isfinite(price) or price <= 0:
                return None
            observed = time.time()
            self._selected_asset = asset
            return {"price": price, "asset": asset, "ts": observed,
                    "observed_at": observed, "timestamp_kind": "observation_time",
                    "source": "broker_browser_sample"}
        except Exception:
            return None

    async def get_current_price(self):
        quote = await self.get_current_quote(self._selected_asset or settings.ASSET)
        return quote["price"] if quote else None

    async def is_demo_account(self) -> bool:
        try:
            label = self._page.locator('[data-test="account-balance-title"]')
            return await label.is_visible() and (await label.inner_text()).strip().lower() == "demo account"
        except Exception:
            return False

    async def get_asset_payout(self, asset: str):
        if await self.get_current_quote(asset) is None:
            return None
        try:
            selected = self._page.locator('[data-test="assets-tabs-tab-selected"]')
            text = await selected.inner_text(timeout=2000)
            match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
            if not match:
                return None
            payout = float(match.group(1)) / 100.0
            return payout if 0 <= payout <= 1 else None
        except Exception:
            return None
