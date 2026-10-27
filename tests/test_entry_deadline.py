import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.agent_pipeline_supervisor import PipelineSupervisorAgent
from bot.browser import OrderUncertainError
from config import settings
from models.predictor import Signal
from test_browser_execution import execution_bot


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    fake_time = SimpleNamespace(time=lambda: clock.now)
    monkeypatch.setattr("bot.browser.time", fake_time)
    monkeypatch.setattr("agents.agent_pipeline_supervisor.time", fake_time)
    monkeypatch.setattr(settings, "CONFIDENCE_THRESHOLD", 0.5)
    return clock


def supervisor(bot=None):
    bot = bot or SimpleNamespace(place_ftt_trade=AsyncMock(return_value=True))
    risk = SimpleNamespace(approve=MagicMock(return_value=(True, "")))
    agent = PipelineSupervisorAgent(bot, risk, None, None)
    agent._current_asset = "EUR/USD OTC"
    return agent


def confirm_order(bot, signal=Signal.BUY):
    bot._order_rows.side_effect = [[], [{
        "id": "deadline-demo-order", "asset": "EUR/USD OTC",
        "amount": float(settings.TRADE_AMOUNT), "direction": signal.value,
        "duration": settings.FTT_DURATION,
    }]]


@pytest.mark.parametrize("deadline", [99.0, 100.0, float("inf"), float("-inf"),
                                      float("nan"), "bad-deadline"])
def test_invalid_or_elapsed_deadline_rejected_before_any_work(clock, deadline):
    bot, button = execution_bot()
    assert not asyncio.run(bot.place_ftt_trade(
        Signal.BUY, settings.FTT_DURATION, "EUR/USD OTC", entry_deadline=deadline))
    bot.is_demo_account.assert_not_awaited()
    button.click.assert_not_awaited()

    agent = supervisor()
    assert not asyncio.run(agent.safe_execute_trade(
        Signal.BUY, 0.95, "ftt", entry_deadline=deadline))
    agent.risk.approve.assert_not_called()
    agent.bot.place_ftt_trade.assert_not_awaited()
    assert not agent.execution_uncertain


@pytest.mark.parametrize("layer", ["browser", "supervisor"])
def test_signal_expiring_while_waiting_for_lock_does_not_execute(clock, layer):
    bot, button = execution_bot()
    agent = supervisor(bot)

    async def run():
        lock = bot._order_lock if layer == "browser" else agent._trade_lock
        await lock.acquire()
        if layer == "browser":
            attempt = bot.place_ftt_trade(
                Signal.BUY, settings.FTT_DURATION, entry_deadline=101.0)
        else:
            attempt = agent.safe_execute_trade(Signal.BUY, 0.95, "ftt", entry_deadline=101.0)
        task = asyncio.create_task(attempt)
        await asyncio.sleep(0)
        clock.now = 101.0
        lock.release()
        assert not await task

    asyncio.run(run())
    bot.is_demo_account.assert_not_awaited()
    button.click.assert_not_awaited()
    assert not agent.execution_uncertain


def test_deadline_checked_after_final_awaited_precondition(clock, caplog):
    bot, button = execution_bot()
    agent = supervisor(bot)
    checks = 0

    async def verify_demo():
        nonlocal checks
        checks += 1
        if checks == 2:
            clock.now = 101.0
        return True

    bot.is_demo_account.side_effect = verify_demo
    with caplog.at_level("INFO"):
        assert not asyncio.run(agent.safe_execute_trade(
            Signal.BUY, 0.95, "ftt", entry_deadline=101.0))
    assert checks == 2
    assert "signal_expired" in caplog.text
    button.click.assert_not_awaited()
    assert not agent.execution_uncertain

    # A known non-click expiry does not block a later fresh signal.
    confirm_order(bot)
    assert asyncio.run(agent.safe_execute_trade(
        Signal.BUY, 0.95, "ftt", entry_deadline=102.0))
    button.click.assert_awaited_once()


def test_supervisor_forwards_deadline_without_changing_classic_contract(clock):
    agent = supervisor()

    async def run():
        assert await agent.safe_execute_trade(Signal.BUY, 0.95, "ftt", entry_deadline=101.0)
        agent.bot.place_ftt_trade.assert_awaited_once_with(
            Signal.BUY, settings.FTT_DURATION, asset="EUR/USD OTC", entry_deadline=101.0)
        for options in ({}, {"entry_deadline": None}):
            agent.bot.place_ftt_trade.reset_mock()
            assert await agent.safe_execute_trade(Signal.SELL, 0.95, "ftt", **options)
            agent.bot.place_ftt_trade.assert_awaited_once_with(
                Signal.SELL, settings.FTT_DURATION, asset="EUR/USD OTC")

    asyncio.run(run())


@pytest.mark.parametrize("remaining, expected", [(5.0, 3000), (0.5, 500), (0.0005, 0.5)])
def test_click_timeout_capped_without_shortening_acknowledgement(clock, remaining, expected):
    bot, button = execution_bot()
    confirm_order(bot)

    async def clicked(**kwargs):
        # A timely submitted order may be acknowledged after the entry deadline.
        clock.now += 10

    button.click.side_effect = clicked
    assert asyncio.run(bot.place_ftt_trade(
        Signal.BUY, settings.FTT_DURATION, "EUR/USD OTC", entry_deadline=100 + remaining))
    assert button.click.await_args.kwargs["timeout"] == pytest.approx(expected)
    assert bot.last_order["id"] == "deadline-demo-order"


def test_classic_browser_click_keeps_existing_timeout(clock):
    bot, button = execution_bot()
    confirm_order(bot)
    assert asyncio.run(bot.place_ftt_trade(Signal.BUY, settings.FTT_DURATION, "EUR/USD OTC"))
    button.click.assert_awaited_once_with(timeout=3000)


@pytest.mark.parametrize("blocker", ["risk", "confidence", "circuit", "mode", "uncertain"])
def test_deadline_does_not_bypass_existing_guards(clock, blocker):
    agent = supervisor()
    confidence, mode = 0.95, "ftt"
    if blocker == "risk":
        agent.risk.approve.return_value = (False, "daily_limit")
    elif blocker == "confidence":
        confidence = 0.1
    elif blocker == "circuit":
        agent.cb.should_stop = lambda: (True, "cooldown")
    elif blocker == "mode":
        mode = "forex"
    else:
        agent.execution_uncertain = True
    assert not asyncio.run(agent.safe_execute_trade(
        Signal.BUY, confidence, mode, entry_deadline=101.0))
    agent.bot.place_ftt_trade.assert_not_awaited()


def test_click_timeout_is_uncertain_and_cannot_be_retried(clock):
    bot, button = execution_bot()
    button.click.side_effect = TimeoutError("possible broker submission")
    agent = supervisor(bot)

    async def run():
        assert not await agent.safe_execute_trade(Signal.BUY, 0.95, "ftt", entry_deadline=101.0)
        assert agent.execution_uncertain
        assert not await agent.safe_execute_trade(Signal.BUY, 0.95, "ftt", entry_deadline=102.0)

    asyncio.run(run())
    button.click.assert_awaited_once()


def test_browser_exposes_uncertain_click_without_converting_it_to_expiry(clock):
    bot, button = execution_bot()

    async def expired_click(**kwargs):
        clock.now = 102.0
        raise TimeoutError("possible broker submission")

    button.click.side_effect = expired_click
    with pytest.raises(OrderUncertainError):
        asyncio.run(bot.place_ftt_trade(
            Signal.BUY, settings.FTT_DURATION, "EUR/USD OTC", entry_deadline=101.0))
    button.click.assert_awaited_once()


@pytest.mark.parametrize("guard_result", [False, None, "true", "raises"])
def test_invalid_signal_guard_rejects_before_any_work(clock, guard_result, caplog):
    bot, button = execution_bot()

    def guard():
        if guard_result == "raises":
            raise RuntimeError("signal state unavailable")
        return guard_result

    with caplog.at_level("INFO"):
        assert not asyncio.run(bot.place_ftt_trade(
            Signal.BUY, settings.FTT_DURATION, "EUR/USD OTC",
            entry_deadline=105.0, entry_guard=guard))
        agent = supervisor()
        assert not asyncio.run(agent.safe_execute_trade(
            Signal.BUY, 0.95, "ftt", entry_deadline=105.0, entry_guard=guard))
    bot.is_demo_account.assert_not_awaited()
    button.click.assert_not_awaited()
    agent.risk.approve.assert_not_called()
    agent.bot.place_ftt_trade.assert_not_awaited()
    assert not agent.execution_uncertain
    assert "signal_invalidated" in caplog.text


@pytest.mark.parametrize("layer", ["browser", "supervisor"])
def test_guard_rechecked_after_waiting_for_lock(clock, layer):
    bot, button = execution_bot()
    agent = supervisor(bot)
    state = SimpleNamespace(valid=True)

    async def run():
        lock = bot._order_lock if layer == "browser" else agent._trade_lock
        await lock.acquire()
        if layer == "browser":
            attempt = bot.place_ftt_trade(
                Signal.BUY, settings.FTT_DURATION,
                entry_deadline=105.0, entry_guard=lambda: state.valid)
        else:
            attempt = agent.safe_execute_trade(
                Signal.BUY, 0.95, "ftt", entry_deadline=105.0,
                entry_guard=lambda: state.valid)
        task = asyncio.create_task(attempt)
        await asyncio.sleep(0)
        state.valid = False
        lock.release()
        assert not await task

    asyncio.run(run())
    bot.is_demo_account.assert_not_awaited()
    button.click.assert_not_awaited()
    assert not agent.execution_uncertain


@pytest.mark.parametrize("during", ["amount", "final_demo", "guard_exception"])
def test_invalidated_signal_cannot_click_after_ui_preparation(clock, during, caplog):
    bot, button = execution_bot()
    agent = supervisor(bot)
    state = SimpleNamespace(valid=True)

    def guard():
        if not state.valid and during == "guard_exception":
            raise RuntimeError("signal state unavailable")
        return state.valid

    async def invalidate_amount(*args):
        state.valid = False
        return True

    demo_checks = 0

    async def invalidate_final_demo():
        nonlocal demo_checks
        demo_checks += 1
        if demo_checks == 2:
            state.valid = False
        return True

    if during == "amount":
        bot.set_amount.side_effect = invalidate_amount
    else:
        bot.is_demo_account.side_effect = invalidate_final_demo
    with caplog.at_level("INFO"):
        assert not asyncio.run(agent.safe_execute_trade(
            Signal.BUY, 0.95, "ftt", entry_deadline=105.0, entry_guard=guard))
    button.click.assert_not_awaited()
    assert not agent.execution_uncertain
    assert "signal_invalidated" in caplog.text

    # A known non-click must not latch the unresolved-order guard.
    state.valid = True
    bot.set_amount.side_effect = None
    bot.is_demo_account.side_effect = None
    confirm_order(bot)
    assert asyncio.run(agent.safe_execute_trade(
        Signal.BUY, 0.95, "ftt", entry_deadline=105.0, entry_guard=guard))
    button.click.assert_awaited_once()


def test_supervisor_only_forwards_guard_when_provided(clock):
    agent = supervisor()
    guard = lambda: True
    assert asyncio.run(agent.safe_execute_trade(
        Signal.BUY, 0.95, "ftt", entry_deadline=105.0, entry_guard=guard))
    agent.bot.place_ftt_trade.assert_awaited_once_with(
        Signal.BUY, settings.FTT_DURATION, asset="EUR/USD OTC",
        entry_deadline=105.0, entry_guard=guard)

    agent.bot.place_ftt_trade.reset_mock()
    assert asyncio.run(agent.safe_execute_trade(Signal.BUY, 0.95, "ftt", entry_guard=None))
    agent.bot.place_ftt_trade.assert_awaited_once_with(
        Signal.BUY, settings.FTT_DURATION, asset="EUR/USD OTC")
