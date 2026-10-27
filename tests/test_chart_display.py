"""Exercise verified chart controls with local broker-shaped DOM, never orders."""
import asyncio
import json

from playwright.async_api import async_playwright

from bot.chart_indicators import configure_strategy_chart, set_chart_timeframe


MARKUP = r'''
<button data-test="trade-time-duration-input" onclick="window.orderDurationClicks++">1m</button>
<button data-test="TimeFrameSelector.selected" onclick="openTimeframes()"></button>
<div data-test="TimeFrameSelector.selector" hidden>
  <button data-test="TimeFrameSelector.selector.button" onclick="selectTimeframe('1m')">1m</button>
  <button data-test="TimeFrameSelector.selector.button" onclick="selectTimeframe('30s')">30s</button>
</div>
<button data-test="indicator-menu" onclick="document.querySelector('#catalog').hidden=false">Indicators</button>
<div id="catalog" hidden>
  <input data-test="techanalysis-instrument-search-field">
  <div data-test="techanalysis-instrument-list-item-title"><span data-trans="indicators_menu_title_ema" onclick="addIndicator('EMA')">EMA</span></div>
  <div data-test="techanalysis-instrument-list-item-title"><span data-trans="indicators_menu_title_rsi" onclick="addIndicator('RSI')">RSI</span></div>
  <div data-test="techanalysis-instrument-list-item-title"><span data-trans="indicators_menu_title_demarker" onclick="addIndicator('DeMarker')">DeMarker</span></div>
</div>
<div id="legends"></div>
<div data-test="drawing-tool" id="drawing">User trend line</div>
<section data-test="dynamic_ChartIndicators" hidden>
  <label data-test="valueInputField">Period <input id="period"></label>
  <label data-test="valueInputField">Upper band <input id="upper" value="70"></label>
  <button data-test="indicatorsFormDone" onclick="saveIndicator()">Done</button>
</section>
<script>
window.orderDurationClicks = 0;
window.changes = 0;
window.target = null;
function openTimeframes() {document.querySelector('[data-test="TimeFrameSelector.selector"]').hidden=false;}
function selectTimeframe(value) {
  state.timeframe = value;
  document.querySelector('[data-test="TimeFrameSelector.selector"]').hidden=true;
  window.changes++;
  render();
}
function addIndicator(name) {
  state.indicators.push({name, period: name === 'EMA' ? 20 : 14});
  window.changes++;
  render();
}
function removeIndicator(index) {state.indicators.splice(index,1); window.changes++; render();}
function editIndicator(index) {
  target = index;
  document.querySelector('[data-test$="_ChartIndicators"]').hidden=false;
  document.querySelector('#period').value=state.indicators[index].period;
}
function saveIndicator() {
  if (!state.ignorePeriodSave) {
    state.indicators[target].period=Number(document.querySelector('#period').value);
  }
  document.querySelector('[data-test$="_ChartIndicators"]').hidden=true;
  window.changes++;
  render();
}
function render() {
  document.querySelector('[data-test="TimeFrameSelector.selected"]').textContent=state.timeframe;
  document.querySelector('#legends').innerHTML=state.indicators.map((item,index) =>
    '<div data-test="techanalysis-active-tool-legend">' +
    '<button data-test="tech-analysis-legend-title" onclick="editIndicator('+index+')">'+item.name+'</button>' +
    '<span data-test="techanalysis-active-instrument-panel-content">'+item.period+'</span>' +
    '<button data-test="tech-analysis-legend-remove" onclick="removeIndicator('+index+')">X</button></div>'
  ).join('');
}
</script>
'''


def run_page(initial, check):
    async def run():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.set_content(MARKUP)
                await page.evaluate("(initial) => {window.state=initial; render();}", initial)
                await check(page)
            finally:
                await browser.close()
    asyncio.run(run())


def test_entry_replaces_bollinger_and_verifies_each_ema_without_order_controls():
    async def check(page):
        result = await configure_strategy_chart(page)
        assert result["verified"], json.dumps(result)
        assert result["observed"] == {
            "timeframe": "1m", "indicators": [
                {"name": "EMA", "period": 9}, {"name": "EMA", "period": 21},
                {"name": "RSI", "period": 14},
            ],
        }
        assert await page.evaluate("window.orderDurationClicks") == 0
        assert await page.locator('[data-test="drawing-tool"]').inner_text() == "User trend line"
        assert await page.locator("#upper").input_value() == "70"
        previous_changes = await page.evaluate("window.changes")
        again = await configure_strategy_chart(page)
        assert again["verified"] and not again["changed"]
        assert again["reason"] == "already_matched"
        assert await page.evaluate("window.changes") == previous_changes
    run_page({"timeframe": "30s", "indicators": [{"name": "Bollinger Bands", "period": 20}]}, check)


def test_trend_profile_removes_duplicate_and_rsi_preserves_matching_ema():
    async def check(page):
        result = await configure_strategy_chart(page, "trend")
        assert result["verified"], json.dumps(result)
        assert result["observed"] == {"timeframe": "30s", "indicators": [
            {"name": "EMA", "period": 20}, {"name": "EMA", "period": 50}]}
        assert await page.evaluate("window.orderDurationClicks") == 0
    run_page({"timeframe": "1m", "indicators": [
        {"name": "EMA", "period": 20}, {"name": "EMA", "period": 20},
        {"name": "RSI", "period": 14}]}, check)


def test_successful_form_click_with_wrong_legend_period_is_not_verified():
    async def check(page):
        result = await configure_strategy_chart(page)
        assert not result["verified"]
        assert result["reason"] == "indicator_readback_mismatch"
        assert {"name": "EMA", "period": 20} in result["observed"]["indicators"]
    run_page({"timeframe": "1m", "indicators": [], "ignorePeriodSave": True}, check)


def test_missing_chart_timeframe_never_clicks_order_duration_or_claims_success(monkeypatch):
    monkeypatch.setattr("bot.chart_indicators.CONTROL_READY_MS", 80)
    async def check(page):
        await page.locator('[data-test="TimeFrameSelector.selected"]').evaluate("(node) => node.remove()")
        assert not await set_chart_timeframe(page, "1m")
        result = await configure_strategy_chart(page)
        assert not result["verified"]
        assert result["reason"] == "timeframe_not_verified"
        assert await page.evaluate("window.orderDurationClicks") == 0
        assert await page.evaluate("window.changes") == 0
    run_page({"timeframe": "30s", "indicators": []}, check)

def test_unmounted_legends_are_rendered_before_removing_duplicates():
    async def check(page):
        # Reproduce the broker: saved instances exist while their chart legends
        # are unmounted until the technical analysis panel is opened.
        await page.evaluate("""() => {
            document.querySelector('#legends').innerHTML = '';
            document.querySelector('[data-test="indicator-menu"]').onclick = () => {
                document.querySelector('#catalog').hidden = false;
                render();
            };
        }""")
        result = await configure_strategy_chart(page)
        assert result["verified"], json.dumps(result)
        assert sorted((item["name"], item["period"]) for item in
                      result["observed"]["indicators"]) == [
            ("EMA", 9), ("EMA", 21), ("RSI", 14)]
    run_page({"timeframe": "1m", "indicators": [
        {"name": "Bollinger Bands", "period": 20},
        {"name": "EMA", "period": 9}, {"name": "EMA", "period": 9},
        {"name": "EMA", "period": 21}, {"name": "RSI", "period": 14}]}, check)


def test_delayed_indicator_menu_does_not_leave_default_timeframe_or_fail():
    async def check(page):
        await page.evaluate("""() => {
            const menu = document.querySelector('[data-test="indicator-menu"]');
            menu.remove();
            setTimeout(() => document.body.appendChild(menu), 3200);
        }""")
        result = await configure_strategy_chart(page)
        assert result["verified"], json.dumps(result)
        assert result["observed"]["timeframe"] == "1m"
        assert result["attempts"] == 1
        assert await page.evaluate("window.orderDurationClicks") == 0
    run_page({"timeframe": "5s", "indicators": [
        {"name": "EMA", "period": 9}, {"name": "EMA", "period": 21},
        {"name": "RSI", "period": 14}]}, check)


def test_partial_indicator_save_retries_actual_state_without_duplicate(monkeypatch):
    import bot.chart_indicators as display
    original = display._add_indicator
    calls = []
    async def interrupted(page, name, period):
        await original(page, name, period)
        calls.append((name, period))
        if len(calls) == 1:
            raise TimeoutError("UI response lost after successful save")
    monkeypatch.setattr(display, "_add_indicator", interrupted)
    async def check(page):
        result = await display.configure_strategy_chart(page)
        assert result["verified"], json.dumps(result)
        assert result["attempts"] == 2
        assert len(result["observed"]["indicators"]) == 3
        assert calls.count(("EMA", 9)) == 1
        assert await page.evaluate("window.orderDurationClicks") == 0
    run_page({"timeframe": "5s", "indicators": []}, check)
