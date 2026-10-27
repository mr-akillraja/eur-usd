# Short-candle entries

The default `hybrid` mode evaluates both the original 30s/1m strategy and a
separate strategy that can enter from completed 5s/10s/15s candles. A failing
original gate no longer prevents a passing short setup from reaching execution.
`strategies/closed_candle.py` retains the original rules unchanged.

The short strategy uses the existing EMA and Wilder RSI calculations:

| Period | Role | BUY requirements (SELL reverses direction) |
| --- | --- | --- |
| 15 seconds | Trend | EMA9 above EMA21, close above EMA9, EMA9 rising over three closed intervals |
| 10 seconds | Confirmation | EMA9 above EMA21 and close above EMA9 |
| 5 seconds | Entry | Bullish body, close above the previous candle's high, RSI14 rising in 50–70 |

SELL requires a bearish 5s candle closing below the previous low and falling
RSI14 in 30–50. The 5s confirmation range must be positive and no greater than
twice the mean range of the previous ten 5s candles in either direction.
The original 1m swing, 1m pullback, and 30s resistance/support filters are not
requirements of this separate short strategy.

All three series must be valid, aligned, contiguous, and current. Forming bars,
startup partial bars, and bars interrupted by a feed gap cannot qualify.
Histories are built from the actual one-second browser observations and remain
labelled `broker_browser_sample`; these are not tick-complete native candles.

Warmup needs 16 completed 5s bars, 21 completed 10s bars, and 21 completed 15s
bars: about 5 minutes 15 seconds to 5 minutes 30 seconds of uninterrupted
sampling after setup. A feed outage restarts warmup. Classic signals remain
available during this period when their existing requirements pass.

Each fresh 5s close can create a new short candidate. It expires when the next
5s interval closes. Freshness is checked before submission and the browser
checks the deadline and feed validity again immediately before clicking. A slow
form preparation or a feed interruption therefore cannot reuse an expired or
invalidated short setup.

Both strategies use the existing shared execution path: demo-account checks,
configured stake, payout threshold, daily limits, pending exposure, circuit
breaker, one unresolved order at a time, durable order state, and confirmed
broker settlement. The trade duration remains **60 seconds**. The 5s/10s/15s
periods describe signal candles, not option expiry. Accepted short orders carry
`state_key=SHORT_CANDLE_GATE`; they count toward the same risk budget as classic
orders. Restarting does not reset the day's journal or risk limits.

Each strategy attempts an entry candle at most once, including known rejections.
If fresh classic and short signals disagree, that poll does not submit an order.
If both agree, the classic candidate takes priority and only one is submitted.

Restart the existing process to load the change; a process already running keeps
its previous code. The normal command now enables both strategies:

```bash
conda run --no-capture-output -n tg_algo python multi_agent_orchestrator.py --entry-mode hybrid
```

Other modes:

- `--entry-mode classic`: original 30s/1m entries; short candles remain monitoring-only.
- `--entry-mode short`: short entries only; original gates remain diagnostic.
- Add `--observe-only` to evaluate without submitting orders.

Look for `[EntryMode] mode=hybrid`, `[CandleMonitor]` with
`used_for_entries: true`, and `[ShortBuyGate]` / `[ShortSellGate]`. Short gates
report warmup, stale data, and individual failed rules. A qualifying setup must
also pass execution checks; `[Execution]` reports the result.

Tests exercise passing BUY and SELL signals through the order path with mocked
broker calls, along with rejection, stale signals, duplicate checks, conflicting
signals, and risk limits. This validates execution behavior, not profitability
or a guaranteed trade count.

An offline replay of the recorded 2026-10-04 13:03–13:24 IST quote observations
evaluated 1,261 polls and found 25 distinct qualifying 5s candles (22 BUY, 3 SELL).
For example, a short BUY qualified at 13:09:50 while both classic gates failed.
The [replay report](short_entry_replay_20261004.json) records these candidate
signals; it placed no orders and does not account for execution or risk blocks.
