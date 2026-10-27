# One-second candle checks

Update: the default `hybrid` entry mode now allows a separate short-candle
strategy to place orders. See [short-candle entries](short_candle_entries.md)
for its rules and startup command. The original monitoring-only behavior
described below is still available through `--entry-mode classic`.

The orchestrator now targets one quote and BUY/SELL gate check per second.
Broker history requests run in one background task, so waiting for native
30-second and 1-minute candles does not pause quote sampling. Browser response
time and processing can still extend an individual poll beyond one second.

The trading rules in `strategies/closed_candle.py` are unchanged. Indicators,
swing breaks, RSI bands, and obstacle clearance continue to use completed
30-second/1-minute candles. The existing five-second confirmation window,
one submission attempt per entry candle, payout checks, daily limits, and
pending-order protections still apply. An entry blocked by a temporary payout
lookup failure can be reconsidered on the next second within that same window.

`agents/candle_monitor.py` separately aggregates 5-, 10-, and 15-second candles
from the observed browser quotes. These are sampled OHLC, not broker-native
tick-complete candles. They describe observed movement and do not predict the
next move or authorize trades. Startup partial candles and candles interrupted
by a sampling outage are discarded from completed history. The first complete
bars become available after the next full aligned interval.

Each valid poll writes:

- `[CandleMonitor]`: 5s/10s/15s forming OHLC, latest completed OHLC, observed
  up/down/flat movement, price change, source, freshness, and warmup status.
  `used_for_entries` is `false`.
- `[BuyOnlyGate]` and `[SellOnlyGate]`: all existing diagnostics, plus
  `checked_at`, `check_interval_seconds`, and `closed_snapshot_changed`.
  Repeated checks can show the same failures until a strategy candle closes.
  `candle_event` counts actual strategy candle updates, not polling iterations.

For the supplied SELL log, the unchanged checks reject the entry because:

- The 1-minute close of 1.36018 is above the swing low of 1.36013 and the
  previous candle low of 1.36017.
- RSI 32.0655 is below the existing SELL band of 35–50.
- Support distance 0.00005 is smaller than the required confirmation range
  of 0.00028.

Faster monitoring does not make these failed conditions pass or guarantee more
trades. Trading directly from 5s/10s/15s signals would require different rules.

Restart the existing orchestrator to load the updated files. To inspect the
new logs without order clicks, use the existing observation mode:

```bash
conda run --no-capture-output -n tg_algo python multi_agent_orchestrator.py --entry-mode classic --observe-only --max-runtime 65
```

Regression checks run offline with mocked broker operations and local browser
fixtures. Disable unrelated installed pytest plugins for these tests. The
process-lock regression explicitly enables locking in its isolated environment,
regardless of the local `.env` setting for multiple sessions:

```bash
conda run --no-capture-output -n tg_algo env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```
