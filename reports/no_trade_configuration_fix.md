# Trade-limit configuration fix — 2026-10-05

The October 4 evening run had already accepted 20 trades that day. The
orchestrator imposed a hardcoded 20-trade cap even though `config/.env`
specified a different limit. Valid short BUY signals at 18:35:10 and 18:35:15
passed the strategy and then hit that daily cap. See
[restored accounting](../logs/runs/unified_20261004_182616.log) and the
`[DailyRisk] status=blocked accepted=20` records following those signals.

The October 5 run started with zero accepted trades. Its first five minutes
were short-strategy warmup, followed by ordinary strategy rejections; that
initial wait was not caused by an exhausted daily cap.

## Applied settings

| Setting | Previous effective runtime value | New configured value |
| --- | ---: | ---: |
| `MAX_DAILY_TRADES` | 20 | 50 |
| `TRADE_AMOUNT` | 200 | 200 |
| `MAX_DAILY_LOSS_USD` | 2000 | 2000 |
| `CONFIDENCE_THRESHOLD` | 0.50 | 0.50 |
| `RESET_TODAY_LOGS_ON_STARTUP` | Ignored by accounting | false |

The previously ignored `.env` entries were 2000 trades, a 1000 stake and a
10000 daily loss limit. They are now aligned with the intended effective
profile above. Duplicate stake and confidence entries have been consolidated.

`MultiAgentTradingSystem` now reads the current settings when constructed.
Explicit command-line or constructor overrides take precedence. The runtime
and risk manager share those same daily limits. Configured daily limits are
passed explicitly even in training mode, preserving the previous limit
selection behavior. Settings fallbacks also use a 200 stake, 50 trades and
2000 daily loss limit.

Both strategy evaluator files, risk approval rules and supervisor execution
logic are unchanged. Hybrid mode still uses native 30s/1m classic candles and
sampled 5s/10s/15s short candles; the uploaded 20s display is not a new strategy
input. Expiry remains 60 seconds. Freshness, payout, loss cooldowns, pending
exposure, single unresolved order and journal persistence still apply.

## Loading the change

A running process retains its old settings. Stop it normally and restart with:

```bash
conda run --no-capture-output -n tg_algo python multi_agent_orchestrator.py --entry-mode hybrid
```

The startup `[Risk] Effective limits` line should show
`max_daily_trades=50 max_daily_loss=2000.00`. Remove any old explicit
`--max-daily-trades 20` flag to use the configuration. Add `--observe-only`
to evaluate without submitting orders. Short signals need roughly 5m15s–5m30s
of uninterrupted observations after startup.

The change allows qualifying signals beyond trade 20 when the remaining checks
pass. It does not force entries or establish a profitable win rate. Verification
uses local tests and mocked broker calls; no orders were placed for this fix.

## Verification

Full suite: **344 passed** with:

```bash
conda run --no-capture-output -n tg_algo env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

New configuration regressions cover settings read at construction, individual
CLI overrides, risk approval after 20 accepted trades, rejection at 50 after
restart, and continued loss-budget, pending-exposure, per-trade, concurrency
and confidence enforcement. Existing execution tests cover qualifying BUY
and SELL submissions using mocked broker calls. File comparisons confirm both
strategy evaluators, `bot/risk.py` and the supervisor are unchanged.
