# Taurus

An autonomous agent that watches the market, researches it, learns from what
it finds, and trades an aggressive strategy on that basis.

It runs in **simulation by default**. Live trading requires two independent
switches to be thrown deliberately.

---

## What it actually does

```
   market data  ─▶  features  ─▶  labels  ─▶  learned model
                                                   │
                                                   ▼
   risk limits ◀──  position sizing  ◀──  signals ─┘
        │
        ▼
   broker (paper │ live)  ─▶  decision journal
```

1. **Watches** — pulls OHLCV bars for a configurable universe and caches them.
2. **Researches** — builds 29 features per bar (momentum, mean reversion,
   volatility, participation, relative strength) and labels each bar by the
   triple-barrier method: did the profit target hit before the stop?
3. **Learns** — trains a calibrated gradient-boosted classifier to estimate
   `P(profit barrier before stop)`, validated by *purged, embargoed*
   walk-forward. Retrains on a schedule as new data arrives.
4. **Trades** — converts model probabilities into position sizes via
   volatility targeting and fractional Kelly, scaled by an aggression knob
   and bounded by hard caps.
5. **Records** — writes every decision, with its reasoning, to an append-only
   journal.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Try it in 60 seconds, with no data feed

```bash
python scripts/make_demo_data.py                       # synthetic market
python -m taurus.cli research --config config.offline.yaml
python -m taurus.cli backtest --config config.offline.yaml
```

## Against real data

```bash
python -m taurus.cli validate  --config config.yaml    # audit the risk settings
python -m taurus.cli research  --config config.yaml    # walk-forward + train
python -m taurus.cli backtest  --config config.yaml    # replay history
python -m taurus.cli signals   --config config.yaml    # today's picks, no trades
python -m taurus.cli trade     --config config.yaml    # run the agent
```

`data.provider` selects the feed: `yfinance` (free, default) or `csv`, which
reads `<csv_dir>/<SYMBOL>.csv` — use that for a paid vendor's export or behind
a restricted network.

---

## How "aggressive" is expressed

Aggression lives in sizing and signal parameters, in `config.yaml`:

| Setting | Default | Effect |
|---|---|---|
| `aggression` | 2.0 | Scales the whole book. 1.0 is a conventional vol-targeted portfolio. |
| `target_annual_vol` | 35% | Portfolio volatility target — roughly 2x a typical equity fund. |
| `max_gross_leverage` | 2.0x | Ceiling on total exposure. |
| `max_position_weight` | 25% | A single name may be a quarter of the book. |
| `max_positions` | 8 | Concentrated: few, large bets. |
| `kelly_fraction` | 0.5 | Half-Kelly on the model's estimated edge. |

## The guardrails, which do not turn off

Aggressive sizing without limits is not a strategy, it is a countdown. These
are enforced on every cycle regardless of configuration:

- **Drawdown kill switch** (20%) — flattens the book and stops new entries.
  It *latches*: recovering does not resume trading, because a breach means
  the strategy needs a human to look at it.
- **Daily loss halt** (6%) — stops the day from compounding. Clears overnight.
- **Progressive de-risking** — position sizes scale down from 100% to 25% as
  drawdown approaches the halt, rather than trading full size into a cliff.
- **Per-position stops** — initial stop at 2 ATR, trailing at 3 ATR.
- **Confidence floor** — nothing below `min_signal_confidence` is traded.
  Holding cash is a position.
- **Leverage and concentration caps** — applied last, always.

## Why you should believe the backtest (and where not to)

Most backtests are wrong in the same few ways. This one is built against them:

- **No lookahead.** Signals are computed from the close of bar `t`; orders fill
  at the **open of bar `t+1`**. `tests/test_backtest.py` asserts that a
  strategy is never handed a bar dated after the day it is deciding on — if
  that test fails, every number the system produces is fiction.
- **Causal indicators.** Appending future bars provably does not change any
  historical indicator value; this is tested directly.
- **Purged, embargoed walk-forward.** Triple-barrier labels resolve up to
  `max_holding_days` later, so training bars near a fold boundary contain
  test-period information. They are dropped, and an embargo is applied on the
  other side too.
- **Costs are charged.** Slippage (5 bps each way) and commission on every
  fill. A test asserts a zero-cost run beats a realistic-cost run — if it does
  not, costs are not actually being applied.
- **The accounting reconciles.** A test asserts that the sum of realized trade
  P&L equals the change in the equity curve. If the trade log and the equity
  curve disagree, there is no way to know which is lying.
- **Pessimistic ambiguity.** When both barriers fall inside one bar, the stop
  is assumed. When a bar gaps through a stop, the fill is the gap, not the
  stop price.
- **Buy-and-hold is reported next to every result.** A strategy that takes
  leverage and concentration risk to underperform an index fund has no reason
  to exist, and the tool says so.

**What it still does not model:** intraday fills (daily bars only), market
impact beyond a flat slippage estimate, borrow cost and locate for shorts
(shorts are off by default), dividends and corporate actions beyond
split/dividend-adjusted closes, financing cost on margin, and survivorship
bias in whatever universe you configure. Yahoo data is revised without notice.
Each of these makes live results worse than the backtest, never better.

## The model gate

A trained model is only deployed if it clears **three** gates:

1. AUC > 0.5 — below that it ranks worse than chance and its probabilities
   cannot be used for sizing.
2. Accuracy ≥ `min_oos_accuracy`.
3. **Accuracy above the majority-class baseline.**

The third matters most and is the one usually missed. With a 2:1 barrier
ratio only about one bar in three is a winner, so a model that *never*
predicts a winner scores ~68% accuracy while being worthless. The default run
on synthetic data does exactly this, and is correctly rejected:

```
  mean OOS accuracy 0.6754   mean AUC 0.4971
  majority-class baseline 0.6752   -> edge +0.0002
  MODEL REJECTED
```

The verdict is saved with the model, and **enforced at load time**: a
rejected model is not picked up by `backtest`, `signals`, or `trade`. A gate
checked when training and ignored when trading is not a gate. `--force-model`
overrides it for research comparisons; `trade` ignores that flag entirely.

When a model is rejected the agent trades the rules-based strategy alone.
**Do not respond to a rejection by lowering the gate.** Change the features
or the labels.

## Going live

Live trading needs **both**:

```yaml
execution:
  broker: alpaca
  allow_live: true
```

```bash
export TAURUS_ALLOW_LIVE=1
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
```

Two independent switches, so a stray config edit or a copied example file
cannot by itself put money at risk. Credentials come from the environment
only — never from a file that could be committed. `taurus trade` additionally
prompts for typed confirmation before its first live order.

Run against Alpaca's **paper** endpoint (the default when either switch is
missing) until the strategy has a track record you actually trust.

## Layout

```
taurus/
  config.py              all tunables; the risk posture of a run in one object
  data/providers.py      yfinance / CSV providers, disk cache
  features/              causal indicators and the feature matrix builder
  research/              labeling, model, purged walk-forward, dataset panel
  strategy/              base interface, ML alpha, momentum breakout, ensemble
  risk/                  sizing, portfolio limits, kill switch, allocator
  execution/             broker interface, paper broker, gated Alpaca adapter
  backtest/              event-driven engine, performance metrics
  agent/                 the trading loop and the decision journal
  cli.py                 research | backtest | signals | trade | validate
tests/                   86 tests
```

The backtester and the live agent share one allocator (`risk/allocator.py`)
rather than each having their own. If they diverged, the backtest would stop
describing what the live agent does — and a backtest that does not describe
live behavior is worse than none, because it is trusted.

## Tests

```bash
python -m pytest tests/ -q      # 86 passed
```

---

## Before you trade real money with this

This is working software, not a profitable strategy. Nothing here has been
shown to have edge — the shipped strategies underperform buy-and-hold on the
synthetic data, and the model is rejected by its own gate. That is the system
being honest, not broken.

The realistic path is: supply good data, do the research to find an actual
edge, validate it walk-forward, paper trade it for months, and only then
consider size. Aggressive leverage applied to a strategy with no edge does
not produce aggressive returns — it produces the drawdown kill switch, faster.

Trading involves risk of substantial loss, and leverage can lose more than
the capital deployed. This is not financial advice.
