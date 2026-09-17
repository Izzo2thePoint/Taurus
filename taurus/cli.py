"""Command-line interface.

    python -m taurus.cli research   --config config.yaml
    python -m taurus.cli backtest   --config config.yaml
    python -m taurus.cli signals    --config config.yaml
    python -m taurus.cli trade      --config config.yaml --cycles 1
    python -m taurus.cli validate   --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from .agent.journal import Journal
from .agent.loop import TradingAgent
from .backtest.engine import BacktestEngine
from .config import Config
from .data.providers import load_universe, make_provider
from .execution.alpaca import make_broker
from .execution.paper import PaperBroker
from .research.dataset import build_panel
from .research.model import AlphaModel
from .research.walkforward import run_walk_forward, train_production_model
from .strategy.ensemble import EnsembleStrategy
from .strategy.ml_alpha import MLAlphaStrategy
from .strategy.momentum import MomentumBreakoutStrategy

log = logging.getLogger("taurus")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.ERROR)


def _load_data(config: Config, end: str | None = None):
    provider = make_provider(config.data)
    log.info("loading %d symbols via %s from %s", len(config.data.universe),
             type(provider).__name__, config.data.history_start)
    bars = load_universe(provider, config.data.universe,
                         config.data.history_start, end, config.data.interval)
    panel = build_panel(bars, config)
    return bars, panel


def _load_gated_model(config: Config, force: bool = False) -> AlphaModel | None:
    """Load the saved model, but only return it if it cleared its gate.

    A gate that is checked at training time and ignored at trading time is
    not a gate. `force` is for research comparisons, never for live trading.
    """
    try:
        model = AlphaModel.load(config.model.model_dir)
    except FileNotFoundError:
        log.warning("no saved model found; run `research` first. Using rules only.")
        return None

    metrics = model.metrics
    if metrics is None:
        log.warning("saved model has no recorded metrics; treating as untested. "
                    "Using rules only.")
        return None if not force else model

    if not metrics.tradeable:
        message = (f"saved model FAILED its gate (accuracy {metrics.accuracy:.4f} "
                   f"vs baseline {metrics.baseline_accuracy:.4f}, "
                   f"AUC {metrics.auc:.4f})")
        if force:
            log.warning("%s — using it anyway because --force-model was passed. "
                        "Do not do this with real money.", message)
            return model
        log.warning("%s. Trading rules only.", message)
        return None

    log.info("loaded model: accuracy %.4f (baseline %.4f), AUC %.4f",
             metrics.accuracy, metrics.baseline_accuracy, metrics.auc)
    return model


def _build_strategy(config: Config, model: AlphaModel | None):
    rules = MomentumBreakoutStrategy(config.risk)
    if model is None:
        return rules
    return EnsembleStrategy([MLAlphaStrategy(model, config.risk), rules],
                            weights=[0.65, 0.35])


# --- commands -------------------------------------------------------------

def cmd_research(args) -> int:
    """Walk-forward evaluation, then fit and save the production model."""
    config = Config.load(args.config)
    _, panel = _load_data(config, args.end)

    print("\nWalk-forward validation (purged, embargoed):")
    result = run_walk_forward(panel, config.model)
    print(result.summary().to_string(index=False))
    baseline = sum(f.metrics.baseline_accuracy for f in result.folds) / len(result.folds)
    print(f"\n  mean OOS accuracy {result.mean_accuracy:.4f}"
          f"   mean AUC {result.mean_auc:.4f}   folds {len(result.folds)}")
    print(f"  majority-class baseline {baseline:.4f}"
          f"   -> edge {result.mean_accuracy - baseline:+.4f}")

    gate = config.model.min_oos_accuracy
    if (result.mean_accuracy < gate or result.mean_auc <= 0.5
            or result.mean_accuracy <= baseline):
        print(f"\n  MODEL REJECTED: it does not clear all three gates "
              f"(accuracy >= {gate}, AUC > 0.5,\n  and accuracy above the "
              f"majority-class baseline of {baseline:.4f}).")
        print("  Accuracy above the baseline is the one that matters: a model "
              "can look 68%\n  accurate purely by never predicting a winner.")
        print("  The agent will trade rules only. Do not fix this by lowering "
              "the gate —\n  change the features or the labels.")

    model, metrics = train_production_model(panel, config.model)
    path = model.save(config.model.model_dir)
    print(f"\n  Production model saved to {path}")
    print(f"  Holdout accuracy {metrics.accuracy:.4f} "
          f"(baseline {metrics.baseline_accuracy:.4f}, "
          f"edge {metrics.accuracy_edge:+.4f})  AUC {metrics.auc:.4f}"
          f"  Brier {metrics.brier:.4f}")
    print(f"  Gate verdict: {'TRADEABLE' if metrics.tradeable else 'NOT TRADEABLE'}"
          f" — {'the agent will use it' if metrics.tradeable else 'the agent will trade rules only'}")

    if args.importance:
        from .research.dataset import feature_matrix, training_rows
        rows = training_rows(panel)
        X, _ = feature_matrix(rows)
        imp = model.feature_importance(X.tail(3000), rows["target"].astype(int).tail(3000))
        print("\n  Top features by permutation importance:")
        for name, value in imp.head(12).items():
            print(f"    {name:<22} {value:+.5f}")

    Journal(config.agent.journal_path).log_research(
        date.today(),
        {"mean_oos_accuracy": result.mean_accuracy, "mean_auc": result.mean_auc,
         "folds": len(result.folds), "holdout_accuracy": metrics.accuracy})
    return 0


def cmd_backtest(args) -> int:
    """Replay the strategy over history and report performance."""
    config = Config.load(args.config)
    bars, panel = _load_data(config, args.end)

    model = None if args.rules_only else _load_gated_model(config, args.force_model)
    strategy = _build_strategy(config, model)
    result = BacktestEngine(config, strategy).run(panel, bars, args.start, args.end)

    print(f"\nBacktest — {type(strategy).__name__}")
    print(f"  Aggression {config.risk.aggression}  "
          f"max leverage {config.risk.max_gross_leverage}x  "
          f"vol target {config.risk.target_annual_vol:.0%}\n")
    print(result.summary())

    # Buy-and-hold is the bar to clear. A strategy that takes leverage and
    # concentration risk to underperform an index fund has no reason to exist.
    curve = result.equity_curve
    if len(curve) > 1:
        d0, d1 = curve.index[0], curve.index[-1]
        rets = [bars[s].loc[d1, "close"] / bars[s].loc[d0, "close"] - 1.0
                for s in bars if d0 in bars[s].index and d1 in bars[s].index]
        if rets:
            bh = sum(rets) / len(rets)
            print(f"\n  Equal-weight buy & hold over the same window: {bh:+.2%}")
            verdict = "BEATS" if result.report.total_return > bh else "TRAILS"
            print(f"  Strategy {verdict} buy & hold.")

    if args.out:
        curve.to_csv(args.out)
        print(f"\n  Equity curve written to {args.out}")
    return 0


def cmd_signals(args) -> int:
    """Print today's signals without trading anything."""
    config = Config.load(args.config)
    bars, panel = _load_data(config, args.end)

    model = _load_gated_model(config, args.force_model)
    strategy = _build_strategy(config, model)
    agent = TradingAgent(config, PaperBroker(config.execution.starting_cash),
                         model=model, strategy=strategy)
    agent._bars, agent._panel = bars, panel

    snapshot = agent._latest_snapshot(date.today())
    if snapshot is None:
        print("No snapshot available.")
        return 1

    signals = strategy.generate(snapshot)
    print(f"\nSignals as of {snapshot.as_of} ({len(signals)} actionable)\n")
    if not signals:
        print("  Nothing clears the confidence threshold. Holding cash is a position.")
        return 0

    incumbents: set[str] = set()
    from .risk.allocator import target_weights
    weights = target_weights(signals, config, incumbents)

    print(f"  {'symbol':<8}{'dir':>5}{'conf':>8}{'vol':>8}{'weight':>9}   rationale")
    for sig in signals:
        w = weights.get(sig.symbol, 0.0)
        print(f"  {sig.symbol:<8}{sig.direction:>5}{sig.confidence:>8.3f}"
              f"{sig.volatility:>8.2f}{w:>9.2%}   {sig.rationale[:60]}")
    print(f"\n  Gross exposure if taken: {sum(abs(v) for v in weights.values()):.2%}")
    return 0


def cmd_trade(args) -> int:
    """Run live decision cycles against the configured broker."""
    config = Config.load(args.config)
    broker = make_broker(config)

    is_live = config.live_enabled() and config.execution.broker == "alpaca"
    if is_live:
        print("\n  *** LIVE TRADING — REAL MONEY ***")
        if not args.yes:
            reply = input("  Type 'trade live' to continue: ").strip()
            if reply != "trade live":
                print("  Aborted.")
                return 1
    else:
        print(f"\n  Simulation mode ({type(broker).__name__}). No real orders.")

    # Live trading never forces a rejected model, whatever the flag says.
    model = _load_gated_model(config, force=False)
    agent = TradingAgent(config, broker, model=model)
    for result in agent.run(cycles=args.cycles):
        print(f"  {result}")

    account = broker.account()
    print(f"\n  Equity {account.equity:,.2f}  cash {account.cash:,.2f}  "
          f"positions {len(account.positions)}")
    for symbol, pos in account.positions.items():
        print(f"    {symbol:<8} {pos.quantity:>7d} @ {pos.avg_price:>9.2f}")
    return 0


def cmd_validate(args) -> int:
    """Check the config for settings that would be dangerous in live trading."""
    config = Config.load(args.config)
    warnings: list[str] = []
    r = config.risk

    if r.aggression > 3.0:
        warnings.append(f"aggression {r.aggression} is very high; sizing is bounded "
                        "by the caps but drawdowns will be severe")
    if r.max_gross_leverage > 2.0:
        warnings.append(f"max_gross_leverage {r.max_gross_leverage}x exceeds Reg-T "
                        "overnight margin for a retail cash account")
    if r.max_drawdown_halt > 0.35:
        warnings.append(f"max_drawdown_halt {r.max_drawdown_halt:.0%} is loose; "
                        "the kill switch may not fire before serious damage")
    if r.max_position_weight > 0.4:
        warnings.append(f"max_position_weight {r.max_position_weight:.0%} means a "
                        "single name can dominate the book")
    if r.min_signal_confidence < 0.5:
        warnings.append(f"min_signal_confidence {r.min_signal_confidence} is below a "
                        "coin flip; the agent will trade noise")
    if config.execution.slippage_bps < 1.0:
        warnings.append("slippage_bps under 1 will flatter the backtest")
    if config.execution.allow_live:
        warnings.append("execution.allow_live is TRUE — live trading is one env var away")

    print(f"\nConfig: {args.config or '(defaults)'}")
    print(f"  universe        {len(config.data.universe)} symbols")
    print(f"  aggression      {r.aggression}")
    print(f"  vol target      {r.target_annual_vol:.0%} annualized")
    print(f"  max leverage    {r.max_gross_leverage}x")
    print(f"  max position    {r.max_position_weight:.0%}")
    print(f"  max positions   {r.max_positions}")
    print(f"  drawdown halt   {r.max_drawdown_halt:.0%}")
    print(f"  daily loss halt {r.daily_loss_halt:.0%}")
    print(f"  live enabled    {config.live_enabled()}")

    if warnings:
        print(f"\n  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("\n  No warnings.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="taurus",
        description="Market research and trading agent. Simulation by default.")
    parser.add_argument("--config", default="config.yaml", help="path to config YAML")
    parser.add_argument("-v", "--verbose", action="store_true")

    # Repeated on every subcommand so `taurus research --config x.yaml` works
    # as well as `taurus --config x.yaml research`; requiring the flag before
    # the subcommand is a papercut nobody remembers.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("-v", "--verbose", action="store_true",
                        default=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("research", parents=[common], help="walk-forward validation and model training")
    p.add_argument("--end", default=None)
    p.add_argument("--importance", action="store_true", help="show feature importance")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("backtest", parents=[common], help="replay the strategy over history")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--rules-only", action="store_true", help="ignore the trained model")
    p.add_argument("--out", default=None, help="write the equity curve to CSV")
    p.add_argument("--force-model", action="store_true",
                   help="use the saved model even if it failed its gate (research only)")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("signals", parents=[common], help="print today's signals, trade nothing")
    p.add_argument("--end", default=None)
    p.add_argument("--force-model", action="store_true",
                   help="use the saved model even if it failed its gate")
    p.set_defaults(func=cmd_signals)

    p = sub.add_parser("trade", parents=[common], help="run decision cycles against the broker")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--yes", action="store_true", help="skip the live confirmation prompt")
    p.set_defaults(func=cmd_trade)

    p = sub.add_parser("validate", parents=[common], help="audit the config for dangerous settings")
    p.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        log.error("%s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
