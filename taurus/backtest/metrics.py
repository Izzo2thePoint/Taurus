"""Performance and risk statistics for an equity curve.

An aggressive strategy is judged on risk-adjusted return, not total return:
doubling the money with an 80% drawdown along the way is not a strategy
anyone can actually hold. Max drawdown and Calmar carry as much weight here
as CAGR.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceReport:
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_trade_return: float
    best_day: float
    worst_day: float
    exposure: float
    final_equity: float

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"  Total return    {self.total_return:>9.2%}\n"
            f"  CAGR            {self.cagr:>9.2%}\n"
            f"  Volatility      {self.annual_volatility:>9.2%}\n"
            f"  Sharpe          {self.sharpe:>9.2f}\n"
            f"  Sortino         {self.sortino:>9.2f}\n"
            f"  Max drawdown    {self.max_drawdown:>9.2%}\n"
            f"  Calmar          {self.calmar:>9.2f}\n"
            f"  Win rate        {self.win_rate:>9.2%}\n"
            f"  Profit factor   {self.profit_factor:>9.2f}\n"
            f"  Trades          {self.n_trades:>9d}\n"
            f"  Avg trade       {self.avg_trade_return:>9.2%}\n"
            f"  Best / worst day{self.best_day:>8.2%} / {self.worst_day:.2%}\n"
            f"  Time in market  {self.exposure:>9.2%}\n"
            f"  Final equity    {self.final_equity:>9,.0f}"
        )


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline, as a positive fraction."""
    if equity.empty:
        return 0.0
    return float((1.0 - equity / equity.cummax()).max())


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe. Returns 0 for a constant curve rather than inf."""
    if returns.empty:
        return 0.0
    excess = returns - risk_free / TRADING_DAYS
    sd = excess.std()
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Like Sharpe but penalizes only downside deviation."""
    if returns.empty:
        return 0.0
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if downside.empty:
        return 0.0
    dd = downside.std()
    if not np.isfinite(dd) or dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate implied by the curve's length."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / TRADING_DAYS
    if years <= 0:
        return 0.0
    ratio = equity.iloc[-1] / equity.iloc[0]
    if ratio <= 0:
        return -1.0
    return float(ratio ** (1 / years) - 1.0)


def trade_stats(trade_returns: list[float]) -> tuple[float, float, float]:
    """Return (win_rate, profit_factor, average trade return)."""
    if not trade_returns:
        return 0.0, 0.0, 0.0
    arr = np.asarray(trade_returns, dtype=float)
    wins, losses = arr[arr > 0], arr[arr < 0]
    win_rate = float(len(wins) / len(arr))
    gross_loss = float(-losses.sum())
    # No losing trades makes profit factor undefined; report it as inf only
    # when there were actual wins to divide.
    profit_factor = (float(wins.sum() / gross_loss) if gross_loss > 0
                     else (float("inf") if len(wins) else 0.0))
    return win_rate, profit_factor, float(arr.mean())


def build_report(equity: pd.Series, trade_returns: list[float] | None = None,
                 exposure: float = 0.0) -> PerformanceReport:
    """Assemble the full scorecard from an equity curve."""
    equity = equity.dropna()
    if equity.empty:
        # Named explicitly: positional construction silently breaks the moment
        # a field is added to the report.
        return PerformanceReport(
            total_return=0.0, cagr=0.0, annual_volatility=0.0, sharpe=0.0,
            sortino=0.0, max_drawdown=0.0, calmar=0.0, win_rate=0.0,
            profit_factor=0.0, n_trades=0, avg_trade_return=0.0,
            best_day=0.0, worst_day=0.0, exposure=exposure, final_equity=0.0,
        )

    returns = equity.pct_change().dropna()
    mdd = max_drawdown(equity)
    growth = cagr(equity)
    win_rate, profit_factor, avg_trade = trade_stats(trade_returns or [])

    return PerformanceReport(
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        cagr=growth,
        annual_volatility=float(returns.std() * np.sqrt(TRADING_DAYS)) if not returns.empty else 0.0,
        sharpe=sharpe_ratio(returns),
        sortino=sortino_ratio(returns),
        max_drawdown=mdd,
        calmar=float(growth / mdd) if mdd > 1e-9 else 0.0,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trades=len(trade_returns or []),
        avg_trade_return=avg_trade,
        best_day=float(returns.max()) if not returns.empty else 0.0,
        worst_day=float(returns.min()) if not returns.empty else 0.0,
        exposure=exposure,
        final_equity=float(equity.iloc[-1]),
    )
