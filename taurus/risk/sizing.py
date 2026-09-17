"""Position sizing.

Aggression is expressed here, and only here. Three ideas compose:

  1. Volatility targeting — size inversely to the name's volatility, so a
     6%-a-day stock and a 1%-a-day ETF contribute comparable risk.
  2. Fractional Kelly — scale by the model's edge, so a 0.75 signal gets more
     capital than a 0.56 one.
  3. The aggression multiplier — a single knob that scales the whole book,
     bounded by the hard caps in RiskConfig.

The caps are not suggestions: `max_position_weight` and `max_gross_leverage`
are applied last and always.
"""
from __future__ import annotations

import numpy as np

from ..config import RiskConfig


def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Kelly stake for a bet with the given edge and payoff ratio.

    f* = p - (1-p)/b. Negative means no edge — the correct size is zero, not
    a short, because the payoff ratio is asymmetric by construction.
    """
    if win_loss_ratio <= 0:
        return 0.0
    f = win_prob - (1.0 - win_prob) / win_loss_ratio
    return max(0.0, float(f))


def volatility_scalar(asset_vol: float, target_vol: float,
                      max_scalar: float = 3.0) -> float:
    """Leverage that brings an asset's vol to the portfolio target."""
    if not np.isfinite(asset_vol) or asset_vol <= 1e-6:
        return 0.0
    return float(min(target_vol / asset_vol, max_scalar))


def position_weight(confidence: float, asset_vol: float, config: RiskConfig,
                    win_loss_ratio: float = 2.0) -> float:
    """Target weight for one position, as a fraction of equity.

    Returns 0 when confidence is below the entry threshold — the strategy sits
    out rather than taking a marginal bet, which is what keeps an aggressive
    book from being merely a permanently maxed-out one.
    """
    if not np.isfinite(confidence) or confidence < config.min_signal_confidence:
        return 0.0

    kelly = kelly_fraction(confidence, win_loss_ratio) * config.kelly_fraction
    vol_scalar = volatility_scalar(asset_vol, config.target_annual_vol)
    raw = kelly * vol_scalar * config.aggression
    return float(np.clip(raw, 0.0, config.max_position_weight))


def apply_portfolio_caps(weights: dict[str, float], config: RiskConfig,
                         incumbents: set[str] | None = None,
                         incumbent_bonus: float = 0.15) -> dict[str, float]:
    """Enforce concentration and leverage limits across the whole book.

    Keeps the highest-conviction `max_positions` names, then scales every
    weight down proportionally if gross exposure exceeds the leverage cap.
    Scaling proportionally (rather than truncating the tail) preserves the
    relative conviction the model expressed.

    Positions already held get a ranking bonus so a name is not sold on
    Tuesday and bought back on Wednesday because something else edged past it
    by a hair. The bonus affects ranking only, never the final weight.
    """
    if not weights:
        return {}

    incumbents = incumbents or set()

    def rank_key(item: tuple[str, float]) -> float:
        symbol, weight = item
        score = abs(weight)
        return score * (1.0 + incumbent_bonus) if symbol in incumbents else score

    ranked = sorted(weights.items(), key=rank_key, reverse=True)
    kept = {k: v for k, v in ranked[: config.max_positions] if abs(v) > 1e-6}
    if not kept:
        return {}

    capped = {k: float(np.clip(v, -config.max_position_weight, config.max_position_weight))
              for k, v in kept.items()}

    gross = sum(abs(v) for v in capped.values())
    if gross > config.max_gross_leverage:
        scale = config.max_gross_leverage / gross
        capped = {k: v * scale for k, v in capped.items()}
    return capped


def stop_price(entry: float, atr: float, config: RiskConfig,
               direction: int = 1) -> float:
    """Initial stop, `per_trade_stop_atr` ATR away from entry."""
    distance = config.per_trade_stop_atr * atr
    return float(entry - distance if direction > 0 else entry + distance)


def trailing_stop_price(peak: float, atr: float, config: RiskConfig,
                        direction: int = 1) -> float:
    """Trailing stop referenced to the best price reached since entry.

    A wider band than the initial stop, so a winner is given room to breathe
    instead of being shaken out by ordinary noise.
    """
    distance = config.trailing_stop_atr * atr
    return float(peak - distance if direction > 0 else peak + distance)


def shares_for_weight(weight: float, equity: float, price: float) -> int:
    """Convert a target weight into whole shares."""
    if price <= 0 or not np.isfinite(price):
        return 0
    return int(weight * equity / price)
