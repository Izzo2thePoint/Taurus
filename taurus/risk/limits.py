"""Portfolio-level risk limits and the kill switch.

This is the part of the system that stays switched on regardless of how
aggressive the strategy is configured to be. `RiskManager.check` is called
before every trading decision; when it halts, the agent flattens the book and
stops opening positions for the rest of the session.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from ..config import RiskConfig

log = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Running risk telemetry for the current session."""
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    current_date: date | None = None
    halted: bool = False
    halt_reason: str = ""
    breaches: list[str] = field(default_factory=list)


@dataclass
class RiskDecision:
    """Outcome of a risk check."""
    allowed: bool
    halt: bool = False
    reason: str = ""
    scale: float = 1.0  # multiplier applied to intended position sizes


class RiskManager:
    """Enforces drawdown, daily-loss, and exposure limits."""

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.state = RiskState()

    def reset(self, starting_equity: float, when: date | None = None) -> None:
        """Start a fresh risk session at `starting_equity`.

        `when` anchors the daily-loss baseline. Passing it matters when an
        agent restarts part-way through a session: without an anchor the
        first check would re-baseline to current equity and the daily limit
        would not see a loss that had already happened.
        """
        self.state = RiskState(
            peak_equity=starting_equity,
            day_start_equity=starting_equity,
            current_date=when,
        )

    def drawdown(self, equity: float) -> float:
        """Current peak-to-trough drawdown as a positive fraction."""
        if self.state.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - equity / self.state.peak_equity)

    def daily_loss(self, equity: float) -> float:
        """Loss since the session's open, as a positive fraction."""
        if self.state.day_start_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - equity / self.state.day_start_equity)

    def on_new_day(self, when: date, equity: float) -> None:
        """Roll the daily loss baseline. Does not clear a drawdown halt.

        A daily-loss halt expires overnight — it exists to stop a bad day
        compounding. A max-drawdown halt does not: that one says the strategy
        itself is broken and wants a human to look at it.
        """
        self.state.current_date = when
        self.state.day_start_equity = equity
        if self.state.halted and self.state.halt_reason.startswith("daily_loss"):
            log.info("daily loss halt cleared for new session %s", when)
            self.state.halted = False
            self.state.halt_reason = ""

    def check(self, equity: float, when: date | None = None) -> RiskDecision:
        """Evaluate every limit against current equity."""
        st, cfg = self.state, self.config

        if when is not None:
            if st.current_date is None:
                # First dated check of this session. The baseline set by
                # reset() stands — do not overwrite it with current equity,
                # or any loss already taken today becomes invisible.
                st.current_date = when
            elif when > st.current_date:
                self.on_new_day(when, equity)

        st.peak_equity = max(st.peak_equity, equity)

        if st.halted:
            return RiskDecision(allowed=False, halt=True, reason=st.halt_reason)

        dd = self.drawdown(equity)
        if dd >= cfg.max_drawdown_halt:
            reason = f"max_drawdown {dd:.1%} >= {cfg.max_drawdown_halt:.1%}"
            self._halt(reason)
            return RiskDecision(allowed=False, halt=True, reason=reason)

        dl = self.daily_loss(equity)
        if dl >= cfg.daily_loss_halt:
            reason = f"daily_loss {dl:.1%} >= {cfg.daily_loss_halt:.1%}"
            self._halt(reason)
            return RiskDecision(allowed=False, halt=True, reason=reason)

        # De-risk progressively as drawdown approaches the halt, instead of
        # trading full size right up to the cliff edge and then stopping dead.
        scale = 1.0
        soft_floor = cfg.max_drawdown_halt * 0.5
        if dd > soft_floor:
            span = cfg.max_drawdown_halt - soft_floor
            scale = float(max(0.25, 1.0 - (dd - soft_floor) / span))
            log.info("drawdown %.1f%% -> sizing scaled to %.2f", dd * 100, scale)

        return RiskDecision(allowed=True, scale=scale)

    def _halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.breaches.append(reason)
        log.error("RISK HALT: %s", reason)

    def force_halt(self, reason: str) -> None:
        """Manual kill switch."""
        self._halt(f"manual: {reason}")
