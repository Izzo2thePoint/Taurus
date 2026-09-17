"""Append-only decision journal.

Every decision the agent makes is written here as one JSON object per line:
what it saw, what it chose, and why. This is what makes the system auditable
after a bad week — without it, an autonomous trader is a black box that lost
money for unknowable reasons.

The journal is also the agent's memory: `recent` and `performance_summary`
read it back so past decisions can inform current ones.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, date
from typing import Any, Iterator

log = logging.getLogger(__name__)


class Journal:
    """Newline-delimited JSON log of agent activity."""

    def __init__(self, path: str = "runs/journal.jsonl"):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def write(self, kind: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
            "kind": kind,
            **payload,
        }
        try:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            # A journal failure must never take the trading loop down with it.
            log.error("journal write failed: %s", exc)

    # --- typed helpers -----------------------------------------------------

    def log_decision(self, when: date, symbol: str, action: str, quantity: int,
                     price: float, confidence: float, rationale: str,
                     **extra: Any) -> None:
        self.write("decision", {
            "date": str(when), "symbol": symbol, "action": action,
            "quantity": quantity, "price": round(price, 4),
            "confidence": round(confidence, 4), "rationale": rationale, **extra,
        })

    def log_equity(self, when: date, equity: float, cash: float,
                   n_positions: int, gross_exposure: float,
                   drawdown: float) -> None:
        self.write("equity", {
            "date": str(when), "equity": round(equity, 2), "cash": round(cash, 2),
            "n_positions": n_positions,
            "gross_exposure": round(gross_exposure, 4),
            "drawdown": round(drawdown, 4),
        })

    def log_research(self, when: date, metrics: dict[str, Any],
                     top_features: dict[str, float] | None = None) -> None:
        self.write("research", {"date": str(when), "metrics": metrics,
                                "top_features": top_features or {}})

    def log_risk(self, when: date, event: str, detail: str) -> None:
        self.write("risk", {"date": str(when), "event": event, "detail": detail})

    # --- reading back ------------------------------------------------------

    def read(self) -> Iterator[dict[str, Any]]:
        if not os.path.exists(self.path):
            return iter(())
        def _gen() -> Iterator[dict[str, Any]]:
            with open(self.path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # A torn final line (killed mid-write) should not make
                        # the whole history unreadable.
                        log.warning("skipping malformed journal line")
        return _gen()

    def recent(self, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        records = [r for r in self.read() if kind is None or r.get("kind") == kind]
        return records[-limit:]

    def performance_summary(self) -> dict[str, Any]:
        """Condense the journal into the agent's view of how it has been doing."""
        equity = [r for r in self.read() if r.get("kind") == "equity"]
        if not equity:
            return {}
        first, last = equity[0], equity[-1]
        peak = max(r["equity"] for r in equity)
        decisions = sum(1 for r in self.read() if r.get("kind") == "decision")
        return {
            "start_equity": first["equity"],
            "current_equity": last["equity"],
            "total_return": last["equity"] / first["equity"] - 1.0 if first["equity"] else 0.0,
            "peak_equity": peak,
            "current_drawdown": 1.0 - last["equity"] / peak if peak else 0.0,
            "observations": len(equity),
            "decisions": decisions,
        }
