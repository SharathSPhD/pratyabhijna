"""Per-item free-energy budget ledger.

The cascade pays/earns *bits of free energy* across the cit-iccha-jnana-vimarsa
sequence. The ledger has three signed contributors:

1. ``earn_jnana(delta_F)`` -- jñāna BMR ΔF (bits of evidence the surface
   actually fits the constraint posterior). Positive ΔF earns budget.
2. ``earn_aspect(distance)`` -- the embedding cosine *distance* (= 1 - cosine)
   between the committed surface and the aspect prior. Lower distance =>
   smaller cost; higher distance is a cost the cascade pays.
3. ``earn_tokens(n_tokens)`` -- a flat per-token cost of
   ``self.token_cost_bits`` bits/token. Always negative.

When ``balance() <= abort_threshold`` the cascade is told to skip the
revision pass via :meth:`should_continue_revision`. The default
``abort_threshold = -2.0`` corresponds to "the surface costs ~3 bits more
than the BMR earned, and the second pass would only widen the gap unless
the brief is tight". The threshold is conservative because revision passes
in v0.2 always helped the prove-gate items.

The ledger is per-prompt: the cascade creates one
:class:`FreeEnergyBudget` at the top of ``run_cascade`` and discards it at
the bottom. There is *no* cross-prompt budget bleed; the per-domain
storehouse is the only stateful component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


class BudgetExceededError(RuntimeError):
    """Raised when a caller tries to keep going past ``abort_threshold``.

    The cascade catches this and either commits the draft (if event-gated
    skipped revision because of cost) or surfaces it to the audit log.
    """


@dataclass(frozen=True)
class BudgetLedgerEntry:
    """One signed entry in the per-item budget ledger."""

    source: Literal["jnana", "aspect", "tokens", "manual"]
    bits: float
    note: str = ""


@dataclass
class FreeEnergyBudget:
    """Per-item free-energy ledger.

    Default constants are calibrated against the v0.2 audit (mean draft pass:
    ~110 tokens; mean ΔF: ~0.2 bits; mean aspect cosine: ~0.45). The
    revision pass is allowed when there is at least ~2 bits of headroom.
    """

    initial_bits: float = 4.0
    abort_threshold: float = -2.0
    token_cost_bits: float = 0.01  # ~1 bit per 100 tokens
    aspect_cost_scale: float = 2.0
    jnana_earn_scale: float = 1.0
    entries: list[BudgetLedgerEntry] = field(default_factory=list)

    def balance(self) -> float:
        """Total bits remaining (initial + sum of signed entries)."""
        return float(self.initial_bits + sum(e.bits for e in self.entries))

    def earn_jnana(self, delta_F: float, *, note: str = "") -> float:
        """Add (signed) jñāna ΔF to the ledger; clipped to a safe range."""
        bits = float(self.jnana_earn_scale * float(delta_F))
        bits = max(-10.0, min(10.0, bits))  # clip to avoid -inf cascading
        self.entries.append(BudgetLedgerEntry(source="jnana", bits=bits, note=note))
        return self.balance()

    def earn_aspect(self, distance: float, *, note: str = "") -> float:
        """Pay aspect cost: -aspect_cost_scale * distance bits."""
        d = max(0.0, float(distance))
        bits = -float(self.aspect_cost_scale * d)
        self.entries.append(BudgetLedgerEntry(source="aspect", bits=bits, note=note))
        return self.balance()

    def earn_tokens(self, n_tokens: int, *, note: str = "") -> float:
        """Pay flat per-token cost: -token_cost_bits * n_tokens bits."""
        n = max(0, int(n_tokens))
        bits = -float(self.token_cost_bits * n)
        self.entries.append(BudgetLedgerEntry(source="tokens", bits=bits, note=note))
        return self.balance()

    def add_manual(self, bits: float, *, note: str = "") -> float:
        """Add a caller-supplied signed adjustment (used by tests/debug)."""
        self.entries.append(BudgetLedgerEntry(source="manual", bits=float(bits), note=note))
        return self.balance()

    def should_continue_revision(self) -> bool:
        """True iff the ledger still has headroom for a revision pass."""
        return self.balance() > self.abort_threshold

    def to_audit(self) -> dict[str, object]:
        """Render the ledger for inclusion in :class:`pce.types.CascadeState.audit`."""
        return {
            "initial_bits": float(self.initial_bits),
            "abort_threshold": float(self.abort_threshold),
            "token_cost_bits": float(self.token_cost_bits),
            "aspect_cost_scale": float(self.aspect_cost_scale),
            "jnana_earn_scale": float(self.jnana_earn_scale),
            "balance_bits": float(self.balance()),
            "entries": [
                {"source": str(e.source), "bits": float(e.bits), "note": str(e.note)}
                for e in self.entries
            ],
        }
