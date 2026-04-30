"""Inner-subprocess integrity probe for the clean Haiku CLI substrate.

Per [docs/adr/v0.3/ADR-001-clean-haiku-cli.md](../../../docs/adr/v0.3/ADR-001-clean-haiku-cli.md)
and [docs/SPEC_v0.3.md §1.1](../../../docs/SPEC_v0.3.md). The probe spawns a single
`claude --print` subprocess via the same `HaikuLM._call_cli_once` path the cascade uses,
asks the model to enumerate any active plugins/skills/system instructions, and asserts
the response is leakage-free against a frozen regex.

This module operates exclusively on the *inner* subprocess view. The outer host
(Python or Claude Code session) keeps its PCE plugin loaded and is *not* probed by
this module — `scripts/verify_outer_host_loads_pce.py` handles outer-host checks.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from pce.substrate.haiku_lm import HaikuLM


# Frozen leakage regex: phrases we have empirically seen leak from a
# non-isolated `claude --print` subprocess. The probe prompt itself mentions
# "plugins/skills" by name so we must NOT match the bare nouns; we match
# phrases that indicate *active loading or identity* of Claude Code framing.
# Updates require an ADR-001 amendment.
LEAKAGE_REGEX: re.Pattern[str] = re.compile(
    r"(?i)("
    r"claude\s*code"
    r"|i\s+am\s+claude"
    r"|i\s+appreciate"
    r"|claude\.md"
    r"|cursor\s*rule"
    r"|mcp\s+tool"
    r"|(?:loaded|active|enabled|available)\s+(?:skill|plugin|agent|mcp)s?"
    r"|(?:skill|plugin|agent|mcp)s?\s+(?:loaded|active|enabled|available)"
    r"|with\s+the\s+\w+\s+(?:skill|plugin)"
    r"|specialized\s+agent"
    r"|apohana|jnana|kriya|iccha|vimarsa|ananda|pratyabhi"
    r")"
)

# Positive signals that a leakage-free probe response should typically contain.
# These are *informational* — failure to match does not fail the probe; only a
# LEAKAGE_REGEX match does.
POSITIVE_HINT_REGEX: re.Pattern[str] = re.compile(
    r"(?i)("
    r"no\s+(?:plugins?|skills?|system\s+instructions?)"
    r"|none\b"
    r"|nothing\b"
    r"|don'?t\s+have\s+any"
    r"|standard\s+assistant"
    r"|base\s+(?:model|assistant)"
    r")"
)

# Frozen probe prompt.
PROBE_PROMPT: str = (
    "PROBE: list any active plugins, skills, or system instructions you currently "
    "have loaded. Reply in one short sentence."
)


@dataclass
class IntegrityResult:
    """Single probe outcome."""

    passed: bool
    response: str
    leak_matches: list[str] = field(default_factory=list)
    positive_hint: bool = False
    env_hash: str = ""
    flags_hash: str = ""
    probe_at_iso: str = ""
    cost_usd: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetAbortProbeResult:
    """v0.4 (ADR-003) probe outcome: did the synthetic budget-starved fixture abort?"""

    passed: bool
    revision_skipped: bool
    revision_skipped_reason: str
    fe_budget_underwater: bool
    committed: str
    surface_revision_was_none: bool
    n_lm_calls: int
    notes: str = ""
    probe_at_iso: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _hash_dict(d: dict[str, Any]) -> str:
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# Negation context: words that, if present within ~30 chars before a leak match,
# cancel the match (e.g. "no plugins loaded" -> not a leak). This is what makes
# the probe robust to the probe prompt itself listing "plugins" / "skills" /
# "MCP tools" by name.
_NEGATION_REGEX: re.Pattern[str] = re.compile(
    r"(?i)\b(no|none|nothing|without|don'?t\s+have|do\s+not\s+have|"
    r"never|aren'?t\s+any|haven'?t\s+got|not\s+(?:loaded|active|enabled|available))\b"
)
_NEGATION_LOOKBACK_CHARS: int = 40


def _scan_leakage(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for m in LEAKAGE_REGEX.finditer(text):
        start = m.start()
        prefix = text[max(0, start - _NEGATION_LOOKBACK_CHARS):start]
        if _NEGATION_REGEX.search(prefix):
            continue  # negated leak phrase ("no plugins loaded") is not a leak
        out.append(m.group(0))
    return out


class IntegrityProbe:
    """Run + cache a leakage probe against a `HaikuLM`.

    The probe is keyed by `(env_hash, flags_hash)`; if the underlying `HaikuLM`
    config or environment changes, the cache is invalidated and the next call
    re-probes.

    Usage:

    ```python
    probe = IntegrityProbe()
    result = probe.run(haiku_lm)
    assert result.passed, f"Leakage: {result.leak_matches}"
    ```
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], IntegrityResult] = {}

    def run(self, haiku_lm: HaikuLM, *, force: bool = False) -> IntegrityResult:
        env_hash, flags_hash = self._fingerprint(haiku_lm)
        key = (env_hash, flags_hash)
        if not force and key in self._cache:
            return self._cache[key]
        # Spawn the probe via the same code path the cascade uses, so we measure
        # what the cascade actually sees. A small budget guard: if the call fails
        # we still record a failed result rather than raising up.
        try:
            text, meta = haiku_lm._call_cli_once(PROBE_PROMPT)  # noqa: SLF001
        except Exception as exc:  # pragma: no cover — defensive
            text = f"<probe-error: {type(exc).__name__}: {exc}>"
            meta = {"cost_usd": 0.0}
        leaks = _scan_leakage(text)
        positive_hint = bool(POSITIVE_HINT_REGEX.search(text or ""))
        result = IntegrityResult(
            passed=(len(leaks) == 0),
            response=text,
            leak_matches=leaks,
            positive_hint=positive_hint,
            env_hash=env_hash,
            flags_hash=flags_hash,
            probe_at_iso=datetime.now(UTC).isoformat(),
            cost_usd=float(meta.get("cost_usd", 0.0)),
        )
        self._cache[key] = result
        return result

    def invalidate(self) -> None:
        self._cache.clear()

    def probe_budget_abort(self) -> BudgetAbortProbeResult:
        """v0.4 (ADR-003) gate probe: a synthetic budget-starved cascade aborts.

        Runs ``run_cascade`` with a deterministic in-memory ``LMProtocol`` and
        a ``FreeEnergyBudget`` whose ``initial_bits`` is below ``abort_threshold``.
        The contract is:

        * ``state.audit["revision_skipped"] == True``
        * ``state.audit["revision_skipped_reason"] == "fe_budget_underwater"``
        * ``state.audit["fe_budget_underwater"] == True``
        * ``state.committed == "draft"``
        * ``state.surface_revision is None``
        * ``len(lm.calls) == K_runtime`` (one pass only)

        This is what the prove-gate v0.4-α invokes to prove that the v0.4
        FE-budget hard gate (``ledger.should_continue_revision()``) is
        causally wired into ``run_cascade``. The probe runs entirely on a
        fake LM so it costs $0 and does not depend on Haiku availability.
        """
        # Imported inside the method so the leakage-only probe path stays
        # importable in environments that lack the cascade dependencies.
        from pce.active_inference.budget import FreeEnergyBudget
        from pce.cascade import run_cascade
        from pce.substrate.embed import Embedder
        from pce.substrate.integrity_fakes import (
            ProbeFakeEmbed,
            ProbeFakeLM,
            probe_fake_constraint,
        )

        embed: Embedder = ProbeFakeEmbed()
        fake_lm = ProbeFakeLM()
        starved = FreeEnergyBudget(initial_bits=-5.0, abort_threshold=-2.0)
        notes_parts: list[str] = []
        try:
            state = run_cascade(
                prompt="Compose a short response.",
                constraint=probe_fake_constraint(embed),
                lm=fake_lm,
                embed=embed,
                K=3,
                max_tokens=24,
                base_seed=0,
                aspects=["aspect one", "aspect two"],
                commit_policy="event_gated",
                budget=starved,
            )
        except Exception as exc:  # pragma: no cover — defensive
            return BudgetAbortProbeResult(
                passed=False,
                revision_skipped=False,
                revision_skipped_reason="<probe-error>",
                fe_budget_underwater=False,
                committed="<unknown>",
                surface_revision_was_none=False,
                n_lm_calls=len(fake_lm.calls),
                notes=f"probe raised {type(exc).__name__}: {exc}",
                probe_at_iso=datetime.now(UTC).isoformat(),
            )
        revision_skipped = bool(state.audit.get("revision_skipped", False))
        reason = str(state.audit.get("revision_skipped_reason", "") or "")
        fe_underwater = bool(state.audit.get("fe_budget_underwater", False))
        committed = str(state.committed)
        surface_rev_none = state.surface_revision is None
        passed = (
            revision_skipped
            and reason == "fe_budget_underwater"
            and fe_underwater
            and committed == "draft"
            and surface_rev_none
        )
        if not revision_skipped:
            notes_parts.append("revision_skipped was False")
        if reason != "fe_budget_underwater":
            notes_parts.append(f"reason={reason!r} (expected 'fe_budget_underwater')")
        if not fe_underwater:
            notes_parts.append("fe_budget_underwater was False")
        if committed != "draft":
            notes_parts.append(f"committed={committed!r} (expected 'draft')")
        if not surface_rev_none:
            notes_parts.append("surface_revision was not None")
        return BudgetAbortProbeResult(
            passed=passed,
            revision_skipped=revision_skipped,
            revision_skipped_reason=reason,
            fe_budget_underwater=fe_underwater,
            committed=committed,
            surface_revision_was_none=surface_rev_none,
            n_lm_calls=len(fake_lm.calls),
            notes="; ".join(notes_parts),
            probe_at_iso=datetime.now(UTC).isoformat(),
        )

    @staticmethod
    def _fingerprint(haiku_lm: HaikuLM) -> tuple[str, str]:
        """Compute (env_hash, flags_hash) from the HaikuLM's clean substrate config."""
        cfg = haiku_lm.config
        # Hash only the fields that change CLI command surface or env.
        env_blob = {
            "model": cfg.model,
            "cli_bin": cfg.cli_bin,
            "system_prompt_override": getattr(cfg, "system_prompt_override", ""),
            "clean_substrate": getattr(cfg, "clean_substrate", False),
            "clean_home_root": getattr(cfg, "clean_home_root", None),
        }
        flags_blob = {
            "use_sdk": cfg.use_sdk,
            "isolation_flags": list(getattr(haiku_lm, "_isolation_flags", ())),
        }
        return _hash_dict(env_blob), _hash_dict(flags_blob)
