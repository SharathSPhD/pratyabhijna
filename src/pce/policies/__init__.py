"""v0.4 commit-policy layer (ADR-002).

Re-exports the public API: :class:`CommitPolicy`, :class:`PolicyFeatures`,
and the five concrete policies (``AlwaysDraft``, ``AlwaysRevise``,
``EventGated``, ``LearnedGate``, ``OracleCommit``).
"""
from __future__ import annotations

from pce.policies.commit import (
    AlwaysDraft,
    AlwaysRevise,
    CommitPolicy,
    EventGated,
    LearnedGate,
    OracleCommit,
    PolicyFeatures,
    extract_features_from_audit,
    policy_for_name,
)

__all__ = [
    "AlwaysDraft",
    "AlwaysRevise",
    "CommitPolicy",
    "EventGated",
    "LearnedGate",
    "OracleCommit",
    "PolicyFeatures",
    "extract_features_from_audit",
    "policy_for_name",
]
