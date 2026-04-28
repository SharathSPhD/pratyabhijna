"""Sleep / consolidation routines for the Hopfield storehouse."""
from pce.consolidation.sleep import (
    is_consolidated,
    run_rem,
    run_sleep_cycle,
    run_sws,
)

__all__ = ["is_consolidated", "run_rem", "run_sleep_cycle", "run_sws"]
