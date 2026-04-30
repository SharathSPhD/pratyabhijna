"""Allow ``python -m pce`` to invoke the standalone CLI."""
from __future__ import annotations

from pce.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
