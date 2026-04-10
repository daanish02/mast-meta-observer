from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEEPAGENTS_SRC = ROOT.parent / "deepagents" / "libs" / "deepagents"


def main() -> int:
    """Run the observer vs baseline benchmark CLI."""
    from dotenv import load_dotenv

    if DEEPAGENTS_SRC.exists():
        sys.path.insert(0, str(DEEPAGENTS_SRC))
    sys.path.insert(0, str(ROOT / "src"))
    load_dotenv(ROOT / ".env")
    from mast_meta_observer.runner import run_benchmark_cli

    return asyncio.run(run_benchmark_cli())


if __name__ == "__main__":
    raise SystemExit(main())
