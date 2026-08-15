"""Shared setup. Import this first in any script or notebook."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
load_dotenv(ROOT / ".env")


def require(*keys: str) -> None:
    """Fail fast with a useful message instead of a confusing 401 later."""
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise SystemExit(
            f"Missing env var(s): {', '.join(missing)}\n"
            f"Copy .env.example to .env and fill them in."
        )


# Change this once, everything downstream follows.
# Format is "provider:model" in LangChain 1.x.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai:gpt-4o-mini")
