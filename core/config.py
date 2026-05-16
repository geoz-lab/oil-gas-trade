"""Global configuration — loaded once at startup."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# ── Anthropic ──────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = "claude-sonnet-4-6"

# ── Optional data-source keys ──────────────────────────────────────────────
ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "")
EIA_API_KEY: str = os.getenv("EIA_API_KEY", "")
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")

# ── Forecast defaults ──────────────────────────────────────────────────────
DEFAULT_HORIZON_WEEKS: int = 8
HISTORY_DAYS: int = 365 * 2          # 2 years of history for baseline models
CACHE_DIR: Path = ROOT / "data" / "cache"
REPORTS_DIR: Path = ROOT / "reports"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
