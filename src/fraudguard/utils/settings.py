"""Runtime settings loaded from environment variables / .env file.

Use this for SECRETS and ENVIRONMENT-SPECIFIC values (DB URIs, API keys,
ports). For static project constants, use config.py instead.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from fraudguard.utils.config import PROJECT_ROOT
from fraudguard.utils.exceptions import ConfigError


# load_dotenv() reads `.env` and adds entries to os.environ. It does NOT
# overwrite variables already set in the real environment.
load_dotenv(PROJECT_ROOT / ".env")


def _require(key: str) -> str:
    """Fetch a required env var or raise ConfigError if missing."""
    value = os.environ.get(key)
    if value is None or value == "":
        raise ConfigError(f"Required environment variable not set: {key}")
    return value


def _optional(key: str, default: str) -> str:
    """Fetch an optional env var, falling back to `default`."""
    return os.environ.get(key, default)


def _bool_env(key: str, default: bool = False) -> bool:
    """Parse a string env var as a boolean.

    Accepts 'true', '1', 'yes' (case-insensitive) as True.
    """
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in ("true", "1", "yes")


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment."""

    mongo_uri: str
    jwt_secret: str
    api_port: int
    debug: bool

    @classmethod
    def load(cls) -> "Settings":
        """Construct Settings from current env / .env contents."""
        return cls(
            mongo_uri=_optional("MONGO_URI", "mongodb://localhost:27017"),
            jwt_secret=_require("JWT_SECRET"),
            api_port=int(_optional("API_PORT", "8000")),
            debug=_bool_env("DEBUG", default=False),
        )
