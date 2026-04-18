"""Environment utilities for the muse package.

Loads environment variables from a ``.env`` file (if present) and provides
the Materials Project API key.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

MP_API_KEY: str | None = os.getenv("MP_API_KEY")

if MP_API_KEY is None:
    logger.warning(
        "MP_API_KEY environment variable is not set. "
        "Materials Project API queries will fail. "
        "Set it in your environment or in a .env file."
    )
