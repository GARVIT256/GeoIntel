"""
utils/logging.py — Structured logging for GEO-INTEL.

Every module should call ``get_logger(__name__)`` at the top. Logging level
is controlled by the LOG_LEVEL env var (default: INFO).

No spatial operations; no CRS assumptions.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a logger for ``name`` configured with a consistent format.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : str, optional
        Override log level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        Defaults to ``LOG_LEVEL`` env var, then 'INFO'.

    Returns
    -------
    logging.Logger
    """
    _level_str = level or os.environ.get("LOG_LEVEL", "INFO")
    _level = getattr(logging, _level_str.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (e.g. in pytest); don't add duplicate handlers
        logger.setLevel(_level)
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    logger.setLevel(_level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
