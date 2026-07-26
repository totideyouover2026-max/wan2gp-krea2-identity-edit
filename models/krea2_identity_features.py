"""Local feature switches for unfinished plugin capabilities."""

from __future__ import annotations

import os


_REID_EXPERIMENTS_ENV = "KREA2_IDENTITY_ENABLE_REID_EXPERIMENTS"
_TRUE_VALUES = {"1", "true", "yes", "on"}
REID_EXPERIMENTS_DISABLED_MESSAGE = (
    "ReID experiments are disabled in this plugin build. "
    "Set KREA2_IDENTITY_ENABLE_REID_EXPERIMENTS=1 before starting WanGP "
    "to enable them in a private development workspace."
)


def reid_experiments_enabled() -> bool:
    """Return whether local developer ReID experiments are enabled."""
    return os.environ.get(_REID_EXPERIMENTS_ENV, "").strip().lower() in _TRUE_VALUES