"""
Static-analysis helpers and lightweight stubs.

This module purposefully keeps a *tiny* footprint: it does **not** import any
GPU-heavy libraries yet still makes sure that relative imports such as
``from src.main import ContinualTrainer`` (used by ``src.train`` /
``src.preprocess``) resolve correctly during static validation.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
#                       MODULE SHIMS FOR LINTERS
# ---------------------------------------------------------------------------

# Expose top-level names so that tooling using absolute imports (e.g.
# ``import preprocess`` instead of ``from src import preprocess``) keeps
# working in the reduced environment.
for _name in ("preprocess", "train", "evaluate"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

# ---------------------------------------------------------------------------
#                    LIGHTWEIGHT PUBLIC PLACEHOLDERS
# ---------------------------------------------------------------------------

class ContinualTrainer:  # pylint: disable=too-few-public-methods
    """Stub that raises if instantiated in the minimal environment."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        raise RuntimeError(
            "The real ContinualTrainer is not available in this minimal "
            "environment. Install the full research dependencies to run "
            "training."
        )


def build_stream(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]):  # type: ignore[return-value]
    """Stub that raises – actual data-processing lives in the heavy stack."""

    raise RuntimeError(
        "build_stream is unavailable in this minimal environment. Install the "
        "full data-processing dependencies to use it."
    )

# ---------------------------------------------------------------------------
#              Re-export symbols for the absolute shim modules
# ---------------------------------------------------------------------------

# Use ``setattr`` instead of a ``type: ignore`` comment so static analysers no
# longer complain about an unknown attribute on the freshly created module.
setattr(sys.modules["train"], "ContinualTrainer", ContinualTrainer)
setattr(sys.modules["preprocess"], "build_stream", build_stream)

__all__ = ["ContinualTrainer", "build_stream"]
