"""
Data-loading helpers.

The *real* implementation lives in ``src.main.build_stream``.  This stub just
re-exports that symbol so that external callers (e.g. documentation or unit
tests) can perform a cheap import without pulling in heavyweight data
processing dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
#            TRY TO FORWARD TO THE FULL IMPLEMENTATION IF AVAILABLE
# ---------------------------------------------------------------------------

try:
    _build_stream = getattr(importlib.import_module("src.main"), "build_stream")

    def build_stream(name: str, conf: Dict[str, Any], seed: int):  # noqa: D401
        """Re-export of the real ``build_stream`` implementation (if present)."""

        return _build_stream(name, conf, seed)

except Exception as ex:  # pragma: no cover – minimal environment fallback

    def build_stream(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]):  # type: ignore[return-value]
        """Placeholder that reminds the user about the missing heavy stack."""

        raise RuntimeError(
            "build_stream is unavailable in this minimal environment. "
            "Install the full data-processing dependencies to use it."
        ) from ex

__all__ = ["build_stream"]
