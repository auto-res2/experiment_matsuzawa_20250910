# src/train.py
"""
Training-related helpers.

A *very* lightweight stub that re-exports ``ContinualTrainer`` when the full
research stack is available.  If that heavy stack is missing (as during static
analysis in this minimal environment) we expose a placeholder that fails fast
on instantiation.
"""

from __future__ import annotations

import importlib
from typing import Any

# ---------------------------------------------------------------------------
#                 TRY TO IMPORT THE REAL IMPLEMENTATION (IF ANY)
# ---------------------------------------------------------------------------

try:
    # Prefer the fully-featured implementation located in ``src.main``.  We use
    # ``importlib`` instead of a relative import so that the file works both as
    # part of the ``src`` package *and* when the package is absent from
    # ``sys.path`` during static validation.
    _RealTrainer = getattr(importlib.import_module("src.main"), "ContinualTrainer")
    ContinualTrainer = _RealTrainer  # noqa: N816 – public re-export
except Exception:  # pragma: no cover – minimal environment stub

    class ContinualTrainer:  # pylint: disable=too-few-public-methods
        """Placeholder that points users to the missing heavy dependencies."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            raise RuntimeError(
                "The real ContinualTrainer is not available in this minimal "
                "environment. Install the optional heavy dependencies and make "
                "sure `src.main` can be imported in order to run training."
            )

__all__ = ["ContinualTrainer"]
