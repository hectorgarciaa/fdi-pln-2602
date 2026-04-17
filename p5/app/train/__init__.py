from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .train import TextDataset, train_model

__all__ = ["TextDataset", "train_model"]


def __getattr__(name: str):
    if name in __all__:
        from .train import TextDataset, train_model

        exports = {
            "TextDataset": TextDataset,
            "train_model": train_model,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
