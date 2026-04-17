from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .inference import (
        InferenceArtifacts,
        generate_text,
        load_artifacts,
        read_train_config,
    )

__all__ = [
    "InferenceArtifacts",
    "generate_text",
    "load_artifacts",
    "read_train_config",
]


def __getattr__(name: str):
    if name in __all__:
        from .inference import (
            InferenceArtifacts,
            generate_text,
            load_artifacts,
            read_train_config,
        )

        exports = {
            "InferenceArtifacts": InferenceArtifacts,
            "generate_text": generate_text,
            "load_artifacts": load_artifacts,
            "read_train_config": read_train_config,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
