from typing import Any

__all__ = ["load_model", "generate"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .inference import generate, load_model

        exports = {
            "load_model": load_model,
            "generate": generate,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
