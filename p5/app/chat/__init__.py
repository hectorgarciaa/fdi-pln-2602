from typing import Any

__all__ = ["chat"]


def __getattr__(name: str) -> Any:
    if name == "chat":
        from .chat import chat

        return chat
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
