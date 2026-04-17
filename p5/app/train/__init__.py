__all__ = ["TextDataset", "train_model"]


def __getattr__(name: str):
    if name in __all__:
        from .train import TextDataset, train_model

        return {"TextDataset": TextDataset, "train_model": train_model}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
