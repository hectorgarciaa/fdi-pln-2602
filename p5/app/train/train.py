from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from ..model.model import LLM
from ..tokenizer.tokenizer import MiniBPETokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def resolve_device(device: str | None) -> torch.device:
    requested = device.lower() if device is not None else None
    auto_selected_cuda = requested is None
    candidate = requested if requested is not None else "cuda"

    if candidate == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")

        try:
            driver_device_count = torch.cuda.device_count()
        except RuntimeError:
            driver_device_count = 0

        if auto_selected_cuda:
            print("CUDA no está disponible; se usará CPU para el entrenamiento.")
            return torch.device("cpu")

        if driver_device_count > 0 and torch.version.cuda is not None:
            raise RuntimeError(
                "Se solicitó --device cuda, pero la build de PyTorch no puede inicializar CUDA "
                f"(torch={torch.__version__}, build CUDA={torch.version.cuda}). "
                "La GPU parece visible para el driver, así que probablemente hay una incompatibilidad "
                "entre la versión de CUDA de PyTorch y el driver instalado. "
                "Instala una build de PyTorch compatible con tu driver o actualiza el driver."
            )

        raise RuntimeError(
            "Se solicitó --device cuda, pero CUDA no está disponible en este entorno."
        )

    return torch.device(candidate)


class TextDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        if seq_len < 2:
            raise ValueError("seq_len debe ser al menos 2.")
        if len(token_ids) <= seq_len:
            raise ValueError("No hay suficientes tokens para construir ejemplos de entrenamiento.")

        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.token_ids[index : index + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def read_corpus(data_dir: Path) -> str:
    text_files = sorted(data_dir.glob("*.txt"))
    if not text_files:
        raise FileNotFoundError(f"No se encontraron ficheros .txt en {data_dir}")

    return "\n\n".join(path.read_text(encoding="utf-8") for path in text_files)


def build_dataloaders(
    token_ids: list[int],
    seq_len: int,
    batch_size: int,
    train_split: float,
) -> tuple[DataLoader, DataLoader]:
    dataset = TextDataset(token_ids, seq_len)

    train_size = max(1, int(len(dataset) * train_split))
    val_size = max(1, len(dataset) - train_size)
    if train_size + val_size > len(dataset):
        train_size = len(dataset) - 1
        val_size = 1

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_batches += 1

    model.train()
    return total_loss / max(1, total_batches)


def train_model(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    vocab_size: int = 256,
    seq_len: int = 64,
    batch_size: int = 16,
    epochs: int = 5,
    learning_rate: float = 3e-4,
    dim_embedding: int = 64,
    dim_attention: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    train_split: float = 0.9,
    device: str | None = None,
) -> tuple[LLM, MiniBPETokenizer]:
    data_path = Path(data_dir)
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    raw_text = read_corpus(data_path)

    tokenizer = MiniBPETokenizer()
    tokenizer.train(raw_text, vocab_size=vocab_size)
    token_ids = tokenizer.encode(raw_text)
    if len(token_ids) <= seq_len:
        raise ValueError(
            f"El corpus tokenizado solo tiene {len(token_ids)} tokens y seq_len={seq_len}."
        )

    train_loader, val_loader = build_dataloaders(
        token_ids=token_ids,
        seq_len=seq_len,
        batch_size=batch_size,
        train_split=train_split,
    )

    target_device = resolve_device(device)
    model = LLM(
        vocab_size=len(tokenizer.vocab),
        dim_embedding=dim_embedding,
        dim_attention=dim_attention,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len,
    ).to(target_device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_val_loss = math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for x, y in train_loader:
            x = x.to(target_device)
            y = y.to(target_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        train_loss = total_loss / max(1, total_batches)
        val_loss = evaluate(model, val_loader, target_device)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"perplexity={math.exp(val_loss):.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), artifacts_path / "best_model.pt")

    tokenizer.save(artifacts_path / "tokenizer.json")
    torch.save(model.state_dict(), artifacts_path / "last_model.pt")

    metadata = {
        "vocab_size": len(tokenizer.vocab),
        "seq_len": seq_len,
        "dim_embedding": dim_embedding,
        "dim_attention": dim_attention,
        "num_heads": num_heads,
        "num_layers": num_layers,
    }
    (artifacts_path / "train_config.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in metadata.items()),
        encoding="utf-8",
    )

    return model, tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena el mini transformer con textos de data/.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dim-embedding", type=int, default=64)
    parser.add_argument("--dim-attention", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_model(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dim_embedding=args.dim_embedding,
        dim_attention=args.dim_attention,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        train_split=args.train_split,
        device=args.device,
    )


if __name__ == "__main__":
    main()
