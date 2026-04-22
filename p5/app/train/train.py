import time
import math
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from ..model import LLM
from ..tokenizer import MiniBPETokenizer
from .evaluate import evaluate
from .utils import read_corpus, build_dataloaders, build_parser


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts"

def train_model(
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
    if epochs < 1:
        raise ValueError("epochs debe ser mayor o igual que 1.")

    data_path = Path(DEFAULT_DATA_DIR)
    artifacts_path = Path(DEFAULT_ARTIFACTS_DIR)

    now = time.strftime("%Y%m%d-%H%M%S")
    artifacts_path_exp = artifacts_path / now
    artifacts_path_exp.mkdir(parents=True, exist_ok=True)

    raw_text = read_corpus(data_path)

    tokenizer = MiniBPETokenizer()
    tokenizer.train(raw_text, vocab_size=vocab_size)
    token_ids = tokenizer.encode(raw_text)

    train_loader, val_loader = build_dataloaders(
        token_ids=token_ids,
        seq_len=seq_len,
        batch_size=batch_size,
        train_split=train_split,
    )

    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
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
    epochs_data = []

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
        perplexity = math.exp(val_loss)

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "perplexity": round(perplexity, 2),
        }
        epochs_data.append(epoch_info)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"perplexity={perplexity:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), artifacts_path_exp / "best_model.pt")

    tokenizer.save(artifacts_path_exp / "tokenizer.json")
    torch.save(model.state_dict(), artifacts_path_exp / "last_model.pt")

    metadata = {
        "vocab_size": len(tokenizer.vocab),
        "seq_len": seq_len,
        "dim_embedding": dim_embedding,
        "dim_attention": dim_attention,
        "num_heads": num_heads,
        "num_layers": num_layers,
    }
    (artifacts_path_exp / "train_config.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in metadata.items()),
        encoding="utf-8",
    )

    results_path = artifacts_path_exp / "results.txt"
    results_path.write_text(
        json.dumps(epochs_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    current_best_epoch = min(epochs_data, key=lambda x: x["perplexity"])
    best_model_path = artifacts_path / "best"
    if best_model_path.exists():
        best_results_path = best_model_path / "results.txt"
        if best_results_path.exists():
            best_results = json.loads(best_results_path.read_text(encoding="utf-8"))
            best_epoch = min(best_results, key=lambda x: x["perplexity"])
            if current_best_epoch["perplexity"] < best_epoch["perplexity"]:
                shutil.copytree(artifacts_path_exp, best_model_path, dirs_exist_ok=True)
        else:
            shutil.copytree(artifacts_path_exp, best_model_path, dirs_exist_ok=True)
    else:
        shutil.copytree(artifacts_path_exp, best_model_path, dirs_exist_ok=True)

    return model, tokenizer


def main() -> None:
    args = build_parser().parse_args()
    train_model(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        dim_embedding=args.dim_embedding,
        dim_attention=args.dim_attention,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_split=args.train_split,
        device=args.device,
    )


if __name__ == "__main__":
    main()
