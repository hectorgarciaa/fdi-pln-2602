import argparse
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..model import LLM
from ..tokenizer import MiniBPETokenizer
from ..inference import load_model as load_pretrained_backbone


DEFAULT_DATA_PATH = Path("data") / "ner_dataset.json"
DEFAULT_LLM_ARTIFACTS_DIR = Path("artifacts") / "llm" / "best_general"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "ner"
DEFAULT_TRAIN_SPLIT = 0.85
DEFAULT_SEED = 42
PAD_LABEL = "<pad_label>"


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


class NERDataset(Dataset):
    def __init__(self, examples: list[EncodedExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EncodedExample:
        return self.examples[idx]


class TransformerNERHead(nn.Module):
    def __init__(self, backbone: LLM, num_labels: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.dim_embedding, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone.backbone_forward(x)
        return self.classifier(h)


def read_dataset(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Dataset inválido en {path}: se esperaba lista no vacía")
    return rows


def build_label_vocab(rows: list[dict]) -> dict[str, int]:
    labels = sorted({label for row in rows for label in row["labels"]})
    vocab = {PAD_LABEL: 0}
    for label in labels:
        vocab[label] = len(vocab)
    return vocab


def encode_rows(
    rows: list[dict], tokenizer: MiniBPETokenizer, label_to_id: dict[str, int]
) -> list[EncodedExample]:
    encoded: list[EncodedExample] = []
    for row in rows:
        words = row["words"]
        labels = row["labels"]
        if len(words) != len(labels):
            raise ValueError("words y labels tienen distinta longitud")

        input_ids: list[int] = []
        out_labels: list[int] = []
        for w, lbl in zip(words, labels):
            token_ids = tokenizer.encode(str(w))
            if not token_ids:
                continue
            input_ids.extend(token_ids)
            out_labels.extend([label_to_id[lbl]] * len(token_ids))

        if input_ids:
            encoded.append(EncodedExample(input_ids=input_ids, labels=out_labels))
    return encoded


def collate_batch(
    batch: list[EncodedExample], max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_len = min(max(len(item.input_ids) for item in batch), max_seq_len)
    x = torch.zeros((len(batch), padded_len), dtype=torch.long)
    y = torch.zeros((len(batch), padded_len), dtype=torch.long)

    for i, item in enumerate(batch):
        seq_x = item.input_ids[:padded_len]
        seq_y = item.labels[:padded_len]
        x[i, : len(seq_x)] = torch.tensor(seq_x, dtype=torch.long)
        y[i, : len(seq_y)] = torch.tensor(seq_y, dtype=torch.long)

    return x, y


def token_f1_micro(
    logits: torch.Tensor, y: torch.Tensor, pad_label_id: int = 0
) -> float:
    pred = logits.argmax(dim=-1)
    mask = y != pad_label_id
    tp = ((pred == y) & mask).sum().item()
    n = mask.sum().item()
    if n == 0:
        return 0.0
    precision = tp / n
    recall = tp / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_f1 = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_f1 += token_f1_micro(logits, y, pad_label_id=0)
            count += 1
    return total_loss / max(1, count), total_f1 / max(1, count)


def train(
    data_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    freeze_backbone: bool,
) -> tuple[Path, dict[str, float]]:
    seed = DEFAULT_SEED
    train_split = DEFAULT_TRAIN_SPLIT

    random.seed(seed)
    torch.manual_seed(seed)

    rows = read_dataset(data_path)
    random.shuffle(rows)
    split = int(len(rows) * train_split)
    train_rows = rows[:split]
    val_rows = rows[split:]
    if not train_rows or not val_rows:
        raise ValueError("Split train/val inválido")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, tokenizer, _ = load_pretrained_backbone(artifacts_dir, device)
    label_to_id = build_label_vocab(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_data = encode_rows(train_rows, tokenizer, label_to_id)
    val_data = encode_rows(val_rows, tokenizer, label_to_id)

    collate = lambda b: collate_batch(b, max_seq_len=backbone.max_seq_len)
    train_loader = DataLoader(
        NERDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        NERDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    model = TransformerNERHead(backbone=backbone, num_labels=len(label_to_id)).to(
        device
    )

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )

    best_f1 = -1.0
    history: list[dict[str, float]] = []

    print(f"Guardando run NER en: {run_dir}")
    print(
        f"Train samples: {len(train_data)} | Val samples: {len(val_data)} | Device: {device}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            n += 1

        train_loss = tr_loss / max(1, n)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_token_f1_micro": val_f1,
            }
        )
        print(
            f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    torch.save(model.state_dict(), run_dir / "last_model.pt")
    (run_dir / "label_to_id.json").write_text(
        json.dumps(label_to_id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "id_to_label.json").write_text(
        json.dumps(
            {idx: label for label, idx in label_to_id.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "train_config.json").write_text(
        json.dumps(
            {
                "train_split": train_split,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "seed": seed,
                "freeze_backbone": freeze_backbone,
                "artifacts_dir": str(artifacts_dir),
                "max_seq_len": backbone.max_seq_len,
                "dim_embedding": backbone.dim_embedding,
                "dim_attention": backbone.dim_attention,
                "num_heads": backbone.num_heads,
                "num_layers": backbone.num_layers,
                "vocab_size": len(tokenizer.vocab),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_dir = output_dir / "best"
    current_best = max(history, key=lambda row: row["val_token_f1_micro"])
    if best_dir.exists():
        best_history_path = best_dir / "history.json"
        if best_history_path.exists():
            best_history = json.loads(best_history_path.read_text(encoding="utf-8"))
            previous_best = max(best_history, key=lambda row: row["val_token_f1_micro"])
            if current_best["val_token_f1_micro"] > previous_best["val_token_f1_micro"]:
                shutil.rmtree(best_dir)
                shutil.copytree(run_dir, best_dir)
                print(
                    "Nuevo mejor modelo NER global: "
                    f"{current_best['val_token_f1_micro']:.4f} > {previous_best['val_token_f1_micro']:.4f}"
                )
            else:
                print(
                    "El best global de NER no mejora: "
                    f"{previous_best['val_token_f1_micro']:.4f} >= {current_best['val_token_f1_micro']:.4f}"
                )
        else:
            shutil.rmtree(best_dir)
            shutil.copytree(run_dir, best_dir)
            print(
                f"Best global de NER actualizado en {best_dir} porque no tenia history.json valido."
            )
    else:
        shutil.copytree(run_dir, best_dir)
        print(f"Primer best global de NER guardado en {best_dir}.")

    print(f"Modelo NER guardado en: {run_dir}")
    return run_dir, current_best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrenamiento NER simple sobre backbone preentrenado"
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_LLM_ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
