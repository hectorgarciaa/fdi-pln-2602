import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

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

def build_dataloaders(token_ids: list[int], seq_len: int, batch_size: int, train_split: float) -> tuple[DataLoader, DataLoader]:
    split_point = int(len(token_ids) * train_split)
    train_ids = token_ids[:split_point]
    val_ids = token_ids[split_point:]
    
    train_dataset = TextDataset(train_ids, seq_len)
    val_dataset = TextDataset(val_ids, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena el mini transformer con textos de data/.")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--dim-embedding", type=int, default=64)
    parser.add_argument("--dim-attention", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None)
    return parser