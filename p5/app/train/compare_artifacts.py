import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from ..model import LLM
    from ..tokenizer import MiniBPETokenizer
    from .utils import TextDataset, read_corpus
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from app.model import LLM
    from app.tokenizer import MiniBPETokenizer
    from app.train.utils import TextDataset, read_corpus


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DEFAULT_OUTPUT_PATH = DEFAULT_ARTIFACTS_DIR / "artifact_comparison.json"


def parse_train_config(path: Path) -> dict[str, int | float | str]:
    config: dict[str, int | float | str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        value = raw_value.strip()
        try:
            parsed: int | float | str = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
        config[key.strip()] = parsed
    return config


def read_best_epoch(run_dir: Path) -> dict[str, int | float] | None:
    results_path = run_dir / "results.txt"
    if not results_path.exists():
        return None

    epochs_data = json.loads(results_path.read_text(encoding="utf-8"))
    if not epochs_data:
        return None

    return min(epochs_data, key=lambda row: row["val_loss"])


def normalize_text_for_tokenizer(text: str) -> str:
    return " ".join(text.lower().split())


def build_validation_text(raw_text: str, train_split: float) -> str:
    split_point = int(len(raw_text) * train_split)
    return raw_text[split_point:]


def evaluate_run(
    run_dir: Path,
    raw_val_text: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, int | float | str]:
    config = parse_train_config(run_dir / "train_config.txt")
    tokenizer = MiniBPETokenizer.load(run_dir / "tokenizer.json")

    model = LLM(
        vocab_size=len(tokenizer.vocab),
        dim_embedding=int(config["dim_embedding"]),
        dim_attention=int(config["dim_attention"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        max_seq_len=int(config["seq_len"]),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    token_ids = tokenizer.encode(raw_val_text)
    if len(token_ids) <= int(config["seq_len"]):
        raise ValueError(
            f"Validación insuficiente: {len(token_ids)} tokens para seq_len={config['seq_len']}"
        )
    dataset = TextDataset(token_ids, seq_len=int(config["seq_len"]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_nll = 0.0
    total_target_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            total_nll += loss.item()
            total_target_tokens += y.numel()

    if total_target_tokens == 0:
        raise ValueError("No hay suficientes tokens de validación para evaluar este experimento.")

    char_count = len(raw_val_text)
    unk_id = tokenizer.vocab[tokenizer.unk_token]
    unk_count = sum(1 for token_id in token_ids if token_id == unk_id)
    best_epoch = read_best_epoch(run_dir)

    result: dict[str, int | float | str] = {
        "run_dir": str(run_dir),
        "vocab_size": int(config["vocab_size"]),
        "seq_len": int(config["seq_len"]),
        "dim_embedding": int(config["dim_embedding"]),
        "dim_attention": int(config["dim_attention"]),
        "num_heads": int(config["num_heads"]),
        "num_layers": int(config["num_layers"]),
        "batch_size": int(config.get("batch_size", -1)),
        "epochs": int(config.get("epochs", -1)),
        "learning_rate": float(config["learning_rate"]) if "learning_rate" in config else None,
        "train_split": float(config.get("train_split", 0.9)),
        "token_count": len(token_ids),
        "char_count": char_count,
        "tokens_per_char": len(token_ids) / char_count,
        "unk_rate": unk_count / max(1, len(token_ids)),
        "nll_per_token": total_nll / total_target_tokens,
        "token_perplexity": math.exp(total_nll / total_target_tokens),
        "nll_per_char": total_nll / char_count,
        "bits_per_char": total_nll / (char_count * math.log(2)),
    }

    if best_epoch is not None:
        result["best_epoch"] = int(best_epoch["epoch"])
        result["saved_val_loss"] = float(best_epoch["val_loss"])
        result["saved_perplexity"] = float(best_epoch["perplexity"])

    return result


def discover_runs(artifacts_dir: Path) -> list[Path]:
    required_files = [
        artifacts_dir / "best_model.pt",
        artifacts_dir / "tokenizer.json",
        artifacts_dir / "train_config.txt",
    ]
    if artifacts_dir.is_dir() and all(file_path.exists() for file_path in required_files):
        return [artifacts_dir]

    runs: list[Path] = []
    for path in sorted(artifacts_dir.iterdir()):
        if not path.is_dir() or path.name == "best":
            continue

        required_files = [
            path / "best_model.pt",
            path / "tokenizer.json",
            path / "train_config.txt",
        ]
        if all(file_path.exists() for file_path in required_files):
            runs.append(path)
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara experimentos guardados en artifacts con métricas normalizadas por carácter."
    )
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    raw_text = read_corpus(args.data_dir)
    raw_val_text = normalize_text_for_tokenizer(build_validation_text(raw_text, args.train_split))
    if not raw_val_text:
        raise ValueError("El texto de validación está vacío tras aplicar el split indicado.")

    runs = discover_runs(args.artifacts_dir)
    if not runs:
        raise FileNotFoundError(f"No se encontraron runs evaluables en {args.artifacts_dir}")

    results = []
    skipped = []
    total_runs = len(runs)
    print(f"Evaluando {total_runs} runs en {args.artifacts_dir} usando {device}...")
    for index, run_dir in enumerate(runs, start=1):
        print(f"[{index}/{total_runs}] Evaluando {run_dir.name}...", flush=True)
        try:
            result = evaluate_run(
                run_dir=run_dir,
                raw_val_text=raw_val_text,
                batch_size=args.batch_size,
                device=device,
            )
            results.append(result)
            print(
                f"[{index}/{total_runs}] OK {run_dir.name} | "
                f"bpc={result['bits_per_char']:.4f} | "
                f"ppl_tok={result['token_perplexity']:.4f}",
                flush=True,
            )
        except Exception as exc:
            skipped.append({"run_dir": str(run_dir), "error": str(exc)})
            print(f"[{index}/{total_runs}] Saltado {run_dir.name}: {exc}", flush=True)

    results.sort(key=lambda row: row["bits_per_char"])
    payload = {
        "artifacts_dir": str(args.artifacts_dir),
        "data_dir": str(args.data_dir),
        "device": str(device),
        "train_split_used_for_comparison": args.train_split,
        "validation_chars": len(raw_val_text),
        "ranking_metric": "bits_per_char",
        "results": results,
        "skipped": skipped,
    }

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Comparados {len(results)} experimentos")
    if results:
        best = results[0]
        print(
            "Mejor run por bits_per_char: "
            f"{best['run_dir']} | bpc={best['bits_per_char']:.4f} | "
            f"tpc={best['tokens_per_char']:.4f} | ppl_tok={best['token_perplexity']:.4f}"
        )
    if skipped:
        print(f"Saltados {len(skipped)} experimentos")
    print(f"Resultado guardado en: {args.output}")


if __name__ == "__main__":
    main()
