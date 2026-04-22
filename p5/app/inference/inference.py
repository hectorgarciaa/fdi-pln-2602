import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from ..model import LLM
from ..tokenizer import MiniBPETokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DEFAULT_CHECKPOINT = "best_model.pt"

@dataclass(frozen=True)
class InferenceArtifacts:
    model: LLM
    tokenizer: MiniBPETokenizer
    device: torch.device


def read_train_config(config_path: str | Path) -> dict[str, int]:
    config: dict[str, int] = {}
    for line in Path(config_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue

        key, value = stripped.split("=", maxsplit=1)
        config[key] = int(value)

    return config


def _resolve_checkpoint_path(artifacts_dir: Path, checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_absolute() or checkpoint_path.exists():
        return checkpoint_path

    return artifacts_dir / checkpoint_path


def _build_model(config: dict[str, int], tokenizer: MiniBPETokenizer) -> LLM:
    required_keys = [
        "seq_len",
        "dim_embedding",
        "dim_attention",
        "num_heads",
        "num_layers",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Faltan claves en train_config.txt: {missing}")

    config_vocab_size = config.get("vocab_size")
    if config_vocab_size is not None and config_vocab_size != len(tokenizer.vocab):
        raise ValueError(
            "El vocabulario del tokenizer no coincide con train_config.txt: "
            f"{len(tokenizer.vocab)} != {config_vocab_size}"
        )

    return LLM(
        vocab_size=len(tokenizer.vocab),
        dim_embedding=config["dim_embedding"],
        dim_attention=config["dim_attention"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["seq_len"],
    )


def load_artifacts(
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    device: str | torch.device | None = None,
) -> InferenceArtifacts:
    artifacts_path = Path(artifacts_dir)
    tokenizer = MiniBPETokenizer.load(artifacts_path / "tokenizer.json")
    config = read_train_config(artifacts_path / "train_config.txt")

    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = _build_model(config, tokenizer).to(target_device)

    checkpoint_path = _resolve_checkpoint_path(artifacts_path, checkpoint)
    state_dict = torch.load(
        checkpoint_path,
        map_location=target_device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return InferenceArtifacts(model=model, tokenizer=tokenizer, device=target_device)


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
) -> int:
    if temperature < 0:
        raise ValueError("temperature debe ser mayor o igual que 0.")

    if temperature == 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k debe ser mayor o igual que 1.")

        k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, k=k)
        probabilities = torch.softmax(values, dim=-1)
        sampled_index = torch.multinomial(probabilities, num_samples=1)
        return int(indices[sampled_index].item())

    probabilities = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


@torch.no_grad()
def generate_text(
    artifacts: InferenceArtifacts,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = 40,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens debe ser mayor o igual que 0.")

    token_ids = artifacts.tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("El prompt no ha producido ningún token.")

    generated = list(token_ids)
    for _ in range(max_new_tokens):
        context = generated[-artifacts.model.max_seq_len :]
        x = torch.tensor([context], dtype=torch.long, device=artifacts.device)
        logits = artifacts.model(x)[0, -1]
        next_token_id = _sample_next_token(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
        )
        generated.append(next_token_id)

    return artifacts.tokenizer.decode(generated)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera texto con un modelo ya entrenado.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    artifacts = load_artifacts(
        artifacts_dir=args.artifacts_dir,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    generated_text = generate_text(
        artifacts=artifacts,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(generated_text)


if __name__ == "__main__":
    main()
