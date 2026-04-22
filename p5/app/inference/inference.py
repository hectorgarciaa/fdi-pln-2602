import argparse
from pathlib import Path

import torch

from ..model import LLM
from ..tokenizer import MiniBPETokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "best"


def load_model(artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR, device: str | None = None):
    """Carga el modelo, tokenizer y config."""
    artifacts_path = Path(artifacts_dir)
    
    # Leer config
    config = {}
    for line in (artifacts_path / "train_config.txt").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=")
            config[k] = int(v)
    
    # Cargar tokenizer
    tokenizer = MiniBPETokenizer.load(artifacts_path / "tokenizer.json")
    
    # Crear modelo
    model = LLM(
        vocab_size=len(tokenizer.vocab),
        dim_embedding=config["dim_embedding"],
        dim_attention=config["dim_attention"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["seq_len"],
    )
    
    # Cargar pesos
    target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dict = torch.load(artifacts_path / "model.pt", map_location=target_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(target_device).eval()
    
    return model, tokenizer, target_device


@torch.no_grad()
def generate(model: LLM, tokenizer: MiniBPETokenizer, prompt: str, max_tokens = 200, 
             temperature = 1.0, top_k = 40, device = None):
    """Genera texto a partir de un prompt."""
    token_ids = tokenizer.encode(prompt)
    generated = list(token_ids)
    
    for _ in range(max_tokens):
        context = generated[-model.max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        
        if temperature == 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_k:
                k = min(top_k, len(logits))
                values, indices = torch.topk(logits, k=k)
                probs = torch.softmax(values, dim=-1)
                next_id = int(indices[torch.multinomial(probs, 1)].item())
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
        
        generated.append(next_id)
    
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Genera texto con el modelo.")
    parser.add_argument("--prompt", type=str, required=True, help="Texto inicial")
    parser.add_argument("--max-tokens", type=int, default=50, help="Máximo de tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperatura")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo")
    parser.add_argument("--seed", type=int, default=None, help="Seed")
    args = parser.parse_args()
    
    if args.seed:
        torch.manual_seed(args.seed)
    
    model, tokenizer, device = load_model(device=args.device)
    text = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, device)
    print(text)


if __name__ == "__main__":
    main()
