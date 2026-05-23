from pathlib import Path

from ..inference.inference import DEFAULT_ARTIFACTS_DIR, generate, load_model


def chat(
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    weight_dir: str | Path | None = None,
    device: str | None = None,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
) -> None:
    print("Bienvenido al chat con el LLM. Escribe 'salir' para terminar.")
    model, tokenizer, device = load_model(
        artifacts_dir=artifacts_dir,
        device=device,
        weight_dir=weight_dir,
    )
    try:
        prompt = input("Tú: ")
        while prompt.lower() != "salir":
            response = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device,
            )
            print(f"LLM: {response}\n")
            prompt = input("Tú: ")
    except EOFError:
        print()


if __name__ == "__main__":
    chat()
