from ..inference.inference import generate, load_model

def chat():
    print("Bienvenido al chat con el LLM. Escribe 'salir' para terminar.")
    model, tokenizer, device = load_model()
    prompt = input("Tú: ")
    while prompt.lower() != "salir":
        response = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=100,
            temperature=0.7,
            top_k=40,
            device=device,
        )
        print(f"LLM: {response}\n")
        prompt = input("Tú: ")

if __name__ == "__main__":
    chat()
