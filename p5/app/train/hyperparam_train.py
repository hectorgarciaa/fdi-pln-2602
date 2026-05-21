import argparse
import json
from pathlib import Path

from .train import DEFAULT_ARTIFACTS_BASE_DIR, DEFAULT_DATA_DIR, train_model


DEFAULT_VOCAB_SIZES = [64, 95, 128]
DEFAULT_DIM_EMBEDDINGS = [70, 105, 140]
DEFAULT_DIM_ATTENTIONS = [140, 210, 280]
DEFAULT_NUM_LAYERS = [2]
DEFAULT_NUM_HEADS = [2]
DEFAULT_EPOCHS = [5]
DEFAULT_BATCH_SIZES = [16]
DEFAULT_SEQ_LENS = [96, 128]
DEFAULT_LEARNING_RATES = [1e-4]


def parse_int_list(raw_value: str) -> list[int]:
    return [int(value.strip()) for value in raw_value.split(",") if value.strip()]


def parse_float_list(raw_value: str) -> list[float]:
    return [float(value.strip()) for value in raw_value.split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid search de hiperparametros para el LLM."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS_BASE_DIR)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--vocab-sizes", type=parse_int_list, default=DEFAULT_VOCAB_SIZES
    )
    parser.add_argument("--seq-lens", type=parse_int_list, default=DEFAULT_SEQ_LENS)
    parser.add_argument(
        "--dim-embeddings", type=parse_int_list, default=DEFAULT_DIM_EMBEDDINGS
    )
    parser.add_argument(
        "--dim-attentions", type=parse_int_list, default=DEFAULT_DIM_ATTENTIONS
    )
    parser.add_argument("--num-heads", type=parse_int_list, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-layers", type=parse_int_list, default=DEFAULT_NUM_LAYERS)
    parser.add_argument(
        "--batch-sizes", type=parse_int_list, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--epochs-list", type=parse_int_list, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--learning-rates", type=parse_float_list, default=DEFAULT_LEARNING_RATES
    )
    return parser


def run_hyperparam_search(
    data_dir: Path,
    output_dir: Path,
    device: str | None,
    vocab_sizes: list[int],
    seq_lens: list[int],
    dim_embeddings: list[int],
    dim_attentions: list[int],
    num_heads_values: list[int],
    num_layers_values: list[int],
    batch_sizes: list[int],
    epochs_list: list[int],
    learning_rates: list[float],
) -> None:
    llm_artifacts_dir = output_dir
    llm_artifacts_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for vocab_size in vocab_sizes:
        for seq_len in seq_lens:
            for dim_embedding in dim_embeddings:
                for dim_attention in dim_attentions:
                    for num_heads in num_heads_values:
                        for num_layers in num_layers_values:
                            for batch_size in batch_sizes:
                                for epochs in epochs_list:
                                    for learning_rate in learning_rates:
                                        config = {
                                            "vocab_size": vocab_size,
                                            "seq_len": seq_len,
                                            "dim_embedding": dim_embedding,
                                            "dim_attention": dim_attention,
                                            "num_heads": num_heads,
                                            "num_layers": num_layers,
                                            "batch_size": batch_size,
                                            "epochs": epochs,
                                            "learning_rate": learning_rate,
                                        }

                                        print(
                                            f"\nProbando: vocab={vocab_size}, seq={seq_len}, "
                                            f"emb={dim_embedding}, attn={dim_attention}, "
                                            f"heads={num_heads}, layers={num_layers}, "
                                            f"batch={batch_size}, epochs={epochs}, lr={learning_rate}"
                                        )

                                        try:
                                            _, _, run_dir = train_model(
                                                vocab_size=vocab_size,
                                                seq_len=seq_len,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                learning_rate=learning_rate,
                                                dim_embedding=dim_embedding,
                                                dim_attention=dim_attention,
                                                num_heads=num_heads,
                                                num_layers=num_layers,
                                                device=device,
                                                artifacts_base_dir=llm_artifacts_dir,
                                                data_dir=data_dir,
                                            )

                                            epochs_data = json.loads(
                                                (run_dir / "results.txt").read_text(
                                                    encoding="utf-8"
                                                )
                                            )
                                            best_epoch = min(
                                                epochs_data, key=lambda x: x["val_loss"]
                                            )
                                            result = {
                                                "config": config,
                                                "run_dir": str(run_dir),
                                                "best_val_loss": best_epoch["val_loss"],
                                                "best_perplexity": best_epoch[
                                                    "perplexity"
                                                ],
                                                "best_epoch": best_epoch["epoch"],
                                                "status": "completed",
                                            }
                                            results.append(result)
                                            print(
                                                f"Val Loss: {best_epoch['val_loss']:.4f}, "
                                                f"PPL: {best_epoch['perplexity']:.2f}"
                                            )
                                        except Exception as exc:
                                            result = {
                                                "config": config,
                                                "status": "failed",
                                                "error": str(exc),
                                            }
                                            results.append(result)
                                            print(f"Error: {exc}")

    output_path = llm_artifacts_dir / "hyperparam_results.json"
    if output_path.exists():
        previous_results = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(previous_results, list):
            raise ValueError(
                f"Contenido invalido en {output_path}: se esperaba una lista JSON"
            )
        results = previous_results + results
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nResultados guardados en: {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    run_hyperparam_search(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        vocab_sizes=args.vocab_sizes,
        seq_lens=args.seq_lens,
        dim_embeddings=args.dim_embeddings,
        dim_attentions=args.dim_attentions,
        num_heads_values=args.num_heads,
        num_layers_values=args.num_layers,
        batch_sizes=args.batch_sizes,
        epochs_list=args.epochs_list,
        learning_rates=args.learning_rates,
    )


if __name__ == "__main__":
    main()
