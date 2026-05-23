import argparse
import json
from pathlib import Path

from .train import (
    DEFAULT_DATA_PATH,
    DEFAULT_LLM_ARTIFACTS_DIR,
    DEFAULT_OUTPUT_DIR,
    train,
)


DEFAULT_BATCH_SIZES = [8, 16]
DEFAULT_EPOCHS = [8]
DEFAULT_LEARNING_RATES = [1e-3, 5e-4]
DEFAULT_FREEZE_BACKBONE = [False, True]


def parse_int_list(raw_value: str) -> list[int]:
    return [int(value.strip()) for value in raw_value.split(",") if value.strip()]


def parse_float_list(raw_value: str) -> list[float]:
    return [float(value.strip()) for value in raw_value.split(",") if value.strip()]


def parse_bool_list(raw_value: str) -> list[bool]:
    mapping = {"true": True, "false": False}
    values = []
    for value in raw_value.split(","):
        normalized = value.strip().lower()
        if not normalized:
            continue
        if normalized not in mapping:
            raise ValueError(f"Valor booleano invalido: {value}")
        values.append(mapping[normalized])
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid search de hiperparametros para NER."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_LLM_ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--batch-sizes", type=parse_int_list, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--epochs-list", type=parse_int_list, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--learning-rates", type=parse_float_list, default=DEFAULT_LEARNING_RATES
    )
    parser.add_argument(
        "--freeze-backbone-options",
        type=parse_bool_list,
        default=DEFAULT_FREEZE_BACKBONE,
    )
    return parser


def run_hyperparam_search(
    data_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    batch_sizes: list[int],
    epochs_list: list[int],
    learning_rates: list[float],
    freeze_backbone_options: list[bool],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for batch_size in batch_sizes:
        for epochs in epochs_list:
            for learning_rate in learning_rates:
                for freeze_backbone in freeze_backbone_options:
                    config = {
                        "data_path": str(data_path),
                        "artifacts_dir": str(artifacts_dir),
                        "output_dir": str(output_dir),
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "freeze_backbone": freeze_backbone,
                    }

                    print(
                        "\nProbando NER: "
                        f"batch={batch_size}, epochs={epochs}, "
                        f"lr={learning_rate}, freeze_backbone={freeze_backbone}"
                    )

                    try:
                        run_dir, best_epoch = train(
                            data_path=data_path,
                            artifacts_dir=artifacts_dir,
                            output_dir=output_dir,
                            batch_size=batch_size,
                            epochs=epochs,
                            learning_rate=learning_rate,
                            freeze_backbone=freeze_backbone,
                        )

                        result = {
                            "config": config,
                            "run_dir": str(run_dir),
                            "best_val_loss": best_epoch["val_loss"],
                            "best_val_token_accuracy": best_epoch["val_token_accuracy"],
                            "best_val_entity_token_f1_micro": best_epoch[
                                "val_entity_token_f1_micro"
                            ],
                            "best_epoch": best_epoch["epoch"],
                            "status": "completed",
                        }
                        results.append(result)
                        print(
                            f"Val Loss: {best_epoch['val_loss']:.4f}, "
                            f"Entity F1: {best_epoch['val_entity_token_f1_micro']:.4f}"
                        )
                    except Exception as exc:
                        result = {
                            "config": config,
                            "status": "failed",
                            "error": str(exc),
                        }
                        results.append(result)
                        print(f"Error: {exc}")

    output_path = output_dir / "hyperparam_results.json"
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
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        batch_sizes=args.batch_sizes,
        epochs_list=args.epochs_list,
        learning_rates=args.learning_rates,
        freeze_backbone_options=args.freeze_backbone_options,
    )


if __name__ == "__main__":
    main()
