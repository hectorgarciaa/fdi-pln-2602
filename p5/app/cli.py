import argparse
from pathlib import Path

import torch

from .chat.chat import chat
from .inference.inference import DEFAULT_ARTIFACTS_DIR as DEFAULT_LLM_ARTIFACTS_DIR
from .inference.inference import generate, load_model
from .ner.hyperparam_train import (
    DEFAULT_BATCH_SIZES as DEFAULT_NER_BATCH_SIZES,
    DEFAULT_EPOCHS as DEFAULT_NER_EPOCHS,
    DEFAULT_FREEZE_BACKBONE as DEFAULT_NER_FREEZE_BACKBONE,
    DEFAULT_LEARNING_RATES as DEFAULT_NER_LEARNING_RATES,
    parse_bool_list,
    parse_float_list as parse_ner_float_list,
    parse_int_list as parse_ner_int_list,
    run_hyperparam_search as run_ner_hyperparam_search,
)
from .ner.predict import (
    DEFAULT_ARTIFACTS_DIR as DEFAULT_NER_ARTIFACTS_DIR,
    load_ner_model,
    predict_entities,
    resolve_input_text,
)
from .ner.train import (
    DEFAULT_DATA_PATH as DEFAULT_NER_DATA_PATH,
    DEFAULT_LLM_ARTIFACTS_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_NER_OUTPUT_DIR,
    train as train_ner,
)
from .train.hyperparam_train import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_DIM_ATTENTIONS,
    DEFAULT_DIM_EMBEDDINGS,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATES,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_SEQ_LENS,
    DEFAULT_VOCAB_SIZES,
    parse_float_list,
    parse_int_list,
    run_hyperparam_search as run_llm_hyperparam_search,
)
from .train.train import DEFAULT_ARTIFACTS_BASE_DIR, DEFAULT_DATA_DIR, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fdi-pln-2602-p5",
        description="CLI principal de la practica 5: entrenamiento LLM, generacion y NER.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_llm = subparsers.add_parser(
        "train-llm", help="Entrena una configuracion del LLM."
    )
    train_llm.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train_llm.add_argument("--vocab-size", type=int, default=256)
    train_llm.add_argument("--seq-len", type=int, default=64)
    train_llm.add_argument("--dim-embedding", type=int, default=64)
    train_llm.add_argument("--dim-attention", type=int, default=128)
    train_llm.add_argument("--num-heads", type=int, default=4)
    train_llm.add_argument("--num-layers", type=int, default=2)
    train_llm.add_argument("--batch-size", type=int, default=16)
    train_llm.add_argument("--epochs", type=int, default=5)
    train_llm.add_argument("--learning-rate", type=float, default=3e-4)
    train_llm.add_argument("--train-split", type=float, default=0.9)
    train_llm.add_argument("--device", type=str, default=None)
    train_llm.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ARTIFACTS_BASE_DIR
    )

    train_hp_llm = subparsers.add_parser(
        "train-hp-llm", help="Grid search de hiperparametros para el LLM."
    )
    train_hp_llm.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train_hp_llm.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ARTIFACTS_BASE_DIR
    )
    train_hp_llm.add_argument("--device", type=str, default=None)
    train_hp_llm.add_argument(
        "--vocab-sizes", type=parse_int_list, default=DEFAULT_VOCAB_SIZES
    )
    train_hp_llm.add_argument(
        "--seq-lens", type=parse_int_list, default=DEFAULT_SEQ_LENS
    )
    train_hp_llm.add_argument(
        "--dim-embeddings", type=parse_int_list, default=DEFAULT_DIM_EMBEDDINGS
    )
    train_hp_llm.add_argument(
        "--dim-attentions", type=parse_int_list, default=DEFAULT_DIM_ATTENTIONS
    )
    train_hp_llm.add_argument(
        "--num-heads", type=parse_int_list, default=DEFAULT_NUM_HEADS
    )
    train_hp_llm.add_argument(
        "--num-layers", type=parse_int_list, default=DEFAULT_NUM_LAYERS
    )
    train_hp_llm.add_argument(
        "--batch-sizes", type=parse_int_list, default=DEFAULT_BATCH_SIZES
    )
    train_hp_llm.add_argument(
        "--epochs-list", type=parse_int_list, default=DEFAULT_EPOCHS
    )
    train_hp_llm.add_argument(
        "--learning-rates", type=parse_float_list, default=DEFAULT_LEARNING_RATES
    )

    generate_llm = subparsers.add_parser(
        "generate-llm", help="Genera texto o abre chat con el LLM."
    )
    generate_llm.add_argument(
        "--artifacts-dir", type=Path, default=DEFAULT_LLM_ARTIFACTS_DIR
    )
    generate_llm.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt inicial para generacion puntual",
    )
    generate_llm.add_argument(
        "--chat",
        action="store_true",
        help="Abre un chat interactivo en lugar de una generacion puntual",
    )
    generate_llm.add_argument("--max-tokens", type=int, default=50)
    generate_llm.add_argument("--temperature", type=float, default=1.0)
    generate_llm.add_argument("--top-k", type=int, default=40)
    generate_llm.add_argument("--device", type=str, default=None)
    generate_llm.add_argument("--seed", type=int, default=None)

    train_ner_parser = subparsers.add_parser(
        "train-ner", help="Entrena el modelo NER sobre el backbone del LLM."
    )
    train_ner_parser.add_argument(
        "--data-path", type=Path, default=DEFAULT_NER_DATA_PATH
    )
    train_ner_parser.add_argument(
        "--artifacts-dir", type=Path, default=DEFAULT_LLM_ARTIFACTS_DIR
    )
    train_ner_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_NER_OUTPUT_DIR
    )
    train_ner_parser.add_argument("--batch-size", type=int, default=16)
    train_ner_parser.add_argument("--epochs", type=int, default=8)
    train_ner_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_ner_parser.add_argument("--freeze-backbone", action="store_true")

    train_hp_ner = subparsers.add_parser(
        "train-hp-ner", help="Grid search de hiperparametros para NER."
    )
    train_hp_ner.add_argument("--data-path", type=Path, default=DEFAULT_NER_DATA_PATH)
    train_hp_ner.add_argument(
        "--artifacts-dir", type=Path, default=DEFAULT_LLM_ARTIFACTS_DIR
    )
    train_hp_ner.add_argument("--output-dir", type=Path, default=DEFAULT_NER_OUTPUT_DIR)
    train_hp_ner.add_argument(
        "--batch-sizes", type=parse_ner_int_list, default=DEFAULT_NER_BATCH_SIZES
    )
    train_hp_ner.add_argument(
        "--epochs-list", type=parse_ner_int_list, default=DEFAULT_NER_EPOCHS
    )
    train_hp_ner.add_argument(
        "--learning-rates",
        type=parse_ner_float_list,
        default=DEFAULT_NER_LEARNING_RATES,
    )
    train_hp_ner.add_argument(
        "--freeze-backbone-options",
        type=parse_bool_list,
        default=DEFAULT_NER_FREEZE_BACKBONE,
    )

    detect_ner = subparsers.add_parser(
        "detect-ner", help="Detecta entidades nombradas con el mejor modelo NER."
    )
    detect_ner.add_argument(
        "--artifacts-dir", type=Path, default=DEFAULT_NER_ARTIFACTS_DIR
    )
    detect_ner.add_argument("--device", type=str, default=None)
    detect_ner.add_argument("--show-labels", action="store_true")
    detect_ner_input = detect_ner.add_mutually_exclusive_group(required=True)
    detect_ner_input.add_argument(
        "--text", type=str, help="Texto sobre el que detectar entidades"
    )
    detect_ner_input.add_argument(
        "--input-file",
        type=Path,
        help="Fichero de texto sobre el que detectar entidades",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train-llm":
        train_model(
            data_dir=args.data_dir,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            dim_embedding=args.dim_embedding,
            dim_attention=args.dim_attention,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            train_split=args.train_split,
            device=args.device,
            artifacts_base_dir=args.output_dir,
        )
        return

    if args.command == "train-hp-llm":
        run_llm_hyperparam_search(
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
        return

    if args.command == "generate-llm":
        if args.seed is not None:
            torch.manual_seed(args.seed)

        if args.chat:
            chat(
                artifacts_dir=args.artifacts_dir,
                device=args.device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            return

        if not args.prompt:
            parser.error("generate-llm requiere --prompt o bien --chat.")

        model, tokenizer, target_device = load_model(
            artifacts_dir=args.artifacts_dir,
            device=args.device,
        )
        text = generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=target_device,
        )
        print(text)
        return

    if args.command == "train-ner":
        train_ner(
            data_path=args.data_path,
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            freeze_backbone=args.freeze_backbone,
        )
        return

    if args.command == "train-hp-ner":
        run_ner_hyperparam_search(
            data_path=args.data_path,
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
            batch_sizes=args.batch_sizes,
            epochs_list=args.epochs_list,
            learning_rates=args.learning_rates,
            freeze_backbone_options=args.freeze_backbone_options,
        )
        return

    if args.command == "detect-ner":
        input_text = resolve_input_text(args.text, args.input_file)
        model, tokenizer, id_to_label, target_device = load_ner_model(
            artifacts_dir=args.artifacts_dir,
            device=args.device,
        )
        words, labels, entities = predict_entities(
            model,
            tokenizer,
            id_to_label,
            input_text,
            target_device,
        )

        if args.show_labels:
            print("Etiquetas por palabra:")
            for word, label in zip(words, labels):
                print(f"{word}\t{label}")
            print()

        if not entities:
            print("No se detectaron entidades.")
            return

        print("Entidades detectadas:")
        for entity in entities:
            print(f"{entity['type']}: {entity['text']}")
        return

    parser.error(f"Comando no soportado: {args.command}")


if __name__ == "__main__":
    main()
