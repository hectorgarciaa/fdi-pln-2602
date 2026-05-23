import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch

from ..inference import load_model as load_pretrained_backbone
from .train import DEFAULT_OUTPUT_DIR, TransformerNERHead


DEFAULT_ARTIFACTS_DIR = DEFAULT_OUTPUT_DIR / "best"


def load_ner_model(
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    device: str | None = None,
    weight_dir: Path | None = None,
) -> tuple[TransformerNERHead, object, dict[int, str], torch.device]:
    artifacts_path = Path(artifacts_dir)
    config = json.loads(
        (artifacts_path / "train_config.json").read_text(encoding="utf-8")
    )
    label_to_id = json.loads(
        (artifacts_path / "label_to_id.json").read_text(encoding="utf-8")
    )
    id_to_label = {
        int(idx): label
        for idx, label in json.loads(
            (artifacts_path / "id_to_label.json").read_text(encoding="utf-8")
        ).items()
    }

    backbone, tokenizer, target_device = load_pretrained_backbone(
        config["artifacts_dir"],
        device=device,
    )
    model = TransformerNERHead(backbone=backbone, num_labels=len(label_to_id))
    weights_path = (
        artifacts_path / "best_model.pt" if weight_dir is None else Path(weight_dir)
    )
    if weights_path.suffix not in {".pt", ".pth"}:
        raise ValueError(
            f"weight_dir debe apuntar a un fichero .pt o .pth, recibido: {weights_path}"
        )
    state_dict = torch.load(
        weights_path,
        map_location=target_device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(target_device).eval()
    return model, tokenizer, id_to_label, target_device


def split_words(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def chunk_words(words: list[str], tokenizer, max_seq_len: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    current_len = 0

    for word in words:
        token_ids = tokenizer.encode(word)
        token_len = max(1, len(token_ids))

        if current_chunk and current_len + token_len > max_seq_len:
            chunks.append(current_chunk)
            current_chunk = [word]
            current_len = token_len
            continue

        current_chunk.append(word)
        current_len += token_len

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def choose_word_label(subtoken_label_ids: list[int]) -> int:
    counts = Counter(subtoken_label_ids)
    return counts.most_common(1)[0][0]


def predict_word_labels(
    model, tokenizer, words: list[str], device: torch.device
) -> list[int]:
    predicted_labels: list[int] = []

    for chunk in chunk_words(words, tokenizer, model.backbone.max_seq_len):
        input_ids: list[int] = []
        spans: list[tuple[int, int]] = []

        for word in chunk:
            token_ids = tokenizer.encode(word)
            if not token_ids:
                token_ids = [tokenizer.vocab[tokenizer.unk_token]]

            start = len(input_ids)
            input_ids.extend(token_ids)
            end = len(input_ids)
            spans.append((start, end))

        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)[0]
        pred_ids = logits.argmax(dim=-1).tolist()

        for start, end in spans:
            predicted_labels.append(choose_word_label(pred_ids[start:end]))

    return predicted_labels


def decode_entities(
    words: list[str], label_ids: list[int], id_to_label: dict[int, str]
) -> list[dict[str, str | int]]:
    entities: list[dict[str, str | int]] = []
    current_type: str | None = None
    current_words: list[str] = []
    start_index = 0

    for index, (word, label_id) in enumerate(zip(words, label_ids)):
        label = id_to_label[label_id]

        if label == "O":
            if current_type is not None:
                entities.append(
                    {
                        "type": current_type,
                        "text": " ".join(current_words),
                        "start_word": start_index,
                        "end_word": index - 1,
                    }
                )
                current_type = None
                current_words = []
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or current_type != entity_type:
            if current_type is not None:
                entities.append(
                    {
                        "type": current_type,
                        "text": " ".join(current_words),
                        "start_word": start_index,
                        "end_word": index - 1,
                    }
                )
            current_type = entity_type
            current_words = [word]
            start_index = index
        else:
            current_words.append(word)

    if current_type is not None:
        entities.append(
            {
                "type": current_type,
                "text": " ".join(current_words),
                "start_word": start_index,
                "end_word": len(words) - 1,
            }
        )

    return entities


def predict_entities(
    model: TransformerNERHead,
    tokenizer,
    id_to_label: dict[int, str],
    text: str,
    device: torch.device,
) -> tuple[list[str], list[str], list[dict[str, str | int]]]:
    words = split_words(text)
    if not words:
        return [], [], []

    label_ids = predict_word_labels(model, tokenizer, words, device)
    labels = [id_to_label[label_id] for label_id in label_ids]
    entities = decode_entities(words, label_ids, id_to_label)
    return words, labels, entities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detecta entidades nombradas con el mejor modelo NER."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
    )
    source_group.add_argument(
        "--weight-dir",
        type=Path,
        default=None,
        help=(
            "Ruta directa a un fichero .pt o .pth. "
            "Si se usa, la config y las etiquetas se toman de artifacts/ner/best."
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Muestra tambien las etiquetas por palabra",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", type=str, help="Texto sobre el que detectar entidades"
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Fichero de texto sobre el que detectar entidades",
    )
    return parser


def resolve_input_text(text: str | None, input_file: Path | None) -> str:
    if text is not None:
        return text
    if input_file is not None:
        # Accept UTF-8 files with or without BOM to keep CLI behavior stable
        # across editors and Windows/Linux environments.
        return input_file.read_text(encoding="utf-8-sig")
    raise ValueError("Hay que proporcionar --text o --input-file.")


def main() -> None:
    args = build_parser().parse_args()
    model, tokenizer, id_to_label, device = load_ner_model(
        artifacts_dir=args.artifacts_dir,
        device=args.device,
        weight_dir=args.weight_dir,
    )
    input_text = resolve_input_text(args.text, args.input_file)
    words, labels, entities = predict_entities(
        model, tokenizer, id_to_label, input_text, device
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


if __name__ == "__main__":
    main()
