import json
from collections import Counter
from pathlib import Path
from itertools import pairwise
from typing import Iterable


END_OF_WORD = "</w>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

class MiniBPETokenizer:
    """Versión sencilla de un tokenizer BPE para uso didáctico."""

    def __init__(self) -> None:
        self.special_tokens = [PAD_TOKEN, UNK_TOKEN]
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._reset()

    def _reset(self) -> None:
        self.vocab = {token: index for index, token in enumerate(self.special_tokens)}
        self.id_to_token = {index: token for token, index in self.vocab.items()}

    def _split_text(self, text: str) -> list[str]:
        return text.lower().split()

    def _word_to_symbols(self, word: str) -> list[str]:
        return list(word) + [END_OF_WORD]

    def _merge_word(self, symbols: list[str], pair: tuple[str, str]) -> list[str]:
        merged: list[str] = []
        i = 0

        while i < len(symbols):
            current_pair = tuple(symbols[i : i + 2])
            if len(current_pair) == 2 and current_pair == pair:
                merged.append("".join(current_pair))
                i += 2
                continue

            merged.append(symbols[i])
            i += 1

        return merged

    def _count_pairs(self, word_freqs: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for word, freq in word_freqs.items():
            for pair in pairwise(word):
                pair_counts[pair] += freq
        return pair_counts

    def train(self, text: str, vocab_size: int = 256) -> None:
        words = self._split_text(text)
        if not words:
            raise ValueError("No hay texto suficiente para entrenar el tokenizer.")

        word_freqs: Counter[tuple[str, ...]] = Counter(
            tuple(self._word_to_symbols(word)) for word in words
        )

        self._reset()
        alphabet = sorted({symbol for word in word_freqs for symbol in word})
        for symbol in alphabet:
            if symbol in self.vocab:
                continue

            token_id = len(self.vocab)
            self.vocab[symbol] = token_id
            self.id_to_token[token_id] = symbol

        self.merges = []
        while len(self.vocab) < vocab_size:
            pair_counts = self._count_pairs(word_freqs)
            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            self.merges.append(best_pair)
            word_freqs = Counter(
                {tuple(self._merge_word(list(word), best_pair)): freq for word, freq in word_freqs.items()}
            )

            merged_token = "".join(best_pair)
            if merged_token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[merged_token] = token_id
                self.id_to_token[token_id] = merged_token

    def _apply_merges(self, word: str) -> list[str]:
        symbols = self._word_to_symbols(word)
        for pair in self.merges:
            symbols = self._merge_word(symbols, pair)
        return symbols

    def encode(self, text: str) -> list[int]:
        unk_id = self.vocab[self.unk_token]
        return [
            self.vocab.get(token, unk_id)
            for word in self._split_text(text)
            for token in self._apply_merges(word)
        ]

    def decode(self, token_ids: Iterable[int]) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), self.unk_token)
            if token in self.special_tokens:
                continue

            if token.endswith(END_OF_WORD):
                pieces.append(token[: -len(END_OF_WORD)])
                pieces.append(" ")
            else:
                pieces.append(token)

        return "".join(pieces).strip()

    def save(self, output_path: str | Path) -> None:
        payload = {
            "merges": self.merges,
            "vocab": self.vocab,
        }
        Path(output_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "MiniBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.merges = [tuple(pair) for pair in payload["merges"]]
        tokenizer.vocab = {token: int(idx) for token, idx in payload["vocab"].items()}
        tokenizer.id_to_token = {idx: token for token, idx in tokenizer.vocab.items()}
        return tokenizer
