from __future__ import annotations

from pathlib import Path
import math
import sys

import typer

DEFAULT_OFFSET = 77
UPPERCASE_MARK = 0x82
ACUTE_MARK = 0x7F
DIAERESIS_MARK = 0x80
ENYE_MARK = 0x81

DIGIT_TO_PLNCG26 = {str(i): 0x89 + i for i in range(10)}
PLNCG26_TO_DIGIT = {value: key for key, value in DIGIT_TO_PLNCG26.items()}

PUNCTUATION_TO_PLNCG26 = {
    ".": 0x93,
    ",": 0x94,
    ";": 0x95,
    ":": 0x96,
    "?": 0x97,
    "!": 0x98,
    "¿": 0x99,
    "¡": 0x9A,
    '"': 0x9B,
    "'": 0x9C,
    "(": 0x9E,
    ")": 0x9F,
}
EXTRA_PUNCTUATION = (
    "[",
    "]",
    "{",
    "}",
    "<",
    ">",
    "-",
    "_",
    "/",
    "\\",
    "@",
    "#",
    "$",
    "%",
    "&",
    "*",
    "+",
    "=",
    "|",
    "~",
    "^",
    "`",
    "«",
    "»",
    "“",
    "”",
    "…",
    "·",
)
for codepoint, symbol in enumerate(EXTRA_PUNCTUATION, start=0xA0):
    PUNCTUATION_TO_PLNCG26[symbol] = codepoint

PLNCG26_TO_PUNCTUATION = {value: key for key, value in PUNCTUATION_TO_PLNCG26.items()}
PLNCG26_TO_PUNCTUATION[0x9D] = '"'

ACCENTED_TO_PLNCG26 = {
    "á": (ord("a"), ACUTE_MARK),
    "é": (ord("e"), ACUTE_MARK),
    "í": (ord("i"), ACUTE_MARK),
    "ó": (ord("o"), ACUTE_MARK),
    "ú": (ord("u"), ACUTE_MARK),
    "ü": (ord("u"), DIAERESIS_MARK),
    "ñ": (ord("n"), ENYE_MARK),
    "Á": (ord("a"), UPPERCASE_MARK, ACUTE_MARK),
    "É": (ord("e"), UPPERCASE_MARK, ACUTE_MARK),
    "Í": (ord("i"), UPPERCASE_MARK, ACUTE_MARK),
    "Ó": (ord("o"), UPPERCASE_MARK, ACUTE_MARK),
    "Ú": (ord("u"), UPPERCASE_MARK, ACUTE_MARK),
    "Ü": (ord("u"), UPPERCASE_MARK, DIAERESIS_MARK),
    "Ñ": (ord("n"), UPPERCASE_MARK, ENYE_MARK),
}

ACUTE_DECODE = {
    "a": "á",
    "e": "é",
    "i": "í",
    "o": "ó",
    "u": "ú",
    "A": "Á",
    "E": "É",
    "I": "Í",
    "O": "Ó",
    "U": "Ú",
}
DIAERESIS_DECODE = {"u": "ü", "U": "Ü"}
ENYE_DECODE = {"n": "ñ", "N": "Ñ"}

COMMON_WORDS = (" de ", " la ", " que ", " el ", " y ", " en ")

app = typer.Typer(add_completion=False, help="Conversor UTF-8 <-> PLNCG26")


def utf8_to_plncg26(text: str) -> bytes:
    out = bytearray()
    for ch in text:
        if ch in ACCENTED_TO_PLNCG26:
            out.extend(ACCENTED_TO_PLNCG26[ch])
            continue
        if ch in DIGIT_TO_PLNCG26:
            out.append(DIGIT_TO_PLNCG26[ch])
            continue
        if ch in PUNCTUATION_TO_PLNCG26:
            out.append(PUNCTUATION_TO_PLNCG26[ch])
            continue
        if "A" <= ch <= "Z":
            out.extend((ord(ch.lower()), UPPERCASE_MARK))
            continue
        if "a" <= ch <= "z":
            out.append(ord(ch))
            continue
        if ch == " ":
            out.append(ord("X"))
            continue
        if ch in "\n\r\t":
            out.append(ord(ch))
            continue
        if 0x7F <= ord(ch) <= 0xBF:
            out.append(ord(ch))
            continue
        raise ValueError(f"Caracter no soportado por PLNCG26: {ch!r} (U+{ord(ch):04X})")
    return bytes(out)


def decode_letter_token(data: bytes, start: int) -> tuple[str, int]:
    base = chr(data[start])
    idx = start + 1
    uppercase = False

    if idx < len(data) and data[idx] == UPPERCASE_MARK:
        uppercase = True
        idx += 1

    if idx < len(data):
        mark = data[idx]
        if mark == ACUTE_MARK and (base.upper() if uppercase else base) in ACUTE_DECODE:
            token = base.upper() if uppercase else base
            return ACUTE_DECODE[token], idx + 1
        if (
            mark == DIAERESIS_MARK
            and (base.upper() if uppercase else base) in DIAERESIS_DECODE
        ):
            token = base.upper() if uppercase else base
            return DIAERESIS_DECODE[token], idx + 1
        if mark == ENYE_MARK and (base.upper() if uppercase else base) in ENYE_DECODE:
            token = base.upper() if uppercase else base
            return ENYE_DECODE[token], idx + 1

    if uppercase:
        return base.upper(), idx
    return base, idx


def plncg26_to_utf8(data: bytes) -> str:
    out: list[str] = []
    idx = 0

    while idx < len(data):
        byte = data[idx]

        if byte in (ord("X"), ord("W")):
            out.append(" ")
            idx += 1
            continue
        if byte in (ord("\n"), ord("\r"), ord("\t")):
            out.append(chr(byte))
            idx += 1
            continue
        if byte in PLNCG26_TO_DIGIT:
            out.append(PLNCG26_TO_DIGIT[byte])
            idx += 1
            continue
        if byte in PLNCG26_TO_PUNCTUATION:
            out.append(PLNCG26_TO_PUNCTUATION[byte])
            idx += 1
            continue
        if ord("a") <= byte <= ord("z"):
            decoded, next_idx = decode_letter_token(data, idx)
            out.append(decoded)
            idx = next_idx
            continue
        if byte in (UPPERCASE_MARK, ACUTE_MARK, DIAERESIS_MARK, ENYE_MARK):
            idx += 1
            continue
        if 0x7F <= byte <= 0xBF:
            out.append(chr(byte))
            idx += 1
            continue
        if 32 <= byte <= 126:
            out.append(chr(byte))
            idx += 1
            continue
        raise ValueError(f"Byte PLNCG26 no reconocido: 0x{byte:02X}")

    return "".join(out)


def encode_bytes(plain_text: str, offset: int) -> bytes:
    plncg26 = utf8_to_plncg26(plain_text)
    return bytes((byte - offset) % 256 for byte in plncg26)


def decode_bytes(cipher_data: bytes, offset: int) -> str:
    plncg26 = bytes((byte + offset) % 256 for byte in cipher_data)
    return plncg26_to_utf8(plncg26)


def score_plain_text(text: str) -> float:
    if not text:
        return float("-inf")
    length = len(text)
    valid = sum(ch.isprintable() or ch in "\n\r\t" for ch in text) / length
    alpha = sum(ch.isalpha() for ch in text) / length
    lower = f" {text.lower()} "
    common = sum(lower.count(token) for token in COMMON_WORDS)
    weird = sum(ord(ch) < 9 or ord(ch) == 127 for ch in text)
    return (valid * 100.0) + (alpha * 20.0) + (common * 2.0) - (weird * 10.0)


def detect_offset(data: bytes) -> tuple[int, float]:
    scored: list[tuple[int, float]] = []
    for offset in range(256):
        try:
            text = decode_bytes(data, offset)
        except ValueError:
            continue
        scored.append((offset, score_plain_text(text)))
    if not scored:
        raise ValueError("No se pudo estimar un offset valido.")
    scored.sort(key=lambda item: item[1], reverse=True)
    if len(scored) == 1:
        return scored[0][0], 1.0

    best_offset, best_score = scored[0]
    second_score = scored[1][1]
    probability = 1.0 / (1.0 + math.exp((second_score - best_score) / 10.0))
    return best_offset, probability


@app.command("encode")
def encode_cmd(
    input_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    offset: int = typer.Option(DEFAULT_OFFSET, "--offset", min=0, max=255),
) -> None:
    """Pasa de UTF-8 a PLNCG26 (bytes), escribiendo a stdout."""
    text = input_file.read_text(encoding="utf-8")
    cipher_data = encode_bytes(text, offset)
    sys.stdout.buffer.write(cipher_data)
    sys.stdout.buffer.flush()


@app.command("decode")
def decode_cmd(
    input_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    offset: int = typer.Option(DEFAULT_OFFSET, "--offset", min=0, max=255),
) -> None:
    """Pasa de PLNCG26 (bytes) a UTF-8, escribiendo a stdout."""
    plain_text = decode_bytes(input_file.read_bytes(), offset)
    sys.stdout.write(plain_text)
    sys.stdout.flush()


@app.command("detect")
def detect_cmd(
    input_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Opcional: estima si un fichero parece PLNCG26 y el offset mas probable."""
    offset, probability = detect_offset(input_file.read_bytes())
    typer.echo(f"offset={offset} probabilidad={probability:.3f}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
