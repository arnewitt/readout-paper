"""Command line interface for readout-paper."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .tts import (
    DEFAULT_VOICE,
    LANG_NAMES,
    VOICES,
    synth_to_wav,
    wav_duration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="readout-paper",
        description="Read text aloud with a local Kokoro-82M model.",
    )
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument(
        "-i", "--input-file", type=Path, help="Read text from this file instead"
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=Path("out.wav"),
        help="Where to write the WAV (default: out.wav)",
    )
    parser.add_argument(
        "-m", "--voice", default=DEFAULT_VOICE, help=f"Voice (default: {DEFAULT_VOICE})"
    )
    parser.add_argument(
        "-l",
        "--lang",
        choices=sorted(LANG_NAMES),
        help="Language code (default: first letter of the voice)",
    )
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        help="Torch device (default: auto -- CPU on Apple silicon)",
    )
    parser.add_argument(
        "--play", action="store_true", help="Play the result when it is done"
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="List known voices and exit"
    )
    return parser


def list_voices() -> None:
    for code, voices in VOICES.items():
        print(f"{code}  {LANG_NAMES[code]}")
        for voice in voices:
            print(f"      {voice}")


def resolve_text(args: argparse.Namespace) -> str:
    if args.text is not None and args.input_file is not None:
        raise SystemExit("error: pass either TEXT or --input-file, not both")
    if args.text is not None:
        text = args.text
    elif args.input_file is not None:
        if not args.input_file.is_file():
            raise SystemExit(f"error: no such file: {args.input_file}")
        text = args.input_file.read_text()
    else:
        if sys.stdin.isatty():
            print("Reading text from stdin, press Ctrl-D when done.", file=sys.stderr)
        text = sys.stdin.read()
    if not text.strip():
        raise SystemExit("error: no text to read out")
    return text


def main() -> None:
    args = build_parser().parse_args()

    if args.list_voices:
        list_voices()
        return

    text = resolve_text(args)

    lang = args.lang or args.voice[0]
    if lang not in LANG_NAMES:
        raise SystemExit(
            f"error: cannot infer a language from voice {args.voice!r}; pass --lang"
        )
    if args.voice[0] != lang:
        print(
            f"warning: voice {args.voice} does not look like a {LANG_NAMES[lang]} voice",
            file=sys.stderr,
        )
    if args.output_file.suffix.lower() != ".wav":
        print(
            f"warning: {args.output_file} will be written as a WAV regardless of its suffix",
            file=sys.stderr,
        )

    try:
        out = synth_to_wav(
            text,
            args.output_file,
            voice=args.voice,
            lang_code=lang,
            speed=args.speed,
            device=args.device,
        )
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}")

    print(f"Wrote {out} ({wav_duration(out):.1f}s)")

    if args.play:
        player = shutil.which("afplay") or shutil.which("aplay")
        if player is None:
            print("warning: no audio player found (afplay/aplay)", file=sys.stderr)
        else:
            subprocess.run([player, str(out)])


if __name__ == "__main__":
    main()
