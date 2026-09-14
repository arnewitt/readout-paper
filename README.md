# readout-paper

Local text-to-speech using [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
via the `kokoro` package's `KPipeline`. No network calls after the model is cached.

## Usage

```bash
uv run readout-paper "Hello world" -o out.wav
uv run readout-paper -i paper.txt -o paper.wav --play
cat paper.txt | uv run readout-paper -o paper.wav
uv run readout-paper "Bonjour tout le monde" -m ff_siwis -o fr.wav
uv run readout-paper --list-voices
```

Options: `-m/--voice` (default `af_heart`), `-l/--lang`, `-s/--speed`,
`--device {cpu,mps,cuda}`, `--play`, `-o/--output-file` (default `out.wav`).
The language is inferred from the first letter of the voice name.

Output is 24 kHz 16-bit mono WAV.

## Web UI

```bash
uv run readout-paper --ui          # opens http://127.0.0.1:8765
uv run readout-paper --ui --port 9000
uv run readout-paper --ui --host 0.0.0.0   # reachable from other machines
```

`--host` defaults to `127.0.0.1`; the browser is opened only for a loopback host.

Paste text, pick a voice and speed, and press *Read it aloud* (or Cmd/Ctrl-Enter).
The result plays in the page on a waveform you can click to seek, and downloads
as a WAV. It is a stdlib-only server bound to localhost -- no extra dependencies,
no network calls.

## As a library

```python
from readout_paper import synth_to_wav

synth_to_wav("Some text", "out.wav", voice="af_heart", speed=1.0)
```

The pipeline is cached per language, so repeated calls in one process only pay
the model load once.

## Docker

```bash
docker compose up --build          # web UI on http://127.0.0.1:8765
```

Or without compose:

```bash
docker build -t readout-paper .
docker run --rm -p 127.0.0.1:8765:8765 -v hf-cache:/home/app/.cache/huggingface readout-paper
```

The image entrypoint is `readout-paper` itself, so the CLI works too — mount a
directory on `/data`, which is the working directory:

```bash
docker run --rm -v hf-cache:/home/app/.cache/huggingface -v "$PWD:/data" \
  readout-paper -i input/paper.txt -o paper.wav
```

Roughly 2 GB, CPU-only. `--play` does nothing in the container (no audio device)
and `--device mps` is unavailable; write a WAV to a mount instead.

Keep the `hf-cache` volume: the model weights are downloaded on first run and
would otherwise be fetched again by every fresh container.

## Notes

- On Linux, `torch` comes from the PyTorch CPU index, so the container skips
  the CUDA wheels PyPI serves there (~8 GB of nvidia packages). macOS still
  installs the PyPI build, which is the one that supports `--device mps`.
- Requires Python 3.12 — `kokoro` and `misaki` both declare `requires-python <3.13`.
- `espeak-ng` ships inside the `espeakng-loader` wheel; no system install needed.
- `en_core_web_sm` is a declared dependency so `misaki` never shells out to `pip`
  to download it at runtime.
- First run downloads the model weights and voice pack into `~/.cache/huggingface`.
