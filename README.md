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

## As a library

```python
from readout_paper import synth_to_wav

synth_to_wav("Some text", "out.wav", voice="af_heart", speed=1.0)
```

The pipeline is cached per language, so repeated calls in one process only pay
the model load once.

## Notes

- Requires Python 3.12 — `kokoro` and `misaki` both declare `requires-python <3.13`.
- `espeak-ng` ships inside the `espeakng-loader` wheel; no system install needed.
- `en_core_web_sm` is a declared dependency so `misaki` never shells out to `pip`
  to download it at runtime.
- First run downloads the model weights and voice pack into `~/.cache/huggingface`.
