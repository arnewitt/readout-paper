"""A local web UI for readout-paper.

Stdlib only: one page, one endpoint. POST /synth returns a WAV the browser
plays and can download.
"""

from __future__ import annotations

import json
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .tts import DEFAULT_VOICE, LANG_NAMES, VOICES, synth_to_wav

PAGE = Path(__file__).with_name("index.html")
MAX_TEXT = 200_000

# KPipeline is not thread-safe and the model is big; one synthesis at a time.
_lock = threading.Lock()


def render_page() -> bytes:
    groups = [
        {"name": LANG_NAMES[code], "voices": list(voices)}
        for code, voices in VOICES.items()
    ]
    html = PAGE.read_text()
    html = html.replace("__VOICES__", json.dumps(groups))
    html = html.replace("__DEFAULT_VOICE__", json.dumps(DEFAULT_VOICE))
    return html.encode()


def synth_bytes(text: str, voice: str, speed: float, device: str | None) -> bytes:
    """Synthesize to a temporary WAV and return its bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.wav"
        with _lock:
            synth_to_wav(
                text,
                out,
                voice=voice,
                lang_code=voice[0],
                speed=speed,
                device=device,
            )
        return out.read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version = "readout-paper"
    device: str | None = None

    def log_message(self, fmt: str, *args) -> None:  # quieter than the default
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        self._send(code, json.dumps({"error": message}).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, render_page(), "text/html; charset=utf-8")
        else:
            self._send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path != "/synth":
            self._send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = str(payload.get("text", ""))
            voice = str(payload.get("voice", DEFAULT_VOICE))
            speed = float(payload.get("speed", 1.0))
        except (ValueError, TypeError):
            self._send_error(400, "Could not read that request.")
            return

        if not text.strip():
            self._send_error(400, "Nothing to read yet — add some text above.")
            return
        if len(text) > MAX_TEXT:
            self._send_error(400, f"Text is longer than {MAX_TEXT:,} characters.")
            return
        if voice[:1] not in LANG_NAMES:
            self._send_error(400, f"Unknown voice: {voice}")
            return
        if not 0.5 <= speed <= 2.0:
            self._send_error(400, "Speed must be between 0.5 and 2.0.")
            return

        try:
            wav = synth_bytes(text, voice, speed, self.device)
        except RuntimeError as exc:
            self._send_error(400, str(exc))
            return
        except Exception as exc:  # a bad voice name only fails inside kokoro
            self._send_error(500, f"Synthesis failed: {exc}")
            return

        self._send(200, wav, "audio/wav")


def serve(host: str = "127.0.0.1", port: int = 8765, device: str | None = None,
          open_browser: bool = True) -> None:
    """Serve the UI until interrupted."""
    Handler.device = device
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{httpd.server_port}"
    print(f"readout-paper UI on {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
