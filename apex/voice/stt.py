"""Speech-to-text — transcribe voice memos via a Whisper-compatible API.

Sent audio (e.g. Telegram voice notes) is transcribed through the
configured provider's ``/audio/transcriptions`` endpoint (OpenAI-compatible),
so it works against OpenAI, Groq, or a local server. No local ML deps.
"""

from __future__ import annotations

import json
import secrets
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from apex.agent.config import AgentConfig

Transport = Callable[[str, dict[str, str], bytes], dict[str, Any]]


class VoiceError(RuntimeError):
    """Raised when a voice endpoint cannot be reached or errors."""


def _urllib_transport(
    url: str, headers: dict[str, str], body: bytes
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise VoiceError(f"STT endpoint returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise VoiceError(f"Cannot reach STT endpoint {url}: {exc.reason}") from exc


class SpeechToText:
    """Whisper-API-compatible transcription client."""

    def __init__(
        self, config: AgentConfig, *, transport: Transport | None = None
    ) -> None:
        self.config = config
        self._transport = transport or _urllib_transport

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe an audio file and return the recognized text."""
        path = Path(audio_path)
        boundary = "apexvoice" + secrets.token_hex(8)
        model = self.config.voice.stt_model
        body = b""
        body += (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model"\r\n\r\n'
            f"{model}\r\n"
        ).encode()
        body += (
            f"--{boundary}\r\n"
            "Content-Disposition: form-data; "
            f'name="file"; filename="{path.name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        body += path.read_bytes()
        body += f"\r\n--{boundary}--\r\n".encode()

        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        url = self.config.resolved_base_url().rstrip("/") + "/audio/transcriptions"
        data = self._transport(url, headers, body)
        return str(data.get("text", ""))
