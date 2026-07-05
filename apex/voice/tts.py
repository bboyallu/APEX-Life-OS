"""Text-to-speech — spoken replies via an OpenAI-compatible ``/audio/speech``.

Audio is written to a file (playable locally or sent back through the
messaging gateway) — no audio hardware needed on a VPS.
"""

from __future__ import annotations

import json
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from apex.agent.config import AgentConfig
from apex.voice.stt import VoiceError

Transport = Callable[[str, dict[str, str], bytes], bytes]


def _urllib_transport(url: str, headers: dict[str, str], body: bytes) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise VoiceError(f"TTS endpoint returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise VoiceError(f"Cannot reach TTS endpoint {url}: {exc.reason}") from exc


class TextToSpeech:
    """OpenAI-compatible speech synthesis client."""

    def __init__(
        self, config: AgentConfig, *, transport: Transport | None = None
    ) -> None:
        self.config = config
        self._transport = transport or _urllib_transport

    def synthesize_to_file(
        self, text: str, output_path: str | Path | None = None
    ) -> Path:
        """Synthesize ``text`` to an mp3 file and return its path."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        payload = json.dumps(
            {
                "model": self.config.voice.tts_model,
                "voice": self.config.voice.tts_voice,
                "input": text[:4000],
            }
        ).encode("utf-8")
        url = self.config.resolved_base_url().rstrip("/") + "/audio/speech"
        audio = self._transport(url, headers, payload)
        if output_path is None:
            handle = tempfile.NamedTemporaryFile(
                prefix="apex-voice-", suffix=".mp3", delete=False
            )
            output_path = handle.name
            handle.close()
        path = Path(output_path)
        path.write_bytes(audio)
        return path
