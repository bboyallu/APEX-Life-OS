"""apex.voice — optional voice interaction (off by default).

API-based speech-to-text and text-to-speech, VPS-friendly: no audio
hardware required on the server. Enable with ``/voice on`` in chat or
``voice.enabled: true`` in ``~/.apex/config.json``.
"""

from apex.voice.stt import SpeechToText
from apex.voice.tts import TextToSpeech

__all__ = ["SpeechToText", "TextToSpeech"]
