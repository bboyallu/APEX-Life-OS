"""Agent configuration — persisted at ``~/.apex/config.json``.

API keys are read from environment variables (or a ``.env`` file you manage
yourself) and are **never** written to the config file.

Environment variables:

* ``APEX_API_KEY``    — API key for the configured provider.
* ``APEX_BASE_URL``   — override the provider base URL.
* ``APEX_MODEL``      — override the model name.
* ``APEX_HOME``       — override the ``~/.apex`` state directory.
* ``TELEGRAM_BOT_TOKEN`` — Telegram gateway bot token.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

#: Built-in provider presets. All speak the OpenAI-compatible
#: ``/chat/completions`` API, so one client covers them all.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-3.5-sonnet",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.2"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "model": "local-model"},
    "vllm": {"base_url": "http://localhost:8000/v1", "model": "local-model"},
}


def apex_home() -> Path:
    """Return the APEX state directory (``~/.apex`` by default)."""
    return Path(os.environ.get("APEX_HOME", str(Path.home() / ".apex")))


class VoiceConfig(BaseModel):
    """Optional voice interaction settings (off by default)."""

    enabled: bool = False
    stt_model: str = "whisper-1"
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"


class AgentConfig(BaseModel):
    """Persistent agent settings (no secrets)."""

    provider: str = "openai"
    base_url: str = PROVIDER_PRESETS["openai"]["base_url"]
    model: str = PROVIDER_PRESETS["openai"]["model"]
    temperature: float = 0.7
    max_tool_rounds: int = 8
    memory_nudge_every: int = 5
    voice: VoiceConfig = Field(default_factory=VoiceConfig)

    @property
    def api_key(self) -> str:
        """API key from the environment — never persisted."""
        return os.environ.get("APEX_API_KEY", "")

    def resolved_base_url(self) -> str:
        return os.environ.get("APEX_BASE_URL", self.base_url)

    def resolved_model(self) -> str:
        return os.environ.get("APEX_MODEL", self.model)

    def use_provider(self, provider: str, model: str | None = None) -> None:
        """Switch to a preset provider (and optionally a specific model)."""
        preset = PROVIDER_PRESETS.get(provider)
        if preset is None:
            raise ValueError(
                f"Unknown provider {provider!r}. "
                f"Choose from: {', '.join(sorted(PROVIDER_PRESETS))}"
            )
        self.provider = provider
        self.base_url = preset["base_url"]
        self.model = model or preset["model"]


def detect_ollama_models(
    base_url: str | None = None,
    *,
    fetch: Callable[[str, float], str] | None = None,
    timeout: float = 0.8,
) -> list[str]:
    """Return installed model names if a local Ollama server is online.

    Probes the Ollama native ``/api/tags`` endpoint (stdlib only). Returns
    an empty list when the server is unreachable — never raises, so callers
    can use it as a cheap availability check at startup.
    """
    root = (base_url or PROVIDER_PRESETS["ollama"]["base_url"]).rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    url = root + "/api/tags"
    fetch = fetch or _urlopen_text
    try:
        data = json.loads(fetch(url, timeout))
        return [
            model["name"]
            for model in data.get("models", [])
            if isinstance(model, dict) and model.get("name")
        ]
    except (OSError, ValueError, KeyError, TypeError):
        # Detection is best-effort: offline server, bad JSON, or an
        # unexpected payload shape all mean "no local models".
        return []


def _urlopen_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def autodetect_local_provider(config: AgentConfig) -> str | None:
    """Switch *config* to Ollama when it is running locally.

    Mutates *config* in place (provider, base_url, model) when a switch
    happens; callers should persist it with :func:`save_config`. Only
    applies when the user has no API key exported and is still on a
    remote provider (i.e. chat would fail anyway). Prefers a model that is
    actually installed. Returns the detected model name, or ``None`` when
    no switch happened.
    """
    if config.api_key or config.provider in ("ollama", "lmstudio", "vllm"):
        return None
    models = detect_ollama_models()
    if not models:
        return None
    preset_model = PROVIDER_PRESETS["ollama"]["model"]
    chosen = next(
        (m for m in models if m.split(":")[0] == preset_model), models[0]
    )
    config.use_provider("ollama", chosen)
    return chosen


def config_path(home: Path | None = None) -> Path:
    return (home or apex_home()) / "config.json"


def load_config(home: Path | None = None) -> AgentConfig:
    """Load config from ``~/.apex/config.json`` (defaults if missing)."""
    path = config_path(home)
    if path.exists():
        return AgentConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return AgentConfig()


def save_config(config: AgentConfig, home: Path | None = None) -> Path:
    """Persist config (secrets excluded by construction)."""
    path = config_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
