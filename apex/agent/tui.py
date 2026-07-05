"""Interactive terminal chat — the ``apex chat`` command.

Zero-dependency TUI (ANSI + ``input()``) so it works on any VPS over SSH.
Slash commands mirror Hermes, plus APEX-only governance commands.
"""

from __future__ import annotations

import sys
from pathlib import Path

from apex.agent.config import (
    PROVIDER_PRESETS,
    AgentConfig,
    autodetect_local_provider,
    load_config,
    save_config,
)
from apex.agent.llm import LLMClient, LLMError
from apex.agent.loop import AgentLoop
from apex.agent.sessions import SessionStore
from apex.agent.skills import SkillStore
from apex.agent.tools import build_default_tools
from apex.core.types import ThresholdLevel
from apex.system import ApexSystem
from apex import __version__

_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_EMERALD_BRIGHT = "\033[38;5;48m"
_EMERALD = "\033[38;5;42m"
_EMERALD_DEEP = "\033[38;5;29m"
_RESET = "\033[0m"

# Big pixel-style wordmark, shaded top-to-bottom in emerald green.
_LOGO_ROWS = (
    (_EMERALD_BRIGHT, " █████╗ ██████╗ ███████╗██╗  ██╗"),
    (_EMERALD_BRIGHT, "██╔══██╗██╔══██╗██╔════╝╚██╗██╔╝"),
    (_EMERALD, "███████║██████╔╝█████╗   ╚███╔╝ "),
    (_EMERALD, "██╔══██║██╔═══╝ ██╔══╝   ██╔██╗ "),
    (_EMERALD_DEEP, "██║  ██║██║     ███████╗██╔╝ ██╗"),
    (_EMERALD_DEEP, "╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝"),
)

# Display grouping for the banner; unknown tools land in "other".
_TOOL_GROUPS = {
    "search_knowledge": "knowledge",
    "generate_report": "knowledge",
    "remember": "memory",
    "search_memories": "memory",
    "publish_metric": "evolution",
    "run_evolution_cycle": "evolution",
    "run_shell": "shell",
    "save_skill": "skills",
    "use_skill": "skills",
    "search_past_conversations": "sessions",
}

_PANEL_WIDTH = 72

HELP = f"""{_BOLD}Slash commands{_RESET}
  /new              start a new session
  /model [p] [m]    show or switch provider/model ({', '.join(sorted(PROVIDER_PRESETS))})
  /memory [query]   list or search long-term memories
  /skills           list saved skills
  /sessions         list recent sessions
  /cycle            run one knowledge-informed MAPE-K cycle
  /report <query>   generate a knowledge report into outputs/
  /audit            verify the audit chain
  /insight <text>   record an insight into knowledge raw/
  /voice on|off     toggle voice replies (needs [voice] setup)
  /help             show this help
  /quit             exit
"""


def _terminal_approval(name: str, summary: str, level: ThresholdLevel) -> bool:
    print(
        f"{_YELLOW}⚠ approval required ({level.value}): {summary}{_RESET}"
    )
    answer = input("approve? [y/N] ").strip().lower()
    return answer in ("y", "yes")


class ChatShell:
    """Line-based chat shell around the shared :class:`AgentLoop`."""

    def __init__(
        self,
        *,
        knowledge_root: str | Path = ".",
        config: AgentConfig | None = None,
        session_store: SessionStore | None = None,
        system: ApexSystem | None = None,
        client: LLMClient | None = None,
    ) -> None:
        self.config = config or load_config()
        self.knowledge_root = Path(knowledge_root).resolve()
        self.system = system or ApexSystem(knowledge_root=knowledge_root)
        self.sessions = session_store or SessionStore()
        self.skills = SkillStore(audit_ledger=self.system.audit_ledger)
        self.client = client or LLMClient(self.config)
        tools = build_default_tools(
            self.system,
            approval_callback=_terminal_approval,
            skill_store=self.skills,
            session_store=self.sessions,
        )
        self.loop = AgentLoop(
            system=self.system,
            config=self.config,
            client=self.client,
            tools=tools,
            session_store=self.sessions,
            channel="terminal",
            knowledge_root=knowledge_root,
            skill_store=self.skills,
        )
        self._speaker = None

    # ------------------------------------------------------------------

    def handle_slash(self, line: str) -> str | None:
        """Handle a slash command; return output text, or None to exit."""
        parts = line.split()
        command, args = parts[0], parts[1:]

        if command in ("/quit", "/exit"):
            return None
        if command == "/help":
            return HELP
        if command == "/new":
            session = self.loop.new_session()
            return f"new session {session[:8]}"
        if command == "/model":
            if not args:
                return (
                    f"provider={self.config.provider} "
                    f"model={self.config.resolved_model()} "
                    f"base_url={self.config.resolved_base_url()}"
                )
            try:
                self.config.use_provider(args[0], args[1] if len(args) > 1 else None)
            except ValueError as exc:
                return str(exc)
            save_config(self.config)
            return f"switched to {self.config.provider}:{self.config.model}"
        if command == "/memory":
            entries = (
                self.system.search_memories(" ".join(args))
                if args
                else self.system.recall_memories()
            )
            return (
                "\n".join(f"- [{e.subject}] {e.fact}" for e in entries)
                or "no memories stored"
            )
        if command == "/skills":
            skills = self.skills.list()
            return (
                "\n".join(
                    f"- {s.name} (uses: {s.uses}) — {s.description}" for s in skills
                )
                or "no skills saved yet"
            )
        if command == "/sessions":
            return "\n".join(
                f"- {s.session_id[:8]} [{s.channel}] {s.message_count} msgs "
                f"({s.created_at[:19]})"
                for s in self.sessions.sessions(limit=10)
            ) or "no sessions"
        if command == "/cycle":
            report = self.system.run_knowledge_informed_cycle()
            return f"cycle severity={report.overall_severity.value}"
        if command == "/report":
            if not args:
                return "usage: /report <query>"
            return self.system.generate_knowledge_report(" ".join(args))
        if command == "/audit":
            valid, message = self.system.verify_audit_chain()
            return ("✔ " if valid else "✘ ") + message
        if command == "/insight":
            if not args:
                return "usage: /insight <text>"
            path = self.loop.record_insight(" ".join(args))
            return f"recorded to {path}"
        if command == "/voice":
            if args and args[0] in ("on", "off"):
                self.config.voice.enabled = args[0] == "on"
                save_config(self.config)
                if self.config.voice.enabled:
                    self._speaker = None  # re-created lazily
                return f"voice {'enabled' if self.config.voice.enabled else 'disabled'}"
            return f"voice is {'on' if self.config.voice.enabled else 'off'} (usage: /voice on|off)"
        return f"unknown command {command} (try /help)"

    def _maybe_speak(self, text: str) -> None:
        if not self.config.voice.enabled:
            return
        try:
            from apex.voice.tts import TextToSpeech

            if self._speaker is None:
                self._speaker = TextToSpeech(self.config)
            path = self._speaker.synthesize_to_file(text)
            print(f"{_DIM}(voice reply saved to {path}){_RESET}")
        except Exception as exc:  # noqa: BLE001 — voice is best-effort
            print(f"{_DIM}(voice unavailable: {exc}){_RESET}")

    def _grouped_tools(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for name in sorted(self.loop.tools.tools):
            groups.setdefault(_TOOL_GROUPS.get(name, "other"), []).append(name)
        return groups

    def print_banner(self) -> None:
        """Print the Hermes-style startup banner."""
        print()
        for color, row in _LOGO_ROWS:
            print(f"  {color}{_BOLD}{row}{_RESET}")
        print()

        groups = self._grouped_tools()
        tool_count = sum(len(names) for names in groups.values())
        skills = self.skills.list()

        title = f" APEX Life OS v{__version__} · self-evolving agent "
        pad = _PANEL_WIDTH - len(title)
        left = pad // 2
        print(
            f"{_EMERALD_DEEP}┌{'─' * left}{_RESET}{_BOLD}{_EMERALD_BRIGHT}{title}"
            f"{_RESET}{_EMERALD_DEEP}{'─' * (pad - left)}┐{_RESET}"
        )

        def panel(text: str = "") -> None:
            print(f"{_EMERALD_DEEP}│{_RESET} {text}")

        panel(f"{_BOLD}{_EMERALD_BRIGHT}Available Tools{_RESET}")
        for group in sorted(groups):
            names = ", ".join(groups[group])
            panel(f"  {_EMERALD_DEEP}{group}:{_RESET} {names}")
        panel()
        panel(f"{_BOLD}{_EMERALD_BRIGHT}Available Skills{_RESET}")
        if skills:
            for skill in skills:
                panel(f"  {_EMERALD_DEEP}{skill.name}:{_RESET} {skill.description}")
        else:
            panel(f"  {_DIM}none yet — APEX learns skills automatically{_RESET}")
        panel()
        panel(
            f"{_DIM}model:{_RESET} {_EMERALD}"
            f"{self.config.provider}:{self.config.resolved_model()}{_RESET}"
        )
        panel(f"{_DIM}root:  {self.knowledge_root}{_RESET}")
        panel(f"{_DIM}session: {self.loop.session_id}{_RESET}")
        panel()
        panel(
            f"{_DIM}{tool_count} tools · {len(skills)} skills · "
            f"/help for commands{_RESET}"
        )
        print(f"{_EMERALD_DEEP}└{'─' * _PANEL_WIDTH}┘{_RESET}")
        print()
        print("Welcome to APEX! Type your message or /help for commands.")
        print(
            f"{_DIM}✦ Tip: APEX grows with you — /insight records ideas, "
            f"/cycle evolves the system.{_RESET}"
        )

    def run(self) -> int:
        detected = autodetect_local_provider(self.config)
        if detected:
            save_config(self.config)
        self.print_banner()
        if detected:
            print(
                f"{_GREEN}✔ detected local Ollama server — using "
                f"{detected} (switch anytime with /model){_RESET}"
            )
        elif not self.config.api_key and self.config.provider not in (
            "ollama",
            "lmstudio",
            "vllm",
        ):
            print(
                f"{_YELLOW}warning: APEX_API_KEY is not set and no local "
                f"Ollama server was found — chat will fail until you export "
                f"a key or start Ollama.{_RESET}"
            )
        while True:
            try:
                line = input(f"{_CYAN}you ▸ {_RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line.startswith("/"):
                output = self.handle_slash(line)
                if output is None:
                    return 0
                print(output)
                continue
            try:
                turn = self.loop.send(line)
            except LLMError as exc:
                print(f"{_YELLOW}llm error: {exc}{_RESET}")
                continue
            for name in turn.tool_calls:
                print(f"{_DIM}⚙ {name}{_RESET}")
            print(f"{_GREEN}apex ▸ {_RESET}{turn.reply}")
            self._maybe_speak(turn.reply)


def run_chat(knowledge_root: str | Path = ".") -> int:
    """Entry point used by ``apex chat``."""
    try:
        return ChatShell(knowledge_root=knowledge_root).run()
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(run_chat())
