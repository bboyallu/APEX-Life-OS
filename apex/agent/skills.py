"""Skills — reusable procedures the agent learns from experience.

Mirrors the Hermes closed learning loop, with an APEX upgrade: every skill
creation, update, use, and deletion is recorded on the tamper-evident audit
ledger. Skills are stored as human-readable markdown in ``~/.apex/skills/``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from apex.agent.config import apex_home
from apex.governance.audit import AuditLedger

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower()).strip("-") or "skill"


class Skill(BaseModel):
    name: str
    description: str = ""
    steps: list[str] = Field(default_factory=list)
    uses: int = 0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def slug(self) -> str:
        return _slugify(self.name)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.name}",
            "",
            f"- created: {self.created_at}",
            f"- uses: {self.uses}",
            "",
            self.description,
            "",
            "## Steps",
            "",
        ]
        lines += [f"{i}. {step}" for i, step in enumerate(self.steps, start=1)]
        return "\n".join(lines) + "\n"

    @classmethod
    def from_markdown(cls, text: str) -> "Skill":
        name = "skill"
        description_lines: list[str] = []
        steps: list[str] = []
        uses = 0
        created_at = datetime.now(timezone.utc).isoformat()
        in_steps = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and name == "skill":
                name = stripped[2:].strip()
            elif stripped.startswith("- created:"):
                created_at = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- uses:"):
                try:
                    uses = int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    uses = 0
            elif stripped.startswith("## Steps"):
                in_steps = True
            elif in_steps and re.match(r"^\d+\.", stripped):
                steps.append(re.sub(r"^\d+\.\s*", "", stripped))
            elif not in_steps and stripped and not stripped.startswith("#"):
                description_lines.append(stripped)
        return cls(
            name=name,
            description=" ".join(description_lines),
            steps=steps,
            uses=uses,
            created_at=created_at,
        )


class SkillStore:
    """File-backed skill library with mandatory audit logging."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        audit_ledger: AuditLedger | None = None,
    ) -> None:
        self.root = Path(root) if root else apex_home() / "skills"
        self.root.mkdir(parents=True, exist_ok=True)
        self.audit_ledger = audit_ledger

    def _audit(self, event: str, payload: dict) -> None:
        if self.audit_ledger is not None:
            self.audit_ledger.append(event, actor="skill_store", payload=payload)

    def _path(self, slug: str) -> Path:
        return self.root / f"{slug}.md"

    def save(self, skill: Skill) -> Path:
        path = self._path(skill.slug)
        existed = path.exists()
        path.write_text(skill.to_markdown(), encoding="utf-8")
        self._audit(
            "skill_updated" if existed else "skill_created",
            {"skill": skill.slug, "steps": len(skill.steps)},
        )
        return path

    def get(self, name: str) -> Skill | None:
        path = self._path(_slugify(name))
        if not path.exists():
            return None
        return Skill.from_markdown(path.read_text(encoding="utf-8"))

    def use(self, name: str) -> Skill | None:
        """Fetch a skill and increment its use counter (audit-logged)."""
        skill = self.get(name)
        if skill is None:
            return None
        skill.uses += 1
        self._path(skill.slug).write_text(skill.to_markdown(), encoding="utf-8")
        self._audit("skill_used", {"skill": skill.slug, "uses": skill.uses})
        return skill

    def list(self) -> list[Skill]:
        skills = [
            Skill.from_markdown(p.read_text(encoding="utf-8"))
            for p in sorted(self.root.glob("*.md"))
        ]
        return skills

    def delete(self, name: str) -> bool:
        path = self._path(_slugify(name))
        if not path.exists():
            return False
        path.unlink()
        self._audit("skill_deleted", {"skill": _slugify(name)})
        return True
