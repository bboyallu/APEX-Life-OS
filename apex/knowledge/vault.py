"""KnowledgeVault — file-backed personal knowledge base built into APEX.

Implements the three-folder architecture from ``KNOWLEDGE_BASE.md``:

* ``raw/``     — the "junk drawer"; immutable input, never modified here.
* ``wiki/``    — compiled, cross-referenced articles; written only by APEX.
* ``outputs/`` — append-only on-demand reports, briefings, and answers.

Processing is deterministic and idempotent: re-ingesting an unchanged raw
file is a no-op, while a changed file updates (rather than duplicates) its
wiki section.  Cross-references between articles are derived from shared
significant keywords.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

#: Raw-file extensions treated as ingestible text.
_TEXT_EXTENSIONS = {".md", ".txt", ".text", ".markdown", ".rst", ".csv", ".log"}

#: Words too common to be meaningful cross-reference keywords.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "not", "of", "on", "or",
    "that", "the", "this", "to", "was", "were", "what", "when", "which",
    "will", "with", "you", "your",
}

_TOPIC_RE = re.compile(r"^topic\s*:\s*(?P<topic>.+)$", re.IGNORECASE | re.MULTILINE)
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{3,}")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def _title_from_slug(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


class WikiArticle(BaseModel):
    """A compiled thematic article in the wiki."""

    slug: str
    title: str
    sources: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)

    @property
    def filename(self) -> str:
        return f"{self.slug}.md"


class IngestReport(BaseModel):
    """Result of one :meth:`KnowledgeVault.process_raw` run."""

    ingested: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    articles: list[str] = Field(default_factory=list)


class KnowledgeVault:
    """File-backed knowledge base following the ``KNOWLEDGE_BASE.md`` schema.

    Parameters
    ----------
    root:
        Directory containing (or that will contain) the ``raw/``, ``wiki/``
        and ``outputs/`` folders.  Defaults to the current working directory.
    """

    def __init__(self, root: str | Path = ".") -> None:
        self._root = Path(root)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        return self._root / "raw"

    @property
    def wiki_dir(self) -> Path:
        return self._root / "wiki"

    @property
    def outputs_dir(self) -> Path:
        return self._root / "outputs"

    @property
    def _state_path(self) -> Path:
        return self.wiki_dir / ".vault_state.json"

    # ------------------------------------------------------------------
    # Core function 1–4: ingestion, organization, cross-referencing, wiki
    # ------------------------------------------------------------------

    def process_raw(self) -> IngestReport:
        """Scan ``raw/`` and fold new or changed material into ``wiki/``.

        Idempotent: unchanged files are skipped; changed files replace their
        existing wiki section instead of duplicating it.
        """
        with self._lock:
            self.wiki_dir.mkdir(parents=True, exist_ok=True)
            state = self._load_state()
            report = IngestReport()

            present = {f"raw/{p.name}" for p in self._raw_files()}
            for rel in [r for r in state["files"] if r not in present]:
                del state["files"][rel]

            for path in self._raw_files():
                rel = f"raw/{path.name}"
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                previous = state["files"].get(rel)
                if previous and previous["sha256"] == digest:
                    report.skipped.append(rel)
                    continue

                content = path.read_text(encoding="utf-8", errors="replace")
                topic = self._derive_topic(path, content)
                state["files"][rel] = {"sha256": digest, "topic": topic}
                (report.updated if previous else report.ingested).append(rel)

            self._rebuild_wiki(state)
            self._save_state(state)
            report.articles = sorted(
                {info["topic"] for info in state["files"].values()}
            )
            return report

    def articles(self) -> list[WikiArticle]:
        """Return the compiled wiki articles."""
        with self._lock:
            state = self._load_state()
            return self._build_articles(state)

    # ------------------------------------------------------------------
    # Core function 5: on-demand report generation
    # ------------------------------------------------------------------

    def generate_report(self, query: str, *, title: str | None = None) -> Path:
        """Answer ``query`` from the knowledge base into a new outputs file.

        Outputs are append-only: each call creates a fresh, date-prefixed
        Markdown file and never overwrites a previous one.
        """
        with self._lock:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            heading = title or query
            matches = self._search_knowledge(query)

            lines = [
                f"# {heading}",
                "",
                f"*Generated {datetime.now(timezone.utc).isoformat()} for query: `{query}`*",
                "",
            ]
            if matches:
                lines.append("## Findings")
                lines.append("")
                for source, snippet in matches:
                    lines.append(f"- {snippet}  ")
                    lines.append(f"  *Source: {source}*")
                lines.append("")
            else:
                lines.append("_No matching knowledge found in the wiki or raw material._")
                lines.append("")

            path = self._unique_output_path(heading)
            path.write_text("\n".join(lines), encoding="utf-8")
            return path

    def search(self, query: str) -> list[tuple[str, str]]:
        """Search the compiled wiki and raw material for ``query``.

        Returns ``(source, snippet)`` pairs without writing any files.
        """
        with self._lock:
            return self._search_knowledge(query)

    # ------------------------------------------------------------------
    # Private helpers — ingestion
    # ------------------------------------------------------------------

    def _raw_files(self) -> list[Path]:
        if not self.raw_dir.is_dir():
            return []
        return sorted(
            p
            for p in self.raw_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in _TEXT_EXTENSIONS
            and p.name.lower() != "readme.md"
        )

    @staticmethod
    def _derive_topic(path: Path, content: str) -> str:
        match = _TOPIC_RE.search(content)
        if match:
            return _slugify(match.group("topic"))
        stem = re.sub(r"^\d{4}-\d{2}(-\d{2})?[-_ ]*", "", path.stem)
        return _slugify(stem)

    # ------------------------------------------------------------------
    # Private helpers — wiki compilation and cross-referencing
    # ------------------------------------------------------------------

    def _build_articles(self, state: dict) -> list[WikiArticle]:
        by_topic: dict[str, list[str]] = {}
        for rel, info in sorted(state["files"].items()):
            by_topic.setdefault(info["topic"], []).append(rel)

        keywords = {
            topic: self._topic_keywords(sources)
            for topic, sources in by_topic.items()
        }
        articles = []
        for topic, sources in sorted(by_topic.items()):
            related = sorted(
                other
                for other in by_topic
                if other != topic and len(keywords[topic] & keywords[other]) >= 3
            )
            articles.append(
                WikiArticle(
                    slug=topic,
                    title=_title_from_slug(topic),
                    sources=sources,
                    related=related,
                )
            )
        return articles

    def _topic_keywords(self, sources: list[str]) -> set[str]:
        words: set[str] = set()
        for rel in sources:
            path = self._root / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            words.update(
                w for w in _WORD_RE.findall(text) if w not in _STOPWORDS
            )
        return words

    def _rebuild_wiki(self, state: dict) -> None:
        articles = self._build_articles(state)

        expected = {a.filename for a in articles} | {"index.md", ".vault_state.json"}
        for stale in self.wiki_dir.glob("*.md"):
            if stale.name not in expected:
                stale.unlink()

        for article in articles:
            self._write_article(article)
        self._write_index(articles)

    def _write_article(self, article: WikiArticle) -> None:
        lines = [
            f"# {article.title}",
            "",
            "> Compiled automatically by APEX. Do not edit files here by hand.",
            "",
        ]
        for rel in article.sources:
            path = self._root / rel
            body = (
                path.read_text(encoding="utf-8", errors="replace").strip()
                if path.exists()
                else "_(raw source no longer present)_"
            )
            lines += [f"## From `{rel}`", "", body, "", f"*Source: {rel}*", ""]
        if article.related:
            lines.append("## Related")
            lines.append("")
            lines += [
                f"- [{_title_from_slug(slug)}](./{slug}.md)"
                for slug in article.related
            ]
            lines.append("")
        (self.wiki_dir / article.filename).write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _write_index(self, articles: list[WikiArticle]) -> None:
        lines = [
            "# Wiki Index",
            "",
            "> This folder is written and updated automatically by APEX.",
            "> **Do not edit files here by hand.** See [`../KNOWLEDGE_BASE.md`](../KNOWLEDGE_BASE.md).",
            "",
            "Master index of the knowledge base, grouped by theme.",
            "",
            "## Themes",
            "",
        ]
        if articles:
            lines += [
                f"- [{a.title}](./{a.slug}.md) — {len(a.sources)} source(s)"
                for a in articles
            ]
        else:
            lines.append(
                "*(No articles yet — drop material into `../raw/` and run "
                "`process_raw()`.)*"
            )
        lines.append("")
        (self.wiki_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Private helpers — reports
    # ------------------------------------------------------------------

    def _search_knowledge(self, query: str) -> list[tuple[str, str]]:
        terms = [
            w for w in _WORD_RE.findall(query.lower()) if w not in _STOPWORDS
        ] or [query.lower().strip()]
        matches: list[tuple[str, str]] = []
        candidates = [
            p for p in sorted(self.wiki_dir.glob("*.md")) if p.name != "index.md"
        ] if self.wiki_dir.is_dir() else []
        candidates += self._raw_files()
        for path in candidates:
            rel = path.relative_to(self._root).as_posix()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                lowered = line.lower()
                if line.strip() and any(t in lowered for t in terms):
                    matches.append((rel, line.strip()))
        return matches

    def _unique_output_path(self, heading: str) -> Path:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        base = f"{date}-{_slugify(heading)[:60]}"
        path = self.outputs_dir / f"{base}.md"
        counter = 2
        while path.exists():
            path = self.outputs_dir / f"{base}-{counter}.md"
            counter += 1
        return path

    # ------------------------------------------------------------------
    # Private helpers — state
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        if self._state_path.exists():
            try:
                state = json.loads(self._state_path.read_text(encoding="utf-8"))
                if isinstance(state, dict) and isinstance(state.get("files"), dict):
                    return state
            except json.JSONDecodeError:
                pass
        return {"files": {}}

    def _save_state(self, state: dict) -> None:
        self._state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )
