"""Tests for the ``apex`` command-line interface."""

from __future__ import annotations

import pytest

from apex import __version__
from apex.cli import build_parser, main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_requires_command():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_cycle(tmp_path, capsys):
    assert main(["--knowledge-root", str(tmp_path), "cycle"]) == 0
    assert "cycle complete" in capsys.readouterr().out


def test_knowledge_cycle(tmp_path, capsys):
    assert main(["--knowledge-root", str(tmp_path), "knowledge-cycle"]) == 0
    assert "knowledge-informed cycle complete" in capsys.readouterr().out


def test_process_knowledge(tmp_path, capsys):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "note.md").write_text("# Deep Work\n\nFocus matters.\n")
    assert main(["--knowledge-root", str(tmp_path), "process-knowledge"]) == 0
    assert "ingested=1" in capsys.readouterr().out


def test_report(tmp_path, capsys):
    assert main(["--knowledge-root", str(tmp_path), "report", "deep work"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.endswith(".md")


def test_verify_audit(tmp_path):
    assert main(["--knowledge-root", str(tmp_path), "verify-audit"]) == 0


def test_parser_lists_all_commands():
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "cycle",
        "knowledge-cycle",
        "process-knowledge",
        "report",
        "verify-audit",
        "daemon",
    ):
        assert command in help_text
