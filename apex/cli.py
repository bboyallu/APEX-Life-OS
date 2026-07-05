"""Command-line interface for APEX Life OS.

Installed as the ``apex`` console script (see ``pyproject.toml``), and also
runnable as ``python -m apex``. Designed so the system can be operated on a
headless server (e.g. a VPS via systemd or Docker) without writing any code.
"""

from __future__ import annotations

import argparse
import sys
import time

from apex import ApexSystem, __version__


def _build_system(args: argparse.Namespace) -> ApexSystem:
    return ApexSystem(knowledge_root=args.knowledge_root)


def _cmd_cycle(args: argparse.Namespace) -> int:
    system = _build_system(args)
    report = system.run_cycle()
    print(f"cycle complete: severity={report.overall_severity.name.lower()}")
    return 0


def _cmd_knowledge_cycle(args: argparse.Namespace) -> int:
    system = _build_system(args)
    report = system.run_knowledge_informed_cycle()
    print(
        "knowledge-informed cycle complete: "
        f"severity={report.overall_severity.name.lower()}"
    )
    return 0


def _cmd_process_knowledge(args: argparse.Namespace) -> int:
    system = _build_system(args)
    report = system.process_knowledge()
    print(
        f"knowledge processed: ingested={len(report.ingested)} "
        f"updated={len(report.updated)} articles={len(report.articles)}"
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    system = _build_system(args)
    path = system.generate_knowledge_report(args.query)
    print(path)
    return 0


def _cmd_verify_audit(args: argparse.Namespace) -> int:
    system = _build_system(args)
    valid, message = system.verify_audit_chain()
    print(message)
    return 0 if valid else 1


def _cmd_daemon(args: argparse.Namespace) -> int:
    system = _build_system(args)
    interval = args.interval
    print(
        f"apex daemon started (interval={interval}s, "
        f"knowledge_root={args.knowledge_root})",
        flush=True,
    )
    try:
        while True:
            system.heartbeat()
            report = system.run_knowledge_informed_cycle()
            system.process_alert_timeouts()
            print(
                "cycle complete: "
                f"severity={report.overall_severity.name.lower()}",
                flush=True,
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        print("apex daemon stopped", flush=True)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apex",
        description="APEX Life OS — self-evolving AI system.",
    )
    parser.add_argument(
        "--version", action="version", version=f"apex-life-os {__version__}"
    )
    parser.add_argument(
        "--knowledge-root",
        default=".",
        help="Directory containing raw/, wiki/ and outputs/ (default: current directory)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cycle", help="Run one MAPE-K adaptation cycle")
    p.set_defaults(func=_cmd_cycle)

    p = sub.add_parser(
        "knowledge-cycle", help="Run one knowledge-informed evolution cycle"
    )
    p.set_defaults(func=_cmd_knowledge_cycle)

    p = sub.add_parser(
        "process-knowledge", help="Fold new material from raw/ into the wiki/"
    )
    p.set_defaults(func=_cmd_process_knowledge)

    p = sub.add_parser(
        "report", help="Answer a query from the knowledge base into outputs/"
    )
    p.add_argument("query", help="Question to answer from the knowledge base")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("verify-audit", help="Verify the audit chain integrity")
    p.set_defaults(func=_cmd_verify_audit)

    p = sub.add_parser(
        "daemon",
        help="Run continuously: heartbeat + knowledge-informed cycles",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between cycles (default: 300)",
    )
    p.set_defaults(func=_cmd_daemon)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
