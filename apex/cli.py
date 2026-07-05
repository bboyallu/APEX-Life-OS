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
    from apex.agent.scheduler import Scheduler

    system = _build_system(args)
    scheduler = Scheduler(system)
    interval = args.interval
    print(
        f"apex daemon started (interval={interval}s, "
        f"knowledge_root={args.knowledge_root})",
        flush=True,
    )
    last_cycle: float | None = None
    try:
        while True:
            system.heartbeat()
            now = time.monotonic()
            if last_cycle is None or now - last_cycle >= interval:
                last_cycle = now
                report = system.run_knowledge_informed_cycle()
                print(
                    "cycle complete: "
                    f"severity={report.overall_severity.name.lower()}",
                    flush=True,
                )
            for action in scheduler.run_due():
                print(f"scheduled task run: {action}", flush=True)
            system.process_alert_timeouts()
            time.sleep(30)
    except KeyboardInterrupt:
        print("apex daemon stopped", flush=True)
        return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from apex.agent.tui import run_chat

    return run_chat(args.knowledge_root)


def _cmd_model(args: argparse.Namespace) -> int:
    from apex.agent.config import PROVIDER_PRESETS, load_config, save_config

    config = load_config()
    if not args.provider:
        print(
            f"provider={config.provider} model={config.resolved_model()} "
            f"base_url={config.resolved_base_url()}"
        )
        print(f"available providers: {', '.join(sorted(PROVIDER_PRESETS))}")
        return 0
    try:
        config.use_provider(args.provider, args.model)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    save_config(config)
    print(f"switched to {config.provider}:{config.model}")
    return 0


def _cmd_gateway(args: argparse.Namespace) -> int:
    from apex.agent.gateway import run_gateway

    return run_gateway(args.knowledge_root)


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from apex.agent.dashboard import run_dashboard

    return run_dashboard(args.knowledge_root, host=args.host, port=args.port)


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

    p = sub.add_parser(
        "chat", help="Interactive terminal chat with the APEX agent"
    )
    p.set_defaults(func=_cmd_chat)

    p = sub.add_parser("model", help="Show or switch the LLM provider/model")
    p.add_argument("provider", nargs="?", help="Provider preset name")
    p.add_argument("model", nargs="?", help="Model name (optional)")
    p.set_defaults(func=_cmd_model)

    p = sub.add_parser(
        "gateway", help="Run the Telegram messaging gateway (long polling)"
    )
    p.set_defaults(func=_cmd_gateway)

    p = sub.add_parser(
        "dashboard", help="Run the web dashboard (default 127.0.0.1:9119)"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9119)
    p.set_defaults(func=_cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
