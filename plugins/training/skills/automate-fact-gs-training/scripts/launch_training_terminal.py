#!/usr/bin/env python3
"""Launch a training command in a separate graphical terminal with durable status."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


TERMINALS = ("ghostty", "ptyxis", "gnome-terminal", "x-terminal-emulator")


def choose_terminal(requested: str) -> str:
    if requested != "auto":
        if not shutil.which(requested):
            raise SystemExit(f"Terminal executable not found: {requested}")
        return requested
    preferred = []
    if os.environ.get("GHOSTTY_RESOURCES_DIR"):
        preferred.append("ghostty")
    if os.environ.get("PTYXIS_ID") or os.environ.get("TERM_PROGRAM") == "ptyxis":
        preferred.append("ptyxis")
    preferred.extend(TERMINALS)
    for name in dict.fromkeys(preferred):
        if shutil.which(name):
            return name
    raise SystemExit("No supported graphical terminal found")


def terminal_command(terminal: str, cwd: Path, runner: Path) -> list[str]:
    bash_command = ["bash", str(runner)]
    if terminal == "ghostty":
        return [terminal, f"--working-directory={cwd}", "-e", *bash_command]
    if terminal == "ptyxis":
        return [terminal, "--standalone", f"--working-directory={cwd}", "--", *bash_command]
    if terminal == "gnome-terminal":
        return [terminal, f"--working-directory={cwd}", "--", *bash_command]
    return [terminal, "-e", *bash_command]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", required=True, type=Path, help="Training working directory")
    parser.add_argument("--run-dir", required=True, type=Path, help="Directory for log/PID/status files")
    parser.add_argument("--title", default="FaCT-GS Training")
    parser.add_argument("--terminal", choices=("auto", *TERMINALS), default="auto")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    cwd = args.cwd.expanduser().resolve()
    if not cwd.is_dir():
        parser.error(f"working directory does not exist: {cwd}")
    environment: dict[str, str] = {}
    for item in args.env:
        if "=" not in item or not item.split("=", 1)[0].isidentifier():
            parser.error(f"invalid --env value: {item!r}")
        key, value = item.split("=", 1)
        environment[key] = value
    terminal = choose_terminal(args.terminal)
    run_dir = args.run_dir.expanduser()
    if not run_dir.is_absolute():
        run_dir = cwd / run_dir
    run_dir = run_dir.resolve()
    runner = run_dir / "run.sh"
    log = run_dir / "terminal.log"
    pid_file = run_dir / "pid"
    status_file = run_dir / "exit_code"
    env_command = ["env", *[f"{key}={value}" for key, value in environment.items()], *command]
    launch = terminal_command(terminal, cwd, runner)
    summary = {
        "terminal": terminal,
        "launch": launch,
        "command": env_command,
        "run_dir": str(run_dir),
        "log": str(log),
        "pid_file": str(pid_file),
        "status_file": str(status_file),
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    runner_text = f"""#!/usr/bin/env bash
set +e
cd -- {shlex.quote(str(cwd))}
printf '\\033]0;%s\\007' {shlex.quote(args.title)}
rm -f -- {shlex.quote(str(status_file))}
script -q -f -e {shlex.quote(str(log))} -c {shlex.quote(shlex.join(env_command))} &
training_pid=$!
printf '%s\\n' "$training_pid" > {shlex.quote(str(pid_file))}
wait "$training_pid"
training_status=$?
printf '%s\\n' "$training_status" > {shlex.quote(str(status_file))}
printf '\\nFaCT-GS training finished with exit code %s.\\n' "$training_status"
read -r -p 'Press Enter to close this terminal...'
exit "$training_status"
"""
    runner.write_text(runner_text, encoding="utf-8")
    runner.chmod(0o700)
    try:
        subprocess.Popen(launch, cwd=cwd, start_new_session=True)
    except Exception:
        runner.unlink(missing_ok=True)
        run_dir.rmdir()
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
