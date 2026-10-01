"""Run an agent CLI (Codex, Gemini CLI) headless and stream its JSON events.

Both harnesses print one JSON object per line in headless mode. This reads
them as they arrive so the caller can count tool calls as they happen and
stop a run that passes its ceiling -- neither CLI has a max-turns setting
that means the same thing as the other routes' turn budget.

The prompt goes in on stdin, never as an argument: a task prompt runs to
many lines, and on Windows a newline in an argument that passes through a
.cmd shim ends the command right there.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time

from common.agent_workspace import _kill_tree

#: A run with no event for this long is taken as hung. Generous: one tool
#: call may legitimately be a long SolidWorks build.
DEFAULT_IDLE_S = int(os.getenv("AGENT_CLI_IDLE_S", str(60 * 60)))
#: Wall-clock ceiling for a whole run.
DEFAULT_WALL_S = int(os.getenv("AGENT_CLI_WALL_S", str(8 * 60 * 60)))
#: How long ONE model request may go silent before the harness gives up on
#: it. Every harness defaults to about five minutes, and a reasoning model
#: at high effort can think longer than that before its first visible byte:
#: the harness then drops the request and retries it from scratch, forever
#: (an Opus 5.5 living-hinge run did exactly that for half an hour). Used by
#: all three routes -- Claude Code's API_TIMEOUT_MS and Codex's
#: stream_idle_timeout_ms. Gemini CLI has no such setting; its ~5 min limit
#: is Node's own, and the run-idle ceiling above is its backstop.
MODEL_REQUEST_TIMEOUT_S = int(os.getenv("AGENT_MODEL_TIMEOUT_S",
                                        str(60 * 60)))


def run_jsonl(cmd, *, env, cwd, stdin_text: str, on_event,
              idle_s: int = DEFAULT_IDLE_S, wall_s: int = DEFAULT_WALL_S):
    """Run `cmd`, calling on_event(dict) for each JSON line on stdout.

    on_event may return the string "stop" to end the run early (the turn
    ceiling). Returns (returncode, stderr_tail, stopped_why) where
    stopped_why is None, "stop", "idle" or "wall".
    """
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(cwd), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1)
    proc.stdin.write(stdin_text)
    proc.stdin.close()

    err_lines: list[str] = []
    t_err = threading.Thread(
        target=lambda: [err_lines.append(ln) for ln in proc.stderr],
        daemon=True)
    t_err.start()

    last = {"t": time.time()}
    lines: list[str] = []
    done = threading.Event()

    def read_out():
        for ln in proc.stdout:
            last["t"] = time.time()
            lines.append(ln)
        done.set()

    t_out = threading.Thread(target=read_out, daemon=True)
    t_out.start()

    start, why, seen = time.time(), None, 0
    while True:
        finished = done.wait(1.0)
        while seen < len(lines):
            ln = lines[seen].strip()
            seen += 1
            if not ln.startswith("{"):
                continue
            try:
                ev = json.loads(ln)
            except ValueError:
                continue
            if on_event(ev) == "stop" and why is None:
                why = "stop"
        if finished and seen >= len(lines):
            break
        if why is None and time.time() - last["t"] > idle_s:
            why = "idle"
        if why is None and time.time() - start > wall_s:
            why = "wall"
        if why:
            _kill_tree(proc)
            break
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
    t_err.join(timeout=5)
    return proc.returncode, "".join(err_lines)[-4000:], why
