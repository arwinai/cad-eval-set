"""Provider-neutral scratch-directory scaffolding shared by the three routes.

All three routes drive their own agent loop over the four tools implemented
here -- bash / write_file / read_file / list_dir against a scratch cwd. None
of the three providers supplies a tool runtime we use: Claude reaches Azure
through the plain Messages API, which has no built-in tools either.

THIS MODULE EXISTS SO THE THREE ROUTES SHARE A PROMPT, NOT JUST A SIGNATURE.
An eval that gives Claude "inspect the files, run your script, save
solution.py" and GPT something worded differently is comparing prompts as
much as models. `agent_task_suffix()` and `attached_files_note()` are the
single source of both, parameterised only by what each provider calls its
tools.

It deliberately does NOT import any provider SDK. Each `call_*` module
imports its own at module scope -- see the note in `call_llm` about the
import being the point of failure -- so putting the shared pieces in one of
them would make the other two unimportable wherever that one SDK is
missing, which is exactly the coupling the route split is meant to avoid.
"""
from __future__ import annotations

import os
import shutil
import re
import subprocess
from pathlib import Path

# One turn = one API round trip, which may contain several tool calls.
DEFAULT_AGENT_MAX_TURNS = 40

TOOL_OUTPUT_LIMIT = 15000
BASH_TIMEOUT_DEFAULT_S = 300
BASH_TIMEOUT_MAX_S = 900

AGENT_SOLUTION_FILENAME = "solution.py"

_SHELL = "cmd.exe" if os.name == "nt" else "/bin/sh"


class Transcript(list):
    """A run's transcript that stamps every entry as it is appended.

    `t` is seconds since the run started and `at` the wall-clock time (UTC,
    ISO 8601), so a reader can see where an hour-long run spent its time --
    model thinking, a slow build, retries. Every route builds its transcript
    with this, so the stamps appear whichever model try_model called; entries
    that already carry a `t` keep it.
    """

    def __init__(self, *args):
        super().__init__(*args)
        import time
        self._t0 = time.time()

    def append(self, entry):
        if isinstance(entry, dict) and "t" not in entry:
            import time
            from datetime import datetime, timezone
            now = time.time()
            entry = {**entry, "t": round(now - self._t0, 2),
                     "at": datetime.fromtimestamp(now, timezone.utc)
                     .isoformat(timespec="seconds")}
        super().append(entry)


class AgentProgress:
    """Provider-neutral live progress for one agent run.

    WHY THIS EXISTS
    ----------------
    Agent harnesses normally return their metadata only after `run_agent()`
    finishes.  That is too late for an interrupted run: Ctrl-C, an API error,
    a CLI crash or another abnormal exit may happen after the model has already
    spent hours and made hundreds of tool calls, but before the harness reaches
    its normal `return final_text, meta`.

    The harness already knows its progress while it is running.  Codex,
    Gemini CLI, Kimi Code and Claude Code all maintain a live tool-call
    counter.  This object gives them one provider-neutral place to publish
    that information so try_model can recover it even when run_agent() never
    returns normally.

    The harness remains authoritative about what counts as a tool call.
    AgentProgress does NOT inspect transcripts or dispatch calls and does not
    try to count anything itself.  It only stores snapshots reported by the
    harness.  This matters because different agent implementations observe
    their tools at different layers.

    A fresh AgentProgress belongs to exactly one model session.  Resume starts
    a new session with a new AgentProgress; cumulative accounting across
    sessions remains try_model's responsibility.
    """

    def __init__(self):
        self._meta: dict = {}

    def update(self, **meta) -> None:
        """Publish the latest known run metadata.

        Updates are incremental: a harness may first publish `turns`, later
        add its transcript or usage data, and update `turns` repeatedly as the
        run progresses.  Existing fields not mentioned by an update survive.
        """
        self._meta.update(meta)

    def snapshot(self) -> dict:
        """Return a detached snapshot safe for try_model to modify.

        try_model adds interruption/error/resume fields to solve metadata.
        Returning a copy prevents those reporting changes from mutating the
        live object still owned by the harness.
        """
        return dict(self._meta)


def image_placeholder(path, size_hint: str = "") -> str:
    """What the transcript records when a tool handed the model an image.

    The bytes went to the model, not into the log; this names the file so a
    reader can open the same image from the run's kept workspace.
    """
    name = str(path or "?")
    return f"[image returned to the model: {name}{' ' + size_hint if size_hint else ''}]"


def keep_native_logs(src_root, patterns, label: str) -> list[str]:
    """Copy a harness's own session files out of its throwaway home.

    Claude Code, Codex and Gemini CLI each write a complete session log --
    every message, retry, timeout, compaction and the system prompt -- into
    the temp home they are given, which is deleted when the run ends. This
    copies the matching files (relative paths kept) into a fresh temp dir
    and returns their paths; try_model moves them into the run folder as
    `native/`. Routes with no harness return none: their transcript is the
    whole record already.
    """
    import tempfile
    src_root = Path(src_root)
    out_dir = Path(tempfile.mkdtemp(prefix=f"native_{label}_"))
    kept = []
    for pat in patterns:
        for f in src_root.glob(pat):
            if f.is_file():
                dest = out_dir / f.relative_to(src_root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                kept.append(str(dest))
    return kept


def clip(text: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    """Keep the head and (mostly) the tail of an over-long tool result.

    Tail-weighted on purpose: when a build script fails, the traceback is at
    the end, and a head-only clip throws away the only part worth reading.
    """
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 4], text[-(limit * 3 // 4):]
    return head + f"\n...[{len(text) - limit} chars clipped]...\n" + tail


# --------------------------------------------------------------------------
# Tool implementations. Every one takes cwd first and returns a plain string;
# a tool that raises is reported back to the model as text rather than
# killing the run, since a recoverable mistake (bad path, bad JSON) is
# something the model can act on and a crash is not.
# --------------------------------------------------------------------------

#: After the tree is killed, how long to wait for the pipes to close.
BASH_DRAIN_S = 10


def _kill_tree(proc):
    """Kill the shell AND everything it spawned.

    Killing only the direct child is not enough, and the difference is not
    theoretical: a SolidWorks run wedged for 286 minutes because the shell
    launched ScriptRunner.exe, ScriptRunner blocked on a COM call, and the
    timeout killed `cmd.exe` while the orphan lived on.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


#: KILLING BY PID IS REFUSED, and the reason is a run that ended in one.
#: `tasklist | findstr /i python` lists EVERY python on the machine --
#: including `try_model.py`, which is what hosts the agent, holds the
#: curtain over the repository and writes the report at the end. An agent
#: that believed its own build script had hung ran `taskkill /PID 19780
#: /F` on the only python it could see. That was the parent. The run died
#: with the deliverable already saved and no report written, and the
#: curtain stayed shut over the whole checkout until it was swept by hand.
#:
#: Nothing legitimate is lost. Every command this tool starts is killed
#: with its whole tree when it times out (`_kill_tree`), so an agent never
#: needs to reap anything itself -- and a command it did not start is not
#: its business.
#:
#: A SPEED BUMP, NOT A SANDBOX, exactly like the curtain: this reads the
#: command line, and a determined agent can spell a kill in ways no
#: pattern catches. It is here to stop the accident, which is what
#: actually happened.
_KILL_PATTERNS = re.compile(
    r"(?:^|[\s|&;(`])(?:"
    r"taskkill|tskill|pskill|"              # Windows
    r"stop-process|"                        # PowerShell
    r"kill|pkill|killall|"                  # POSIX
    r"wmic\s+process[^\n]*\bdelete\b"      # the WMI spelling
    r")(?:$|[\s/|&;)])", re.IGNORECASE)


def _refuses_to_kill(command: str) -> str | None:
    """The refusal text, or None when the command kills nothing."""
    if not _KILL_PATTERNS.search(command or ""):
        return None
    return ("ERROR: this command was NOT executed. It kills processes by "
            "id or by name, and the process list on this machine includes "
            "the one that is running you -- an earlier agent ended its own "
            "run that way, half a second after saving a finished "
            "deliverable. You never need to: every command you start is "
            "killed with its whole tree when it times out, so a script "
            "that stops responding is already bounded by the `timeout_s` "
            "you passed. If something you started seems stuck, do not "
            "wait on it and do not reap it -- run the next step and look "
            "for its result file on disk. If a modal dialog is blocking "
            "the SolidWorks COM thread, arm the watchdog "
            "(`sws.arm_unattended(app)`) rather than killing anything.")


def tool_bash(cwd, command, timeout_s=None):
    """Run a command, and be genuinely killable.

    NOT `subprocess.run(..., timeout=)`, which does not survive contact with
    a command that outlives its shell. On timeout it kills the direct child
    and then reads the pipes again -- and a surviving grandchild still holds
    the write end, so the read blocks forever. The timeout looks present and
    does nothing; the agent loop hangs with no turn, no error and no ceiling.
    Popen plus an explicit tree kill is what actually bounds the call.
    """
    t = min(int(timeout_s or BASH_TIMEOUT_DEFAULT_S), BASH_TIMEOUT_MAX_S)

    refusal = _refuses_to_kill(command)
    if refusal:
        return refusal

    # A COMMAND WITH EMBEDDED NEWLINES DOES NOT RUN ON WINDOWS, and fails in
    # the worst possible way: cmd.exe breaks at the first newline, the body
    # never executes, and the call returns a bare `exit=0` with no output and
    # no stderr. That is indistinguishable from "ran fine, printed nothing".
    # One agent read it as success forty turns in a row while doing nothing
    # at all. Refusing it with an explanation costs one turn; the silence
    # cost forty.
    if os.name == "nt" and chr(10) in command:
        return ("ERROR: this command was NOT executed. It contains newlines, "
                "and cmd.exe splits on them -- the body after the first line "
                "never runs, which is why such commands appear to succeed "
                "with empty output. Write the script to a file with "
                "write_file and run that file instead, e.g. "
                "write_file('probe.py', ...) then bash('python probe.py').")

    kw = {} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(command, shell=True, cwd=str(cwd),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace", **kw)
    try:
        out, err = proc.communicate(timeout=t)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=BASH_DRAIN_S)
        except subprocess.TimeoutExpired:
            return (f"TIMEOUT: killed after {t}s, but something it spawned "
                    "still holds the output pipe, so its output is lost. "
                    "Whatever you started is detached -- do not wait on it "
                    "again; check for a result file on disk instead.")
        partial = (out or "")
        if err:
            partial += ("\n--- stderr ---\n" if partial else
                        "--- stderr ---\n") + err
        head = (f"TIMEOUT: command killed after {t}s, along with every "
                "process it started")
        return (head + ("\n" + partial if partial.strip() else "")).strip()

    out = out or ""
    if err:
        out += ("\n--- stderr ---\n" if out else "--- stderr ---\n") + err
    return f"exit={proc.returncode}\n{out}".strip()


def _inside(cwd, path):
    """`path` resolved against cwd, or None if it lands outside cwd.

    `Path(cwd) / path` is NOT a confinement: an absolute path replaces cwd
    outright and `..` walks out of it, so read_file("C:/.../solution/x")
    used to succeed. Every file tool goes through this instead.
    """
    root = Path(cwd).resolve()
    p = (root / path).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None
    return p


def _outside_msg(path):
    return (f"ERROR: {path} is outside the working directory; only files "
            "inside it can be read or written")


def tool_write_file(cwd, path, content):
    p = _inside(cwd, path)
    if p is None:
        return _outside_msg(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def tool_read_file(cwd, path, start_line=None, line_count=None):
    p = _inside(cwd, path)
    if p is None:
        return _outside_msg(path)
    if not p.is_file():
        return f"ERROR: {path} does not exist"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"ERROR reading {path}: {exc}"
    start = max(int(start_line or 1), 1) - 1
    stop = start + int(line_count) if line_count else len(lines)
    return "\n".join(lines[start:stop]) or "(empty)"


def tool_list_dir(cwd, pattern=None):
    cwd = Path(cwd).resolve()
    rows = []
    if pattern and (Path(pattern).is_absolute() or ".." in Path(pattern).parts):
        return _outside_msg(pattern)
    for p in sorted(cwd.glob(pattern or "*")):
        try:
            size = "<dir>" if p.is_dir() else str(p.stat().st_size)
        except OSError:
            size = "?"
        rows.append(f"{p.relative_to(cwd)}  {size}")
    return "\n".join(rows) or "(no matches)"


TOOL_IMPL = {
    "bash": tool_bash,
    "write_file": tool_write_file,
    "read_file": tool_read_file,
    "list_dir": tool_list_dir,
}


# Neutral JSON-Schema specs. `call_gpt` wraps these in OpenAI's
# {"type": "function", ...} envelope and `call_gemini` converts them to
# `types.FunctionDeclaration`, so both models see the same four tools with
# the same descriptions.
TOOL_SPECS = [
    {
        "name": "bash",
        "description": (
            f"Run a shell command ({_SHELL}) in the working directory and "
            "return exit code + stdout/stderr.  `python` is on PATH.  Prefer "
            "writing scripts with write_file and running them here over long "
            "one-liners."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string",
                        "description": "the command line to run"},
            "timeout_s": {"type": "integer",
                          "description": f"seconds (default "
                                         f"{BASH_TIMEOUT_DEFAULT_S}, max "
                                         f"{BASH_TIMEOUT_MAX_S})"},
        }, "required": ["command"]},
    },
    {
        "name": "write_file",
        "description": "Write a UTF-8 text file (path relative to the "
                       "working directory), creating parent dirs.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative file path"},
            "content": {"type": "string", "description": "full file contents"},
        }, "required": ["path", "content"]},
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file (path relative to the working "
                       "directory).  Optional 1-based start line and line "
                       "count.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative file path"},
            "start_line": {"type": "integer", "description": "1-based"},
            "line_count": {"type": "integer",
                           "description": "how many lines to return"},
        }, "required": ["path"]},
    },
    {
        "name": "list_dir",
        "description": "List files matching a glob pattern relative to the "
                       "working directory (default '*'; '**/*' recurses).",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "glob, e.g. '**/*'"},
        }},
    },
]


def dispatch(cwd, name, args):
    impl = TOOL_IMPL.get(name)
    if impl is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        return impl(cwd, **args)
    except TypeError as exc:
        return f"ERROR: bad arguments for {name}: {exc}"
    except Exception as exc:
        return f"ERROR: {name} raised {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Prompt scaffolding
# --------------------------------------------------------------------------

BINARY_DATA_EXTS = {".ply", ".stl", ".step", ".stp", ".obj", ".3mf"}

BINARY_DATA_HINT = (
    "Do NOT open it with {read} (it's binary) -- inspect it from {bash} with "
    "Python, e.g. `python3 -c \"import trimesh, numpy as np; "
    "v = np.asarray(trimesh.load('{name}').vertices); print(v.shape, "
    "v.min(0), v.max(0))\"`. You have trimesh/numpy available, so you can "
    "measure the real coordinates (extents, cross-sections, clusters) "
    "rather than estimating from renders."
)


def attach_files(cwd, files) -> list[str]:
    """Copy reference files into the scratch dir, keeping their own names."""
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    names = []
    for f in files:
        f = Path(f)
        dest = cwd / f.name
        if f.resolve() != dest.resolve():
            shutil.copy2(f, dest)
        names.append(f.name)
    return names


def attached_files_note(names, *, read_tool: str = "Read",
                        bash_tool: str = "Bash") -> str:
    names = list(names)
    if not names:
        return ""
    lines = ["\n\nThese files are in your current working directory:"]
    for name in names:
        if Path(name).suffix.lower() in BINARY_DATA_EXTS:
            lines.append(f"- `{name}` -- " + BINARY_DATA_HINT.format(
                name=name, read=read_tool, bash=bash_tool))
        else:
            lines.append(f"- `{name}`")
    return "\n".join(lines)


def agent_task_suffix(*, bash_tool: str = "Bash",
                      solution_filename: str = AGENT_SOLUTION_FILENAME) -> str:
    return (
        "\n\nYou're working in a scratch directory that has the reference "
        "material above copied in as files (listed below) -- inspect them "
        f"yourself if that helps. Feel free to run your script with "
        f"{bash_tool} to check it actually builds before finalizing (e.g. "
        "`python3 script.py`). When you're done, write your final, complete, "
        f"self-contained script to a file named exactly "
        f"`{solution_filename}` in the current directory."
    )


def read_back_solution(workdir, text: str,
                       filename: str = AGENT_SOLUTION_FILENAME) -> str:
    """The graded artefact is the file on disk, not the chat text.

    An agent that writes solution.py and then says "done" would otherwise be
    scored on the word "done"; when the file is missing we fall back to the
    final message so a one-shot-style answer still grades.
    """
    p = Path(workdir) / filename
    if p.is_file():
        return f"```python\n{p.read_text(encoding='utf-8', errors='replace')}\n```"
    return text
