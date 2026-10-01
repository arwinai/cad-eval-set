from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import tempfile
from pathlib import Path

import dotenv
from openai import AzureOpenAI

from common.agent_workspace import (
    Transcript,
    keep_native_logs,
    AGENT_SOLUTION_FILENAME,
    DEFAULT_AGENT_MAX_TURNS,
    TOOL_SPECS,
    agent_task_suffix,
    attach_files,
    attached_files_note,
    clip,
    dispatch,
    read_back_solution,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(REPO_ROOT / ".env")
#: AND the package root, which is NOT REPO_ROOT -- see the note in
#: call_claude. In the built package common/ sits one level deeper, so
#: REPO_ROOT lands one above the package root and never sees its .env.
dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# GPT-5.6 (2026-07-09) on the gtm-research resource. The old default --
# deployment "gpt-5" on jenni-m7rybsi6-eastus2 -- was plain GPT-5
# (2025-08-07), despite the name.
GPT_CONFIG = {
    # One resource serves both the GPT and the Claude deployments, so the
    # plain AZURE_API_KEY is the usual answer; the GTM_RESEARCH name
    # stays as an override for a second resource.
    "api_key": os.getenv("AZURE_API_KEY_GTM_RESEARCH")
               or os.getenv("AZURE_API_KEY"),
    "endpoint": os.getenv(
        "AZURE_GPT_ENDPOINT", "https://gtm-research.openai.azure.com/"),
    "deployment_name": os.getenv("AZURE_GPT_DEPLOYMENT", "gpt-5.6-sol"),
    # 2025-03-01-preview is the floor for the Responses API, which
    # `run_agent` needs; chat completions (the one-shot path) is unaffected
    # by the bump.
    "api_version": os.getenv("AZURE_GPT_API_VERSION", "2025-04-01-preview"),
    "reasoning_effort": "xhigh",
}

DEFAULT_MAX_COMPLETION_TOKENS = 100000


def _image_data_url(path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


DEFAULT_POINTCLOUD_SAMPLE = 3000
POINTCLOUD_SAMPLE_SEED = 0


def pointcloud_to_text(path, max_points: int = DEFAULT_POINTCLOUD_SAMPLE) -> str:
    import numpy as np
    import trimesh

    path = Path(path)
    v = np.asarray(trimesh.load(path).vertices, dtype=float)
    total = len(v)
    if total == 0:
        raise ValueError(f"{path.name} contains no points")

    mins, maxs = v.min(axis=0), v.max(axis=0)
    centroid = v.mean(axis=0)

    if total > max_points:
        idx = np.sort(np.random.default_rng(POINTCLOUD_SAMPLE_SEED).choice(
            total, size=max_points, replace=False))
        sample = v[idx]
        sampled_note = (
            f"{max_points} of them, sampled uniformly at random, are listed "
            "below (the stats above are over all points, not just these)")
    else:
        sample = v
        sampled_note = "all of them are listed below"

    fmt = lambda a: ", ".join(f"{x:.3f}" for x in a)
    lines = [
        f"Point cloud `{path.name}` - {total} points; {sampled_note}.",
        f"  bounding box min (x, y, z): {fmt(mins)}",
        f"  bounding box max (x, y, z): {fmt(maxs)}",
        f"  bounding box size (x, y, z): {fmt(maxs - mins)}",
        f"  centroid (x, y, z): {fmt(centroid)}",
        "",
        "XYZ coordinates, one point per line:",
        "```",
        *(f"{x:.3f} {y:.3f} {z:.3f}" for x, y, z in sample),
        "```",
    ]
    return "\n".join(lines)


def client():
    return AzureOpenAI(
        api_key=GPT_CONFIG["api_key"],
        azure_endpoint=GPT_CONFIG["endpoint"],
        api_version=GPT_CONFIG["api_version"],
    )


def call_gpt(prompt: str, *, images=(), image_b64=(), pointclouds=(),
             pointcloud_sample: int = DEFAULT_POINTCLOUD_SAMPLE,
             max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
             return_raw: bool = False):
    content = [{"type": "text", "text": prompt}]
    for ply in pointclouds:
        content.append({"type": "text",
                        "text": pointcloud_to_text(ply, pointcloud_sample)})
    for img in images:
        content.append({"type": "image_url",
                        "image_url": {"url": _image_data_url(img)}})
    for b64 in image_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + b64}})

    resp = client().chat.completions.create(
        model=GPT_CONFIG["deployment_name"],
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=GPT_CONFIG["reasoning_effort"],
    )
    text = resp.choices[0].message.content or ""
    return (text, resp.model_dump()) if return_raw else text


# ---------------------------------------------------------------------------
# Agent loop -- GPT counterpart to call_claude.run_agent, built on the chat
# completions tool-calling API.  The model gets bash / write_file / read_file
# / list_dir against a scratch cwd and iterates until it stops calling tools
# (or the turn budget runs out).  One turn = one API round trip; a turn may
# contain several tool calls.
# ---------------------------------------------------------------------------

#: The tool implementations, schemas and output clipping moved to
#: `agent_workspace` when the Gemini route needed the same four tools.
#: GPT's names for them, used in the prompt wording so the model is told
#: to use the tools it actually has.
AGENT_READ_TOOL = "read_file"
AGENT_BASH_TOOL = "bash"
SHARED_READ_TOOL, SHARED_BASH_TOOL = AGENT_READ_TOOL, AGENT_BASH_TOOL


def _agent_tool_schemas_responses():
    """Responses-API tool shape: flat, not wrapped in a "function" object."""
    return [{"type": "function", **spec} for spec in TOOL_SPECS]


def _input_content(prompt, images, image_b64):
    content = [{"type": "input_text", "text": prompt}]
    for img in images:
        content.append({"type": "input_image",
                        "image_url": _image_data_url(img)})
    for b64 in image_b64:
        content.append({"type": "input_image",
                        "image_url": "data:image/png;base64," + b64})
    return content


def run_agent_shared_tools(prompt: str, *, cwd, images=(), image_b64=(),
              files=(), announce_files: bool = True,
              max_turns: int = DEFAULT_AGENT_MAX_TURNS,
              max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
              reasoning_effort: str | None = None,
              require_file: str | None = None,
              verbose: bool = True):
    """The shared four-tool loop on the Responses API.  Returns (final_text, meta).

    Not what try_model uses for GPT any more (see `run_agent`, which is
    Codex); kept so GPT can still be scored on the identical tool surface.

    NOT chat completions, which is what the one-shot `call_gpt` still uses.
    gpt-5.6-sol rejects function tools combined with reasoning_effort there:
    "Function tools with reasoning_effort are not supported ... use
    /v1/responses or set reasoning_effort to 'none'".  Dropping to effort
    'none' would hand the eval a non-reasoning GPT, so the agent moved API
    instead.  Needs api-version 2025-03-01-preview or later.

    Turns are chained with `previous_response_id` rather than by resending
    the transcript: it keeps the model's reasoning state across tool calls
    server-side, which is the reasoning-model equivalent of passing Gemini's
    thought signatures back.

    files: copied into cwd and announced, as in the other two routes.
    require_file: if set and the model stops before that file exists in cwd,
    it gets one nudge to keep going.
    """
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)

    if files:
        names = attach_files(cwd, files)
        if announce_files:
            prompt = prompt + attached_files_note(
                names, read_tool=SHARED_READ_TOOL, bash_tool=SHARED_BASH_TOOL)

    cl = client()
    tools = _agent_tool_schemas_responses()
    effort = reasoning_effort or GPT_CONFIG["reasoning_effort"]

    pending = [{"role": "user",
                "content": _input_content(prompt, images, image_b64)}]
    prev_id = None
    transcript = Transcript()
    final_text, nudged, turn = "", False, 0

    while turn < max_turns:
        turn += 1
        resp = cl.responses.create(
            model=GPT_CONFIG["deployment_name"],
            input=pending,
            tools=tools,
            reasoning={"effort": effort},
            max_output_tokens=max_completion_tokens,
            previous_response_id=prev_id,
        )
        prev_id = resp.id

        calls = [it for it in (resp.output or [])
                 if getattr(it, "type", "") == "function_call"]
        text = resp.output_text or ""
        transcript.append({"role": "assistant", "content": text,
                           "function_calls": [{"name": c.name,
                                               "arguments": c.arguments}
                                              for c in calls],
                           "status": resp.status})

        if not calls:
            if (require_file and not (cwd / require_file).is_file()
                    and not nudged and turn < max_turns):
                nudged = True
                note = (f"`{require_file}` does not exist in the working "
                        "directory yet -- your work is only graded from "
                        "that file.  Please continue and save it.")
                pending = [{"role": "user",
                            "content": [{"type": "input_text", "text": note}]}]
                transcript.append({"role": "user", "content": note})
                if verbose:
                    print(f"    [gpt turn {turn}] stopped without "
                          f"{require_file}; nudged once")
                continue
            final_text = text
            break

        pending = []
        for c in calls:
            try:
                args = json.loads(c.arguments or "{}")
                result = dispatch(cwd, c.name, args)
            except json.JSONDecodeError as exc:
                result = f"ERROR: tool arguments were not valid JSON: {exc}"
            result = clip(result)
            if verbose:
                preview = (c.arguments or "")[:110].replace(chr(10), " ")
                print(f"    [gpt turn {turn}] {c.name}({preview})"
                      f" -> {len(result)} chars")
            pending.append({"type": "function_call_output",
                            "call_id": c.call_id, "output": result})
            transcript.append({"role": "tool", "call_id": c.call_id,
                               "name": c.name, "content": result})
    else:
        if verbose:
            print(f"    [gpt] turn budget exhausted ({max_turns})")

    return final_text, {"model": GPT_CONFIG["deployment_name"],
                        "turns": turn, "transcript": transcript,
                        "nudged": nudged}

# ---------------------------------------------------------------------------
# Codex agent (2026-09-25) -- what try_model runs for GPT
#
# `run_agent` drives OpenAI's own coding agent, the Codex CLI, headless
# (`codex exec --json`) against the same Azure deployment, the way
# call_claude drives Claude Code. Codex brings its own system prompt,
# apply_patch editing, compaction and judgement about when to stop.
#
# THE SANDBOX HOLDS the same way it does for Claude Code:
#   * Codex's shell is switched off (features shell_tool, unified_exec).
#     Its tools are then reached through code mode's `exec`, which can
#     call only apply_patch, MCP resource helpers and MCP tools -- verified
#     by asking it, and by watching it fail to read outside the workspace
#     any other way. Reads go through `mcp__workspace__bash`, served by
#     `common.workspace_mcp` in this process, which runs wherever try_model
#     put the shell. apply_patch writes are held to the workspace by
#     Codex's own workspace-write sandbox.
#   * Web search, browser/computer use, apps, plugins and image generation
#     are off.
#   * CODEX_HOME is a fresh temp dir: no ~/.codex config, AGENTS.md, auth,
#     sessions or skills reach the agent.
# ---------------------------------------------------------------------------

#: The agent's tool names as Codex shows them to the model. Files are read
#: with the shell, so both names point at it.
CODEX_BASH_TOOL = "mcp__workspace__bash"
AGENT_BASH_TOOL = CODEX_BASH_TOOL
AGENT_READ_TOOL = CODEX_BASH_TOOL

#: Codex features switched off for an eval run -- its own shell, and
#: everything that reaches beyond the workspace or the task.
CODEX_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "multi_agent", "browser_use",
    "browser_use_external", "computer_use", "in_app_browser", "apps",
    "plugins", "remote_plugin", "image_generation", "goals", "sleep_tool",
    "tool_suggest", "skill_search", "realtime_conversation",
    "workspace_dependencies", "worktrees", "view_image",
)

#: Codex wants the v1 Responses path on the resource, not a versioned one.
CODEX_BASE_URL = os.getenv(
    "CODEX_AZURE_BASE_URL",
    GPT_CONFIG["endpoint"].rstrip("/") + "/openai/v1")

#: The key reaches Codex through this variable only (config names it).
CODEX_KEY_ENV = "EVAL_CODEX_AZURE_KEY"


def _codex_cli() -> str:
    """codex.exe itself, not the npm .cmd shim, where it can be found."""
    if os.getenv("CODEX_CLI"):
        return os.environ["CODEX_CLI"]
    shim = shutil.which("codex")
    if shim and os.name == "nt":
        root = Path(shim).parent / "node_modules" / "@openai" / "codex"
        for exe in root.glob("node_modules/@openai/codex-win32-*/vendor/*/"
                             "bin/codex.exe"):
            return str(exe)
    if shim:
        return shim
    raise RuntimeError("the Codex CLI is not installed: npm install -g @openai/codex")


def _codex_config(mcp_url: str, effort: str) -> list[str]:
    from common.agent_cli import MODEL_REQUEST_TIMEOUT_S
    from common.agent_workspace import BASH_TIMEOUT_MAX_S
    return [
        'model_provider="azure"',
        'model_providers.azure.name="Azure OpenAI"',
        f'model_providers.azure.base_url="{CODEX_BASE_URL}"',
        f'model_providers.azure.env_key="{CODEX_KEY_ENV}"',
        'model_providers.azure.wire_api="responses"',
        #: Codex's default is 5 min, shorter than gpt-5.6-sol at xhigh can
        #: stay silent while it reasons
        'model_providers.azure.stream_idle_timeout_ms='
        f'{MODEL_REQUEST_TIMEOUT_S * 1000}',
        f'model_reasoning_effort="{effort}"',
        'web_search="disabled"',
        'approval_policy="never"',
        #: without this, workspace-write on Windows degrades to read-only
        #: and apply_patch is refused
        'windows.sandbox="unelevated"',
        #: workspace-write ALSO makes %TEMP% and /tmp writable by default --
        #: and try_model's workspaces live under %TEMP%, next to other runs.
        #: Only the workspace itself may be written.
        'sandbox_workspace_write.exclude_tmpdir_env_var=true',
        'sandbox_workspace_write.exclude_slash_tmp=true',
        f'mcp_servers.workspace.url="{mcp_url}"',
        f'mcp_servers.workspace.tool_timeout_sec={BASH_TIMEOUT_MAX_S + 60}',
        'mcp_servers.workspace.default_tools_approval_mode="approve"',
    ]


def run_agent(prompt: str, *, cwd, images=(), image_b64=(),
              files=(), announce_files: bool = True,
              max_turns: int = DEFAULT_AGENT_MAX_TURNS,
              reasoning_effort: str | None = None,
              require_file: str | None = None,
              verbose: bool = True, **_ignored):
    """Codex on the task. Returns (final_text, meta), like the other routes.

    Codex decides when it is finished. `max_turns` caps its tool calls --
    Codex has no turn setting of its own -- and passing it ends the run.
    If it stops without `require_file`, the same session is resumed once
    and asked to save it.
    """
    from common.agent_cli import run_jsonl
    from common.workspace_mcp import WorkspaceMCP

    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    #: The long form. Temp paths here come back 8.3-shortened (STORYG~1),
    #: and Codex's Windows sandbox registers the long form as writable --
    #: so apply_patch on the short one fails, every time.
    cwd = cwd.resolve()
    if files:
        names = attach_files(cwd, files)
        if announce_files:
            prompt = prompt + attached_files_note(
                names, read_tool=AGENT_READ_TOOL, bash_tool=AGENT_BASH_TOOL)
    if not GPT_CONFIG["api_key"]:
        raise RuntimeError("no Azure credential for GPT: set AZURE_API_KEY in .env")

    effort = reasoning_effort or GPT_CONFIG["reasoning_effort"]
    home = Path(tempfile.mkdtemp(prefix="codex_home_"))
    img_dir = Path(tempfile.mkdtemp(prefix="codex_img_"))
    image_paths = [str(Path(i)) for i in images]
    for n, b64 in enumerate(image_b64):
        p = img_dir / f"image_{n}.png"
        p.write_bytes(base64.b64decode(b64))
        image_paths.append(str(p))

    env = {**os.environ, "CODEX_HOME": str(home),
           CODEX_KEY_ENV: GPT_CONFIG["api_key"]}
    for k in ("OPENAI_API_KEY", "CODEX_API_KEY"):
        env.pop(k, None)

    transcript = Transcript()
    state = {"thread": None, "final": "", "calls": 0, "usage": [],
             "error": None, "exhausted": False}

    # try_model may provide a live progress channel for partial-run accounting.
    # Codex already owns the authoritative tool-call counter below; publishing it
    # here lets try_model recover the work completed before Ctrl-C, an API error,
    # or a CLI failure prevents run_agent() from reaching its normal return.
    progress = _ignored.get("progress")

    if progress is not None:
        progress.update(model=GPT_CONFIG["deployment_name"], turns=0,
                        transcript=transcript, harness="codex exec",)

    def on_event(ev):
        kind = ev.get("type")
        if kind == "thread.started":
            state["thread"] = ev.get("thread_id")
        elif kind == "turn.completed":
            state["usage"].append(ev.get("usage") or {})
        elif kind in ("turn.failed", "error"):
            err = ev.get("error") or {}
            state["error"] = (err.get("message") if isinstance(err, dict)
                              else None) or ev.get("message") or str(ev)
            # Keep a provider/CLI failure beside the last live work count.  If Codex
            # cannot return normally, try_model will still have both the partial progress
            # and the reason the session stopped.
            if progress is not None:
                progress.update(error=state["error"])
        elif kind == "item.completed":
            it = ev.get("item") or {}
            t = it.get("type")
            if t == "agent_message":
                state["final"] = it.get("text") or ""
                transcript.append({"role": "assistant",
                                   "content": state["final"]})
            elif t in ("mcp_tool_call", "file_change", "command_execution"):
                state["calls"] += 1
                if t == "mcp_tool_call":
                    name = f"mcp__{it.get('server')}__{it.get('tool')}"
                    args = it.get("arguments")
                    res = it.get("result") or {}
                    out = ("\n".join(c.get("text", "") for c in
                                     res.get("content") or [])
                           if res else (it.get("error") or {}).get("message", ""))
                elif t == "file_change":
                    name, args = "apply_patch", {"changes": it.get("changes")}
                    out = it.get("status", "")
                else:
                    name, args = "command", {"command": it.get("command")}
                    out = it.get("aggregated_output", "")
                transcript.append({"role": "assistant", "content": "",
                                   "tool_calls": [{"name": name,
                                                   "input": args}]})
                transcript.append({"role": "tool", "name": name,
                                   "content": clip(str(out))})
                # Publish only after the transcript contains both sides of this completed
                # Codex action.  The snapshot then describes exactly the same amount of work
                # as `turns`, rather than claiming N calls while recording only N-1.
                if progress is not None:
                    progress.update(turns=state["calls"], 
                                    transcript=transcript,)
                if verbose:
                    print(f"    [codex {state['calls']}] {name}"
                          f"({str(args)[:110]})")
                if state["calls"] >= max_turns:
                    state["exhausted"] = True
                    return "stop"
        return None

    def run(extra: list[str], text: str):
        with WorkspaceMCP(cwd) as mcp:
            cmd = [_codex_cli(), "exec", "--json", "--strict-config",
                   "--skip-git-repo-check",
                   "-m", GPT_CONFIG["deployment_name"],
                   "-s", "workspace-write", "-C", str(cwd)]
            for c in _codex_config(mcp.url, effort):
                cmd += ["-c", c]
            for f in CODEX_DISABLED_FEATURES:
                cmd += ["--disable", f]
            cmd += extra
            code, err, why = run_jsonl(cmd, env=env, cwd=cwd,
                                       stdin_text=text, on_event=on_event)
        if why in ("idle", "wall"):
            state["error"] = f"codex run stopped: no progress ({why})"
        elif code not in (0, None) and not state["exhausted"] \
                and not state["error"]:
            state["error"] = f"codex exited {code}: {err[-800:]}"
        # Some failures are detected by the process runner rather than arriving
        # as Codex JSON events (idle/wall timeout or an abnormal CLI exit).
        # Publish those too so the live snapshot follows the same error path
        # regardless of where the harness detected the failure.
        if state["error"] and progress is not None:
            progress.update(error=state["error"])

    nudged = False
    try:
        first = []
        for p in image_paths:
            first += ["-i", p]
        run(first + ["-"], prompt)
        if (require_file and not state["error"] and not state["exhausted"]
                and state["thread"] and not (cwd / require_file).is_file()):
            nudged = True
            note = (f"`{require_file}` does not exist in the working "
                    "directory yet -- your work is only graded from that "
                    "file.  Please continue and save it.")
            transcript.append({"role": "user", "content": note})
            if verbose:
                print(f"    [codex] stopped without {require_file}; nudged once")
            run(["resume", state["thread"], "-"], note)
    finally:
        #: Codex's own rollout files: every event of every turn, resume
        #: included
        native = keep_native_logs(home, ["sessions/**/*.jsonl"], "codex")
        shutil.rmtree(home, ignore_errors=True)
        shutil.rmtree(img_dir, ignore_errors=True)

    if state["error"] and not state["final"]:
        raise RuntimeError(f"Codex run failed: {state['error']}")
    if verbose:
        print(f"    [codex] {state['calls']} tool calls"
              + (f" -- turn budget exhausted ({max_turns})"
                 if state["exhausted"] else ""))
    return state["final"], {
        "model": GPT_CONFIG["deployment_name"],
        #: tool calls, and the ceiling itself when it ran out -- what
        #: try_model's `_exhausted` reads
        "turns": max_turns if state["exhausted"] else state["calls"],
        "transcript": transcript, "nudged": nudged,
        "usage": state["usage"], "harness": "codex exec",
        "error": state["error"], "native_logs": native,
    }


def label() -> str:
    return (f"GPT (Codex, Azure, deployment={GPT_CONFIG['deployment_name']}, "
            f"reasoning_effort={GPT_CONFIG['reasoning_effort']})")


def solve(user_text, image_files=(), pdf_pages=(), pdf_files=(),
          code_files=(), pointcloud_files=()):
    """Agent solver, mirroring `call_claude.solve`.

    Images go through the vision channel rather than onto disk: the
    read_file tool is UTF-8 text only, so a PNG copied into the scratch
    dir would be unreadable to this agent. Everything else -- code, PDFs,
    point clouds -- is staged as a file, which is what lets the model
    measure a .ply with trimesh instead of reading a sampled text dump.
    """
    workdir = tempfile.mkdtemp(prefix="gpt_agent_")
    try:
        files = (list(code_files) + list(pdf_files)
                 + list(pointcloud_files))
        text, meta = run_agent(
            user_text + agent_task_suffix(bash_tool=AGENT_BASH_TOOL),
            cwd=workdir,
            images=list(image_files) + list(pdf_pages),
            files=files,
            require_file=AGENT_SOLUTION_FILENAME,
        )
        return read_back_solution(workdir, text), meta
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
