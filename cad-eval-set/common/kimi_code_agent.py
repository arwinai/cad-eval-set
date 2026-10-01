"""Run Moonshot's Kimi Code CLI headless on a task, against an Azure Foundry model.

The agent harness behind the `kimi` and `grok` routes (call_kimi / call_grok),
as Claude Code is behind `claude`, Codex behind `gpt` and Gemini CLI behind
`gemini`: Kimi Code brings its own system prompt, loop, context compaction
and judgement about when it is done.

THE SANDBOX, and why it differs from the other harnesses:
  * Kimi Code's own file tools are NOT offered. Its `Read` read a file
    outside the workspace, and its permission rules did not stop it in
    headless (-p) mode. So an agent file allows ONLY the four shared tools,
    served by `common.workspace_mcp`: bash (wherever try_model put the
    shell) and read_file / write_file / list_dir, which agent_workspace
    confines to the workspace. No Bash, web tools, sub-agents or skills.
  * KIMI_CODE_HOME is a fresh temp dir per run: no ~/.kimi-code config,
    AGENTS.md, memory, credentials, sessions or plugins reach the agent, and
    --skills-dir points at an empty folder so no skill is discovered.

AZURE FOUNDRY needs a pass-through (`common.request_filter_proxy`) that
  * drops `prompt_cache_key`, which Kimi Code always sends and Foundry
    rejects with a 400;
  * renames tool-call ids to call_0, call_1, ... -- with Kimi Code's
    `functions_<tool>_<n>` ids in the history, Foundry's Kimi K2.7 ends every
    later turn with an empty stop.

`max_turns` caps tool calls over the whole run (nudge included), like the
other harness routes. If the agent stops without `require_file`, the session
is continued once (-c) and asked to save it.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from common.agent_cli import run_jsonl
from common.agent_workspace import (
    Transcript,
    attach_files,
    image_placeholder,
    attached_files_note,
    clip,
    keep_native_logs,
)
from common.request_filter_proxy import FilterProxy
from common.workspace_mcp import WorkspaceMCP

KIMI_CODE_CLI = os.getenv("KIMI_CODE_CLI") or str(
    Path.home() / ".kimi-code" / "bin" / "kimi.exe")

#: The agent's tools, as Kimi Code names MCP tools.
BASH_TOOL = "mcp__workspace__bash"
READ_TOOL = "mcp__workspace__read_file"
#: the four shared tools plus a confined image viewer -- Grok 4.6, Kimi
#: K2.7-Code and Kimi K3 all accept images on Foundry (checked)
SHARED_TOOLS = ("bash", "read_file", "write_file", "list_dir", "view_image")
VIEW_TOOL = "mcp__workspace__view_image"

AGENT_FILE = """---
name: eval-agent
description: CAD eval agent confined to its workspace
tools:
  - mcp__workspace__*
disallowedTools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - ReadMediaFile
  - WebSearch
  - FetchURL
  - Agent
  - AgentSwarm
---
${base_prompt}

Your tools are the workspace tools: mcp__workspace__bash runs a shell
command in the working directory; mcp__workspace__read_file / write_file /
list_dir work on files inside it (paths relative to it); and
mcp__workspace__view_image shows you an image file, so you can look at
drawings and renders yourself.
"""


def _config(proxy_url: str, model_id: str, context: int, capabilities) -> str:
    caps = ", ".join(f'"{c}"' for c in capabilities)
    return f'''default_model = "evalmodel"

[providers.foundry]
type = "openai"
base_url = "{proxy_url}"
api_key_env = "EVAL_FOUNDRY_KEY"

[models.evalmodel]
provider = "foundry"
model = "{model_id}"
max_context_size = {context}
capabilities = [{caps}]
'''

# `progress` is an optional live metadata channel supplied by try_model.
# It does not control the agent or resume a session; it only exposes work the
# harness has already measured before the normal return becomes available.
def run_kimi_code(prompt: str, *, cwd, model_id: str, base_url: str,
                  api_key: str, max_turns: int, files=(), inline_files=(),
                  announce_files: bool = True, require_file: str | None = None,
                  context: int = 200000, capabilities=("tool_use", "image_in"),
                  label: str = "kimi code", verbose: bool = True,
                  progress=None):
    """(final_text, meta) in the shape every route returns."""
    from common.agent_workspace import BASH_TIMEOUT_MAX_S

    if not Path(KIMI_CODE_CLI).is_file():
        raise RuntimeError(
            "Kimi Code CLI is not installed: see https://code.kimi.com "
            f"(expected {KIMI_CODE_CLI}, or set KIMI_CODE_CLI)")
    if not api_key:
        raise RuntimeError(f"{label}: no API key for {base_url}")
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    cwd = cwd.resolve()
    if files:
        names = attach_files(cwd, files)
        if announce_files:
            prompt += attached_files_note(names, read_tool=READ_TOOL,
                                          bash_tool=BASH_TOOL)
    if inline_files:
        #: staged as files; the agent looks at them with view_image
        att = cwd / "_attached"
        att.mkdir(exist_ok=True)
        for f in inline_files:
            shutil.copy2(f, att / Path(f).name)
        prompt += ("\n\nDrawings/pages for this task are in `_attached/`: "
                   + ", ".join(Path(f).name for f in inline_files)
                   + f". Look at them with {VIEW_TOOL}.\n")

    home = Path(tempfile.mkdtemp(prefix="kimi_home_")).resolve()
    (home / "no_skills").mkdir()
    agent_file = home / "eval-agent.md"
    agent_file.write_text(AGENT_FILE, encoding="utf-8")
    env = {**os.environ, "KIMI_CODE_HOME": str(home),
           "EVAL_FOUNDRY_KEY": api_key}

    transcript = Transcript()
    names_by_id, args_by_id = {}, {}
    state = {"final": "", "calls": 0, "error": None, "exhausted": False}

    # Publish the initial state as well.  If the run fails before its first tool
    # call, try_model can still identify the harness/model and preserve the
    # transcript rather than receiving a completely empty progress snapshot.
    if progress is not None:
        progress.update(model=model_id, turns=0, transcript=transcript,
                        harness=f"kimi code ({label})",)

    def on_event(ev):
        role = ev.get("role")
        if role == "assistant":
            if ev.get("content"):
                state["final"] = ev["content"]
                transcript.append({"role": "assistant",
                                   "content": ev["content"]})
            calls = ev.get("tool_calls") or []
            for c in calls:
                fn = c.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = fn.get("arguments")
                names_by_id[c.get("id")] = fn.get("name")
                args_by_id[c.get("id")] = args
                state["calls"] += 1
                transcript.append({"role": "assistant", "content": "",
                                   "tool_calls": [{"name": fn.get("name"),
                                                   "input": args}]})

                # `state["calls"]` is Kimi Code's authoritative benchmark tool-call
                # counter.  Publish it immediately while the run is alive.  If Ctrl-C,
                # a provider error or the CLI itself stops the run before the final return,
                # try_model can recover the exact amount of work already performed.
                #
                # AgentProgress deliberately does not increment anything itself: keeping the
                # harness as the single source of truth avoids two counters drifting apart.
                if progress is not None:
                    progress.update(turns=state["calls"], 
                                    transcript=transcript,)
                
                if verbose:
                    print(f"    [{label} {state['calls']}] {fn.get('name')}"
                          f"({str(args)[:110]})")
                if state["calls"] >= max_turns:
                    state["exhausted"] = True
                    return "stop"
        elif role == "tool":
            tid = ev.get("tool_call_id")
            text = str(ev.get("content") or "")
            if "image_url" in text and "base64," in text:
                #: the stream carries the image itself; name it instead
                a = args_by_id.get(tid) or {}
                text = image_placeholder(a.get("path") if isinstance(a, dict) else "?")
            transcript.append({"role": "tool", "name": names_by_id.get(tid, "?"),
                               "content": clip(text)})
        elif role == "meta" and ev.get("type") == "error":
            state["error"] = str(ev.get("message") or ev)
            # The failure reason is useful together with the last live tool count,
            # especially for resumable 429/network/provider failures.
            if progress is not None:
                progress.update(error=state["error"])
        return None

    def run(text: str, resume: bool, proxy_url: str, mcp_url: str):
        (home / "config.toml").write_text(
            _config(proxy_url, model_id, context, capabilities), encoding="utf-8")
        (home / "mcp.json").write_text(json.dumps({"mcpServers": {"workspace": {
            "url": mcp_url, "toolTimeoutMs": (BASH_TIMEOUT_MAX_S + 60) * 1000}}}),
            encoding="utf-8")
        cmd = [KIMI_CODE_CLI, "-p", text, "--output-format", "stream-json",
               "--skills-dir", str(home / "no_skills")]
        #: the agent is bound when the session is created; a resumed session
        #: restores it and the flag is not allowed alongside --continue
        cmd += ["-c"] if resume else ["--agent-file", str(agent_file)]
        code, err, why = run_jsonl(cmd, env=env, cwd=cwd, stdin_text="",
                                   on_event=on_event)
        if why in ("idle", "wall"):
            state["error"] = f"{label} run stopped: no progress ({why})"
        elif code not in (0, None) and not state["exhausted"] and not state["error"]:
            state["error"] = f"{label} exited {code}: {err.strip()[-600:]}"

    nudged = False
    try:
        with FilterProxy(base_url, drop={"prompt_cache_key"},
                         plain_tool_call_ids=True) as proxy, \
                WorkspaceMCP(cwd, tools=SHARED_TOOLS) as mcp:
            run(prompt, False, proxy.url, mcp.url)
            if (require_file and not state["error"] and not state["exhausted"]
                    and not (cwd / require_file).is_file()):
                nudged = True
                note = (f"`{require_file}` does not exist in the working "
                        "directory yet -- your work is only graded from that "
                        "file.  Please continue and save it.")
                transcript.append({"role": "user", "content": note})
                if verbose:
                    print(f"    [{label}] stopped without {require_file}; nudged once")
                run(note, True, proxy.url, mcp.url)
    finally:
        native = keep_native_logs(home, ["sessions/**/*", "logs/*"], "kimi")
        shutil.rmtree(home, ignore_errors=True)

    if state["error"] and not state["final"]:
        raise RuntimeError(f"{label} run failed: {state['error']}")
    if verbose:
        print(f"    [{label}] {state['calls']} tool calls"
              + (f" -- turn budget exhausted ({max_turns})"
                 if state["exhausted"] else ""))
    return state["final"], {
        "model": model_id,
        "turns": max_turns if state["exhausted"] else state["calls"],
        "transcript": transcript, "nudged": nudged,
        "harness": f"kimi code ({label})", "error": state["error"],
        "native_logs": native,
    }
