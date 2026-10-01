"""Gemini route: Gemini 3.1 Pro with Deep Think, or Gemini 3.8 Flash.

Same shape as `call_gpt` and `call_claude` -- `call_gemini()` for a one-shot
judge call, `solve()` / `label()` for a solver run -- so a caller swaps route
by swapping the import.

MODEL IDS ARE STILL ONE ENV VAR AWAY. The defaults below were confirmed
against `models.list()` on 2026-09-16 with the AI Studio key in .env
(GOOGLE_API_KEY): `gemini-3.8-flash` is GA, while Pro is only published as
`gemini-3.1-pro-preview` -- there is no unsuffixed `gemini-3.1-pro` on that
catalogue. When the GA id lands, or on a credential that sees a different
catalogue, re-check with:

    python3 -c "from common.call_gemini import list_models; list_models()"

and set GEMINI_MODEL_PRO / GEMINI_MODEL_FLASH in .env. Nothing else in this
module needs touching.

DEEP THINK IS A THINKING LEVEL, NOT A SEPARATE MODEL. The SDK exposes it as
`ThinkingConfig(thinking_level=...)` with the ladder MINIMAL / LOW / MEDIUM /
HIGH (google.genai 2.23.0), so "Pro + Deep Think" is the Pro model at HIGH
with thought summaries returned. If a future SDK ships a distinct Deep Think
model id instead, set GEMINI_MODEL_PRO to it and the HIGH level stays correct.
"""
from __future__ import annotations

import functools
import mimetypes
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import dotenv
from google import genai
from google.genai import types

from common.agent_workspace import (
    Transcript,
    image_placeholder,
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

# Same two lines as call_gpt/call_claude: the .env sits one level above the
# checkout in this layout, not inside it.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(REPO_ROOT / ".env")
dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODELS = {
    "pro": os.getenv("GEMINI_MODEL_PRO", "gemini-3.1-pro-preview"),
    "flash": os.getenv("GEMINI_MODEL_FLASH", "gemini-3.8-flash"),
}

DEFAULT_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", "pro")

DISPLAY_NAMES = {
    "pro": "Gemini 3.1 Pro (Deep Think)",
    "flash": "Gemini 3.8 Flash",
}

# Pro is the careful route, so it runs at the top of the thinking ladder and
# returns its thought summaries -- a judge that has to justify a score is
# worth the tokens. Flash is the cheap route and is asked to think as little
# as the ladder allows; raising it would spend Pro money on a Flash call and
# defeat the point of having two routes.
THINKING_LEVEL = {
    "pro": types.ThinkingLevel.HIGH,
    "flash": types.ThinkingLevel.HIGH,
}
INCLUDE_THOUGHTS = {"pro": True, "flash": False}

DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "32768"))

# Deterministic by default. These calls grade CAD models; two runs of the same
# judge on the same evidence disagreeing is a defect, not variety.
DEFAULT_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0"))

CALL_RETRIES = 5
CALL_RETRY_BACKOFF_S = 10

# Anything not on this list is read as bytes and handed over with a guessed
# mime type; these are just the ones worth naming.
_TEXT_SUFFIXES = {".py", ".txt", ".md", ".json", ".toml", ".csv", ".step",
                  ".stp", ".scad"}


def resolve_model(model: str = DEFAULT_MODEL) -> str:
    """'pro'/'flash' -> the id actually sent. A full id passes through, so a
    caller can name a model this module has never heard of."""
    return MODELS.get(model, model)


def api_keys() -> list[str]:
    """Every AI Studio key in .env, best first.

    GEMINI_API_KEY and GOOGLE_API_KEY are the single-key names. The numbered
    GOOGLE_API_KEY_1..N are a pool: AI Studio quota is per key, so a batch of
    judge calls that exhausts one can carry on with the next rather than
    failing the run. Sorted numerically, because GOOGLE_API_KEY_10 must not
    sort before GOOGLE_API_KEY_2.
    """
    keys, seen = [], set()
    numbered = sorted(
        ((int(m.group(1)), v) for k, v in os.environ.items()
         if (m := re.fullmatch(r"GOOGLE_API_KEY_(\d+)", k)) and v.strip()),
        key=lambda kv: kv[0])
    for v in ([os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY")]
              + [v for _, v in numbered]):
        v = (v or "").strip()
        if v and v not in seen:
            seen.add(v)
            keys.append(v)
    return keys


#: THE SDK DOES NOT RETRY BY DEFAULT, which is not what the code reads
#: like. 408/429/500/502/503/504 are all in its retriable set with a
#: 1/2/4/8s backoff, but `retry_args(None)` returns
#: `stop_after_attempt(1)` -- so without this, one transient blip ends the
#: call. In an agent loop that means the whole run dies with its work half
#: done: a 503 killed a 26-turn run, and a 429 whose own payload said
#: "Please retry in 7.6s" killed another. Five attempts covers both.

#: A wall-clock ceiling per request, in MILLISECONDS. Without one the
#: SDK waits forever: a Deep Think call carrying a 1.2 MB inline PDF sat for
#: 42 minutes having never reached the tool loop -- container idle at 0% CPU,
#: nothing written, no error. Retries do not help there, because nothing ever
#: fails. 15 minutes is well past the slowest healthy call seen here (~6 min)
#: and still short enough to fail visibly.
REQUEST_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", str(15 * 60 * 1000)))

RETRY_HTTP_OPTIONS = types.HttpOptions(
    retry_options=types.HttpRetryOptions(attempts=5),
    timeout=REQUEST_TIMEOUT_MS)


@functools.lru_cache(maxsize=8)
def client(key_index: int = 0):
    """API key if there is one, Vertex otherwise. Built once per key.

    The cache is not just about connection reuse: `models.list()` returns a
    lazy pager, so a throwaway client is garbage-collected -- taking its
    httpx pool with it -- before the pager ever issues the request, which
    fails with "Cannot send a request, as the client has been closed".

    key_index selects from `api_keys()`; out of range means the pool is
    used up, which is a hard error rather than a silent fall-through to
    Vertex -- see below for why that distinction matters.

    Two routes because the two credentials in play here are different things:
    an AI Studio key is a key, while `.gcp_service_account.json` is the same
    service account the Drive tooling uses and reaches Gemini only through
    Vertex. Vertex additionally needs the API enabled on the project, which
    is a console action and is NOT done from here -- on this repo's project
    it is disabled, so the Vertex path 403s. That is exactly why a missing
    key must not quietly fall through to it: the failure then surfaces as a
    confusing "Agent Platform API has not been used in project ..." instead
    of "you have no key".
    """
    keys = api_keys()
    if keys:
        if key_index >= len(keys):
            raise RuntimeError(
                f"key_index {key_index} out of range: only {len(keys)} "
                f"Gemini API key(s) in .env")
        return genai.Client(api_key=keys[key_index],
                            http_options=RETRY_HTTP_OPTIONS)

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    sa_path = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
               or str(Path(__file__).resolve().parent.parent
                      / ".gcp_service_account.json"))
    if Path(sa_path).is_file():
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not project:
            import json
            project = json.loads(Path(sa_path).read_text())["project_id"]
        return genai.Client(vertexai=True, project=project,
                            location=location, credentials=creds,
                            http_options=RETRY_HTTP_OPTIONS)

    raise RuntimeError(
        "no Gemini credentials: set GEMINI_API_KEY or GOOGLE_API_KEY (or "
        "GOOGLE_API_KEY_1, _2, ...) in .env, or provide a service account "
        "with the Vertex AI API enabled on its project")

def list_models(substring: str = "gemini") -> list[str]:
    """Print and return the model ids this credential can actually see.

    The point of entry for confirming GEMINI_MODEL_PRO / GEMINI_MODEL_FLASH,
    since the defaults in this file are unverified.
    """
    names = [m.name for m in client().models.list()]
    hits = [n for n in names if substring.lower() in n.lower()]
    for n in hits:
        print(n)
    if not hits:
        print(f"no model name contains {substring!r}; {len(names)} visible")
    return hits


def _part(path) -> types.Part:
    path = Path(path)
    if path.suffix.lower() in _TEXT_SUFFIXES:
        return types.Part.from_text(
            text=f"--- {path.name} ---\n{path.read_text(errors='replace')}")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


def _config(model: str, *, system: str | None = None,
            max_output_tokens: int | None = None,
            temperature: float | None = None,
            thinking_level: types.ThinkingLevel | None = None,
            tools=None):
    return types.GenerateContentConfig(
        system_instruction=system,
        temperature=(DEFAULT_TEMPERATURE if temperature is None
                     else temperature),
        max_output_tokens=max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
        tools=tools,
        # We drive the tool loop ourselves in `run_agent`, so the SDK's
        # automatic function calling stays off: it only fires for Python
        # callables anyway, and leaving it enabled prints a warning on
        # every single call.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True),
        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level or THINKING_LEVEL.get(
                model, types.ThinkingLevel.MEDIUM),
            include_thoughts=INCLUDE_THOUGHTS.get(model, False),
        ),
    )


def _split_response(resp):
    """(answer_text, thought_text). Thought summaries come back as ordinary
    parts flagged `thought=True`, so a caller that just concatenates every
    part silently pastes the model's reasoning into its own answer."""
    answer, thoughts = [], []
    for cand in (resp.candidates or []):
        content = getattr(cand, "content", None)
        for p in (getattr(content, "parts", None) or []):
            text = getattr(p, "text", None)
            if not text:
                continue
            (thoughts if getattr(p, "thought", False) else answer).append(text)
    return "".join(answer).strip(), "".join(thoughts).strip()


def generate(prompt: str, *, model: str = DEFAULT_MODEL, files=(),
             system: str | None = None, max_output_tokens: int | None = None,
             temperature: float | None = None,
             thinking_level: types.ThinkingLevel | None = None):
    """One call. Returns (answer_text, meta) with the thought summary, the
    resolved model id and the token counts in meta."""
    model_id = resolve_model(model)
    parts = [types.Part.from_text(text=prompt)] + [_part(f) for f in files]
    resp = client().models.generate_content(
        model=model_id,
        contents=[types.Content(role="user", parts=parts)],
        config=_config(model, system=system,
                       max_output_tokens=max_output_tokens,
                       temperature=temperature,
                       thinking_level=thinking_level),
    )
    answer, thoughts = _split_response(resp)
    usage = getattr(resp, "usage_metadata", None)
    return answer, {
        "model": model_id,
        "thoughts": thoughts,
        "finish_reason": str(getattr(resp.candidates[0], "finish_reason", ""))
                         if resp.candidates else "",
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }


def call_gemini(prompt: str, *, model: str = DEFAULT_MODEL, files=(),
                retries: int = CALL_RETRIES) -> str:
    """Judge-call counterpart to `call_claude` / `call_gpt`: text in, text out.

    Retries on transport failures AND on an empty answer. Empty is a real
    outcome here rather than a hypothetical: a Pro call that spends its whole
    output budget inside Deep Think returns finish_reason MAX_TOKENS with
    every part flagged `thought`, so the answer is blank while the call itself
    looks successful.
    """
    last = None
    for attempt in range(retries):
        try:
            text, meta = generate(prompt, model=model, files=files)
            if text.strip():
                return text
            last = RuntimeError(
                f"empty answer (finish_reason={meta['finish_reason']}, "
                f"thought tokens={meta['thoughts_tokens']})")
        except Exception as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(CALL_RETRY_BACKOFF_S * (attempt + 1))
    raise RuntimeError(
        f"call_gemini failed after {retries} attempts: {last}") from last


# ---------------------------------------------------------------------------
# Agent loop -- Gemini counterpart to call_claude.run_agent / call_gpt.run_agent.
# The model gets the same four tools from `agent_workspace` against a scratch
# cwd and iterates until it stops calling them (or the turn budget runs out).
# One turn = one API round trip; a turn may contain several function calls.
# ---------------------------------------------------------------------------

AGENT_READ_TOOL = "read_file"
AGENT_BASH_TOOL = "bash"
SHARED_READ_TOOL, SHARED_BASH_TOOL = AGENT_READ_TOOL, AGENT_BASH_TOOL


def _agent_tools():
    """Neutral specs as Gemini function declarations.

    `parameters_json_schema` takes our JSON Schema as-is. The older
    `parameters=types.Schema(...)` field wants Google's own uppercase type
    enum ("OBJECT"/"STRING"), so it would need every spec transcribed.
    """
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=spec["name"],
            description=spec["description"],
            parameters_json_schema=spec["parameters"],
        ) for spec in TOOL_SPECS
    ])]


#: How many of the most recent tool results keep their full text. Older
#: ones are replaced by a one-line note. Six is enough to keep the working
#: set -- what the last few commands printed -- while dropping the feature
#: dumps and STL comparisons from the start of the run.
TOOL_RESULT_KEEP_FULL = 6

#: Below this, eliding costs more in confusion than it saves in tokens.
COMPACT_MIN_CHARS = 400

_ELIDED = ("[{chars} chars of output elided to stay inside the "
           "input-token-per-minute quota. Re-run the command if you need "
           "it again.]")


def _compact_tool_results(contents, keep_recent: int = TOOL_RESULT_KEEP_FULL):
    """Shrink old tool output in place. Returns (contents, chars_freed).

    WHY THIS EXISTS. These APIs are stateless: every turn resends the whole
    conversation, so a run's input-token cost grows with its own history.
    A CAD agent's history is mostly tool output -- 15 KB feature dumps, STL
    comparisons -- that mattered for one turn and is dead weight forever
    after. Left alone, a long run walks into
    `RESOURCE_EXHAUSTED ... input_token_count` and dies with its work half
    done. That is not hypothetical; it is where this function came from.

    ONLY USER-ROLE FUNCTION RESPONSES ARE TOUCHED. Model turns are returned
    verbatim no matter how old, because Gemini 3 binds a thought signature
    to the turn that produced it and rewriting one breaks the reasoning
    chain across tool calls.
    """
    idx = [i for i, c in enumerate(contents)
           if getattr(c, "role", None) == "user"
           and any(getattr(p, "function_response", None)
                   for p in (getattr(c, "parts", None) or []))]
    stale = idx[:-keep_recent] if keep_recent else idx
    if not stale:
        return contents, 0

    out, freed = list(contents), 0
    for i in stale:
        c, parts, changed = out[i], [], False
        for p in (getattr(c, "parts", None) or []):
            fr = getattr(p, "function_response", None)
            body = (dict(fr.response or {}).get("output")
                    if fr is not None else None)
            if isinstance(body, str) and len(body) > COMPACT_MIN_CHARS:
                freed += len(body)
                parts.append(types.Part.from_function_response(
                    name=fr.name,
                    response={"output": _ELIDED.format(chars=len(body))}))
                changed = True
            else:
                parts.append(p)
        if changed:
            out[i] = types.Content(role=c.role, parts=parts)
    return out, freed


def run_agent_shared_tools(prompt: str, *, cwd, model: str = DEFAULT_MODEL, files=(),
              inline_files=(), announce_files: bool = True,
              max_turns: int = DEFAULT_AGENT_MAX_TURNS,
              require_file: str | None = None, system: str | None = None,
              max_output_tokens: int | None = None,
              temperature: float | None = None,
              thinking_level: types.ThinkingLevel | None = None,
              verbose: bool = True):
    """The shared four-tool loop. Returns (final_text, meta).

    Not what try_model uses for Gemini any more (see `run_agent`, which is
    Gemini CLI); kept so Gemini can still be scored on the identical tool
    surface.

    files: copied into cwd and announced, as in the other two routes.
    inline_files: sent as prompt parts instead (images -- `read_file` is
    UTF-8 only, so a PNG on disk is invisible to this agent).

    require_file: if set and the model stops before that file exists in cwd,
    it gets one nudge to keep going.

    THE MODEL'S OWN `Content` IS APPENDED VERBATIM rather than rebuilt from
    its text. Gemini 3 thinking models return a thought signature alongside
    the function call, and the signature has to come back on the next turn or
    the model loses its reasoning chain across tool calls.
    """
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)

    if files:
        names = attach_files(cwd, files)
        if announce_files:
            prompt = prompt + attached_files_note(
                names, read_tool=SHARED_READ_TOOL, bash_tool=SHARED_BASH_TOOL)

    model_id = resolve_model(model)
    parts = [types.Part.from_text(text=prompt)] + [
        _part(f) for f in inline_files]
    contents = [types.Content(role="user", parts=parts)]

    cl = client()
    config = _config(model, system=system,
                     max_output_tokens=max_output_tokens,
                     temperature=temperature, thinking_level=thinking_level,
                     tools=_agent_tools())

    transcript = Transcript()
    final_text, nudged, turn = "", False, 0

    compacted_chars = 0
    while turn < max_turns:
        turn += 1
        contents, freed = _compact_tool_results(contents)
        if freed:
            compacted_chars += freed
            if verbose:
                print(f"    [gemini] compacted {freed} chars of old tool "
                      f"output out of the request")
        resp = cl.models.generate_content(
            model=model_id, contents=contents, config=config)

        cand = (resp.candidates or [None])[0]
        content = getattr(cand, "content", None)
        answer, thoughts = _split_response(resp)
        calls = list(resp.function_calls or [])

        transcript.append({
            "role": "assistant", "content": answer, "thoughts": thoughts,
            "function_calls": [{"name": c.name, "args": dict(c.args or {})}
                               for c in calls],
            "finish_reason": str(getattr(cand, "finish_reason", "")),
        })

        # A candidate with no parts at all (e.g. the whole budget spent in
        # Deep Think) cannot be appended as history -- the next request would
        # be rejected for an empty Content -- and there is nothing to act on.
        if content is None or not getattr(content, "parts", None):
            break
        contents.append(content)

        if not calls:
            if (require_file and not (cwd / require_file).is_file()
                    and not nudged and turn < max_turns):
                nudged = True
                note = (f"`{require_file}` does not exist in the working "
                        "directory yet -- your work is only graded from "
                        "that file.  Please continue and save it.")
                contents.append(types.Content(
                    role="user", parts=[types.Part.from_text(text=note)]))
                transcript.append({"role": "user", "content": note})
                if verbose:
                    print(f"    [gemini turn {turn}] stopped without "
                          f"{require_file}; nudged once")
                continue
            final_text = answer
            break

        responses = []
        for fc in calls:
            result = clip(dispatch(cwd, fc.name, dict(fc.args or {})))
            if verbose:
                preview = str(dict(fc.args or {}))[:110]
                print(f"    [gemini turn {turn}] {fc.name}({preview})"
                      f" -> {len(result)} chars")
            responses.append(types.Part.from_function_response(
                name=fc.name, response={"output": result}))
            transcript.append({"role": "tool", "name": fc.name,
                               "content": result})
        contents.append(types.Content(role="user", parts=responses))
    else:
        if verbose:
            print(f"    [gemini] turn budget exhausted ({max_turns})")

    return final_text, {"model": model_id, "turns": turn,
                        "transcript": transcript, "nudged": nudged,
                        "compacted_chars": compacted_chars}


# ---------------------------------------------------------------------------
# Gemini CLI agent (2026-09-25) -- what try_model runs for Gemini
#
# `run_agent` drives Google's own coding agent, Gemini CLI, headless
# (`gemini -p ... -o stream-json`), the way call_claude drives Claude Code
# and call_gpt drives Codex. It brings its own system prompt, file tools,
# compression and judgement about when to stop.
#
# THE SANDBOX HOLDS:
#   * run_shell_command, web_fetch, google_web_search, take_snapshot and
#     invoke_agent (subagents) are excluded. The shell is
#     `mcp_workspace_bash`, served by `common.workspace_mcp` in this
#     process, which runs wherever try_model put the shell.
#   * Gemini CLI's own file tools refuse any path outside the workspace
#     ("Path not in workspace") -- verified.
#   * GEMINI_CLI_HOME is a fresh temp dir: no ~/.gemini settings, GEMINI.md,
#     memory, extensions or skills reach the agent.
# ---------------------------------------------------------------------------

GEMINI_CLI_BASH_TOOL = "mcp_workspace_bash"
#: what `solve` names in its prompt -- the CLI's shell, not the shared one
AGENT_BASH_TOOL = GEMINI_CLI_BASH_TOOL

#: Built-in tools switched off for an eval run.
GEMINI_CLI_EXCLUDED_TOOLS = ["run_shell_command", "web_fetch",
                             "google_web_search", "take_snapshot",
                             "invoke_agent"]

#: Where inline files (drawings, PDF pages) are staged so the prompt can
#: attach them with Gemini CLI's @path syntax.
GEMINI_CLI_ATTACH_DIR = "_attached"


def _gemini_cli() -> str:
    exe = os.getenv("GEMINI_CLI") or shutil.which("gemini")
    if not exe:
        raise RuntimeError(
            "Gemini CLI is not installed: npm install -g @google/gemini-cli")
    return exe


def _gemini_settings(mcp_url: str, max_turns: int) -> dict:
    from common.agent_workspace import BASH_TIMEOUT_MAX_S
    return {
        "security": {"auth": {"selectedType": "gemini-api-key"},
                     "folderTrust": {"enabled": False}},
        #: deprecated in favour of the policy engine, still honoured (0.61)
        "tools": {"exclude": GEMINI_CLI_EXCLUDED_TOOLS},
        "mcpServers": {"workspace": {
            "httpUrl": mcp_url, "trust": True,
            "timeout": (BASH_TIMEOUT_MAX_S + 60) * 1000}},
        #: unlimited (-1): the CLI counts user, model and tool messages
        #: as turns, so passing max_turns here ended runs with
        #: FatalTurnLimitedError at ~460 tool calls of a 500 budget.
        #: run_agent's own on_event counter enforces the tool-call cap.
        "model": {"maxSessionTurns": -1},
        "general": {"disableAutoUpdate": True},
        "privacy": {"usageStatisticsEnabled": False},
    }


def run_agent(prompt: str, *, cwd, model: str = DEFAULT_MODEL, files=(),
              inline_files=(), announce_files: bool = True,
              max_turns: int = DEFAULT_AGENT_MAX_TURNS,
              require_file: str | None = None, verbose: bool = True,
              **_ignored):
    """Gemini CLI on the task. Returns (final_text, meta), like the other routes.

    Gemini CLI decides when it is finished; `max_turns` caps its tool calls,
    counted here -- the CLI's own maxSessionTurns is left unlimited. If it
    stops without
    `require_file`, the session is resumed once and asked to save it.
    """
    import json

    from common.agent_cli import run_jsonl
    from common.workspace_mcp import WorkspaceMCP

    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    cwd = cwd.resolve()
    if files:
        names = attach_files(cwd, files)
        if announce_files:
            prompt = prompt + attached_files_note(
                names, read_tool="read_file", bash_tool=GEMINI_CLI_BASH_TOOL)
    if inline_files:
        att = cwd / GEMINI_CLI_ATTACH_DIR
        att.mkdir(exist_ok=True)
        refs = []
        for f in inline_files:
            shutil.copy2(f, att / Path(f).name)
            refs.append(f"@{GEMINI_CLI_ATTACH_DIR}/{Path(f).name}")
        prompt += "\n\nAttached drawings/pages: " + " ".join(refs) + "\n"

    keys = api_keys()
    if not keys:
        raise RuntimeError("no Gemini API key: set GEMINI_API_KEY or "
                           "GOOGLE_API_KEY_1.. in .env")
    model_id = resolve_model(model)

    transcript = Transcript()
    names_by_id, inputs_by_id = {}, {}
    state = {"final": [], "calls": 0, "stats": [], "error": None,
             "exhausted": False}
    # try_model may provide a live progress channel for partial-run accounting.
    # Gemini CLI already owns the authoritative tool-call counter below; publish
    # that existing state so Ctrl-C, quota errors, network failures or CLI crashes
    # do not erase the work completed before run_agent() can return normally.
    progress = _ignored.get("progress")

    if progress is not None:
        progress.update(model=model_id, turns=0,
                        transcript=transcript, harness="gemini cli",)

    def on_event(ev):
        kind = ev.get("type")
        if kind == "message" and ev.get("role") == "assistant":
            state["final"].append(ev.get("content") or "")
        elif kind == "tool_use":
            state["calls"] += 1
            name, args = ev.get("tool_name"), ev.get("parameters")
            names_by_id[ev.get("tool_id")] = name
            inputs_by_id[ev.get("tool_id")] = args or {}
            transcript.append({"role": "assistant", "content": "",
                               "tool_calls": [{"name": name, "input": args}]})
            # Publish after recording the tool call so `turns=N` and the transcript
            # describe the same point in the run. AgentProgress does not count calls
            # independently; Gemini CLI's own event stream remains the source of truth.
            if progress is not None:
                progress.update(turns=state["calls"], transcript=transcript,)
            if verbose:
                print(f"    [gemini cli {state['calls']}] {name}"
                      f"({str(args)[:110]})")
            if state["calls"] > max_turns:
                state["exhausted"] = True
                return "stop"
        elif kind == "tool_result":
            tid = ev.get("tool_id")
            out = str(ev.get("output") or "")
            args = inputs_by_id.get(tid) or {}
            src = args.get("file_path") or args.get("absolute_path") or ""
            if (not out and Path(str(src)).suffix.lower() in
                    (".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf")):
                out = image_placeholder(src)
            transcript.append({"role": "tool",
                               "name": names_by_id.get(tid, "?"),
                               "content": clip(out or str(ev.get("status")
                                                          or ""))})
        elif kind == "result":
            state["stats"].append(ev.get("stats") or {})
            if ev.get("status") not in (None, "success"):
                state["error"] = str(ev.get("error") or ev.get("status"))
                if progress is not None:
                    progress.update(error=state["error"])
        elif kind == "error":
            state["error"] = str(ev.get("message") or ev)
            if progress is not None:
                progress.update(error=state["error"])
        return None

    def run(key: str, text: str, resume: bool):
        home = Path(tempfile.mkdtemp(prefix="gemini_home_")) if not resume \
            else state["home"]
        state["home"] = home
        (home / ".gemini").mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "GEMINI_CLI_HOME": str(home),
               "GEMINI_API_KEY": key}
        for k in ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI",
                  "GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"):
            env.pop(k, None)
        with WorkspaceMCP(cwd) as mcp:
            (home / ".gemini" / "settings.json").write_text(
                json.dumps(_gemini_settings(mcp.url, max_turns)),
                encoding="utf-8")
            cmd = [_gemini_cli(), "-m", model_id, "--approval-mode", "yolo",
                   "-o", "stream-json"]
            if resume:
                cmd += ["--resume", "latest"]
            #: the prompt arrives on stdin; `-p ""` only selects headless
            cmd += ["-p", ""]
            code, err, why = run_jsonl(cmd, env=env, cwd=cwd,
                                       stdin_text=text, on_event=on_event)
        if why in ("idle", "wall"):
            state["error"] = f"gemini cli run stopped: no progress ({why})"
        elif code not in (0, None) and not state["exhausted"] \
                and not state["error"]:
            state["error"] = f"gemini exited {code}: {err[-800:]}"
        # Idle/wall timeouts and abnormal CLI exits are detected outside Gemini's
        # JSON event stream. Publish them through the same live channel so abnormal
        # exits have consistent partial metadata regardless of where they originated.
        if state["error"] and progress is not None:
            progress.update(error=state["error"])

    nudged = False
    try:
        #: AI Studio quota is per key: a run that fails before doing any
        #: work moves on to the next key, as the one-shot path does.
        for n, key in enumerate(keys):
            state.update(error=None, final=[])
            if progress is not None:
                progress.update(error=None)
            run(key, prompt, resume=False)
            if not state["error"] or state["calls"] or n == len(keys) - 1:
                break
            if verbose:
                print(f"    [gemini cli] key {n + 1} failed before any work "
                      f"({state['error'][:120]}); trying the next")
            shutil.rmtree(state["home"], ignore_errors=True)
        if (require_file and not state["error"] and not state["exhausted"]
                and not (cwd / require_file).is_file()):
            nudged = True
            note = (f"`{require_file}` does not exist in the working "
                    "directory yet -- your work is only graded from that "
                    "file.  Please continue and save it.")
            transcript.append({"role": "user", "content": note})
            if verbose:
                print(f"    [gemini cli] stopped without {require_file}; "
                      "nudged once")
            state["final"] = []
            run(key, note, resume=True)
    finally:
        native = []
        if state.get("home"):
            #: Gemini CLI's own chat and log files for the session
            native = keep_native_logs(state["home"], [".gemini/tmp/**/*"],
                                      "gemini")
            shutil.rmtree(state["home"], ignore_errors=True)

    final = "".join(state["final"]).strip()
    if state["error"] and not final:
        raise RuntimeError(f"Gemini CLI run failed: {state['error']}")
    transcript.append({"role": "assistant", "content": final})
    if verbose:
        print(f"    [gemini cli] {state['calls']} tool calls"
              + (f" -- turn budget exhausted ({max_turns})"
                 if state["exhausted"] else ""))
    return final, {
        "model": model_id,
        "turns": max_turns if state["exhausted"] else state["calls"],
        "transcript": transcript, "nudged": nudged,
        "stats": state["stats"], "harness": "gemini cli",
        "error": state["error"], "native_logs": native,
    }


def label(model: str = DEFAULT_MODEL) -> str:
    pretty = DISPLAY_NAMES.get(model, model)
    return f"{pretty} (Gemini CLI, {resolve_model(model)})"


def solve(user_text, image_files=(), pdf_pages=(), pdf_files=(),
          code_files=(), pointcloud_files=(), *,
          model: str = DEFAULT_MODEL):
    """Agent solver, mirroring `call_claude.solve` / `call_gpt.solve`.

    Point clouds are staged as files now, not flattened through
    `call_gpt.pointcloud_to_text`: an agent with Bash and trimesh can measure
    the real mesh, and handing it a pre-sampled text dump would cap it at the
    one-shot route's view of the data.
    """
    workdir = tempfile.mkdtemp(prefix="gemini_agent_")
    try:
        files = (list(code_files) + list(pdf_files)
                 + list(pointcloud_files))
        text, meta = run_agent(
            user_text + agent_task_suffix(bash_tool=AGENT_BASH_TOOL),
            cwd=workdir, model=model,
            inline_files=list(image_files) + list(pdf_pages),
            files=files,
            require_file=AGENT_SOLUTION_FILENAME,
        )
        return read_back_solution(workdir, text), meta
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
if __name__ == "__main__":
    import sys

    if "--list" in sys.argv:
        list_models()
        raise SystemExit(0)
    which = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(label(which))
    answer, meta = generate(
        "In one sentence, what is the difference between a gate valve and a "
        "ball valve?", model=which)
    print(f"\n{answer}\n")
    print({k: v for k, v in meta.items() if k != "thoughts"})
