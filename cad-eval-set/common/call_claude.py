"""Claude route: Claude Sonnet 5, Claude Opus 5.5 or Claude Fable 5.1, on Azure.

AZURE ONLY. There is no Bedrock path in this module any more, and no AWS
credential is read anywhere in it. All three models are deployments on the same
`jenni-m7rybsi6-eastus2` resource that serves the GPT route, reached with
`AnthropicFoundry` -- Azure is "Microsoft Foundry" to the Anthropic SDK, and
it takes the resource NAME, not the `*.cognitiveservices.azure.com` URL (a
base_url of that form 404s).

THE AGENT IS REAL CLAUDE CODE (2026-09-25). `run_agent` drives the `claude`
CLI through the Claude Agent SDK, pointed at the same Azure deployments via
Claude Code's Foundry backend (CLAUDE_CODE_USE_FOUNDRY), which did not exist
when this module first moved off the SDK. So the Claude route gets Claude
Code's own system prompt, its Read/Write/Edit/Glob/Grep tools, subagents,
automatic compaction on long runs, and its own judgement about when it is
done -- not the four-tool loop GPT and Gemini share. Scores are therefore
harness-vs-harness, not model-vs-model. The shared loop is kept as
`run_agent_shared_tools` for a like-for-like comparison.

THE SANDBOX STILL HOLDS, and that took three changes to Claude Code:

  * Its built-in Bash is switched off. The agent's shell is an in-process
    MCP tool (`mcp__workspace__bash`) that goes through `agent_workspace`
    -- which try_model patches to run in a container or as the restricted
    Windows user. A host shell can read solution/ and git history; models
    have done exactly that.
  * A PreToolUse hook refuses any file tool (Read, Write, Edit, Glob, Grep,
    NotebookEdit) whose path resolves outside the workspace, subagents
    included. WebFetch and WebSearch are not offered: the source models of
    several tasks are public.
  * It runs with a fresh CLAUDE_CONFIG_DIR, so nothing from the operator's
    ~/.claude -- settings, CLAUDE.md, memory, plugins -- reaches the agent.

The judge path (`call_claude`) is unchanged: one plain API call, no harness.
Model ids are first-party ids (`claude-sonnet-5`), not Bedrock's
`us.anthropic.*` inference-profile ids.

THINKING IS CONFIGURED PER MODEL AND THE MODELS DISAGREE. Fable 5.1
thinks unconditionally and rejects any explicit thinking config, so the
parameter is omitted for it entirely; Sonnet 5 needs `{"type": "adaptive"}`
to think at all. Opus 5.5 always thinks too, but accepts `{"type": "adaptive"}`
(only `disabled` and `budget_tokens` 400), so it takes Sonnet's config; its
effort DEFAULTS TO `medium`, not `high`, which is why EFFORT names it
explicitly. `budget_tokens` is dead on all three -- verified, not assumed:
it comes back 400 `"thinking.type.enabled" is not supported`. Depth is set
with `output_config.effort` on all three.
"""
from __future__ import annotations

import base64
import functools
import json
import mimetypes
import os
import shutil
import tempfile
import time
from pathlib import Path

import dotenv
from anthropic import AnthropicFoundry

from common.agent_workspace import (
    Transcript,
    image_placeholder,
    keep_native_logs,
    AGENT_SOLUTION_FILENAME,
    BINARY_DATA_EXTS,
    BINARY_DATA_HINT,
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
#: AND the package root, which is NOT REPO_ROOT. In the source tree
#: common/ sits one below the git root and the line above finds .env. In
#: the built package common/ sits one level deeper, so REPO_ROOT lands
#: one ABOVE the package root and that .env is never read -- every judged
#: task then dies on "no Azure credential". call_gemini and call_grok have
#: carried this second line for a while; claude and gpt did not, which is
#: why those two were the routes that failed. make_release seeds the
#: package .env this reads.
dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")

#: Deployment names on the Azure resource, which happen to equal the
#: first-party model ids. Overridable because a second resource would very
#: likely label them differently.
MODELS = {
    "sonnet5": os.getenv("CLAUDE_MODEL_SONNET5", "claude-sonnet-5"),
    "fable51": os.getenv("CLAUDE_MODEL_FABLE51", "claude-fable-5-1"),
    "opus55": os.getenv("CLAUDE_MODEL_OPUS55", "claude-opus-5-5"),
}

DEFAULT_MODEL = os.getenv("CLAUDE_DEFAULT_MODEL", "sonnet5")

#: `xhigh` is the sweet spot for agentic/coding work on both of these; the
#: judge path drops to `high`, since it only has to justify a score.
EFFORT = {"sonnet5": "xhigh", "fable51": "xhigh", "opus55": "xhigh"}

DISPLAY_NAMES = {"sonnet5": "Claude Sonnet 5", "fable51": "Claude Fable 5.1",
                 "opus55": "Claude Opus 5.5"}

AZURE_RESOURCE = (os.getenv("AZURE_CLAUDE_RESOURCE")
                  or "jenni-m7rybsi6-eastus2")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

#: Non-streaming ceiling. The SDK wants streaming for anything much larger,
#: and neither a judge call nor a solver turn needs it.
DEFAULT_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "16000"))

CALL_RETRIES = int(os.getenv("CLAUDE_CALL_RETRIES", "5"))
CALL_RETRY_BACKOFF_S = int(os.getenv("CLAUDE_CALL_BACKOFF_S", "10"))

#: SOME FAILURES ARE NOT WORTH A SECOND TRY, and retrying them is not free.
#: Measured twice: task 59 passed an alias the endpoint does not have and
#: paid 340 s per model in retries of a 404; task 75 ran on a machine whose
#: proxy is a socks4 URL httpx will never accept and paid 315 s per model
#: of the identical error. Five inner attempts with a growing backoff,
#: inside three outer ones, is fifteen repetitions of a sentence that was
#: true the first time.
#:
#: A transport fault -- a timeout, a reset, a rate limit, a 5xx -- is worth
#: repeating. A configuration fault is not: nothing between two attempts
#: changes a deployment name, an API key or a proxy scheme. These are
#: matched against the exception's text, lower-cased.
FATAL_SIGNATURES = (
    "unknown scheme for proxy",
    "deploymentnotfound",
    "invalid api key",
    "invalid_api_key",
    "authentication_error",
    "unsupported_country_region_territory",
    "no such deployment",
)


def is_configuration_error(exc) -> bool:
    """True when asking again cannot possibly help."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(sign in text for sign in FATAL_SIGNATURES)

#: Tool names as the Claude Code agent sees them. Its shell is the MCP tool
#: below, which Claude Code prefixes with `mcp__<server>__`.
AGENT_READ_TOOL = "Read"
AGENT_BASH_TOOL = "mcp__workspace__bash"

AGENT_TASK_SUFFIX = agent_task_suffix(bash_tool=AGENT_BASH_TOOL)

#: The same wording for `run_agent_shared_tools`, which names the four
#: shared tools.
SHARED_READ_TOOL = "read_file"
SHARED_BASH_TOOL = "bash"

#: BINARY_DATA_EXTS / BINARY_DATA_HINT / attach_files / attached_files_note
#: are re-exported above: `judge_runner` imports them from this module.


def resolve_model(model: str) -> tuple[str, str | None]:
    """'sonnet5'/'opus55'/'fable51' -> (deployment id, effort).

    Returns a pair rather than a string because callers unpack it; a full
    id passes through with the default effort.
    """
    if model in MODELS:
        return MODELS[model], EFFORT.get(model)
    return model, EFFORT.get(DEFAULT_MODEL)


_resolve_model = resolve_model


@functools.lru_cache(maxsize=1)
def client():
    """The Foundry client, built once and reused."""
    if not AZURE_API_KEY:
        raise RuntimeError(
            "no Azure credential for Claude: set AZURE_API_KEY in .env")
    return AnthropicFoundry(api_key=AZURE_API_KEY, resource=AZURE_RESOURCE)


def thinking_config(model: str):
    """Fable 5.1 rejects an explicit thinking config; Sonnet 5 requires one,
    and Opus 5.5 accepts it (adaptive is its only mode anyway).

    Returns None to mean "omit the parameter", which is not the same as
    disabling thinking -- on Fable 5.1 omitting it is the only way to get
    its always-on thinking without a 400.
    """
    if MODELS.get(model, model) == MODELS["fable51"]:
        return None
    return {"type": "adaptive"}


#: `cache_control` marks a prefix boundary: everything BEFORE it is cached
#: and re-read at a fraction of the input price on the next turn. The API
#: allows at most four, and this module uses three -- tools, the opening
#: user message, and one rolling marker on the newest tool results.
CACHE_MARK = {"type": "ephemeral"}


def _tools(cache: bool = True):
    """The shared specs as Anthropic tool definitions.

    Tools render before system and messages, never change during a run, and
    are several KB of JSON -- so the last one carries a cache breakpoint and
    the whole block is read from cache on every turn after the first.
    """
    defs = [{"name": s["name"], "description": s["description"],
             "input_schema": s["parameters"]} for s in TOOL_SPECS]
    if cache and defs:
        defs[-1] = {**defs[-1], "cache_control": CACHE_MARK}
    return defs


def _mark_rolling_cache(messages) -> None:
    """Move the rolling breakpoint to the newest user message.

    An agent transcript only ever grows at the end, so the whole history up
    to the last tool result is a stable prefix. Marking it means each turn
    re-reads the conversation from cache instead of paying full input price
    for it again -- which is the thing that makes a 40-turn CAD run
    affordable. The previous marker is cleared first: the cap is four, and
    leaving one per turn would blow through it by turn five.
    """
    last_list = None
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict):
                block.pop("cache_control", None)
        last_list = content
    if last_list and isinstance(last_list[-1], dict):
        last_list[-1]["cache_control"] = CACHE_MARK


def _media_block(path):
    """An image or PDF as a content block; anything else as labelled text."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or ""
    if mime.startswith("image/"):
        return {"type": "image", "source": {
            "type": "base64", "media_type": mime,
            "data": base64.b64encode(path.read_bytes()).decode("ascii")}}
    if mime == "application/pdf":
        return {"type": "document", "source": {
            "type": "base64", "media_type": "application/pdf",
            "data": base64.b64encode(path.read_bytes()).decode("ascii")}}
    return {"type": "text",
            "text": f"--- {path.name} ---\n"
                    f"{path.read_text(errors='replace')}"}


def _create(messages, *, model, tools, max_tokens, effort, system=None):
    model_id, default_effort = resolve_model(model)
    kwargs = dict(model=model_id, max_tokens=max_tokens, messages=messages,
                  output_config={"effort": effort or default_effort})
    think = thinking_config(model)
    if think is not None:
        kwargs["thinking"] = think
    if tools:
        kwargs["tools"] = tools
    if system:
        kwargs["system"] = system
    return client().messages.create(**kwargs)


def cache_usage(resp) -> dict:
    """The two numbers that say whether caching is actually working.

    `cache_read_input_tokens` staying at zero across turns means something
    is invalidating the prefix, and that is invisible without looking.
    """
    u = getattr(resp, "usage", None)
    return {
        "input": getattr(u, "input_tokens", None),
        "cache_read": getattr(u, "cache_read_input_tokens", None),
        "cache_write": getattr(u, "cache_creation_input_tokens", None),
        "output": getattr(u, "output_tokens", None),
    }


def _refusal_error(resp):
    """Safety classifiers decline with HTTP 200 and stop_reason 'refusal'.

    Worth naming rather than letting it surface as an empty answer: a
    refused judge call and a model that simply said nothing need different
    responses from whoever reads the log.
    """
    if getattr(resp, "stop_reason", None) != "refusal":
        return None
    det = getattr(resp, "stop_details", None)
    return RuntimeError(
        f"refused by the model (category={getattr(det, 'category', None)}): "
        f"{getattr(det, 'explanation', '') or 'no explanation given'}")


def _text_of(resp) -> str:
    return "".join(b.text for b in resp.content
                   if getattr(b, "type", "") == "text").strip()


def run_agent_shared_tools(prompt: str, *, cwd=None, model: str = DEFAULT_MODEL,
              tools=None, max_turns: int = DEFAULT_AGENT_MAX_TURNS,
              effort: str | None = None, files=(), inline_files=(),
              announce_files: bool = True, require_file: str | None = None,
              max_tokens: int = DEFAULT_MAX_TOKENS, system: str | None = None,
              verbose: bool = True):
    """The shared four-tool loop GPT and Gemini run. Returns (final_text, meta).

    Not what try_model uses for Claude any more (see `run_agent`); kept so
    Claude can still be scored on the identical tool surface. tools=[] means "no tools at all" -- that is how a one-turn call asks for
    a plain answer, and it is why this defaults to None rather than to the
    tool list.

    files: copied into cwd and announced, as in the other two routes.
    inline_files: sent as content blocks instead (images and PDFs, which
    `read_file` cannot open).
    """
    if files and cwd is None:
        raise ValueError("run_agent(files=...) needs a cwd to copy them into")
    if cwd is not None:
        cwd = Path(cwd)
        cwd.mkdir(parents=True, exist_ok=True)

    if files:
        names = attach_files(cwd, files)
        if announce_files:
            prompt = prompt + attached_files_note(
                names, read_tool=SHARED_READ_TOOL, bash_tool=SHARED_BASH_TOOL)

    content = [{"type": "text", "text": prompt}]
    content += [_media_block(f) for f in inline_files]
    # The opening message carries the whole task: instructions, the staged
    # file list, and any drawings. It never changes again, so it is worth a
    # breakpoint of its own.
    content[-1] = {**content[-1], "cache_control": CACHE_MARK}
    messages = [{"role": "user", "content": content}]

    tool_defs = _tools() if tools is None else list(tools)
    cache_stats = []
    transcript = Transcript()
    final_text, nudged, turn = "", False, 0

    while turn < max_turns:
        turn += 1
        _mark_rolling_cache(messages)
        resp = _create(messages, model=model, tools=tool_defs,
                       max_tokens=max_tokens, effort=effort, system=system)
        cache_stats.append(cache_usage(resp))

        refusal = _refusal_error(resp)
        if refusal is not None:
            raise refusal

        text = _text_of(resp)
        calls = [b for b in resp.content
                 if getattr(b, "type", "") == "tool_use"]
        transcript.append({"role": "assistant", "content": text,
                           "tool_calls": [{"name": c.name, "input": c.input}
                                          for c in calls],
                           "stop_reason": resp.stop_reason})

        # The whole content list goes back, not the text pulled out of it:
        # thinking blocks have to be echoed unchanged on the same model.
        messages.append({"role": "assistant", "content": resp.content})

        if not calls:
            if (require_file and cwd is not None
                    and not (cwd / require_file).is_file()
                    and not nudged and turn < max_turns):
                nudged = True
                note = (f"`{require_file}` does not exist in the working "
                        "directory yet -- your work is only graded from "
                        "that file.  Please continue and save it.")
                messages.append({"role": "user", "content": note})
                transcript.append({"role": "user", "content": note})
                if verbose:
                    print(f"    [claude turn {turn}] stopped without "
                          f"{require_file}; nudged once")
                continue
            final_text = text
            break

        results = []
        for c in calls:
            result = clip(dispatch(cwd or Path.cwd(), c.name, dict(c.input)))
            if verbose:
                preview = str(dict(c.input))[:110]
                print(f"    [claude turn {turn}] {c.name}({preview})"
                      f" -> {len(result)} chars")
            results.append({"type": "tool_result", "tool_use_id": c.id,
                            "content": result})
            transcript.append({"role": "tool", "name": c.name,
                               "content": result})
        messages.append({"role": "user", "content": results})
    else:
        if verbose:
            print(f"    [claude] turn budget exhausted ({max_turns})")

    model_id, _ = resolve_model(model)
    return final_text, {"model": model_id, "turns": turn,
                        "transcript": transcript, "nudged": nudged,
                        "cache": cache_stats}


# ---------------------------------------------------------------------------
# Claude Code agent
# ---------------------------------------------------------------------------

#: Claude Code's own tools the agent keeps. Bash is deliberately absent (its
#: shell is `mcp__workspace__bash`), and so are WebFetch/WebSearch.
CC_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit",
            "TodoWrite", "Task"]

#: Which argument of each file tool names a path, for the workspace guard.
CC_PATH_ARGS = {"Read": ("file_path",), "Write": ("file_path",),
                "Edit": ("file_path",), "NotebookEdit": ("notebook_path",),
                "Glob": ("path", "pattern"), "Grep": ("path",)}

#: Credentials that would make Claude Code pick a provider other than
#: Foundry, and the marker of the Claude Code session this may run inside.
#: Hidden from the child for the length of a run.
CC_ENV_HIDDEN = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                 "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDECODE",
                 "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")


def _cc_cli():
    """The installed `claude`, which has the Foundry backend; else the SDK's own."""
    return os.getenv("CLAUDE_CODE_CLI") or shutil.which("claude")


def _outside(cwd: Path, value) -> bool:
    if not value or not isinstance(value, str):
        return False
    p = Path(value)
    if not p.is_absolute():
        p = cwd / p
    try:
        p.resolve().relative_to(cwd.resolve())
        return False
    except ValueError:
        return True


def _workspace_guard(cwd: Path):
    """PreToolUse hook: no file tool may look outside the workspace."""
    async def guard(input_data, tool_use_id, context):
        name = input_data.get("tool_name", "")
        args = input_data.get("tool_input") or {}
        for key in CC_PATH_ARGS.get(name, ()):
            if _outside(cwd, args.get(key)):
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason":
                        f"{name} is limited to the working directory {cwd}; "
                        f"{args.get(key)!r} is outside it."}}
        return {}
    return guard


def _bash_server(cwd: Path):
    """The agent's shell: the shared `bash`, so try_model's sandbox applies."""
    import asyncio

    from claude_agent_sdk import create_sdk_mcp_server, tool

    spec = next(s for s in TOOL_SPECS if s["name"] == "bash")

    @tool("bash", spec["description"], spec["parameters"])
    async def bash(args):
        # dispatch looks TOOL_IMPL up at call time, so try_model's
        # install_bash_backend redirect is honoured. It blocks for up to the
        # command's timeout, hence the thread.
        out = await asyncio.to_thread(dispatch, cwd, "bash", dict(args))
        return {"content": [{"type": "text", "text": clip(out)}]}

    return create_sdk_mcp_server("workspace", tools=[bash])


def _cc_env(model_id: str, config_dir: str) -> dict:
    from common.agent_cli import MODEL_REQUEST_TIMEOUT_S
    from common.agent_workspace import BASH_TIMEOUT_MAX_S
    return {
        "CLAUDE_CODE_USE_FOUNDRY": "1",
        "ANTHROPIC_FOUNDRY_RESOURCE": AZURE_RESOURCE,
        "ANTHROPIC_FOUNDRY_API_KEY": AZURE_API_KEY or "",
        #: Claude Code asks for its "haiku"/"sonnet"/"opus" aliases for
        #: background work and subagents. Only the chosen deployment exists
        #: on the resource, so every alias maps to it.
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_id,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model_id,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model_id,
        "CLAUDE_CONFIG_DIR": config_dir,
        #: An MCP call times out long before a SolidWorks build does.
        "MCP_TOOL_TIMEOUT": str((BASH_TIMEOUT_MAX_S + 60) * 1000),
        #: Opus 5.5 at xhigh can think for over five minutes before its
        #: first visible byte, and Claude Code's own request timeout then
        #: drops the request and retries it from scratch -- a living-hinge
        #: run did that every 5 min for half an hour and never came back.
        "API_TIMEOUT_MS": str(MODEL_REQUEST_TIMEOUT_S * 1000),
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def _block_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") if isinstance(c, dict) else str(c)
                         for c in content)
    return "" if content is None else str(content)


def run_agent(prompt: str, *, cwd=None, model: str = DEFAULT_MODEL,
              max_turns: int = DEFAULT_AGENT_MAX_TURNS,
              effort: str | None = None, files=(), inline_files=(),
              announce_files: bool = True, require_file: str | None = None,
              system: str | None = None, verbose: bool = True, **_ignored):
    """Claude Code on the task. Returns (final_text, meta), like the other routes.

    Claude Code decides when it is finished. `max_turns` caps TOOL CALLS
    over the whole run, nudge included -- the same unit and scope as the
    Codex and Gemini CLI routes, so `budget_exhausted` means one thing
    across models. It is enforced exactly: a PreToolUse hook counts every
    call (subagents' too) and denies the first one over the budget, and the
    session is then interrupted. Claude Code's own --max-turns (model round
    trips) stays on as a looser backstop.
    If it stops without `require_file`, it is asked once, in the same
    session, to save it.
    """
    import asyncio

    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions,
                                  ClaudeSDKClient, HookMatcher, ResultMessage,
                                  SystemMessage, TextBlock, ThinkingBlock,
                                  ToolResultBlock, ToolUseBlock, UserMessage)

    if cwd is None:
        raise ValueError("the Claude Code agent needs a cwd to work in")
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    if files:
        names = attach_files(cwd, files)
        if announce_files:
            prompt = prompt + attached_files_note(
                names, read_tool=AGENT_READ_TOOL, bash_tool=AGENT_BASH_TOOL)
    if not AZURE_API_KEY:
        raise RuntimeError(
            "no Azure credential for Claude: set AZURE_API_KEY in .env")

    model_id, default_effort = resolve_model(model)
    config_dir = tempfile.mkdtemp(prefix="claude_cfg_")
    state = {"turns": 0, "calls": 0, "cost": 0.0, "final": "",
             "subtype": None, "error": None, "cli": None, "exhausted": False}
    # try_model may provide a live progress channel for partial-run accounting.
    # Claude Code already owns the authoritative tool-call counter in its
    # PreToolUse hook below.  Publishing that existing counter lets try_model
    # preserve work completed before Ctrl-C or another abnormal exit prevents
    # run_agent() from reaching its normal return.
    progress = _ignored.get("progress")
    if progress is not None:
        progress.update(model=model_id, turns=0, harness="claude code",)

    async def budget(input_data, tool_use_id, context):
        state["calls"] += 1
        # PreToolUse is the authoritative place where Claude Code counts benchmark
        # tool calls, including built-in file tools and subagent calls that do not
        # necessarily pass through WorkspaceMCP.  Publish the same counter rather
        # than trying to reconstruct it elsewhere.
        if progress is not None:
            # PreToolUse also observes the first call that is denied for exceeding the
            # budget.  That denied attempt is not counted by run_agent's normal metadata,
            # so publish the same capped value here rather than making interrupted runs
            # appear to have spent one extra tool call.
            progress.update(turns=min(state["calls"], max_turns))
        if state["calls"] <= max_turns:
            return {}
        state["exhausted"] = True
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason":
                f"tool budget exhausted ({max_turns} calls)"}}

    options = ClaudeAgentOptions(
        model=model_id,
        cwd=str(cwd),
        cli_path=_cc_cli(),
        system_prompt=({"type": "preset", "preset": "claude_code",
                        "append": system} if system
                       else {"type": "preset", "preset": "claude_code"}),
        tools=CC_TOOLS,
        mcp_servers={"workspace": _bash_server(cwd)},
        allowed_tools=CC_TOOLS + [AGENT_BASH_TOOL],
        permission_mode="bypassPermissions",
        hooks={"PreToolUse": [HookMatcher(
            hooks=[budget, _workspace_guard(cwd)])]},
        #: round trips, a backstop only: a run makes at most one per tool
        #: call plus its closing replies, so this never trips before the
        #: tool-call budget does
        max_turns=max_turns + 5,
        effort=effort or default_effort,
        env=_cc_env(model_id, config_dir),
        #: One stream-json line per message, and a Read of a rendered
        #: drawing page is a base64 image inside one line. The SDK's 1 MB
        #: default killed a sling-lift run on its second page.
        max_buffer_size=64 * 1024 * 1024,
        #: Stream thinking SUMMARIES instead of hiding thinking. Hidden
        #: thinking sends no bytes at all until the model is done -- not even
        #: pings (measured: 493 s of silence) -- and Azure Foundry drops a
        #: request that is silent for ~10 min. Summaries keep bytes flowing
        #: (longest gap 7.6 s over an 11.5 min request that completed) and
        #: leave the model's reasoning unchanged. They also land in the
        #: transcript, so a Claude run's reasoning is on record.
        #: the CLI flag, not the showThinkingSummaries setting: headless
        #: (stream-json) runs honour only an explicit display choice
        extra_args={"thinking-display": "summarized"},
    )

    content = [{"type": "text", "text": prompt}]
    content += [_media_block(f) for f in inline_files]

    async def opening():
        yield {"type": "user", "session_id": "",
               "parent_tool_use_id": None,
               "message": {"role": "user", "content": content}}

    transcript, usage, names_by_id = Transcript(), [], {}
    # The transcript is created after the budget hook because ClaudeAgentOptions
    # needs that hook during setup.  Publish the live object once it exists;
    # subsequent appends remain visible in snapshots without maintaining a second
    # transcript.
    if progress is not None:
        progress.update(transcript=transcript)
    inputs_by_id = {}

    from common.agent_cli import DEFAULT_IDLE_S, DEFAULT_WALL_S
    deadline = time.time() + DEFAULT_WALL_S

    async def messages(client):
        """receive_response(), with the same idle and wall-clock ceilings
        the Codex and Gemini CLI runs have: a message must arrive within
        the idle limit, and the run must end by the deadline."""
        it = client.receive_response().__aiter__()
        while True:
            left = deadline - time.time()
            wait = min(DEFAULT_IDLE_S, left)
            if wait <= 0:
                state["error"] = "claude code run stopped: no progress (wall)"
                if progress is not None:
                    progress.update(error=state["error"])
                return
            try:
                yield await asyncio.wait_for(it.__anext__(), timeout=wait)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                state["error"] = ("claude code run stopped: no progress ("
                                  + ("wall" if wait < DEFAULT_IDLE_S
                                     else "idle") + ")")
                if progress is not None:
                    progress.update(error=state["error"])
                return

    async def drain(client):
        async for msg in messages(client):
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                state["cli"] = (msg.data or {}).get("claude_code_version")
                state["tools"] = (msg.data or {}).get("tools")
            elif isinstance(msg, AssistantMessage):
                text = "".join(b.text for b in msg.content
                               if isinstance(b, TextBlock)).strip()
                calls = [b for b in msg.content if isinstance(b, ToolUseBlock)]
                thought = "\n".join(b.thinking for b in msg.content
                                    if isinstance(b, ThinkingBlock)
                                    and b.thinking).strip()
                if thought:
                    transcript.append({"role": "assistant", "content": "",
                                       "thinking": thought})
                for c in calls:
                    names_by_id[c.id] = c.name
                    inputs_by_id[c.id] = c.input
                    if verbose:
                        print(f"    [claude code] {c.name}"
                              f"({str(c.input)[:110]})")
                if text or calls:
                    transcript.append({
                        "role": "assistant", "content": text,
                        "tool_calls": [{"name": c.name, "input": c.input}
                                       for c in calls]})
                if state["exhausted"] and not state.get("interrupted"):
                    state["interrupted"] = True
                    if verbose:
                        print(f"    [claude code] tool budget exhausted "
                              f"({max_turns}); interrupting")
                    await client.interrupt()
            elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for b in msg.content:
                    if isinstance(b, ToolResultBlock):
                        text = _block_text(b.content)
                        images = [c for c in (b.content if isinstance(
                            b.content, list) else []) if isinstance(c, dict)
                            and c.get("type") == "image"]
                        if images:
                            args = inputs_by_id.get(b.tool_use_id) or {}
                            src = (args.get("file_path") or args.get("path")
                                   or "?")
                            size = sum(len(((c.get("source") or {})
                                            .get("data")) or "")
                                       for c in images) * 3 // 4
                            note = image_placeholder(
                                src, f"(~{size // 1024} KB)")
                            text = f"{text}\n{note}" if text else note
                        transcript.append({
                            "role": "tool",
                            "name": names_by_id.get(b.tool_use_id, "?"),
                            "content": clip(text)})
            elif isinstance(msg, ResultMessage):
                state["turns"] += msg.num_turns or 0
                state["cost"] += msg.total_cost_usd or 0.0
                state["subtype"] = msg.subtype
                usage.append(msg.usage or {})
                if (msg.is_error and msg.subtype != "error_max_turns"
                        and not state["exhausted"]):
                    state["error"] = msg.result or msg.subtype
                    if progress is not None:
                        progress.update(error=state["error"])
                if msg.result:
                    state["final"] = msg.result

    async def main():
        async with ClaudeSDKClient(options=options) as client:
            await client.query(opening())
            await drain(client)
            if (require_file and not state["error"]
                    and not state["exhausted"]
                    and state["subtype"] != "error_max_turns"
                    and not (cwd / require_file).is_file()):
                note = (f"`{require_file}` does not exist in the working "
                        "directory yet -- your work is only graded from "
                        "that file.  Please continue and save it.")
                transcript.append({"role": "user", "content": note})
                if verbose:
                    print(f"    [claude code] stopped without {require_file};"
                          " nudged once")
                state["nudged"] = True
                await client.query(note)
                await drain(client)

    hidden = {k: os.environ.pop(k) for k in CC_ENV_HIDDEN if k in os.environ}
    try:
        asyncio.run(main())
    finally:
        os.environ.update(hidden)
        #: Claude Code's own session files: every message, retry, timeout
        #: and compaction, subagents included
        native = keep_native_logs(config_dir, ["projects/**/*.jsonl"],
                                  "claude")
        shutil.rmtree(config_dir, ignore_errors=True)

    if state["error"] and not state["final"]:
        raise RuntimeError(f"Claude Code run failed: {state['error']}")
    exhausted = state["exhausted"] or state["subtype"] == "error_max_turns"
    if verbose:
        print(f"    [claude code] {min(state['calls'], max_turns)} tool calls, "
              f"{state['turns']} model turns, ${state['cost']:.2f}"
              + (f" -- turn budget exhausted ({max_turns})" if exhausted
                 else ""))
    return state["final"], {
        "model": model_id,
        #: tool calls, like the Codex and Gemini CLI routes; the ceiling
        #: itself when it ran out, which is what try_model's `_exhausted`
        #: reads
        "turns": max_turns if exhausted else min(state["calls"], max_turns),
        "model_turns": state["turns"],
        "transcript": transcript, "nudged": bool(state.get("nudged")),
        "cache": usage, "cost_usd": round(state["cost"], 4),
        "harness": f"claude code {state['cli'] or '?'}",
        "error": state["error"],
        "native_logs": native,
        #: what Claude Code reported it could use -- the proof that its own
        #: Bash and the web tools were really off for this run
        "tools": state.get("tools"),
    }


def label(model: str = DEFAULT_MODEL) -> str:
    model_id, effort = resolve_model(model)
    pretty = DISPLAY_NAMES.get(model, model)
    return f"{pretty} (Claude Code, Azure, {model_id}, effort={effort})"


def solve(user_text, image_files=(), pdf_pages=(), pdf_files=(),
          code_files=(), pointcloud_files=(), *, model: str = DEFAULT_MODEL):
    workdir = tempfile.mkdtemp(prefix="claude_agent_")
    try:
        files = list(code_files) + list(pointcloud_files)
        text, meta = run_agent(
            user_text + AGENT_TASK_SUFFIX,
            cwd=workdir, model=model, files=files,
            inline_files=(list(image_files) + list(pdf_pages)
                          + list(pdf_files)),
            require_file=AGENT_SOLUTION_FILENAME,
        )
        return read_back_solution(workdir, text), meta
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def call_claude(prompt: str, *, model: str = DEFAULT_MODEL,
                images=(), effort: str = "high",
                retries: int = CALL_RETRIES) -> str:
    """Judge call: text (and optionally pictures) in, text out, no tools."""
    last = None
    for attempt in range(retries):
        try:
            content = [{"type": "text", "text": prompt}]
            content += [_media_block(i) for i in images]
            resp = _create([{"role": "user", "content": content}],
                           model=model, tools=None,
                           max_tokens=DEFAULT_MAX_TOKENS, effort=effort)
            refusal = _refusal_error(resp)
            if refusal is not None:
                raise refusal
            text = _text_of(resp)
            if text:
                return text
            last = RuntimeError(
                f"empty reply (stop_reason={resp.stop_reason})")
        except Exception as exc:                            # noqa: BLE001
            last = exc
            if is_configuration_error(exc):
                raise RuntimeError(
                    f"call_claude cannot run here, and trying again will "
                    f"not change it: {exc}") from exc
        if attempt < retries - 1:
            time.sleep(CALL_RETRY_BACKOFF_S * (attempt + 1))
    raise RuntimeError(
        f"call_claude failed after {retries} attempts: {last}") from last


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(label(which))
    print(call_claude("In one sentence, what is a living hinge?", model=which))
