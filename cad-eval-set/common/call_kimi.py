"""Kimi route: Moonshot AI Kimi models on Microsoft Foundry.

Uses the OpenAI-compatible endpoint exposed by Microsoft Foundry.

The solver uses the same provider-neutral tools from `agent_workspace`
as the GPT, Claude, Gemini and Grok routes so tasks are evaluated with
the same tool surface and prompt scaffolding.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import tempfile
import time
from pathlib import Path

import dotenv
from openai import APITimeoutError, BadRequestError, OpenAI, RateLimitError

from common.agent_workspace import (
    Transcript,
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
dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")


MODELS = {
    "k27code": os.getenv("KIMI_MODEL_K27_CODE", "Kimi-K2.7-Code",),
    "k3": os.getenv("KIMI_MODEL_K3", "FW-Kimi-K3",),
}

MODEL_CONFIG = {
    "k27code": {"reasoning_effort": None,},
    "k3": {"reasoning_effort": "high", "max_tokens": 65536,},
}

#: Other spellings people use for the same keys -- try_model's own help
#: advertises `kimi:k2.7-code`, and .env's KIMI_DEFAULT_MODEL uses it too.
MODEL_ALIASES = {"k2.7-code": "k27code", "k2.7code": "k27code",
                 "k27-code": "k27code", "kimi-k3": "k3"}

DEFAULT_MODEL = os.getenv("KIMI_DEFAULT_MODEL", "k27code",)
DEFAULT_MODEL = MODEL_ALIASES.get(DEFAULT_MODEL, DEFAULT_MODEL)

DISPLAY_NAMES = {
    "k27code": "Kimi K2.7 Code",
    "k3": "Kimi K3",
}

#: every route shares the eastus2 resource and its AZURE_API_KEY; the
#: specific names stay as overrides for a second resource
AZURE_API_KEY = (os.getenv("AZURE_KIMI_API_KEY")
                 or os.getenv("AZURE_FOUNDRY_API_KEY")
                 or os.getenv("AZURE_API_KEY"))

AZURE_ENDPOINT = (os.getenv("AZURE_KIMI_ENDPOINT")
                  or os.getenv("AZURE_FOUNDRY_ENDPOINT")
                  or "https://jenni-m7rybsi6-eastus2.services.ai.azure.com/openai/v1")

DEFAULT_MAX_TOKENS = int(os.getenv("KIMI_MAX_TOKENS", "32768"))
API_MAX_RETRIES = int(os.getenv("KIMI_API_MAX_RETRIES", "6"))
API_RETRY_BASE_S = int(os.getenv("KIMI_API_RETRY_BASE_S", "15"))
API_RETRY_MAX_S = int(os.getenv("KIMI_API_RETRY_MAX_S", "120"))

AGENT_READ_TOOL = "read_file"
AGENT_BASH_TOOL = "bash"
SHARED_READ_TOOL, SHARED_BASH_TOOL = AGENT_READ_TOOL, AGENT_BASH_TOOL

def resolve_model(model: str = DEFAULT_MODEL):
    model_id = MODELS.get(model, model)
    config = MODEL_CONFIG.get(model, {})
    return model_id, config


def _image_data_url(path) -> str:
    """Encode an image as a data URL for the OpenAI-compatible API."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def client():
    if not AZURE_API_KEY:
        raise RuntimeError("no Azure credential for Kimi: set AZURE_KIMI_API_KEY "
                           "(or AZURE_API_KEY) in .env")

    if not AZURE_ENDPOINT:
        raise RuntimeError("no Azure endpoint for Kimi: set AZURE_KIMI_ENDPOINT "
                           "(or AZURE_FOUNDRY_ENDPOINT) in .env")

    return OpenAI(base_url=AZURE_ENDPOINT, 
                  api_key=AZURE_API_KEY,
                  timeout=180.0,)


def _chat_create_with_retry(cl, kwargs, *, verbose=True):
    """Retry temporary Kimi/Foundry rate-limit or capacity errors."""
    attempt = 0

    while True:
        try:
            return cl.chat.completions.create(**kwargs)

        except (RateLimitError, APITimeoutError):
            attempt += 1
            if attempt > API_MAX_RETRIES:
                raise

            wait_s = min(
                API_RETRY_BASE_S * (2 ** (attempt - 1)),
                API_RETRY_MAX_S,
            )

            if verbose:
                print(
                    f"    [kimi] temporary API error; "
                    f"retry {attempt}/{API_MAX_RETRIES} in {wait_s}s"
                )

            time.sleep(wait_s)


def call_kimi(prompt: str, *, model: str = DEFAULT_MODEL, 
              images=(), image_b64=(),
              max_tokens: int = DEFAULT_MAX_TOKENS,
              return_raw: bool = False):
    
    model_id, _ = resolve_model(model)
    content = [{"type": "text", "text": prompt}]

    for img in images:
        content.append({"type": "image_url",
                        "image_url": {"url": _image_data_url(img)}})
    for b64 in image_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + b64}})

    resp = client().chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )

    text = resp.choices[0].message.content or ""
    return (text, resp.model_dump()) if return_raw else text


def _agent_tools():
    return [{"type": "function", "function": spec} for spec in TOOL_SPECS]


def _input_content(prompt, images=(), image_b64=()):
    content = [{"type": "text", "text": prompt}]

    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": _image_data_url(img)},
        })

    for b64 in image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + b64},
        })

    return content


def run_agent_shared_tools(prompt: str, *, cwd, model: str = DEFAULT_MODEL,
              images=(), image_b64=(), files=(),
              announce_files: bool = True,
              max_turns: int = DEFAULT_AGENT_MAX_TURNS,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              require_file: str | None = None,
              verbose: bool = True):
    """Tool-using Kimi agent loop on the Chat Completions API."""
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)

    if files:
        names = attach_files(cwd, files)
        if announce_files:
            prompt += attached_files_note(
                names, read_tool=SHARED_READ_TOOL, bash_tool=SHARED_BASH_TOOL)

    model_id, config = resolve_model(model)
    model_max_tokens = config.get("max_tokens", max_tokens)
    reasoning_effort = config.get("reasoning_effort")

    cl = client()
    tools = _agent_tools()

    messages = [{
        "role": "user",
        "content": _input_content(prompt, images, image_b64),
    }]

    transcript = Transcript()
    final_text = ""
    nudged = False
    turn = 0
    malformed_recoveries = 0
    MAX_MALFORMED_RECOVERIES = 3

    while turn < max_turns:
        turn += 1

        kwargs = {
            "model": model_id,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": model_max_tokens,
        }

        # K2.7 uses native always-on reasoning.
        # K3 supports reasoning_effort="max" through Chat Completions.
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            resp = _chat_create_with_retry(cl, kwargs, verbose=verbose)

        except BadRequestError as exc:
            err = str(exc)

            malformed = (
                "function.arguments must be valid JSON" in err
                or "tool call function.arguments must be valid JSON" in err
            )

            if not malformed or malformed_recoveries >= MAX_MALFORMED_RECOVERIES:
                raise

            malformed_recoveries += 1

            if verbose:
                print(
                    f"    [kimi] malformed tool-call JSON rejected by API; "
                    f"recovery {malformed_recoveries}/"
                    f"{MAX_MALFORMED_RECOVERIES}"
                )

            # Remove the most recent assistant tool-call exchange that Azure
            # refuses to accept back in conversation history.
            while messages and messages[-1].get("role") == "tool":
                messages.pop()

            if messages and getattr(messages[-1], "role", None) == "assistant":
                messages.pop()
            elif messages and messages[-1].get("role") == "assistant":
                messages.pop()

            recovery_note = (
                "Your previous tool call contained invalid JSON arguments and was "
                "rejected by the API. Retry the same intended action using valid "
                "JSON tool arguments only. Do not include any extra text or "
                "characters inside or after the tool-call arguments."
            )

            messages.append({
                "role": "user",
                "content": recovery_note,
            })

            transcript.append({
                "role": "user",
                "content": recovery_note,
            })

            # This was not a completed model turn.
            turn -= 1
            continue

        msg = resp.choices[0].message
        text = msg.content or ""
        calls = msg.tool_calls or []

        transcript.append({
            "role": "assistant",
            "content": text,
            "function_calls": [
                {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
                for call in calls
            ],
        })

        if not calls:
            missing = require_file and not (cwd / require_file).is_file()

            if missing and not nudged and turn < max_turns:
                nudged = True
                note = (
                    f"`{require_file}` does not exist in the working directory "
                    "yet -- your work is only graded from that file. "
                    "Please continue and save it."
                )

                messages.append(msg)
                messages.append({"role": "user", "content": note})
                transcript.append({"role": "user", "content": note})

                if verbose:
                    print(
                        f"    [kimi turn {turn}] stopped without "
                        f"{require_file}; nudged once"
                    )
                continue

            final_text = text
            break

        # Preserve the assistant's tool calls in conversation history.
        messages.append(msg)

        for call in calls:
            try:
                args = json.loads(call.function.arguments or "{}")
                result = dispatch(cwd, call.function.name, args)
            except json.JSONDecodeError as exc:
                result = (
                    f"ERROR: tool arguments were not valid JSON: {exc}"
                )

            result = clip(result)

            if verbose:
                preview = (
                    call.function.arguments or ""
                )[:110].replace("\n", " ")

                print(
                    f"    [kimi turn {turn}] "
                    f"{call.function.name}({preview}) "
                    f"-> {len(result)} chars"
                )

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

            transcript.append({
                "role": "tool",
                "call_id": call.id,
                "name": call.function.name,
                "content": result,
            })

    else:
        if verbose:
            print(f"    [kimi] turn budget exhausted ({max_turns})")

    return final_text, {
        "model": model_id,
        "model_key": model,
        "turns": turn,
        "transcript": transcript,
        "nudged": nudged,
    }



# ---------------------------------------------------------------------------
# Kimi Code agent (2026-09-25) -- what try_model runs for Kimi. See
# common/kimi_code_agent.py: Moonshot's own harness, confined to the
# workspace, on the Foundry deployment through a small request filter.
# ---------------------------------------------------------------------------
from common import kimi_code_agent as _kc  # noqa: E402

AGENT_BASH_TOOL = _kc.BASH_TOOL
AGENT_READ_TOOL = _kc.READ_TOOL


def run_agent(prompt: str, *, cwd, model: str = DEFAULT_MODEL,
              images=(), image_b64=(), files=(),
              announce_files: bool = True,
              max_turns: int = DEFAULT_AGENT_MAX_TURNS,
              require_file: str | None = None,
              verbose: bool = True, **_ignored):
    """Kimi Code on the task. Returns (final_text, meta), like the other routes."""
    import base64 as _b64
    model_id, _cfg = resolve_model(model)
    tmp = Path(tempfile.mkdtemp(prefix="kimi_img_"))
    inline = [str(i) for i in images]
    for n, b in enumerate(image_b64):
        p = tmp / f"image_{n}.png"
        p.write_bytes(_b64.b64decode(b))
        inline.append(str(p))
    # try_model may provide a provider-neutral live progress channel.  This route
    # delegates the actual agent loop to Kimi Code, so it only forwards the object;
    # the shared harness remains the single source of truth for tool-call counts.
    progress = _ignored.get("progress")
    try:
        return _kc.run_kimi_code(
            prompt, cwd=cwd, model_id=model_id, base_url=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY, max_turns=max_turns, files=files,
            inline_files=inline, announce_files=announce_files,
            require_file=require_file, label=f"kimi {model}", verbose=verbose,
            progress=progress,)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def label(model: str = DEFAULT_MODEL) -> str:
    model_id, model_config = resolve_model(model)
    pretty = DISPLAY_NAMES.get(model, model)

    reasoning_effort = model_config.get("reasoning_effort")
    reasoning_label = reasoning_effort or "native"

    return (
        f"{pretty} (Kimi Code, Azure Foundry, "
        f"deployment={model_id}, reasoning={reasoning_label})"
    )


def solve(user_text, image_files=(), pdf_pages=(),
          pdf_files=(), code_files=(), pointcloud_files=(),
          *, model: str = DEFAULT_MODEL,):
    """Agent solver, mirroring call_gpt.solve()."""

    workdir = tempfile.mkdtemp(prefix="kimi_agent_")

    try:
        files = (list(code_files) + list(pdf_files) + list(pointcloud_files))

        text, meta = run_agent(
            user_text + agent_task_suffix(bash_tool=AGENT_BASH_TOOL),
            cwd=workdir,
            model=model,
            images=list(image_files) + list(pdf_pages),
            files=files,
            require_file=AGENT_SOLUTION_FILENAME,
        )
        return read_back_solution(workdir, text), meta

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    print(label(which))
    print(call_kimi("Reply with exactly: OK", model=which,))