"""GLM route: GLM on Microsoft Foundry.

Uses the OpenAI-compatible endpoint exposed by Microsoft Foundry.

The solver uses the same provider-neutral tools from `agent_workspace`
as the GPT, Claude and Gemini routes so tasks are evaluated with the same
tool surface and prompt scaffolding.
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

GLM_CONFIG = {
    "api_key": (os.getenv("AZURE_GLM_API_KEY")
                or os.getenv("AZURE_FOUNDRY_API_KEY")
                or os.getenv("AZURE_API_KEY")),
    "endpoint": (os.getenv("AZURE_GLM_ENDPOINT") 
                 or os.getenv("AZURE_FOUNDRY_ENDPOINT")
                 or "https://jenni-m7rybsi6-eastus2.services.ai.azure.com/openai/v1"),
    "deployment_name": os.getenv("AZURE_GLM_DEPLOYMENT", "FW-GLM-5.3-Flash"),
}

DEFAULT_MODEL = GLM_CONFIG["deployment_name"]

DEFAULT_MAX_TOKENS = int(os.getenv("GLM_MAX_TOKENS", "32768"))
API_MAX_RETRIES = int(os.getenv("GLM_API_MAX_RETRIES", "6"))
API_RETRY_BASE_S = int(os.getenv("GLM_API_RETRY_BASE_S", "15"))
API_RETRY_MAX_S = int(os.getenv("GLM_API_RETRY_MAX_S", "120"))

AGENT_READ_TOOL = "read_file"
AGENT_BASH_TOOL = "bash"


def _image_data_url(path) -> str:
    """Encode an image as a data URL for the OpenAI-compatible API."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def client():
    if not GLM_CONFIG["api_key"]:
        raise RuntimeError("no Azure credential for GLM: " \
        "set AZURE_GLM_API_KEY (or AZURE_FOUNDRY_API_KEY) in .env")

    if not GLM_CONFIG["endpoint"]:
        raise RuntimeError("no Azure endpoint for GLM: " \
        "set AZURE_GLM_ENDPOINT (or AZURE_FOUNDRY_ENDPOINT) in .env")
    
    return OpenAI(base_url=GLM_CONFIG["endpoint"],
                  api_key=GLM_CONFIG["api_key"],
                  timeout=180.0,)


def _chat_create_with_retry(cl, kwargs, *, verbose=True):
    """Retry temporary GLM/Foundry rate-limit or capacity errors."""
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
                    f"    [glm] temporary API error; "
                    f"retry {attempt}/{API_MAX_RETRIES} in {wait_s}s"
                )

            time.sleep(wait_s)


def call_glm(prompt: str, *, model: str = DEFAULT_MODEL,
             images=(), image_b64=(),
             max_tokens: int = DEFAULT_MAX_TOKENS,
             return_raw: bool = False):
    
    content = [{"type": "text", "text": prompt}]

    for img in images:
        content.append({"type": "image_url",
                        "image_url": {"url": _image_data_url(img)}})
    for b64 in image_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + b64}})

    resp = client().chat.completions.create(
        model=model,
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
        content.append({"type": "image_url",
                        "image_url": {"url": _image_data_url(img)},})

    for b64 in image_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + b64},})

    return content


def run_agent_shared_tools(prompt: str, *, cwd, model: str = DEFAULT_MODEL,
                           images=(), image_b64=(), files=(),
                           announce_files: bool = True,
                           max_turns: int = DEFAULT_AGENT_MAX_TURNS,
                           max_tokens: int = DEFAULT_MAX_TOKENS,
                           require_file: str | None = None,
                           verbose: bool = True):
    """Tool-using glm agent loop on the Chat Completions API."""
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)

    if files:
        names = attach_files(cwd, files)
        if announce_files:
            prompt += attached_files_note(
                names, read_tool=AGENT_READ_TOOL, bash_tool=AGENT_BASH_TOOL)

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
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
        }

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
                    f"    [glm] malformed tool-call JSON rejected by API; "
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
            "function_calls": [{
                    "name": call.function.name,
                    "arguments": call.function.arguments,}
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
                        f"    [glm turn {turn}] stopped without "
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
                preview = (call.function.arguments or ""
                           )[:110].replace("\n", " ")

                print(f"    [glm turn {turn}] "
                      f"{call.function.name}({preview}) "
                      f"-> {len(result)} chars")

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
            print(f"    [glm] turn budget exhausted ({max_turns})")

    return final_text, {
        "model": model,
        "model_key": model,
        "turns": turn,
        "transcript": transcript,
        "nudged": nudged,
    }


# ---------------------------------------------------------------------------
# Kimi Code agent (2026-09-25) -- what try_model runs for GLM. Codex
# was tried first and fails: Foundry's GLM errors on every Codex tool call. See
# common/kimi_code_agent.py: Moonshot's own harness, confined to the
# workspace, on the Foundry deployment through a small request filter.
# ---------------------------------------------------------------------------
from common import kimi_code_agent as _kc  # noqa: E402

AGENT_BASH_TOOL = _kc.BASH_TOOL
AGENT_READ_TOOL = _kc.READ_TOOL


def run_agent(prompt: str, *, cwd,
              images=(), image_b64=(), files=(),
              announce_files: bool = True,
              max_turns: int = DEFAULT_AGENT_MAX_TURNS,
              require_file: str | None = None,
              verbose: bool = True, **_ignored):
    """GLM on Kimi Code. Returns (final_text, meta), like the other routes."""
    import base64 as _b64
    model_id = GLM_CONFIG["deployment_name"]
    tmp = Path(tempfile.mkdtemp(prefix="glm_img_"))
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
            prompt, cwd=cwd, model_id=model_id, base_url=GLM_CONFIG["endpoint"],
            api_key=GLM_CONFIG["api_key"], max_turns=max_turns, files=files,
            inline_files=inline, announce_files=announce_files,
            require_file=require_file, label="glm", verbose=verbose,
            progress=progress)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def label() -> str:
    return (f"GLM (Kimi Code, Azure Foundry, "
            f"deployment={GLM_CONFIG['deployment_name']})")


def solve(user_text, image_files=(), pdf_pages=(),
          pdf_files=(), code_files=(), pointcloud_files=(),):
    """Agent solver, mirroring call_gpt.solve()."""

    workdir = tempfile.mkdtemp(prefix="glm_agent_")

    try:
        files = (list(code_files) + list(pdf_files) + list(pointcloud_files))

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


if __name__ == "__main__":
    print(label())
    print(call_glm("Reply with exactly: OK"))