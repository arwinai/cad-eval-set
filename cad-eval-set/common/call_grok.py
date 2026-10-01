"""Grok route: xAI Grok on Microsoft Foundry.

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
from openai import OpenAI

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

GROK_CONFIG = {
    #: every route shares the eastus2 resource and its AZURE_API_KEY; the
    #: specific names stay as overrides for a second resource
    "api_key": (os.getenv("AZURE_GROK_API_KEY")
                or os.getenv("AZURE_FOUNDRY_API_KEY")
                or os.getenv("AZURE_API_KEY")),
    "endpoint": (os.getenv("AZURE_GROK_ENDPOINT")
                 or os.getenv("AZURE_FOUNDRY_ENDPOINT")
                 or "https://jenni-m7rybsi6-eastus2.services.ai.azure.com/openai/v1"),
    "deployment_name": os.getenv("AZURE_GROK_DEPLOYMENT", "grok-4.6"),
}

DEFAULT_MODEL = GROK_CONFIG["deployment_name"]

DEFAULT_MAX_TOKENS = int(os.getenv("GROK_MAX_TOKENS", "32768"))

AGENT_READ_TOOL = "read_file"
AGENT_BASH_TOOL = "bash"
SHARED_READ_TOOL, SHARED_BASH_TOOL = AGENT_READ_TOOL, AGENT_BASH_TOOL


def _image_data_url(path) -> str:
    """Encode an image as a data URL for the OpenAI-compatible API."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def client():
    if not GROK_CONFIG["api_key"]:
        raise RuntimeError("no Azure credential for Grok: set AZURE_GROK_API_KEY in .env")

    if not GROK_CONFIG["endpoint"]:
        raise RuntimeError("no Azure endpoint for Grok: set AZURE_GROK_ENDPOINT in .env")
    
    return OpenAI(base_url=GROK_CONFIG["endpoint"],
                  api_key=GROK_CONFIG["api_key"],)


def call_grok(prompt: str, *, images=(), image_b64=(),
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
        model=GROK_CONFIG["deployment_name"],
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    return (text, resp.model_dump()) if return_raw else text


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
              max_tokens: int = DEFAULT_MAX_TOKENS,
              require_file: str | None = None,
              verbose: bool = True,):
    """Tool-using Grok agent loop on the Responses API.
    Returns (final_text, meta).
    """
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)

    if files:
        names = attach_files(cwd, files)

        if announce_files:
            prompt = prompt + attached_files_note(
                names, read_tool=SHARED_READ_TOOL, bash_tool=SHARED_BASH_TOOL,)

    cl = client()
    tools = _agent_tool_schemas_responses()

    pending = [{"role": "user",
                "content": _input_content(prompt, images, image_b64),}]

    prev_id = None
    transcript = Transcript()

    final_text = ""
    nudged = False
    turn = 0

    while turn < max_turns:
        turn += 1

        resp = cl.responses.create(
            model=GROK_CONFIG["deployment_name"],
            input=pending,
            tools=tools,
            max_output_tokens=max_tokens,
            previous_response_id=prev_id,
        )

        prev_id = resp.id

        calls = [item for item in (resp.output or [])
                 if getattr(item, "type", "") == "function_call"]

        text = resp.output_text or ""

        transcript.append(
            {
                "role": "assistant",
                "content": text,
                "function_calls": [
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    for call in calls
                ],
                "status": resp.status,
            }
        )

        if not calls:
            if (
                require_file
                and not (cwd / require_file).is_file()
                and not nudged
                and turn < max_turns
            ):
                nudged = True

                note = (
                    f"`{require_file}` does not exist in the working "
                    "directory yet -- your work is only graded from "
                    "that file. Please continue and save it."
                )

                pending = [{"role": "user",
                            "content": [{"type": "input_text", "text": note,}],}]

                transcript.append({"role": "user", "content": note,})

                if verbose:
                    print(f"    [grok turn {turn}] stopped without "
                          f"{require_file}; nudged once")
                continue

            final_text = text
            break

        pending = []

        for call in calls:
            try:
                args = json.loads(call.arguments or "{}")
                result = dispatch(cwd, call.name, args)

            except json.JSONDecodeError as exc:
                result = f"ERROR: tool arguments were not valid JSON: {exc}"

            result = clip(result)

            if verbose:
                preview = (call.arguments or "")[:110].replace("\n", " ")

                print(
                    f"    [grok turn {turn}] "
                    f"{call.name}({preview}) "
                    f"-> {len(result)} chars"
                )

            pending.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                }
            )

            transcript.append(
                {
                    "role": "tool",
                    "call_id": call.call_id,
                    "name": call.name,
                    "content": result,
                }
            )

    else:
        if verbose:
            print(f"    [grok] turn budget exhausted ({max_turns})")

    return final_text, {
        "model": GROK_CONFIG["deployment_name"],
        "turns": turn,
        "transcript": transcript,
        "nudged": nudged,
    }



# ---------------------------------------------------------------------------
# Kimi Code agent (2026-09-25) -- what try_model runs for Grok. Codex
# was tried first and fails: Foundry's Grok errors on every Codex tool call. See
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
    """Grok on Kimi Code. Returns (final_text, meta), like the other routes."""
    import base64 as _b64
    model_id = GROK_CONFIG["deployment_name"]
    tmp = Path(tempfile.mkdtemp(prefix="grok_img_"))
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
            prompt, cwd=cwd, model_id=model_id, base_url=GROK_CONFIG["endpoint"],
            api_key=GROK_CONFIG["api_key"], max_turns=max_turns, files=files,
            inline_files=inline, announce_files=announce_files,
            require_file=require_file, label="grok", verbose=verbose,
            progress=progress,)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def label() -> str:
    return (f"Grok (Kimi Code, Azure Foundry, "
            f"deployment={GROK_CONFIG['deployment_name']})")


def solve(user_text, image_files=(), pdf_pages=(),
          pdf_files=(), code_files=(), pointcloud_files=(),):
    """Agent solver, mirroring call_gpt.solve()."""

    workdir = tempfile.mkdtemp(prefix="grok_agent_")

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
    print(call_grok("Reply with exactly: OK"))