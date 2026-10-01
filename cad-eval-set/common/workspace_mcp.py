"""The agent's sandboxed shell, served as an MCP tool to agent CLIs.

Codex and Gemini CLI run as separate processes and bring their own shell,
which runs on the HOST -- where solution/, the release copy of every task
and git history are all readable. try_model's whole defence is that the
agent's shell runs somewhere else (a container, or a restricted Windows
account), and it installs that by patching `agent_workspace.TOOL_IMPL`.
So each harness has its own shell switched off and is given this tool
instead: `bash`, answered in THIS process through `dispatch`, which honours
the patch.

It speaks MCP's streamable-HTTP transport in its simplest form -- one JSON
reply per POST, no SSE stream -- over 127.0.0.1 on a random port, behind a
random path. Hand-rolled rather than built on the `mcp` package, whose 2.x
server API already broke one SDK in this repo; four JSON-RPC methods do
not justify that exposure.

    with WorkspaceMCP(cwd) as mcp:
        ...  # hand mcp.url to the CLI; mcp.calls records every call
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from common.agent_workspace import (TOOL_SPECS, _inside, _outside_msg, clip,
                                    dispatch, image_placeholder)

#: Image viewer for harnesses whose own viewer cannot be confined (Kimi Code's
#: ReadMediaFile reads any path). Served here instead, inside the workspace.
VIEW_IMAGE_SPEC = {
    "name": "view_image",
    "description": ("Look at an image file in the working directory (PNG, JPG, "
                    "GIF, WEBP): the picture itself is returned to you. Path "
                    "relative to the working directory. Large images are "
                    "downscaled to at most 2000 px on their long side."),
    "inputSchema": {"type": "object", "properties": {
        "path": {"type": "string", "description": "relative image path"}},
        "required": ["path"]},
}
_IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".gif": "image/gif", ".webp": "image/webp"}
VIEW_IMAGE_MAX_PX = 2000


def _view_image(cwd, path):
    """(MCP content blocks, log text) for one image inside cwd."""
    import base64
    import io
    p = _inside(cwd, path or "")
    if p is None:
        msg = _outside_msg(path)
        return [{"type": "text", "text": msg}], msg
    mime = _IMAGE_MIME.get(p.suffix.lower())
    if not p.is_file() or mime is None:
        msg = (f"ERROR: {path} is not an image file in the working directory"
               " (png, jpg, gif, webp)")
        return [{"type": "text", "text": msg}], msg
    data = p.read_bytes()
    note = ""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        w, h = im.size
        if max(w, h) > VIEW_IMAGE_MAX_PX:
            im.thumbnail((VIEW_IMAGE_MAX_PX, VIEW_IMAGE_MAX_PX))
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "PNG")
            data, mime = buf.getvalue(), "image/png"
            note = f" (downscaled from {w}x{h} to {im.size[0]}x{im.size[1]})"
        else:
            note = f" ({w}x{h})"
    except Exception:
        pass
    blocks = [{"type": "image", "data": base64.b64encode(data).decode("ascii"),
               "mimeType": mime},
              {"type": "text", "text": f"{path}{note}"}]
    return blocks, image_placeholder(path, f"{note.strip()} ~{len(data) // 1024} KB")

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "workspace"


def _spec(name):
    if name == "view_image":
        return VIEW_IMAGE_SPEC
    spec = next(s for s in TOOL_SPECS if s["name"] == name)
    return {"name": name, "description": spec["description"],
            "inputSchema": spec["parameters"]}


class WorkspaceMCP:
    """A one-tool MCP server for the length of a `with` block."""

    def __init__(self, cwd, tools=("bash",)):
        """tools: which of the shared tools to serve. bash alone for harnesses
        that keep their own workspace-confined file tools (Claude Code, Codex,
        Gemini CLI); all four for one whose file tools cannot be confined
        (Kimi Code), so every file access goes through agent_workspace."""
        self.cwd = Path(cwd)
        self.tools = tuple(tools)
        self.token = secrets.token_urlsafe(16)
        #: every call, in order: {"name", "input", "output", "seconds"}
        self.calls: list[dict] = []
        self._server = None
        self._thread = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/mcp/{self.token}"

    def _handle(self, req: dict):
        method, rid = req.get("method"), req.get("id")
        if rid is None:                     # a notification: nothing to say
            return None
        if method == "initialize":
            asked = (req.get("params") or {}).get("protocolVersion")
            result = {"protocolVersion": asked or PROTOCOL_VERSION,
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": SERVER_NAME, "version": "1.0"}}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [_spec(t) for t in self.tools]}
        elif method == "tools/call":
            params = req.get("params") or {}
            name, args = params.get("name"), params.get("arguments") or {}
            if name not in self.tools:
                return {"jsonrpc": "2.0", "id": rid, "error": {
                    "code": -32602, "message": f"unknown tool {name!r}"}}
            t0 = time.time()
            if name == "view_image":
                blocks, out = _view_image(self.cwd, args.get("path"))
            else:
                out = clip(dispatch(self.cwd, name, dict(args)))
                blocks = [{"type": "text", "text": out}]
            self.calls.append({"name": name, "input": args, "output": out,
                               "seconds": round(time.time() - t0, 1)})
            result = {"content": blocks, "isError": False}
        else:
            return {"jsonrpc": "2.0", "id": rid, "error": {
                "code": -32601, "message": f"method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def __enter__(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):      # keep the agent's console clean
                pass

            def _path_ok(self):
                if self.path.split("?")[0] != f"/mcp/{outer.token}":
                    self.send_error(404)
                    return False
                return True

            def do_POST(self):
                if not self._path_ok():
                    return
                body = self.rfile.read(int(self.headers.get("Content-Length")
                                           or 0))
                try:
                    msg = json.loads(body or b"null")
                except ValueError:
                    self.send_error(400)
                    return
                batch = isinstance(msg, list)
                replies = [r for r in (outer._handle(m) for m in
                                       (msg if batch else [msg])) if r]
                if not replies:
                    self.send_response(202)
                    self.end_headers()
                    return
                data = json.dumps(replies if batch else replies[0]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Mcp-Session-Id", outer.token)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):               # no server-initiated stream
                if self._path_ok():
                    self.send_response(405)
                    self.send_header("Allow", "POST")
                    self.end_headers()

            def do_DELETE(self):
                if self._path_ok():
                    self.send_response(200)
                    self.end_headers()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        return False
