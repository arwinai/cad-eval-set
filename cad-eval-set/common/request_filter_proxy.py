"""A local pass-through to an OpenAI-compatible endpoint that drops fields it rejects.

Kimi Code sends `prompt_cache_key` on every chat-completions request, and
Azure Foundry answers `400 Unrecognized request argument supplied:
prompt_cache_key` -- with no setting in Kimi Code to stop sending it. So
the harness is pointed at this proxy instead: each JSON request body has
the named top-level keys removed and is forwarded to the real endpoint
unchanged otherwise, headers (the API key) included; the response streams
back chunk by chunk, so SSE thinking and tool calls still arrive live.

    with FilterProxy("https://x.services.ai.azure.com/openai/v1",
                     drop={"prompt_cache_key"}) as proxy:
        ...  # give proxy.url to the harness as its base_url
"""
from __future__ import annotations

import json
import queue
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import requests

#: seconds of upstream silence before an SSE keep-alive comment is sent
KEEPALIVE_S = 20
#: Longest the upstream may stay silent before its request is given up (the
#: client then retries it). The keep-alives above would otherwise hide a dead
#: connection for as long as this: a Grok sling-lift request sat open for
#: 40 min with nothing coming. Azure drops a request silent for ~10 min, so a
#: live think never outlasts this.
UPSTREAM_SILENCE_S = 12 * 60

#: request headers that must not be copied verbatim to the upstream
_HOP = {"host", "content-length", "connection", "accept-encoding",
        "transfer-encoding", "keep-alive"}


class FilterProxy:
    def __init__(self, upstream: str, drop=("prompt_cache_key",),
                 timeout_s: int = UPSTREAM_SILENCE_S, capture: str | None = None,
                 strip_message_keys=(), plain_tool_call_ids: bool = False):
        self.upstream = upstream.rstrip("/")
        self.drop = set(drop)
        self.timeout_s = timeout_s
        self.token = secrets.token_urlsafe(12)
        #: keys removed from every entry of `messages` -- see Kimi below
        self.strip_message_keys = set(strip_message_keys)
        #: Rename tool-call ids to call_0, call_1, ... within each request.
        #: Kimi Code echoes ids shaped like `functions_mcp__workspace__bash_0`,
        #: and Foundry's Kimi K2.7 then answers every later turn with an empty
        #: stop -- replaying one captured request with ONLY the id changed to
        #: `call_0` brought the tool calls back. Ids need only agree within a
        #: request, so the assistant's calls and the tool results that answer
        #: them are mapped together.
        self.plain_tool_call_ids = plain_tool_call_ids
        self.dropped = 0
        #: debug only: append each request body and raw response here
        self.capture = capture
        self._server = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/{self.token}"

    def __enter__(self):
        outer = self
        session = requests.Session()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _forward(self, method):
                prefix = f"/{outer.token}"
                if not self.path.startswith(prefix):
                    self.send_error(404)
                    return
                target = outer.upstream + self.path[len(prefix):]
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if body and "json" in (self.headers.get("Content-Type") or ""):
                    try:
                        data = json.loads(body)
                        changed = False
                        if isinstance(data, dict) and outer.drop & data.keys():
                            for k in outer.drop:
                                data.pop(k, None)
                            changed = True
                        if isinstance(data, dict) and outer.strip_message_keys:
                            for msg in data.get("messages") or []:
                                if isinstance(msg, dict) and                                         outer.strip_message_keys & msg.keys():
                                    for k in outer.strip_message_keys:
                                        msg.pop(k, None)
                                    changed = True
                        if isinstance(data, dict) and outer.plain_tool_call_ids:
                            ids = {}
                            for msg in data.get("messages") or []:
                                if not isinstance(msg, dict):
                                    continue
                                for c in msg.get("tool_calls") or []:
                                    if isinstance(c, dict) and c.get("id"):
                                        c["id"] = ids.setdefault(c["id"], "call_%d" % len(ids))
                                        changed = True
                                if msg.get("tool_call_id"):
                                    msg["tool_call_id"] = ids.setdefault(
                                        msg["tool_call_id"], "call_%d" % len(ids))
                                    changed = True
                        if changed:
                            outer.dropped += 1
                            body = json.dumps(data).encode()
                    except ValueError:
                        pass
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in _HOP}
                headers["Host"] = urlsplit(outer.upstream).netloc
                streaming = b'"stream": true' in body or b'"stream":true' in body

                # The upstream is read on a thread into a queue, so this
                # handler can keep the client's connection alive while the
                # model is silent. Grok on chat completions streams nothing
                # while it reasons, and Node (Kimi Code) drops a response
                # body that is silent for ~5 min -- "terminated", then a
                # retry from scratch. An SSE comment line every few seconds
                # is ignored by every stream parser and keeps it open.
                q: "queue.Queue" = queue.Queue()

                def pump():
                    try:
                        r = session.request(method, target, data=body or None,
                                            headers=headers, stream=True,
                                            timeout=(30, outer.timeout_s))
                        q.put(("head", r))
                        for chunk in r.iter_content(chunk_size=None):
                            if chunk:
                                q.put(("data", chunk))
                        r.close()
                        q.put(("end", None))
                    except requests.RequestException as exc:
                        q.put(("fail", exc))

                threading.Thread(target=pump, daemon=True).start()
                cap = open(outer.capture, "ab") if outer.capture else None
                if cap:
                    cap.write(b"\n=== REQUEST " + target.encode() + b"\n"
                              + (body or b"") + b"\n=== RESPONSE\n")

                def chunk_out(b):
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
                    self.wfile.flush()

                headers_sent = False
                try:
                    while True:
                        try:
                            kind, item = q.get(timeout=KEEPALIVE_S)
                        except queue.Empty:
                            if not streaming:
                                continue
                            if not headers_sent:
                                # no status yet: commit to an SSE 200 now
                                self.send_response(200)
                                self.send_header("Content-Type", "text/event-stream")
                                self.send_header("Transfer-Encoding", "chunked")
                                self.end_headers()
                                headers_sent = True
                            chunk_out(b": keep-alive\n\n")
                            continue
                        if kind == "head":
                            if headers_sent:
                                if item.status_code != 200:
                                    err = item.text[:2000].replace("\n", " ")
                                    chunk_out(b"data: " + json.dumps({"error": {
                                        "message": f"upstream {item.status_code}: {err}",
                                        "code": item.status_code}}).encode() + b"\n\n")
                                    break
                                continue
                            self.send_response(item.status_code)
                            for k, v in item.headers.items():
                                if k.lower() not in _HOP | {"content-encoding"}:
                                    self.send_header(k, v)
                            self.send_header("Transfer-Encoding", "chunked")
                            self.end_headers()
                            headers_sent = True
                        elif kind == "data":
                            if cap:
                                cap.write(item)
                            chunk_out(item)
                        elif kind == "fail":
                            if not headers_sent:
                                self.send_error(502, str(item)[:200])
                                return
                            chunk_out(b"data: " + json.dumps({"error": {
                                "message": f"upstream failed: {str(item)[:300]}"}}).encode()
                                + b"\n\n")
                            break
                        else:                                   # end
                            break
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    if cap:
                        cap.close()

            def do_POST(self):
                self._forward("POST")

            def do_GET(self):
                self._forward("GET")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        #: a client closing its keep-alive socket is normal, not an error
        self._server.handle_error = lambda *a: None
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        return False
