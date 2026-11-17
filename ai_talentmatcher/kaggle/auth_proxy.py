"""Tiny bearer-token proxy in front of Ollama (standard library only).

Ollama has no authentication. The Cloudflare tunnel makes it reachable from the internet,
so this proxy rejects every request that doesn't carry `Authorization: Bearer <PROXY_TOKEN>`.
That's the same header LiteLLM sends as `api_key` for OpenAI-compatible endpoints.

    PROXY_TOKEN=... python auth_proxy.py      # listens on 127.0.0.1:8080 -> Ollama 127.0.0.1:11434
"""

import hmac
import http.server
import json
import os
import urllib.error
import urllib.request

TOKEN = os.environ["PROXY_TOKEN"]
UPSTREAM = os.getenv("PROXY_UPSTREAM", "http://127.0.0.1:11434")
HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PORT = int(os.getenv("PROXY_PORT", "8080"))


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type or "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str) -> None:
        self._send(code, json.dumps({"error": {"message": message}}).encode())

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None  # always drain the request body

        supplied = self.headers.get("Authorization", "").encode()
        if not hmac.compare_digest(supplied, f"Bearer {TOKEN}".encode()):  # constant-time compare
            return self._error(401, "Invalid or missing API token.")

        request = urllib.request.Request(
            UPSTREAM + self.path,
            data=body,
            method=method,
            headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as resp:
                self._send(resp.status, resp.read(), resp.headers.get("Content-Type"))
        except urllib.error.HTTPError as exc:  # Ollama answered with an error: pass it through
            self._send(exc.code, exc.read(), exc.headers.get("Content-Type"))
        except Exception as exc:  # Ollama not running, crashed, ...
            self._error(502, f"Upstream model server unavailable: {exc}")

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def log_message(self, fmt, *args):  # one short line per request, never the headers
        print(f"[proxy] {self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    print(f"[proxy] listening on {HOST}:{PORT} -> {UPSTREAM}", flush=True)
    http.server.ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
