"""Self-hosted endpoint (Kaggle/Ollama): LiteLLM settings and the token proxy."""

import http.server
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import litellm
import pytest

from talentmatcher import config, llm
from talentmatcher.ratelimit import RateLimiter


def test_api_base_and_key_are_passed_to_litellm(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_BASE", "https://example.trycloudflare.com/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "secret")
    monkeypatch.setattr(llm, "rate_limiter", RateLimiter(0))
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "hi"}}])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    assert llm.chat([{"role": "user", "content": "x"}], temperature=0) == "hi"
    assert seen["api_base"] == "https://example.trycloudflare.com/v1" and seen["api_key"] == "secret"


def test_connection_error_mentions_self_hosted_server(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_BASE", "https://gone.trycloudflare.com/v1")
    monkeypatch.setattr(llm, "rate_limiter", RateLimiter(0))

    def down(**kwargs):
        raise litellm.APIConnectionError("refused", llm_provider="openai", model="m")

    monkeypatch.setattr(litellm, "completion", down)
    with pytest.raises(llm.LLMError) as info:
        llm.chat([{"role": "user", "content": "x"}], temperature=0)
    assert info.value.retryable and "Kaggle" in str(info.value)


# ---------- kaggle/auth_proxy.py ----------

class Upstream(http.server.BaseHTTPRequestHandler):
    received_auth = []

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        Upstream.received_auth.append(self.headers.get("Authorization"))
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _serve(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def proxy_url(monkeypatch):
    monkeypatch.setenv("PROXY_TOKEN", "t0ken")
    path = Path(__file__).resolve().parent.parent / "kaggle" / "auth_proxy.py"
    spec = importlib.util.spec_from_file_location("auth_proxy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    upstream = _serve(Upstream)
    module.UPSTREAM = f"http://127.0.0.1:{upstream.server_port}"
    module.Handler.log_message = lambda *a: None
    proxy = _serve(module.Handler)
    yield f"http://127.0.0.1:{proxy.server_port}"
    proxy.shutdown()
    upstream.shutdown()


def _post(url, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url + "/v1/chat/completions", data=b"{}", headers=headers)
    return urllib.request.urlopen(req, timeout=5)


def test_proxy_forwards_with_valid_token_without_leaking_it(proxy_url):
    Upstream.received_auth.clear()
    assert json.load(_post(proxy_url, "t0ken")) == {"ok": True}
    assert Upstream.received_auth == [None]


@pytest.mark.parametrize("token", [None, "wrong", "t0ken-extra"])
def test_proxy_rejects_bad_tokens(proxy_url, token):
    with pytest.raises(urllib.error.HTTPError) as info:
        _post(proxy_url, token)
    assert info.value.code == 401
