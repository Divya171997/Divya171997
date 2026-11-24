"""Offline tests for agentkit. No API keys or network needed.

* OpenAI and Ollama adapters are exercised end to end (real LangChain + SDK code) against local fake HTTP servers
  that speak each provider's wire protocol, including native tool calls.
* The Gemini adapter is tested with an injected chat model (its transport goes to Google's servers).
* Facade logic (selection, cache labelling, retries, fallback, structured-output repair, tool loop) uses a scripted provider.
Run:  pytest -q agentkit/tests          (set AGENTKIT_LIVE=1 to also run one live local-model test)
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agentkit import LLM, Settings, resolve, CacheMiss, StructuredOutputError                  # noqa: E402
from agentkit.config import choose_ollama_model, pick_hf_model                                  # noqa: E402
from agentkit.providers import Completion, ProviderError, OpenAIProvider, OllamaProvider, GeminiProvider  # noqa: E402


# ------------------------------------------------------------------ helpers
def get_price(ticker: str, date: str) -> dict:
    """Closing price of a stock on a date.

    Args:
        ticker: stock ticker, e.g. AAPL
        date: YYYY-MM-DD
    """
    return {"ticker": ticker, "date": date, "close": 139.79}


class Pick(BaseModel):
    tickers: list
    year: int


class Scripted:
    """A provider that replays a fixed script of Completions (or raises scripted errors)."""
    def __init__(self, script, name="scripted", model="scripted-1"):
        self.script, self.name, self.model, self.seen = list(script), name, model, []

    def complete(self, messages, max_tokens=256, tools=None, json_mode=False):
        self.seen.append({"messages": messages, "tools": tools, "json_mode": json_mode})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def env_clear(monkeypatch):
    for k in ["LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OLLAMA_HOST",
              "OLLAMA_MODEL", "HF_MODEL", "LLM_CACHE", "LLM_FALLBACK"]:
        monkeypatch.delenv(k, raising=False)


class FakeServer:
    def __init__(self, handler_fn):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(*handler_fn("GET", self.path, None, self.headers))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                outer.requests.append(body)
                self._send(*handler_fn("POST", self.path, body, self.headers))

        self.requests = []
        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


# ------------------------------------------------------------------ selection
def test_auto_selection_order(monkeypatch):
    env_clear(monkeypatch)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")           # nothing listens here
    s = Settings.from_env()
    p, m, why = resolve(s)
    assert p == "huggingface" and m.startswith("Qwen/Qwen3-") and "offline" in why
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert resolve(Settings.from_env())[:2] == ("gemini", "gemini-3.1-flash-lite")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert resolve(Settings.from_env())[:2] == ("openai", "gpt-4.1-mini")      # OpenAI first when both are set
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert resolve(Settings.from_env())[0] == "gemini"                         # explicit choice wins
    monkeypatch.setenv("LLM_MODEL", "gemini-3.6-flash")
    assert resolve(Settings.from_env())[1] == "gemini-3.6-flash"


def test_secrets_never_exposed(monkeypatch):
    env_clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-123")
    assert "sk-secret-123" not in json.dumps(Settings.from_env().public())


def test_local_model_tiers_and_ollama_preference():
    assert pick_hf_model(40) == "Qwen/Qwen3-8B"
    assert pick_hf_model(12) == "Qwen/Qwen3-4B"
    assert pick_hf_model(7.7) == "Qwen/Qwen3-1.7B"
    assert pick_hf_model(1.0) == "Qwen/Qwen3-0.6B"
    assert choose_ollama_model(["llama3.2:3b", "qwen3:8b"]) == "qwen3:8b"
    assert choose_ollama_model(["phi4:latest"]) == "phi4:latest"


def test_invalid_settings_rejected(monkeypatch):
    env_clear(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "claude-local")
    with pytest.raises(ValueError):
        Settings.from_env()


# ------------------------------------------------------------------ OpenAI adapter (fake server)
def openai_handler(method, path, body, headers):
    if headers.get("Authorization", "").endswith("bad-key"):
        return 401, {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error", "code": "invalid_api_key"}}
    msgs = body["messages"]
    usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    if body.get("tools") and not any(m["role"] == "tool" for m in msgs):
        msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_a", "type": "function",
               "function": {"name": "get_price", "arguments": json.dumps({"ticker": "AAPL", "date": "2017-03-01"})}}]}
    elif body.get("response_format", {}).get("type") == "json_object":
        msg = {"role": "assistant", "content": '{"tickers": ["MSFT", "AAPL"], "year": 2017}'}
    elif any(m["role"] == "tool" for m in msgs):
        msg = {"role": "assistant", "content": "Apple closed at 139.79."}
    else:
        msg = {"role": "assistant", "content": "hello"}
    return 200, {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                 "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage}


@pytest.fixture
def openai_server():
    srv = FakeServer(openai_handler)
    yield srv
    srv.close()


def test_openai_adapter_chat_json_and_tools(openai_server, monkeypatch, tmp_path):
    env_clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "good-key")
    monkeypatch.setenv("OPENAI_BASE_URL", openai_server.url + "/v1")
    llm = LLM(out_dir=tmp_path)
    assert (llm.provider, llm.model_name) == ("openai", "gpt-4.1-mini")
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert openai_server.requests[-1]["temperature"] == 0 and openai_server.requests[-1]["seed"] == 0
    p = llm.structured([{"role": "user", "content": "Compare Microsoft and Apple in 2017. Reply in JSON."}], Pick)
    assert p.tickers == ["MSFT", "AAPL"] and openai_server.requests[-1]["response_format"] == {"type": "json_object"}
    run = llm.run_agent([{"role": "user", "content": "Apple close on 2017-03-01?"}], [get_price])
    assert run.tools_used == ["get_price"] and "139.79" in run.final_text
    assert openai_server.requests[-1]["messages"][-1]["role"] == "tool"          # tool result was sent back
    rep = llm.report(show=False)
    assert rep["stats"]["live_calls"] == 4 and rep["stats"]["cache_hits"] == 0
    assert all(r["est_cost_usd"] is not None for r in rep["by_source"])          # priced model


def test_openai_bad_key_falls_back_to_local(openai_server, monkeypatch, tmp_path):
    env_clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "bad-key")
    monkeypatch.setenv("OPENAI_BASE_URL", openai_server.url + "/v1")
    local = Scripted([Completion("local answer")], name="huggingface", model="Qwen/Qwen3-1.7B")
    import agentkit.llm as L
    monkeypatch.setattr(L, "build_provider", lambda n, m, s: local if n == "huggingface" else OpenAIProvider(m, s))
    llm = LLM(out_dir=tmp_path)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "local answer"
    assert llm.provider == "huggingface" and llm.stats["fallbacks"] == 1 and "FELL BACK" in llm.selection_reason
    recs = [json.loads(l) for l in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    assert recs[-1]["provider"] == "huggingface"                                 # the answer is attributed to who gave it


def test_reasoning_models_omit_temperature(openai_server, monkeypatch):
    env_clear(monkeypatch)
    s = Settings(openai_api_key="good-key", openai_base_url=openai_server.url + "/v1")
    OpenAIProvider("gpt-5-mini", s).complete([{"role": "user", "content": "hi"}])
    assert "temperature" not in openai_server.requests[-1]


# ------------------------------------------------------------------ Ollama adapter (fake server)
def ollama_handler(method, path, body, headers):
    if path == "/api/tags":
        return 200, {"models": [{"name": "llama3.2:3b"}, {"name": "qwen3:8b"}]}
    tools, msgs = body.get("tools"), body["messages"]
    if tools and not any(m["role"] == "tool" for m in msgs):
        msg = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_price", "arguments": {"ticker": "AAPL", "date": "2017-03-01"}}}]}
    elif body.get("format") == "json":
        msg = {"role": "assistant", "content": '{"tickers": ["MSFT"], "year": 2016}'}
    else:
        msg = {"role": "assistant", "content": "Apple closed at 139.79." if tools else "hi from ollama"}
    chunk = {"model": body["model"], "created_at": "2026-01-01T00:00:00Z", "message": msg, "done": True,
             "done_reason": "stop", "prompt_eval_count": 9, "eval_count": 4}
    return 200, json.dumps(chunk) + "\n"


def test_ollama_discovery_and_adapter(monkeypatch, tmp_path):
    srv = FakeServer(ollama_handler)
    try:
        env_clear(monkeypatch)
        monkeypatch.setenv("OLLAMA_HOST", srv.url)
        llm = LLM(out_dir=tmp_path)
        assert (llm.provider, llm.model_name) == ("ollama", "qwen3:8b")          # best installed model chosen
        assert llm.chat([{"role": "user", "content": "hi"}]) == "hi from ollama"
        assert llm.structured([{"role": "user", "content": "json please"}], Pick).year == 2016
        run = llm.run_agent([{"role": "user", "content": "Apple close?"}], [get_price])
        assert run.tools_used == ["get_price"] and run.steps[0].ok and "139.79" in run.final_text
        req = srv.requests[-1]
        assert req["options"]["temperature"] == 0 and req["options"]["seed"] == 0
    finally:
        srv.close()


# ------------------------------------------------------------------ Gemini adapter (injected chat model)
def test_gemini_adapter(monkeypatch, tmp_path):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class FakeGemini(BaseChatModel):
        bound_tools: list = []

        @property
        def _llm_type(self):
            return "fake-gemini"

        def bind_tools(self, tools, **kw):
            return self.model_copy(update={"bound_tools": tools})

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            if self.bound_tools and messages[-1].type != "tool":
                m = AIMessage(content="", tool_calls=[{"name": "get_price", "args": {"ticker": "AAPL", "date": "2017-03-01"}, "id": "g1"}])
            elif self.bound_tools:
                m = AIMessage(content="139.79")
            else:
                m = AIMessage(content=[{"type": "text", "text": '{"tickers": ["GOOGL"], "year": 2015}'}],
                              usage_metadata={"input_tokens": 5, "output_tokens": 6, "total_tokens": 11})
            return ChatResult(generations=[ChatGeneration(message=m)])

    s = Settings(gemini_api_key="g")
    real = GeminiProvider("gemini-3.1-flash-lite", s)._build(128, True)          # constructs the real client offline
    assert real.model.endswith("gemini-3.1-flash-lite") and real.response_mime_type == "application/json"
    prov = GeminiProvider("gemini-3.1-flash-lite", s, chat_factory=lambda mt, js: FakeGemini())
    llm = LLM(_provider_override=prov, out_dir=tmp_path)
    assert llm.structured([{"role": "user", "content": "x"}], Pick).tickers == ["GOOGL"]
    run = llm.run_agent([{"role": "user", "content": "price?"}], [get_price])
    assert run.tools_used == ["get_price"] and run.final_text == "139.79"


# ------------------------------------------------------------------ facade behaviour
def test_cache_off_by_default_and_replays_are_labelled(tmp_path, monkeypatch, caplog):
    env_clear(monkeypatch)
    prov = Scripted([Completion("first"), Completion("second")])
    llm = LLM(_provider_override=prov, out_dir=tmp_path)
    m = [{"role": "user", "content": "same question"}]
    assert llm.settings.cache_mode == "off"
    assert [llm.chat(m), llm.chat(m)] == ["first", "second"]                       # both live
    llm2 = LLM(_provider_override=Scripted([Completion("live once")]), out_dir=tmp_path, cache="readwrite")
    assert llm2.chat(m) == "live once"
    llm3 = LLM(_provider_override=Scripted([]), out_dir=tmp_path, cache="readwrite")
    with caplog.at_level("WARNING"):
        assert llm3.chat(m) == "live once"
    assert "[CACHE REPLAY]" in caplog.text and llm3.stats["cache_hits"] == 1 and llm3.stats["live_calls"] == 0
    rec = json.loads((tmp_path / "llm_calls.jsonl").read_text().splitlines()[-1])
    assert rec["source"] == "cache"
    llm4 = LLM(_provider_override=Scripted([]), out_dir=tmp_path, cache="replay")
    with pytest.raises(CacheMiss):
        llm4.chat([{"role": "user", "content": "never asked before"}])


def test_transient_errors_retry_then_succeed(tmp_path, monkeypatch):
    env_clear(monkeypatch)
    monkeypatch.setattr("agentkit.llm.time.sleep", lambda s: None)
    prov = Scripted([ProviderError("429 rate limit", transient=True), Completion("ok")])
    llm = LLM(_provider_override=prov, out_dir=tmp_path)
    assert llm.chat([{"role": "user", "content": "x"}]) == "ok" and llm.stats["retries"] == 1


def test_structured_output_repair_and_failure(tmp_path, monkeypatch):
    env_clear(monkeypatch)
    prov = Scripted([Completion("Sure! {'tickers': ['KO'], 'year': None}"), Completion('{"tickers": ["KO"], "year": 2016}')])
    llm = LLM(_provider_override=prov, out_dir=tmp_path)
    assert llm.structured([{"role": "user", "content": "x"}], Pick).year == 2016
    assert "not valid" in prov.seen[1]["messages"][-1]["content"]                 # the model saw the validation error
    bad = LLM(_provider_override=Scripted([Completion("nope")] * 3), out_dir=tmp_path)
    with pytest.raises(StructuredOutputError):
        bad.structured([{"role": "user", "content": "x"}], Pick, retries=2)
    assert issubclass(StructuredOutputError, RuntimeError)


def test_agent_loop_handles_bad_tool_calls(tmp_path, monkeypatch):
    env_clear(monkeypatch)
    prov = Scripted([Completion("", tool_calls=[{"id": "1", "name": "no_such_tool", "arguments": {}},
                                                {"id": "2", "name": "get_price", "arguments": {"ticker": "AAPL"}}]),
                     Completion("", tool_calls=[{"id": "3", "name": "get_price", "arguments": {"ticker": "AAPL", "date": "2017-03-01"}}]),
                     Completion("done")])
    run = LLM(_provider_override=prov, out_dir=tmp_path).run_agent([{"role": "user", "content": "x"}], [get_price])
    assert [s.ok for s in run.steps] == [False, False, True] and run.final_text == "done"
    assert "unknown tool" in run.steps[0].result and "invalid arguments" in run.steps[1].result


def test_agent_repeat_calls_are_not_rerun(tmp_path, monkeypatch):
    env_clear(monkeypatch)
    calls = []
    def counter(x: int) -> int:
        """Count.

        Args:
            x: a number
        """
        calls.append(x)
        return x
    rep = Completion("", tool_calls=[{"id": "1", "name": "counter", "arguments": {"x": 1}}])
    prov = Scripted([rep, rep, Completion("final")])
    run = LLM(_provider_override=prov, out_dir=tmp_path).run_agent([{"role": "user", "content": "x"}], [counter])
    assert calls == [1] and "Repeat call" in run.steps[1].result and run.final_text == "final"


def test_agent_stops_at_max_steps(tmp_path, monkeypatch):
    env_clear(monkeypatch)
    loop = Completion("", tool_calls=[{"id": "1", "name": "get_price", "arguments": {"ticker": "A", "date": "d"}}])
    run = LLM(_provider_override=Scripted([loop] * 3), out_dir=tmp_path).run_agent([{"role": "user", "content": "x"}], [get_price], max_steps=3)
    assert run.stopped == "max_steps" and len(run.steps) == 3


# ------------------------------------------------------------------ optional live local-model test
@pytest.mark.skipif(os.getenv("AGENTKIT_LIVE") != "1", reason="set AGENTKIT_LIVE=1 to load the local model")
def test_live_local_model(monkeypatch, tmp_path):
    env_clear(monkeypatch)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    llm = LLM(out_dir=tmp_path)
    assert llm.provider == "huggingface"
    assert llm.structured([{"role": "system", "content": 'Reply with JSON: {"tickers": [str], "year": int}'},
                           {"role": "user", "content": "Compare Microsoft and Apple in 2017"}], Pick).year == 2017
    run = llm.run_agent([{"role": "user", "content": "What was Apple's (AAPL) closing price on 2017-03-01?"}], [get_price])
    assert "get_price" in run.tools_used
