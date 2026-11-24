"""Provider-independent LLM facade used by every project.

    llm = LLM()                                   # automatic provider/model selection from the environment
    text = llm.chat(messages)                     # plain completion
    obj  = llm.structured(messages, MySchema)     # validated Pydantic object (native JSON mode where available)
    run  = llm.run_agent(messages, [tool_fn, ...])  # genuine tool-calling loop
    llm.report()                                  # provenance: provider, model, live vs cached calls, tokens, cost

Transparency rules:
  * Caching is OFF by default; every call is a live model call.
  * With LLM_CACHE=readwrite or replay, every response served from the cache is logged at WARNING level
    ("[CACHE REPLAY]"), tagged source="cache" in outputs/llm_calls.jsonl, and counted separately in the report.
  * A fallback from an API provider to the local model is logged at WARNING level ("[FALLBACK]") and every
    later call records the provider that actually answered.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import platform
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Type

from pydantic import BaseModel, ValidationError

from .config import DEFAULT_FALLBACK_MODELS, KNOWN_PRICES, Settings, load_env, pick_hf_model, free_memory_gb, resolve
from .providers import Completion, ProviderError, build_provider
from .tools import execute_tool, tool_schema

log = logging.getLogger("agentkit")
_NOISY = ("httpx", "httpcore", "urllib3", "filelock", "chromadb", "sentence_transformers", "transformers",
          "huggingface_hub", "openai", "google_genai", "google", "langchain", "ollama")


def configure_logging(out_dir="outputs", logfile="run.log") -> logging.Logger:
    """Full detail to outputs/run.log; warnings (cache replays, fallbacks, failures) also to the notebook."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        handlers=[logging.FileHandler(out / logfile, mode="w"), console], force=True)
    for n in _NOISY:
        logging.getLogger(n).setLevel(logging.ERROR if n == "huggingface_hub" else logging.WARNING)
    return logging.getLogger("agents")


def retry_delay(message: str, attempt: int) -> float:
    """Seconds to wait before retrying: the provider's own hint if it gives one ("retry in 12s", "retryDelay": "12s"),
    otherwise exponential backoff 3, 6, 12, 24 ... capped at 60 s."""
    m = re.search(r"retry(?:ing)?\s+in\s+([\d.]+)\s*s|retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", message, re.I)
    if m:
        return min(120.0, float(m.group(1) or m.group(2)) + 1.0)
    return min(60.0, 3.0 * 2 ** attempt)


class StructuredOutputError(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    pass


@dataclass
class AgentStep:
    tool: str
    arguments: dict
    result: str
    ok: bool


@dataclass
class AgentRun:
    final_text: str
    steps: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    stopped: str = "answered"          # answered | max_steps

    @property
    def tools_used(self) -> list:
        return [s.tool for s in self.steps]


class LLM:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, cache: Optional[str] = None,
                 out_dir="outputs", settings: Optional[Settings] = None, _provider_override=None):
        self.env_files = load_env()
        self.settings = settings or Settings.from_env()
        if provider:
            self.settings.provider = provider
        if model:
            self.settings.model = model
        if cache:
            self.settings.cache_mode = cache
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.out / "llm_calls.jsonl"
        self.calls_path.write_text("")
        self.cache_path = self.out / "llm_cache.json"
        self.cache = {}
        if self.settings.cache_mode != "off" and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())
        self.stats = {"calls": 0, "live_calls": 0, "cache_hits": 0, "seconds": 0.0, "input_tokens": 0,
                      "output_tokens": 0, "retries": 0, "fallbacks": 0, "errors": 0}
        self.events = []
        self._last_call = 0.0
        self._model_queue = []
        if _provider_override is not None:                       # used by the offline test-suite
            self._p, self.selection_reason = _provider_override, "injected provider (tests)"
        else:
            name, mdl, self.selection_reason = resolve(self.settings)
            fb = self.settings.fallback_models if self.settings.fallback_models is not None else DEFAULT_FALLBACK_MODELS.get(name, [])
            self._model_queue = [m for m in fb if m != mdl]
            self._p = self._build(name, mdl)
        log.info("LLM ready: %s", self.describe().replace("\n", " | "))

    # ------------------------------------------------------------------ setup
    def _build(self, name: str, model: str):
        try:
            return build_provider(name, model, self.settings)
        except Exception as e:
            hint = (" (install it with: pip install -r agentkit/requirements-providers.txt)" if isinstance(e, ImportError) else "")
            if name != "huggingface" and self.settings.fallback_to_local:
                self._fallback(f"could not initialise {name}/{model}: {type(e).__name__}: {e}{hint}")
                return self._p
            raise

    def _fallback(self, why: str):
        free, device = free_memory_gb()
        model = self.settings.hf_model or pick_hf_model(free)
        log.warning("[FALLBACK] %s -> local huggingface/%s. Responses from now on come from the local model.", why, model)
        self.stats["fallbacks"] += 1
        self.events.append({"ts": time.time(), "event": "fallback", "reason": why[:300], "to": f"huggingface/{model}"})
        self._p = build_provider("huggingface", model, self.settings)
        self.selection_reason += f"; FELL BACK to local model ({why[:120]})"

    @property
    def provider(self) -> str:
        return self._p.name

    @property
    def model_name(self) -> str:
        return self._p.model

    def describe(self) -> str:
        s = self.settings
        rev = getattr(self._p, "revision", None)
        lines = [f"Provider: {self.provider}   Model: {self.model_name}" + (f" (revision {rev[:10]})" if rev else ""),
                 f"Why: {self.selection_reason}",
                 f"API keys: OpenAI {'set' if s.openai_api_key else 'not set'}, Gemini {'set' if s.gemini_api_key else 'not set'}"
                 f" | cache: {s.cache_mode}" + (" (responses replayed from cache are labelled)" if s.cache_mode != "off" else " (every call is live)")
                 + f" | fallback to local: {'on' if s.fallback_to_local else 'off'}",
                 f".env files loaded: {[Path(f).parent.name + '/.env' for f in self.env_files] or 'none'}"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ core call
    def _key(self, messages, max_tokens, tools, json_mode) -> str:
        blob = json.dumps({"p": self.provider, "m": self.model_name, "msgs": messages, "max": max_tokens,
                           "tools": tools, "json": json_mode}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _record(self, c: Completion, seconds: float, purpose: str, error: Optional[str] = None):
        price = (self.settings.price_input_per_m, self.settings.price_output_per_m)
        if price[0] is None:
            price = KNOWN_PRICES.get(self.model_name, (None, None))
        cost = None
        if c.source == "live" and price[0] is not None and c.input_tokens is not None:
            cost = (c.input_tokens * price[0] + (c.output_tokens or 0) * price[1]) / 1e6
        rec = {"ts": round(time.time(), 3), "purpose": purpose, "provider": c.provider or self.provider,
               "model": c.model or self.model_name, "source": c.source, "seconds": round(seconds, 3),
               "input_tokens": c.input_tokens, "output_tokens": c.output_tokens, "tool_calls": len(c.tool_calls),
               "est_cost_usd": cost, "error": error}
        with self.calls_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def complete(self, messages: list, max_tokens: int = 256, tools: Optional[list] = None, json_mode: bool = False,
                 purpose: str = "chat") -> Completion:
        key = self._key(messages, max_tokens, tools, json_mode)
        self.stats["calls"] += 1
        if self.settings.cache_mode in ("readwrite", "replay") and key in self.cache:
            c = Completion(**{**self.cache[key], "source": "cache"})
            self.stats["cache_hits"] += 1
            log.warning("[CACHE REPLAY] %s response from %s/%s was NOT generated in this run (LLM_CACHE=%s)",
                        purpose, c.provider, c.model, self.settings.cache_mode)
            self._record(c, 0.0, purpose)
            return c
        if self.settings.cache_mode == "replay":
            raise CacheMiss(f"LLM_CACHE=replay but no cached response exists for this {purpose} call")
        attempt, t0 = 0, time.time()
        while True:
            if self.provider != "huggingface" and self.settings.min_interval > 0:   # pace API calls (free-tier RPM limits)
                gap = time.time() - self._last_call
                if gap < self.settings.min_interval:
                    time.sleep(self.settings.min_interval - gap)
                self._last_call = time.time()
            try:
                c = self._p.complete(messages, max_tokens=max_tokens, tools=tools, json_mode=json_mode)
                break
            except ProviderError as e:
                self.stats["errors"] += 1
                remote = self.provider != "huggingface"
                if e.transient and not e.auth and attempt < self.settings.max_retries:
                    wait = retry_delay(str(e), attempt)
                    log.warning("transient %s error (attempt %d/%d), retrying in %ss: %s", self.provider, attempt + 1,
                                self.settings.max_retries, wait, str(e)[:200])
                    self.stats["retries"] += 1
                    attempt += 1
                    time.sleep(wait)
                    continue
                if remote and not e.auth and self._model_queue:      # same provider, next model
                    nxt = self._model_queue.pop(0)
                    log.warning("[MODEL FALLBACK] %s/%s failed (%s) -> trying %s/%s", self.provider, self.model_name, str(e)[:160], self.provider, nxt)
                    self.events.append({"ts": time.time(), "event": "model_fallback", "from": self.model_name, "to": nxt, "reason": str(e)[:300]})
                    self.stats["fallbacks"] += 1
                    self.selection_reason += f"; switched to {nxt} after {self.model_name} failed"
                    self._p = build_provider(self.provider, nxt, self.settings)
                    attempt = 0
                    continue
                if remote and self.settings.fallback_to_local:
                    self._fallback(str(e))
                    attempt = 0
                    continue
                self._record(Completion("", provider=self.provider, model=self.model_name), time.time() - t0, purpose, str(e)[:300])
                raise
        dt = time.time() - t0
        c.provider, c.model, c.source = self.provider, self.model_name, "live"
        self.stats["live_calls"] += 1
        self.stats["seconds"] += dt
        self.stats["input_tokens"] += c.input_tokens or 0
        self.stats["output_tokens"] += c.output_tokens or 0
        log.info("LLM call (%s, live): %s/%s, %s in / %s out tokens, %d tool call(s), %.1fs", purpose, c.provider, c.model,
                 c.input_tokens, c.output_tokens, len(c.tool_calls), dt)
        self._record(c, dt, purpose)
        if self.settings.cache_mode == "readwrite":
            self.cache[key] = {k: v for k, v in c.__dict__.items() if k != "source"}
            self.cache_path.write_text(json.dumps(self.cache))
        return c

    # ------------------------------------------------------------------ public API
    def chat(self, messages: list, max_tokens: int = 256, purpose: str = "chat") -> str:
        return self.complete(messages, max_tokens=max_tokens, purpose=purpose).text

    @staticmethod
    def extract_json(text: str):
        """First JSON object/array in a reply; tolerates code fences, prose and Python literals (None, True, ')."""
        text = re.sub(r"```(?:json)?", "", text)
        starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
        if not starts:
            raise ValueError("no JSON found in model reply")
        start = min(starts)
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        return ast.literal_eval(chunk)
        raise ValueError("unterminated JSON in model reply")

    def structured(self, messages: list, schema: Type[BaseModel], max_tokens: int = 256, retries: int = 2,
                   purpose: Optional[str] = None) -> BaseModel:
        """Validated Pydantic output. Uses the provider's native JSON mode where available; on invalid output the model
        sees the validation error and tries again. Raises StructuredOutputError (a RuntimeError) after `retries`."""
        msgs, last = list(messages), None
        purpose = purpose or f"structured:{schema.__name__}"
        for attempt in range(retries + 1):
            reply = self.complete(msgs, max_tokens=max_tokens, json_mode=True, purpose=purpose).text
            try:
                data = self.extract_json(reply)
                if isinstance(data, list) and len(schema.model_fields) == 1:     # bare list for a one-field schema
                    data = {next(iter(schema.model_fields)): data}
                return schema.model_validate(data)
            except (ValueError, SyntaxError, ValidationError) as e:
                last = e
                log.info("structured output invalid (attempt %d): %s", attempt + 1, str(e)[:200])
                msgs = msgs + [{"role": "assistant", "content": reply},
                               {"role": "user", "content": f"That was not valid. Error: {str(e)[:300]}\n"
                                                           f"Reply with ONLY a JSON object matching: {json.dumps(schema.model_json_schema())[:1200]}"}]
        raise StructuredOutputError(f"structured output failed after {retries + 1} attempts: {last}")

    def call_tools(self, messages: list, tools: list, max_tokens: int = 256) -> Completion:
        schemas = [t if isinstance(t, dict) else tool_schema(t) for t in tools]
        return self.complete(messages, max_tokens=max_tokens, tools=schemas, purpose="tool_call")

    def run_agent(self, messages: list, tools: list, max_steps: int = 5, max_tokens: int = 256) -> AgentRun:
        """Tool-calling loop: the model decides which tools to call and with what arguments; tools run locally and their
        results go back to the model, until it answers without a tool call or `max_steps` is reached."""
        fn_map = {t.__name__: t for t in tools}
        schemas = [tool_schema(t) for t in tools]
        msgs, steps, seen = list(messages), [], {}
        for _ in range(max_steps):
            c = self.complete(msgs, max_tokens=max_tokens, tools=schemas, purpose="agent")
            if not c.tool_calls:
                return AgentRun(c.text, steps, msgs + [{"role": "assistant", "content": c.text}], "answered")
            msgs.append({"role": "assistant", "content": c.text, "tool_calls": c.tool_calls})
            for call in c.tool_calls:
                sig = (call["name"], json.dumps(call.get("arguments") or {}, sort_keys=True))
                if sig in seen:          # repeated identical call: don't re-run it, and nudge the model to answer
                    result, ok = seen[sig] + "\n(Repeat call: you already have this result. If you have enough evidence, give your answer now.)", True
                else:
                    result, ok = execute_tool(fn_map, call)
                    seen[sig] = result
                steps.append(AgentStep(call["name"], call.get("arguments") or {}, result, ok))
                log.info("tool %s(%s) -> %s%s", call["name"], json.dumps(call.get("arguments"))[:200], result[:160], "" if ok else " [error]")
                msgs.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result})
        return AgentRun("", steps, msgs, "max_steps")

    # ------------------------------------------------------------------ provenance
    def report(self, show: bool = True):
        """Print the provenance summary (show=True) or return it as a dict (show=False). Always written to outputs/llm_provenance.json."""
        recs = [json.loads(l) for l in self.calls_path.read_text().splitlines() if l.strip()]
        by = {}
        for r in recs:
            k = (r["provider"], r["model"], r["source"])
            d = by.setdefault(k, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "est_cost_usd": None, "errors": 0})
            d["calls"] += 1
            d["input_tokens"] += r["input_tokens"] or 0
            d["output_tokens"] += r["output_tokens"] or 0
            d["seconds"] += r["seconds"]
            if r["est_cost_usd"] is not None:          # None = no known price (not "free")
                d["est_cost_usd"] = (d["est_cost_usd"] or 0.0) + r["est_cost_usd"]
            d["errors"] += bool(r["error"])
        summary = {"provider": self.provider, "model": self.model_name, "revision": getattr(self._p, "revision", None),
                   "selection": self.selection_reason, "cache_mode": self.settings.cache_mode, "stats": self.stats,
                   "events": self.events, "python": platform.python_version(),
                   "by_source": [{"provider": p, "model": m, "source": s, **v} for (p, m, s), v in by.items()]}
        (self.out / "llm_provenance.json").write_text(json.dumps(summary, indent=1, default=str))
        if show:
            print("LLM provenance for this run")
            print(f"  {self.provider}/{self.model_name}: {self.selection_reason}")
            for row in summary["by_source"]:
                cost = (", $0 (local)" if row["provider"] in ("huggingface", "ollama") or row["source"] == "cache"
                        else f", est. ${row['est_cost_usd']:.4f}" if row["est_cost_usd"] is not None
                        else ", cost not estimated (set LLM_PRICE_INPUT_PER_M / LLM_PRICE_OUTPUT_PER_M; free-tier keys are not billed)")
                print(f"  {row['source'].upper():5s} {row['provider']}/{row['model']}: {row['calls']} calls, "
                      f"{row['input_tokens']} in / {row['output_tokens']} out tokens, {row['seconds']:.0f}s{cost}")
            print(f"  live calls: {self.stats['live_calls']} | replayed from cache: {self.stats['cache_hits']} | "
                  f"retries: {self.stats['retries']} | fallbacks: {self.stats['fallbacks']}")
            if self.stats["cache_hits"]:
                print(f"  NOTE: {self.stats['cache_hits']} response(s) above were replayed from outputs/llm_cache.json, not generated in this run.")
            return None
        return summary
