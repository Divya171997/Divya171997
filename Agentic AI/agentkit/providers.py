"""Provider adapters with one interface: `complete(messages, max_tokens, tools=None, json_mode=False) -> Completion`.

Messages are plain dicts in OpenAI style:
  {"role": "system"|"user"|"assistant"|"tool", "content": str,
   "tool_calls": [{"id", "name", "arguments": dict}]   (assistant, optional),
   "tool_call_id": str, "name": str                    (tool)}
Tools are OpenAI-format function schemas (see agentkit.tools).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import Settings


@dataclass
class Completion:
    text: str
    tool_calls: list = field(default_factory=list)      # [{"id", "name", "arguments": dict}]
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    provider: str = ""
    model: str = ""
    source: str = "live"                                # "live" or "cache"


class ProviderError(RuntimeError):
    """A provider call failed. `transient` errors are retried; `auth` errors trigger the local fallback."""

    def __init__(self, msg: str, transient: bool = False, auth: bool = False):
        super().__init__(msg)
        self.transient, self.auth = transient, auth


_TRANSIENT = ("ratelimit", "rate limit", "timeout", "timed out", "temporarily", "unavailable", "overloaded",
              "connection", "resourceexhausted", "429", "500", "502", "503", "504", "internalserver")
_AUTH = ("authentication", "unauthorized", "permission", "api key", "api_key", "invalid key", "401", "403")


def classify(e: Exception, provider: str) -> ProviderError:
    text = f"{type(e).__name__}: {e}"
    low = text.lower()
    return ProviderError(f"{provider}: {text[:400]}", transient=any(t in low for t in _TRANSIENT),
                         auth=any(t in low for t in _AUTH))


# --------------------------------------------------------------------------------------------------
# LangChain-backed providers: OpenAI (and OpenAI-compatible servers), Google Gemini, Ollama
# --------------------------------------------------------------------------------------------------
def to_langchain(messages: list) -> list:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    out = []
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content))
        elif role == "user":
            out.append(HumanMessage(content))
        elif role == "assistant":
            calls = [{"name": c["name"], "args": c["arguments"], "id": c["id"], "type": "tool_call"} for c in m.get("tool_calls") or []]
            out.append(AIMessage(content=content, tool_calls=calls))
        elif role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=m["tool_call_id"]))
        else:
            raise ValueError(f"unknown role {role!r}")
    return out


def _text_of(msg) -> str:
    c = msg.content
    if isinstance(c, list):                       # some providers return content blocks
        c = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return c or ""


class LangChainProvider:
    name = "langchain"

    def __init__(self, model: str, settings: Settings, chat_factory=None):
        self.model, self.settings = model, settings
        self._factory = chat_factory or self._build
        self._instances = {}

    def _build(self, max_tokens: int, json_mode: bool):   # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _chat(self, max_tokens: int, json_mode: bool):
        key = (max_tokens, json_mode)
        if key not in self._instances:
            self._instances[key] = self._factory(max_tokens, json_mode)
        return self._instances[key]

    def complete(self, messages, max_tokens=256, tools=None, json_mode=False) -> Completion:
        chat = self._chat(max_tokens, json_mode and not tools)
        if tools:
            chat = chat.bind_tools(tools)
        try:
            msg = chat.invoke(to_langchain(messages))
        except Exception as e:
            raise classify(e, self.name) from e
        usage = getattr(msg, "usage_metadata", None) or {}
        calls = [{"id": c.get("id") or f"call_{i}", "name": c["name"], "arguments": c.get("args") or {}}
                 for i, c in enumerate(getattr(msg, "tool_calls", None) or [])]
        return Completion(text=_text_of(msg), tool_calls=calls, input_tokens=usage.get("input_tokens"),
                          output_tokens=usage.get("output_tokens"), provider=self.name, model=self.model)


class OpenAIProvider(LangChainProvider):
    name = "openai"

    def _build(self, max_tokens, json_mode):
        from langchain_openai import ChatOpenAI
        s = self.settings
        reasoning = self.model.startswith(("o1", "o3", "o4", "gpt-5"))   # these reject temperature != 1
        kw = dict(model=self.model, api_key=s.openai_api_key or "not-needed", timeout=s.timeout, max_retries=0,
                  max_tokens=max_tokens, seed=s.seed)
        if s.openai_base_url:
            kw["base_url"] = s.openai_base_url
        if not reasoning:
            kw["temperature"] = 0
        if json_mode:
            kw["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kw)


class GeminiProvider(LangChainProvider):
    name = "gemini"

    def _build(self, max_tokens, json_mode):
        from langchain_google_genai import ChatGoogleGenerativeAI
        s = self.settings
        kw = dict(model=self.model, google_api_key=s.gemini_api_key, max_output_tokens=max_tokens, timeout=s.timeout, max_retries=0)
        if not re.match(r"gemini-([3-9]|\d{2,})", self.model):   # Gemini 3+ use fixed sampling and warn if temperature is sent
            kw["temperature"] = 0
        if json_mode:
            kw["response_mime_type"] = "application/json"
        return ChatGoogleGenerativeAI(**kw)


class OllamaProvider(LangChainProvider):
    name = "ollama"

    def _build(self, max_tokens, json_mode):
        from langchain_ollama import ChatOllama
        s = self.settings
        kw = dict(model=self.model, base_url=s.ollama_host, temperature=0, seed=s.seed, num_predict=max_tokens)
        if json_mode:
            kw["format"] = "json"
        if "qwen3" in self.model:
            kw["reasoning"] = False        # skip the thinking phase: faster and cheaper
        return ChatOllama(**kw)


# --------------------------------------------------------------------------------------------------
# Local Hugging Face Transformers provider (fully offline once weights are downloaded)
# --------------------------------------------------------------------------------------------------
_TOOL_RX = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_THINK_RX = re.compile(r"<think>.*?</think>", re.S)


class HuggingFaceProvider:
    name = "huggingface"

    def __init__(self, model: str, settings: Settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model, self.settings, self._torch = model, settings, torch
        torch.manual_seed(settings.seed)
        cuda = torch.cuda.is_available()
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="auto" if cuda else None)
        self.lm.eval()
        self.device = "cuda" if cuda else "cpu"
        self.revision = getattr(self.lm.config, "_commit_hash", None)
        self.load_seconds = round(time.time() - t0, 1)
        self._template_kw = {"enable_thinking": False} if "qwen3" in model.lower() else {}

    @staticmethod
    def _to_hf(messages: list) -> list:
        out = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                out.append({"role": "assistant", "content": m.get("content") or "",
                            "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                           for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": m["content"], "name": m.get("name", "")})
            else:
                out.append({"role": m["role"], "content": m.get("content") or ""})
        return out

    def complete(self, messages, max_tokens=256, tools=None, json_mode=False) -> Completion:
        torch = self._torch
        try:
            enc = self.tok.apply_chat_template(self._to_hf(messages), tools=tools or None, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True, **self._template_kw).to(self.lm.device)
            with torch.no_grad():
                out = self.lm.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                       pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        except Exception as e:
            raise ProviderError(f"huggingface: {type(e).__name__}: {e}") from e
        new = out[0][enc["input_ids"].shape[1]:]
        raw = self.tok.decode(new, skip_special_tokens=False)
        for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"):
            raw = raw.replace(t, "")
        raw = _THINK_RX.sub("", raw)
        calls = []
        for i, m in enumerate(_TOOL_RX.finditer(raw)):
            try:
                d = json.loads(m.group(1))
                args = d.get("arguments") or d.get("parameters") or {}
                calls.append({"id": f"call_{i}", "name": d["name"], "arguments": json.loads(args) if isinstance(args, str) else args})
            except (ValueError, KeyError):
                continue                       # malformed call: left in the text, so the caller sees it
        text = _TOOL_RX.sub("", raw).strip() if calls else raw.strip()
        return Completion(text=text, tool_calls=calls, input_tokens=int(enc["input_ids"].shape[1]),
                          output_tokens=int(len(new)), provider=self.name, model=self.model)


def build_provider(name: str, model: str, settings: Settings):
    cls = {"openai": OpenAIProvider, "gemini": GeminiProvider, "ollama": OllamaProvider, "huggingface": HuggingFaceProvider}[name]
    return cls(model, settings)
