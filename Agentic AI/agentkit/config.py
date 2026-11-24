"""Configuration and automatic provider/model selection.

Everything is driven by environment variables (or a `.env` file), so adding an API key later
requires no code change. See `agentkit/README.md` for the full list.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("agentkit")

PROVIDERS = ("openai", "gemini", "ollama", "huggingface")

# Cost-efficient, non-reasoning defaults (verified current as of Nov 2026); override with OPENAI_MODEL / GEMINI_MODEL.
DEFAULT_API_MODELS = {"openai": "gpt-4.1-mini", "gemini": "gemini-3.5-flash-lite"}
# Tried in order on the same provider when the primary model keeps failing (overload, removal), before the local fallback.
DEFAULT_FALLBACK_MODELS = {"openai": ["gpt-4.1-nano", "gpt-4o-mini"], "gemini": ["gemini-flash-lite-latest", "gemini-3.1-flash-lite"],
                           "ollama": [], "huggingface": []}

# Local Hugging Face models, best first. All Apache-2.0, all support native tool calling via their chat template.
# (repo id, approximate bf16 weight size in GB)
LOCAL_HF_MODELS = [
    ("Qwen/Qwen3-8B", 16.4),
    ("Qwen/Qwen3-4B", 8.1),
    ("Qwen/Qwen3-1.7B", 4.1),
    ("Qwen/Qwen3-0.6B", 1.5),
]
MEMORY_HEADROOM = 1.25          # weights x headroom must fit in free memory (activations, other notebook data)

# Ollama models, best first (prefix match against `ollama list`); otherwise the first installed model is used.
OLLAMA_PREFERENCE = ["qwen3:32b", "qwen3:30b", "qwen3:14b", "qwen2.5:14b", "qwen3:8b", "llama3.1:8b",
                     "qwen2.5:7b", "mistral", "qwen3:4b", "llama3.2", "qwen3:1.7b"]

# List prices in USD per 1M tokens (input, output) for cost estimates; override with LLM_PRICE_INPUT_PER_M / ..._OUTPUT_PER_M.
KNOWN_PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40), "gpt-4o-mini": (0.15, 0.60),
                "gpt-5-mini": (0.25, 2.00), "gpt-5-nano": (0.05, 0.40)}


def load_env(start: Optional[Path] = None, levels: int = 3) -> list:
    """Load `.env` files from the working directory and up to `levels` parents (closest wins). Returns the files loaded."""
    try:
        from dotenv import load_dotenv
    except ImportError:          # python-dotenv is optional
        return []
    here = Path(start or Path.cwd()).resolve()
    loaded = []
    for d in [here, *list(here.parents)[:levels]]:
        f = d / ".env"
        if f.is_file():
            load_dotenv(f, override=False)
            loaded.append(str(f))
    return loaded


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default


@dataclass
class Settings:
    provider: str = "auto"                 # auto | openai | gemini | ollama | huggingface
    model: Optional[str] = None            # overrides the provider's default model
    openai_api_key: Optional[str] = None
    openai_model: str = DEFAULT_API_MODELS["openai"]
    openai_base_url: Optional[str] = None  # any OpenAI-compatible server (vLLM, LM Studio, Azure proxy, ...)
    gemini_api_key: Optional[str] = None
    gemini_model: str = DEFAULT_API_MODELS["gemini"]
    ollama_host: str = "http://localhost:11434"
    ollama_model: Optional[str] = None
    hf_model: Optional[str] = None         # default: best local model that fits in memory
    cache_mode: str = "off"                # off | readwrite | replay
    fallback_to_local: bool = True         # API failure -> local model (logged and labelled)
    max_retries: int = 4
    min_interval: float = 0.0              # seconds between live API calls (pacing for free-tier rate limits)
    fallback_models: Optional[list] = None # same-provider models to try before the local fallback (None = defaults)
    timeout: float = 120.0
    seed: int = 0
    price_input_per_m: Optional[float] = None
    price_output_per_m: Optional[float] = None

    @classmethod
    def from_env(cls) -> "Settings":
        def f(name):
            v = _env(name)
            return float(v) if v else None
        s = cls(
            provider=(_env("LLM_PROVIDER", "auto") or "auto").lower(),
            model=_env("LLM_MODEL"),
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_model=_env("OPENAI_MODEL", DEFAULT_API_MODELS["openai"]),
            openai_base_url=_env("OPENAI_BASE_URL"),
            gemini_api_key=_env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY"),
            gemini_model=_env("GEMINI_MODEL", DEFAULT_API_MODELS["gemini"]),
            ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=_env("OLLAMA_MODEL"),
            hf_model=_env("HF_MODEL"),
            cache_mode=(_env("LLM_CACHE", "off") or "off").lower(),
            fallback_to_local=(_env("LLM_FALLBACK", "local") or "local").lower() not in ("none", "off", "false", "0"),
            max_retries=int(_env("LLM_MAX_RETRIES", "4")),
            min_interval=float(_env("LLM_MIN_INTERVAL", "0")),
            fallback_models=[m.strip() for m in _env("LLM_FALLBACK_MODELS").split(",") if m.strip()] if _env("LLM_FALLBACK_MODELS") else None,
            timeout=float(_env("LLM_TIMEOUT", "120")),
            seed=int(_env("LLM_SEED", "0")),
            price_input_per_m=f("LLM_PRICE_INPUT_PER_M"),
            price_output_per_m=f("LLM_PRICE_OUTPUT_PER_M"),
        )
        if s.provider not in ("auto", *PROVIDERS):
            raise ValueError(f"LLM_PROVIDER must be one of auto, {', '.join(PROVIDERS)}; got {s.provider!r}")
        if s.cache_mode not in ("off", "readwrite", "replay"):
            raise ValueError(f"LLM_CACHE must be off, readwrite or replay; got {s.cache_mode!r}")
        return s

    def public(self) -> dict:
        """Settings safe to print or log: secrets are reduced to present/absent."""
        d = dict(self.__dict__)
        for k in ("openai_api_key", "gemini_api_key"):
            d[k] = "set" if d[k] else "not set"
        return d


def free_memory_gb() -> tuple:
    """(free GB, device) for the device a local model would run on."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free / 1e9, "cuda"
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9, "cpu"
    except Exception:
        return 4.0, "cpu"


def pick_hf_model(free_gb: float) -> str:
    for repo, size in LOCAL_HF_MODELS:
        if size * MEMORY_HEADROOM <= free_gb:
            return repo
    return LOCAL_HF_MODELS[-1][0]


def discover_ollama(host: str, timeout: float = 1.0) -> Optional[list]:
    """Installed Ollama models, or None if no server is reachable."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return None


def choose_ollama_model(names: list) -> Optional[str]:
    for pref in OLLAMA_PREFERENCE:
        for n in names:
            if n == pref or n.startswith(pref + ":") or n.startswith(pref + "-") or (":" not in pref and n.split(":")[0] == pref):
                return n
    return names[0] if names else None


def resolve(s: Settings) -> tuple:
    """Pick (provider, model, reason). Explicit LLM_PROVIDER wins; otherwise:
    OpenAI key -> Gemini key -> running Ollama server with a model -> local Hugging Face model."""
    p = s.provider
    if p == "auto":
        if s.openai_api_key:
            p, why = "openai", "OPENAI_API_KEY is set"
        elif s.gemini_api_key:
            p, why = "gemini", "GOOGLE_API_KEY/GEMINI_API_KEY is set"
        else:
            names = discover_ollama(s.ollama_host)
            if names and (s.ollama_model or choose_ollama_model(names)):
                p, why = "ollama", f"no API key; Ollama server at {s.ollama_host} has {len(names)} model(s)"
            else:
                p, why = "huggingface", "no API key and no Ollama server: running offline with a local Hugging Face model"
    else:
        why = f"LLM_PROVIDER={p}"
    if s.model:
        return p, s.model, why + "; model from LLM_MODEL"
    if p == "openai":
        return p, s.openai_model, why
    if p == "gemini":
        return p, s.gemini_model, why
    if p == "ollama":
        model = s.ollama_model or choose_ollama_model(discover_ollama(s.ollama_host) or [])
        if not model:
            raise RuntimeError(f"LLM_PROVIDER=ollama but no model is installed at {s.ollama_host} (run `ollama pull qwen3:8b`) "
                               "or set OLLAMA_MODEL")
        return p, model, why
    if s.hf_model:
        return p, s.hf_model, why + "; model from HF_MODEL"
    free, device = free_memory_gb()
    return p, pick_hf_model(free), why + f"; best Apache-2.0 model that fits {free:.1f} GB free {device} memory"
