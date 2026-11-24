"""Tool definitions from plain Python functions (type hints + docstring) and safe execution."""
from __future__ import annotations

import inspect
import json
from typing import Callable


def tool_schema(fn: Callable) -> dict:
    """OpenAI-format function schema, understood by every provider adapter."""
    from langchain_core.utils.function_calling import convert_to_openai_tool
    return convert_to_openai_tool(fn)


def execute_tool(fn_map: dict, call: dict, max_chars: int = 4000) -> tuple:
    """Run one tool call. Returns (result_text, ok). Unknown tools, bad arguments and exceptions are
    returned to the model as error messages instead of crashing the agent."""
    fn = fn_map.get(call["name"])
    if fn is None:
        return json.dumps({"error": f"unknown tool {call['name']!r}; available: {sorted(fn_map)}"}), False
    args = call.get("arguments") or {}
    try:
        bound = inspect.signature(fn).bind(**args)
    except TypeError as e:
        return json.dumps({"error": f"invalid arguments for {call['name']}: {e}"}), False
    try:
        result = fn(*bound.args, **bound.kwargs)
    except Exception as e:            # the tool's own failure is information for the agent
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), False
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return text[:max_chars], True
