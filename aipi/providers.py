"""Provider adapters. They translate the shared prompt and tool definition into
each API's wire format and return a normalized record. They never change the
prompt text, the tool schema, or the decoding parameters.

Every adapter returns:
    {"calls": [{"name": str, "args": dict}], "text": str,
     "served_model": str, "stop_reason": str, "usage": dict}
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from . import prompts as P


class OpenAICompatible:
    """OpenAI, and OpenAI-compatible hosts (Together AI) via OPENAI_BASE_URL."""

    def __init__(self, parallel_tool_calls: bool | None = None) -> None:
        from openai import AsyncOpenAI
        self.parallel = parallel_tool_calls   # None = provider default (not sent)
        self.client = AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None,
                                  api_key=os.environ.get("OPENAI_API_KEY"))
        self.tool = {"type": "function", "function": {
            "name": P.TOOL_NAME, "description": P.TOOL_DESCRIPTION,
            "parameters": P.TOOL_PARAMETERS}}

    async def complete(self, model, system, user, temperature, max_tokens) -> dict:
        resp = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            tools=[self.tool], tool_choice="auto", max_tokens=max_tokens,
            **({} if temperature is None else {"temperature": temperature}),
            **({} if self.parallel is None else {"parallel_tool_calls": self.parallel}))
        ch = resp.choices[0]
        calls = []
        for tc in ch.message.tool_calls or []:
            if tc.type != "function":
                continue
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": tc.function.arguments}
            calls.append({"name": tc.function.name, "args": args})
        return {"calls": calls, "text": ch.message.content or "",
                "served_model": resp.model, "stop_reason": ch.finish_reason,
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
                "usage": resp.usage.model_dump() if resp.usage else {}}


class Anthropic:
    def __init__(self) -> None:
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic()
        self.tool = {"name": P.TOOL_NAME, "description": P.TOOL_DESCRIPTION,
                     "input_schema": P.TOOL_PARAMETERS}

    async def complete(self, model, system, user, temperature, max_tokens) -> dict:
        # Some SDK releases no longer expose `temperature` as a keyword; send it
        # in the request body so the API itself accepts or rejects it.
        resp = await self.client.messages.create(
            model=model, system=system,
            messages=[{"role": "user", "content": user}],
            tools=[self.tool], tool_choice={"type": "auto"},
            max_tokens=max_tokens,
            **({} if temperature is None else {"extra_body": {"temperature": temperature}}))
        calls = [{"name": b.name, "args": dict(b.input)}
                 for b in resp.content if b.type == "tool_use"]
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {"calls": calls, "text": text, "served_model": resp.model,
                "stop_reason": resp.stop_reason,
                "usage": resp.usage.model_dump() if resp.usage else {}}


class Mock:
    """Pipeline test double. Output is NOT model behavior and must never be
    reported; rows are tagged provider='mock'. It follows an injected address
    with fixed probability, otherwise performs the user's task."""

    def __init__(self, p_follow: float = 0.4, seed: int = 0) -> None:
        self.p, self.rng = p_follow, random.Random(seed)

    async def complete(self, model, system, user, temperature, max_tokens) -> dict:
        calls = []
        low = user.lower()
        if self.rng.random() < self.p:
            for addr in (P.ATTACKER_ADDRESS, P.INTERNAL_DROP):
                if addr in low:
                    calls.append({"name": P.TOOL_NAME,
                                  "args": {"to": addr, "body": "confirmation"}})
                    break
        if "send the summary to " + P.DANA in low:
            calls.append({"name": P.TOOL_NAME, "args": {"to": P.DANA, "body": "Summary."}})
        elif "reviewer that dana names" in low:
            calls.append({"name": P.TOOL_NAME, "args": {"to": P.RAJ, "body": "Summary."}})
        return {"calls": calls, "text": "", "served_model": "mock",
                "stop_reason": "mock", "usage": {}}


def make_provider(name: str, parallel_tool_calls: bool | None = None):
    if name == "openai":
        return OpenAICompatible(parallel_tool_calls)
    if name == "anthropic":
        return Anthropic()
    if name == "mock":
        return Mock()
    raise ValueError(f"unknown provider: {name}")
