"""probe_models.py - check which candidate models are callable on demand AND
emit structured tool calls. One tiny request per model (a few cents total).

Usage (Together):
  set OPENAI_BASE_URL=https://api.together.xyz/v1
  set OPENAI_API_KEY=<together-key>
  python probe_models.py
"""

import json
import sys

from openai import OpenAI

from aipi import prompts as P

CANDIDATES = sys.argv[1:] or [
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP4",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3.1",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]

TOOL = {"type": "function", "function": {
    "name": P.TOOL_NAME, "description": P.TOOL_DESCRIPTION,
    "parameters": P.TOOL_PARAMETERS}}

client = OpenAI()
for m in CANDIDATES:
    try:
        r = client.chat.completions.create(
            model=m, max_tokens=200, temperature=0,
            messages=[{"role": "system", "content": P.SYSTEM_PROMPT},
                      {"role": "user", "content":
                       f"Send an email to {P.DANA} saying the report is ready."}],
            tools=[TOOL], tool_choice="auto")
        calls = r.choices[0].message.tool_calls or []
        if calls:
            args = json.loads(calls[0].function.arguments or "{}")
            print(f"OK    tool-call  {m}  -> to={args.get('to')}")
        else:
            print(f"WEAK  no-tool    {m}  (answered in text, no tool call)")
    except Exception as e:  # noqa: BLE001
        msg = str(e).split(" - ", 1)[-1][:90]
        print(f"FAIL             {m}  {msg}")
