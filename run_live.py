"""run_live.py - authorization-conditioned live evaluation (one harness, all providers).

Each cell = (attack, regime, prompt condition, transform, sample index).
Raw provider output is logged per cell; scoring and structural defenses are
applied afterwards by analyze.py, so the log is the single source of truth.

Examples:
  # OpenAI
  python run_live.py --provider openai --model gpt-4o-mini --subset 1000 --samples 3 --temperature 0.7
  # Together AI (OpenAI-compatible): set OPENAI_BASE_URL and OPENAI_API_KEY first
  python run_live.py --provider openai --model meta-llama/Llama-3.3-70B-Instruct-Turbo --subset 1000 --samples 3 --temperature 0.7
  # Anthropic
  python run_live.py --provider anthropic --model claude-sonnet-4-6 --subset 1000 --samples 3 --temperature 0.7
  # Pipeline check without API calls (output is not reportable)
  python run_live.py --provider mock --model mock --subset 100 --samples 2 --out-dir results_mock
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from aipi import prompts as P
from aipi.providers import make_provider


def load_attacks(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            obj = r.get("objectives")
            obj = json.loads(obj) if isinstance(obj, str) else (obj or {})
            if obj.get("email.retrieved") is not True:
                continue                      # honest denominator: attacks that reached the model
            r["_id"] = str(r.get("RowKey") or r.get("job_id"))
            r["_marker"] = r.get("content_marker") or "confirmation"
            out.append(r)
    return out


def stratified_subset(attacks: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional by scenario (S1-S4 via level prefix), deterministic."""
    if not n or n >= len(attacks):
        return attacks
    by = defaultdict(list)
    for a in attacks:
        by[a.get("scenario", "")[:6]].append(a)
    key = lambda a: hashlib.sha256(f"{seed}:{a['_id']}".encode()).hexdigest()
    # largest-remainder allocation so strata sum exactly to n
    quotas = {s: n * len(v) / len(attacks) for s, v in by.items()}
    alloc = {s: int(q) for s, q in quotas.items()}
    for s in sorted(quotas, key=lambda s: quotas[s] - alloc[s], reverse=True)[: n - sum(alloc.values())]:
        alloc[s] += 1
    picked = []
    for stratum, items in sorted(by.items()):
        picked += sorted(items, key=key)[: alloc[stratum]]
    return picked


def cell_key(aid, regime, cond, transform, s) -> str:
    return f"{aid}|{regime}|{cond}|{transform}|{s}"


async def main_async(a) -> None:
    attacks = stratified_subset(load_attacks(a.data), a.subset, a.seed)
    logging.info("attacks: %d", len(attacks))
    safe = a.model.replace("/", "_").replace(":", "_")
    out = Path(a.out_dir) / f"live_{safe}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if "error" not in r:
                done.add(r["key"])
    cells = [(atk, rg, c, t, s)
             for atk in attacks for rg in a.regimes for c in a.conditions
             for t in a.transforms for s in range(a.samples)
             if cell_key(atk["_id"], rg, c, t, s) not in done]
    random.Random(a.seed).shuffle(cells)          # spread provider drift across conditions
    logging.info("pending cells: %d (done %d)", len(cells), len(done))

    provider = make_provider(a.provider, False if a.no_parallel_tools else None)
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    fh = open(out, "a", encoding="utf-8")
    n, t0 = 0, time.time()

    async def run(atk, regime, cond, transform, s):
        nonlocal n
        if a.context == "legacy":
            system, user = P.build_legacy_prompt(atk, cond)
        else:
            emails = P.build_context(atk, transform, a.seed)
            system, user = P.build_prompt(regime, cond, emails)
        row = {"key": cell_key(atk["_id"], regime, cond, transform, s),
               "id": atk["_id"], "scenario": atk.get("scenario"),
               "regime": regime, "condition": cond, "transform": transform,
               "sample": s, "note_first": P.note_first(atk["_id"], a.seed),
               "provider": a.provider, "model": a.model,
               "temperature": a.temperature, "max_tokens": a.max_tokens,
               "seed": a.seed, "marker": atk["_marker"], "context": a.context,
               "parallel_tool_calls": False if a.no_parallel_tools else "default"}
        async with sem:
            delay, err = 1.0, None
            for _ in range(a.max_retries):
                try:
                    res = await provider.complete(a.model, system, user,
                                                  a.temperature, a.max_tokens)
                    row.update(res)
                    err = None
                    break
                except Exception as e:            # noqa: BLE001 - logged, retried
                    err = f"{type(e).__name__}: {e}"
                    status = getattr(e, "status_code", None)
                    logging.warning("call failed (%s): %s", status, err[:300])
                    if status in (400, 401, 402, 403, 404, 422) or isinstance(e, TypeError):
                        logging.error("non-retryable %s: %s", status, err[:500])
                        break                     # request itself is wrong; do not retry
                    await asyncio.sleep(delay + random.random() * delay)
                    delay = min(delay * 2, 60)
            if err:
                row["error"] = err
        row["ts"] = datetime.now(timezone.utc).isoformat()
        async with lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n += 1
            if n % 100 == 0:
                logging.info("%d/%d (%.1f/s)", n, len(cells), n / (time.time() - t0))

    await asyncio.gather(*(run(*c) for c in cells))
    fh.close()
    logging.info("wrote %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", required=True, choices=["openai", "anthropic", "mock"])
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/slice_2500.jsonl")
    p.add_argument("--subset", type=int, default=0, help="stratified subset size (0 = all)")
    p.add_argument("--regimes", nargs="+", default=list(P.REGIMES), choices=P.REGIMES)
    p.add_argument("--conditions", nargs="+", default=list(P.CONDITIONS), choices=P.CONDITIONS)
    p.add_argument("--transforms", nargs="+", default=["none"], choices=P.TRANSFORMS)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=lambda v: None if v == "default" else float(v),
                   default=0.0, help="number, or 'default' to omit (provider default sampling)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--out-dir", default="results_v2")
    p.add_argument("--context", choices=["two_email", "legacy"], default="two_email",
                   help="legacy = original paper's single-email context (R0 only; ablation)")
    p.add_argument("--no-parallel-tools", action="store_true",
                   help="ask the provider for at most one tool call per response (OpenAI only)")
    a = p.parse_args()
    if a.no_parallel_tools:
        if a.provider != "openai":
            raise SystemExit("--no-parallel-tools is supported for --provider openai only")
        if "noparallel" not in a.out_dir:
            raise SystemExit("use an output directory whose name contains 'noparallel'")
    if a.context == "legacy":
        if a.regimes != ["R0"] or a.transforms != ["none"]:
            raise SystemExit("--context legacy requires --regimes R0 --transforms none")
        if "legacy" not in a.out_dir:
            raise SystemExit("legacy output must go to a directory whose name contains 'legacy'")
    if a.provider == "mock" and "mock" not in a.out_dir:
        raise SystemExit("mock output must go to a directory whose name contains 'mock'")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
