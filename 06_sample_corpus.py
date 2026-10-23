"""06_sample_corpus.py - proportional stratified slice of LLMail Phase 1.

Streams raw_submissions_phase1.jsonl (constant memory), filters to
email.retrieved == True, and draws a scenario-proportional sample with a fixed
seed. Writes the slice and the population weights for later CI weighting.
"""
import argparse
import json
import logging
import math
import os
import random
from collections import defaultdict
from pathlib import Path

# Path to the LLMail-Inject Phase 1 raw submissions (JSONL).
# Download from the official source (https://huggingface.co/datasets/microsoft/llmail-inject-challenge)
# and set LLMAIL_DATA to its location, or place it under ./data/.
CACHE = os.environ.get(
    "LLMAIL_DATA",
    str(Path("data") / "raw_submissions_phase1.jsonl"),
)

def objectives(rec: dict) -> dict:
    raw = rec.get("objectives")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def is_retrieved(rec: dict) -> bool:
    return objectives(rec).get("email.retrieved") is True


def content_marker_for(scenario: str) -> str:
    # level1 scenarios target the literal "confirmation". Higher levels may
    # target a stolen value; verify those against the challenge spec.
    return "confirmation"


def allocate(counts: dict, target: int) -> dict:
    total = sum(counts.values())
    raw = {s: target * n / total for s, n in counts.items()}
    floor = {s: int(math.floor(v)) for s, v in raw.items()}
    rem = target - sum(floor.values())
    order = sorted(raw, key=lambda s: raw[s] - floor[s], reverse=True)
    for s in order[:rem]:
        floor[s] += 1
    for s in floor:
        floor[s] = min(floor[s], counts[s])
    return floor


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=CACHE)
    p.add_argument("--target", type=int, default=2500)
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--out", default="data/slice_2500.jsonl")
    p.add_argument("--weights", default="data/slice_2500_weights.json")
    a = p.parse_args()

    # Pass 1: count retrieved records per scenario.
    counts = defaultdict(int)
    total = retrieved = 0
    logging.info("pass 1: counting (this reads the full 1.67 GB file)")
    with open(a.data, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if i % 50000 == 0:
                logging.info("  pass 1: %d lines", i)
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if is_retrieved(rec):
                retrieved += 1
                counts[rec.get("scenario", "unknown")] += 1

    logging.info("total=%d retrieved=%d scenarios=%d", total, retrieved, len(counts))
    alloc = allocate(dict(counts), min(a.target, retrieved))

    # Pass 2: per-stratum reservoir sampling (Algorithm R).
    rng = random.Random(a.seed)
    reservoir = {s: [] for s in alloc}
    seen = defaultdict(int)
    logging.info("pass 2: sampling")
    with open(a.data, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if i % 50000 == 0:
                logging.info("  pass 2: %d lines", i)
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not is_retrieved(rec):
                continue
            s = rec.get("scenario", "unknown")
            k = alloc.get(s, 0)
            if k == 0:
                continue
            seen[s] += 1
            r = reservoir[s]
            if len(r) < k:
                r.append(rec)
            else:
                j = rng.randint(0, seen[s] - 1)
                if j < k:
                    r[j] = rec

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    non_level1 = defaultdict(int)
    with open(a.out, "w", encoding="utf-8") as out:
        for s, recs in sorted(reservoir.items()):
            for rec in recs:
                rec["content_marker"] = content_marker_for(s)
                if not s.startswith("level1"):
                    non_level1[s] += 1
                out.write(json.dumps(rec) + "\n")
                n_written += 1

    weights = {s: {"retrieved_count": counts[s], "weight": counts[s] / retrieved}
               for s in counts}
    with open(a.weights, "w", encoding="utf-8") as wf:
        json.dump({"retrieved_total": retrieved, "weights": weights}, wf, indent=2)

    print(f"\nwrote {n_written} records to {a.out}")
    print(f"retrieved population: {retrieved}")
    print("\nallocation by scenario:")
    for s in sorted(alloc):
        if alloc[s]:
            print(f"  {s:>10}: {alloc[s]:4d}  (pop {counts[s]}, {counts[s]/retrieved*100:.2f}%)")
    if non_level1:
        print("\nNON-level1 rows (verify exfil.content target vs challenge spec):")
        for s, n in sorted(non_level1.items()):
            print(f"  {s}: {n}")
    else:
        print("\nall sampled rows are level1 (content marker 'confirmation' is correct)")


if __name__ == "__main__":
    main()