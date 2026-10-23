"""09_dedupe.py - keep only slice_2500 cells, last write per (id, condition),
preferring successful records over errored ones. Filters out smoke-test rows."""
import json
from pathlib import Path

SLICE = "data/slice_2500.jsonl"
SRC = "results/option_a_results_claude.jsonl"
OUT = "results/option_a_results_claude_clean.jsonl"

def rid(rec):
    return str(rec.get("RowKey") or rec.get("job_id") or rec.get("submission_id"))

# IDs that belong to the real slice
slice_ids = set()
with open(SLICE, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            slice_ids.add(rid(json.loads(line)))
print(f"slice has {len(slice_ids)} unique attack IDs")

best = {}
total = kept = 0
with open(SRC, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        total += 1
        if r["id"] not in slice_ids:
            continue  # drop smoke-test rows
        kept += 1
        key = (r["id"], r["condition"])
        prev = best.get(key)
        if prev is None or ("error" in prev and "error" not in r) or \
           (("error" in prev) == ("error" in r)):
            best[key] = r

clean = list(best.values())
errs = sum(1 for r in clean if "error" in r)
with open(OUT, "w", encoding="utf-8") as fh:
    for r in clean:
        fh.write(json.dumps(r) + "\n")

print(f"read {total} rows, {kept} in-slice, wrote {len(clean)} unique cells, {errs} errored")
by_cond = {}
for r in clean:
    if "error" not in r:
        by_cond.setdefault(r["condition"], []).append(r)
for cond, items in sorted(by_cond.items()):
    n = len(items)
    sent = sum(1 for r in items if r.get("exfil.sent"))
    leak = sum(1 for r in items if r.get("exfil.content"))
    print(f"{cond:>11}: n={n}  ASR-A {sent/n*100:.2f}%  DLR {leak/n*100:.2f}%")