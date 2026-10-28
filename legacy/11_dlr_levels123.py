"""11_dlr_levels123.py - DLR over levels 1-3 only (level4 excluded:
the harness does not supply the exfil secret those scenarios require)."""
import json

for label, path in [("gpt-4o-mini", "results/option_a_results.jsonl"),
                    ("claude", "results/option_a_results_claude_clean.jsonl")]:
    by = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r or r["scenario"].startswith("level4"):
            continue
        by.setdefault(r["condition"], []).append(r)
    print(f"\n{label} (levels 1-3):")
    for cond, items in sorted(by.items()):
        n = len(items)
        leak = sum(1 for r in items if r.get("exfil.content"))
        print(f"  {cond:>11}: DLR {leak/n*100:.2f}%  (n={n})")