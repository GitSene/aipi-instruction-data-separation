"""12_wilson_ci.py - Wilson 95% CIs for the headline ASRs.
Counts successes/trials per model x condition from the saved result files."""
import json, math

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (p*100, max(0, (c-h))*100, min(1, (c+h))*100)

files = {"gpt-4o-mini": "results/option_a_results.jsonl",
         "claude": "results/option_a_results_claude_clean.jsonl"}

for label, path in files.items():
    counts = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        c = r["condition"]
        counts.setdefault(c, [0, 0])
        counts[c][1] += 1
        if r.get("exfil.sent"):
            counts[c][0] += 1
    print(f"\n{label}:")
    for cond, (k, n) in sorted(counts.items()):
        p, lo, hi = wilson(k, n)
        print(f"  {cond:>11}: ASR {p:.1f}%  95% CI [{lo:.1f}, {hi:.1f}]  (k={k}, n={n})")