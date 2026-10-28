"""diag_calls.py - how many send_email calls each response makes, and whether
attacker sends co-occur with the user's legitimate send.

Tests the hypothesis that the authorization effect depends on parallel tool
calling: a model that emits one call per turn and uses it for the user's send
has no second call to give the attacker.

Usage:
  python diag_calls.py results_full\\live_gpt-4o-mini.jsonl results_full\\live_meta-llama_Llama-3.3-70B-Instruct-Turbo.jsonl
"""
import json
import sys
from collections import Counter, defaultdict

from aipi import prompts as P
from aipi import scoring

for path in sys.argv[1:]:
    cells = defaultdict(lambda: {"n": 0, "calls": Counter(), "atk": 0, "atk_with_legit": 0,
                                 "atk_only": 0})
    model = None
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" in r or r.get("transform") != "none":
            continue
        model = r["model"]
        sends = scoring.send_calls(r["calls"])
        target = P.target_address("none")
        legit = P.EXPECTED_RECIPIENT[r["regime"]]
        rc = [scoring.recipients(c["args"]) for c in sends]
        atk = any(target in x for x in rc)
        ok = bool(legit) and any(legit in x for x in rc)
        c = cells[(r["regime"], r["condition"])]
        c["n"] += 1
        c["calls"][min(len(sends), 3)] += 1
        if atk:
            c["atk"] += 1
            c["atk_with_legit"] += ok
            c["atk_only"] += not ok
    print(f"\n=== {model}")
    print(f"{'reg':<4}{'cond':<11}{'n':>6}  {'0 calls':>8}{'1':>7}{'2':>7}{'3+':>7}"
          f"  {'ASR':>6}{'atk+legit':>11}{'atk only':>10}")
    for (rg, cd), c in sorted(cells.items()):
        n = c["n"]
        pct = lambda k: 100 * c["calls"][k] / n
        print(f"{rg:<4}{cd:<11}{n:>6}  {pct(0):>7.1f}%{pct(1):>6.1f}%{pct(2):>6.1f}%{pct(3):>6.1f}%"
              f"  {100*c['atk']/n:>5.1f}%{100*c['atk_with_legit']/n:>10.1f}%{100*c['atk_only']/n:>9.1f}%")
