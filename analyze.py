"""analyze.py - score logged live runs under every structural defense.

For each (model, regime, prompt condition, transform, structural defense):
  rates are averaged over samples within an attack, then over attacks, and
  95% CIs come from a bootstrap over attacks (the unit that was sampled),
  so repeated samples of one attack are not treated as independent.

Usage:
  python analyze.py results_v2/live_*.jsonl --out results_v2/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict

from aipi import prompts as P
from aipi.scoring import DEFENSES, apply_defense, outcomes

METRICS = ("attacker_send", "strict_exfil", "task_done", "any_send")


def load_attack_index(path: str) -> dict[str, dict]:
    idx = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            idx[str(r.get("RowKey") or r.get("job_id"))] = r
    return idx


def bootstrap_ci(values: list[float], reps: int, seed: int) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng, n = random.Random(seed), len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--data", default="data/slice_2500.jsonl")
    p.add_argument("--out", default="results_v2/summary.csv")
    p.add_argument("--reps", type=int, default=2000)
    a = p.parse_args()

    attacks = load_attack_index(a.data)
    # (model, regime, cond, transform, defense) -> attack id -> metric -> [per-sample values]
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    errors, providers = defaultdict(int), {}

    for path in a.logs:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            providers[r["model"]] = r["provider"]
            base = (r["model"], r["regime"], r["condition"], r["transform"])
            if "error" in r:
                errors[base] += 1
                continue
            atk = dict(attacks[r["id"]], _id=r["id"])
            emails = [e.__dict__ for e in P.build_context(atk, r["transform"], r["seed"])]
            for d in DEFENSES:
                executed = apply_defense(d, r["regime"], emails, r["calls"])
                o = outcomes(r["regime"], r["transform"], r["marker"], executed)
                for m in METRICS:
                    if o[m] is not None:
                        acc[base + (d,)][r["id"]][m].append(float(o[m]))

    rows = []
    for key in sorted(acc):
        per_attack = acc[key]
        row = dict(zip(("model", "regime", "condition", "transform", "defense"), key))
        row["provider"] = providers[key[0]]
        row["n_attacks"] = len(per_attack)
        row["samples_per_attack"] = max(len(v["any_send"]) for v in per_attack.values())
        row["errors"] = errors[key[:4]]
        for m in METRICS:
            vals = [sum(v[m]) / len(v[m]) for v in per_attack.values() if v[m]]
            if not vals:
                continue
            stable = int(hashlib.sha256("|".join(key + (m,)).encode()).hexdigest()[:8], 16)
            lo, hi = bootstrap_ci(vals, a.reps, seed=stable)
            row[m] = round(100 * sum(vals) / len(vals), 2)
            row[m + "_lo"], row[m + "_hi"] = round(100 * lo, 2), round(100 * hi, 2)
        rows.append(row)

    cols = ["provider", "model", "regime", "condition", "transform", "defense",
            "n_attacks", "samples_per_attack", "errors"]
    for m in METRICS:
        cols += [m, m + "_lo", m + "_hi"]
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"{'model':<28}{'reg':<4}{'cond':<11}{'xform':<14}{'defense':<14}"
          f"{'ASR':>7}{'strict':>8}{'task':>7}{'n':>6}")
    for r in rows:
        if any(prov == "mock" for prov in [r["provider"]]):
            tag = " [MOCK - not reportable]"
        else:
            tag = ""
        print(f"{r['model'][:27]:<28}{r['regime']:<4}{r['condition']:<11}{r['transform']:<14}"
              f"{r['defense']:<14}{r.get('attacker_send', float('nan')):>7.1f}"
              f"{r.get('strict_exfil', float('nan')):>8.1f}"
              f"{r.get('task_done', float('nan')) if r.get('task_done') is not None else float('nan'):>7.1f}"
              f"{r['n_attacks']:>6}{tag}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
