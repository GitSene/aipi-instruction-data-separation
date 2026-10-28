"""stats_report.py - every number the paper reports, with 95% bootstrap CIs.

Resampling unit is the attack: per-attack rates are averaged over samples
first, then attacks are resampled. Differences between two cells of the same
model are paired on attack (same attacks, same resample).

Usage (labels are free text; one LABEL=PATH per run):
  python stats_report.py gpt=results_full\\live_gpt-4o-mini.jsonl ^
      llama=results_full\\live_meta-llama_Llama-3.3-70B-Instruct-Turbo.jsonl ^
      oss=results_full\\live_openai_gpt-oss-120b.jsonl ^
      claude=results_full\\live_claude-sonnet-4-6.jsonl ^
      gpt_noparallel=results_noparallel\\live_gpt-4o-mini.jsonl

Writes stats_report.txt next to the script and prints the same text.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

from aipi import prompts as P
from aipi.scoring import DEFENSES, apply_defense, outcomes

REPS = 5000
SEED = 20261028
METRICS = ("attacker_send", "strict_exfil", "task_done")
OUT = []


def say(line: str = "") -> None:
    print(line)
    OUT.append(line)


def load_attacks(path="data/slice_2500.jsonl"):
    idx = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            idx[str(r.get("RowKey") or r.get("job_id"))] = r
    return idx


def load_run(path, attacks):
    """(regime, cond, transform, defense) -> metric -> {attack: mean over samples}"""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            continue
        atk = dict(attacks[r["id"]], _id=r["id"])
        emails = ([] if r.get("context") == "legacy" else
                  [e.__dict__ for e in P.build_context(atk, r["transform"], r["seed"])])
        for d in DEFENSES:
            o = outcomes(r["regime"], r["transform"], r["marker"],
                         apply_defense(d, r["regime"], emails, r["calls"]))
            for m in METRICS:
                if o[m] is not None:
                    acc[(r["regime"], r["condition"], r["transform"], d)][m][r["id"]].append(float(o[m]))
    return {k: {m: {a: sum(v) / len(v) for a, v in per.items()} for m, per in ms.items()}
            for k, ms in acc.items()}


def ci(vals):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(SEED)
    means = v[rng.integers(0, len(v), (REPS, len(v)))].mean(1)
    return 100 * v.mean(), 100 * np.quantile(means, .025), 100 * np.quantile(means, .975)


def paired(a: dict, b: dict):
    """b - a, paired on attacks present in both. Returns diff, lo, hi, p, n."""
    ids = sorted(set(a) & set(b))
    d = np.array([b[i] - a[i] for i in ids])
    rng = np.random.default_rng(SEED)
    means = d[rng.integers(0, len(d), (REPS, len(d)))].mean(1)
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return 100 * d.mean(), 100 * np.quantile(means, .025), 100 * np.quantile(means, .975), p, len(ids)


def fmt(t):
    return f"{t[0]:5.1f} [{t[1]:4.1f}, {t[2]:4.1f}]"


def fmtd(t):
    p = "<0.001" if t[3] < 0.001 else f"{t[3]:.3f}"
    return f"{t[0]:+6.1f} [{t[1]:+5.1f}, {t[2]:+5.1f}]  p={p}  n={t[4]}"


def cell(run, regime, cond, transform="none", defense="none", metric="attacker_send"):
    return run.get((regime, cond, transform, defense), {}).get(metric)


def main():
    pairs = [a.split("=", 1) for a in sys.argv[1:]]
    if not pairs:
        raise SystemExit(__doc__)
    attacks = load_attacks()
    runs = {lab: load_run(path, attacks) for lab, path in pairs}

    say("=" * 78)
    say("A. Main table: transform none, no structural check. Value [95% CI], %")
    say("=" * 78)
    for lab, run in runs.items():
        for reg in P.REGIMES:
            for cond in P.CONDITIONS:
                row = []
                for m in METRICS:
                    c = cell(run, reg, cond, metric=m)
                    row.append(f"{m.split('_')[0]:>8} {fmt(ci(list(c.values())))}" if c else f"{m.split('_')[0]:>8}  n/a")
                n = len(cell(run, reg, cond) or {})
                if n:
                    say(f"{lab:<17}{reg} {cond:<11} n={n:<5}" + "  ".join(row))

    say("")
    say("=" * 78)
    say("B. Paired differences within a model (later minus earlier), percentage points")
    say("=" * 78)
    for lab, run in runs.items():
        for cond in P.CONDITIONS:
            base = cell(run, "R0", cond)
            for reg in ("R1", "R2"):
                other = cell(run, reg, cond)
                if base and other:
                    say(f"{lab:<17}ASR {reg}-R0 ({cond:<10}) {fmtd(paired(base, other))}")
        for reg in P.REGIMES:
            a, b = cell(run, reg, "none"), cell(run, reg, "separation")
            if a and b:
                say(f"{lab:<17}ASR sep-none ({reg})        {fmtd(paired(a, b))}")
            a, b = (cell(run, reg, "none", metric="task_done"),
                    cell(run, reg, "separation", metric="task_done"))
            if a and b:
                say(f"{lab:<17}TASK sep-none ({reg})       {fmtd(paired(a, b))}")

    labs = list(runs)
    np_labs = [l for l in labs if "noparallel" in l]
    if np_labs:
        say("")
        say("=" * 78)
        say("C. Parallel tool calls disabled minus enabled (same model, paired on attack)")
        say("=" * 78)
        for nl in np_labs:
            base_lab = nl.replace("_noparallel", "")
            if base_lab not in runs:
                continue
            for reg in ("R1", "R2"):
                for cond in P.CONDITIONS:
                    for m, name in (("attacker_send", "ASR "), ("task_done", "TASK")):
                        a = cell(runs[base_lab], reg, cond, metric=m)
                        b = cell(runs[nl], reg, cond, metric=m)
                        if a and b:
                            say(f"{base_lab:<17}{name} off-on {reg} {cond:<10} {fmtd(paired(a, b))}")

    say("")
    say("=" * 78)
    say("D. Structural checks x adaptive variants, R2, no prompt defense: ASR [95% CI], task")
    say("=" * 78)
    for lab, run in runs.items():
        if not cell(run, "R2", "none", "spoof_drop"):
            continue
        say(f"--- {lab}")
        for d in DEFENSES:
            parts = []
            for t in P.TRANSFORMS:
                c = cell(run, "R2", "none", t, d)
                parts.append(f"{t}: {fmt(ci(list(c.values())))}" if c else f"{t}: n/a")
            task = cell(run, "R2", "none", "none", d, "task_done")
            parts.append(f"task: {fmt(ci(list(task.values())))}" if task else "")
            say(f"  {d:<14}" + " | ".join(parts))
        for cond in ("separation",):
            c = cell(run, "R2", cond, "spoof_drop", "dir_prov")
            if c:
                say(f"  dir_prov, spoof_drop, R2 + {cond}: {fmt(ci(list(c.values())))}")

    with open("stats_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")
    print("\nwrote stats_report.txt")


if __name__ == "__main__":
    main()
