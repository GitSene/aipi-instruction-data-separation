"""07_defense_family_rollup.py - aggregate Option A per-level results into
the by-defense-family comparison for the rebuttal.

Reads results/option_a_per_scenario_gpt-4o-mini.json and the population weights
from data/slice_2500_weights.json. Each level suffix maps to the defense the
model already had (phase-1 level map from the LLMail-Inject spec); each prefix
maps to the scenario. Reports population-weighted no_defense vs family_b ASR
and the delta per defense family and per scenario.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PER_SCENARIO = "results/option_a_per_scenario_claude_clean.json"
WEIGHTS = "data/slice_2500_weights.json"

# Phase-1 level suffix -> (defense family, model). From the challenge spec.
SUFFIX = {
    "a": ("PromptShield", "Phi-3"), "b": ("PromptShield", "GPT"),
    "c": ("TaskTracker", "Phi-3"),  "d": ("TaskTracker", "GPT"),
    "e": ("Spotlight", "Phi-3"),    "f": ("Spotlight", "GPT"),
    "g": ("LLM-judge", "Phi-3"),    "h": ("LLM-judge", "GPT"),
    "i": ("All-combined", "Phi-3"), "j": ("All-combined", "GPT"),
}
SCENARIO = {
    "level1": "S1 two-email no-retrieval",
    "level2": "S2 ten-email no-retrieval",
    "level3": "S3 retrieval",
    "level4": "S4 retrieval + exfil",
}


def split_level(level: str):
    prefix, suffix = level[:-1], level[-1]
    fam, model = SUFFIX.get(suffix, ("unknown", "unknown"))
    scen = SCENARIO.get(prefix, prefix)
    return fam, model, scen


def main() -> None:
    per = json.loads(Path(PER_SCENARIO).read_text(encoding="utf-8"))
    wdoc = json.loads(Path(WEIGHTS).read_text(encoding="utf-8"))
    weights = {lvl: v["weight"] for lvl, v in wdoc["weights"].items()}

    # Accumulators keyed by grouping. Each holds weighted sums and raw counts.
    def acc():
        return {"w": 0.0, "wn_nd": 0.0, "wn_fb": 0.0, "n": 0}

    by_family = defaultdict(acc)
    by_scenario = defaultdict(acc)
    by_family_scenario = defaultdict(acc)
    rows = []

    for level, conds in per.items():
        if "no_defense" not in conds or "family_b" not in conds:
            continue
        fam, model, scen = split_level(level)
        w = weights.get(level, 0.0)
        nd = conds["no_defense"]["ASR_A"]
        fb = conds["family_b"]["ASR_A"]
        n = min(conds["no_defense"]["n"], conds["family_b"]["n"])
        rows.append((level, scen, fam, model, n, nd, fb, nd - fb))
        for store, key in ((by_family, fam), (by_scenario, scen),
                           (by_family_scenario, (fam, scen))):
            s = store[key]
            s["w"] += w
            s["wn_nd"] += w * nd
            s["wn_fb"] += w * fb
            s["n"] += n

    def weighted(store):
        out = {}
        for key, s in store.items():
            if s["w"] == 0:
                continue
            nd = s["wn_nd"] / s["w"]
            fb = s["wn_fb"] / s["w"]
            out[key] = (nd, fb, nd - fb, s["n"])
        return out

    fam_w = weighted(by_family)
    scen_w = weighted(by_scenario)

    print("\n" + "=" * 70)
    print("BY DEFENSE FAMILY  (population-weighted ASR; delta = no_def - fam_b)")
    print("=" * 70)
    print(f"{'defense':<14}{'no_def':>9}{'family_b':>10}{'delta':>9}{'rel.red':>9}{'n':>7}")
    order = ["PromptShield", "TaskTracker", "Spotlight", "LLM-judge", "All-combined"]
    for fam in order:
        if fam not in fam_w:
            continue
        nd, fb, d, n = fam_w[fam]
        rel = (d / nd * 100) if nd else 0.0
        print(f"{fam:<14}{nd*100:8.1f}%{fb*100:9.1f}%{d*100:8.1f}%{rel:8.0f}%{n:7d}")

    print("\n" + "=" * 70)
    print("BY SCENARIO  (population-weighted)")
    print("=" * 70)
    print(f"{'scenario':<28}{'no_def':>9}{'family_b':>10}{'delta':>9}{'n':>7}")
    for scen in ["S1 two-email no-retrieval", "S2 ten-email no-retrieval",
                 "S3 retrieval", "S4 retrieval + exfil"]:
        if scen not in scen_w:
            continue
        nd, fb, d, n = scen_w[scen]
        print(f"{scen:<28}{nd*100:8.1f}%{fb*100:9.1f}%{d*100:8.1f}%{n:7d}")

    print("\n" + "=" * 70)
    print("PER-LEVEL DETAIL  (* = n<20, treat as directional only)")
    print("=" * 70)
    print(f"{'level':<9}{'scenario':<26}{'defense':<14}{'mdl':<6}{'n':>5}{'no_def':>9}{'fam_b':>8}{'delta':>8}")
    for level, scen, fam, model, n, nd, fb, d in sorted(rows):
        flag = "*" if n < 20 else " "
        print(f"{level:<9}{scen[:25]:<26}{fam:<14}{model:<6}{n:>4}{flag}{nd*100:8.1f}%{fb*100:7.1f}%{d*100:7.1f}%")

    Path("results").mkdir(exist_ok=True)
    out = {
        "by_defense_family": {k: {"no_defense": v[0], "family_b": v[1],
                                  "delta": v[2], "n": v[3]} for k, v in fam_w.items()},
        "by_scenario": {k: {"no_defense": v[0], "family_b": v[1],
                            "delta": v[2], "n": v[3]} for k, v in scen_w.items()},
    }
    Path("results/defense_family_rollup.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote results/defense_family_rollup.json")


if __name__ == "__main__":
    main()