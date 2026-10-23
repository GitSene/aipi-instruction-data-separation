"""07b_rollup.py - defense-family rollup for ANY model.

Same logic as 07_defense_family_rollup.py, but takes the model on the command
line so you do not hand-edit paths. Reads the per-scenario JSON for that model
and the shared population weights, prints the by-family / by-scenario tables,
and writes a model-specific rollup JSON.

Usage:
    python 07b_rollup.py --in results/option_a_per_scenario_meta-llama_Llama-3.3-70B-Instruct-Turbo.json --tag llama
    python 07b_rollup.py --in results/option_a_per_scenario_Qwen_Qwen2.5-7B-Instruct-Turbo.json --tag qwen
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

WEIGHTS = "data/slice_2500_weights.json"

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
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True,
                   help="per-scenario JSON for the model")
    p.add_argument("--tag", required=True,
                   help="short label for the output file, e.g. llama or qwen")
    a = p.parse_args()

    per = json.loads(Path(a.infile).read_text(encoding="utf-8"))
    wdoc = json.loads(Path(WEIGHTS).read_text(encoding="utf-8"))
    weights = {lvl: v["weight"] for lvl, v in wdoc["weights"].items()}

    def acc():
        return {"w": 0.0, "wn_nd": 0.0, "wn_fb": 0.0, "n": 0}

    by_family = defaultdict(acc)
    by_scenario = defaultdict(acc)
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
        for store, key in ((by_family, fam), (by_scenario, scen)):
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
    print(f"MODEL TAG: {a.tag}")
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

    # Print figure-ready arrays so you can paste straight into make_figure.py
    order_fig = ["PromptShield", "TaskTracker", "Spotlight", "LLM-judge", "All-combined"]
    nodef_arr = [round(fam_w[f][0] * 100, 1) if f in fam_w else 0.0 for f in order_fig]
    sep_arr = [round(fam_w[f][1] * 100, 1) if f in fam_w else 0.0 for f in order_fig]
    print("\n--- paste into make_figure.py ---")
    print(f"{a.tag}_nodef = {nodef_arr}")
    print(f"{a.tag}_sep   = {sep_arr}")

    Path("results").mkdir(exist_ok=True)
    out = {
        "by_defense_family": {k: {"no_defense": v[0], "family_b": v[1],
                                  "delta": v[2], "n": v[3]} for k, v in fam_w.items()},
        "by_scenario": {k: {"no_defense": v[0], "family_b": v[1],
                            "delta": v[2], "n": v[3]} for k, v in scen_w.items()},
    }
    outpath = f"results/defense_family_rollup_{a.tag}.json"
    Path(outpath).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()