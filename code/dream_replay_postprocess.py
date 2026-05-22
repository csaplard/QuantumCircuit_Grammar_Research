#!/usr/bin/env python3
"""
dream_replay_postprocess.py
============================
Aggregates and plots results from dream_replay_poc.py after the crash
at the very end (cohens_d_std missing from fieldnames).

All per-seed CSVs are already saved. This script:
  1. Loads per-seed layer_sep CSVs for T2D and T2C
  2. Aggregates (mean + std of Cohen's d across seeds)
  3. Saves T2D_layer_sep_mean.csv, T2C_layer_sep_mean.csv
  4. Loads T0/T1 metrics and all participation/path_speed from metrics JSONs
  5. Generates summary.png
  6. Prints final diagnostic table
"""

import csv, json, os
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results/dream_replay")
BOOTSTRAP_K = 5
REGIMES     = ["factual", "mathematical", "creative", "philosophical"]


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_layer_sep(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "layer":     int(row["layer"]),
                "regime_a":  row["regime_a"],
                "regime_b":  row["regime_b"],
                "cohens_d":  float(row["cohens_d"]),
            })
    return rows


def aggregate_sep(sep_list: list[list[dict]]) -> list[dict]:
    combined = defaultdict(list)
    for sep in sep_list:
        for row in sep:
            key = (row["layer"], row["regime_a"], row["regime_b"])
            combined[key].append(row["cohens_d"])
    return [
        {
            "layer":        k[0],
            "regime_a":     k[1],
            "regime_b":     k[2],
            "cohens_d":     round(float(np.mean(vs)), 4),
            "cohens_d_std": round(float(np.std(vs)), 4),
        }
        for k, vs in sorted(combined.items())
    ]


def save_layer_sep(rows: list[dict], path: Path):
    fields = ["layer", "regime_a", "regime_b", "cohens_d", "cohens_d_std"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[save] {path}", flush=True)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def mean_d_zone(rows, n_layers, zone):
    lo = {"early": 0,           "mid": n_layers//3,       "late": 2*n_layers//3}[zone]
    hi = {"early": n_layers//3, "mid": 2*n_layers//3,     "late": n_layers}[zone]
    v  = [r["cohens_d"] for r in rows if lo <= r["layer"] < hi]
    return round(float(np.mean(v)), 3) if v else 0.0


# ─── aggregate ────────────────────────────────────────────────────────────────

print("[load] T0 layer sep ...", flush=True)
t0_sep = load_layer_sep(RESULTS_DIR / "T0_layer_sep.csv")
t1_sep = load_layer_sep(RESULTS_DIR / "T1_layer_sep.csv")
n_layers = max(r["layer"] for r in t0_sep) + 1
print(f"[info] n_layers={n_layers}", flush=True)

print("[load] T2D per-seed ...", flush=True)
t2d_seps = []
for s in range(BOOTSTRAP_K):
    p = RESULTS_DIR / f"T2D_layer_sep_s{s}.csv"
    if p.exists():
        t2d_seps.append(load_layer_sep(p))
    else:
        print(f"  [warn] missing {p}", flush=True)

print("[load] T2C per-seed ...", flush=True)
t2c_seps = []
for s in range(BOOTSTRAP_K):
    p = RESULTS_DIR / f"T2C_layer_sep_s{s}.csv"
    if p.exists():
        t2c_seps.append(load_layer_sep(p))
    else:
        print(f"  [warn] missing {p}", flush=True)

t2d_mean = aggregate_sep(t2d_seps)
t2c_mean = aggregate_sep(t2c_seps)
save_layer_sep(t2d_mean, RESULTS_DIR / "T2D_layer_sep_mean.csv")
save_layer_sep(t2c_mean, RESULTS_DIR / "T2C_layer_sep_mean.csv")

# ─── load scalar metrics ───────────────────────────────────────────────────────

print("[load] metrics JSONs ...", flush=True)
t0_m  = load_json(RESULTS_DIR / "T0_metrics.json")
t1_m  = load_json(RESULTS_DIR / "T1_metrics.json")
t2d_m = load_json(RESULTS_DIR / "T2D_metrics.json")  if (RESULTS_DIR / "T2D_metrics.json").exists() else {}
t2c_m = load_json(RESULTS_DIR / "T2C_metrics.json")  if (RESULTS_DIR / "T2C_metrics.json").exists() else {}
probe = load_json(RESULTS_DIR / "probe_transfer.json") if (RESULTS_DIR / "probe_transfer.json").exists() else {}

# patch layer_sep into scalar metric dicts for plotting
t0_m["layer_sep"]  = t0_sep
t1_m["layer_sep"]  = t1_sep
t2d_m["layer_sep"] = t2d_mean
t2c_m["layer_sep"] = t2c_mean
all_results = {"T0": t0_m, "T1": t1_m, "T2D": t2d_m, "T2C": t2c_m,
               "probe_transfer": probe}

# ─── diagnostic summary table ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

cond_sep_map = {"T0": t0_sep, "T1": t1_sep, "T2D": t2d_mean, "T2C": t2c_mean}
for cond, sep in cond_sep_map.items():
    m = all_results[cond]
    pr = m.get("participation", {})
    ps = m.get("path_speed", {})
    print(f"\n  {cond}:")
    print(f"    Cohen's d  early={mean_d_zone(sep, n_layers, 'early')}  "
          f"mid={mean_d_zone(sep, n_layers, 'mid')}  "
          f"late={mean_d_zone(sep, n_layers, 'late')}")
    print(f"    Participation ratio : {pr}")
    print(f"    Path speed (mean F) : {ps}")

print("\n  Probe transfer (T0-trained linear probe):")
for k, v in probe.items():
    print(f"    {k}: {v:.2%}")

# ─── key comparison: T2D vs T2C ───────────────────────────────────────────────

print("\n  T2D vs T2C comparison (mid-layer zone, averaged across pairs):")
mid_d   = mean_d_zone(t2d_mean, n_layers, "mid")
mid_c   = mean_d_zone(t2c_mean, n_layers, "mid")
delta   = round(mid_d - mid_c, 3)
print(f"    T2D mid Cohen's d = {mid_d}  |  T2C mid Cohen's d = {mid_c}  |  delta = {delta}")
print(f"    -> {'T2D > T2C (structured dropout preserves more regime geometry)' if delta > 0.05 else 'T2D ~ T2C (structure does not matter much)' if abs(delta) <= 0.05 else 'T2C > T2D (random dropout preserves more)'}")

t2d_probe_mean = np.mean([v for k, v in probe.items() if k.startswith("T2D")])
t2c_probe_mean = np.mean([v for k, v in probe.items() if k.startswith("T2C")])
print(f"\n    T2D probe accuracy (mean) = {t2d_probe_mean:.2%}")
print(f"    T2C probe accuracy (mean) = {t2c_probe_mean:.2%}")
print(f"    T0 train accuracy         = {probe.get('T0_train', 0):.2%}")

# ─── plot ─────────────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dream Replay — Fisher Geometry Analysis", fontsize=14, fontweight="bold")

    colors = {"T0": "black", "T1": "navy", "T2D": "steelblue", "T2C": "tomato"}
    lss    = {"T0": "-",    "T1": "--",   "T2D": "-",          "T2C": ":"}
    cond_order = ["T0", "T1", "T2D", "T2C"]

    # Panel 1: Cohen's d per layer
    ax = axes[0, 0]
    sep_map_plot = {"T0": t0_sep, "T1": t1_sep, "T2D": t2d_mean, "T2C": t2c_mean}
    for cond in cond_order:
        rows = sep_map_plot[cond]
        layers_idx = sorted(set(r["layer"] for r in rows))
        mean_d_vals = [np.mean([r["cohens_d"] for r in rows if r["layer"] == li])
                       for li in layers_idx]
        ax.plot(layers_idx, mean_d_vals, label=cond,
                color=colors[cond], ls=lss[cond], lw=2)
        # Bootstrap std band for T2D/T2C
        if cond in ("T2D", "T2C"):
            std_vals = [np.mean([r.get("cohens_d_std", 0) for r in rows if r["layer"] == li])
                        for li in layers_idx]
            ax.fill_between(layers_idx,
                            np.array(mean_d_vals) - np.array(std_vals),
                            np.array(mean_d_vals) + np.array(std_vals),
                            alpha=0.15, color=colors[cond])
    ax.axvspan(0,             n_layers//3,       alpha=0.06, color="green")
    ax.axvspan(n_layers//3,   2*n_layers//3,     alpha=0.06, color="orange")
    ax.axvspan(2*n_layers//3, n_layers,           alpha=0.06, color="purple")
    ax.text(n_layers//6,            ax.get_ylim()[0], "early", fontsize=8, color="green")
    ax.text(n_layers//2,            ax.get_ylim()[0], "mid",   fontsize=8, color="darkorange")
    ax.text(5*n_layers//6,          ax.get_ylim()[0], "late",  fontsize=8, color="purple")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean Cohen's d (across pairs)")
    ax.set_title("Regime separation per layer")
    ax.legend(fontsize=9)

    # Panel 2: Participation ratio
    ax = axes[0, 1]
    x = np.arange(len(REGIMES))
    bar_w = 0.8 / len(cond_order)
    for ci, cond in enumerate(cond_order):
        pr = all_results[cond].get("participation", {})
        vals = [pr.get(r, 0) for r in REGIMES]
        ax.bar(x + ci * bar_w, vals, bar_w, label=cond,
               color=colors[cond], alpha=0.75)
    ax.set_xticks(x + bar_w * len(cond_order) / 2)
    ax.set_xticklabels(REGIMES, rotation=15, fontsize=9)
    ax.set_ylabel("Participation ratio")
    ax.set_title("Temporal participation ratio\n(high = signal diffuse over time)")
    ax.legend(fontsize=9)

    # Panel 3: Fisher path speed
    ax = axes[1, 0]
    for ci, cond in enumerate(cond_order):
        ps = all_results[cond].get("path_speed", {})
        vals = [ps.get(r, 0) for r in REGIMES]
        ax.bar(x + ci * bar_w, vals, bar_w, label=cond,
               color=colors[cond], alpha=0.75)
    ax.set_xticks(x + bar_w * len(cond_order) / 2)
    ax.set_xticklabels(REGIMES, rotation=15, fontsize=9)
    ax.set_ylabel("Mean F(t)")
    ax.set_title("Fisher path speed per regime")
    ax.legend(fontsize=9)

    # Panel 4: Probe transfer
    ax = axes[1, 1]
    if probe:
        keys  = list(probe.keys())
        vals  = [probe[k] for k in keys]
        bcols = ["black" if k == "T0_train" else
                 colors["T2D"] if k.startswith("T2D") else
                 colors["T2C"] for k in keys]
        ax.bar(range(len(keys)), vals, color=bcols, alpha=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, fontsize=8)
        ax.axhline(1/len(REGIMES), color="gray", ls="--",
                   label=f"chance ({1/len(REGIMES):.0%})")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title("Probe transfer: T0-trained, T2D/T2C-tested\n"
                     "(blue=T2D structured, red=T2C random)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "summary.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n[plot] -> {out}", flush=True)

except Exception as exc:
    print(f"[plot] failed: {exc}", flush=True)

print("\n[done]", flush=True)
