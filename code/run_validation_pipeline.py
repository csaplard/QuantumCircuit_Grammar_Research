import argparse
import csv
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from grammar_learner import extract_grammar, train_model

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "results", "readout_raw_data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CRITERIA_PATH = os.path.join(REPO_ROOT, "validation_criteria.json")
ORIGINAL_AUDIT_PATH = os.path.join(RESULTS_DIR, "robustness_audit_sax7.txt")
SHUFFLED_OUT_PATH = os.path.join(RESULTS_DIR, "shuffled_control_sax7.txt")
SWEEP_OUT_PATH = os.path.join(RESULTS_DIR, "parameter_sweep_sax7.csv")
REPORT_OUT_PATH = os.path.join(RESULTS_DIR, "validation_report.txt")

GROUND_TRUTH = {
    "14q": "1D_Snake",
    "18q": "1D_Snake",
    "22q": "1D_Snake",
    "28q": "1D_Snake",
    "32q": "1D_Snake",
    "47q": "1D_Snake",
    "49q": "1D_Snake",
    "12q": "2D_Block",
    "16q": "2D_Block",
    "20q": "2D_Block",
    "24q": "2D_Block",
    "30q": "2D_Block",
    "34q": "2D_Block",
    "39q": "2D_Block",
    "40q": "2D_Block",
    "41q": "2D_Block",
    "42q": "2D_Block",
    "43q": "2D_Block",
    "44q": "2D_Block",
    "45q": "2D_Block",
    "50q": "2D_Block",
    "26q": "Bulk_Full",
    "36q": "Bulk_Full",
    "38q": "Bulk_Full",
    "46q": "Bulk_Full",
    "48q": "Bulk_Full",
    "51q": "Bulk_Full",
    "53q": "Bulk_Full",
}


def frobenius_distance(m1: np.ndarray, m2: np.ndarray) -> float:
    return float(np.sqrt(np.sum((m1 - m2) ** 2)))


def list_data_files(data_dir: str | None = None) -> List[str]:
    d = data_dir if data_dir is not None else DATA_DIR
    files = [f for f in os.listdir(d) if f.endswith(".txt")]
    files.sort(key=lambda x: int(x.split("q")[0]))
    return files


def run_single_validation(
    *,
    seed: int,
    alphabet_size: int,
    epochs: int,
    max_pts: int,
    shuffle: bool,
    data_dir: str | None = None,
    ground_truth: Dict[str, str] | None = None,
    output_bit_index: int | None = None,
) -> Tuple[float, List[float], List[str]]:
    """
    If output_bit_index is set (0 = LSB), each timestep uses only that bit of the integer
    readout: (int(output) >> output_bit_index) & 1 → 0/1 time series. Otherwise the full
    integer is cast to float (existing behaviour).
    """
    bit_note = f" output_bit={output_bit_index}(LSB)" if output_bit_index is not None else ""
    print(
        f"\n>>> seed={seed} alphabet={alphabet_size} epochs={epochs} max_pts={max_pts}"
        f" shuffle={shuffle}{bit_note}",
        flush=True,
    )
    ddir = data_dir if data_dir is not None else DATA_DIR
    gt = ground_truth if ground_truth is not None else GROUND_TRUTH
    files = list_data_files(ddir)
    fingerprints = []
    labels = []

    params = {"hidden_dim": 32, "seq_len": 16, "epochs": epochs, "lr": 0.01}
    rng = np.random.default_rng(seed) if shuffle else None

    for f in files:
        label = f.split("_")[0]
        f_path = os.path.join(ddir, f)
        df = pd.read_csv(f_path, sep=" ", on_bad_lines="skip")
        raw = df["output"].values
        if len(raw) > max_pts:
            raw = raw[:max_pts]
        if output_bit_index is not None:
            k = int(output_bit_index)
            signal = np.array([((int(v) >> k) & 1) for v in raw], dtype=np.float64)
        else:
            signal = raw.astype(np.float64)
        if shuffle:
            signal = rng.permutation(signal)

        _, _, model, val_data = train_model(
            signal,
            label,
            alphabet_size=alphabet_size,
            data_is_array=True,
            seed=seed,
            **params,
        )
        if model is not None:
            fp = extract_grammar(model, val_data, seq_len=params["seq_len"])
            fingerprints.append(fp)
            labels.append(label)

    n_f = len(fingerprints)
    dist_matrix = np.zeros((n_f, n_f), dtype=np.float64)
    for i in range(n_f):
        for j in range(n_f):
            dist_matrix[i, j] = frobenius_distance(fingerprints[i], fingerprints[j])

    Z = linkage(squareform(dist_matrix), method="ward")
    clusters = fcluster(Z, t=3, criterion="maxclust")
    results = pd.DataFrame(
        {
            "q_label": labels,
            "cluster": clusters,
            "true_regime": [gt[q] for q in labels],
        }
    )

    correct = 0
    cluster_purities = []
    cluster_lines = []
    for cluster_id in range(1, 4):
        cluster_data = results[results["cluster"] == cluster_id]
        if cluster_data.empty:
            cluster_purities.append(0.0)
            cluster_lines.append(f"Cluster {cluster_id}: EMPTY (Purity: 0.0%)")
            continue
        majority_regime = cluster_data["true_regime"].value_counts().idxmax()
        majority_count = cluster_data["true_regime"].value_counts().max()
        purity = float((majority_count / len(cluster_data)) * 100)
        correct += int(majority_count)
        cluster_purities.append(purity)
        cluster_lines.append(f"Cluster {cluster_id}: {majority_regime} (Purity: {purity:.1f}%)")

    accuracy = float((correct / n_f) * 100)
    return accuracy, cluster_purities, cluster_lines


def run_shuffled_control() -> Dict[int, Tuple[float, List[str]]]:
    seeds = [0, 123, 999]
    out = {}
    for seed in seeds:
        acc, _, cluster_lines = run_single_validation(
            seed=seed, alphabet_size=7, epochs=50, max_pts=10000, shuffle=True
        )
        out[seed] = (acc, cluster_lines)
        print(f"--- SHUFFLED seed={seed} accuracy={acc:.2f}% ---", flush=True)

    with open(SHUFFLED_OUT_PATH, "w", encoding="utf-8") as f:
        f.write("=== SHUFFLED CONTROL TEST (SAX=7) ===\n")
        f.write("Pipeline: same as original (epochs=50, max_pts=10000, hidden_dim=32, seq_len=16, lr=0.01)\n")
        f.write("Only difference: input sequence shuffled per file.\n\n")
        f.write("Summary:\n")
        for seed in [0, 123, 999]:
            f.write(f"  Seed {seed}: {out[seed][0]:.2f}%\n")
        mean_acc = np.mean([v[0] for v in out.values()])
        std_acc = np.std([v[0] for v in out.values()])
        f.write(f"\nMean accuracy: {mean_acc:.2f}%\n")
        f.write(f"Std accuracy: {std_acc:.2f}\n\n")
        for seed in [0, 123, 999]:
            f.write(f"Seed {seed}:\n")
            for line in out[seed][1]:
                f.write(f"  {line}\n")
            f.write("\n")
    print(f"Saved shuffled control results: {SHUFFLED_OUT_PATH}", flush=True)
    return out


def load_existing_sweep_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    keys = set()
    for _, r in df.iterrows():
        keys.add((int(r["epochs"]), int(r["max_pts"]), int(r["seed"])))
    return keys


def run_parameter_sweep() -> None:
    epochs_list = [10, 20, 30, 40, 50]
    max_pts_list = [3000, 5000, 8000, 10000]
    seeds = [0, 123, 999]

    header = [
        "epochs",
        "max_pts",
        "seed",
        "accuracy",
        "cluster1_purity",
        "cluster2_purity",
        "cluster3_purity",
    ]
    existing = load_existing_sweep_keys(SWEEP_OUT_PATH)
    write_header = not os.path.exists(SWEEP_OUT_PATH)
    with open(SWEEP_OUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for epochs in epochs_list:
            for max_pts in max_pts_list:
                for seed in seeds:
                    key = (epochs, max_pts, seed)
                    if key in existing:
                        print(f"Skipping existing row: epochs={epochs}, max_pts={max_pts}, seed={seed}", flush=True)
                        continue
                    acc, purities, _ = run_single_validation(
                        seed=seed,
                        alphabet_size=7,
                        epochs=epochs,
                        max_pts=max_pts,
                        shuffle=False,
                    )
                    writer.writerow(
                        [
                            epochs,
                            max_pts,
                            seed,
                            f"{acc:.6f}",
                            f"{purities[0]:.6f}",
                            f"{purities[1]:.6f}",
                            f"{purities[2]:.6f}",
                        ]
                    )
                    f.flush()
                    print(
                        f"Saved sweep row: epochs={epochs}, max_pts={max_pts}, seed={seed}, acc={acc:.2f}%",
                        flush=True,
                    )
    print(f"Saved parameter sweep: {SWEEP_OUT_PATH}", flush=True)


def run_parameter_sweep_watchdog(timeout_sec: int) -> None:
    """
    Sweep runner with per-combination timeout watchdog.
    Each (epochs, max_pts, seed) combo runs in a separate subprocess via `--tasks sweep_one`.
    If one combo hangs, only that combo is marked as NaN and the sweep continues.
    """
    epochs_list = [10, 20, 30, 40, 50]
    max_pts_list = [3000, 5000, 8000, 10000]
    seeds = [0, 123, 999]

    header = [
        "epochs",
        "max_pts",
        "seed",
        "accuracy",
        "cluster1_purity",
        "cluster2_purity",
        "cluster3_purity",
    ]
    existing = load_existing_sweep_keys(SWEEP_OUT_PATH)
    write_header = not os.path.exists(SWEEP_OUT_PATH)
    with open(SWEEP_OUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for epochs in epochs_list:
            for max_pts in max_pts_list:
                for seed in seeds:
                    key = (epochs, max_pts, seed)
                    if key in existing:
                        print(f"Skipping existing row: epochs={epochs}, max_pts={max_pts}, seed={seed}", flush=True)
                        continue

                    cmd = [
                        sys.executable,
                        os.path.abspath(__file__),
                        "--tasks",
                        "sweep_one",
                        "--sweep-epochs",
                        str(epochs),
                        "--sweep-max-pts",
                        str(max_pts),
                        "--sweep-seed",
                        str(seed),
                    ]
                    try:
                        proc = subprocess.run(
                            cmd,
                            cwd=REPO_ROOT,
                            capture_output=True,
                            text=True,
                            timeout=timeout_sec,
                            check=False,
                        )
                    except subprocess.TimeoutExpired:
                        print(
                            f"Timeout: epochs={epochs}, max_pts={max_pts}, seed={seed} exceeded {timeout_sec}s",
                            flush=True,
                        )
                        writer.writerow([epochs, max_pts, seed, "nan", "nan", "nan", "nan"])
                        f.flush()
                        continue

                    if proc.returncode != 0:
                        print(
                            f"Error combo epochs={epochs}, max_pts={max_pts}, seed={seed}, code={proc.returncode}",
                            flush=True,
                        )
                        print(proc.stdout[-5000:], flush=True)
                        print(proc.stderr[-5000:], flush=True)
                        writer.writerow([epochs, max_pts, seed, "nan", "nan", "nan", "nan"])
                        f.flush()
                        continue

                    # Parse last RESULT line from child stdout
                    result_line = None
                    for line in reversed(proc.stdout.splitlines()):
                        if line.startswith("RESULT,"):
                            result_line = line
                            break
                    if result_line is None:
                        print(
                            f"Missing RESULT line for epochs={epochs}, max_pts={max_pts}, seed={seed}",
                            flush=True,
                        )
                        writer.writerow([epochs, max_pts, seed, "nan", "nan", "nan", "nan"])
                        f.flush()
                        continue

                    _, acc_s, p1_s, p2_s, p3_s = result_line.split(",")
                    writer.writerow([epochs, max_pts, seed, acc_s, p1_s, p2_s, p3_s])
                    f.flush()
                    print(
                        f"Saved sweep row: epochs={epochs}, max_pts={max_pts}, seed={seed}, acc={float(acc_s):.2f}%",
                        flush=True,
                    )
    print(f"Saved parameter sweep (watchdog): {SWEEP_OUT_PATH}", flush=True)


def parse_original_accuracies(path: str) -> List[float]:
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    vals = re.findall(r"^\s*(?:0|123|999)\s+([0-9]+\.[0-9]+)%\s*$", txt, flags=re.MULTILINE)
    return [float(v) for v in vals]


def parse_shuffled_accuracies(path: str) -> List[float]:
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    vals = re.findall(r"Seed\s+(?:0|123|999):\s+([0-9]+\.[0-9]+)%", txt)
    return [float(v) for v in vals]


def generate_report() -> None:
    with open(CRITERIA_PATH, encoding="utf-8") as f:
        criteria = json.load(f)

    orig_acc = parse_original_accuracies(ORIGINAL_AUDIT_PATH)
    shuf_acc = parse_shuffled_accuracies(SHUFFLED_OUT_PATH)
    sweep_df = pd.read_csv(SWEEP_OUT_PATH)

    original_mean = float(np.mean(orig_acc))
    original_std = float(np.std(orig_acc))
    shuffled_mean = float(np.mean(shuf_acc))
    diff = original_mean - shuffled_mean

    checks = {
        "minimum_accuracy_original": original_mean >= float(criteria["minimum_accuracy_original"]),
        "maximum_seed_std": original_std <= float(criteria["maximum_seed_std"]),
        "maximum_shuffled_accuracy": shuffled_mean <= float(criteria["maximum_shuffled_accuracy"]),
        "minimum_accuracy_difference_vs_shuffled": diff >= float(criteria["minimum_accuracy_difference_vs_shuffled"]),
    }
    verdict = "SIGNAL IS REAL" if all(checks.values()) else "SIGNAL IS ARTIFACT"

    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        f.write("=== VALIDATION REPORT ===\n\n")
        f.write("Loaded decision criteria from validation_criteria.json\n")
        f.write(json.dumps(criteria, indent=2))
        f.write("\n\nOriginal robustness (SAX=7):\n")
        f.write(f"  Accuracies: {orig_acc}\n")
        f.write(f"  Mean accuracy: {original_mean:.2f}%\n")
        f.write(f"  Std across seeds: {original_std:.2f}\n\n")
        f.write("Shuffled control (SAX=7, same pipeline):\n")
        f.write(f"  Accuracies: {shuf_acc}\n")
        f.write(f"  Mean accuracy: {shuffled_mean:.2f}%\n\n")
        f.write("Parameter sweep summary:\n")
        f.write(f"  Rows: {len(sweep_df)}\n")
        grouped = (
            sweep_df.groupby(["epochs", "max_pts"])["accuracy"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(["epochs", "max_pts"])
        )
        for _, row in grouped.iterrows():
            f.write(
                f"  epochs={int(row['epochs'])}, max_pts={int(row['max_pts'])}: "
                f"mean={row['mean']:.2f}%, std={row['std']:.2f}\n"
            )
        f.write("\nCriteria evaluation:\n")
        f.write(
            f"  minimum_accuracy_original ({criteria['minimum_accuracy_original']}): "
            f"{'PASS' if checks['minimum_accuracy_original'] else 'FAIL'}\n"
        )
        f.write(
            f"  maximum_seed_std ({criteria['maximum_seed_std']}): "
            f"{'PASS' if checks['maximum_seed_std'] else 'FAIL'}\n"
        )
        f.write(
            f"  maximum_shuffled_accuracy ({criteria['maximum_shuffled_accuracy']}): "
            f"{'PASS' if checks['maximum_shuffled_accuracy'] else 'FAIL'}\n"
        )
        f.write(
            f"  minimum_accuracy_difference_vs_shuffled ({criteria['minimum_accuracy_difference_vs_shuffled']}): "
            f"{'PASS' if checks['minimum_accuracy_difference_vs_shuffled'] else 'FAIL'}\n"
        )
        f.write(f"\nAccuracy difference (original - shuffled): {diff:.2f}%\n")
        f.write(f"\nVERDICT: {verdict}\n")
    print(f"Saved validation report: {REPORT_OUT_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated tasks: shuffled,sweep,report or all",
    )
    parser.add_argument(
        "--watchdog-timeout-sec",
        type=int,
        default=7200,
        help="Per-combination timeout for sweep watchdog mode (seconds).",
    )
    parser.add_argument("--sweep-epochs", type=int, default=None)
    parser.add_argument("--sweep-max-pts", type=int, default=None)
    parser.add_argument("--sweep-seed", type=int, default=None)
    args = parser.parse_args()
    if args.tasks == "all":
        tasks = ["shuffled", "sweep", "report"]
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if "shuffled" in tasks:
        run_shuffled_control()
    if "sweep" in tasks:
        run_parameter_sweep_watchdog(timeout_sec=args.watchdog_timeout_sec)
    if "sweep_one" in tasks:
        if args.sweep_epochs is None or args.sweep_max_pts is None or args.sweep_seed is None:
            raise ValueError("sweep_one requires --sweep-epochs, --sweep-max-pts, --sweep-seed")
        acc, purities, _ = run_single_validation(
            seed=args.sweep_seed,
            alphabet_size=7,
            epochs=args.sweep_epochs,
            max_pts=args.sweep_max_pts,
            shuffle=False,
        )
        print(f"RESULT,{acc:.6f},{purities[0]:.6f},{purities[1]:.6f},{purities[2]:.6f}", flush=True)
    if "report" in tasks:
        generate_report()


if __name__ == "__main__":
    main()

