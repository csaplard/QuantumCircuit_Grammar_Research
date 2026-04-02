import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
SHOTS = 8192
REPS = 5
OUTPUT_DIR = "results/ibm_raw_shots"

PREFERRED_BACKENDS = [
    "ibm_brisbane",
    "ibm_osaka",
    "ibm_kyoto",
    "ibm_sherbrooke",
]


# ============================================================
# CIRCUITS
# ============================================================
def make_circuits():
    """Create diverse test circuits for Grammar Fingerprinting."""
    from qiskit import QuantumCircuit

    circuits = {}

    for n_qubits in [1, 5, 10, 20]:
        qc = QuantumCircuit(n_qubits, n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        qc.measure(range(n_qubits), range(n_qubits))
        circuits[f"hadamard_{n_qubits}q"] = qc

    for n_qubits in [5, 10, 20]:
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cx(0, i)
        qc.measure(range(n_qubits), range(n_qubits))
        circuits[f"ghz_{n_qubits}q"] = qc

    for n_qubits in [10, 20]:
        qc = QuantumCircuit(n_qubits, n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(n_qubits):
            qc.rz(0.7 * i, i)
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        qc.measure(range(n_qubits), range(n_qubits))
        circuits[f"layers_{n_qubits}q"] = qc

    for n_qubits in [10, 20]:
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.measure(range(n_qubits), range(n_qubits))
        circuits[f"identity_{n_qubits}q"] = qc

    return circuits


# ============================================================
# COLLECTION
# ============================================================
def bitstring_to_int(bitstring: str) -> int:
    """Convert Qiskit bitstring (LSB order) to integer."""
    return int(bitstring, 2)


def _count_shot_lines(filepath: str) -> int:
    """Number of data rows (excluding header) in Sycamore-style shot file."""
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return 0
    if lines[0].strip().lower().startswith("input"):
        return max(0, len(lines) - 1)
    return len(lines)


def collect_shots(
    backend_name: str | None = None,
    shots: int | None = None,
    reps: int | None = None,
    resume: bool = False,
):
    """Run circuits on IBM Quantum and save raw shot sequences."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    n_shots = shots if shots is not None else SHOTS
    n_reps = reps if reps is not None else REPS

    # Newer qiskit-ibm-runtime uses `ibm_quantum_platform` channel name.
    service = QiskitRuntimeService(channel="ibm_quantum_platform")

    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=20,
        )

    actual_backend = backend.name
    print(f"Using backend: {actual_backend}")
    print(f"Qubits: {backend.num_qubits}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    circuits = make_circuits()

    expected_total = n_shots * n_reps

    for circuit_name, qc in circuits.items():
        n_qubits = qc.num_qubits
        if n_qubits > backend.num_qubits:
            print(
                f"Skipping {circuit_name}: needs {n_qubits} qubits, backend has {backend.num_qubits}"
            )
            continue

        out_name = f"{actual_backend}_{circuit_name}_{expected_total}shots.txt"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        print(f"\n--- {circuit_name} ({n_qubits} qubits) ---")
        if resume and os.path.isfile(out_path):
            n_have = _count_shot_lines(out_path)
            if n_have == expected_total:
                print(f"  SKIP (--resume): {out_name} already has {n_have} shots.")
                continue
            if n_have > 0:
                print(
                    f"  RESUME: partial file ({n_have}/{expected_total} rows) — re-running full circuit."
                )

        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        transpiled = pm.run(qc)

        all_shots = []

        for rep in range(n_reps):
            print(f"  Rep {rep + 1}/{n_reps}...", end=" ", flush=True)

            sampler = SamplerV2(mode=backend)
            job = sampler.run([transpiled], shots=n_shots)
            result = job.result()

            pub_result = result[0]
            data = pub_result.data
            creg_name = list(data.__dict__.keys())[0]
            bit_array = getattr(data, creg_name)
            bitstrings = bit_array.get_bitstrings()

            print(f"got {len(bitstrings)} shots")
            all_shots.extend(bitstrings)

        input_val = 0

        filename = f"{actual_backend}_{circuit_name}_{expected_total}shots.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("input output\n")
            for bs in all_shots:
                output_val = bitstring_to_int(bs)
                f.write(f"{input_val} {output_val}\n")

        print(f"  Saved: {filepath} ({len(all_shots)} shots)")

    meta_path = os.path.join(OUTPUT_DIR, f"{actual_backend}_metadata.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"Backend: {actual_backend}\n")
        f.write(f"Qubits: {backend.num_qubits}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Shots per rep: {n_shots}\n")
        f.write(f"Reps: {n_reps}\n")
        f.write(f"Total shots per circuit: {expected_total}\n")
        f.write(f"Circuits: {list(circuits.keys())}\n")

    print(f"\nDone! All files saved to {OUTPUT_DIR}/")
    print(f"Metadata: {meta_path}")


def quick_validate():
    """Run Grammar Fingerprinting on collected IBM data."""
    import glob

    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_*shots.txt")))
    if not files:
        print("No shot files found. Run collect_shots() first.")
        return

    print(f"\nFound {len(files)} shot files:")
    for f in files:
        name = os.path.basename(f)
        with open(f, encoding="utf-8") as fh:
            lines = fh.readlines()
        print(f"  {name}: {len(lines) - 1} shots")

    print("\nTo run Grammar Fingerprinting on these files:")
    print("  1. Copy files to results/readout_raw_data/")
    print("  2. Update GROUND_TRUTH in run_validation_pipeline.py")
    print("  3. Run: python code/run_validation_pipeline.py --tasks sweep,report")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect raw quantum shots for Grammar Fingerprinting"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Specific backend name (e.g., ibm_brisbane). Default: least busy.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=8192,
        help="Shots per execution (default: 8192)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Repetitions per circuit (default: 5)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run quick validation on collected data",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip circuits whose output file already has the full expected shot count; re-run partials.",
    )

    args = parser.parse_args()
    SHOTS = args.shots
    REPS = args.reps

    if args.validate:
        quick_validate()
    else:
        collect_shots(
            backend_name=args.backend,
            shots=args.shots,
            reps=args.reps,
            resume=args.resume,
        )
