"""Replay saved observations against a new native build, without rerunning SBL.

All input arrays and baseline results remain untouched. This is an analysis
tool, not a claim that lower perturbation sensitivity establishes accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

from compare import angular, stats


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    report = {"native_sha256": digest(args.native), "source_sha256": {
        name: digest(root / name) for name in
        ("src/robust_fit.hpp", "src/photometric.cpp", "src/image_io.cpp", "tools/compare_robust/replay.py")}, "cases": {}}
    for folder in sorted(args.results.iterdir()):
        if not (folder / "samples.npz").exists():
            continue
        data = dict(np.load(folder / "samples.npz"))
        metadata = json.loads((folder / "dataset.json").read_text())
        output = args.out / (folder.name + ".bin")
        subprocess.run([str(args.native.resolve()), str((folder / "input.bin").resolve()), str(output.resolve())], check=True)
        count = len(data["b"])
        old = np.fromfile(folder / "native.bin", dtype="<f8").reshape(count, 4, 6)
        new = np.fromfile(output, dtype="<f8").reshape(count, 4, 6)
        normals = {"before": old[:, 3, :3], "after": new[:, 3, :3], "ls": new[:, 0, :3]}
        common = np.all(np.isfinite(normals["before"]), axis=1) & np.all(np.isfinite(normals["after"]), axis=1)
        case = {"input_sha256": digest(folder / "input.bin"), "samples": count, "common_count": int(common.sum()), "methods": {}}
        pairs = metadata["pair_count"]
        for name, n in normals.items():
            valid = np.all(np.isfinite(n), axis=1)
            truth_ok = valid & np.all(np.isfinite(data["truth"]), axis=1)
            truth_error = np.full(count, 180.0)
            truth_error[truth_ok] = angular(n[truth_ok], data["truth"][truth_ok])
            result = {"coverage": float(valid.mean()), "truth_error": stats(truth_error[truth_ok]),
                      "common_truth_error": stats(truth_error[truth_ok & common])}
            if np.all(np.isfinite(data["truth"])):
                result["unsupported_penalty_mean"] = float(truth_error.mean())
            if pairs:
                repeated = np.tile(n[:pairs], (count // pairs - 1, 1))
                result["pair_change"] = stats(angular(n[pairs:], repeated))
            result["regions"] = {str(region): {"coverage": float(valid[data["region"] == region].mean()),
                "truth_error": stats(truth_error[truth_ok & (data["region"] == region)])}
                for region in np.unique(data["region"])}
            case["methods"][name] = result
        case["total_native_seconds"] = {"before": float(old[:, 3, 5].sum()), "after": float(new[:, 3, 5].sum())}
        report["cases"][folder.name] = case
        print(folder.name, " ".join(f"{name}: err={r['truth_error']['mean']}, p99={r.get('pair_change',{}).get('p99')}, coverage={r['coverage']:.4f}"
              for name, r in case["methods"].items()), flush=True)
    if not report["cases"]:
        raise ValueError("No saved comparison cases found")
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
