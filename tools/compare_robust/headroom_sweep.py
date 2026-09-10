"""Research-only headroom-width experiment. Does not change production settings."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(), "widths": {}}
    for width in (0.02, 0.03, 0.04):
        def headroom(raw, white=255):
            peak = raw.max(axis=-1) if raw.ndim == 3 else raw
            t = np.clip((white - peak.astype(float)) / (width * white), 0, 1)
            return np.rint(255 * t * t * (3 - 2 * t)).astype(np.uint8)
        compare.headroom = headroom
        records = {}
        for dataset in (*compare.analytic_cases(), *compare.rendered_cases(), *compare.fish_cases()):
            if not (dataset["name"].startswith(("noisy_color", "fish")) or dataset["meta"]["source"] == "committed rendered validation"):
                continue
            folder = args.out / f"{width:.2f}" / dataset["name"]
            folder.mkdir(parents=True)
            result, _ = compare.native_run(dataset, "headroom", args.native.resolve(), folder)
            normals = result[:, 3, :3]
            valid = np.isfinite(normals).all(axis=1)
            truth_ok = np.isfinite(dataset["truth"]).all(axis=1) & valid
            record = {"coverage": float(valid.mean()),
                      "truth_error": compare.stats(compare.angular(normals[truth_ok], dataset["truth"][truth_ok])),
                      "input_sha256": hashlib.sha256((folder / "input.bin").read_bytes()).hexdigest()}
            count = dataset["pair_count"]
            if count:
                repeated = np.tile(normals[:count], (len(normals) // count - 1, 1))
                record["pair_change"] = compare.stats(compare.angular(normals[count:], repeated))
            records[dataset["name"]] = record
            print(width, dataset["name"], record, flush=True)
        report["widths"][str(width)] = records
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
