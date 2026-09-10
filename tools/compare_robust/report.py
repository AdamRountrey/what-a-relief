"""Audit saved comparison arrays and export compact, reproducible result tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

METHODS = ["ls", "huber10", "huber80", "cauchy", "l1_unit", "l1_255", "sbl_unit", "sbl_255"]


def audit(root, summary, require_fish=True):
    largest_ls_difference = 0.0
    count = 0
    solver_status={}
    for case, results in summary.items():
        folder = root/case
        with np.load(folder/"samples.npz") as archive:
            data = {key:archive[key] for key in archive.files}
        meta = json.loads((folder/"dataset.json").read_text())
        n, lights = data["b"].shape
        native = np.fromfile(folder/"native.bin", dtype="<f8").reshape(n, 4, 6)
        outputs = {method:native[:,i] for i,method in enumerate(METHODS[:4])}
        outputs.update({method:np.load(folder/(method+".npy")) for method in METHODS[4:]})
        solver_status[case]={}
        for method, value in outputs.items():
            assert value.shape == (n, 6), (case, method)
            valid = np.isfinite(value[:,:3]).all(axis=1)
            assert abs(valid.mean()-results[method]["coverage"]) < 1e-12
            assert np.all(np.abs(np.linalg.norm(value[valid,:3],axis=1)-1) < 1e-5)
            assert np.all(value[valid,2] > 0)
            solver_status[case][method]=dict(unsupported=int((~valid).sum()),
                valid_not_converged=int(np.sum(valid & (value[:,4]==0))),
                solved_mean_iterations=float(value[valid,3].mean()) if valid.any() else None)
        weights = data["h"].astype(float)/255 if meta["policy"]=="headroom" else np.ones((n,lights))
        for p in range(n):
            good = (data["b"][p] > np.float32(.02)) & (data["clipped"][p]==0) & (weights[p]>0)
            if good.sum() < 3: continue
            a = data["a"][p,good].astype(float)*np.sqrt(weights[p,good,None])
            b = data["b"][p,good].astype(float)*np.sqrt(weights[p,good])
            if np.linalg.matrix_rank(a) < 3: continue
            g = np.linalg.lstsq(a,b,rcond=None)[0]
            if g[2] <= 0 or np.linalg.norm(g) <= 1e-8: continue
            reference = g/np.linalg.norm(g)
            assert np.isfinite(native[p,0,:3]).all(), (case,p,"LS unexpectedly unsupported")
            difference = float(np.linalg.norm(reference-native[p,0,:3]))
            largest_ls_difference = max(largest_ls_difference,difference)
            assert difference < 1e-6, (case,p,difference)
            count += 1
    for method in METHODS:
        assert summary["pristine_8_binary"][method]["error"]["maximum"] < .001
        assert summary["pristine_8_binary"][method]["coverage"] == 1
    required=["pristine_8_binary", "broad_gloss_8_binary"]
    for lights in (8,16,40,64):
        required.extend([f"clean_noise_{lights}_binary",f"noisy_color_{lights}_binary",f"noisy_color_{lights}_headroom"])
    for lights in (8,40):
        required.extend([f"gain_mismatch_ring_{lights}_binary",f"gain_mismatch_irregular_{lights}_binary"])
    for name in ("robust_v1", "textured_primitives_v1", "holdout_relief_v1"):
        required.extend([name+"_binary",name+"_headroom"])
    if require_fish:
        required.extend([f"fish_{mode}_{policy}" for mode in ("fixed","dynamic") for policy in ("binary","headroom")])
    for case in required:
        assert case in summary, f"Incomplete comparison: {case}"
    return dict(audited_ls_fits=count,largest_ls_unit_vector_difference=largest_ls_difference,
                pristine_maximum_error_limit_degrees=.001,case_count=len(summary),required_case_count=len(required),
                fish_required=require_fish,solver_status=solver_status)


def number(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def strata(root, summary):
    result={}
    repo=Path(__file__).resolve().parents[2]
    for case in summary:
        if not any(case.startswith(name) for name in ("robust_v1", "textured_primitives_v1", "holdout_relief_v1")): continue
        folder=root/case
        meta=json.loads((folder/"dataset.json").read_text())
        with np.load(folder/"samples.npz") as archive:
            data={key:archive[key] for key in archive.files}
        fixture=repo/"tests/fixtures/mitsuba"/meta["name"]
        n=len(data["b"])
        native=np.fromfile(folder/"native.bin",dtype="<f8").reshape(n,4,6)
        outputs={method:native[:,i,:3] for i,method in enumerate(METHODS[:4])}
        outputs.update({method:np.load(folder/(method+".npy"))[:,:3] for method in METHODS[4:]})
        labels={kind:np.stack([np.asarray(Image.open(p)).ravel()[meta["points"]]!=0
            for p in sorted((fixture/(kind+"_truth")).glob("*.png"))],axis=1)
            for kind in ("shadow","highlight","saturation")}
        # The committed 12-bit ADC endpoint packs to 65534 or 65535 in uint16.
        observed_saturation=np.stack([np.asarray(Image.open(p)).ravel()[meta["points"]]>=65534
            for p in sorted((fixture/"images").glob("*.png"))],axis=1)
        assert np.array_equal(labels["saturation"],observed_saturation), case
        groups={"any_shadow":labels["shadow"].any(axis=1),
                "any_highlight":labels["highlight"].any(axis=1),
                "at_least_half_shadow":labels["shadow"].mean(axis=1)>=.5,
                "no_labelled_outlier":~np.logical_or.reduce([v.any(axis=1) for v in labels.values()])}
        result[case]={}
        for group,selection in groups.items():
            if not selection.any(): continue
            rows={}
            for method,normal in outputs.items():
                valid=np.isfinite(normal).all(axis=1)
                normal=normal/np.linalg.norm(normal,axis=1,keepdims=True)
                error=np.degrees(np.arccos(np.clip(np.sum(normal*data["truth"],axis=1),-1,1)))
                good=selection&valid
                rows[method]=dict(coverage=float(valid[selection].mean()),
                    mean_error=float(error[good].mean()) if good.any() else None,
                    penalized_mean_error=float(np.where(valid,error,180)[selection].mean()))
            result[case][group]=dict(n=int(selection.sum()),methods=rows)
    return result


def fish_audit(root, summary):
    result={}
    for case in summary:
        if not case.startswith("fish"): continue
        folder=root/case
        with np.load(folder/"samples.npz") as archive:
            data={key:archive[key] for key in archive.files}
        meta=json.loads((folder/"dataset.json").read_text())
        pairs=meta["pair_count"]
        lights=data["b"].shape[1]
        b=data["b"].reshape(-1,pairs,lights)
        clipped=data["clipped"].reshape(b.shape)
        weights=data["h"].reshape(b.shape) if meta["policy"]=="headroom" else np.full(b.shape,255,np.uint8)
        effective=weights*((b>np.float32(.02)) & (clipped==0))
        unchanged=np.all(effective[1:]==effective[:1],axis=2).ravel()
        n=len(data["b"])
        native=np.fromfile(folder/"native.bin",dtype="<f8").reshape(n,4,6)
        outputs={method:native[:,i,:3] for i,method in enumerate(METHODS[:4])}
        outputs.update({method:np.load(folder/(method+".npy"))[:,:3] for method in METHODS[4:]})
        rows={}
        for method,normal in outputs.items():
            normal=normal/np.linalg.norm(normal,axis=1,keepdims=True)
            normal=normal.reshape(-1,pairs,3)
            changes=np.degrees(np.arccos(np.clip(np.sum(normal[1:]*normal[:1],axis=2),-1,1))).ravel()
            good=unchanged & np.isfinite(changes)
            values=changes[good]
            rows[method]=dict(n=len(values),pair_failures=int(np.sum(unchanged & ~np.isfinite(changes))),
                mean=float(values.mean()),p99=float(np.quantile(values,.99)),maximum=float(values.max()),
                above10=int(np.sum(values>10)))
        result[case]=dict(total_pairs=int(unchanged.size),unchanged_effective_weights=int(unchanged.sum()),
            lower_cutoff_changes=int(np.sum(np.any((b[1:]>np.float32(.02))!=(b[:1]>np.float32(.02)),axis=2))),
            clipping_changes=int(np.sum(np.any(clipped[1:]!=clipped[:1],axis=2))),
            headroom_changes=int(np.sum(np.any(weights[1:]!=weights[:1],axis=2))),methods=rows)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--allow-missing-fish",action="store_true")
    args=parser.parse_args()
    summary=json.loads((args.results/"summary.json").read_text())
    evidence=audit(args.results,summary,require_fish=not args.allow_missing_fish)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"environment.json").write_text(json.dumps(dict(
        primary=json.loads((args.results/"environment.json").read_text()),
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+"\n")
    (args.out/"audit.json").write_text(json.dumps(evidence,indent=2)+"\n")
    (args.out/"strata.json").write_text(json.dumps(strata(args.results,summary),indent=2,allow_nan=False)+"\n")
    (args.out/"fish-audit.json").write_text(json.dumps(fish_audit(args.results,summary),indent=2,allow_nan=False)+"\n")
    with (args.out/"metrics.csv").open("w",newline="") as f:
        fields=["case","method","coverage","mean_error","p95_error","p99_error","common_mean_error",
                "penalized_mean_error","change_p99","change_max","change_above10","pair_failures",
                "mean_iterations","valid_not_converged","unsupported","seconds"]
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for case,results in summary.items():
            for method,r in results.items():
                change=r.get("change",{})
                writer.writerow(dict(case=case,method=method,coverage=r["coverage"],mean_error=r["error"]["mean"],
                    p95_error=r["error"]["p95"],p99_error=r["error"]["p99"],common_mean_error=r["common_error"]["mean"],
                    penalized_mean_error=r["penalized_error"]["mean"],change_p99=change.get("p99"),
                    change_max=change.get("maximum"),change_above10=change.get("above10"),pair_failures=r.get("pair_failures"),
                    mean_iterations=evidence["solver_status"][case][method]["solved_mean_iterations"],
                    valid_not_converged=evidence["solver_status"][case][method]["valid_not_converged"],
                    unsupported=evidence["solver_status"][case][method]["unsupported"],seconds=r["seconds"]))
    lines=["# Solver Comparison Tables", "", "Generated from saved arrays; lower angular errors are better.", "",
           "## Mean Angular Error (Degrees)", "", "| Case | "+" | ".join(METHODS)+" |",
           "| --- | "+" | ".join(["---:"]*len(METHODS))+" |"]
    for case,results in summary.items():
        if not case.startswith("fish"):
            lines.append("| "+case+" | "+" | ".join(number(results[m]["error"]["mean"]) for m in METHODS)+" |")
    lines += ["", "## Fish Perturbations: P99 Change (Degrees)", "",
              "| Case | "+" | ".join(METHODS)+" |", "| --- | "+" | ".join(["---:"]*len(METHODS))+" |"]
    for case,results in summary.items():
        if case.startswith("fish"):
            lines.append("| "+case+" | "+" | ".join(number(results[m]["change"]["p99"]) for m in METHODS)+" |")
    lines += ["", "## Coverage And Convergence", "", "Counts below use each method's own stopping rule.", "",
              "| Case | Method | Coverage % | Solved mean iterations | Solved but not converged |", "| --- | --- | ---: | ---: | ---: |"]
    for case,results in summary.items():
        for method,r in results.items():
            status=evidence["solver_status"][case][method]
            if status["valid_not_converged"] or r["coverage"] < 1:
                lines.append(f"| {case} | {method} | {100*r['coverage']:.2f} | {number(status['solved_mean_iterations'],1)} | {status['valid_not_converged']} |")
    lines += ["", "## Rendered Per-Object Mean Error", "",
              "| Scene / policy / object | "+" | ".join(METHODS)+" |", "| --- | "+" | ".join(["---:"]*len(METHODS))+" |"]
    for case,results in summary.items():
        if not any(case.startswith(name) for name in ("robust_v1", "textured_primitives_v1", "holdout_relief_v1")): continue
        meta=json.loads((args.results/case/"dataset.json").read_text())
        names={str(value):key for key,value in meta["region_names"].items()}
        for region in results["ls"]["regions"]:
            label=case+" / "+names.get(region,region)
            lines.append("| "+label+" | "+" | ".join(number(results[m]["regions"][region]["error"]["mean"]) for m in METHODS)+" |")
    (args.out/"tables.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in evidence.items() if k!="solver_status"},indent=2))
    print("Tables:", args.out)


if __name__=="__main__": main()
