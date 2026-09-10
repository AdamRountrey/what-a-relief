"""Secondary SBL budget check; does not replace the default-budget comparison."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from compare import angular, stats, unit, np

CASES = ["clean_noise_8_binary", "gain_mismatch_irregular_8_binary", "robust_v1_binary",
         "holdout_relief_v1_binary", "fish_dynamic_headroom"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    source=args.reference/"rpsnumerics.py"
    environment=json.loads((args.results/"environment.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==environment["reference_sha256"]
    spec=importlib.util.spec_from_file_location("reference_sbl",source)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    func=module.sparse_bayesian_learning
    recorded={}
    def trace(frame,event,arg):
        if event=="return" and frame.f_code is func.__code__:
            recorded["iterations"]=frame.f_locals.get("ite",10000)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"environment.json").write_text(json.dumps(dict(
        primary_environment=environment,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tolerance=1e-8,budget=10000,sample_count_per_case=32,cases=CASES),indent=2)+"\n")
    results={}
    for case in CASES:
        folder=args.results/case
        with np.load(folder/"samples.npz") as archive:
            data={key:archive[key] for key in archive.files}
        # Predeclared even sampling, independent of either error or convergence.
        points=np.linspace(0,len(data["b"])-1,32,dtype=int)
        h=data["h"] if case.endswith("headroom") else np.full_like(data["h"],255)
        for method in ("sbl_unit","sbl_255"):
            original=np.load(folder/(method+".npy"))[points]
            normals=np.full((len(points),3),np.nan)
            iterations=[]
            start=time.perf_counter()
            for j,p in enumerate(points):
                b=data["b"][p]; a=data["a"][p]
                good=np.isfinite(b)&(b>np.float32(.02))&(data["clipped"][p]==0)&(h[p]>0)
                if good.sum()<3 or np.linalg.matrix_rank(a[good])<3: continue
                row_scale=np.sqrt(h[p,good].astype(float)/255)
                aa=a[good].astype(float)*row_scale[:,None]
                bb=(b[good].astype(float)*row_scale*(255 if method.endswith("255") else 1))[:,None]
                recorded.clear()
                try:
                    sys.setprofile(trace)
                    g=func(aa,bb,max_ite=10000).ravel()
                finally:
                    sys.setprofile(None)
                iterations.append(recorded.get("iterations",10000))
                if np.isfinite(g).all() and g[2]>0 and np.linalg.norm(g)>1e-8:
                    normals[j]=unit(g)
            key=case+"_"+method
            np.savez_compressed(args.out/(key+".npz"),points=points,normals=normals,iterations=iterations)
            results[key]=dict(points=points.tolist(),budget=10000,seconds=time.perf_counter()-start,
                coverage=float(np.isfinite(normals).all(axis=1).mean()),iterations=stats(iterations),
                nonconverged=int(np.sum(np.asarray(iterations)>=10000)),
                change_from_default=stats(angular(normals,original[:,:3])),
                default_error=stats(angular(original[:,:3],data["truth"][points])),
                extended_error=stats(angular(normals,data["truth"][points])))
            (args.out/"summary.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
            r=results[key]
            print(key,"mean change",r["change_from_default"]["mean"],"max change",r["change_from_default"]["maximum"],
                  "error",r["default_error"]["mean"],"->",r["extended_error"]["mean"],
                  "not converged",r["nonconverged"],flush=True)


if __name__=="__main__": main()
