"""Controlled research comparison. Never changes application source or input files."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_COMMIT = "f03aa95b57e746a7d31df76b1c0fa0a83584a3c1"
METHODS = ["ls", "huber10", "huber80", "cauchy", "l1_unit", "l1_255", "sbl_unit", "sbl_255"]


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def decode(x):
    x = np.asarray(x, dtype=np.float64) / 255.0
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def encode(x):
    x = np.clip(x, 0, 1)
    return np.rint(255 * np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)).astype(np.uint8)


def headroom(raw, white=255):
    maximum = raw.max(axis=-1) if raw.ndim == 3 else raw
    t = np.clip((white-maximum.astype(float))/(.02*white), 0, 1)
    return np.rint(255*t*t*(3-2*t)).astype(np.uint8)


def luminance(rgb):
    return rgb @ np.array([.2126, .7152, .0722])


def lights(count, ring=False):
    i = np.arange(count)
    azimuth = 2*np.pi*i/count if ring else i*np.pi*(3-np.sqrt(5))
    z = np.full(count, .6) if ring else .25+.6*(i+.5)/count
    r = np.sqrt(1-z*z)
    return np.stack([r*np.cos(azimuth), r*np.sin(azimuth), z], axis=-1)


def pack(name, a, b, truth=None, clipped=None, h=None, region=None, pair_count=0, **meta):
    b = np.asarray(b, dtype="<f4")
    if a.ndim == 2:
        a = np.broadcast_to(a, (*b.shape, 3))
    result = dict(name=name, a=np.ascontiguousarray(a, dtype="<f4"), b=b,
                clipped=np.zeros(b.shape, np.uint8) if clipped is None else np.asarray(clipped, np.uint8),
                h=np.full(b.shape, 255, np.uint8) if h is None else np.asarray(h, np.uint8),
                truth=np.full((len(b), 3), np.nan) if truth is None else np.asarray(truth),
                region=np.ones(len(b), int) if region is None else np.asarray(region),
                pair_count=pair_count, meta=meta)
    if result["a"].shape != (*b.shape, 3) or result["h"].shape != b.shape or result["clipped"].shape != b.shape:
        raise ValueError(f"Inconsistent sample shapes: {name}")
    return result


def analytic_cases():
    y, x = np.mgrid[:16, :16]
    u, v = (x.ravel()-7.5)/8, (y.ravel()-7.5)/8
    offset = u-.35*v
    ridge = np.exp(-.5*(offset/.16)**2)
    du = .08+.08*u-.08*offset*ridge/.16**2
    dv = -.04+.06*v+.35*.08*offset*ridge/.16**2
    truth = unit(np.stack([-du, dv, np.ones_like(u)], -1))
    for count in (8, 16, 40, 64):
        a = lights(count)
        base = (.40+.12*np.sin(5*u)*np.cos(4*v))[:, None]*np.maximum(0, truth@a.T)
        rng = np.random.default_rng(11920+count)
        for mode in ("clean_noise", "noisy_color"):
            b, c, h = [], [], []
            for repeat in range(2):
                if mode == "clean_noise":
                    sample = np.maximum(0, base+rng.normal(size=base.shape)*.003)
                    b.append(sample); c.append(np.zeros_like(base)); h.append(np.full_like(base, 255))
                else:
                    reflectance = np.where((u < 0)[:, None], [.95,.45,.12], [.16,.38,.85])
                    signal = 1.8*(.8+.16*np.sin(3*u)*np.cos(3*v))[:,None,None]*\
                        (np.maximum(0, truth@a.T)+.04)[...,None]*reflectance[:,None,:]
                    raw = encode(np.maximum(0, signal+rng.normal(size=signal.shape)*np.sqrt(signal/20000+.001**2)))
                    b.append(luminance(decode(raw)))
                    c.append(np.any(raw==255, axis=-1)*255); h.append(headroom(raw))
            yield pack(f"{mode}_{count}", a, np.concatenate(b), np.tile(truth,(2,1)),
                       np.concatenate(c), np.concatenate(h), pair_count=len(truth),
                       pair_kind="independent sensor noise on identical geometry", source="new analytic control")
        if count in (8,40):
            for ring in (False, True):
                a = lights(count, ring)
                gain = 1+.2*np.cos(2*np.pi*np.arange(count)/count-.4)+.12*np.sin(4*np.pi*np.arange(count)/count)
                albedo = .2+.25*(.5+.5*np.sin(11*u)*np.cos(7*v))
                half = unit(a+np.array([0,0,1]))
                signal = gain*(albedo[:,None]*np.maximum(0,truth@a.T)+.08*(1+u)[:,None]*np.maximum(0,truth@half.T)**24)+.08*albedo[:,None]
                raw = encode(signal)
                changed = np.clip(raw.astype(int)+rng.integers(-1,2,raw.shape),0,255).astype(np.uint8)
                joined = np.concatenate([raw,changed])
                yield pack(f"gain_mismatch_{'ring' if ring else 'irregular'}_{count}",a,decode(joined),
                           np.tile(truth,(2,1)), (joined==255)*255, headroom(joined), pair_count=len(truth),
                           pair_kind="one-code perturbation", source="new analytic model-mismatch control")
    a = lights(8,True)
    a[:, :2] *= np.sqrt(1-.75**2)/np.sqrt(1-.6**2); a[:,2]=.75
    diffuse = .12*np.maximum(0,truth@a.T)
    gloss = .70*np.maximum(0,truth@unit(a+np.array([0,0,1])).T)**12
    yield pack("broad_gloss_8",a,np.minimum(1,diffuse+gloss),truth, source="analytic dense-gloss limitation")
    yield pack("pristine_8",a,.6*(truth@a.T),truth, source="exact Lambertian sanity check")


def rendered_cases():
    for name in ("robust_v1", "textured_primitives_v1", "holdout_relief_v1"):
        root = ROOT/"tests/fixtures/mitsuba"/name
        meta = json.loads((root/"manifest.json").read_text())
        mask = np.asarray(Image.open(root/"solve_mask.png")) != 0
        points = np.flatnonzero(mask)
        rng = np.random.default_rng(887122)
        points = np.sort(rng.choice(points,min(2000,len(points)),replace=False))
        b,c,h = [],[],[]
        for file in sorted((root/"images").glob("*.png")):
            raw = np.asarray(Image.open(file))
            white = 65535 if raw.dtype.itemsize > 1 else 255
            linear = raw.astype(float)/white
            if raw.ndim==3: linear=luminance(linear)
            b.append(linear.ravel()[points])
            c.append(np.asarray(Image.open(root/"saturation_truth"/file.name)).ravel()[points])
            maximum = raw.max(axis=-1) if raw.ndim==3 else raw
            h.append(headroom(maximum.ravel(),white)[points])
        b,c,h = np.stack(b,1), np.stack(c,1), np.stack(h,1)
        truth = unit(np.stack([np.asarray(Image.open(root/f"normal_{ax}.png")).ravel()[points]/65535*2-1 for ax in "xyz"],1))
        region = np.asarray(Image.open(root/"shape_index.png")).ravel()[points]
        a = np.asarray(meta["light_directions"])
        if meta["lighting_model"] == "near_field_ring":
            height,width=mask.shape
            px,py=points%width,points//width
            scale=meta["pixel_scale_mm"]
            receiver=np.stack([(px-(width-1)*.5)*scale,((height-1)*.5-py)*scale,np.zeros(len(points))],1)
            # Same planar reference-surface approximation as production, not true AOV depth.
            az = unit(np.column_stack([a[:,:2],np.zeros(len(a))]))
            source=az*meta["ring_radius_mm"]; source[:,2]=meta["ring_height_mm"]
            d=source[None,:,:]-receiver[:,None,:]
            distance2=np.sum(d*d,axis=-1)
            a=unit(d)*((meta["ring_radius_mm"]**2+meta["ring_height_mm"]**2)/distance2)[...,None]
        yield pack(name,a,b,truth,c,h,region,source="committed rendered validation", points=points.tolist(),
                   region_names=meta["files"].get("shape_indices",{}),
                   fixture_manifest_sha256=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest())


def fish_cases():
    root = ROOT/"build/fish-noise-review/crop_inputs"
    if not root.exists(): return
    paths=sorted(root.glob("*.png"))
    calibration=ROOT/"build/fish-noise-review/continuous-final-smoke/light_vectors.csv"
    with calibration.open(newline="") as f:
        rows=list(csv.reader(f))
    a=np.asarray([[float(v) for v in row[-3:]] for row in rows[1:]])
    if len(a)!=len(paths): raise ValueError("Fish calibration/input mismatch")
    x,y=np.meshgrid(np.arange(12,640,24),np.arange(12,640,24))
    points=np.stack([x.ravel(),y.ravel()],1)
    selected=np.sort(np.random.default_rng(719981).choice(len(points),192,replace=False))
    points=points[selected]
    raw=[]; peak=0
    source_hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    for path in paths:
        rgb=np.asarray(Image.open(path).convert("RGB"))
        peak=max(peak,float(luminance(decode(rgb)).max()))
        raw.append(rgb[points[:,1],points[:,0]])
    raw=np.stack(raw,1)
    rng=np.random.default_rng(718832)
    stack=[raw]
    for _ in range(8): stack.append(np.clip(raw.astype(int)+rng.integers(-1,2,raw.shape),0,255).astype(np.uint8))
    samples=np.concatenate(stack)
    values=luminance(decode(samples))/peak
    clipped=np.any(samples==255,axis=-1)*255
    h=headroom(samples)
    for fixed in (True,False):
        yield pack(f"fish_{'fixed' if fixed else 'dynamic'}",a,values,
                   clipped=np.tile(clipped[:len(points)],(9,1)) if fixed else clipped,
                   h=np.tile(h[:len(points)],(9,1)) if fixed else h,
                   pair_count=len(points), source="real crop, no ground truth", points=points.tolist(),
                   pair_kind="eight one-code RGB perturbations", normalization_peak=peak,
                   source_sha256=source_hashes, calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest())


def stats(v):
    v=np.asarray(v); v=v[np.isfinite(v)]
    if not len(v): return dict(n=0,mean=None,p95=None,p99=None,maximum=None,above10=0)
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),
                p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),maximum=float(v.max()),above10=int((v>10).sum()))


def angular(a,b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a)*unit(b),axis=-1),-1,1)))


def native_run(dataset, policy, executable, folder):
    count=len(dataset["b"])
    weights=dataset["h"] if policy=="headroom" else np.full_like(dataset["h"],255)
    source=folder/"input.bin"
    with source.open("wb") as f:
        np.array([count,dataset["b"].shape[1]],dtype="<u4").tofile(f)
        dataset["a"].tofile(f); dataset["b"].tofile(f)
        dataset["clipped"].tofile(f); weights.tofile(f)
    target=folder/"native.bin"
    subprocess.run([str(executable),str(source),str(target)],check=True)
    native=np.fromfile(target,dtype="<f8").reshape(count,4,6)
    return native,weights


def reference_run(dataset, weights, module, method):
    count=len(dataset["b"])
    output=np.full((count,6),np.nan)
    l1=method.startswith("l1")
    scale=255. if method.endswith("255") else 1.
    func=module.L1_residual_min if l1 else module.sparse_bayesian_learning
    observed={}
    def trace(frame,event,arg):
        if event=="return" and frame.f_code is func.__code__:
            observed["iterations"]=frame.f_locals.get("iter" if l1 else "ite",0)
    for p,(a,b) in enumerate(zip(dataset["a"],dataset["b"])):
        ok=np.isfinite(b)&(b>np.float32(.02))&(dataset["clipped"][p]==0)&(weights[p]>0)
        if ok.sum()<3 or np.linalg.matrix_rank(a[ok])<3: continue
        w=weights[p,ok].astype(float)/255
        row_scale=w if l1 else np.sqrt(w)
        aa=a[ok].astype(float)*row_scale[:,None]
        bb=(b[ok].astype(float)*row_scale*scale)[:,None]
        start=time.perf_counter()
        observed.clear()
        try:
            sys.setprofile(trace)
            g=func(aa,bb).ravel()
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):
            g=np.full(3,np.nan)
        finally: sys.setprofile(None)
        elapsed=time.perf_counter()-start
        iterations=observed.get("iterations",1000)
        if np.isfinite(g).all() and np.linalg.norm(g)>1e-8 and g[2]>0:
            output[p,:3]=unit(g)
        output[p,3:]=[iterations,int(iterations<1000),elapsed]
    return output


def evaluate(dataset, outputs):
    has_truth=np.isfinite(dataset["truth"]).all(axis=1)
    common=np.logical_and.reduce([np.isfinite(v[:,:3]).all(axis=1) for v in outputs.values()])
    result={}
    pair=dataset["pair_count"]
    for method,v in outputs.items():
        valid=np.isfinite(v[:,:3]).all(axis=1)
        err=angular(v[:,:3],dataset["truth"])
        item=dict(coverage=float(valid.mean()),error=stats(err[valid&has_truth]),
                  common_error=stats(err[common&has_truth]),
                  penalized_error=stats(np.where(valid,err,180)[has_truth]),
                  iterations=stats(v[:,3]),nonconverged=int(np.sum(v[:,4]==0)),seconds=float(np.nansum(v[:,5])))
        item["regions"]={str(r):dict(n=int(np.sum(dataset["region"]==r)),
            coverage=float(valid[dataset["region"]==r].mean()),
            error=stats(err[(dataset["region"]==r)&valid&has_truth])) for r in np.unique(dataset["region"])}
        if pair:
            changes=[]; failures=0
            for k in range(1,len(v)//pair):
                good=valid[:pair]&valid[k*pair:(k+1)*pair]
                failures+=int((~good).sum())
                changes.extend(angular(v[:pair,:3][good],v[k*pair:(k+1)*pair,:3][good]))
            item["change"]=stats(changes); item["pair_failures"]=failures
        result[method]=item
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--only",default="")
    args=parser.parse_args()
    ref=args.reference.resolve(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(["git","-C",str(ref),"rev-parse","HEAD"],text=True).strip()
    if commit!=REFERENCE_COMMIT: raise ValueError(f"Unexpected reference commit {commit}")
    file=ref/"rpsnumerics.py"
    spec=importlib.util.spec_from_file_location("author_rpsnumerics",file)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    environment=dict(reference_commit=commit,reference_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                     python=sys.version,numpy=np.__version__,platform=platform.platform(),
                     native_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest(),
                     harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     production_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (ROOT/"src/robust_fit.hpp",ROOT/"src/photometric.cpp")})
    env_file=out/"environment.json"
    if env_file.exists() and json.loads(env_file.read_text())!=environment:
        raise ValueError("Environment or harness changed: choose a new output directory")
    env_file.write_text(json.dumps(environment,indent=2)+"\n")
    summary=json.loads((out/"summary.json").read_text()) if (out/"summary.json").exists() else {}
    from itertools import chain
    for dataset in chain(analytic_cases(),rendered_cases(),fish_cases()):
        if args.only and args.only not in dataset["name"]: continue
        for policy in ("binary","headroom"):
            if policy=="headroom" and not np.any((dataset["h"]>0)&(dataset["h"]<255)): continue
            name=dataset["name"]+"_"+policy
            folder=out/name; folder.mkdir(exist_ok=True)
            print("CASE",name,"samples",len(dataset["b"]),flush=True)
            digest=hashlib.sha256()
            for key in ("a","b","h","clipped","truth","region"):
                digest.update(np.ascontiguousarray(dataset[key]).tobytes())
            digest.update(json.dumps(dataset["meta"],sort_keys=True).encode())
            fingerprint=digest.hexdigest()
            stamp=folder/"sample_sha256.txt"
            if stamp.exists() and stamp.read_text().strip()!=fingerprint:
                raise ValueError("Input data changed: choose a new output directory")
            stamp.write_text(fingerprint+"\n")
            np.savez_compressed(folder/"samples.npz",**{k:dataset[k] for k in ("a","b","h","clipped","truth","region")})
            (folder/"dataset.json").write_text(json.dumps(dict(name=dataset["name"],policy=policy,
                pair_count=dataset["pair_count"],**dataset["meta"]),indent=2)+"\n")
            native,weights=native_run(dataset,policy,args.native.resolve(),folder)
            outputs={method:native[:,i,:] for i,method in enumerate(METHODS[:4])}
            for method in METHODS[4:]:
                cache=folder/(method+".npy")
                if cache.exists(): result=np.load(cache)
                else:
                    result=reference_run(dataset,weights,module,method)
                    np.save(cache,result)
                outputs[method]=result
                print(" ",method,"seconds",round(float(np.nansum(result[:,5])),2),flush=True)
            results=evaluate(dataset,outputs)
            (folder/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
            summary[name]=results
            for method in METHODS:
                r=results[method]
                print(" ",method,"mean",r["error"]["mean"],"coverage",r["coverage"],
                      "change_p99",r.get("change",{}).get("p99"),flush=True)
            (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print("Comparison complete:",out,flush=True)


if __name__=="__main__": main()
