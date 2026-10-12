#!/usr/bin/env python3
"""Script 93 — Prof. Saeed encoding-aware Yhat + candidate reverse bridge.

REAL Script-73 NHANES input ONLY; no synthetic fallback. Never modifies frozen files.
Design: train 2005-06; calibrate phenotype likelihood SD on 2007-08;
evaluate frozen on 2009-18 and 2021-23 (optionally also report 2007-08).

Forward: compare raw Z, quantized decoded Z, all decimal interleaved digits,
structured decimal digits + decoded coords, and COMPLETE Cantor integer bytes.
Two-coordinate only: factorized prime-exponent and 2-adic representations
are equivalent to quantized coordinate pairs, NOT new information.

Reverse: posterior over 2005-06 frozen-Z codebook using training NHANES
weights as empirical prior and Gaussian likelihood around learned h(code).
Report MAP candidate and posterior mean, compared to direct ridge Y->Z;
posterior marginal 90% coverage/width are descriptive, not conformal.

Ridge/HGB models consume all-digit/byte features, never truncated bit windows.
Do not infer that exact coding improves identifiability or is compression.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ALL_Z = [
    "SP_SHARED1", "SP_SHARED2", "SP_DISCORD1", "SP_DISCORD2",
    "SP_A_PRIVATE3", "SP_A_PRIVATE4", "SP_G_PRIVATE3"
]
PERIODS = [("2007-2008_CAL", "0708"), ("2009-2018", None), ("2021-2023", None)]


def load_data(path: Path, cols: list[str], outcome: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Script-73 real dataset not found: {path}")
    want = ["SEQN", "CYCLE", "PERIOD", "SURVEY_WT", "ELIGIBLE_ADULT_NONPREG", outcome] + cols
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(want) - set(header))
    if missing:
        raise ValueError(f"Script-73 dataset is missing: {missing}")
    d = pd.read_csv(path, usecols=list(dict.fromkeys(want)), low_memory=False)
    for c in cols + [outcome, "SURVEY_WT"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["CYCLE"] = (d["CYCLE"].astype(str).str.strip()
                  .str.replace(r"\.0$", "", regex=True).str.zfill(4))
    d = d[d["ELIGIBLE_ADULT_NONPREG"].eq(1)
          & d[cols + [outcome, "SURVEY_WT"]].notna().all(axis=1)
          & d["SURVEY_WT"].gt(0)].copy()
    if d["SEQN"].duplicated().any():
        raise AssertionError("Duplicate SEQN in filtered Script-73 observations")
    return d


def wnorm(d: pd.DataFrame) -> np.ndarray:
    w = d["SURVEY_WT"].to_numpy(float)
    return w / w.mean()


def wr2(y: np.ndarray, yh: np.ndarray, w: np.ndarray) -> float:
    y, yh = np.asarray(y), np.asarray(yh)
    mid = np.average(y, weights=w)
    den = np.average((y-mid)**2, weights=w)
    return float(1.0-np.average((y-yh)**2, weights=w)/den) if den>0 else float("nan")


def zwscore(z: np.ndarray, zh: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    center = np.average(z, axis=0, weights=w)
    sse = np.sum(w[:, None]*(z-zh)**2)
    sst = np.sum(w[:, None]*(z-center)**2)
    r2 = float(1-sse/sst) if sst>0 else float("nan")
    rmse = float(np.sqrt(np.average(np.mean((z-zh)**2,axis=1),weights=w)))
    return r2, rmse


def model(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=12,
                learning_rate=.05, min_samples_leaf=35, l2_regularization=3.0,
                random_state=93)
    raise ValueError(name)


def fit(model_obj, name: str, x: np.ndarray, y: np.ndarray, w: np.ndarray):
    if name == "ridge":
        model_obj.fit(x, y, ridge__sample_weight=w)
    else:
        model_obj.fit(x, y, sample_weight=w)
    return model_obj


def quantize(Z: np.ndarray, places: int) -> tuple[np.ndarray, np.ndarray]:
    scl = 10**places
    q = np.rint(Z*scl).astype(np.int64)
    u = np.where(q>=0, 2*q, -2*q-1)
    if np.any(u<0): raise OverflowError("Quantization overflow")
    return q, u


def decimal_features(u: np.ndarray, width: int) -> np.ndarray:
    """Fixed-width interleaving ordered by digit position, then dimension.

    A 7x7 digit matrix is flattened row-major by place, not by coordinate.
    This contains EVERY digit of the exact decimal code.
    """
    n,d = u.shape
    seq = np.empty((n, width*d), dtype=np.uint8)
    for j in range(d):
        digits = np.array([list(f"{int(t):0{width}d}") for t in u[:,j]],dtype=np.uint8)
        seq[:,j::d] = digits
    return seq.astype(np.float32)


def cantor_pair(a: int, b: int) -> int:
    s=a+b
    return s*(s+1)//2+b


def cantor_unpair(c: int) -> tuple[int,int]:
    w=(math.isqrt(8*c+1)-1)//2
    t=w*(w+1)//2
    return w-(c-t),c-t


def cantor_tuple(vals) -> int:
    ans=int(vals[0])
    for v in vals[1:]:
        ans=cantor_pair(ans,int(v))
    return ans


def uncantor(code: int, d: int) -> tuple[int,...]:
    tail=[]
    for _ in range(d-1):
        code,z=cantor_unpair(code)
        tail.append(z)
    return tuple(reversed(tail+[code]))


def cantor_bytes(u: np.ndarray, cap_bytes: int) -> tuple[np.ndarray, int]:
    """Big-endian left-pad, complete bytes from each exact Cantor integer."""
    n,d = u.shape
    out = np.zeros((n,cap_bytes+1),dtype=np.float32)
    largest=0
    for i in range(n):
        code=cantor_tuple(u[i])
        if i<8 or i==n-1:
            if uncantor(code,d)!=tuple(int(v) for v in u[i]):
                raise AssertionError("Cantor roundtrip failed")
        required=max(1,(code.bit_length()+7)//8)
        largest=max(largest,required)
        if required>cap_bytes:
            raise OverflowError(f"Cantor needs {required} bytes, --max-cantor-bytes={cap_bytes}; "
                                "increase the limit (no silent truncation)")
        out[i,0]=float(required)
        out[i,1+cap_bytes-required:] = np.frombuffer(code.to_bytes(required,byteorder="big"),dtype=np.uint8)
    return out,largest


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    idx=np.argsort(x)
    x,w=x[idx],w[idx]
    # Standard empirical weighted quantile; not conformal.
    pos=np.searchsorted(np.cumsum(w)/np.sum(w),q,side="left")
    return float(x[min(pos,len(x)-1)])


def posterior_batch(obs_y, candidate_pred, Zcodebook, prior, sd, batch_size=120,
                    interval_n=0):
    """Empirical Bayes finite codebook posterior; no test Z appears here."""
    n,d = len(obs_y), Zcodebook.shape[1]
    means=np.empty((n,d),float)
    maps=np.empty((n,d),float)
    # Construct weighted 90% marginal intervals only if requested.
    lows=np.empty((n,d),float) if interval_n else None
    highs=np.empty((n,d),float) if interval_n else None
    for start in range(0,n,batch_size):
        stop=min(n,start+batch_size)
        v=obs_y[start:stop,None]-candidate_pred[None,:]
        logw=np.log(prior)[None,:]-.5*(v/sd)**2
        logw-=np.max(logw,axis=1,keepdims=True)
        w=np.exp(logw)
        w/=w.sum(axis=1,keepdims=True)
        means[start:stop]=w@Zcodebook
        maps[start:stop]=Zcodebook[np.argmax(w,axis=1)]
        if interval_n:
            for row in range(start,stop):
                for j in range(d):
                    lows[row,j]=weighted_quantile(Zcodebook[:,j],w[row-start],.05)
                    highs[row,j]=weighted_quantile(Zcodebook[:,j],w[row-start],.95)
    return means,maps,lows,highs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path,default=ROOT/"Results"/"73_frozen_phenotype_map_input.csv")
    ap.add_argument("--out",type=Path,default=ROOT/"Results")
    ap.add_argument("--components",default="",help="Optional two frozen components, e.g. SP_SHARED1,SP_A_PRIVATE3; default all seven")
    ap.add_argument("--outcome",default="PHQ9_TOTAL",choices=["PHQ9_TOTAL","SOMATIC_SCORE","COGAFF_SUM"])
    ap.add_argument("--decimals",type=int,default=3)
    ap.add_argument("--width",type=int,default=7,help="Fixed decimal digits per unsigned quantized component")
    ap.add_argument("--max-cantor-bytes",type=int,default=160)
    ap.add_argument("--intervals",action="store_true",help="Compute descriptive weighted posterior marginal 90%% interval coverage (slow)")
    ap.add_argument("--no-cantor",action="store_true",help="Skip exact Cantor byte feature model")
    args=ap.parse_args()
    if args.decimals<0 or args.decimals>6 or args.width<1 or args.width>14 or args.max_cantor_bytes<1:
        ap.error("Invalid precision/width/byte limit")
    comps=[x.strip() for x in args.components.split(",") if x.strip()] if args.components else ALL_Z
    if len(comps)<2 or len(set(comps))!=len(comps) or any(x not in ALL_Z for x in comps):
        ap.error("Use 2 to 7 distinct frozen Z columns")

    d=load_data(args.data,comps,args.outcome)
    tr=d[d.CYCLE.eq("0506")].copy()
    cal=d[d.CYCLE.eq("0708")].copy()
    tests={tag:(d[d.CYCLE.eq(c)].copy() if c else d[d.PERIOD.eq(tag)].copy()) for tag,c in PERIODS}
    tests={tag:v for tag,v in tests.items() if len(v)}
    if len(tr)<100 or len(cal)<100 or not tests:
        raise ValueError(f"Insufficient train={len(tr)}, calibration={len(cal)}, or held-out rows")
    datasets={"train":tr,**tests}
    raw={tag:dd[comps].to_numpy(float) for tag,dd in datasets.items()}
    Y={tag:dd[args.outcome].to_numpy(float) for tag,dd in datasets.items()}
    W={tag:wnorm(dd) for tag,dd in datasets.items()}
    qints={}; unsigned={}; decimal={}; decoded={}
    for tag,zz in raw.items():
        q,u=quantize(zz,args.decimals)
        qints[tag]=q
        unsigned[tag]=u
        if np.any(np.array([len(str(int(x))) for x in u.flat])>args.width):
            raise ValueError(f"Decimal encoded width insufficient in {tag}; increase --width")
        decimal[tag]=decimal_features(u,args.width)
        decoded[tag]=q/(10**args.decimals)
        max_err=float(np.max(np.abs(decoded[tag]-zz)))
        if max_err>0.5/(10**args.decimals)+1e-10:
            raise AssertionError("Quantization error bound failed")
        # Every interleaved digit must recover the underlying integer by fixed-position parse.
        if tag=="train":
            sample=decimal[tag][:min(20,len(zz))].astype(int)
            for row_index,row in enumerate(sample):
                parsed=tuple(int(''.join(str(t) for t in row[j::len(comps)])) for j in range(len(comps)))
                if parsed!=tuple(int(x) for x in u[row_index]):
                    raise AssertionError("Decimal interleaving inverse failure")

    features={
        "raw_Z":raw,
        "decoded_decimal_Z":decoded,
        "decimal_digit_tokens":decimal,
        "decimal_multiscale":{tag:np.column_stack([decoded[tag], decimal[tag]]) for tag in datasets},
    }
    audits=[dict(representation="decimal_digit_tokens",state="exact_after_quantization",
                 n_components=len(comps),n_features=args.width*len(comps),
                 max_Z_quant_error=float(max(np.max(np.abs(decoded[t]-raw[t])) for t in datasets)),
                 notes="Complete interleaved digits; no code digits truncated")]
    if not args.no_cantor:
        try:
            cb={}; caps=[]
            for tag in datasets:
                cb[tag],cap=cantor_bytes(unsigned[tag],args.max_cantor_bytes)
                caps.append(cap)
            features["cantor_complete_bytes"]=cb
            audits.append(dict(representation="cantor_complete_bytes",state="complete",
                               n_components=len(comps),n_features=args.max_cantor_bytes+1,
                               max_cantor_bytes_used=max(caps),notes="All encoded integer bytes provided; no truncation"))
        except OverflowError as err:
            audits.append(dict(representation="cantor_complete_bytes",state="skipped",
                               notes=str(err)))
    if len(comps)==2:
        # These exact factorizations of the code contain the exponent fields.
        # They are explicitly equivalent to unsigned q(Z); integer products are
        # unnecessary (and can be huge). This is an honest factorized baseline.
        features["prime_exponent_fields"]={tag:unsigned[tag].astype(float) for tag in datasets}
        features["two_adic_fields"]={tag:unsigned[tag].astype(float) for tag in datasets}
        for n in ("prime_exponent_fields","two_adic_fields"):
            audits.append(dict(representation=n,state="factorized_exact",
                               n_components=2,n_features=2,
                               notes="Exponents/valuation and odd part; equivalent to quantized coordinates, not independent compression"))
    else:
        for n in ("prime_exponents","two_adic"):
            audits.append(dict(representation=n,state="not_run",
                               notes="Unbounded giant integer encodings; 2-adic only two-coordinate implementation"))

    # Standardized, weighted direct Y->Z comparator identical for every method.
    rev=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    rev.fit(Y["train"][:,None],raw["train"],ridge__sample_weight=W["train"])
    reverse=[]
    for tag in tests:
        pred=rev.predict(Y[tag][:,None]); r,rm=zwscore(raw[tag],pred,W[tag])
        reverse.append(dict(method="direct_ridge_Y_to_Z",test_period=tag,
                  reverse_variant="conditional_mean",weighted_Z_R2=r,weighted_Z_RMSE=rm,
                  marginal_90_coverage=np.nan,marginal_90_mean_width=np.nan))

    fw=[]
    candidate_models={}
    print(f"SCRIPT 93 | REAL NHANES Script-73 | {len(comps)} Z components | train 2005-06 n={len(tr)}")
    print(f"Calibration 2007-08 n={len(cal)}; test "+", ".join(f"{t} n={len(v)}" for t,v in tests.items()))
    print(f"Quantization={args.decimals} dp; complete decimal digits={args.width*len(comps)}")
    for representation,fs in features.items():
        for mname in ("ridge","hgb"):
            # Skip tree model on high dimension Cantor if excessive overhead.
            if representation=="cantor_complete_bytes" and fs["train"].shape[1]>300 and mname=="hgb":
                audits.append(dict(representation=representation+"_hgb",state="skipped",
                                   notes="Cantor tree model skipped due to >300 untruncated byte features; ridge evaluated"))
                continue
            mdl=fit(model(mname),mname,fs["train"],Y["train"],W["train"])
            key=representation+"__"+mname
            cal_pred=mdl.predict(fs["2007-2008_CAL"])
            sd=float(np.sqrt(np.average((Y["2007-2008_CAL"]-cal_pred)**2,
                                              weights=W["2007-2008_CAL"])))
            sd=max(sd,1e-3)
            ytrainhat=mdl.predict(fs["train"])
            # Posterior candidates are actual coded/training states; for raw
            # baseline use real Z, otherwise use accurately decoded q(Z).
            candidate_z=raw["train"] if representation=="raw_Z" else decoded["train"]
            candidate_models[key]=(ytrainhat,candidate_z,sd)
            for tag in tests:
                pred=mdl.predict(fs[tag])
                fw.append(dict(representation=representation,model=mname,test_period=tag,
                      n_train=len(tr),n_test=len(tests[tag]),weighted_Y_R2=wr2(Y[tag],pred,W[tag]),
                      weighted_Y_RMSE=float(np.sqrt(np.average((Y[tag]-pred)**2,weights=W[tag]))),
                      calibrated_likelihood_sd=sd))
            print(f"  {key}: calibrated likelihood SD={sd:.3f}")

    prior=W["train"]/np.sum(W["train"])
    for key,(candidate_y,candidate_z,sd) in candidate_models.items():
        for tag in tests:
            obs=Y[tag]
            mean,point,lo,hi=posterior_batch(obs,candidate_y,candidate_z,prior,sd,
                    interval_n=1 if args.intervals else 0)
            for variant,pred in (("posterior_mean",mean),("posterior_MAP_candidate",point)):
                r,rm=zwscore(raw[tag],pred,W[tag])
                cover=np.nan; width=np.nan
                if variant=="posterior_mean" and args.intervals:
                    inc=(raw[tag]>=lo)&(raw[tag]<=hi)
                    cover=float(np.average(inc.mean(axis=1),weights=W[tag]))
                    width=float(np.average((hi-lo).mean(axis=1),weights=W[tag]))
                reverse.append(dict(method=key,test_period=tag,reverse_variant=variant,
                      weighted_Z_R2=r,weighted_Z_RMSE=rm,
                      marginal_90_coverage=cover,marginal_90_mean_width=width))

    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(fw).to_csv(args.out/"93_forward_digit_representation.csv",index=False)
    pd.DataFrame(reverse).to_csv(args.out/"93_reverse_codebook_posterior.csv",index=False)
    pd.DataFrame(audits).to_csv(args.out/"93_encoding_integrity_audit.csv",index=False)
    (args.out/"93_run_manifest.json").write_text(json.dumps(dict(
        provenance="real NHANES-derived frozen Script 73 observations ONLY",
        training_cycle="0506",calibration_cycle="0708",test_periods={tag:len(t) for tag,t in tests.items()},
        components=comps,outcome=args.outcome,decimals=args.decimals,decimal_width=args.width,
        n_training=len(tr),calibration_n=len(cal),
        flags=["No synthetic fallback", "No modifications to frozen Script 72-75 artifacts",
               "Forward scores omit covariate adjustment; not comparable to adjusted Script 73 regressions",
               "Calibrated sigma is evaluated on 2007-08 and reused for 2009-18/2021-23",
               "Posterior uses empirical discrete 2005-06 codebook with NHANES survey-weight proxy prior",
               "Gaussian phenotype likelihood is an assumption, not verified clinical law",
               "Posterior marginal 90% intervals are not conformal or guaranteed calibrated",
               "A code cannot restore lost inverse information or identify personal biological causation",
               "Prime-exponent/two-adic factorized fields are reversible coordinate encodings and provide no extra information"]
    ),indent=2),encoding="utf-8")

    fdf=pd.DataFrame(fw)
    print("\nFORWARD weighted Y R2 (best model for each representation; later held-out cycles)")
    later=fdf[fdf.test_period.isin(["2009-2018","2021-2023"])]
    best=(later.sort_values("weighted_Y_R2",ascending=False)
          .drop_duplicates(["representation","test_period"]))
    print(best.pivot(index="representation",columns="test_period",values="weighted_Y_R2").round(4).to_string())
    rdf=pd.DataFrame(reverse)
    print("\nREVERSE weighted Z R2 (2009-18; posterior mean vs direct ridge)")
    rr=rdf[rdf.test_period.eq("2009-2018") & rdf.reverse_variant.isin(["posterior_mean","conditional_mean"])]
    print(rr[["method","weighted_Z_R2","weighted_Z_RMSE"]].round(4).to_string(index=False))
    print("\nSaved EDA/Results/93_*; audit learning, quantization and reverse uncertainty separately.")

if __name__=="__main__":
    main()
