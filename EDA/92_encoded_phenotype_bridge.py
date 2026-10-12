#!/usr/bin/env python3
"""Script 92: Prof. Saeed's encoding -> Yhat -> candidate-Z bridge.

Real input ONLY: EDA/Results/73_frozen_phenotype_map_input.csv.
Do not overwrite frozen decomposition. No synthetic fallback.

Compare 4 *reversible* integer encoders: Cantor, prime exponent, 2-adic,
decimal interleave. Train a phenotype predictor h(code) and compare on held-out
NHANES cycles to the SAME-input raw-Z regression and nonlinear baseline.
Then search the 2005-06 encoded-state codebook using h(code) and held-out Y.

IMPORTANT: Very long integer codes are represented to the learning model by
LIMITED high/low bytes + bit-length. The code itself is reversible, but these
learning features may NOT be. The output audits this explicitly.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
ALL_Z = ["SP_SHARED1", "SP_SHARED2", "SP_DISCORD1", "SP_DISCORD2",
         "SP_A_PRIVATE3", "SP_A_PRIVATE4", "SP_G_PRIVATE3"]
PRIMES = (2, 3, 5, 7, 11, 13, 17)
METHODS = ("decimal_interleave", "cantor", "two_adic", "prime_exponents")
PERIODS = [("2007-2008", "0708"), ("2009-2018", None), ("2021-2023", None)]


def qint(x: float, places: int) -> int:
    return int((Decimal(str(float(x))) * (10 ** places))
               .to_integral_value(rounding=ROUND_HALF_EVEN))


def unsigned(n: int) -> int:
    return 2*n if n >= 0 else -2*n-1


def signed(n: int) -> int:
    return n//2 if n % 2 == 0 else -(n+1)//2


def pair(a: int, b: int) -> int:
    t=a+b
    return t*(t+1)//2+b


def unpair(c: int) -> tuple[int,int]:
    t=(math.isqrt(8*c+1)-1)//2
    b=c-t*(t+1)//2
    return t-b,b


def cantor(vals: tuple[int,...]) -> int:
    out=vals[0]
    for v in vals[1:]:
        out=pair(out,v)
    return out


def uncantor(c: int, n: int) -> tuple[int,...]:
    vs=[]
    for _ in range(n-1):
        c,v=unpair(c)
        vs.append(v)
    vs.append(c)
    return tuple(reversed(vs))


def encode(method: str, vals: tuple[int,...], width: int, max_bits: int):
    """Exact code object; fail explicitly if impractically large."""
    if method == "decimal_interleave":
        a=[str(v).zfill(width) for v in vals]
        if any(len(v)!=width for v in a):
            raise ValueError("Decimal width exceeded; increase --width")
        return "".join(v[i] for i in range(width) for v in a)
    if method == "cantor":
        # Bound on length grows ~2^(n-1) with successive pairing.
        x=vals[0]
        for v in vals[1:]:
            if 2*max(x.bit_length(),v.bit_length())+3 > max_bits:
                raise OverflowError("Cantor integer exceeds --max-bits")
            x=pair(x,v)
        return x
    if method == "two_adic":
        if len(vals)!=2: raise ValueError("2-adic method here defined for exactly 2 dimensions")
        a,b=vals
        if a+(2*b+1).bit_length()>max_bits:
            raise OverflowError("2-adic integer exceeds --max-bits")
        return (2*b+1)<<a
    if method == "prime_exponents":
        if len(vals)>len(PRIMES): raise ValueError("Insufficient primes")
        estimated=sum(v*math.log2(PRIMES[i]) for i,v in enumerate(vals))
        if estimated+2 > max_bits:
            raise OverflowError("Prime integer exceeds --max-bits")
        ans=1
        for p,exp in zip(PRIMES,vals):
            ans*=pow(p,exp)
        return ans
    raise ValueError(method)


def decode(method: str, code, count: int, width: int) -> tuple[int,...]:
    if method=="decimal_interleave":
        if len(code)!=count*width: raise ValueError("Bad code length")
        return tuple(int(code[j::count]) for j in range(count))
    if method=="cantor":return uncantor(code,count)
    if method=="two_adic":
        if count!=2: raise ValueError("Expected 2 dims")
        n=(code&-code).bit_length()-1
        return n,((code>>n)-1)//2
    if method=="prime_exponents":
        orig=code
        powers=[]
        for prime in PRIMES[:count]:
            e=0
            while orig%prime==0:
                orig//=prime
                e+=1
            powers.append(e)
        if orig!=1: raise AssertionError("Unrecognized prime factor")
        return tuple(powers)
    raise ValueError(method)


def code_features(method: str, code, width: int, n: int, flank_bytes: int):
    """Model inputs derived ONLY from c (not from separately supplied Z).

    Decimal: all interleaved digits, exactly.
    Integer: bit length + leading and trailing fixed-size byte windows.
    Audit indicates when middle bits are omitted.
    """
    if method=="decimal_interleave":
        return np.fromiter((ord(ch)-48 for ch in code),dtype=np.float64,count=len(code)), False, len(code)
    bl=max(1,code.bit_length())
    shift=max(0,bl-8*flank_bytes)
    high=code>>shift
    mask=(1<<(8*flank_bytes))-1
    low=code&mask
    high_bytes=[(high>>(8*i))&255 for i in range(flank_bytes-1,-1,-1)]
    low_bytes=[(low>>(8*i))&255 for i in range(flank_bytes-1,-1,-1)]
    # All original integer bits are visible if bit_length <= 2*flank_bytes*8.
    truncated=bl>16*flank_bytes
    return np.asarray([math.log2(bl)]+high_bytes+low_bytes,dtype=np.float64),truncated,bl


def load_data(path: Path, components: list[str], outcome: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Script 73 real observations: {path}")
    hdr=set(pd.read_csv(path,nrows=0).columns)
    cols=["SEQN","CYCLE","PERIOD","SURVEY_WT","ELIGIBLE_ADULT_NONPREG",outcome]+components
    absent=sorted(set(cols)-hdr)
    if absent: raise ValueError(f"Missing columns in Script 73: {absent}")
    d=pd.read_csv(path,usecols=list(dict.fromkeys(cols)),low_memory=False)
    for k in components+[outcome,"SURVEY_WT"]:
        d[k]=pd.to_numeric(d[k],errors="coerce")
    d["CYCLE"]=(d["CYCLE"].astype(str).str.strip()
                .str.replace(r"\.0$","",regex=True).str.zfill(4))
    d=d[d["ELIGIBLE_ADULT_NONPREG"].eq(1)]
    d=d[d[components+[outcome,"SURVEY_WT"]].notna().all(axis=1) & d["SURVEY_WT"].gt(0)].copy()
    return d


def wmse(y, yhat, wt):
    return float(np.average((np.asarray(y)-np.asarray(yhat))**2,weights=wt))


def wr2(y, yhat, wt):
    den=float(np.average((y-np.average(y,weights=wt))**2,weights=wt))
    return 1-wmse(y,yhat,wt)/den if den>0 else float('nan')


def z_w_r2(z, zhat, wt):
    center=np.average(z,axis=0,weights=wt)
    sse=float(np.sum(wt[:,None]*(z-zhat)**2))
    sst=float(np.sum(wt[:,None]*(z-center)**2))
    return 1-sse/sst if sst>0 else float('nan')


def new_model(kind: str):
    if kind=="ridge": return make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    if kind=="hgb":
        return HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=12,
                   learning_rate=0.05,l2_regularization=3.0,
                   min_samples_leaf=35,random_state=92)
    raise ValueError(kind)


def weights(d:pd.DataFrame):
    w=d["SURVEY_WT"].to_numpy(float)
    return w/w.mean()


def rank_candidates(y_test:np.ndarray, predicted_y_train:np.ndarray,
                    train_z:np.ndarray, train_w:np.ndarray, k:int):
    """Return top-1 decoded codebook Z and top-k weighted candidate mean."""
    idx=np.argsort(predicted_y_train,kind="mergesort")
    s=predicted_y_train[idx]
    top1=np.empty((len(y_test),train_z.shape[1]))
    topk=np.empty_like(top1)
    for i,y in enumerate(y_test):
        cut=np.searchsorted(s,y)
        left=max(0,cut-k)
        right=min(len(s),cut+k)
        selection=idx[left:right]
        ranked=selection[np.argsort(np.abs(predicted_y_train[selection]-y),kind="stable")[:k]]
        if len(ranked)==0: raise AssertionError("Empty candidate codebook")
        top1[i]=train_z[ranked[0]]
        topk[i]=np.average(train_z[ranked],axis=0,weights=train_w[ranked])
    return top1,topk


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path,default=ROOT/"Results"/"73_frozen_phenotype_map_input.csv")
    ap.add_argument("--out",type=Path,default=ROOT/"Results")
    ap.add_argument("--components",default="SP_SHARED1,SP_A_PRIVATE3",help="Comma-separated frozen Z columns")
    ap.add_argument("--all7",action="store_true",help="Use seven frozen components; non-pair codings may be infeasible")
    ap.add_argument("--outcome",default="PHQ9_TOTAL",choices=["PHQ9_TOTAL","SOMATIC_SCORE","COGAFF_SUM"])
    ap.add_argument("--decimals",type=int,default=3)
    ap.add_argument("--width",type=int,default=10,help="Fixed decimal width per unsigned integer")
    ap.add_argument("--flank-bytes",type=int,default=24,help="High and low integer-code bytes supplied to h(code)")
    ap.add_argument("--max-bits",type=int,default=100000,help="Skip methods with unmanageably long codes")
    ap.add_argument("--k",type=int,default=15,help="Number of training codes averaged for reverse candidates")
    ap.add_argument("--max-train",type=int,default=None,help="Optional development-only limit")
    args=ap.parse_args()
    comps=ALL_Z if args.all7 else [x.strip() for x in args.components.split(",") if x.strip()]
    if len(comps)<2 or len(set(comps))!=len(comps) or any(x not in ALL_Z for x in comps):
        ap.error("Choose >=2 distinct columns from the seven frozen Z values")
    if not (0<=args.decimals<=6 and 1<=args.width<=18 and 4<=args.flank_bytes<=256 and args.k>=1):
        ap.error("Bad numeric argument")
    d=load_data(args.data,comps,args.outcome)
    tr=d[d["CYCLE"].eq("0506")].copy()
    if args.max_train is not None: tr=tr.head(args.max_train)
    if len(tr)<100: raise ValueError("Need at least 100 complete 2005–06 rows")
    tests={tag:(d[d["CYCLE"].eq(cy)].copy() if cy else d[d["PERIOD"].eq(tag)].copy())
           for tag,cy in PERIODS}
    tests={k:v for k,v in tests.items() if len(v)>0}
    if not tests: raise ValueError("No held-out cycles found")

    # Freeze train/test selection before any feature construction.
    ytr=tr[args.outcome].to_numpy(float)
    ztr=tr[comps].to_numpy(float)
    wtr=weights(tr)
    forward=[]
    reverse=[]
    audit=[]
    candidate_models={}
    ysets={tag:te[args.outcome].to_numpy(float) for tag,te in tests.items()}
    zsets={tag:te[comps].to_numpy(float) for tag,te in tests.items()}
    wsets={tag:weights(te) for tag,te in tests.items()}
    print(f"SCRIPT 92 | REAL Script-73 data | components={len(comps)} | train 2005-06 n={len(tr)}")
    print("Test periods:",", ".join(f"{tag} n={len(te)}" for tag,te in tests.items()))

    # Reference inverse regression Y->Z, fit only on discovery.
    rev=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    rev.fit(ytr.reshape(-1,1),ztr,ridge__sample_weight=wtr)
    for tag in tests:
        zhat=rev.predict(ysets[tag].reshape(-1,1))
        reverse.append(dict(method="ridge_Y_to_Z",test_period=tag,model="ridge",
              candidates=0,weighted_Z_R2=z_w_r2(zsets[tag],zhat,wsets[tag]),
              weighted_Z_RMSE=float(np.sqrt(np.average(np.mean((zsets[tag]-zhat)**2,axis=1),weights=wsets[tag])))))

    # Same-dimensional raw-Z baselines ensure apples-to-apples forward comparisons.
    zbaseline={}
    for model_name in ("ridge","hgb"):
        mod=new_model(model_name)
        if model_name=="ridge": mod.fit(ztr,ytr,ridge__sample_weight=wtr)
        else: mod.fit(ztr,ytr,sample_weight=wtr)
        yhat_tr=mod.predict(ztr)
        zbaseline[model_name]=mod
        candidate_models[f"raw_Z_{model_name}"]=(yhat_tr,ztr)
        for tag in tests:
            yh=mod.predict(zsets[tag])
            forward.append(dict(method="raw_Z",model=model_name,test_period=tag,
                        train_n=len(tr),test_n=len(tests[tag]),weighted_Y_R2=wr2(ysets[tag],yh,wsets[tag]),
                        weighted_Y_RMSE=math.sqrt(wmse(ysets[tag],yh,wsets[tag]))))

    sets={"train":tr,**tests}
    state_sets={name:[tuple(unsigned(qint(v,args.decimals)) for v in zz)
                      for zz in subset[comps].to_numpy(float)]
                for name,subset in sets.items()}
    for method in METHODS:
        if method=="two_adic" and len(comps)!=2:
            audit.append(dict(method=method,status="SKIP: 2-adic encoder implemented for exactly 2 values"))
            continue
        try:
            feature_sets={}
            code_sets={}
            code_bits={}
            truncations=[]
            all_codes=[]
            checks=0
            max_quant_error=0.0
            for tag, subset in sets.items():
                feats=[]
                codes=[]
                for i,(vals,z) in enumerate(zip(state_sets[tag],subset[comps].to_numpy(float))):
                    code=encode(method,vals,args.width,args.max_bits)
                    feat,is_truncated,bits=code_features(method,code,args.width,len(comps),args.flank_bytes)
                    feats.append(feat)
                    codes.append(code)
                    truncations.append(is_truncated)
                    code_bits.setdefault(tag,[]).append(bits)
                    reconstructed=np.array([signed(u)/10**args.decimals for u in vals])
                    error=float(np.max(np.abs(reconstructed-z)))
                    max_quant_error=max(max_quant_error,error)
                    if i<12 or i==len(state_sets[tag])-1:
                        if decode(method,code,len(comps),args.width)!=vals:
                            raise AssertionError(f"{method} roundtrip failure at {tag}:{i}")
                        checks+=1
                feature_sets[tag]=np.vstack(feats)
                code_sets[tag]=codes
                all_codes.extend(codes)
            # Candidate physiology returned by the codebook is the quantized,
            # reversibly encoded Z, not the original full-precision Z.
            ztrain_decoded=np.asarray([[signed(v)/(10**args.decimals) for v in state]
                                      for state in state_sets["train"]],dtype=float)
            tr_feat=feature_sets["train"]
            # This tests whether same code has been assigned to distinct QUANTIZED states.
            key_to_state={}
            collisions=0
            for tag in sets:
                for code,vals in zip(code_sets[tag],state_sets[tag]):
                    if code in key_to_state and key_to_state[code]!=vals:
                        collisions+=1
                    else:key_to_state[code]=vals
            if collisions: raise AssertionError(f"{method} has coding collisions")
            audit.append(dict(method=method,status="OK",train_rows=len(tr),
                 total_rows_encoded=sum(len(x) for x in sets.values()),
                 audited_roundtrips=checks,quantization_decimals=args.decimals,
                 max_abs_Z_rounding_error=max_quant_error,
                 distinct_train_codes=len(set(code_sets["train"])),
                 code_collisions_between_different_quantized_states=collisions,
                 mean_train_code_size_bits_or_digits=float(np.mean(code_bits["train"])),
                 max_code_size_bits_or_digits=int(max(max(x) for x in code_bits.values())),
                 h_feature_dimension=tr_feat.shape[1],
                 fraction_h_feature_vectors_missing_middle_code_bits=float(np.mean(truncations))))
            for model_name in ("ridge","hgb"):
                mod=new_model(model_name)
                if model_name=="ridge":mod.fit(tr_feat,ytr,ridge__sample_weight=wtr)
                else:mod.fit(tr_feat,ytr,sample_weight=wtr)
                yhtrain=mod.predict(tr_feat)
                candidate_models[f"{method}_{model_name}"]=(yhtrain,ztrain_decoded)
                for tag in tests:
                    yh=mod.predict(feature_sets[tag])
                    forward.append(dict(method=method,model=model_name,test_period=tag,
                              train_n=len(tr),test_n=len(tests[tag]),weighted_Y_R2=wr2(ysets[tag],yh,wsets[tag]),
                              weighted_Y_RMSE=math.sqrt(wmse(ysets[tag],yh,wsets[tag]))))
            print(f"  {method}: OK; maximum code size={audit[-1]['max_code_size_bits_or_digits']}, "
                  f"truncated middle-bit features={audit[-1]['fraction_h_feature_vectors_missing_middle_code_bits']:.1%}")
        except (OverflowError,ValueError,MemoryError) as exc:
            audit.append(dict(method=method,status=f"SKIPPED (feasibility): {type(exc).__name__}: {exc}"))
            print(f"  {method}: SKIPPED ({exc})")
            continue

    # Reverse comparison: evaluate each learned forward model's predictions as an
    # index over training candidate states. No test physiology enters codebook.
    for method,(yhtrain,zcode) in candidate_models.items():
        for tag in tests:
            z1,zk=rank_candidates(ysets[tag],yhtrain,zcode,wtr,min(args.k,len(tr)))
            for klabel,zh in (("top1",z1),(f"top{args.k}_weighted_mean",zk)):
                reverse.append(dict(method=method,test_period=tag,model=klabel,
                    candidates=min(args.k,len(tr)),weighted_Z_R2=z_w_r2(zsets[tag],zh,wsets[tag]),
                    weighted_Z_RMSE=float(np.sqrt(np.average(np.mean((zsets[tag]-zh)**2,axis=1),weights=wsets[tag])))))

    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(audit).to_csv(args.out/"92_encoding_feasibility_audit.csv",index=False)
    pd.DataFrame(forward).to_csv(args.out/"92_forward_Yhat_comparison.csv",index=False)
    pd.DataFrame(reverse).to_csv(args.out/"92_reverse_candidate_Z_comparison.csv",index=False)
    report=dict(provenance="REAL NHANES-derived frozen Script 73 Z and observed PHQ Y; no synthetic fallback",
        train_cycle="0506",test_periods={k:len(v) for k,v in tests.items()},n_train=len(tr),
        components=comps,outcome=args.outcome,decimals=args.decimals,
        q_error_bound=.5*10**(-args.decimals),
        important_limits=["No demographic covariates; this is NOT a refit of Script 73's adjusted regression.",
          "Code integer is reversible at stated quantization, but integer-code model features may omit middle bits: check 92 audit.",
          "Reverse method searches only observed training-codebook states, not a learned mathematical inverse.",
          "Top-k weighted mean Z may not itself correspond to one valid encoded state.",
          "No valid inference of an individual's physiological cause from PHQ-9 alone.",
          "Not a causal model, no within-person longitudinal inference."])
    (args.out/"92_run_manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    fw=pd.DataFrame(forward)
    if fw.empty: raise RuntimeError("No models completed")
    print("\nFORWARD YHAT (weighted Y R2; negative means worse than mean baseline)")
    print(fw.pivot_table(index=["method","model"],columns="test_period",values="weighted_Y_R2").round(4).to_string())
    rv=pd.DataFrame(reverse)
    print("\nREVERSE Z (weighted Z R2, summarized for later cycles)")
    print(rv[rv.test_period.eq("2009-2018")][["method","model","weighted_Z_R2"]].round(4).to_string(index=False))
    print(f"\nResults saved: {args.out.resolve()}")


if __name__=="__main__":
    main()
