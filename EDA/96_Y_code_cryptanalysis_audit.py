#!/usr/bin/env python3
"""Script 96: Y-centred exhaustive coding and cryptanalytic reverse-bridge audit.

Place in EDA/ of auniyal26/anemic-glycemic-burden-decomposition and run:
    python EDA/96_Y_code_cryptanalysis_audit.py

Source: REAL NHANES Script-73 output and original cycle DPQ XPTs, as Script 95.
Train 2005-06 ONLY; evaluate separately on 2007-08, 2009-18, 2021-23.
Never edit frozen artefacts. No synthetic-data fallback, no Git actions.

Part I: enumerate ALL nonempty subsets of nine PHQ items, seven frozen Z
coordinates, and seven raw X biomarkers (4095+127+127=4349 subsets, including
PHQ total/somatic/cognitive-affective and item combinations). Audit four
coding families for each subset (17396 statuses). Exact inverses apply only to
quantized data. Two-adic coding is defined for PAIRS ONLY and records skips.

Part II: Learn Y -> X and Y -> Z from observed training pairs only. Compare
prior, ridge, exact-score keys, and progressive n-gram key-frequency decoding.
A progressive decoder uses phenotype item singles, pairs, and triples (the
frequency-analysis/cryptanalysis analogy). Test X and Z NEVER enter inference.

Part III: Compare raw versus fully represented encoded Y features for the four
coding families on total score, somatic/cognitive-affective pair and all items.
This assesses representation usefulness, not added information.

Strict noncausal interpretation. This script does not perform intervention
identification, biological disease-subtype validation or person-level followup.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EDA = Path(__file__).resolve().parent
ROOT = EDA.parent
RESULTS = EDA / "Results"
SOURCE = RESULTS / "73_frozen_phenotype_map_input.csv"
OUTPUT = RESULTS
A = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
G = ["LBXGH", "LBXGLU", "LOG_IN"]
X = A + G
Z = ["SP_SHARED1", "SP_SHARED2", "SP_DISCORD1", "SP_DISCORD2",
     "SP_A_PRIVATE3", "SP_A_PRIVATE4", "SP_G_PRIVATE3"]
YITEMS = [f"DPQ0{i}0" for i in range(1,10)]
BASIC_Y = ["PHQ9_TOTAL", "SOMATIC_SCORE", "COGAFF_SUM"]
SOMATIC_ITEMS = ["DPQ030", "DPQ040", "DPQ050"]  # Canonical split used in Script 81.
METHODS = ("decimal_interleave", "cantor", "prime_exponents", "two_adic")
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
CYCLE_SUFFIX = {"0506":"D","0708":"E","0910":"F","1112":"G",
                "1314":"H","1516":"I","1718":"J","2123":"L"}
PERIODS = {"0506":"train_2005-2006", "0708":"2007-2008",
           "0910":"2009-2018", "1112":"2009-2018", "1314":"2009-2018",
           "1516":"2009-2018", "1718":"2009-2018", "2123":"2021-2023"}
SEED = 96


def unsigned(v: int) -> int:
    return 2*v if v >= 0 else -2*v-1


def signed(v: int) -> int:
    return v//2 if v%2==0 else -(v+1)//2


def cantor_pair(a: int,b: int) -> int:
    s=a+b
    return s*(s+1)//2+b


def cantor_unpair(z: int) -> tuple[int,int]:
    s=(math.isqrt(8*z+1)-1)//2
    b=z-s*(s+1)//2
    return s-b,b


def encode(vals: tuple[int,...], method: str, width: int, max_bits: int):
    if not vals: raise ValueError("Cannot encode an empty vector")
    if any(v < 0 for v in vals): raise ValueError("Input to codec must be unsigned")
    if method=="decimal_interleave":
        seq=[str(v).zfill(width) for v in vals]
        if any(len(v)!=width for v in seq): raise OverflowError("decimal_width_exceeded")
        return "".join(q[i] for i in range(width) for q in seq)
    if method=="cantor":
        cur=vals[0]
        for v in vals[1:]:
            if 2*max(cur.bit_length(),v.bit_length())+3>max_bits:
                raise OverflowError("cantor_bit_limit")
            cur=cantor_pair(cur,v)
        return cur
    if method=="prime_exponents":
        if len(vals)>len(PRIMES): raise ValueError("insufficient_primes")
        bits=sum(v*math.log2(PRIMES[i]) for i,v in enumerate(vals))
        if bits > max_bits-2: raise OverflowError("prime_bit_limit")
        r=1
        for p,v in zip(PRIMES,vals): r*=pow(p,v)
        return r
    if method=="two_adic":
        if len(vals)!=2: raise ValueError("two_adic_requires_pair")
        a,b=vals
        if a+(2*b+1).bit_length()>max_bits:
            raise OverflowError("two_adic_bit_limit")
        return (2*b+1)<<a
    raise ValueError(method)


def decode(code, method: str, n: int, width: int) -> tuple[int,...]:
    if method=="decimal_interleave":
        if len(code)!=n*width or not code.isdigit(): raise ValueError("bad_decimal_code")
        return tuple(int(code[i::n]) for i in range(n))
    if method=="cantor":
        cur=int(code); out=[]
        for _ in range(n-1):
            cur,v=cantor_unpair(cur);out.append(v)
        return tuple(reversed(out+[cur]))
    if method=="prime_exponents":
        cur=int(code); out=[]
        for p in PRIMES[:n]:
            e=0
            while cur%p==0:
                cur//=p;e+=1
            out.append(e)
        if cur!=1: raise ValueError("invalid_prime_code")
        return tuple(out)
    if method=="two_adic":
        if n!=2:raise ValueError("two_adic_requires_pair")
        v=int(code)
        if v<=0: raise ValueError("bad_two_adic")
        a=(v & -v).bit_length()-1
        return a,(v//(1<<a)-1)//2
    raise ValueError(method)


def subsets(columns):
    for k in range(1,len(columns)+1):
        yield from itertools.combinations(columns,k)


def q_values(frame: pd.DataFrame, cols: tuple[str,...], decimals: int):
    # np.rint uses banker's rounding; ties are representationally declared.
    vals=frame[list(cols)].to_numpy(float)
    if not np.isfinite(vals).all(): raise ValueError("nonfinite_values")
    if np.max(np.abs(vals)) > 1e9: raise OverflowError("unsafe_raw_input")
    q=np.rint(vals*(10**decimals)).astype(np.int64)
    return np.where(q>=0,2*q,-2*q-1).astype(np.int64)


def audit_codes(d: pd.DataFrame, max_rows: int, max_bits: int):
    # TRAIN ONLY for the exhaustive numerical integrity/feasibility audit.
    # Infer the decimal width from the whole train subset, not the test periods.
    frames={"Y":(BASIC_Y+YITEMS,0),"Z":(Z,3),"X":(X,3)}
    rng=np.random.default_rng(SEED)
    sample=d[d.CYCLE.eq("0506")].copy()
    if len(sample)>max_rows:
        sample=sample.iloc[np.sort(rng.choice(len(sample),max_rows,replace=False))]
    results=[]
    for domain,(columns,precision) in frames.items():
        for ss in subsets(columns):
            sub=sample.loc[sample[list(ss)].notna().all(axis=1),list(ss)]
            if sub.empty:
                for method in METHODS:
                    results.append(dict(domain=domain,fields="|".join(ss),dimensions=len(ss),
                        method=method,status="no_complete_train_rows",tested_rows=0))
                continue
            q=q_values(sub,ss,precision)
            width=max(1,max(len(str(int(v))) for v in q.flat))
            rows=[tuple(map(int,row)) for row in q]
            for method in METHODS:
                row=dict(domain=domain,fields="|".join(ss),dimensions=len(ss),
                         decimals=precision,method=method,tested_rows=len(rows),
                         quantized_unique_states=len(set(rows)),decimal_width=width)
                if method=="two_adic" and len(ss)!=2:
                    row.update(status="not_defined_nonpair",roundtrip_failures=None)
                    results.append(row);continue
                try:
                    codes=[]; failures=0; maxsize=0
                    for v in rows:
                        c=encode(v,method,width,max_bits)
                        dec=decode(c,method,len(ss),width)
                        failures+=int(dec!=v)
                        codes.append(c)
                        maxsize=max(maxsize,len(c) if isinstance(c,str) else c.bit_length())
                    row.update(status="ok" if failures==0 else "failed_roundtrip",
                               roundtrip_failures=failures,
                               distinct_codes=len(set(codes)),
                               quantized_code_collisions=len(set(rows))-len(set(codes)),
                               maximum_code_digits_or_bits=maxsize)
                except (OverflowError,ValueError,MemoryError) as e:
                    row.update(status="not_feasible:"+str(e),roundtrip_failures=None)
                results.append(row)
    return pd.DataFrame(results)


def dpq_location(cycle: str) -> Path:
    parent="NHANES" if cycle in ("0506","0708") else "NHANES_Transfer"
    folder=ROOT/"Data"/parent/cycle
    stem=f"DPQ_{CYCLE_SUFFIX[cycle]}"
    for ext in (".XPT",".xpt"):
        p=folder/(stem+ext)
        if p.is_file():return p
    raise FileNotFoundError(f"Original {stem}.XPT missing from {folder}; see Script 95. Never infer PHQ items from totals.")


def load():
    """Build authoritative PHQ/Z cohort; let X completeness vary by target.

    The previous implementation required ALL raw X, Z and stored phenotype
    sub-scores to be observed in EVERY period. A single sparse X variable (or
    unavailable old sub-score) made the whole experiment fail even though valid
    Y->Z data were present. Do not impute: measure target-specific support.
    """
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing Script 73 input: {SOURCE}")
    head=pd.read_csv(SOURCE,nrows=0).columns.tolist()
    required=["SEQN","CYCLE","SURVEY_WT","ELIGIBLE_ADULT_NONPREG","PHQ9_TOTAL"]+Z
    miss=sorted(set(required)-set(head))
    if miss:
        raise ValueError(f"Script-73 required columns absent: {miss}")
    # Some previously generated input versions lack stored PHQ sub-scores or
    # complete raw X fields. Original item data are authoritative for Y.
    selected=list(dict.fromkeys(required+[c for c in BASIC_Y+X if c in head]))
    d=pd.read_csv(SOURCE,usecols=selected,low_memory=False)
    for col in X+BASIC_Y:
        if col not in d:
            d[col]=np.nan
    d.CYCLE=(d.CYCLE.astype(str).str.strip()
             .str.replace(r"\.0$","",regex=True).str.zfill(4))
    d=d[d.CYCLE.isin(CYCLE_SUFFIX)].copy()
    d.SEQN=pd.to_numeric(d.SEQN,errors="coerce")
    for col in list(dict.fromkeys(["SURVEY_WT"]+BASIC_Y+X+Z)):
        d[col]=pd.to_numeric(d[col],errors="coerce")
    d=d[d.ELIGIBLE_ADULT_NONPREG.eq(1)&d.SURVEY_WT.gt(0)].copy()
    if d.SEQN.duplicated().any():
        raise AssertionError("Duplicate subject IDs in frozen input")
    items=[]
    for cyc in CYCLE_SUFFIX:
        q=pd.read_sas(dpq_location(cyc),format="xport")
        required_items=["SEQN"]+YITEMS
        if not set(required_items).issubset(q.columns):
            raise ValueError(f"Missing item columns in cycle {cyc}")
        q=q[required_items].copy()
        q.SEQN=pd.to_numeric(q.SEQN,errors="coerce")
        for item in YITEMS:
            raw_item = pd.to_numeric(q[item], errors="coerce")
            # CRITICAL NHANES XPT detail: SAS numeric zeros can be decoded by
            # pandas as tiny, nonzero floating-point numbers (~5.4e-79).
            # The verified Script-81/95 loaders normalize those near-zeros
            # BEFORE enforcing the discrete PHQ item domain 0, 1, 2, 3.
            # Without this, almost all valid participants are discarded.
            raw_item = raw_item.mask(raw_item.abs() < 1e-10, 0.0)
            q[item] = raw_item.where(raw_item.isin([0, 1, 2, 3]))
        items.append(q)
    y=pd.concat(items,ignore_index=True)
    if y.SEQN.duplicated().any():
        raise AssertionError("Duplicate raw PHQ subject IDs")
    d=d.merge(y,on="SEQN",how="left",validate="one_to_one")
    d["SPLIT"]=d.CYCLE.map(PERIODS)

    valid=d[YITEMS].notna().all(axis=1)&d[Z].notna().all(axis=1)&d.PHQ9_TOTAL.notna()
    checks=d.loc[valid].copy()
    mismatch=~np.isclose(checks[YITEMS].sum(axis=1),checks.PHQ9_TOTAL)
    if mismatch.any():
        raise AssertionError(f"PHQ total/item mismatch: {int(mismatch.sum())}")
    # Construct PHQ sub-scores from the original answers using the frozen
    # Script-81 item definitions, preserving the standard whole-score identity.
    somatic=checks[SOMATIC_ITEMS].sum(axis=1)
    cogaff=checks.PHQ9_TOTAL-somatic
    for col,derived in [("SOMATIC_SCORE",somatic),("COGAFF_SUM",cogaff)]:
        old=checks[col]
        disagree=old.notna() & ~np.isclose(old.fillna(derived),derived)
        if disagree.any():
            raise AssertionError(f"Raw-DPQ {col} disagrees with stored score for {int(disagree.sum())} participants")
        checks[col]=derived
    d.loc[checks.index,["SOMATIC_SCORE","COGAFF_SUM"]]=checks[["SOMATIC_SCORE","COGAFF_SUM"]]

    # Write a coverage audit EVEN when no period supports full X.
    coverage=[]
    for p in dict.fromkeys(PERIODS.values()):
        section=d[d.SPLIT.eq(p)]
        yz=section.loc[valid.loc[section.index]]
        coverage.append(dict(period=p,eligible_with_weight=len(section),
            complete_PHQ_items_Z=len(yz),complete_A=int(yz[A].notna().all(axis=1).sum()),
            complete_G=int(yz[G].notna().all(axis=1).sum()),
            complete_X_all7=int(yz[X].notna().all(axis=1).sum()),
            complete_stored_somatic=int(section.SOMATIC_SCORE.notna().sum()),
            complete_stored_cogaff=int(section.COGAFF_SUM.notna().sum())))
    cov=pd.DataFrame(coverage)
    cov.to_csv(OUTPUT/"96_cohort_coverage.csv",index=False)
    print("\n96 COVERAGE AUDIT (before target-specific filtering):")
    print(cov.to_string(index=False))
    final=d.loc[valid].copy()
    n=final.groupby("SPLIT").size().to_dict()
    weak={p:n.get(p,0) for p in dict.fromkeys(PERIODS.values()) if n.get(p,0)<50}
    if weak:
        raise ValueError(f"Fewer than 50 genuine complete PHQ-item/Z rows in periods {weak}; "
                         "inspect EDA/Results/96_cohort_coverage.csv")
    return final


def target_cohort(df: pd.DataFrame, split: str, target: list[str]):
    """No physiology imputation; matched complete target rows for every method."""
    rows=df[df.SPLIT.eq(split)]
    return rows.loc[rows[target].notna().all(axis=1)].copy()


def weighted_mean(v,w):
    return np.average(v,weights=w,axis=0)


def standardized_score(truth,pred,w,train_mu,train_sd):
    # Evaluates all dimensions equally on discovery scaling. Never refit test scaler.
    truth=(truth-train_mu)/train_sd
    pred=(pred-train_mu)/train_sd
    m=np.average(truth,weights=w,axis=0)
    den=np.average((truth-m)**2,weights=w,axis=0)
    mse=np.average((truth-pred)**2,weights=w,axis=0)
    return 1.-mse/np.maximum(den,1e-12),mse


def train_ridge(yy, t, w):
    # Fit separate outputs in one multitarget regression (same coefficients as per-output fits).
    m=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    m.fit(yy,t,ridge__sample_weight=w)
    return m


def key_dictionary(train, columns, target, w):
    # Learned 'key': phenotype token pattern -> frequency, weighted X/Z signature.
    dic={}
    x=train[list(columns)].to_numpy(int)
    vals=train[target].to_numpy(float)
    for key,v,wi in zip(map(tuple,x),vals,w):
        if key not in dic:dic[key]=[0.,np.zeros(len(target),float),0]
        dic[key][0]+=float(wi)
        dic[key][1]+=float(wi)*v
        dic[key][2]+=1
    return dic


def decrypt_keys(test, train, target, order: int, smooth: float=20.):
    """Frequency decoder: progressive 1/2/3-token joint motifs, never physiology at test."""
    ww=train.SURVEY_WT.to_numpy(float)
    ww=ww/ww.mean()
    prior=weighted_mean(train[target].to_numpy(float),ww)
    preds=[]
    if order==0:
        patterns=[("PHQ9_TOTAL",)]
    elif order==-1:
        patterns=[("SOMATIC_SCORE","COGAFF_SUM")]
    elif order==-2:
        patterns=[tuple(YITEMS)]
    else:
        patterns=list(itertools.chain.from_iterable(itertools.combinations(YITEMS,k) for k in range(1,order+1)))
    tables=[key_dictionary(train,ss,target,ww) for ss in patterns]
    # A decoded signature is assembled from recurring motifs rather than decoded
    # from a purported hidden deterministic mathematical inverse.
    for ss,table in zip(patterns,tables):
        arr=test[list(ss)].to_numpy(int)
        estimates=np.empty((len(test),len(target)),dtype=float)
        for i,key in enumerate(map(tuple,arr)):
            if key in table:
                support,v,n=table[key]
                local=v/support
                reliability=n/(n+smooth)  # n counts independent observed people
                estimates[i]=prior+reliability*(local-prior)
            else:
                estimates[i]=prior
        preds.append(estimates)
    return np.mean(preds,axis=0)



def cryptanalytic_key_rankings(df: pd.DataFrame):
    """Rank exploratory physiological-pattern keys using TRAIN frequency only.

    Key = one of four frozen, physiology-derived clusters. A higher-ranked
    candidate is NOT a recovered individual biomarker state. Cluster labels
    of future observations are used strictly for evaluation, never inference.
    """
    tr=df.loc[df.SPLIT.eq("train_2005-2006")].copy()
    w=tr.SURVEY_WT.to_numpy(float); w=w/w.mean()
    z=tr[Z].to_numpy(float)
    mu=weighted_mean(z,w)
    sd=np.sqrt(weighted_mean((z-mu)**2,w)).clip(1e-8)
    km=KMeans(n_clusters=4,n_init=10,random_state=SEED)
    labels=km.fit_predict((z-mu)/sd,sample_weight=w)
    prior=np.bincount(labels,weights=w,minlength=4)
    prior=prior/prior.sum()
    modes=[(0,"key_total"),(-1,"key_somatic_cogaff"),(-2,"key_exact_items"),
           (1,"key_single_char"),(2,"key_single_pair"),(3,"key_single_pair_triple")]
    # Pattern -> weighted frequency vector for the four candidate keys.
    pattern_tables={}
    for order,name in modes:
        if order==0: patterns=[("PHQ9_TOTAL",)]
        elif order==-1:patterns=[("SOMATIC_SCORE","COGAFF_SUM")]
        elif order==-2:patterns=[tuple(YITEMS)]
        else:patterns=list(itertools.chain.from_iterable(
            itertools.combinations(YITEMS,j) for j in range(1,order+1)))
        tables=[]
        for cols in patterns:
            table={}
            for ys,ci,wi in zip(map(tuple,tr[list(cols)].to_numpy(int)),labels,w):
                if ys not in table:table[ys]=np.zeros(4,float)
                table[ys][ci]+=wi
            tables.append(table)
        pattern_tables[name]=(patterns,tables)
    rows=[]
    for period in ("2007-2008","2009-2018","2021-2023"):
        te=df.loc[df.SPLIT.eq(period)]
        # Heldout Z is ONLY here, to determine the true group offline.
        truth=km.predict((te[Z].to_numpy(float)-mu)/sd)
        ww=te.SURVEY_WT.to_numpy(float);ww=ww/ww.mean()
        choices={"prior_only":np.tile(prior,(len(te),1))}
        for name,(patterns,tables) in pattern_tables.items():
            v=np.zeros((len(te),4),float)
            for cols,table in zip(patterns,tables):
                for i,key in enumerate(map(tuple,te[list(cols)].to_numpy(int))):
                    cnt=table.get(key)
                    if cnt is None:
                        v[i]+=prior
                    else:
                        v[i]+=(cnt+20.*prior)/(cnt.sum()+20.)
            v/=len(patterns)
            choices[name]=v
        for name,probs in choices.items():
            logloss=np.average(-np.log(np.maximum(1e-12,probs[np.arange(len(te)),truth])),weights=ww)
            top2=np.argsort(probs,axis=1)[:,-2:]
            hit=np.average(np.any(top2==truth[:,None],axis=1),weights=ww)
            rows.append(dict(period=period,method=name,n=len(te),
                             key_type="exploratory_4_cluster_frozen_train_Z",
                             weighted_logloss=float(logloss),weighted_top2_hit=float(hit),
                             information_gain_vs_prior=None))
    out=pd.DataFrame(rows)
    for period in out.period.unique():
        mask=out.period.eq(period)
        base=float(out.loc[mask & out.method.eq("prior_only"),"weighted_logloss"].iloc[0])
        out.loc[mask,"information_gain_vs_prior"]=base-out.loc[mask,"weighted_logloss"]
    return out



def coded_cryptanalytic_key_rankings(df: pd.DataFrame, max_bits: int):
    """Decode a hidden candidate key from frequencies of ENCODED Y characters.

    Symbol streams come from genuine Y, never test physiology. Candidate keys
    are discovery-only physiological groups; bigram/trigram statistics model
    symbols of complete reversible codes (not an RSA key-recovery algorithm).
    """
    tr=df.loc[df.SPLIT.eq("train_2005-2006")].copy()
    ww=tr.SURVEY_WT.to_numpy(float);ww=ww/ww.mean()
    z=tr[Z].to_numpy(float)
    mu=weighted_mean(z,ww)
    sd=np.sqrt(weighted_mean((z-mu)**2,ww)).clip(1e-8)
    cluster=KMeans(n_clusters=4,n_init=10,random_state=SEED)
    group=cluster.fit_predict((z-mu)/sd,sample_weight=ww)
    prior=np.bincount(group,weights=ww,minlength=4)
    prior/=prior.sum()
    tests=[(p,df.loc[df.SPLIT.eq(p)].copy()) for p in
           ("2007-2008","2009-2018","2021-2023")]
    rows=[]
    for yname,cols in direct_feature_sets().items():
        width=2 if yname!="PHQ_items" else 1
        for meth in METHODS:
            if meth=="two_adic" and len(cols)!=2:
                rows.append(dict(phenotype=yname,encoder=meth,order="all",
                                 period="all",status="not_defined_nonpair"))
                continue
            try:
                def as_codes(frame):
                    return [encode(tuple(int(v) for v in r),meth,width,max_bits)
                            for r in frame[cols].to_numpy(int)]
                codes_train=as_codes(tr)
                codes_test={p:as_codes(test) for p,test in tests}
                def fmt(v):
                    # Retain all code digits. Integer codes use complete hex,
                    # leading zero is not discarded by the integer codec.
                    return v if isinstance(v,str) else format(v,"X")
                strings_train=[fmt(c) for c in codes_train]
                strings_test={p:[fmt(c) for c in codes] for p,codes in codes_test.items()}
            except (OverflowError,ValueError,MemoryError) as e:
                rows.append(dict(phenotype=yname,encoder=meth,order="all",
                                 period="all",status="not_feasible:"+str(e)))
                continue
            for order in (1,2,3):
                # Frequency dictionary: n-gram of ciphertext -> group counts.
                table={}
                def ngrams(code):
                    return [code[j:j+order] for j in range(len(code)-order+1)]
                for code,k,w in zip(strings_train,group,ww):
                    grams=ngrams(code)
                    if not grams:continue
                    for token in grams:
                        if token not in table:table[token]=np.zeros(4,float)
                        table[token][k]+=w/len(grams)
                for p,test in tests:
                    ztest=test[Z].to_numpy(float)
                    truth=cluster.predict((ztest-mu)/sd)  # Offline labels only.
                    wtest=test.SURVEY_WT.to_numpy(float)
                    probs=np.empty((len(test),4),float)
                    for i,code in enumerate(strings_test[p]):
                        grams=ngrams(code)
                        if not grams:probs[i]=prior;continue
                        counts=np.zeros(4,float)
                        for tok in grams:
                            state=table.get(tok)
                            if state is not None:counts+=state
                        probs[i]=(counts+20.*prior)/(counts.sum()+20.)
                    loss=np.average(-np.log(np.maximum(probs[np.arange(len(test)),truth],1e-12)),weights=wtest)
                    hit=np.average(np.any(np.argsort(probs,axis=1)[:,-2:]==truth[:,None],axis=1),weights=wtest)
                    base=np.average(-np.log(np.maximum(prior[truth],1e-12)),weights=wtest)
                    rows.append(dict(phenotype=yname,encoder=meth,order=order,period=p,
                                     status="ok",n=len(test),n_symbols=len(table),
                                     weighted_logloss=float(loss),weighted_top2_hit=float(hit),
                                     information_gain_vs_prior=float(base-loss)))
    return pd.DataFrame(rows)


def direct_feature_sets():
    return {"PHQ_total":["PHQ9_TOTAL"],
            "somatic_cogaff":["SOMATIC_SCORE","COGAFF_SUM"],
            "PHQ_items":YITEMS}


def raw_predictive_audit(df):
    period_names=["2007-2008","2009-2018","2021-2023"]
    methods={"prior_only":None,"ridge_PHQ_total":["PHQ9_TOTAL"],
             "ridge_somatic_cogaff":["SOMATIC_SCORE","COGAFF_SUM"],
             "ridge_PHQ_items":YITEMS}
    outputs=[]; splits=[]
    for group,target in [("X_full",X),("A_raw",A),("G_raw",G),("Z_full",Z)]:
        tr=target_cohort(df,"train_2005-2006",target)
        if len(tr)<50:
            for p in period_names:
                outputs.append(dict(period=p,target_space=group,model="all",target="all",
                    n=0,status="skip_insufficient_train_X",n_train=len(tr)))
            continue
        tt=tr[target].to_numpy(float)
        wtr=tr.SURVEY_WT.to_numpy(float);wtr=wtr/wtr.mean()
        mu=weighted_mean(tt,wtr)
        sd=np.sqrt(weighted_mean((tt-mu)**2,wtr)).clip(1e-8)
        models={name:train_ridge(tr[feat].to_numpy(float),tt,wtr)
                for name,feat in methods.items() if feat is not None}
        for p in period_names:
            test=target_cohort(df,p,target)
            if len(test)<50:
                outputs.append(dict(period=p,target_space=group,model="all",target="all",
                    n=len(test),status="skip_insufficient_test_X",n_train=len(tr)))
                continue
            truth=test[target].to_numpy(float)
            w=test.SURVEY_WT.to_numpy(float);w=w/w.mean()
            predictions={"prior_only":np.tile(mu,(len(test),1))}
            for name,feat in methods.items():
                if name in models:
                    predictions[name]=models[name].predict(test[feat].to_numpy(float))
            for ordv,name in [(0,"key_PHQ_total"),(-1,"key_somatic_cogaff"),(-2,"key_full_items"),
                              (1,"cryptanalysis_unigrams"),(2,"cryptanalysis_bigrams"),
                              (3,"cryptanalysis_trigrams")]:
                predictions[name]=decrypt_keys(test,tr,target,ordv)
            for model,pred in predictions.items():
                r2,mse=standardized_score(truth,pred,w,mu,sd)
                for k,col in enumerate(target):
                    outputs.append(dict(period=p,target_space=group,model=model,target=col,
                                        n=len(test),n_train=len(tr),status="ok",
                                        weighted_R2=float(r2[k]),
                                        standardized_MSE=float(mse[k])))
                blocks={"all":list(range(len(target)))}
                if group=="X_full":
                    blocks.update({"A_raw":list(range(4)),"G_raw":list(range(4,7))})
                elif group=="Z_full":
                    blocks.update({"shared":[0,1],"discord":[2,3],
                                   "A_private":[4,5],"G_private":[6]})
                for block,indices in blocks.items():
                    splits.append(dict(period=p,target_space=group,block=block,model=model,
                          n=len(test),n_train=len(tr),status="ok",
                          mean_weighted_R2=float(np.mean(r2[indices])),
                          mean_standardized_MSE=float(np.mean(mse[indices]))))
    return pd.DataFrame(outputs),pd.DataFrame(splits)


def y_feature_code(tr,te,cols,method,max_bits):
    # Inputs here are nonnegative integer-valued PHQ scores or items.
    vals=[tuple(map(int,r)) for r in np.vstack([tr[cols].to_numpy(int),te[cols].to_numpy(int)])]
    width=2 if any(x in ("PHQ9_TOTAL","SOMATIC_SCORE","COGAFF_SUM") for x in cols) else 1
    if method=="two_adic" and len(cols)!=2: raise ValueError("two_adic_requires_pair")
    codes=[encode(v,method,width,max_bits) for v in vals]
    decoded=[decode(c,method,len(cols),width) for c in codes]
    if decoded!=vals:raise AssertionError("encoded Y roundtrip mismatch")
    # For a fair representation test, every code bit/digit/byte is retained.
    if method=="decimal_interleave":
        features=np.asarray([[int(ch) for ch in code] for code in codes],dtype=float)
    else:
        width_bytes=(max(x.bit_length() for x in codes)+7)//8
        if width_bytes==0:width_bytes=1
        features=np.asarray([list(int(c).to_bytes(width_bytes,'big')) for c in codes],dtype=float)
    return features[:len(tr)],features[len(tr):],len(features[0])


def coded_y_models(df,max_bits):
    records=[]
    for ys,cols in direct_feature_sets().items():
        for method in ("raw",)+METHODS:
            for target_name,target in [("X_full",X),("A_raw",A),("G_raw",G),("Z_full",Z)]:
                tr=target_cohort(df,"train_2005-2006",target)
                if len(tr)<50:
                    records.append(dict(phenotype=ys,encoder=method,target_space=target_name,
                                        period="all",status="skip_insufficient_train_X",n_train=len(tr)))
                    continue
                wt=tr.SURVEY_WT.to_numpy(float);wt=wt/wt.mean()
                tt=tr[target].to_numpy(float)
                mu=weighted_mean(tt,wt)
                sd=np.sqrt(weighted_mean((tt-mu)**2,wt)).clip(1e-8)
                for p in ("2007-2008","2009-2018","2021-2023"):
                    te=target_cohort(df,p,target)
                    if len(te)<50:
                        records.append(dict(phenotype=ys,encoder=method,target_space=target_name,
                                            period=p,status="skip_insufficient_test_X",n=len(te),n_train=len(tr)))
                        continue
                    try:
                        if method=="raw":
                            aa=tr[cols].to_numpy(float);bb=te[cols].to_numpy(float)
                        else:
                            aa,bb,_=y_feature_code(tr,te,cols,method,max_bits)
                        model=train_ridge(aa,tt,wt)
                        pred=model.predict(bb)
                        w=te.SURVEY_WT.to_numpy(float);w=w/w.mean()
                        r2,mse=standardized_score(te[target].to_numpy(float),pred,w,mu,sd)
                        records.append(dict(phenotype=ys,encoder=method,target_space=target_name,
                                            period=p,status="ok",n=len(te),n_train=len(tr),features=aa.shape[1],
                                            mean_weighted_R2=float(np.mean(r2)),
                                            mean_standardized_MSE=float(np.mean(mse))))
                    except (OverflowError,ValueError,MemoryError) as e:
                        records.append(dict(phenotype=ys,encoder=method,target_space=target_name,
                                            period=p,status="not_feasible:"+str(e),n=len(te),n_train=len(tr)))
    return pd.DataFrame(records)


def self_test():
    # SAS XPORT reader may return numeric zero as a nonzero subnormal.
    # Verify that the same cleaning rule used by Scripts 81/95 restores it,
    # without treating invalid PHQ responses (7, 9, other) as valid.
    raw_test=pd.Series([5.397605346934028e-79, 0.0, 1.0, 2.0, 3.0, 7.0, 9.0, np.nan])
    cleaned=raw_test.mask(raw_test.abs() < 1e-10, 0.0)
    cleaned=cleaned.where(cleaned.isin([0, 1, 2, 3]))
    assert cleaned.iloc[:5].tolist()==[0.,0.,1.,2.,3.]
    assert cleaned.iloc[5:].isna().all()
    rng=np.random.default_rng(SEED)
    for n in range(1,10):
        for _ in range(40):
            vals=tuple(int(x) for x in rng.integers(0,7,n))
            for method in METHODS:
                if method=="two_adic" and n!=2:continue
                code=encode(vals,method,2,4096)
                assert decode(code,method,n,2)==vals,(method,n,vals)
    print("PASS: deterministic codec inverse tests, four families")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--audit-rows",type=int,default=128,help="Maximum REAL 2005-06 rows used for each exhaustive coding feasibility check")
    parser.add_argument("--max-bits",type=int,default=4096)
    parser.add_argument("--audit-only",action="store_true",help="Skip predictive models, still require real NHANES data")
    parser.add_argument("--self-test",action="store_true",help="Mathematical unit tests only; no empirical claims")
    args=parser.parse_args()
    if args.self_test:return self_test()
    if args.audit_rows<1 or args.max_bits<16:raise ValueError("Invalid audit parameters")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    df=load()
    cohorts=df.groupby("SPLIT").size().to_dict()
    audit=audit_codes(df,args.audit_rows,args.max_bits)
    audit.to_csv(OUTPUT/"96_exhaustive_encoding_feasibility.csv",index=False)
    if not args.audit_only:
        detail, blocks=raw_predictive_audit(df)
        detail.to_csv(OUTPUT/"96_reverse_target_metrics.csv",index=False)
        blocks.to_csv(OUTPUT/"96_reverse_block_metrics.csv",index=False)
        coded=coded_y_models(df,args.max_bits)
        coded.to_csv(OUTPUT/"96_encoded_Y_predictive_comparison.csv",index=False)
        keys=cryptanalytic_key_rankings(df)
        keys.to_csv(OUTPUT/"96_cryptanalytic_key_rankings.csv",index=False)
        token_keys=coded_cryptanalytic_key_rankings(df,args.max_bits)
        token_keys.to_csv(OUTPUT/"96_encoded_symbol_key_rankings.csv",index=False)
    else: coded=pd.DataFrame()
    meta=dict(script="96_Y_code_cryptanalysis_audit.py",base="real frozen Script-73 and PHQ item XPTs",
        training="0506",heldout=["0708","0918","2123"],complete_case_cohorts=cohorts,
        encoding_subsets=int(len(audit)//len(METHODS)),encoder_attempts=len(audit),
        coder_status_counts=audit.status.value_counts().to_dict(),
        precision={"Y":0,"X":3,"Z":3},
        caveats=["No causal identification", "No unique Y->X/Z inverse", "2-adic is pair-only",
        "Encoding cannot add Y information", "Survey-weighted point metrics only: not full NHANES variance estimation",
        "Y/Z analysis uses complete PHQ/Z rows; X prediction uses target-specific complete A, G or X cohorts. Compare only within matched target cohorts",
        "Frequency keys trained only from 2005-06 observed phenotype-physiology pairs",
        "Cluster-based candidate keys are exploratory and not biologically validated",
        "2007-08 is not fully independent of the 2005-08 unsupervised X-to-Z representation fit; 2009-18 and 2021-23 are the principal temporal holdouts",
        "Encoding-symbol frequency search is an analogy to substitution cryptanalysis, not RSA decryption",
        "Exhaustive subsets refer to exact codec auditing; predictive audits use prespecified phenotype representations"],
        coded_model_rows=len(coded),coverage_file="96_cohort_coverage.csv",seed=SEED)
    (OUTPUT/"96_manifest.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
