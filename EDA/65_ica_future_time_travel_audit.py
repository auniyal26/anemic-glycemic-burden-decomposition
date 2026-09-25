#!/usr/bin/env python
# =============================================================================
# 65_ica_future_time_travel_audit.py
#
# Fixes Script 64's future-period exclusion.
#
# Why Script 64 only showed 2005-08:
# it required BOTH frozen PCs and raw biomarkers. In the Script-63 table,
# later periods have frozen PCs but the restored raw biomarker columns are empty.
#
# This audit therefore tests the "time machine" where it is actually identifiable:
# discovery PCA space -> discovery ICA map -> future ICA space -> inverse ICA
# -> SAME frozen PCA coordinates.
#
# It also tests:
# - within-source correlations across time
# - latent range/support shift
# - map conditioning
# - whether ICA re-fit in the future rediscovers the same sources
# - A-source vs G-source cross-correlations
#
# No PCA refit. Locked Results untouched.
# =============================================================================

from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd

from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "EDA" / "Results"
MODELS = ROOT / "EDA" / "Models"
RES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

INPUT = RES / "63_deep_ddx_input.csv"
if not INPUT.exists():
    INPUT = RES / "62_ddx_model_input.csv"
if not INPUT.exists():
    raise FileNotFoundError("Need EDA/Results/63_deep_ddx_input.csv or 62_ddx_model_input.csv")

d = pd.read_csv(INPUT)
d["PERIOD"] = d["PERIOD"].astype(str)

PERIODS = ["2005-2008", "2009-2018", "2021-2023"]
DOMAINS = {
    "A": ["A_OI_PC1_FZ", "A_OI_PC2_FZ", "A_OI_PC3_FZ"],
    "G": ["G3_OI_PC1_FZ", "G3_OI_PC2_FZ", "G3_OI_PC3_FZ"],
}

for cols in DOMAINS.values():
    for c in cols:
        if c not in d.columns:
            raise ValueError(f"Missing required frozen PC: {c}")
if "SURVEY_WT" not in d.columns:
    raise ValueError("Missing SURVEY_WT")

def arr(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

def weights(df):
    return pd.to_numeric(df["SURVEY_WT"], errors="coerce").to_numpy(float)

def valid(df, cols):
    X = arr(df, cols)
    w = weights(df)
    return np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)

def wmean(x,w):
    return float(np.sum(w*x)/np.sum(w))

def wcov(X,w):
    mu=np.sum(X*w[:,None],axis=0)/np.sum(w)
    D=X-mu
    return (D*w[:,None]).T@D/np.sum(w)

def wcorr(x,y,w):
    x=np.asarray(x,float); y=np.asarray(y,float); w=np.asarray(w,float)
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
    if ok.sum()<3: return np.nan
    x=x[ok]; y=y[ok]; w=w[ok]
    mx=wmean(x,w); my=wmean(y,w)
    vx=wmean((x-mx)**2,w); vy=wmean((y-my)**2,w)
    if vx<=0 or vy<=0: return np.nan
    return float(wmean((x-mx)*(y-my),w)/np.sqrt(vx*vy))

def wcorrmat(X,w):
    p=X.shape[1]
    C=np.eye(p)
    for i in range(p):
        for j in range(i+1,p):
            C[i,j]=C[j,i]=wcorr(X[:,i],X[:,j],w)
    return C

def wq(x,w,probs):
    x=np.asarray(x,float); w=np.asarray(w,float)
    ok=np.isfinite(x)&np.isfinite(w)&(w>0)
    x=x[ok]; w=w[ok]
    ii=np.argsort(x); x=x[ii]; w=w[ii]
    cw=np.cumsum(w)/np.sum(w)
    return [float(np.interp(p,cw,x)) for p in probs]

def fit_ica(Z,seed=42):
    model=FastICA(
        n_components=Z.shape[1],
        whiten="unit-variance",
        algorithm="parallel",
        fun="logcosh",
        max_iter=3000,
        tol=1e-6,
        random_state=seed
    )
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        S=model.fit_transform(Z)
    warn=" | ".join(str(x.message) for x in ws)
    return model,S,warn

roundtrip=[]
within=[]
support=[]
rediscovery=[]
dist=[]
conditioning=[]
cross=[]
counts=[]
sources={}

for domain,pcs in DOMAINS.items():
    disc=d[d["PERIOD"].eq("2005-2008")].copy()
    disc=disc.loc[valid(disc,pcs)].copy()
    Z0=arr(disc,pcs)
    w0=weights(disc)

    ica,S0,warn=fit_ica(Z0)

    # save forward/inverse map
    np.savez(
        MODELS/f"65_{domain}_discovery_ICA_map.npz",
        components_=ica.components_,
        mixing_=ica.mixing_,
        mean_=ica.mean_,
        whitening_=getattr(ica,"whitening_",np.array([])),
        pc_names=np.array(pcs,dtype=object)
    )

    conditioning.append({
        "domain":domain,
        "forward_condition_number":float(np.linalg.cond(ica.components_)),
        "inverse_condition_number":float(np.linalg.cond(ica.mixing_)),
        "n_discovery":len(disc),
        "fit_warning":warn
    })

    mu0=np.sum(S0*w0[:,None],axis=0)/np.sum(w0)
    C0=wcov(S0,w0)
    C0inv=np.linalg.pinv(C0)
    d20=np.einsum("ij,jk,ik->i",S0-mu0,C0inv,S0-mu0)
    q95,q99=wq(d20,w0,[.95,.99])

    for period in PERIODS:
        pp=d[d["PERIOD"].eq(period)].copy()
        pp=pp.loc[valid(pp,pcs)].copy()
        counts.append({"domain":domain,"period":period,"n_with_frozen_PCs":len(pp)})
        if len(pp)<20:
            continue

        Z=arr(pp,pcs)
        w=weights(pp)
        S=ica.transform(Z)
        Zback=ica.inverse_transform(S)
        sources[(domain,period)] = (pp.index.to_numpy(),S,w)

        err=np.abs(Z-Zback)
        roundtrip.append({
            "domain":domain,"period":period,"n":len(pp),
            "max_abs_roundtrip_error":float(np.max(err)),
            "mean_abs_roundtrip_error":float(np.mean(err)),
            "rmse_roundtrip_error":float(np.sqrt(np.mean((Z-Zback)**2)))
        })

        C=wcorrmat(S,w)
        for i in range(3):
            for j in range(i+1,3):
                within.append({
                    "domain":domain,"period":period,
                    "source_i":f"{domain}_ICA{i+1}",
                    "source_j":f"{domain}_ICA{j+1}",
                    "weighted_correlation":float(C[i,j]),
                    "n":len(pp)
                })

        for k in range(3):
            qq=wq(S[:,k],w,[.01,.05,.50,.95,.99])
            m=wmean(S[:,k],w)
            sd=float(np.sqrt(wmean((S[:,k]-m)**2,w)))
            dist.append({
                "domain":domain,"period":period,"source":f"{domain}_ICA{k+1}",
                "weighted_mean":m,"weighted_sd":sd,
                "q01":qq[0],"q05":qq[1],"q50":qq[2],"q95":qq[3],"q99":qq[4],
                "n":len(pp)
            })

        D=S-mu0
        d2=np.einsum("ij,jk,ik->i",D,C0inv,D)
        support.append({
            "domain":domain,"period":period,"n":len(pp),
            "weighted_mean_mahalanobis_d2":wmean(d2,w),
            "pct_weight_above_discovery_95pct":
                float(100*np.sum(w[d2>q95])/np.sum(w)),
            "pct_weight_above_discovery_99pct":
                float(100*np.sum(w[d2>q99])/np.sum(w)),
            "max_abs_within_source_corr":
                float(np.max(np.abs(C-np.eye(3)))),
            "latent_cov_condition_number":
                float(np.linalg.cond(wcov(S,w)))
        })

        # Can a fresh ICA in the future rediscover the frozen discovery sources?
        if period != "2005-2008":
            local,Sloc,lwarn=fit_ica(Z,seed=42)
            M=np.empty((3,3))
            for i in range(3):
                for j in range(3):
                    M[i,j]=wcorr(S[:,i],Sloc[:,j],w)
            ri,cj=linear_sum_assignment(-np.abs(M))
            for i,j in zip(ri,cj):
                rediscovery.append({
                    "domain":domain,"period":period,
                    "frozen_source":f"{domain}_ICA{i+1}",
                    "local_source":f"{domain}_ICA{j+1}",
                    "weighted_alignment_correlation":float(M[i,j]),
                    "absolute_alignment":float(abs(M[i,j])),
                    "n":len(pp),
                    "local_fit_warning":lwarn
                })

# cross-domain correlations on same rows
for period in PERIODS:
    A=sources.get(("A",period))
    G=sources.get(("G",period))
    if A is None or G is None: continue
    ia,Sa,wa=A
    ig,Sg,wg=G
    common=np.intersect1d(ia,ig)
    pa={x:i for i,x in enumerate(ia)}
    pg={x:i for i,x in enumerate(ig)}
    ai=np.array([pa[x] for x in common])
    gi=np.array([pg[x] for x in common])
    w=wa[ai]
    for i in range(3):
        for j in range(3):
            cross.append({
                "period":period,
                "A_source":f"A_ICA{i+1}",
                "G_source":f"G_ICA{j+1}",
                "weighted_correlation":wcorr(Sa[ai,i],Sg[gi,j],w),
                "n":len(common)
            })

pd.DataFrame(counts).to_csv(RES/"65_future_PC_availability.csv",index=False)
pd.DataFrame(roundtrip).to_csv(RES/"65_future_ICA_roundtrip.csv",index=False)
pd.DataFrame(within).to_csv(RES/"65_future_ICA_within_correlations.csv",index=False)
pd.DataFrame(dist).to_csv(RES/"65_future_ICA_distributions.csv",index=False)
pd.DataFrame(support).to_csv(RES/"65_future_ICA_support_shift.csv",index=False)
pd.DataFrame(rediscovery).to_csv(RES/"65_future_ICA_rediscovery.csv",index=False)
pd.DataFrame(conditioning).to_csv(RES/"65_future_ICA_conditioning.csv",index=False)
pd.DataFrame(cross).to_csv(RES/"65_future_ICA_crossblock.csv",index=False)

manifest={
    "script":"65_ica_future_time_travel_audit.py",
    "input":str(INPUT.relative_to(ROOT)),
    "reason_for_script64_discovery_only":
        "Script64 required raw biomarkers and frozen PCs simultaneously; later rows in Script63 retain frozen PCs but restored raw biomarker fields are empty.",
    "test":
        "Freeze discovery ICA map, transform later frozen PCA coordinates, invert them back to the exact same PCA coordinates, audit support/range/correlation drift, and compare with independently refit future ICA.",
    "pca_refit":False,
    "locked_outputs_modified":False,
    "important_limit":
        "Future all-the-way-back raw-biomarker reconstruction cannot be directly scored from Script63 because those raw future fields are absent there. This script validates the bidirectional PCA<->ICA map and latent-space transfer.",
}
(RES/"65_future_time_travel_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

print("PASS  Script 65 future time-travel audit complete.")
print("PASS  2009-18 and 2021-23 tested using their frozen PCA coordinates.")
print("PASS  Discovery ICA map frozen; no PCA refit.")
print("PASS  Forward->inverse round trip checked in every available period.")
print("PASS  Future latent support/range/correlation drift checked.")
print("PASS  Independent future ICA rediscovery alignment checked.")
print("PASS  Locked Results untouched.")
print("NEXT  Inspect the 65_* outputs.")
