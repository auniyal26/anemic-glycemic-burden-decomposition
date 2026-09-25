#!/usr/bin/env python
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import FastICA
except ImportError as e:
    raise RuntimeError("scikit-learn is required: pip install scikit-learn") from e

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as e:
    raise RuntimeError("scipy is required: pip install scipy") from e

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "EDA" / "Results"
MODELS = ROOT / "EDA" / "Models"
for p in [RES, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

INPUT = RES / "63_deep_ddx_input.csv"
if not INPUT.exists():
    INPUT = RES / "62_ddx_model_input.csv"
if not INPUT.exists():
    raise FileNotFoundError("Need EDA/Results/63_deep_ddx_input.csv or 62_ddx_model_input.csv")

d = pd.read_csv(INPUT)
d["PERIOD"] = d["PERIOD"].astype(str)

PERIODS = ["2005-2008", "2009-2018", "2021-2023"]
DOMAINS = {
    "A": {
        "pcs": ["A_OI_PC1_FZ", "A_OI_PC2_FZ", "A_OI_PC3_FZ"],
        "raw": ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"],
    },
    "G": {
        "pcs": ["G3_OI_PC1_FZ", "G3_OI_PC2_FZ", "G3_OI_PC3_FZ"],
        "raw": ["LBXGH", "LBXGLU", "LOG_IN"],
    },
}

required = ["PERIOD", "SURVEY_WT"]
for spec in DOMAINS.values():
    required += spec["pcs"] + spec["raw"]
missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

def asnum(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(float)

def valid_rows(df, cols):
    arr = np.column_stack([asnum(df[c]) for c in cols])
    w = asnum(df["SURVEY_WT"])
    return np.all(np.isfinite(arr), axis=1) & np.isfinite(w) & (w > 0)

def wmean(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    return float(np.sum(w*x) / np.sum(w))

def wcov(X, w):
    X = np.asarray(X, float); w = np.asarray(w, float)
    mu = np.sum(X*w[:,None], axis=0) / np.sum(w)
    D = X - mu
    return (D * w[:,None]).T @ D / np.sum(w)

def wcorr(x, y, w):
    x=np.asarray(x,float); y=np.asarray(y,float); w=np.asarray(w,float)
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
    if ok.sum()<3: return np.nan
    x=x[ok]; y=y[ok]; w=w[ok]
    mx=wmean(x,w); my=wmean(y,w)
    vx=wmean((x-mx)**2,w); vy=wmean((y-my)**2,w)
    if vx<=0 or vy<=0: return np.nan
    return float(wmean((x-mx)*(y-my),w)/np.sqrt(vx*vy))

def wquantile(x, w, probs):
    x=np.asarray(x,float); w=np.asarray(w,float)
    ok=np.isfinite(x)&np.isfinite(w)&(w>0)
    x=x[ok]; w=w[ok]
    if len(x)==0: return [np.nan]*len(probs)
    idx=np.argsort(x); x=x[idx]; w=w[idx]
    c=np.cumsum(w)/np.sum(w)
    return [float(np.interp(p,c,x)) for p in probs]

def weighted_r2(y, yhat, w):
    y=np.asarray(y,float); yhat=np.asarray(yhat,float); w=np.asarray(w,float)
    mu=wmean(y,w)
    sse=np.sum(w*(y-yhat)**2)
    sst=np.sum(w*(y-mu)**2)
    return np.nan if sst<=0 else float(1-sse/sst)

def fit_weighted_decoder(Z, X, w):
    D=np.column_stack([np.ones(len(Z)), Z])
    sw=np.sqrt(w)
    return np.linalg.lstsq(D*sw[:,None], X*sw[:,None], rcond=None)[0]

def decode(Z, coef):
    return np.column_stack([np.ones(len(Z)), Z]) @ coef

def weighted_corr_matrix(X, w):
    p=X.shape[1]
    C=np.eye(p)
    for i in range(p):
        for j in range(i+1,p):
            C[i,j]=C[j,i]=wcorr(X[:,i],X[:,j],w)
    return C

def condition_number_cov(X, w):
    C=wcov(X,w)
    eig=np.linalg.eigvalsh(C)
    eig=np.clip(eig,1e-12,None)
    return float(eig.max()/eig.min())

def fit_ica(Z, seed=42):
    model=FastICA(
        n_components=Z.shape[1],
        whiten="unit-variance",
        algorithm="parallel",
        fun="logcosh",
        max_iter=3000,
        tol=1e-6,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        S=model.fit_transform(Z)
    warn=" | ".join(str(x.message) for x in ws)
    return model,S,warn

recon_rows=[]
corr_rows=[]
within_rows=[]
dist_rows=[]
support_rows=[]
rediscovery_rows=[]
condition_rows=[]
model_meta={}
frozen_sources_by_domain_period={}

for domain,spec in DOMAINS.items():
    pcs=spec["pcs"]; raw=spec["raw"]

    disc=d[d["PERIOD"].eq("2005-2008")].copy()
    ok=valid_rows(disc,pcs+raw)
    disc=disc.loc[ok].copy()
    if len(disc)<100:
        raise RuntimeError(f"Too few discovery rows for {domain}: {len(disc)}")

    Z=disc[pcs].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    X=disc[raw].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    w=asnum(disc["SURVEY_WT"])

    ica,Sdisc,warn=fit_ica(Z)
    decoder=fit_weighted_decoder(Z,X,w)

    np.savez(
        MODELS/f"64_{domain}_PCA_to_ICA_map.npz",
        components_=ica.components_,
        mixing_=ica.mixing_,
        mean_=ica.mean_,
        whitening_=getattr(ica,"whitening_",np.array([])),
        decoder_=decoder,
        pc_names=np.array(pcs,dtype=object),
        raw_names=np.array(raw,dtype=object),
    )

    fwd=np.asarray(ica.components_,float)
    inv=np.asarray(ica.mixing_,float)
    condition_rows.append({"domain":domain,"map":"PCA_to_ICA","condition_number":float(np.linalg.cond(fwd))})
    condition_rows.append({"domain":domain,"map":"ICA_to_PCA","condition_number":float(np.linalg.cond(inv))})

    model_meta[domain]={
        "n_discovery":int(len(disc)),
        "pcs":pcs,
        "raw":raw,
        "fit_warning":warn,
        "n_iter":int(getattr(ica,"n_iter_",0)),
        "forward_condition_number":float(np.linalg.cond(fwd)),
        "inverse_condition_number":float(np.linalg.cond(inv)),
    }

    smu=np.sum(Sdisc*w[:,None],axis=0)/np.sum(w)
    Scov=wcov(Sdisc,w)
    Sinv=np.linalg.pinv(Scov)
    d2_disc=np.einsum("ij,jk,ik->i",Sdisc-smu,Sinv,Sdisc-smu)
    q95,q99=wquantile(d2_disc,w,[.95,.99])

    for period in PERIODS:
        pp=d[d["PERIOD"].eq(period)].copy()
        ok=valid_rows(pp,pcs+raw)
        pp=pp.loc[ok].copy()
        if len(pp)<20:
            continue

        Zp=pp[pcs].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        Xp=pp[raw].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        wp=asnum(pp["SURVEY_WT"])

        Sp=ica.transform(Zp)
        frozen_sources_by_domain_period[(domain,period)] = (pp.index.to_numpy(),Sp,wp)

        Zback=ica.inverse_transform(Sp)
        Xback=decode(Zback,decoder)
        Xpca=decode(Zp,decoder)

        recon_rows.append({
            "domain":domain,"period":period,"metric":"max_abs_PCA_roundtrip_error",
            "value":float(np.max(np.abs(Zp-Zback))),"n":len(pp)
        })

        for j,var in enumerate(raw):
            recon_rows.append({
                "domain":domain,"period":period,"metric":f"PCA_decoder_R2__{var}",
                "value":weighted_r2(Xp[:,j],Xpca[:,j],wp),"n":len(pp)
            })
            recon_rows.append({
                "domain":domain,"period":period,"metric":f"ICA_roundtrip_R2__{var}",
                "value":weighted_r2(Xp[:,j],Xback[:,j],wp),"n":len(pp)
            })

        mu=np.sum(Xp*wp[:,None],axis=0)/np.sum(wp)
        sse_pca=np.sum(wp[:,None]*(Xp-Xpca)**2)
        sse_ica=np.sum(wp[:,None]*(Xp-Xback)**2)
        sst=np.sum(wp[:,None]*(Xp-mu)**2)
        recon_rows.append({"domain":domain,"period":period,"metric":"PCA_decoder_global_R2","value":float(1-sse_pca/sst),"n":len(pp)})
        recon_rows.append({"domain":domain,"period":period,"metric":"ICA_roundtrip_global_R2","value":float(1-sse_ica/sst),"n":len(pp)})

        for k in range(Sp.shape[1]):
            for j,var in enumerate(raw):
                corr_rows.append({
                    "domain":domain,"period":period,"source":f"{domain}_ICA{k+1}",
                    "raw_variable":var,"weighted_correlation":wcorr(Sp[:,k],Xp[:,j],wp),"n":len(pp)
                })

        C=weighted_corr_matrix(Sp,wp)
        for i in range(Sp.shape[1]):
            for j in range(i+1,Sp.shape[1]):
                within_rows.append({
                    "domain":domain,"period":period,"source_i":f"{domain}_ICA{i+1}",
                    "source_j":f"{domain}_ICA{j+1}","weighted_correlation":float(C[i,j]),"n":len(pp)
                })

        for k in range(Sp.shape[1]):
            qs=wquantile(Sp[:,k],wp,[.01,.05,.50,.95,.99])
            m=wmean(Sp[:,k],wp)
            dist_rows.append({
                "domain":domain,"period":period,"source":f"{domain}_ICA{k+1}",
                "weighted_mean":m,
                "weighted_sd":float(np.sqrt(wmean((Sp[:,k]-m)**2,wp))),
                "q01":qs[0],"q05":qs[1],"q50":qs[2],"q95":qs[3],"q99":qs[4],"n":len(pp)
            })

        D=Sp-smu
        d2=np.einsum("ij,jk,ik->i",D,Sinv,D)
        support_rows.append({
            "domain":domain,"period":period,
            "weighted_mean_mahalanobis_d2":wmean(d2,wp),
            "pct_weight_above_discovery_95pct":float(100*np.sum(wp[d2>q95])/np.sum(wp)),
            "pct_weight_above_discovery_99pct":float(100*np.sum(wp[d2>q99])/np.sum(wp)),
            "discovery_q95_d2":q95,"discovery_q99_d2":q99,
            "latent_cov_condition_number":condition_number_cov(Sp,wp),
            "max_abs_within_source_corr":float(np.max(np.abs(C-np.eye(C.shape[0])))),
            "n":len(pp)
        })

        if period != "2005-2008":
            try:
                local,Sloc,lwarn=fit_ica(Zp,seed=42)
                M=np.zeros((3,3))
                for i in range(3):
                    for j in range(3):
                        M[i,j]=wcorr(Sp[:,i],Sloc[:,j],wp)
                rr,cc=linear_sum_assignment(-np.abs(M))
                for i,j in zip(rr,cc):
                    rediscovery_rows.append({
                        "domain":domain,"period":period,
                        "frozen_source":f"{domain}_ICA{i+1}",
                        "local_source":f"{domain}_ICA{j+1}",
                        "weighted_alignment_correlation":float(M[i,j]),
                        "absolute_alignment":float(abs(M[i,j])),
                        "local_fit_warning":lwarn,"n":len(pp)
                    })
            except Exception as e:
                rediscovery_rows.append({
                    "domain":domain,"period":period,
                    "frozen_source":"ERROR","local_source":"ERROR",
                    "weighted_alignment_correlation":np.nan,
                    "absolute_alignment":np.nan,
                    "local_fit_warning":str(e),"n":len(pp)
                })

cross_rows=[]
for period in PERIODS:
    a=frozen_sources_by_domain_period.get(("A",period))
    g=frozen_sources_by_domain_period.get(("G",period))
    if a is None or g is None:
        continue
    ia,Sa,wa=a; ig,Sg,wg=g
    common=np.intersect1d(ia,ig)
    if len(common)<20:
        continue
    pa={idx:i for i,idx in enumerate(ia)}
    pg={idx:i for i,idx in enumerate(ig)}
    ai=np.array([pa[x] for x in common])
    gi=np.array([pg[x] for x in common])
    w=wa[ai]
    for i in range(3):
        for j in range(3):
            cross_rows.append({
                "period":period,"A_source":f"A_ICA{i+1}","G_source":f"G_ICA{j+1}",
                "weighted_correlation":wcorr(Sa[ai,i],Sg[gi,j],w),"n":len(common)
            })

pd.DataFrame(recon_rows).to_csv(RES/"64_ica_reconstruction_roundtrip.csv",index=False)
pd.DataFrame(corr_rows).to_csv(RES/"64_ica_source_raw_correlations.csv",index=False)
pd.DataFrame(within_rows).to_csv(RES/"64_ica_within_source_correlations.csv",index=False)
pd.DataFrame(dist_rows).to_csv(RES/"64_ica_source_distributions.csv",index=False)
pd.DataFrame(support_rows).to_csv(RES/"64_ica_support_shift.csv",index=False)
pd.DataFrame(rediscovery_rows).to_csv(RES/"64_ica_rediscovery_alignment.csv",index=False)
pd.DataFrame(condition_rows).to_csv(RES/"64_ica_map_conditioning.csv",index=False)
pd.DataFrame(cross_rows).to_csv(RES/"64_ica_crossblock_source_correlations.csv",index=False)

manifest={
    "script":"64_pca_to_ica_bidirectional_transfer.py",
    "input":str(INPUT.relative_to(ROOT)),
    "design":"Frozen PCA scores -> discovery-fit FastICA -> frozen transfer -> inverse ICA -> weighted raw-variable decoder",
    "pca_refit":False,
    "locked_outputs_modified":False,
    "important_math_note":"Using all 3 ICA components preserves the same retained PCA subspace. ICA should not improve information content merely by rotation; the scientific tests are independence, interpretability, conditioning, transfer, and rediscoverability.",
    "ica_fit_weighting":"FastICA fit is unweighted because sklearn FastICA has no sample_weight; all reported correlations, reconstruction metrics, ranges, covariance/support metrics use NHANES survey weights.",
    "model_metadata":model_meta,
}
(RES/"64_ica_bidirectional_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

print("PASS  Script 64 PCA->ICA bidirectional experiment complete.")
print("PASS  Existing frozen PCA coordinates used; PCA was NOT refit.")
print("PASS  Forward ICA and inverse ICA maps saved under EDA/Models.")
print("PASS  Round-trip reconstruction, source independence, range/support shift, and map conditioning written.")
print("PASS  Frozen-source transfer + later-period ICA rediscovery alignment written.")
print("PASS  A-ICA x G-ICA cross-source correlations written.")
print("PASS  Locked Results untouched.")
print("NEXT  Inspect the 64_* outputs before interpreting any ICA component as a biological source.")
