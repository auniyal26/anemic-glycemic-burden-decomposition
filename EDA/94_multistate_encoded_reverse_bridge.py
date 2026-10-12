#!/usr/bin/env python3
"""Script 94 | Prof. Saeed: multiple plausible encoded states conditioned on Y.

Run from repository root: python EDA/94_multistate_encoded_reverse_bridge.py
Reads ONLY EDA/Results/73_frozen_phenotype_map_input.csv (real NHANES-derived).
Frozen: 2005-06 train; 2007-08 calibrate; 2009-18 and 2021-23 tests.
No CLI arguments, synthetic fallback, frozen-model modification or Git push.

Different from Script 93: assesses an actual multi-hypothesis reverse bridge.
1. Encode all 7 quantized Z coordinates using exact decimal interleaving.
2. Learn h(Z) -> Y (total or somatic+cogaff), comparing decoded/raw and
   encoded-digit multiscale feature models; calibrate likelihood in 2007-08.
3. Compute candidate posterior p(code | Y) over observed training states.
4. Aggregate posterior over 24 physiology-state families learned from train Z,
   rank 5 distinct code candidates (one high-weight code from each family).
5. Audit held-out cluster hit@5, state diversity, information gain, Z R2,
   and compare against an Y-independent prior and direct Y->Z ridge.

NOT patient-specific recovery: codebook comprises observed train states only.
Hit@5 uses the true held-out Z solely for offline evaluation. It is NOT an
individual clinical accuracy metric or evidence of causation. No generated
NHANES 'longitudinal' trajectories. Marginal posterior is NOT conformal.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EDA = Path(__file__).resolve().parent
INPUT = EDA / "Results" / "73_frozen_phenotype_map_input.csv"
OUT = EDA / "Results"
ZCOLS = ["SP_SHARED1", "SP_SHARED2", "SP_DISCORD1", "SP_DISCORD2",
         "SP_A_PRIVATE3", "SP_A_PRIVATE4", "SP_G_PRIVATE3"]
YCASES = {"PHQ_TOTAL": ["PHQ9_TOTAL"], "SOMATIC_COGAFF": ["SOMATIC_SCORE", "COGAFF_SUM"]}
N_CLUSTERS = 24
TOP_K = 5
DECIMALS = 3
DIGITS_PER_COMPONENT = 7
BATCH = 100
SEED = 94


def weights(df):
    w = df["SURVEY_WT"].to_numpy(dtype=float)
    return w / w.mean()


def load():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing REAL Script-73 NHANES input: {INPUT}")
    cols = ["SEQN", "CYCLE", "PERIOD", "SURVEY_WT", "ELIGIBLE_ADULT_NONPREG"] + ZCOLS + list(YCASES["PHQ_TOTAL"]) + list(YCASES["SOMATIC_COGAFF"])
    h = pd.read_csv(INPUT, nrows=0)
    missing = set(cols).difference(h.columns)
    if missing:
        raise ValueError(f"Missing columns in Script 73: {sorted(missing)}")
    df = pd.read_csv(INPUT, usecols=cols, low_memory=False)
    for col in ZCOLS + ["PHQ9_TOTAL", "SOMATIC_SCORE", "COGAFF_SUM", "SURVEY_WT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["CYCLE"] = df["CYCLE"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4)
    df = df[df.ELIGIBLE_ADULT_NONPREG.eq(1) & df[ZCOLS + ["PHQ9_TOTAL", "SOMATIC_SCORE", "COGAFF_SUM", "SURVEY_WT"]].notna().all(axis=1) & df.SURVEY_WT.gt(0)].copy()
    if df.SEQN.duplicated().any():
        raise AssertionError("Duplicate SEQN in the filtered Script-73 input")
    train = df[df.CYCLE.eq("0506")].copy()
    cal = df[df.CYCLE.eq("0708")].copy()
    test = {p: df[df.PERIOD.eq(p)].copy() for p in ["2009-2018", "2021-2023"]}
    if len(train)<100 or len(cal)<100 or any(len(v)<100 for v in test.values()):
        raise ValueError(f"Insufficient actual rows: train={len(train)}, cal={len(cal)}, tests={[len(x) for x in test.values()]}")
    return {"train": train, "cal":cal, **test}


def interleave(z_arrays):
    # Consistent sign-preserving integer representation at 3 decimal places.
    all_u={}
    all_q={}
    max_unsigned=0
    for period,z in z_arrays.items():
        if not np.isfinite(z).all() or np.max(np.abs(z)) > 1e9:
            raise ValueError("Z is nonfinite or out of safe integer encoding range")
        q=np.rint(z*(10**DECIMALS)).astype(np.int64)
        u=np.where(q>=0, 2*q, -2*q-1)
        all_q[period]=q
        all_u[period]=u
        max_unsigned=max(max_unsigned,int(np.max(u)))
    if len(str(max_unsigned)) > DIGITS_PER_COMPONENT:
        raise ValueError(f"Need {len(str(max_unsigned))} digits per coordinate; update DIGITS_PER_COMPONENT explicitly")
    codes={}; tokens={}; decoded={}; err={}
    for period,u in all_u.items():
        n,k = u.shape
        blocks = [np.asarray([list(f"{int(v):0{DIGITS_PER_COMPONENT}d}") for v in u[:,j]],dtype=np.uint8) for j in range(k)]
        seq = np.empty((n,k*DIGITS_PER_COMPONENT),dtype=np.uint8)
        for j,b in enumerate(blocks):
            seq[:,j::k]=b
        inverse = np.zeros_like(u)
        for j in range(k):
            inverse[:,j]=[int("".join(map(str,row))) for row in seq[:,j::k]]
        if not np.array_equal(inverse,u):
            raise AssertionError("Decimal-interleave integer inverse failed")
        qi = np.where(inverse%2==0, inverse//2, -(inverse+1)//2)
        if not np.array_equal(qi,all_q[period]):
            raise AssertionError("Signed integer inverse failed")
        decoded[period]=qi.astype(float)/(10**DECIMALS)
        err[period]=float(np.max(np.abs(decoded[period]-z_arrays[period])))
        if err[period] > .5/(10**DECIMALS) + 1e-8:
            raise AssertionError("Quantization tolerance exceeded")
        tokens[period]=seq.astype(np.float32)
        # Saved only as examples for hypothetical phenotype queries; no patient IDs.
        codes[period]=np.asarray(["".join(map(str,row)) for row in seq],dtype=f"U{k*DIGITS_PER_COMPONENT}")
    return decoded,tokens,codes,err


def wr2(y, pred, w):
    mid=np.average(y, weights=w, axis=0)
    den=np.sum(w[:,None]*(y-mid)**2)
    return float(1-np.sum(w[:,None]*(y-pred)**2)/den) if den>0 else float("nan")


def zr2(z,pred,w):
    mid=np.average(z,weights=w,axis=0)
    den=np.sum(w[:,None]*(z-mid)**2)
    r=1-np.sum(w[:,None]*(z-pred)**2)/den
    rmse=np.sqrt(np.average(np.mean((z-pred)**2,axis=1),weights=w))
    return float(r),float(rmse)


def y_model(name, x, y, w):
    if name=="ridge":
        model=make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x,y,ridge__sample_weight=w)
        return [model]
    fitted=[]
    for j in range(y.shape[1]):
        m=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=12,
              learning_rate=.05,min_samples_leaf=35,l2_regularization=3.0,random_state=93)
        m.fit(x,y[:,j],sample_weight=w)
        fitted.append(m)
    return fitted


def predict(m,method,x):
    if method=="ridge":
        out=m[0].predict(x)
        return out[:,None] if out.ndim==1 else out
    return np.column_stack([obj.predict(x) for obj in m])


def log_cov(residual,w):
    c=residual-np.average(residual,weights=w,axis=0)
    cov=(c*w[:,None]).T@c/w.sum()
    # A small ridge stabilizes near-singular correlated phenotype errors.
    cov += np.eye(c.shape[1])*max(1e-5,float(np.trace(cov))/c.shape[1]*1e-4)
    return cov


def posterior(y, predicted_y, ztrain, clusters, prior, inv_cov, train_codes,
              tr_centers, test_z, test_w, zstd, batch=BATCH):
    """Posterior over exact (quantized) stored codebook, no test Z in estimation."""
    n=len(y); d=ztrain.shape[1]; k=tr_centers.shape[0]
    means=np.empty((n,d),float)
    hits=np.empty(n,dtype=float)
    masses=np.empty(n,float)
    effective=np.empty(n,float)
    kls=np.empty(n,float)
    diversity=np.empty(n,float)
    oracle_err=np.empty(n,float)
    logprior=np.log(prior)
    onehot=np.eye(k,dtype=float)[clusters]
    target_clusters=np.argmin(np.sum(((test_z[:,None,:]-tr_centers[None,:,:])/zstd[None,None,:])**2,axis=2),axis=1)
    for start in range(0,n,batch):
        stop=min(start+batch,n)
        delta=y[start:stop,None,:]-predicted_y[None,:,:]
        # (B,ntrain,Ydim) full conditional Gaussian likelihood.
        loglike=-.5*np.einsum("bni,ij,bnj->bn",delta,inv_cov,delta,optimize=True)
        logpost=loglike+logprior[None,:]
        logpost-=np.max(logpost,axis=1,keepdims=True)
        prob=np.exp(logpost)
        prob/=np.sum(prob,axis=1,keepdims=True)
        means[start:stop]=prob@ztrain
        cmass=prob@onehot
        chosen=np.argsort(-cmass,axis=1)[:,:TOP_K]
        rows=np.arange(stop-start)
        masses[start:stop]=cmass[rows[:,None],chosen].sum(axis=1)
        hits[start:stop]=np.any(chosen==target_clusters[start:stop,None],axis=1)
        effective[start:stop]=np.exp(-np.sum(prob*np.log(prob+1e-300),axis=1))
        kls[start:stop]=np.sum(prob*(np.log(prob+1e-300)-logprior[None,:]),axis=1)
        choices=np.empty((stop-start,TOP_K,d),float)
        for rank in range(TOP_K):
            c=chosen[:,rank]
            for cl in np.unique(c):
                idx=np.where(clusters==cl)[0]
                rows_for_cluster=np.where(c==cl)[0]
                best=idx[np.argmax(prob[np.ix_(rows_for_cluster,idx)],axis=1)]
                choices[rows_for_cluster,rank]=ztrain[best]
                dist=np.linalg.norm((choices[:,:,None,:]-choices[:,None,:,:])/zstd[None,None,None,:],axis=-1)
        diversity[start:stop]=np.mean(dist[:,np.triu_indices(TOP_K,1)[0],np.triu_indices(TOP_K,1)[1]],axis=1)
        oracle_err[start:stop]=np.min(np.sqrt(np.mean((choices-test_z[start:stop,None,:])**2,axis=2)),axis=1)
    weighted=lambda v:float(np.average(v,weights=test_w))
    r,rm=zr2(test_z,means,test_w)
    return dict(weighted_Z_R2=r,weighted_Z_RMSE=rm,cluster_hit_at5=weighted(hits),
                top5_posterior_mass=weighted(masses),effective_codebook_size=weighted(effective),
                KL_posterior_vs_prior_nats=weighted(kls),top5_mean_pairwise_Z_distance_std=weighted(diversity),
                oracle_best_of5_Z_RMSE=weighted(oracle_err))


def example_candidates(query,predicted_y,ztrain,clusters,prior,inv_cov,codes):
    dy=query[None,:]-predicted_y
    ll=-.5*np.einsum("ni,ij,nj->n",dy,inv_cov,dy,optimize=True)
    lp=ll+np.log(prior)
    lp-=lp.max()
    w=np.exp(lp);w/=w.sum()
    cmass=np.bincount(clusters,weights=w,minlength=N_CLUSTERS)
    top=np.argsort(-cmass)[:TOP_K]
    results=[]
    for rank,cl in enumerate(top,1):
        ids=np.where(clusters==cl)[0]
        code_idx=ids[np.argmax(w[ids])]
        results.append(dict(rank=rank,physiology_family=int(cl),
                            family_posterior_mass=float(cmass[cl]),
                            code_posterior_mass=float(w[code_idx]),
                            encoded_decimal_code=str(codes[code_idx]),
                            **{col:float(ztrain[code_idx,j]) for j,col in enumerate(ZCOLS)}))
    return results


def main():
    D=load()
    Z={tag:df[ZCOLS].to_numpy(float) for tag,df in D.items()}
    W={tag:weights(df) for tag,df in D.items()}
    decoded,tokens,codes,errors=interleave(Z)
    F={
        "raw_Z_ridge":(Z,"ridge"),
        "raw_Z_HGB":(Z,"hgb"),
        "decimal_multiscale_HGB":({tag:np.column_stack([decoded[tag],tokens[tag]]) for tag in D},"hgb"),
    }
    ztrain=decoded["train"]
    zmu=np.average(ztrain,weights=W["train"],axis=0)
    zsd=np.sqrt(np.average((ztrain-zmu)**2,weights=W["train"],axis=0))
    zsd=np.where(zsd>1e-8,zsd,1)
    km=KMeans(n_clusters=N_CLUSTERS,random_state=SEED,n_init=10)
    km.fit((ztrain-zmu)/zsd,sample_weight=W["train"])
    cluster=km.labels_
    if len(np.unique(cluster))!=N_CLUSTERS:
        raise RuntimeError("KMeans failed to establish every family")
    prior=W["train"]/W["train"].sum()
    codebook_centers=(km.cluster_centers_*zsd)+zmu
    # Frozen Y-independent candidate-family ranking, for a fair hit@5 baseline.
    baseline_cluster_mass=np.bincount(cluster,weights=prior,minlength=N_CLUSTERS)
    baseline_top_clusters=np.argsort(-baseline_cluster_mass)[:TOP_K]
    baseline_effective=np.exp(-np.sum(prior*np.log(prior+1e-300)))
    print(f"SCRIPT 94 | REAL NHANES Script-73 | 7 components | 3dp decimal-interleaved code")
    print(f"Train 2005-06 n={len(D['train'])}; calibrate 2007-08 n={len(D['cal'])}; tests: "+", ".join(f"{tag} n={len(D[tag])}" for tag in ["2009-2018","2021-2023"]))
    print(f"Roundtrip failures=0; maximum quantization error={max(errors.values()):.7f}; physiology families={N_CLUSTERS}, candidates={TOP_K}")
    print("NOTE: Test-Z is used only for evaluation, never posterior construction.")
    forward=[];reverse=[];hypotheses=[]
    for phenotype,cols in YCASES.items():
        Y={tag:df[cols].to_numpy(float) for tag,df in D.items()}
        # Identical training and test splits for direct Y->Z benchmark.
        direct=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
        direct.fit(Y["train"],Z["train"],ridge__sample_weight=W["train"])
        for tag in ["2009-2018","2021-2023"]:
            pred=direct.predict(Y[tag]);r,rm=zr2(Z[tag],pred,W[tag])
            reverse.append(dict(phenotype=phenotype,forward="direct_Y_to_Z_ridge",period=tag,
                 weighted_Z_R2=r,weighted_Z_RMSE=rm))
            prior_z=np.broadcast_to(np.average(ztrain,weights=prior,axis=0),Z[tag].shape)
            rp,rmp=zr2(Z[tag],prior_z,W[tag])
            test_family=km.predict((Z[tag]-zmu)/zsd)
            hit=np.isin(test_family,baseline_top_clusters).astype(float)
            reverse.append(dict(phenotype=phenotype,forward="prior_only",period=tag,
                 weighted_Z_R2=rp,weighted_Z_RMSE=rmp,
                 cluster_hit_at5=float(np.average(hit,weights=W[tag])),
                 top5_posterior_mass=float(baseline_cluster_mass[baseline_top_clusters].sum()),
                 effective_codebook_size=float(baseline_effective),
                 KL_posterior_vs_prior_nats=0.0))
        for name,(feat,modelname) in F.items():
            model=y_model(modelname,feat["train"],Y["train"],W["train"])
            train_pred=predict(model,modelname,feat["train"])
            cal_pred=predict(model,modelname,feat["cal"])
            cov=log_cov(Y["cal"]-cal_pred,W["cal"])
            inv_cov=np.linalg.inv(cov)
            for tag in ["train","cal","2009-2018","2021-2023"]:
                yp=predict(model,modelname,feat[tag])
                forward.append(dict(phenotype=phenotype,forward=name,period=tag,
                    weighted_Y_R2=wr2(Y[tag],yp,W[tag]),
                    weighted_Y_RMSE=float(np.sqrt(np.average(np.mean((Y[tag]-yp)**2,axis=1),weights=W[tag]))),
                    phenotype_sd=float(np.sqrt(np.average(np.mean((Y[tag]-np.average(Y[tag],weights=W[tag],axis=0))**2,axis=1),weights=W[tag])))))
            for tag in ["2009-2018","2021-2023"]:
                res=posterior(Y[tag],train_pred,ztrain,cluster,prior,inv_cov,codes["train"],
                              codebook_centers,Z[tag],W[tag],zsd)
                reverse.append(dict(phenotype=phenotype,forward=name,period=tag,**res))
            # Hypothetical Y queries: non-patient examples, drawn from training codes.
            queries=[[5.],[10.],[15.]] if phenotype=="PHQ_TOTAL" else [[2.,3.],[3.,7.],[5.,10.]]
            for q in queries:
                for row in example_candidates(np.array(q),train_pred,ztrain,cluster,prior,inv_cov,codes["train"]):
                    hypotheses.append(dict(phenotype=phenotype,forward=name,query_Y="/".join(map(str,q)),**row))

    OUT.mkdir(parents=True,exist_ok=True)
    fdf=pd.DataFrame(forward);rdf=pd.DataFrame(reverse)
    fdf.to_csv(OUT/"94_forward_strength.csv",index=False)
    rdf.to_csv(OUT/"94_reverse_multistate.csv",index=False)
    pd.DataFrame(hypotheses).to_csv(OUT/"94_example_candidate_codes.csv",index=False)
    manifest={"source":"REAL NHANES-derived Script-73 only; no synthetic fallback",
        "train_cycle":"0506", "calibration_cycle":"0708", "heldout_periods":{"2009-2018":len(D["2009-2018"]),"2021-2023":len(D["2021-2023"])},
        "train_n":len(D["train"]),"cal_n":len(D["cal"]),"frozen_Z":ZCOLS,
        "quantization_decimals":DECIMALS,"digits_per_component":DIGITS_PER_COMPONENT,
        "max_Z_quantization_error":errors,"posterior_family_count":N_CLUSTERS,"top_k":TOP_K,
        "notes":["No original frozen decompositions or existing numbered outputs changed",
           "Calibrated Gaussian phenotype likelihood, survey-weight empirical codebook prior",
           "No claims of actual patient-level longitudinal dynamics or causal inference",
           "Somatic and CogAff use two measured dimensions; may carry more information than PHQ total",
           "Codebook consists of 2005-06 observation codes; exact individuals cannot be recovered from phenotype alone",
           "Cluster hit@5 compares a true held-out state cluster solely for retrospective evaluation",
           "Oracle best-of-5 error uses held-out true Z; NOT deployable reconstruction performance",
           "Physiology families formed from training Z, not from a code-specific physical law",
           "Posterior/top-five ranking compared with Y-independent fixed prior top-five family baseline",
           "Posterior probabilities are model-based; NO conformal guarantee, no guarantee of cross-cycle calibration",
           "Survey weights used as empirical-prior proxy, not true physiological prevalence distribution",
           "Forward rawZ Y scores without additional covariates; not directly comparable to adjusted Script-73 maps"]}
    (OUT/"94_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print("\nHELD-OUT FORWARD Y R2 (2009-18)")
    print(fdf[fdf.period.eq("2009-2018")][["phenotype","forward","weighted_Y_R2"]].round(4).to_string(index=False))
    print("\nHELD-OUT REVERSE Z R2 / top-5 physiology-family hit (2009-18)")
    print(rdf[rdf.period.eq("2009-2018")][["phenotype","forward","weighted_Z_R2","cluster_hit_at5","effective_codebook_size"]].round(4).to_string(index=False))
    print("\nSaved EDA/Results/94_* (4 files)")

if __name__=="__main__":
    main()
