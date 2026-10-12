#!/usr/bin/env python3
"""Experiment 91: empirical Y->Z identifiability and codebook baseline on NHANES.

Reads the EXISTING Script-73 patient-level file and frozen Jacobian / PHQ beta
CSV results (Scripts 74 and 75). No synthetic data, no random patient creation,
no revision to the frozen X<->Z map. Training is 2005-06 ONLY; 2007-08 and
later cycles are held out. Compare SAME closest-PHQ-patient retrieval:
  (A) directly return their observed Z
  (B) return their losslessly coded, quantized Z, then decode
The selected patient is identical. Hence code representation cannot add
new evidence and may add quantization error. A separate ridge Y->Z is a
comparison baseline, NOT the original Script-81 exact replication.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

# Import encoder without installing a package; both scripts should be in EDA/.
from importlib import util

HERE=Path(__file__).resolve().parent
spec=util.spec_from_file_location("encoder90",HERE/"90_nhanes_encoding_audit.py")
if spec is None or spec.loader is None:raise ImportError("Put Script 90 in the same EDA/ folder")
encoder90=util.module_from_spec(spec)
spec.loader.exec_module(encoder90)

ZCOLS=encoder90.ZCOLS
NAME=encoder90.NAME
ZSHORT=["Shared1","Shared2","Discord1","Discord2","A_private3","A_private4","G_private3"]
XSHORT=["Hb","RBC","MCV","RDW","HbA1c","Glucose","logInsulin"]


def load_matrix(path: Path, rows: list[str], cols: list[str]) -> np.ndarray:
    if not path.exists():raise FileNotFoundError(f"Missing frozen input: {path}")
    d=pd.read_csv(path,index_col=0)
    return d.loc[rows,cols].to_numpy(float)


def read(path: Path) -> pd.DataFrame:
    if not path.exists():raise FileNotFoundError(f"Missing real Script 73 input: {path}")
    hdr=set(pd.read_csv(path,nrows=0).columns)
    required={"SEQN","CYCLE","PERIOD","PHQ9_TOTAL","SOMATIC_SCORE","COGAFF_SUM",*ZCOLS}
    missing=sorted(required-hdr)
    if missing:raise ValueError(f"Script 73 input missing columns: {missing}")
    extra=[x for x in ["SURVEY_WT","ELIGIBLE_ADULT_NONPREG"] if x in hdr]
    d=pd.read_csv(path,usecols=list(required)+extra,low_memory=False)
    for c in ZCOLS+["PHQ9_TOTAL","SOMATIC_SCORE","COGAFF_SUM"]+(["SURVEY_WT"] if "SURVEY_WT" in d else []):
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d["CYCLE"]=(d["CYCLE"].astype(str).str.strip().str.replace(r"\.0$","",regex=True).str.zfill(4))
    if "ELIGIBLE_ADULT_NONPREG" in d:
        d=d[d["ELIGIBLE_ADULT_NONPREG"].eq(1)]
    valid=d[ZCOLS+["PHQ9_TOTAL","SOMATIC_SCORE","COGAFF_SUM"]].notna().all(axis=1)
    if "SURVEY_WT" in d:valid &= d["SURVEY_WT"].gt(0)
    return d[valid].copy()


def weighted_mean_std(v: np.ndarray, w: np.ndarray):
    m=np.average(v,axis=0,weights=w)
    sd=np.sqrt(np.average((v-m)**2,axis=0,weights=w))
    return m, np.where(sd>1e-10,sd,1.0)


def weighted_r2(true: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    mu=np.average(true,axis=0,weights=w)
    num=float(np.sum(w[:,None]*(true-pred)**2))
    den=float(np.sum(w[:,None]*(true-mu)**2))
    return 1-num/den if den>0 else float('nan')


def weighted_rmse(true:np.ndarray,pred:np.ndarray,w:np.ndarray)->float:
    return float(np.sqrt(np.sum(w[:,None]*(true-pred)**2)/(len(ZCOLS)*np.sum(w))))


def period_group(d:pd.DataFrame, key:str)->pd.DataFrame:
    if key=="2007-2008":return d[d["CYCLE"].eq("0708")]
    return d[d["PERIOD"].astype(str).eq(key)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=HERE/"Results"/"73_frozen_phenotype_map_input.csv")
    p.add_argument("--results-dir",type=Path,default=HERE/"Results")
    p.add_argument("--out",type=Path,default=HERE/"Results")
    p.add_argument("--decimals",type=int,default=3)
    p.add_argument("--alpha",type=float,default=10.0,help="Ridge penalty for the new baseline (not Script 81 replication)")
    a=p.parse_args()
    if not 0<=a.decimals<=6:p.error("decimals must be in 0..6")
    jf=load_matrix(a.results_dir/"74_dZ_dX_matrix.csv",ZSHORT,XSHORT)
    jg=load_matrix(a.results_dir/"74_dX_dZ_matrix.csv",XSHORT,ZSHORT)
    phen=pd.read_csv(a.results_dir/"75_phenotype_derivative_matrix.csv")
    phen=phen[phen["period"].eq("2005-2008")].set_index("outcome")
    beta=phen.loc["PHQ9_TOTAL",ZSHORT].to_numpy(float)
    B=phen.loc[["SOMATIC","COGAFF"],ZSHORT].to_numpy(float)
    if not (np.allclose(jf@jg,np.eye(7),atol=1e-9) and np.allclose(jg@jf,np.eye(7),atol=1e-9)):
        raise AssertionError("Existing frozen forward/reverse maps are not mutual inverses")
    df=read(a.data)
    # Real-row physiology impact of quantization, expressed in raw biomarker units.
    # g(z) = J_g z + b_g, so g(z_quant)-g(z) = J_g (z_quant-z).
    Zall=df[ZCOLS].to_numpy(float)
    Zquant=np.array([[encoder90.quant(v,a.decimals)/(10**a.decimals)
                      for v in row] for row in Zall],dtype=float)
    DX=(Zquant-Zall)@jg.T
    raw_precision=[]
    for j,name in enumerate(XSHORT):
        raw_precision.append({"biomarker":name,"real_rows":len(df),
                              "mean_abs_error_from_Z_quantization":float(np.mean(np.abs(DX[:,j]))),
                              "max_abs_error_from_Z_quantization":float(np.max(np.abs(DX[:,j]))),
                              "theoretical_worst_case_abs_error_bound":float(.5*10**(-a.decimals)*np.sum(np.abs(jg[j,:]))),
                              "units":"original Script 74 raw X units"})
    train=df[df["CYCLE"].eq("0506")].copy()
    if len(train)<100:raise ValueError("Need sufficient real 2005–06 complete cases in CYCLE 0506")
    labels={"Y1_PHQ_TOTAL":["PHQ9_TOTAL"],"Y2_SOMATIC_COGAFF":["SOMATIC_SCORE","COGAFF_SUM"]}
    Xtrain=train[ZCOLS].to_numpy(float)
    wtr=(train["SURVEY_WT"].to_numpy(float) if "SURVEY_WT" in train else np.ones(len(train)))
    result=[]
    for yname, ycols in labels.items():
        Ytrain=train[ycols].to_numpy(float)
        ymu,ysd=weighted_mean_std(Ytrain,wtr)
        zmu,zsd=weighted_mean_std(Xtrain,wtr)
        Ys=(Ytrain-ymu)/ysd
        Zs=(Xtrain-zmu)/zsd
        ridge=Ridge(alpha=a.alpha).fit(Ys,Zs,sample_weight=wtr)
        nn=NearestNeighbors(n_neighbors=1,algorithm="auto").fit(Ys)
        # Codebook stores actual train Z as a fixed-width encoded string.
        states=[tuple(encoder90.to_nat(encoder90.quant(v,a.decimals)) for v in z) for z in Xtrain]
        width=max(len(str(x)) for st in states for x in st)
        codes=[encoder90.interleave(st,width) for st in states]
        decoded=np.asarray([[encoder90.from_nat(k)/(10**a.decimals)
                             for k in encoder90.deinterleave(code,7,width)] for code in codes])
        assert np.max(np.abs(decoded-Xtrain))<=.5*10**(-a.decimals)+1e-12
        for period in ["2007-2008","2009-2018","2021-2023"]:
            te=period_group(df,period)
            if te.empty:
                continue
            Ytest=te[ycols].to_numpy(float)
            Ztrue=te[ZCOLS].to_numpy(float)
            wte=(te["SURVEY_WT"].to_numpy(float) if "SURVEY_WT" in te else np.ones(len(te)))
            Ys_te=(Ytest-ymu)/ysd
            pred_ridge=ridge.predict(Ys_te)*zsd+zmu
            matched=nn.kneighbors(Ys_te,return_distance=False).ravel()
            direct=Xtrain[matched]                # return original train row
            codebook=decoded[matched]             # return SAME train row, decoded from code
            for method, pred in (("ridge_Y_to_Z",pred_ridge),
                                 ("nearest_Y_then_direct_Z",direct),
                                 ("nearest_Y_then_encoded_Z",codebook)):
                result.append({"phenotype":yname,"test_period":period,
                               "method":method,"train_period":"2005-2006",
                               "train_n":len(train),"test_n":len(te),
                               "weighted_Z_R2":weighted_r2(Ztrue,pred,wte),
                               "weighted_Z_RMSE":weighted_rmse(Ztrue,pred,wte)})
            print(f"{yname}, {period}: n={len(te)}; same matched patient direct vs codebook; "
                  f"max codebook quantization difference={np.max(np.abs(direct-codebook)):.6g}")
    # Mathematical nullspace: change Shared1 and Aprivate3 to cancel PHQ9 slope.
    delta=np.zeros(7); delta[0]=.25; delta[4]=-.25*beta[0]/beta[4]
    assert abs(beta@delta)<1e-11
    effect_x=jg@delta
    # All data were real for predictive comparisons; the counterexample is algebraic.
    audit={"data":"real complete rows from EDA/Results/73_frozen_phenotype_map_input.csv",
           "n_real_complete_rows":len(df),"train_0506":len(train),
           "rank_Jf":int(np.linalg.matrix_rank(jf)),
           "inverse_error_JfJg":float(np.max(np.abs(jf@jg-np.eye(7)))),
           "inverse_error_JgJf":float(np.max(np.abs(jg@jf-np.eye(7)))),
           "PHQ9_linear_nullspace_dimensions":6,
           "Somatic_plus_CogAff_linear_nullspace_dimensions":7-int(np.linalg.matrix_rank(B)),
           "constructed_nullspace_delta_Z":dict(zip(ZSHORT,map(float,delta))),
           "predicted_PHQ_delta_at_fixed_covariates":float(beta@delta),
           "implied_raw_biomarker_difference":dict(zip(XSHORT,map(float,effect_x))),
           "warning":"This is not exact replication of Script 81's tuning/training; training exclusively 2005-06 prevents 2007-08 leakage. Codebook nearest-neighbor deliberately matches identical train patient indices. Reversible representation does not imply phenotype identifiability."}
    a.out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(result)
    frame.to_csv(a.out/"91_real_reverse_baselines.csv",index=False)
    pd.DataFrame(raw_precision).to_csv(a.out/"91_real_quantization_to_raw_X_error.csv",index=False)
    (a.out/"91_frozen_map_identifiability.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"\nResults saved: {a.out.resolve()}")


if __name__=="__main__":
    main()
