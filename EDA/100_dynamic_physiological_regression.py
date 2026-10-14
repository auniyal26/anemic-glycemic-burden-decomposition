#!/usr/bin/env python3
"""Script 100 | Interpretable dynamic mixture-of-regressions on frozen NHANES physiology.

Run from the repository root:
    python EDA/100_dynamic_physiological_regression.py
    python EDA/100_dynamic_physiological_regression.py --self-test

Uses Script 98's real-data loader, exact matched cohort, covariate construction
and frozen Script-72/73 Z features. No held-out phenotype enters fitting,
gating, cross-validation, or selection. Later-cycle diagnostics are exploratory:
Scripts 72-99 have already inspected these periods repeatedly.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "EDA"
RESULTS = EDA / "Results"
P98 = EDA / "98_regularized_phenotype_equation.py"
SEED = 100
CV_SEED = 98   # deliberately reuse Script 98's partitioning for comparisons
TRAIN = "train_2005-2008"
TEST = "test_2009-2018"
VALIDATE = "validate_2021-2023"
PERIODS = (TRAIN, TEST, VALIDATE)
Z = ["SP_SHARED1", "SP_SHARED2", "SP_DISCORD1", "SP_DISCORD2",
     "SP_A_PRIVATE3", "SP_A_PRIVATE4", "SP_G_PRIVATE3"]
BLOCKS = {"Shared": Z[:2], "Discordant": Z[2:4],
          "A_private": Z[4:6], "G_private": Z[6:]}
TARGETS = {"PHQ_total": "PHQ9_TOTAL", "Somatic": "SOMATIC_SCORE", "CogAff": "COGAFF_SUM"}


def load_98():
    if not P98.is_file():
        raise FileNotFoundError("Script 98 required in same EDA directory: " + str(P98))
    spec = importlib.util.spec_from_file_location("script98_for_100", P98)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    # Script 98 initializes these module globals inside main(), but Script 100
    # imports its loader directly. Initialize them before load_real() so the
    # NHANES coverage audit and feature builder can run without NameError.
    mod.yield_msg = []
    mod.normalizer = {}
    mod.categories_reference = {}
    mod.categories_feature_names = []
    return mod


def metrics(y, pred, weight):
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    w = np.asarray(weight, dtype=float)
    err = pred - y
    den = np.average((y - np.average(y, weights=w))**2, weights=w)
    mse = np.average(err**2, weights=w)
    return {"R2": float(1-mse/den) if den>0 else float("nan"),
            "RMSE": float(np.sqrt(mse)), "MAE": float(np.average(abs(err), weights=w)),
            "bias": float(np.average(err, weights=w)), "MSE": float(mse)}


def category_features(mod, frame):
    mod.normalizer = {}; mod.categories_reference = {}
    mod.categories_feature_names = []
    out = mod.build_features(frame)
    mod.categories_feature_names = [q for q in out if "=" in q]
    return out


def baseline_specs():
    return [
        {"family": "global_z_ridge", "alpha": 10., "tau": None, "gate": "none", "ratio": None},
        {"family": "global_z_ridge", "alpha": 100., "tau": None, "gate": "none", "ratio": None},
        {"family": "global_z_pairwise", "alpha": 0.05, "tau": None, "gate": "none", "ratio": 0.5},
        {"family": "global_z_pairwise", "alpha": 0.05, "tau": None, "gate": "none", "ratio": 0.8},
    ]


def all_specs():
    specs = baseline_specs()
    # These gates have different assumptions. Model selection uses TRAIN CV only.
    for gate, a, tau in itertools.product(("A_axis", "G_axis", "AG_axes", "kmeans2"),
                                           (10., 100.), (0.75, 1.5)):
        specs.append({"family": "smooth_experts", "gate": gate, "alpha": a,
                      "tau": tau, "ratio": None})
    return specs


def describe(s, scope):
    return (f"{scope}/{s['family']}/{s['gate']}/alpha={s['alpha']:g}"
            + (f"/tau={s['tau']:g}" if s['tau'] is not None else "")
            + (f"/l1={s['ratio']:g}" if s['ratio'] is not None else ""))


class SmoothExperts:
    """Smooth convex mixture of K Ridge equations. Gating determined by train Z alone.

    Gate weights are nonlinear functions of standardized frozen Z.
    Base experts are linear in raw Z and covariate dummy features.
    They are NOT outcome-derived clusters, and NOT medical diagnoses.
    """
    def __init__(self, gate="A_axis", alpha=100., tau=1.0):
        self.gate_type = gate
        self.alpha = float(alpha)
        self.tau = float(tau)

    def _gates(self, Z_raw):
        q = self.zsc.transform(np.asarray(Z_raw,float))
        if self.gate_type == "A_axis":
            p = expit(q[:, 4] / self.tau)
            return np.stack([1-p, p], axis=1)
        if self.gate_type == "G_axis":
            p = expit(q[:, 6] / self.tau)
            return np.stack([1-p, p], axis=1)
        if self.gate_type == "AG_axes":
            a = expit(q[:,4] / self.tau); g = expit(q[:,6] / self.tau)
            return np.stack([(1-a)*(1-g), (1-a)*g, a*(1-g), a*g], axis=1)
        if self.gate_type == "kmeans2":
            dist = np.sum((q[:,None,:] - self.centers[None,:,:])**2, axis=2)
            return softmax(-dist / (self.tau*self.dscale), axis=1)
        raise ValueError(self.gate_type)

    def fit(self, feature_matrix, z_matrix, y, weights):
        x = np.asarray(feature_matrix,float)
        z = np.asarray(z_matrix,float)
        y = np.asarray(y,float)
        w = np.asarray(weights,float)
        if not (len(x)==len(z)==len(y)==len(w)) or len(x)<25:
            raise ValueError("Invalid training shape")
        if np.any(w<=0) or not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError("Invalid weights/features")
        self.xsc = StandardScaler().fit(x,sample_weight=w)
        self.zsc = StandardScaler().fit(z,sample_weight=w)
        xx = self.xsc.transform(x)
        zz = self.zsc.transform(z)
        if self.gate_type == "kmeans2":
            km = KMeans(n_clusters=2, random_state=SEED, n_init=5)
            km.fit(zz, sample_weight=w/w.mean())
            self.centers = np.asarray(km.cluster_centers_,float)
            dist = np.sum((zz[:,None,:]-self.centers[None,:,:])**2,axis=2)
            self.dscale = max(float(np.median(np.min(dist,axis=1))), 1e-8)
        gates = self._gates(z)
        self.experts = []
        for j in range(gates.shape[1]):
            # Weighted expert likelihood; alpha is relative to each expert's
            # normalized effective weighted observations, not arbitrary NHANES WT units.
            ew = w*gates[:,j]
            if float(ew.sum()) <= 1e-10:
                raise ValueError("Empty gate")
            ew = ew / ew.mean()
            mod = Ridge(alpha=self.alpha, random_state=SEED)
            mod.fit(xx,y,sample_weight=ew)
            self.experts.append(mod)
        self.n_experts = len(self.experts)
        return self

    def predict(self, x, z):
        xx = self.xsc.transform(np.asarray(x,float))
        p = self._gates(z)
        raw = np.column_stack([m.predict(xx) for m in self.experts])
        return (p*raw).sum(axis=1)

    def equations(self, names):
        rows=[]
        for j,m in enumerate(self.experts):
            beta=np.asarray(m.coef_,float)/self.xsc.scale_
            intercept=float(m.intercept_ - beta@self.xsc.mean_)
            rows.append({"expert":j, "intercept":intercept,
                         "terms":dict(zip(names,[float(t) for t in beta]))})
        return rows

    def gate_details(self):
        d={"gate":self.gate_type,"temperature":self.tau,
           "z_training_mean":self.zsc.mean_.tolist(),"z_training_scale":self.zsc.scale_.tolist()}
        if self.gate_type=="kmeans2":
            d["centers_standardized_Z"]=self.centers.tolist()
            d["distance_scale"]=self.dscale
        return d


class Global:
    def __init__(self, mod, s): self.mod=mod; self.s=s
    def fit(self,x,z,y,w):
        meth="Ridge" if self.s["family"]=="global_z_ridge" else "ElasticNet"
        self.sc,self.m=self.mod.fit(x,y,w,meth,self.s["alpha"],self.s["ratio"])
        return self
    def predict(self,x,z): return self.mod.prediction(self.sc,self.m,x)
    def equations(self,names):
        beta=np.asarray(self.m.coef_,float)/self.sc.scale_
        return [{"expert":0,"intercept":float(self.m.intercept_-beta@self.sc.mean_),
                 "terms":dict(zip(names,[float(t) for t in beta]))}]
    def gate_details(self): return {"gate":"none","temperature":None}
    def _gates(self,z):return np.ones((len(z),1))


def build_model(mod,s):
    return (SmoothExperts(gate=s["gate"],alpha=s["alpha"],tau=s["tau"])
            if s["family"]=="smooth_experts" else Global(mod,s))


def feature_names(mod, scope, spec):
    features=mod.covariates(scope)+list(Z)
    if spec["family"]=="global_z_pairwise":
        features += [a+"*"+b for a,b in itertools.combinations(Z,2)]
    return features


def prepare_real(mod):
    d=mod.load_real()
    F=category_features(mod,d)
    # Require exact consistency with prior Scripts 98–99.
    counts=d.groupby("SPLIT").size().to_dict()
    if not all(p in counts for p in PERIODS):
        raise AssertionError("Missing cohort split")
    return d,F


def synthetic_data(mod, n=80):
    # Synthetic data for programmatic SELF-TEST only; never used for real analysis.
    rng=np.random.default_rng(100)
    c=np.repeat(["0506","0708","0910","1112","1314","1516","1718","2123"],n)
    d=pd.DataFrame({"CYCLE":c,"SPLIT":[mod.PERIOD[k] for k in c],
                    "SURVEY_WT":rng.uniform(0.8,2,len(c))})
    for v in mod.X+mod.Z:d[v]=rng.normal(size=len(d))
    for v in mod.NUM_COV:d[v]=rng.normal(size=len(d))
    for v in mod.CAT_COV:d[v]=rng.integers(1,4,size=len(d))
    q=d[Z].to_numpy()
    d["SOMATIC_SCORE"]=2 + .5*q[:,4]-.3*q[:,3]+.4*np.tanh(q[:,4])*q[:,6] +rng.normal(0,.6,len(d))
    d["COGAFF_SUM"]=1+.2*q[:,0]+rng.normal(0,.8,len(d))
    d["PHQ9_TOTAL"]=d.SOMATIC_SCORE+d.COGAFF_SUM
    return d,category_features(mod,d)


def model_selection(mod,d,F,nfolds=5,smoke=False):
    tri=np.flatnonzero(d.SPLIT.eq(TRAIN).to_numpy())
    ysets={t:d[v].to_numpy(float) for t,v in TARGETS.items()}
    w=d.SURVEY_WT.to_numpy(float)
    z=d[Z].to_numpy(float)
    cv=list(StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=CV_SEED)
             .split(np.arange(len(tri)),d.CYCLE.to_numpy(str)[tri]))
    configs=all_specs() if not smoke else baseline_specs()[:2]+[
        {"family":"smooth_experts","gate":"A_axis","alpha":10.,"tau":1.5,"ratio":None},
        {"family":"smooth_experts","gate":"kmeans2","alpha":100.,"tau":1.5,"ratio":None}]
    Xcache={}
    for scope in ("lean","full"):
        for s in configs:
            key=(scope,s["family"])
            if key not in Xcache:
                names=feature_names(mod,scope,s)
                Xcache[key]=(names,F[names].to_numpy(float))
    rows=[]; winners={}; fitted={}; selected=[]
    print(f"TRAIN-only CV: {nfolds} folds × {len(configs)} specifications × 2 covariate scopes × 3 targets")
    for target,y in ysets.items():
        target_records=[]
        for scope in ("lean","full"):
            print(f'  CV fitting {target} / {scope} ...', flush=True)
            for spec in configs:
                names,x=Xcache[(scope,spec["family"])]
                fold_mse=[]
                for itr,iva in cv:
                    it=tri[itr]; iv=tri[iva]
                    model=build_model(mod,spec).fit(x[it],z[it],y[it],w[it])
                    pred=model.predict(x[iv],z[iv])
                    fold_mse.append(metrics(y[iv],pred,w[iv])["MSE"])
                rec={"target":target,"scope":scope,**spec,"features":len(names),
                     "label":describe(spec,scope),
                     "cv_weighted_MSE":float(np.mean(fold_mse)),
                     "cv_sd_MSE":float(np.std(fold_mse))}
                target_records.append(rec);rows.append(rec)
        # No held-out outcomes used for winner selection. Prefer simpler on ties.
        ordered=sorted(target_records,key=lambda r:(r["cv_weighted_MSE"],
                        r["family"]=="smooth_experts", r["features"],r["label"]))
        selected_gated=next((r for r in ordered if r["family"]=="smooth_experts"),None)
        selected_global=next(r for r in ordered if r["family"]!="smooth_experts")
        winners[target] = {"cv_winner":ordered[0],"best_dynamic":selected_gated,
                           "best_global":selected_global}
        print(f"  {target}: CV winner = {ordered[0]['label']} (MSE={ordered[0]['cv_weighted_MSE']:.4f})")
        print(f"          best dynamic = {selected_gated['label']} (MSE={selected_gated['cv_weighted_MSE']:.4f})")
        # Fit full training ONLY, for the CV winner + best gated + best global,
        # plus canonical Ridge-100 Z main lean/full matching Script 98.
        keep={q["label"]:q for q in (ordered[0],selected_gated,selected_global)}
        for scope in ("lean","full"):
            ridge100=next(r for r in target_records if r["scope"]==scope and
                          r["family"]=="global_z_ridge" and r["alpha"]==100.)
            keep[ridge100["label"]]=ridge100
        for label, rec in keep.items():
            spec={k:rec[k] for k in ("family","alpha","ratio","tau","gate")}
            names,x=Xcache[(rec["scope"],rec["family"])]
            model=build_model(mod,spec).fit(x[tri],z[tri],y[tri],w[tri])
            fitted[(target,label)]={"rec":rec,"model":model,"names":names,"x":x}
    return pd.DataFrame(rows),winners,fitted


def perturb_prediction(mod, model, x, z, names, zi, step):
    """Evaluate finite difference, rebuilding standardized Z*Z interactions."""
    vv=Z[zi]
    modified_z=np.asarray(z,float).copy()
    modified_z[:,zi]+=step
    modified_x=np.asarray(x,float).copy()
    modified_x[:,names.index(vv)]+=step
    for col,feat in enumerate(names):
        if '*' not in feat:continue
        a,b=feat.split('*')
        if vv not in (a,b):continue
        ja=Z.index(a);jb=Z.index(b)
        na=mod.normalizer[a];nb=mod.normalizer[b]
        modified_x[:,col]=((modified_z[:,ja]-na['mean'])/na['sd']*
                           (modified_z[:,jb]-nb['mean'])/nb['sd'])
    return model.predict(modified_x,modified_z)


def write_outputs(mod,d,F,selection,winners,fitted,smoke=False):
    results=(RESULTS/'100_smoke') if smoke else RESULTS
    results.mkdir(parents=True,exist_ok=True)
    w=d.SURVEY_WT.to_numpy(float)
    z=d[Z].to_numpy(float)
    score=[]; betas=[]; gates=[]; block=[]; derivs=[]; preds={}; details={}
    for (target,label),o in fitted.items():
        y=d[TARGETS[target]].to_numpy(float)
        model=o["model"];x=o["x"];rec=o["rec"]
        details[(target,label)]=model
        for expert in model.equations(o["names"]):
            for name,beta in expert["terms"].items():
                betas.append({"target":target,"label":label,"expert":expert["expert"],
                              "term":name,"coefficient":beta})
            betas.append({"target":target,"label":label,"expert":expert["expert"],
                          "term":"(Intercept)","coefficient":expert["intercept"]})
        for period in PERIODS:
            mask=d.SPLIT.eq(period).to_numpy(); ix=np.flatnonzero(mask)
            p=model.predict(x[mask],z[mask]);preds[(target,label,period)]=p
            score.append({"target":target,"label":label,"period":period,"n":int(mask.sum()),
                          **metrics(y[mask],p,w[mask])})
            g=model._gates(z[mask])
            for j in range(g.shape[1]):
                gates.append({"target":target,"label":label,"period":period,"expert":j,
                              "mean_gate_weight":float(np.average(g[:,j],weights=w[mask]))})
            # Effective partial derivatives include both the expert slopes
            # and changes in the dynamic gate. These are model sensitivities,
            # NOT a causal derivative or a solved PDE.
            for iz,v in enumerate(Z):
                step=1e-4*float(mod.normalizer[v]['sd'])
                plus=perturb_prediction(mod,model,x[mask],z[mask],o['names'],iz,step)
                minus=perturb_prediction(mod,model,x[mask],z[mask],o['names'],iz,-step)
                local=(plus-minus)/(2*step)
                derivs.append({'target':target,'label':label,'period':period,
                               'variable':v,'weighted_mean_dY_dZ':float(np.average(local,weights=w[mask])),
                               'weighted_mean_abs_dY_dZ':float(np.average(abs(local),weights=w[mask])),
                               'weighted_positive_fraction':float(np.average((local>0).astype(float),weights=w[mask])),
                               'note':'Derivative of fitted dynamic equation; not causal; not a PDE solution'})
            # Physiological block contrast: reset one block to its TRAIN mean
            # and re-evaluate both experts AND the gate. Not additive or causal.
            if rec["family"]=="smooth_experts":
                for name,values in BLOCKS.items():
                    zcf=z[mask].copy();xcf=x[mask].copy()
                    for variable in values:
                        idxz=Z.index(variable)
                        train_mean=float(model.zsc.mean_[idxz])
                        zcf[:,idxz]=train_mean
                        xcf[:,o["names"].index(variable)]=train_mean
                    counter=model.predict(xcf,zcf)
                    diff=p-counter
                    block.append({"target":target,"label":label,"period":period,"block":name,
                                  "mean_signed_delta":float(np.average(diff,weights=w[mask])),
                                  "mean_abs_delta":float(np.average(np.abs(diff),weights=w[mask])),
                                  "rms_delta":float(np.sqrt(np.average(diff**2,weights=w[mask]))),
                                  "note":"Model perturbation (not additive and not causal)"})
    scores=pd.DataFrame(score);coefs=pd.DataFrame(betas);gs=pd.DataFrame(gates);bs=pd.DataFrame(block)
    selection.to_csv(results/'100_cv_model_selection.csv',index=False)
    scores.to_csv(results/'100_frozen_performance.csv',index=False)
    coefs.to_csv(results/'100_expert_coefficients.csv',index=False)
    gs.to_csv(results/'100_gate_occupancy.csv',index=False)
    bs.to_csv(results/'100_block_contrasts.csv',index=False)
    pd.DataFrame(derivs).to_csv(results/'100_dynamic_derivatives.csv',index=False)
    # Cross-phenotype algebra: Y_total = Y_somatic + Y_CogAff
    coherence=[]
    for period in PERIODS:
        s=winners['Somatic']['cv_winner']['label'];c=winners['CogAff']['cv_winner']['label'];t=winners['PHQ_total']['cv_winner']['label']
        pm=preds[('Somatic',s,period)]+preds[('CogAff',c,period)]
        pdirect=preds[('PHQ_total',t,period)]
        mask=d.SPLIT.eq(period).to_numpy()
        coherence.append({'period':period,'n':int(mask.sum()),
                          'sum_somatic_cogaff_R2':metrics(d.PHQ9_TOTAL.to_numpy(float)[mask],pm,w[mask])['R2'],
                          'direct_total_R2':metrics(d.PHQ9_TOTAL.to_numpy(float)[mask],pdirect,w[mask])['R2']})
    pd.DataFrame(coherence).to_csv(results/'100_phq_coherence.csv',index=False)
    # Confirm canonical Script-98 Ridge-100 Z_main predictions are numerically
    # reproduced on the unchanged matched NHANES cohort, when baseline exists.
    baseline_file=RESULTS/'98_frozen_holdout_performance.csv'
    if baseline_file.is_file() and not smoke:
        old=pd.read_csv(baseline_file)
        audit=[]
        for target in TARGETS:
            for scope in ('lean','full'):
                label=f'{scope}/global_z_ridge/none/alpha=100'
                for period in (TEST,VALIDATE):
                    expected=old[(old.outcome==target)&(old.scope==scope)&
                                 (old.design=='Z_main')&(old.period==period)&
                                 (old.method=='Ridge')&(old.alpha==100.)]
                    new=scores[(scores.target==target)&(scores.label==label)&
                               (scores.period==period)]
                    if len(expected)!=1 or len(new)!=1:
                        raise AssertionError('Unable to audit Script-98 baseline: '+str((target,scope,period)))
                    delta=float(new.R2.iloc[0]-expected.weighted_R2.iloc[0])
                    audit.append({'target':target,'scope':scope,'period':period,'R2_difference':delta})
                    if abs(delta)>1e-8:
                        raise AssertionError(f'Script-98 baseline reproduction drift: {target} {scope} {period} delta={delta}')
        pd.DataFrame(audit).to_csv(results/'100_script98_reproduction.csv',index=False)
        print('Verified: canonical Script-98 Z_main coefficients/predictions reproduced.')

    equation_lines=['# Script 100 — Dynamically weighted numerical regression equations','',
                    'Each expert is an independently fitted Ridge regression on NHANES 2005–08.','',
                    r'**Complete model:** $Y_i=\sum_{k=1}^{K}w_k(Z_i)(\beta_{0k}+\sum_{j=1}^{7}\beta_{jk}Z_{ij}+\gamma_k^\top C_i)+\varepsilon_i$.',
                    'The gate weights are train-only smooth functions of frozen physiological Z.','',
                    '### Gate definitions', '',
                    r'- A_axis: $p=\sigma(z^*_{A3}/\tau)$, weights $(1-p,p)$.',
                    r'- G_axis: $p=\sigma(z^*_{G3}/\tau)$, weights $(1-p,p)$.',
                    '- AG_axes: independent A/G soft gates, 4 combinations.',
                    r'- kmeans2: $w_k=\mathrm{softmax}_k(-\|z^*-\mu_k\|^2/(\tau d))$; centroids fitted on training physiology only.',
                    '- $z^*$ is standardized with the stored training gate mean and SD.',
                    '',
                    '## CV-selected equations (one per phenotype)', '']
    gate_records={}
    for target in TARGETS:
        for model_kind in ('cv_winner','best_dynamic','best_global'):
            r=winners[target][model_kind];lbl=r['label'];o=fitted[(target,lbl)];model=o['model']
            equations=model.equations(o['names'])
            key=f'{target}/{model_kind}'
            gate_records[key]={'label':lbl,'gate_definition':model.gate_details(),
                               'experts':equations}
            equation_lines += [f'### {target} — {model_kind}: {lbl}', '',
                               f'Training CV weighted MSE: {r["cv_weighted_MSE"]:.6f}.', '']
            for ex in equations:
                equation_lines.append(f'**Expert {ex["expert"]+1}:**')
                terms=[f'{ex["intercept"]:+.8g}']
                terms.extend(f'{b:+.8g}*{name}' for name,b in ex['terms'].items() if abs(b)>1e-11)
                equation_lines.extend(['', '```text', 'Y_hat_'+str(ex['expert']+1)+' = '+' '.join(terms),'```',''])
            equation_lines.extend(['Gate parameters (training-only; needed to reproduce mixture):','',
                                   '```json',json.dumps(model.gate_details(),indent=2),'```',''])
    (results/'100_dynamic_equations.md').write_text('\n'.join(equation_lines),encoding='utf-8')
    (results/'100_equation_bundle.json').write_text(json.dumps(gate_records,indent=2),encoding='utf-8')
    headline=[]
    for target in TARGETS:
        for kind in ('cv_winner','best_dynamic','best_global'):
            rec=winners[target][kind]
            for period in (TEST,VALIDATE):
                row=scores[(scores.target==target)&(scores.label==rec['label'])&(scores.period==period)].iloc[0]
                headline.append({'target':target,'role':kind,'label':rec['label'],
                                 'CV_MSE':rec['cv_weighted_MSE'],'period':period,
                                 'R2':row.R2,'RMSE':row.RMSE,'MAE':row.MAE,'bias':row.bias})
    h=pd.DataFrame(headline)
    lines=['# Script 100 — Dynamic physiological regression: readout','',
           '## Experimental rule','',
           '- 2005–08: training and cycle-stratified 5-fold CV; entire gate/expert fitted inside training folds.',
           '- 2009–18 and 2021–23: frozen predictions only; no tuning or refitting.',
           '- **Historical warning:** both later periods have been repeatedly examined in Scripts 72–99, so results are exploratory replication rather than untouched confirmatory tests.',
           '- NHANES cycles are different participants, not individual longitudinal trajectories.',
           '', '## Selected models and transfer performance','',
           mod.markdown_table(h), '',
           '## PHQ decomposition coherence','',
           mod.markdown_table(pd.DataFrame(coherence)),'',
           '## Interpretation','',
           'The model has a fitted numerical equation for every expert; gates choose a smooth convex mixture. A static model is valid if it beats dynamic alternatives in TRAIN CV.',
           'Physiological block contrasts are model sensitivity measures, **not** causal/clinical attribution or additive percentage shares.',
           'Global interaction models and smooth gating may overfit; low transfer R² does not invalidate the mathematical equation.',
           'All metrics use sample weights only, not complex NHANES survey-design uncertainty.',
           '', '## Next scientific decision', '',
           'Evaluate whether gates beat the best global model on **training CV** first; inspect later-cycle degradation, calibration, and gate occupancy without using those results to select a new winner.',
           'If dynamic models fail, retain the fitted global 7-Z Ridge equation and report the negative result.', '']
    (results/'100_readout.md').write_text('\n'.join(lines),encoding='utf-8')
    manifest={'script':Path(__file__).name,'synthetic_self_test':bool(smoke),
              'reused_script98':str(P98.relative_to(ROOT)) if not smoke else 'synthetic testing only',
              'training_cycles':['0506','0708'], 'test_cycles':['0910','1112','1314','1516','1718'],
              'validation_cycles':['2123'],'survey_weighted_point_metrics':True,
              'revisited_temporal_holdouts':True,'frozen_Z':True,
              'selected_by':'training-only stratified CV weighted MSE',
              'matched_counts':d.groupby('SPLIT').size().to_dict(),
              'n_candidate_specs_per_target':len(all_specs()) if not smoke else 4,
              'no_causality':True,'no_individual_longitudinal':True,
              'winners':{t:{k:v['label'] for k,v in kind.items()} for t,kind in winners.items()}}
    (results/'100_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Ready-to-append log: factual only, computed from this run.
    append=['## Script 100 — Dynamic physiological regression ('+('synthetic self-test' if smoke else '14 October 2026')+')','',
            '**Objective.** Compare static seven-coordinate Ridge/ElasticNet phenotype equations with multiple physiology-gated Ridge equations, preserving the frozen X→Z representation and explicit intercept, coefficients and residual.','',
            '**Protocol.** Train and select exclusively using 2005–08 NHANES with 5-fold CV. Frozen transfer evaluation on 2009–18 and 2021–23. The later periods have already been inspected in earlier scripts; interpretations are exploratory.','',
            r'**Model.** $Y=\sum_k w_k(Z)(\beta_{0k}+\beta_k^\top Z+\gamma_k^\top C)+\varepsilon$. Tested A-private, G-private, joint A/G and unsupervised two-center gating, with lean and full covariates.','',
            '**Training-selected results.**','',mod.markdown_table(h),'',
            '**Decision.** Prefer the training-CV winner; dynamic complexity is not automatically better. Coefficients, gate parameters and block contrasts saved in `100_dynamic_equations.md`, `100_equation_bundle.json`, and `100_block_contrasts.csv`. This establishes statistical associations, not causal pathways or a physiological inverse.','',
            '**Integrity.** Prior models/decomposition unchanged; no validation-based retuning; no new clinical or within-subject longitudinal claims.','']
    (results/'100_experiment_log_append.md').write_text('\n'.join(append),encoding='utf-8')
    print('\nRESULTS — CV winners and best dynamics')
    print(h[h.period.eq(VALIDATE)][['target','role','R2','MAE','label']].to_string(index=False))
    print('FILES:',', '.join(p.name for p in sorted(results.glob('100_*'))))
    return winners,scores


def self_test_regression():
    rng=np.random.default_rng(10)
    x=rng.normal(size=(190,9)); z=x[:,:7].copy(); y=3+.2*x[:,0]-.4*x[:,4]+rng.normal(0,.5,190)
    w=rng.uniform(.4,2,190)
    for gate in ('A_axis','G_axis','AG_axes','kmeans2'):
        m=SmoothExperts(gate,alpha=10,tau=1.5).fit(x,z,y,w)
        p=m.predict(x,z); g=m._gates(z)
        assert p.shape==(190,) and np.isfinite(p).all()
        assert np.allclose(g.sum(axis=1),1,atol=1e-10)
        assert len(m.equations([f'f{i}' for i in range(9)]))==g.shape[1]
        for i in range(g.shape[1]):
            ex=m.equations([f'f{i}' for i in range(9)])[i]
            b=np.array([ex['terms'][f'f{j}'] for j in range(9)])
            assert np.allclose(ex['intercept']+x@b, m.experts[i].predict(m.xsc.transform(x)),rtol=1e-7,atol=1e-7)
    print('All gating / exact expert-equation roundtrip tests PASSED.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cv-folds',type=int,default=5)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.cv_folds<3 or args.cv_folds>10: raise ValueError('cv-folds must be 3..10')
    mod=load_98()
    print('SCRIPT 100 | Frozen X→Z, multi-equation physiological regression')
    if args.self_test:
        self_test_regression()
        d,F=synthetic_data(mod)
    else:
        d,F=prepare_real(mod)
    print('Matched samples:',d.groupby('SPLIT').size().to_dict())
    sel,win,fits=model_selection(mod,d,F,args.cv_folds,smoke=args.self_test)
    write_outputs(mod,d,F,sel,win,fits,smoke=args.self_test)
    print('DONE — 72, 73, 98, 99 unchanged.')

if __name__=='__main__':main()
