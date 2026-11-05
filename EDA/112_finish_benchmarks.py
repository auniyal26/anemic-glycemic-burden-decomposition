#!/usr/bin/env python3
"""Real-data closure of rank-matched benchmarks and H/G/depression ablations.

Run with the original research repository READ ONLY and a separate output path.
Uses original official r.jive and Python AJIVE with its default joint-component
check enabled; exact dense SVD replaces only its numerical SVD backend.
No holdout fitting or rank selection. All later eras were previously inspected.
See 112_protocol.json. This is exploratory evidence, not guaranteed superiority.
"""
from __future__ import annotations
import os,sys,argparse,json,hashlib,importlib,importlib.util,tempfile,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge,LinearRegression

PERIODS=('train_2005-2008','test_2009-2018','validate_2021-2023')
ENV=Path(r'C:\Users\auniy\anaconda3\envs\physiodecomp')

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
def exact_svd(x,rank=None):
    x=np.asarray(x,float);u,s,vt=svd(x,full_matrices=False,lapack_driver='gesvd')
    k=len(s) if rank is None else int(rank)
    if k>len(s):raise ValueError('SVD rank exceeds available dimensions')
    return u[:,:k],s[:k],vt[:k].T

def ajive_module():
    import scipy.sparse.sputils as legacy
    from scipy.sparse import _sputils as modern
    for name in ('isshape','isintlike'):
        if not hasattr(legacy,name):setattr(legacy,name,getattr(modern,name))
    try:return importlib.import_module('jive.AJIVE')
    except ModuleNotFoundError:
        sys.path.append(str(ENV/'Lib/site-packages'))
        return importlib.import_module('jive.AJIVE')

def ajive_fit(x,budget,automatic=False):
    am=ajive_module();backend=am.svd_wrapper;am.svd_wrapper=exact_svd
    try:
        indiv=None if automatic else {'H':2 if budget==5 else 3,'G':1 if budget==5 else 2}
        fit=am.AJIVE(init_signal_ranks={'H':3,'G':2},joint_rank=2,
            indiv_ranks=indiv,center=False,reconsider_joint_components=True,n_jobs=1,store_full=True)
        fit.fit({'H':x[:,:4],'G':x[:,4:]});r=fit.results_dict();pieces=[];svs=[]
        q=np.asarray(r['common']['scores']) if r['common']['scores'] is not None else np.empty((len(x),0))
        if q.size:pieces.append(q)
        for g in ('H','G'):
            item=r[g]['individual']
            if item['scores'] is not None:
                pieces.append(np.asarray(item['scores']));svs.extend(np.asarray(item['svals']).tolist())
            block=x[:,:4] if g=='H' else x[:,4:]
            expected=q@(q.T@block) if q.size else np.zeros_like(block)
            actual=fit.blocks[g].joint.full_
            assert np.max(abs(actual-expected))<1e-8,'Retained joint-space projection invariant failed'
        z=np.column_stack(pieces)
        info=dict(joint_rank=int(fit.joint_rank),individual_ranks={k:int(v) for k,v in fit.indiv_ranks.items()},individual_svals=svs,
                  projection_invariant_pass=True,configuration='native default joint check enabled; deterministic exact SVD',
                  dimension_policy='threshold selected' if automatic else ('fixed rank-5 comparison' if budget==5 else 'explicit full-budget sensitivity'))
        return z,info
    finally:am.svd_wrapper=backend

R_CODE=r'''
args<-commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(r.jive))
d<-as.matrix(read.csv(args[1],check.names=FALSE))
for(k in c(5,7)) {
  set.seed(103)
  ha<-if(k==5)2 else 3; ga<-if(k==5)1 else 2
  m<-jive(list(H=t(d[,1:4]),G=t(d[,5:7])),method="given",rankJ=2,rankA=c(ha,ga),scale=FALSE,center=FALSE,est=TRUE,orthIndiv=TRUE,maxiter=1000,showProgress=FALSE)
  ss<-function(x,k){s<-svd(t(x),nu=k,nv=0);sweep(s$u[,1:k,drop=FALSE],2,s$d[1:k],"*")}
  scores<-cbind(ss(do.call(rbind,m$joint),2),ss(m$individual[[1]],ha),ss(m$individual[[2]],ga))
  write.csv(scores,file.path(args[2],paste0("jive",k,".csv")),row.names=FALSE)
  cat("Official JIVE budget",k,"completed\n");flush.console()
}
'''

def rjive_fits(x,out,fingerprint,rscript):
    archive=out/'112_jive_frozen_maps.npz';meta=out/'112_jive_cache_manifest.json'
    if archive.exists() and meta.exists():
        old=json.loads(meta.read_text());assert old['fingerprint']==fingerprint,'Cached fit fingerprint differs'
        z=np.load(archive);print('Reusing matching discovery-frozen official JIVE maps',flush=True)
        return {k:np.asarray(z[k]) for k in z.files},old['diagnostics']
    tmp_root=out/'temporary';tmp_root.mkdir(exist_ok=True)
    assert tmp_root.resolve().parent==out.resolve()
    env=os.environ.copy();env['R_HOME']=str(ENV/'Lib/R')
    env['PATH']=os.pathsep.join([str(ENV/'Library/bin'),str(ENV/'Lib/R/bin/x64'),env['PATH']])
    with tempfile.TemporaryDirectory(dir=tmp_root,prefix='native_jive_') as td:
        td=Path(td);pd.DataFrame(x).to_csv(td/'x.csv',index=False);(td/'run.R').write_text(R_CODE)
        p=subprocess.run([str(rscript),'--vanilla',str(td/'run.R'),str(td/'x.csv'),str(td)],env=env,capture_output=True,text=True,timeout=1200)
        (out/'112_rjive_execution.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
        if p.returncode:raise RuntimeError(p.stderr[-2500:])
        maps={};diagnostics=[]
        for k in (5,7):
            native=pd.read_csv(td/f'jive{k}.csv').to_numpy(float)
            cf=np.linalg.lstsq(np.c_[np.ones(len(x)),x],native,rcond=1e-12)[0]
            err=np.linalg.norm(np.c_[np.ones(len(x)),x]@cf-native)/np.linalg.norm(native-native.mean(0))
            if err>1e-7:raise RuntimeError(f'JIVE{k} frozen adapter is not faithful: {err}')
            maps[f'JIVE{k}']=cf
            diagnostics.append(dict(method=f'JIVE{k}',adapter_relative_error=float(err),native_rank=int(np.linalg.matrix_rank(native)),score_count=k))
        np.savez(archive,**maps);dump(meta,dict(fingerprint=fingerprint,diagnostics=diagnostics))
        return maps,diagnostics

def paired_psu_weights(frame,analytic,reps,rng):
    keys=['cycle','SDMVSTRA','SDMVPSU'];fr=frame[keys].drop_duplicates().reset_index(drop=True)
    fr['cluster']=np.arange(len(fr));factors=np.zeros((reps,len(fr)))
    for _,group in fr.groupby(['cycle','SDMVSTRA'],sort=True):
        ids=group.cluster.to_numpy();m=len(ids)
        if m<2:raise ValueError('Incomplete full frame: singleton PSU stratum')
        draws=rng.integers(0,m,size=(reps,m-1))
        for j,idx in enumerate(ids):factors[:,idx]=(draws==j).sum(1)*m/(m-1)
    linked=analytic[keys].merge(fr,on=keys,how='left',validate='many_to_one',sort=False)
    if linked.cluster.isna().any():raise ValueError('Analytic PSU absent from complete frame')
    return factors,linked.cluster.to_numpy(int),len(fr)

def bootstrap_metrics(errors,w,reference_sq,frame,analytic,reps,rng,reconstruction=False):
    factors,cluster,p=paired_psu_weights(frame,analytic,reps,rng)
    numer=np.column_stack([np.bincount(cluster,weights=w*errors[:,j],minlength=p) for j in range(errors.shape[1])])
    denominator=np.bincount(cluster,weights=w*(reference_sq if reconstruction else np.ones(len(w))),minlength=p)
    denom=factors@denominator
    if np.any(denom<=0):raise ValueError('Empty bootstrap analytic domain')
    out=(factors@numer)/denom[:,None]
    return 1-out if reconstruction else out

def tests():
    rng=np.random.default_rng(3);x=rng.normal(size=(120,7));y=rng.normal(size=120);w=rng.uniform(.5,2,120)
    a=rng.normal(size=(7,7));z=x@a
    p=LinearRegression().fit(x,y,sample_weight=w).predict(x)
    q=LinearRegression().fit(z,y,sample_weight=w).predict(z)
    assert np.max(abs(p-q))<1e-10
    rows=[]
    for st in (1,2):
        for ps in (1,2):rows.append(dict(cycle='test',SDMVSTRA=st,SDMVPSU=ps))
    frame=pd.DataFrame(rows);analytic=frame.iloc[[0,0,1,2,3]].reset_index(drop=True)
    ww=np.array([1.,2.,3.,4.,5.]);err=np.column_stack([np.arange(5.),np.arange(5.)+.2])
    f,ix,k=paired_psu_weights(frame,analytic,9,np.random.default_rng(5))
    manual=[]
    for rep in f:
        rw=ww*rep[ix];manual.append(np.average(err,axis=0,weights=rw))
    fast=bootstrap_metrics(err,ww,None,frame,analytic,9,np.random.default_rng(5))
    assert np.max(abs(fast-np.asarray(manual)))<1e-12
    shared=rng.normal(size=(120,2))
    coupled=np.c_[3*shared+rng.normal(size=(120,2))*.01,rng.normal(size=(120,2)),3*shared+rng.normal(size=(120,2))*.01,rng.normal(size=120)]
    z5,info=ajive_fit(coupled,5);assert np.linalg.matrix_rank(z5)==z5.shape[1]
    cf=np.linalg.lstsq(np.c_[np.ones(len(coupled)),coupled],z5,rcond=None)[0]
    assert np.max(abs(np.c_[np.ones(len(coupled)),coupled]@cf-z5))<1e-10
    print('PASS: invertible-coordinate OLS invariance; exact cluster/participant bootstrap equivalence; AJIVE retained-projector and affine-score recovery',flush=True)

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--repo',type=Path,default=Path(r'D:\Hematological and Glycemic Decomposition of Depression'))
    pa.add_argument('--out',type=Path,default=None)
    pa.add_argument('--rscript',type=Path,default=ENV/'Lib/R/bin/x64/Rscript.exe')
    pa.add_argument('--self-test',action='store_true');args=pa.parse_args()
    if args.self_test:return tests()
    root=args.repo.resolve();out=(args.out or root/'EDA/Results/112_closure').resolve()
    if out==root:raise ValueError('Use a dedicated results subfolder, not the repository root')
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path(__file__).with_name('112_protocol.json').read_text())
    dump(out/'112_protocol_used.json',protocol)
    source_paths=[root/'EDA/103_jive_ajive_benchmark.py',root/'EDA/98_regularized_phenotype_equation.py',root/'EDA/Results/73_frozen_phenotype_map_input.csv']
    hashes={str(p):sha(p) for p in source_paths}
    fingerprint=hashlib.sha256(json.dumps(dict(hashes=hashes,protocol=protocol),sort_keys=True).encode()).hexdigest()
    b=module('source103_for112',source_paths[0]);d,cov=b.load_matched()
    split=d.SPLIT.to_numpy();tr=np.flatnonzero(split==PERIODS[0]);w=d.SURVEY_WT.to_numpy(float)
    counts={p:int((split==p).sum()) for p in PERIODS}
    assert counts==dict(zip(PERIODS,[3815,9893,2247]))
    raw=d[b.X_COLS].to_numpy(float);z=d[b.Z_COLS].to_numpy(float)
    scaler=StandardScaler().fit(raw[tr]);x=scaler.transform(raw);xt=x[tr]
    z5=b.select_z5_from_physiology(z[tr],xt,w[tr])
    scores={'RawX7':x,'PCA7':b.WeightedPCA(7).fit(xt).transform(x),'FrozenZ7':z,'PCA5':b.WeightedPCA(5).fit(xt).transform(x),'FrozenZ5':z[:,z5]}
    maps,jdiag=rjive_fits(xt,out,fingerprint,args.rscript)
    for name,cf in maps.items():scores[name]=b.affine_predict(x,cf)
    diag=list(jdiag);saved_maps=dict(maps)
    print('Official JIVE 5D/7D discovery fits complete',flush=True)
    for budget in (5,7):
        native,info=ajive_fit(xt,budget)
        repeat,_=ajive_fit(xt,budget)
        qa=np.linalg.qr(native)[0];qb=np.linalg.qr(repeat)[0]
        reproduc=float(np.linalg.svd(qa.T@qb,compute_uv=False).min())
        cf,ad=b.learn_projection(xt,native,min_fit_r2=.99999999)
        rel=np.linalg.norm(b.affine_predict(xt,cf)-native)/np.linalg.norm(native-native.mean(0))
        if native.shape[1]!=budget or np.linalg.matrix_rank(native)!=budget or rel>1e-7:raise RuntimeError('AJIVE reliability gate failed')
        if min(info['individual_svals'])<1e-8*np.linalg.svd(xt,compute_uv=False)[0]:raise RuntimeError('AJIVE contains a numerically null individual axis')
        name=f'AJIVE{budget}_rankchecked';scores[name]=b.affine_predict(x,cf);saved_maps[name]=cf
        diag.append(dict(method=name,adapter_R2=ad['train_score_r2'],adapter_relative_error=float(rel),repeat_subspace_min_cosine=reproduc,**info))
    auto,autoinfo=ajive_fit(xt,5,automatic=True)
    dump(out/'112_AJIVE_threshold_selected_diagnostic.json',autoinfo)
    np.savez(out/'112_frozen_score_maps.npz',**saved_maps)
    dump(out/'112_representation_diagnostics.json',diag)
    name7=['RawX7','PCA7','FrozenZ7','JIVE7','AJIVE7_rankchecked']
    # All downstream names below are fixed before any later-period scoring.
    scores.update(Covariates=np.empty((len(d),0)),H4=x[:,:4],G3=x[:,4:],Threshold2=np.c_[np.maximum(np.where(d.RIAGENDR.eq(1),13.,12.)-raw[:,0],0),np.maximum(raw[:,4]-5.7,0)],Shared2=z[:,:2],Discordant2=z[:,2:4],H_private2=z[:,4:6],G_private1=z[:,6:7])
    feat={name:np.c_[cov,s] for name,s in scores.items()}
    folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=103).split(tr,d.CYCLE.to_numpy()[tr]))
    prediction={};performance=[];selected=[];coefs={};ols_checks=[]
    for yn,ycol in b.PHENOTYPES.items():
        y=d[ycol].to_numpy(float)
        for name,f in feat.items():
            cvm,alpha=b.cv_alpha(f,y,w,tr,folds)
            sc=StandardScaler().fit(f[tr],sample_weight=w[tr]);fit=Ridge(alpha=alpha).fit(sc.transform(f[tr]),y[tr],sample_weight=w[tr]/w[tr].mean())
            yp=fit.predict(sc.transform(f));prediction[yn,name]=yp
            coefs[f'{yn}/{name}']=dict(alpha=alpha,feature_mean=sc.mean_.tolist(),feature_scale=sc.scale_.tolist(),coef=fit.coef_.tolist(),intercept=float(fit.intercept_))
            selected.append(dict(outcome=yn,method=name,discovery_CV_MSE=cvm,alpha=alpha,physiology_dimensions=scores[name].shape[1]))
            for p in PERIODS:
                ix=split==p
                performance.append(dict(outcome=yn,method=name,period=p,n=int(ix.sum()),MSE=b.weighted_mse(y[ix],yp[ix],w[ix]),R2=b.weighted_r2(y[ix],yp[ix],w[ix])))
        base=None
        for name in name7:
            if np.linalg.matrix_rank(np.c_[np.ones(len(tr)),scores[name][tr]])!=8:raise RuntimeError(f'{name} is not full affine rank')
            sc=StandardScaler().fit(feat[name][tr],sample_weight=w[tr]);f=LinearRegression().fit(sc.transform(feat[name][tr]),y[tr],sample_weight=w[tr]/w[tr].mean())
            yp=f.predict(sc.transform(feat[name]))
            if base is None:base=yp
            err=float(np.max(abs(base-yp)));ols_checks.append(dict(outcome=yn,method=name,max_prediction_difference_from_raw_OLS=err))
            if err>1e-7:raise RuntimeError(f'OLS information-equivalence check failed: {name}: {err}')
        print('Completed discovery-only prediction fits:',yn,flush=True)
    pd.DataFrame(performance).to_csv(out/'112_phenotype_performance.csv',index=False)
    pd.DataFrame(selected).to_csv(out/'112_discovery_model_selection.csv',index=False)
    pd.DataFrame(ols_checks).to_csv(out/'112_7D_OLS_invariance.csv',index=False)
    dump(out/'112_frozen_outcome_parameters.json',coefs)
    recon={};reconstruction=[]
    anchor=b.weighted_mean(xt,w[tr]);den=((x-anchor)**2).sum(1)
    for name in ['PCA5','FrozenZ5','JIVE5','AJIVE5_rankchecked']+name7:
        cf,_=b.learn_projection(scores[name][tr],xt,min_fit_r2=-np.inf)
        err=((x-b.affine_predict(scores[name],cf))**2).sum(1);recon[name]=err
        for p in PERIODS:
            ix=split==p;val=1-np.dot(w[ix],err[ix])/np.dot(w[ix],den[ix])
            reconstruction.append(dict(method=name,period=p,n=int(ix.sum()),dimensions=scores[name].shape[1],reconstruction_R2=float(val)))
    pd.DataFrame(reconstruction).to_csv(out/'112_reconstruction.csv',index=False)
    # Full-frame cluster multipliers are shared by every comparison within an era.
    frames=[];links=[]
    for cy,suf in {'0910':'F','1112':'G','1314':'H','1516':'I','1718':'J','2123':'L'}.items():
        folder=root/'Data/NHANES_Transfer'/cy
        p=next(p for p in [folder/f'DEMO_{suf}.XPT',folder/f'DEMO_{suf}.xpt'] if p.exists())
        q=pd.read_sas(p,format='xport')[['SEQN','SDMVSTRA','SDMVPSU']].copy();q['cycle']=cy;q['period']=PERIODS[2] if cy=='2123' else PERIODS[1]
        if q.isna().any().any():raise ValueError('Missing original design identifiers')
        frames.append(q.drop(columns='SEQN').drop_duplicates());links.append(q)
    full=pd.concat(frames,ignore_index=True);link=pd.concat(links,ignore_index=True)
    analytic=pd.DataFrame(dict(SEQN=d.SEQN.to_numpy(),cycle=d.CYCLE.to_numpy(),period=split)).merge(link,on=['SEQN','cycle','period'],how='left',validate='one_to_one',sort=False)
    rng=np.random.default_rng(112);paired=[]
    def add(rows,names,point,boot,p,yn,kind,comparisons,n):
        for method,ref,family in comparisons:
            a=names.index(method);r=names.index(ref)
            direction=1 if kind=='reconstruction' else -1
            gain=direction*(point[a]-point[r]);bgain=direction*(boot[:,a]-boot[:,r]);lo,hi=np.quantile(bgain,[.025,.975])
            rows.append(dict(analysis=kind,family=family,outcome=yn,period=p,method=method,reference=ref,n=n,paired_gain_positive_better=float(gain),approx_PSU_low=float(lo),approx_PSU_high=float(hi),direction='favours_method' if lo>0 else ('favours_reference' if hi<0 else 'inconclusive'),resamples=999,scope='exploratory frozen-model holdout uncertainty'))
    comparisons=[(m,'PCA5','rank5') for m in ['FrozenZ5','JIVE5','AJIVE5_rankchecked']]+[(m,'RawX7','rank7') for m in name7 if m!='RawX7']+[(m,'Covariates','triad') for m in ['H4','G3','RawX7','FrozenZ7','Threshold2','Shared2','Discordant2','H_private2','G_private1']]+[('RawX7','H4','triad'),('RawX7','G3','triad'),('RawX7','Threshold2','triad')]
    for p in PERIODS[1:]:
        ix=split==p;ap=analytic.loc[ix].reset_index(drop=True);frame=full[full.period.eq(p)]
        for yn,ycol in b.PHENOTYPES.items():
            names=list(scores);y=d[ycol].to_numpy(float)[ix]
            errors=np.column_stack([(y-prediction[yn,name][ix])**2 for name in names])
            point=np.average(errors,axis=0,weights=w[ix]);boot=bootstrap_metrics(errors,w[ix],None,frame,ap,999,rng)
            add(paired,names,point,boot,p,yn,'prediction',comparisons,int(ix.sum()))
        names=list(recon);errors=np.column_stack([recon[name][ix] for name in names]);point=1-(w[ix]@errors)/np.dot(w[ix],den[ix])
        boot=bootstrap_metrics(errors,w[ix],den[ix],frame,ap,999,rng,True)
        comps=[(m,'PCA5','rank5') for m in ['FrozenZ5','JIVE5','AJIVE5_rankchecked']]+[(m,'RawX7','rank7') for m in name7 if m!='RawX7']
        add(paired,names,point,boot,p,'physiology','reconstruction',comps,int(ix.sum()))
        print('Completed full-frame paired PSU comparisons:',p,flush=True)
    pp=pd.DataFrame(paired);pp.to_csv(out/'112_paired_PSU_comparisons.csv',index=False)
    pp[pp.family.eq('triad')].to_csv(out/'113_triad_evidence.csv',index=False)
    changes={str(p):sha(p)==hashes[str(p)] for p in source_paths}
    assert all(changes.values()),'A research input changed during execution'
    dump(out/'112_run_manifest.json',dict(protocol=protocol,source_hashes=hashes,source_unchanged=changes,cohort_counts=counts,Z5_indices=z5,full_rank_methods=name7,AJIVE_primary_configuration='original package, default reconsider_joint_components=True, exact dense SVD',AJIVE_auto_reference=autoinfo,participant_rows_exported=False,limits=['exploratory previously inspected eras','fixed-model approximate PSU uncertainty','full-budget 7D fits are not selected denoised ranks','linearly equivalent information is not method superiority']))
    rows=[]
    for (family,method,ref,kind),g in pp.groupby(['family','method','reference','analysis']):
        rows.append(dict(family=family,method=method,reference=ref,analysis=kind,comparisons=len(g),favours_method=int(g.direction.eq('favours_method').sum()),favours_reference=int(g.direction.eq('favours_reference').sum()),inconclusive=int(g.direction.eq('inconclusive').sum())))
    pd.DataFrame(rows).to_csv(out/'112_advantage_scorecard.csv',index=False)
    print('COMPLETE. Corrected rank-matched benchmarks, full-rank invariance and triad comparisons saved.',flush=True)

if __name__=='__main__':main()
