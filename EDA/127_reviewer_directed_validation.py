#!/usr/bin/env python3
"""127 | Reviewer-directed HG7 validation. Place in EDA alongside 106, 118, 123, 125.
No changes to previous results. Default: 999 design bootstrap replicates.
A: reference-frozen HGI and matched-dimensional controls; B: CDC ferritin bridge
on original 118 sample; C: conditional partial R2 survey PSU bootstrap; D: MIDUS
metadata inventory and optional explicitly harmonised external CSV evaluation.
All temporal NHANES tests are exploratory; external validation is gated.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, ast, importlib.util, sys, json, hashlib, traceback, re, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits

EDA = Path(__file__).resolve().parent
R = EDA / 'Results'
SEED = 127
CYCLES = ('0506','0708','2123')
HOLD = ('0708','2123')
CDC_F = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/FERTIN_F.htm'
CDC_I = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/FERTIN_I.htm'
CDC_L = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/FERTIN_L.htm'
EXPECTED = {'LBXHGB':'g/dL','LBXRBCSI':'million cells/uL','LBXMCVSI':'fL','LBXRDW':'%','LBXGH':'%','LBXGLU':'mg/dL','LOG_IN':'ln(insulin uIU/mL)'}


def load(name, filename):
    p = EDA / filename
    if not p.is_file(): raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(name,p)
    m = importlib.util.module_from_spec(spec); sys.modules[name]=m
    spec.loader.exec_module(m); return m


def save(out, name, rows):
    pd.DataFrame(rows).to_csv(out/f'127_{name}.csv',index=False)


def dump(path, obj):
    path.write_text(json.dumps(obj,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def hashfile(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(w):
    w = np.asarray(w,float)
    if np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0: raise ValueError('Invalid weights')
    return w/w.sum()


def design_weights(q, frame, reps, rng):
    # Full cycle design, including PSUs with no eligible domain observations.
    keys=['SDMVSTRA','SDMVPSU']; f=frame[keys].drop_duplicates().reset_index(drop=True).copy()
    if f.isna().any().any(): raise ValueError('Missing design identifiers')
    f['cluster']=np.arange(len(f)); ix=q[keys].merge(f,on=keys,how='left',validate='many_to_one').cluster
    if ix.isna().any(): raise ValueError('Unmatched analytic PSU')
    factors=np.zeros((reps,len(f)))
    for _,sub in f.groupby('SDMVSTRA'):
        ids=sub.cluster.to_numpy(int); m=len(ids)
        if m<2: raise ValueError('Singleton full-frame stratum')
        draws=rng.integers(m,size=(reps,m-1))
        for j,k in enumerate(ids): factors[:,k]=(draws==j).sum(axis=1)*m/(m-1)
    w=q.W.to_numpy(float); ww=np.vstack([w,factors[:,ix.to_numpy(int)]*w])
    totals=ww.sum(1); valid=totals>0
    ww[valid]/=totals[valid,None]; ww[~valid]=np.nan
    return ww


def summary(draws, min_fraction=.95):
    a=np.asarray(draws,float); valid=np.isfinite(a[1:]); n=int(valid.sum())
    ok=n>=max(1,int(np.ceil((len(a)-1)*min_fraction))) and np.isfinite(a[0])
    lo,hi=np.quantile(a[1:][valid],[.025,.975]) if ok else (np.nan,np.nan)
    return dict(estimate=float(a[0]),lo=float(lo),hi=float(hi),valid_reps=n,requested_reps=len(a)-1,status='available' if ok else 'insufficient_valid_replicates')


def wls_rss(x,y,w):
    a=x*np.sqrt(w[:,None]); b=y*np.sqrt(w)
    coef,_,rank,_=np.linalg.lstsq(a,b,rcond=1e-12)
    if rank!=x.shape[1]: raise ValueError('Rank-deficient replicate')
    return float(np.sum((b-a@coef)**2))


def scale_fit(a,w,whiten=True):
    a=np.asarray(a,float); wn=norm(w); mu=wn@a; c=a-mu
    if a.shape[1]==0:return lambda x: np.asarray(x,float)
    if whiten:
        v,u=np.linalg.eigh((c*wn[:,None]).T@c)
        if v[-1]<=0 or v[0]<=v[-1]*1e-10:raise ValueError('Rank-deficient feature set')
        t=(u*v**-.5)@u.T
    else:
        sd=np.sqrt(wn@(c*c))
        if (sd<=1e-12).any():raise ValueError('Constant covariate')
        t=np.diag(1/sd)
    return lambda x:(np.asarray(x,float)-mu)@t


def ridge_model(c,z,y,w):
    cfit=scale_fit(c,w,False); zfit=scale_fit(z,w,True)
    m=Ridge(alpha=10).fit(np.c_[cfit(c),zfit(z)],y,sample_weight=w/w.mean())
    return lambda cc,zz:m.predict(np.c_[cfit(cc),zfit(zz)])


def weighted_mse(y,p,w):return float(norm(w)@((y-p)**2))


def metrics(y,p,w):
    w=norm(w); mse=w@((y-p)**2); var=w@((y-w@y)**2)
    return dict(MSE=float(mse),R2=float(1-mse/var) if var>0 else np.nan)


def fit_hgi(x,w):
    # HGI in HbA1c percentage points; slope estimated on fasting glucose only.
    a=np.c_[np.ones(len(x)),x[:,5]]; sw=np.sqrt(norm(w))
    coef=np.linalg.lstsq(a*sw[:,None],x[:,4]*sw,rcond=None)[0]
    icoef=np.linalg.lstsq(a*sw[:,None],x[:,6]*sw,rcond=None)[0]
    def residuals(xx):
        aa=np.c_[np.ones(len(xx)),xx[:,5]]
        return np.c_[xx[:,4]-aa@coef,xx[:,6]-aa@icoef]
    return residuals,coef,icoef


def feature_builder(s, ref, w):
    x=ref[s.RAW].to_numpy(float)
    # Same strict reference-only weighted CCA implementation as Script 125.
    p=load('s125_for127','125_adversarial_comparative_validation.py')
    hg=p.hg_fit(x,w); residual,coef,icoef=fit_hgi(x,w)
    def features(xx):
        z=hg(xx); d=z['discordant']; h=residual(xx)
        return {'Covariates':np.empty((len(xx),0)),'HGI':h[:,:1],
                'HGI_published':(xx[:,4]-(.024*xx[:,5]+3.1))[:,None],
                'D1':d[:,:1],'D2':d[:,1:2],'HG_Discordant2':d,
                'HGI_FPG2':np.c_[h[:,0],xx[:,5]],
                'DirectionalResidual2':h,'HGI_plus_Discordant3':np.c_[h[:,0],d],
                'HG_Shared2':z['shared'],'Raw7':xx,'HG7':z['all']}
    return features,dict(HGI_intercept=coef[0],HGI_FPG_slope=coef[1],loginsulin_intercept=icoef[0],loginsulin_FPG_slope=icoef[1],reference_n=len(x))


def hgi_analysis(s,samples,frames,features,out,reps):
    perf=[]; contrasts=[]; assoc=[]; failures=[]; models={}
    pairs=[('HGI','D1'),('HGI','D2'),('HGI','HG_Discordant2'),
           ('HGI_published','HG_Discordant2'),('HGI_published','HGI'),
           ('HGI_FPG2','HG_Discordant2'),('DirectionalResidual2','HG_Discordant2'),
           ('HGI','HGI_plus_Discordant3'),('HG_Discordant2','HGI_plus_Discordant3'),
           ('Covariates','HGI'),('Covariates','HG_Discordant2'),('Raw7','HG_Discordant2')]
    for anchor in tqdm(s.ANCHORS,desc='127A | HGI comparisons'):
        if (anchor,'0506') not in samples:continue
        tr=samples[anchor,'0506']; w0=tr.W.to_numpy(float); c0=s.covariates(tr,anchor); y0=tr.Y.to_numpy(float)
        z0=features(tr[s.RAW].to_numpy(float)); fits={}
        for name,z in z0.items():
            try:fits[name]=ridge_model(c0,z,y0,w0)
            except Exception as e:failures.append(dict(anchor=anchor,method=name,stage='fit',reason=str(e)))
        models[anchor]=fits
        for cy in HOLD:
            if (anchor,cy) not in samples:continue
            q=samples[anchor,cy]; w=q.W.to_numpy(float); y=q.Y.to_numpy(float); c=s.covariates(q,anchor); z=features(q[s.RAW].to_numpy(float)); loss={}
            ww=design_weights(q,frames[cy],reps,np.random.default_rng(np.random.SeedSequence([SEED,int(cy),list(s.ANCHORS).index(anchor)])))
            for name,fit in fits.items():
                pred=fit(c,z[name]); loss[name]=(y-pred)**2
                perf.append(dict(anchor=anchor,cycle=cy,method=name,n=len(q),dim=z[name].shape[1],**metrics(y,pred,w)))
            for comp,cand in pairs:
                if comp not in loss or cand not in loss:continue
                draws=ww@(loss[comp]-loss[cand])
                contrasts.append(dict(anchor=anchor,cycle=cy,comparator=comp,candidate=cand,dim_comparator=z[comp].shape[1],dim_candidate=z[cand].shape[1],**summary(draws)))
            # Nested association: does discordance add to HGI, and vice versa?
            # This is a distinct in-period estimand, not frozen prediction.
            try:
                csc=scale_fit(c,w,False)(c); h=z['HGI']; d=z['HG_Discordant2']; a=np.c_[np.ones(len(q)),csc,h,d]
                b,vc,df=s.survey_wls(a,y,w,q,frames[cy]); full=wls_rss(a,y,norm(w))
                for name,ix in [('Discordant_given_HGI',[-2,-1]),('HGI_given_Discordant',[-3])]:
                    ix=[a.shape[1]+j for j in ix]; red=wls_rss(np.delete(a,ix,1),y,norm(w)); F,p,k=s.wald(b,vc,ix,df)
                    assoc.append(dict(anchor=anchor,cycle=cy,test=name,n=len(q),partial_R2=1-full/red,F=F,p=p,df_num=k,design_df=df))
            except Exception as e:failures.append(dict(anchor=anchor,cycle=cy,stage='association',reason=str(e)))
    present={(r['anchor'],r['cycle'],r['test']) for r in assoc}
    for anchor in s.ANCHORS:
        for cy in HOLD:
            if anchor=='OGTT' and cy=='2123':continue
            for test in ('Discordant_given_HGI','HGI_given_Discordant'):
                if (anchor,cy,test) not in present:assoc.append(dict(anchor=anchor,cycle=cy,test=test,p=np.nan,status='unavailable'))
    a=pd.DataFrame(assoc)
    a['p_Holm']=s.holm(a.p.to_numpy())
    save(out,'hgi_performance',perf);save(out,'hgi_paired',contrasts);save(out,'hgi_conditional',a);save(out,'hgi_failures',failures)
    return models,dict(performance=len(perf),contrasts=len(contrasts),conditional_tests=len(assoc),failures=len(failures),complete=(len(assoc)==14 and a.p.notna().all() and len(contrasts)==84 and not failures and all(r['status']=='available' for r in contrasts)))


def bridge(ferritin, stage):
    v=np.asarray(ferritin,float)
    if not np.isfinite(v).all() or (v<=0).any():raise ValueError('Ferritin must be positive')
    if stage=='recorded':return v.copy()
    # CDC FERTIN_F: Hitachi 912 -> Elecsys E170 (2009-10).
    e170=10**(.989*np.log10(v)+.049)
    if stage=='E170_only':return e170
    # CDC FERTIN_I: E170 -> e601 (2015-16).
    if stage=='CDC_chain_e601':return .2243+1.079*e170
    raise ValueError(stage)


def decomp(y,p,ww):
    my=ww@y; mp=ww@p; vy=np.maximum(ww@(y*y)-my*my,0); vp=np.maximum(ww@(p*p)-mp*mp,0); cov=ww@(y*p)-my*mp
    slope=np.divide(cov,vp,out=np.zeros_like(cov),where=vp>1e-14)
    B=(my-mp)**2; C=vp*(slope-1)**2; E=np.maximum(vy-slope*cov,0); mse=ww@((y-p)**2)
    if not np.allclose((B+C+E)[np.isfinite(mse)],mse[np.isfinite(mse)],atol=1e-8):raise ValueError('MSE accounting failure')
    return dict(MSE=mse,mean_bias=B,slope=C,calibration=B+C,residual=E)


def load_118_linear():
    """Execute only the exact linear helpers from local Script 118.

    Script 118 imports torch at module scope for its VAE, which is not needed
    for this analysis. Keep the original cohort loader and ridge implementation
    (including weighted whitening) without importing the neural code.
    The archived-MSE reproduction gate remains unchanged.
    """
    path=EDA/'118_biomedical_utility_benchmark.py'
    source=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    wanted={'weighted_mean','WeightedWhitener','load_inputs','fit_ferritin_models'}
    selected=[node for node in source.body
              if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in wanted]
    if {node.name for node in selected}!=wanted or len(selected)!=len(wanted):
        raise RuntimeError('Script 118 linear helper definitions changed; inspect source before running')
    namespace={'__name__':'s118_linear_for127','__file__':str(path),
               'np':np,'pd':pd,'StandardScaler':StandardScaler,'Ridge':Ridge,
               'traceback':traceback,'CYCLES':CYCLES,'BASE_COV':['RIDAGEYR','BMXBMI','INDFMPIR'],
               'RIDGE_ALPHA':10.0,'load_module':load}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return SimpleNamespace(**{name:namespace[name] for name in wanted})


def ferritin_analysis(out,args):
    # Reuse 118's exact cohort construction and unweighted HG7 map. Only linear
    # candidates needed for the primary Raw7-versus-shared result; no neural refit.
    s=load_118_linear()
    stage=out/'118_input_reproduction';stage.mkdir(exist_ok=True)
    data=s.load_inputs(SimpleNamespace(fetch_missing=args.fetch_missing),stage)
    x=data['x']; m,mu,rho=data['s104'].fit_hg_cca(x['0506'],np.ones(len(x['0506'])))
    z={cy:(x[cy]-mu)@m for cy in CYCLES}
    reps={'Covariates':{cy:np.empty((len(x[cy]),0)) for cy in CYCLES},'RawX7':x,'HG7':z,'HG_Shared2':{cy:z[cy][:,:2] for cy in CYCLES}}
    anchor=data['anchor']; expected={'0506':402,'0708':500,'2123':460}
    if any(len(anchor[cy]['y'])!=expected[cy] for cy in CYCLES):raise ValueError('Original 118 sample sizes differ; inspect 118_input_reproduction before interpretation')
    ww={}
    for cy in HOLD:
        q=anchor[cy]['frame'].copy();q['W']=anchor[cy]['w'];ww[cy]=design_weights(q,data['frames'][cy],args.replicates,np.random.default_rng(np.random.SeedSequence([SEED,2,int(cy)])))
    original={cy:anchor[cy]['y'].copy() for cy in CYCLES}; rows=[]; gaps=[]; changes=[]; audit=[]; pred_results={}
    candidates=list(R.rglob('118_ferritin_performance.csv'))
    if args.reference_118:candidates=[args.reference_118/'118_ferritin_performance.csv']
    old=None
    if len(candidates)==1 and candidates[0].is_file():
        old=pd.read_csv(candidates[0],dtype={'cohort':str});old.cohort=old.cohort.str.zfill(4)
    elif len(candidates)>1:print('Multiple 118 reference tables: specify --reference-118 for exact archived check.')
    # E170-only scenario is intentionally a partial bridge; final comparison uses e601 chain.
    for mode in tqdm(['recorded','CDC_chain_e601','E170_only'],desc='127B | Ferritin scales'):
        for cy in CYCLES:
            yy=original[cy] if cy=='2123' else np.log(bridge(np.exp(original[cy]),mode))
            anchor[cy]['y']=yy
        failures=[];pred,_=s.fit_ferritin_models(data,reps,failures)
        if failures:raise RuntimeError(str(failures))
        pred_results[mode]={}
        for cy in CYCLES:
            y=anchor[cy]['y']; w=anchor[cy]['w']
            for name,p in pred.items():
                pr=p[cy]; value=weighted_mse(y,pr,w)
                truth=y<np.log(15); auc=roc_auc_score(truth,-pr,sample_weight=w) if len(np.unique(truth))==2 else np.nan
                rows.append(dict(scale=mode,cycle=cy,method=name,n=len(y),AUC=auc,**metrics(y,pr,w)))
                if mode=='recorded' and old is not None:
                    match=old[old.cohort.eq(cy)&old.method.eq(name)]
                    if len(match)!=1:raise ValueError(f'Missing archived 118 row {cy}/{name}')
                    delta=value-float(match.weighted_mse.iloc[0]);audit.append(dict(cycle=cy,method=name,delta_MSE=delta,passed=abs(delta)<1e-6))
            if cy not in HOLD:continue
            a=decomp(y,pred['RawX7'][cy],ww[cy]);b=decomp(y,pred['HG_Shared2'][cy],ww[cy]);dg={k:a[k]-b[k] for k in a};pred_results[mode][cy]=dg
            for k,v in dg.items():gaps.append(dict(scale=mode,cycle=cy,metric=k,**summary(v)))
        delta={k:pred_results[mode]['2123'][k]-pred_results[mode]['0708'][k] for k in pred_results[mode]['0708']}
        for k,v in delta.items():changes.append(dict(scale=mode,metric=k,**summary(v)))
        # Ratio unstable if the denominator is near zero; suppress CI if its interval includes zero.
        den=summary(delta['MSE']);ratio=np.divide(delta['calibration'],delta['MSE'],out=np.full_like(delta['MSE'],np.nan),where=np.abs(delta['MSE'])>1e-12)
        rr=summary(ratio)
        if den['lo']<=0<=den['hi'] or not np.isfinite(den['lo']):rr.update(lo=np.nan,hi=np.nan,status='denominator_uncertain')
        changes.append(dict(scale=mode,metric='calibration_fraction',**rr))
    save(out,'ferritin_performance',rows);save(out,'ferritin_gaps',gaps);save(out,'ferritin_gap_changes',changes);save(out,'ferritin_reproduction',audit)
    recorded=next(r for r in changes if r['scale']=='recorded' and r['metric']=='MSE')
    calibrated=next(r for r in changes if r['scale']=='recorded' and r['metric']=='calibration')
    manuscript_match=abs(recorded['estimate']-.315)<.002 and abs(calibrated['estimate']-.284)<.002
    archived_match=bool(audit) and all(r['passed'] for r in audit)
    return dict(archived_118_reproduced=archived_match,archive_checked=old is not None,manuscript_rounded_gap_reproduced=manuscript_match,interpretation='CDC_chain_e601 is chained point-estimate assay conversion; no direct 2005-to-2023 calibration or coefficient-uncertainty propagation. E170_only is partial bridge sensitivity, not e601 harmonisation.',complete=manuscript_match and (old is None or archived_match))


def partial_ci(s,samples,frames,out,reps):
    rows=[];fail=[]
    for (anchor,cy),q in tqdm(samples.items(),desc='127C | Partial R2 intervals'):
        w=q.W.to_numpy(float); y=q.Y.to_numpy(float); c=s.covariates(q,anchor); c=scale_fit(c,w,False)(c); z=q[s.Z].to_numpy(float)
        a=np.c_[np.ones(len(q)),c,z]; offset=a.shape[1]-7
        reduced={g:np.delete(a,[offset+i for i in ix],axis=1) for g,ix in s.GROUPS.items()}
        ww=design_weights(q,frames[cy],reps,np.random.default_rng(np.random.SeedSequence([SEED,3,int(cy),list(s.ANCHORS).index(anchor)])))
        draws={g:np.full(reps+1,np.nan) for g in reduced}; rankfails=0
        for i,wi in enumerate(ww):
            try:
                if not np.isfinite(wi).all():raise ValueError('Empty replicate')
                full=wls_rss(a,y,wi)
                for g,b in reduced.items():
                    r=wls_rss(b,y,wi)
                    if r<=0:raise ValueError('Zero reduced residual')
                    draws[g][i]=max(0,1-full/r)
            except Exception:
                rankfails+=1
                for v in draws.values():v[i]=np.nan
        tests,_=s.conditional_test(q,frames[cy],anchor,cy,'base',expanded=False)
        test={t['group']:t for t in tests}
        for g,v in draws.items():
            ss=summary(v); t=test[g];
            if not np.isclose(ss['estimate'],t['partial_R2'],atol=1e-8):raise RuntimeError(f'Point estimate mismatch {anchor}/{cy}/{g}')
            rows.append(dict(anchor=anchor,cycle=cy,group=g,n=len(q),p=t['p'],F=t['F'],design_df=t['design_df'],**ss))
        if rankfails:fail.append(dict(anchor=anchor,cycle=cy,failed_replicates=rankfails))
    present={(r['anchor'],r['cycle'],r['group']) for r in rows}
    for anchor in s.ANCHORS:
        for cy in CYCLES:
            if anchor=='OGTT' and cy=='2123':continue
            for g in s.GROUPS:
                if (anchor,cy,g) not in present:rows.append(dict(anchor=anchor,cycle=cy,group=g,p=np.nan,status='unavailable'))
    table=pd.DataFrame(rows)
    table['p_Holm']=s.holm(table.p.to_numpy())
    save(out,'partial_r2_intervals',table);save(out,'partial_r2_failures',fail)
    return dict(tests=len(rows),Holm_significant=int((table.p_Holm<.05).sum()) if not table.empty else 0,complete=len(rows)==44 and table.p.notna().all() and all(r['status']=='available' for r in rows),note='Percentile intervals conditional on frozen 2005-08 map; outcome regressions refitted per replicate. Nonnegative partial R2 has boundary bias: use original design-adjusted Wald/Holm tests for significance, not CI exclusion of zero.')


def external_audit(args,out,s,features,models):
    # No automatic variable matching, merging or unit conversion from names.
    roots=[args.midus_dir] if args.midus_dir else [EDA.parent/'Data']
    files=[]; metadata=[]
    for root in roots:
        if not root.exists():continue
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.sav','.zsav','.por'):
                if args.midus_dir or re.search(r'midus|midja',str(p),re.I):files.append(p)
    try:import pyreadstat
    except ImportError:pyreadstat=None
    for p in tqdm(files,desc='127D | External metadata'):
        if pyreadstat is None:
            metadata.append(dict(file=str(p),status='install pyreadstat to inspect labels'));continue
        try:
            reader=pyreadstat.read_por if p.suffix.lower()=='.por' else pyreadstat.read_sav
            _,meta=reader(str(p),metadataonly=True)
            for col,label in zip(meta.column_names,meta.column_labels):
                metadata.append(dict(file=str(p),variable=col,label=label,status='metadata_only'))
        except Exception as e:metadata.append(dict(file=str(p),status=str(e)))
    save(out,'external_inventory',metadata)
    template={'file':'path/to/participant_level_harmonised.csv','id_column':'participant_id','cohort_name':'MIDUS (wave/specimen protocol)','units_verified':False,'fasting_and_assay_compatibility_verified':False,'eligibility_and_missing_codes_verified':False,'units':EXPECTED,'notes':'Use unique eligible nonpregnant adult participants. All columns already converted to exact NHANES units; LOG_IN natural log of positive fasting insulin. Outcome validation requires compatible covariates and assay protocols. No outcome-guided selection or refitting.','validated_anchors':[],'covariates_and_endpoint_definitions_verified':False}
    dump(out/'127_external_manifest_template.json',template)
    if not args.external_manifest:return dict(status='audit_only',files_found=len(files),metadata_rows=len(metadata),validated=False,reason='No verified participant-level mapping; no external validation claimed')
    j=json.loads(args.external_manifest.read_text(encoding='utf-8'))
    if not all(j.get(k) is True for k in ['units_verified','fasting_and_assay_compatibility_verified','eligibility_and_missing_codes_verified']) or j.get('units')!=EXPECTED:raise ValueError('External compatibility manifest not verified')
    path=Path(j['file']);path=path if path.is_absolute() else args.external_manifest.parent/path
    d=pd.read_csv(path);idc=j['id_column']
    if d[idc].isna().any() or d[idc].duplicated().any():raise ValueError('External participant identifiers invalid')
    q=d.copy();q[s.RAW]=q[s.RAW].apply(pd.to_numeric,errors='coerce');q=q.replace([np.inf,-np.inf],np.nan).dropna(subset=s.RAW)
    if len(q)<100 or (q[s.RAW[:-1]]<=0).any().any():raise ValueError('Insufficient or invalid compatible external sample')
    z=features(q[s.RAW].to_numpy(float));rows=[]
    # Frozen map from 0506 (same as HGI); external projection is not outcome validation.
    for name in ('HG7','HG_Discordant2','HG_Shared2','HGI'):
        for k,v in enumerate(z[name].T):rows.append(dict(method=name,coordinate=k+1,n=len(q),mean=float(v.mean()),sd=float(v.std()),cohort=j['cohort_name']))
    save(out,'external_projection_summary',rows);p=[]
    if j.get('validated_anchors') and j.get('covariates_and_endpoint_definitions_verified') is not True:raise ValueError('External endpoint units, assay calibration, sex coding and income-to-poverty-ratio comparability must be verified before frozen outcome evaluation')
    for anchor in j.get('validated_anchors',[]):
        if anchor not in models:raise ValueError(f'Unknown/unfitted anchor {anchor}')
        # Manifest certifies endpoint units; anchor columns are concentrations, not log values.
        cols=s.BASE_COV+['RIAGENDR',anchor];a=q.dropna(subset=cols).copy()
        if s.ANCHORS[anchor]['women_only']:a=a[a.RIAGENDR.eq(2)&a.RIDAGEYR.between(20,49)]
        a=a[a[anchor].gt(0)]
        if len(a)<100:continue
        zz=features(a[s.RAW].to_numpy(float));c=s.covariates(a,anchor);y=np.log(a[anchor].to_numpy(float))
        for name,f in models[anchor].items():p.append(dict(anchor=anchor,method=name,n=len(a),cohort=j['cohort_name'],**metrics(y,f(c,zz[name]),np.ones(len(a)))))
    save(out,'external_performance',p)
    return dict(status='frozen_external_prediction' if p else 'projection_only',validated=bool(p),n_complete=len(q),outcome_results=len(p),note='Unweighted descriptive transport on explicitly harmonised external data; not a clinical utility or population-representative claim.')


def self_test():
    rng=np.random.default_rng(127);x=rng.normal(size=(200,7));x[:,4]=.3*x[:,5]+rng.normal(size=200);w=rng.uniform(.5,2,200)
    h,_,_=fit_hgi(x,w);assert abs(norm(w)@h(x)[:,0])<1e-10
    assert np.allclose(bridge(np.array([10.]),'CDC_chain_e601'),.2243+1.079*10**(.989+.049))
    q=pd.DataFrame({'SDMVSTRA':np.repeat(np.arange(10),20),'SDMVPSU':np.tile(np.repeat([1,2],10),10),'W':w});f=q[['SDMVSTRA','SDMVPSU']].drop_duplicates()
    ww=design_weights(q,f,199,rng);assert np.allclose(ww.sum(1),1)
    y=rng.normal(size=200);p=rng.normal(size=200);d=decomp(y,p,ww);assert np.allclose(d['MSE'],d['calibration']+d['residual'])
    a=np.c_[np.ones(200),x];assert wls_rss(a,y,norm(w))<=wls_rss(a[:,:4],y,norm(w))+1e-10
    pred=ridge_model(x[:,:3],x[:,3:5],y,w);assert pred(x[:,:3],x[:,3:5]).shape==(200,)
    print('127 synthetic numerical self-test PASS')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--fetch-missing',action='store_true');ap.add_argument('--replicates',type=int,default=999);ap.add_argument('--midus-dir',type=Path);ap.add_argument('--external-manifest',type=Path);ap.add_argument('--reference-118',type=Path);ap.add_argument('--out',type=Path,default=R/'127_reviewer_validation');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:self_test();return
    if args.replicates<199:ap.error('Use at least 199 PSU replicates (999 recommended)')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);status={}; errors=[]
    dump(out/'127_completion.json',{'execution_completed':False})
    protocol={'seed':SEED,'replicates':args.replicates,'inputs':{},'assay_sources':[CDC_F,CDC_I,CDC_L], 'ferritin_bridge':'e170=10^(0.989 log10(Hitachi)+0.049); e601=0.2243+1.079 e170','HGI_reference':'2005-06 survey weighted reference; HbA1c (%) minus frozen linear prediction from fasting glucose; 2D controls HGI+FPG and HbA1c/log-insulin residuals vs FPG; published equation sensitivity HbA1c=0.024*FPG(mg/dL)+3.1 (overlaps NHANES development, not independent)', 'HGI_sources':['https://doi.org/10.1002/edm2.299','https://doi.org/10.1002/edm2.442'],'CI':'rescaled stratified PSU bootstrap with full design frame; conditional fixed maps and prediction models; refit WLS for partial R2; 95% marginal intervals','external':'metadata inventory; explicit verified harmonised CSV only','survey_weight_notes':'Cycle-specific fasting subsample WTSAF2YR (OGTT WTSOG2YR); complete-case item nonresponse not separately reweighted. Script118 weight audit preserved; pooled weight construction not independently audited.'}
    for p in [Path(__file__),EDA/'123_biological_specificity_and_leakage_audit.py',EDA/'125_adversarial_comparative_validation.py',EDA/'118_biomedical_utility_benchmark.py',EDA/'106_survey_ferritin_anchor.py',EDA/'104_representation_validation_gate.py',R/'73_frozen_phenotype_map_input.csv',R/'72_shared_private_frozen_transform.npz']:
        if p.is_file():protocol['inputs'][p.name]=hashfile(p)
    dump(out/'127_protocol.json',protocol)
    def run(key,fn):
        try:status[key]=fn()
        except Exception as e:
            status[key]={'complete':False,'error':repr(e)};errors.append(dict(module=key,error=traceback.format_exc()));print(f'{key} FAILED: {e}',flush=True)
        dump(out/'127_completion.json',{'execution_completed':False,'modules':status})
    with threadpool_limits(limits=1):
        run('B_ferritin',lambda:ferritin_analysis(out,args))
        try:
            s=load('s123_for127','123_biological_specificity_and_leakage_audit.py');s.OUT=out
            d=s.load_scaffold(R/'73_frozen_phenotype_map_input.csv');frozen=dict(np.load(R/'72_shared_private_frozen_transform.npz'))
            eligible=d.loc[s.eligible(d)].copy();np.testing.assert_allclose(s.frozen_scores(eligible[s.RAW].to_numpy(float),frozen),eligible[s.Z].to_numpy(float),rtol=0,atol=1e-7)
            samples,frames,coverage=s.build_samples(d,args.fetch_missing);save(out,'anchor_coverage',coverage)
            ref=eligible[eligible.CYCLE.eq('0506')].copy();glu=s.read_xpt('0506','GLU',['SEQN','WTSAF2YR'],args.fetch_missing);ref=ref.drop(columns=['WTSAF2YR'],errors='ignore').merge(glu,on='SEQN',how='left',validate='one_to_one');ref=ref[ref.WTSAF2YR.gt(0)]
            features,hgi=feature_builder(s,ref,ref.WTSAF2YR.to_numpy(float));dump(out/'127_hgi_reference.json',hgi)
            models={}
            def A():
                nonlocal models
                models,info=hgi_analysis(s,samples,frames,features,out,args.replicates);return info
            run('A_HGI',A);run('C_partial_R2',lambda:partial_ci(s,samples,frames,out,args.replicates));run('D_external',lambda:external_audit(args,out,s,features,models))
        except Exception as e:
            errors.append(dict(module='123_input_setup',error=traceback.format_exc()));status['123_input_setup']={'complete':False,'error':repr(e)};print(f'123 inputs FAILED: {e}',flush=True)
    save(out,'failures',errors)
    complete=all(status.get(k,{}).get('complete',False) for k in ['A_HGI','B_ferritin','C_partial_R2'])
    dump(out/'127_completion.json',{'execution_completed':True,'NHANES_modules_complete':complete,'external_outcome_validation':status.get('D_external',{}).get('validated',False),'modules':status})
    lines=['# Script 127 | Reviewer-directed validation','',f'NHANES modules complete: {complete}','', 'External validation is gated; metadata alone is not validation.','']
    for k,v in status.items():lines.extend([f'## {k}','',*['- '+str(a)+': '+str(b) for a,b in v.items()],''])
    (out/'127_READOUT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines));print(f'Outputs: {out}')
    if not complete:sys.exit(1)

if __name__=='__main__':main()
