#!/usr/bin/env python3
"""128 | Final reviewer-directed HG7 checks. Place in EDA and run from repo root.
Uses Scripts 123, 125 and corrected 127 (no PyTorch or R required).
A: frozen CBC-inclusive HGI controls, same sample, matched downstream protocol.
B: matched canonical D-CCA identity plus released D-CCA full-rank contrast if available.
C: independent eight-cycle CDC fasting-weight, design, eligibility and marker audit.
D: dependency/impact gate, not an automatic overwrite or claimed downstream rerun.
Outputs aggregate CSV/JSON only. Previously examined NHANES samples; exploratory.
"""
from pathlib import Path
import argparse, importlib.util, sys, json, hashlib, traceback, platform, subprocess
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits

EDA=Path(__file__).resolve().parent
ROOT=EDA.parent
R=EDA/'Results'
SEED=128
CYCLES={'0506':('2005','D','NHANES',2),'0708':('2007','E','NHANES',2),
        '0910':('2009','F','NHANES_Transfer',5),'1112':('2011','G','NHANES_Transfer',5),
        '1314':('2013','H','NHANES_Transfer',5),'1516':('2015','I','NHANES_Transfer',5),
        '1718':('2017','J','NHANES_Transfer',5),'2123':('2021','L','NHANES_Transfer',1)}
HOLD=('0708','2123')
PERIOD={cy:'2005-2008' if v[3]==2 else '2009-2018' if v[3]==5 else '2021-2023' for cy,v in CYCLES.items()}
RAW=['LBXHGB','LBXRBCSI','LBXMCVSI','LBXRDW','LBXGH','LBXGLU','LOG_IN']
TWO=['HGI_Hb','HGI_RBC','HGI_MCV','HGI_RDW','HGI_CBC_PC1','CBC_PCA2','Hb_RDW','Raw_PCA2','HGI_FPG2','DirectionalResidual2','HG_Shared2']
SOURCES=['https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx',
         'https://wwwn.cdc.gov/nchs/nhanes/tutorials/SampleDesign.aspx',
         'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.htm',
         'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/INS_L.htm',
         'https://doi.org/10.1080/01621459.2018.1543599','https://github.com/shu-hai/D-CCA']

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def save(out,name,rows):
    (rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)).to_csv(out/f'128_{name}.csv',index=False)

def dump(out,name,obj):
    (out/f'128_{name}.json').write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def norm(w):
    w=np.asarray(w,float)
    if w.ndim!=1 or not np.isfinite(w).all() or (w<=0).any():raise ValueError('Invalid weights')
    return w/w.sum()

def pca_fit(x,w,k):
    w=norm(w);mu=w@x;sd=np.sqrt(w@((x-mu)**2));t=(x-mu)/sd
    _,v=np.linalg.eigh((t*w[:,None]).T@t);basis=v[:,::-1][:,:k]
    return lambda z:((z-mu)/sd)@basis

def canonical(p,x,w):
    wn=norm(w);mu,sd=p.std(x,w);t=(x-mu)/sd;h,g=t[:,:4],t[:,4:]
    a=p.invsqrt((h*wn[:,None]).T@h);b=p.invsqrt((g*wn[:,None]).T@g)
    u,r,vt=np.linalg.svd(a@(h*wn[:,None]).T@g@b,full_matrices=True)
    if len(r)<2 or np.any(r[:2]>=1-1e-10):raise ValueError('Degenerate canonical pair')
    wa=a@u;wg=b@vt.T
    def scores(z):
        zz=(z-mu)/sd;return zz[:,:4]@wa,zz[:,4:]@wg
    return scores,mu,sd,wa,wg,r[:2]

def dcca_identity(p,x,w):
    scores,mu,sd,wa,wg,r=canonical(p,x,w)
    a=.5*(1-np.sqrt((1-r)/(1+r)))
    def transform(z):
        u,v=scores(z);u=u[:,:2];v=v[:,:2];c=(u+v)*a
        return ((u-c)-(v-c))/np.sqrt(2*(1-r))
    return transform

def native_dcca(p,x,module):
    # Released implementation is unweighted. Full block ranks prevent sPOET
    # shrinkage; project returned distinctive matrices onto the matched paired
    # canonical directions. Do not subtract raw 4D and 3D matrices directly.
    w=np.ones(len(x));scores,mu,sd,wa,wg,r=canonical(p,x,w);t=(x-mu)/sd
    result=module.dCCA(t[:,:4].T,t[:,4:].T,r_1=4,r_2=3,r_12=2)
    if int(result[8])!=2:raise ValueError('Native D-CCA returned different common rank')
    np.testing.assert_allclose(result[9],r,rtol=1e-7,atol=1e-9)
    dh=np.asarray(result[4]).T;dg=np.asarray(result[5]).T
    contrast=(dh@wa[:,:2]-dg@wg[:,:2])/np.sqrt(2*(1-r))
    beta=p.affine(t,contrast)
    np.testing.assert_allclose(p.pred(t,beta),contrast,rtol=1e-7,atol=1e-8)
    return lambda z:p.pred((z-mu)/sd,beta)

def alignment(a,b):
    # Both bases fitted on training observations, not target or holdout outcomes.
    a=a-a.mean(0);b=b-b.mean(0)
    if np.linalg.matrix_rank(a)!=2 or np.linalg.matrix_rank(b)!=2:raise ValueError('Contrast rank != 2')
    qa=np.linalg.qr(a)[0];qb=np.linalg.qr(b)[0];sv=np.linalg.svd(qa.T@qb,compute_uv=False)
    return float(np.mean(sv**2)),float(np.max(np.abs(a-b)))

def features_fit(p,h,x,w,native=None):
    residual,coef,icoef=h.fit_hgi(x,w);hg=p.hg_fit(x,w)
    hpc=pca_fit(x[:,:4],w,2);rpc=pca_fit(x,w,2);dcca=dcca_identity(p,x,w)
    unweighted=p.hg_fit(x,np.ones(len(x)))
    def features(z):
        a=hg(z);g=residual(z);hh=hpc(z[:,:4]);d=a['discordant'];hgi=g[:,0]
        f={'HG_Discordant2':d,'HG_Shared2':a['shared'],'HGI':g[:,:1],
           'HGI_FPG2':np.c_[hgi,z[:,5]],'DirectionalResidual2':g,
           'HGI_CBC_PC1':np.c_[hgi,hh[:,0]],'CBC_PCA2':hh,'Hb_RDW':z[:,[0,3]],
           'Raw_PCA2':rpc(z),'CBC4':z[:,:4],'Raw7':z,'HG7':a['all'],
           'HGI_CBC4':np.c_[hgi,z[:,:4]],'HGI_CBC4_D2':np.c_[hgi,z[:,:4],d],
           'HGI_D2':np.c_[hgi,d],'DCCA_derived_weighted2':dcca(z),
           'HG_Discordant_unweighted2':unweighted(z)['discordant']}
        for j,m in enumerate(['Hb','RBC','MCV','RDW']):f['HGI_'+m]=np.c_[hgi,z[:,j]]
        if native is not None:f['DCCA_native_derived2']=native(z)
        return f
    return features,{'n':len(x),'HGI_intercept':coef[0],'HGI_slope':coef[1]}

def comparison(p,h,s,samples,frames,ref,out,args,native):
    x=ref[RAW].to_numpy(float);w=ref.WTSAF2YR.to_numpy(float)
    f,meta=features_fit(p,h,x,w,native);dump(out,'reference',meta)
    perf=[];contrasts=[];assoc=[];ident=[];fail=[]
    for anchor in tqdm(s.ANCHORS,desc='128A | CBC-inclusive controls'):
        tr=samples[anchor,'0506'];wt=tr.W.to_numpy(float);ct=s.covariates(tr,anchor);yt=tr.Y.to_numpy(float);ft=f(tr[RAW].to_numpy(float))
        fits={}
        for name,z in ft.items():
            try:fits[name]=h.ridge_model(ct,z,yt,wt)
            except Exception:fail.append({'anchor':anchor,'method':name,'stage':'fit','error':traceback.format_exc()})
        pairs=[(m,'HG_Discordant2') for m in TWO+['HGI','CBC4','Raw7','HGI_CBC4','DCCA_derived_weighted2']]
        pairs += [('HGI_CBC4','HGI_CBC4_D2'),('HGI_CBC4','HGI_D2'),('HG_Discordant2','HGI_D2'),('Raw7','HG7')]
        if native is not None:pairs.append(('DCCA_native_derived2','HG_Discordant_unweighted2'))
        for cy in HOLD:
            if (anchor,cy) not in samples:continue
            q=samples[anchor,cy];c=s.covariates(q,anchor);y=q.Y.to_numpy(float);w=q.W.to_numpy(float);z=f(q[RAW].to_numpy(float));loss={}
            # Reuse exact 127 bootstrap seed and draw protocol for comparable old controls.
            ww=h.design_weights(q,frames[cy],args.replicates,np.random.default_rng(np.random.SeedSequence([127,int(cy),list(s.ANCHORS).index(anchor)])))
            for name,fit in fits.items():
                pred=fit(c,z[name]);loss[name]=(y-pred)**2
                perf.append(dict(anchor=anchor,cycle=cy,method=name,n=len(q),dim=z[name].shape[1],**h.metrics(y,pred,w)))
            for comp,cand in pairs:
                if comp not in loss or cand not in loss:continue
                v=loss[comp]-loss[cand];draw=ww@v;res=h.summary(draw)
                # Design-adjusted intercept-only Wald test of mean paired loss.
                # Fixed fitted models; no inferential claim for algebraic identity controls.
                pv=np.nan;df=np.nan
                primary=anchor=='sTfR' and comp in TWO[:8] and cand=='HG_Discordant2'
                if primary:
                    if np.max(np.abs(v))<=1e-10:pv=1.
                    else:
                        be,vc,df=s.survey_wls(np.ones((len(q),1)),v,w,q,frames[cy]);_,pv,_=s.wald(be,vc,[0],df)
                contrasts.append(dict(anchor=anchor,cycle=cy,comparator=comp,candidate=cand,dim_comparator=z[comp].shape[1],dim_candidate=z[cand].shape[1],primary_family=primary,p_design=pv,design_df=df,**res))
            # Conditional incremental association beyond HGI + complete CBC.
            # This asks whether the contrast includes laboratory-relevant information
            # not captured by those five variables, not causal source identification.
            for baseline in ['HGI_CBC_PC1','HGI_CBC4']:
                try:
                    csc=h.scale_fit(c,w,False)(c);a=np.c_[np.ones(len(q)),csc,z[baseline],z['HG_Discordant2']]
                    b,vc,df=s.survey_wls(a,y,w,q,frames[cy]);full=h.wls_rss(a,y,h.norm(w));red=h.wls_rss(a[:,:-2],y,h.norm(w));F,pv,k=s.wald(b,vc,[a.shape[1]-2,a.shape[1]-1],df)
                    assoc.append(dict(anchor=anchor,cycle=cy,baseline=baseline,n=len(q),partial_R2=1-full/red,F=F,p=pv,df_num=k,design_df=df))
                except Exception:fail.append({'anchor':anchor,'cycle':cy,'method':baseline,'stage':'conditional','error':traceback.format_exc()})
    a=pd.DataFrame(contrasts);a['p_Holm_primary']=np.nan
    mask=a.primary_family.astype(bool);a.loc[mask,'p_Holm_primary']=s.holm(a.loc[mask,'p_design'].to_numpy())
    b=pd.DataFrame(assoc)
    if len(b):b['p_Holm']=s.holm(b.p.to_numpy())
    save(out,'cbc_performance',perf);save(out,'cbc_paired',a);save(out,'cbc_conditional',b);save(out,'cbc_failures',fail)
    # Compare prior Script 127 performance on unchanged methods.
    old=R/'127_reviewer_validation'/'127_hgi_performance.csv';checks=[]
    if old.exists():
        prev=pd.read_csv(old,dtype={'cycle':str});prev.cycle=prev.cycle.str.zfill(4)
        keep=['HG_Discordant2','HG_Shared2','HGI','HGI_FPG2','DirectionalResidual2','Raw7','HG7'];pr=pd.DataFrame(perf)
        merged=pr[pr.method.isin(keep)].merge(prev[prev.method.isin(keep)],on=['anchor','cycle','method'],suffixes=('_new','_old'),validate='one_to_one')
        for _,rr in merged.iterrows():checks.append(dict(anchor=rr.anchor,cycle=rr.cycle,method=rr.method,n_new=rr.n_new,n_old=rr.n_old,delta_MSE=rr.MSE_new-rr.MSE_old,passed=bool(rr.n_new==rr.n_old and abs(rr.MSE_new-rr.MSE_old)<1e-8)))
    save(out,'127_reproduction',checks)
    archive_ok=len(checks)==49 and all(r['passed'] for r in checks)
    return dict(complete=not fail and len(b)==14 and len(a[a.primary_family])==16 and a.loc[mask,'p_design'].notna().all() and a.status.eq('available').all() and archive_ok,contrasts=len(a),primary_tests=int(mask.sum()),primary_holm_significant=int((a.p_Holm_primary<.05).sum()),conditional_tests=len(b),archive_reproduced=archive_ok,failures=len(fail))

def dcca_checks(p,ref,samples,out,args,module):
    x=ref[RAW].to_numpy(float);w=ref.WTSAF2YR.to_numpy(float);records=[];sim=[];native_fail=[]
    for label,train,hold,wt in [('NHANES_reference',x,x,w)]+[(f'NHANES_{cy}',x,samples['sTfR',cy][RAW].to_numpy(float),w) for cy in HOLD]:
        for method,weights in [('weighted_identity',wt),('unweighted_identity',np.ones(len(train)))]:
            a=p.hg_fit(train,weights)(hold)['discordant'];b=dcca_identity(p,train,weights)(hold);ss,err=alignment(a,b)
            records.append(dict(sample=label,method=method,similarity=ss,max_abs_error=float(np.max(np.abs(a-b))),passed=ss>1-1e-10 and err<1e-7))
        if module is not None:
            try:
                a=p.hg_fit(train)(hold)['discordant'];b=native_dcca(p,train,module)(hold);ss,err=alignment(a,b)
                records.append(dict(sample=label,method='released_DCCA_full_rank',similarity=ss,max_abs_error=float(np.max(np.abs(a-b))),passed=ss>1-1e-10 and err<1e-7))
            except Exception:native_fail.append(dict(sample=label,error=traceback.format_exc()))
    for si,sc in enumerate(tqdm(p.SCENARIOS,desc='128B | D-CCA contrast')):
        for rep in range(args.sim_reps):
            rng=np.random.default_rng(np.random.SeedSequence([125,si,rep]));rot=[np.linalg.qr(rng.normal(size=(k,k)))[0] for k in (4,3)]
            a,t=p.generator(rng,500,sc,rot);b,u=p.generator(rng,1500,sc,rot);hg=p.hg_fit(a);d=dcca_identity(p,a,np.ones(len(a)))
            z=hg(a)['discordant'];zz=hg(b)['discordant'];base=p.r2(u['discordant'],p.pred(zz,p.affine(z,t['discordant'])),t['discordant'].mean(0))
            for method,f in [('unweighted_identity',d)]+([('released_DCCA_full_rank',None)] if module is not None else []):
                try:
                    if f is None:f=native_dcca(p,a,module)
                    za=f(a);zb=f(b);value=p.r2(u['discordant'],p.pred(zb,p.affine(za,t['discordant'])),t['discordant'].mean(0));ss,err=alignment(zz,zb)
                    sim.append(dict(scenario=sc,rep=rep,method=method,HG_R2=base,DCCA_R2=value,delta_R2=base-value,similarity=ss,max_abs_error=float(np.max(np.abs(zz-zb))),passed=ss>1-1e-10 and abs(base-value)<1e-8))
                except Exception:native_fail.append(dict(sample=sc,rep=rep,method=method,error=traceback.format_exc()))
    save(out,'dcca_real',records);save(out,'dcca_simulation',sim);save(out,'dcca_failures',native_fail)
    a=pd.DataFrame(sim);summary=a.groupby(['scenario','method']).agg(n=('delta_R2','size'),mean_HG_R2=('HG_R2','mean'),mean_DCCA_R2=('DCCA_R2','mean'),mean_delta_R2=('delta_R2','mean'),min_similarity=('similarity','min'),max_abs_error=('max_abs_error','max'),all_pass=('passed','all')).reset_index();save(out,'dcca_summary',summary)
    return dict(complete=all(r['passed'] for r in records) and bool(a.passed.all()) and not native_fail and len(a)==len(p.SCENARIOS)*args.sim_reps*(2 if module is not None else 1),native_available=module is not None,native_completed=module is not None and not native_fail and len(a[a.method.eq('released_DCCA_full_rank')])==len(p.SCENARIOS)*args.sim_reps,simulation_checks=len(a),failures=len(native_fail),interpretation='Matched identity expected, not a superiority test; native released code unweighted with fixed full block ranks.')

def read_source(s,cy,kind,cols,fetch,hashes):
    # Use 123's CDC XPT downloader, extending its explicit cycle table only.
    year,suffix,folder,_=CYCLES[cy]
    s.CYCLES[cy]=(year,suffix,folder)
    q=s.read_xpt(cy,kind,cols,fetch)
    stem=f'{kind}_{suffix}'
    path=next((p for p in (ROOT/'Data'/folder/cy/f'{stem}.XPT',ROOT/'Data'/folder/cy/f'{stem}.xpt') if p.is_file()),None)
    if path is not None:hashes[str(path.relative_to(ROOT))]=sha(path)
    return q

def raw_frame(s,cy,args,hashes):
    def get(k,c):return read_source(s,cy,k,['SEQN']+c,args.fetch_missing,hashes)
    d=get('DEMO',['RIDAGEYR','RIAGENDR','RIDEXPRG','SDMVSTRA','SDMVPSU'])
    g=get('GLU',['WTSAF2YR','LBXGLU']+(['LBXIN'] if cy in ['0506','0708','0910','1112'] else []))
    if 'LBXIN' not in g:
        i=get('INS',['LBXIN','WTSAF2YR']).rename(columns={'WTSAF2YR':'INS_weight'})
        g=g.merge(i,on='SEQN',how='left',validate='one_to_one')
    else:g['INS_weight']=g.WTSAF2YR
    z=d.merge(g,on='SEQN',how='left',validate='one_to_one')
    z=z[z.WTSAF2YR.gt(0)].copy()
    z=z.merge(get('CBC',RAW[:4]),on='SEQN',how='left',validate='one_to_one')
    z=z.merge(get('GHB',['LBXGH']),on='SEQN',how='left',validate='one_to_one')
    z['LOG_IN']=np.log(z.LBXIN.where(z.LBXIN.gt(0)))
    z['eligible']=z.RIDAGEYR.ge(20)&z.RIDEXPRG.ne(1)
    z['complete']=z.eligible & np.isfinite(z[RAW]).all(axis=1)
    z['strict_unknown_excluded']=z.eligible & z.RIDEXPRG.eq(3)
    z['expected_weight']=z.WTSAF2YR/CYCLES[cy][3]
    z['expected_stratum']=cy+'_'+z.SDMVSTRA.astype('Int64').astype(str)
    z['expected_psu']=z.expected_stratum+'_'+z.SDMVPSU.astype('Int64').astype(str)
    return z,d

def check_frame(cy,raw,archive,label):
    """Compare all positive-fasting participants, not merely outcome complete cases."""
    arc=archive[archive.CYCLE.eq(cy)].copy();arc=arc.rename(columns={k:k+'_arch' for k in arc.columns if k!='SEQN'})
    m=raw.merge(arc,on='SEQN',how='outer',indicator=True,validate='one_to_one');both=m[m._merge.eq('both')].copy()
    def numeric(a,b,rtol=0,atol=1e-8):
        if a not in both or b not in both:return -1
        return int((~np.isclose(pd.to_numeric(both[a],errors='coerce'),pd.to_numeric(both[b],errors='coerce'),rtol=rtol,atol=atol,equal_nan=True)).sum())
    def string(a,b):
        if b not in both:return -1
        return int(both[a].astype(str).ne(both[b].astype(str)).sum())
    counts={'source':label,'cycle':cy,'expected_fasting_n':len(raw),'archived_n':len(arc),'matched_n':len(both),
            'missing_from_archive':int(m._merge.eq('left_only').sum()),'extra_in_archive':int(m._merge.eq('right_only').sum()),
            'weight_mismatch':numeric('expected_weight','SURVEY_WT_arch',rtol=1e-9),
            'stratum_mismatch':numeric('SDMVSTRA','SDMVSTRA_arch'), 'psu_mismatch':numeric('SDMVPSU','SDMVPSU_arch'),
            'nested_stratum_mismatch':string('expected_stratum','STRATUM_arch'),'nested_psu_mismatch':string('expected_psu','PSU_arch'),
            'age_mismatch':numeric('RIDAGEYR','RIDAGEYR_arch'),'sex_mismatch':numeric('RIAGENDR','RIAGENDR_arch'),
            'eligibility_mismatch':numeric('eligible','ELIGIBLE_ADULT_NONPREG_arch'),
            'period_mismatch':int(both.PERIOD_arch.astype(str).ne(PERIOD[cy]).sum()) if 'PERIOD_arch' in both else -1,
            'expected_eligible_n':int(raw.eligible.sum()),'expected_complete7_n':int(raw.complete.sum()),
            'known_indeterminate_pregnancy_n':int(raw.strict_unknown_excluded.sum()),
            'indeterminate_pregnancy_complete7_n':int((raw.strict_unknown_excluded & raw.complete).sum()),
            'expected_weight_sum':float(raw.expected_weight.sum()),'archived_weight_sum':float(pd.to_numeric(arc.SURVEY_WT_arch,errors='coerce').sum())}
    # Some historical tables keep raw design labels only under nested identifiers.
    # Require nested identifiers; raw label checks are optional if columns absent.
    required=['missing_from_archive','extra_in_archive','weight_mismatch','nested_stratum_mismatch','nested_psu_mismatch','age_mismatch','sex_mismatch','eligibility_mismatch','period_mismatch']
    counts['design_pass']=all(counts[k]==0 for k in required) and all(counts[k] in (0,-1) for k in ['stratum_mismatch','psu_mismatch'])
    markers=[]
    if all(k+'_arch' in both for k in RAW):
        arch_valid=both.ELIGIBLE_ADULT_NONPREG_arch.eq(1)&both.SURVEY_WT_arch.gt(0)&np.isfinite(both[[k+'_arch' for k in RAW]]).all(axis=1)
        counts['complete7_membership_mismatch']=int((arch_valid!=both.complete).sum());counts['archived_complete7_n']=int(arch_valid.sum())
        for k in RAW:
            a=pd.to_numeric(both[k],errors='coerce').to_numpy(float);b=pd.to_numeric(both[k+'_arch'],errors='coerce').to_numpy(float)
            scope=(both.complete|arch_valid).to_numpy(bool);finite=scope&np.isfinite(a)&np.isfinite(b)
            missing=scope&(np.isfinite(a)!=np.isfinite(b));diff=finite&~np.isclose(a,b,rtol=1e-8,atol=1e-8)
            markers.append(dict(source=label,cycle=cy,marker=k,n_compared=int(finite.sum()),missingness_mismatch=int(missing.sum()),value_mismatch=int(diff.sum()),max_abs_diff=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else np.nan))
        counts['marker_pass']=counts['complete7_membership_mismatch']==0 and all(r['missingness_mismatch']==r['value_mismatch']==0 for r in markers)
    else:counts['marker_pass']=False
    # Relative weights on eligible matched subjects, plus absolute scale check above.
    q=both[both.complete & both.SURVEY_WT_arch.gt(0)]
    counts['max_normalised_weight_difference']=float(np.max(np.abs(norm(q.expected_weight)-norm(q.SURVEY_WT_arch)))) if len(q) else np.nan
    return counts,markers

def survey_audit(s,d,out,args):
    rows=[];markers=[];designs=[];ins=[];errors=[];hashes={};pool={p:[] for p in set(PERIOD.values())}
    # 63 is the input from which Script 72 actually estimated its map.
    a63=R/'63_deep_ddx_input.csv';archive63=None
    if a63.is_file():
        archive63=pd.read_csv(a63,low_memory=False);archive63.CYCLE=s.cycle_string(archive63.CYCLE)
    oldcycles=s.CYCLES.copy()
    try:
        for cy in tqdm(CYCLES,desc='128C | Independent CDC audit'):
            try:
                raw,demo=raw_frame(s,cy,args,hashes)
                for label,arc in [('73',d)]+([('63',archive63)] if archive63 is not None else []):
                    row,ms=check_frame(cy,raw,arc,label);rows.append(row);markers.extend(ms)
                f=demo[['SDMVSTRA','SDMVPSU']].dropna().drop_duplicates();f=f.assign(cycle=cy);pool[PERIOD[cy]].append(f)
                full=raw[['SDMVSTRA','SDMVPSU']].dropna().drop_duplicates()
                designs.append(dict(cycle=cy,full_demo_strata=f.SDMVSTRA.nunique(),full_demo_psus=len(f),full_demo_df=len(f)-f.SDMVSTRA.nunique(),fasting_strata=full.SDMVSTRA.nunique(),fasting_psus=len(full),fasting_df=len(full)-full.SDMVSTRA.nunique(),singleton_fasting_strata=int((full.groupby('SDMVSTRA').size()<2).sum()),missing_demo_design=int(raw[['SDMVSTRA','SDMVPSU']].isna().any(axis=1).sum())))
                q=raw[raw.complete];ins.append(dict(cycle=cy,n_complete7=len(q),insulin_fasting_weight_mismatch=int((~np.isclose(q.INS_weight,q.WTSAF2YR,rtol=1e-9,atol=1e-8)).sum())))
            except Exception:errors.append(dict(cycle=cy,error=traceback.format_exc()))
            save(out,'survey_audit',rows);save(out,'raw_marker_audit',markers);save(out,'survey_design',designs);save(out,'insulin_weight_audit',ins);save(out,'survey_failures',errors)
    finally:s.CYCLES=oldcycles
    pooled=[]
    for period,fs in sorted(pool.items()):
        if not fs:continue
        f=pd.concat(fs,ignore_index=True);orig=f[['SDMVSTRA','SDMVPSU']].drop_duplicates();nest=f.drop_duplicates()
        pooled.append(dict(period=period,cycles=f.cycle.nunique(),original_strata=orig.SDMVSTRA.nunique(),nested_strata=len(f[['cycle','SDMVSTRA']].drop_duplicates()),original_psus=len(orig),nested_psus=len(nest),original_df=len(orig)-orig.SDMVSTRA.nunique(),nested_df=len(nest)-len(f[['cycle','SDMVSTRA']].drop_duplicates()),identical_partition=orig.SDMVSTRA.nunique()==len(f[['cycle','SDMVSTRA']].drop_duplicates())))
    save(out,'pooled_design',pooled);dump(out,'raw_source_hashes',hashes)
    # Verify coverage and the exact weight/design for the 2005-08 map source.
    required=len(CYCLES)*2
    design_ok=len(rows)==required and all(r['design_pass'] for r in rows) and not errors and all(r['missing_demo_design']==0 and r['singleton_fasting_strata']==0 for r in designs) and len(pooled)==3 and all(r['identical_partition'] for r in pooled) and all(r['insulin_fasting_weight_mismatch']==0 for r in ins)
    # Script 63 may have incomplete/older raw marker values; these are recorded
    # separately from the verified weight and design partition. Its map input
    # is important, so marker discrepancies block a clean full-source verdict.
    marker_ok=len(rows)==required and all(r['marker_pass'] for r in rows)
    # No item-nonresponse adjustment is inferred or silently added.
    return dict(complete=design_ok and marker_ok,design_pass=design_ok,raw_marker_pass=marker_ok,source63_available=archive63 is not None,cycles_audited=len(designs),errors=len(errors),indeterminate_pregnancy_complete7_n=sum(r['indeterminate_pregnancy_complete7_n'] for r in rows if r['source']=='73'),note='Eligibility replicates Script 47: age >=20, exclude RIDEXPRG==1; code 3 indeterminate counted separately. Full survey design audited, not item-nonresponse reweighting or assay harmonisation. Raw 63 and 73 marker disagreements reported separately.')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--fetch-missing',action='store_true');ap.add_argument('--replicates',type=int,default=999);ap.add_argument('--sim-reps',type=int,default=50);ap.add_argument('--dcca',type=Path,default=ROOT/'D-CCA'/'dcca.py');ap.add_argument('--out',type=Path,default=R/'128_final_validation');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:return self_test()
    if args.replicates<199 or args.sim_reps<5:ap.error('Use >=199 bootstrap and >=5 simulation repetitions; defaults recommended.')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'128_protocol.json').exists():raise FileExistsError('Existing run preserved. Use --out EDA/Results/128_final_validation_rerun or another empty directory.')
    s=load(EDA/'123_biological_specificity_and_leakage_audit.py','s123_for128');h=load(EDA/'127_reviewer_directed_validation.py','s127_for128');p=load(EDA/'125_adversarial_comparative_validation.py','s125_for128')
    source=R/'73_frozen_phenotype_map_input.csv';frozen=R/'72_shared_private_frozen_transform.npz';files=[Path(__file__),Path(s.__file__),Path(h.__file__),Path(p.__file__),source,frozen,R/'63_deep_ddx_input.csv',ROOT/'NHANES_Scripts'/'47_domain_corrected_combined_AG_inference.py',args.dcca,R/'127_reviewer_validation'/'127_hgi_performance.csv']
    try:commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    except Exception:commit=None
    dump(out,'protocol',dict(seed=SEED,replicates=args.replicates,sim_reps=args.sim_reps,git_commit=commit,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,source_hashes={str(f):sha(f) for f in files if f.is_file()},sources=SOURCES,primary_family='16 fixed-model sTfR paired weighted-loss comparisons: eight CBC-inclusive or all-panel 2D controls, two holdouts, Holm-adjusted design-based tests. Remaining prediction intervals marginal and exploratory.',rank='2 fixed reference canonical pairs',weights='WTSAF2YR / 2, / 5, / 1; raw masked strata/PSU partition versus inherited nested identifiers',limitations=['Previously examined NHANES cycles','No independent external validation','Conditional bootstrap intervals do not include model-development uncertainty','No claim of superiority over algebraically equivalent D-CCA','No automatic changes to upstream or published outputs']))
    status={};fail=[]
    def run(key,fn):
        try:status[key]=fn()
        except Exception:status[key]={'complete':False,'error':traceback.format_exc()};fail.append(dict(module=key,error=status[key]['error']));print(key,'FAILED',status[key]['error'],flush=True)
        dump(out,'completion',dict(execution_completed=False,modules=status))
    with threadpool_limits(limits=1):
        d=s.load_scaffold(source);s.OUT=out
        run('C_survey',lambda:survey_audit(s,d,out,args))
        # Independently of audit result, exploratory comparisons on original 127
        # sample may be diagnostic; do not use them to claim cleared submission
        # if weight/eligibility discrepancies are found.
        try:
            q=d[s.eligible(d)].copy();np.testing.assert_allclose(s.frozen_scores(q[s.RAW].to_numpy(float),dict(np.load(frozen))),q[s.Z].to_numpy(float),rtol=0,atol=1e-7)
            samples,frames,coverage=s.build_samples(d,args.fetch_missing);save(out,'anchor_coverage',coverage)
            if len(samples)!=11:raise ValueError('Expected 11 available assay-cycle samples')
            ref=q[q.CYCLE.eq('0506')].drop(columns=['WTSAF2YR'],errors='ignore').merge(s.read_xpt('0506','GLU',['SEQN','WTSAF2YR'],args.fetch_missing),on='SEQN',how='left',validate='one_to_one');ref=ref[ref.WTSAF2YR.gt(0)]
            if len(ref)!=1796:raise ValueError(f'Expected original 127 reference n=1796, found {len(ref)}')
            module=None;native=None
            if args.dcca.is_file():
                try:module=load(args.dcca,'native_dcca_for128');native=native_dcca(p,ref[RAW].to_numpy(float),module)
                except Exception:fail.append(dict(module='native_DCCA_setup',error=traceback.format_exc()))
            run('A_CBC',lambda:comparison(p,h,s,samples,frames,ref,out,args,native))
            run('B_DCCA',lambda:dcca_checks(p,ref,samples,out,args,module))
        except Exception:fail.append(dict(module='setup',error=traceback.format_exc()));print(fail[-1]['error'])
    clean=all(status.get(k,{}).get('complete',False) for k in ['A_CBC','B_DCCA','C_survey']) and status.get('B_DCCA',{}).get('native_completed',False) and not fail
    # Fail closed: do not rerun scripts in place or edit weights in a frozen map.
    impact={'complete':clean,'status':'no_source_discrepancy_detected' if status.get('C_survey',{}).get('complete') else 'review_and_rebuild_required','automatic_rerun_performed':False,'affected_if_weight_or_eligibility_diff':['47 source survey frame','62/63 physiological scaffold','69 raw bank','72 frozen representation','73 projection','101-127 affected analyses and figures'],'instruction':'If the audit fails, inspect the aggregate mismatch tables; resolve source version, eligibility, raw marker or survey-design differences before rebuilding the full dependency chain in a separate run. Do not patch only downstream weights or overwrite existing results. If D-CCA absent rerun with --dcca path/to/dcca.py.'}
    status['D_impact_gate']=impact;save(out,'failures',fail);dump(out,'completion',dict(execution_completed=True,all_checks_complete=clean,external_validation=False,modules=status))
    lines=['# Script 128 | Final reviewer-directed validation','',f'All checks complete: {clean}','', 'Completion does not mean HG7 won comparisons. Inspect all effect estimates and unfavourable results.', '']
    for k,v in status.items():lines+=['## '+k,'',*['- '+str(a)+': '+str(b) for a,b in v.items()],'']
    (out/'128_READOUT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines));print('Outputs:',out)
    return 0 if clean else 1

def self_test():
    # No local datasets required. Use actual existing implementations if present.
    h=load(EDA/'127_reviewer_directed_validation.py','s127_selftest128');p=load(EDA/'125_adversarial_comparative_validation.py','s125_selftest128')
    rng=np.random.default_rng(128);x=rng.normal(size=(500,7));x[:,4:6]+=.4*x[:,:2];w=rng.uniform(.5,2,len(x));y=rng.normal(size=(150,7))
    hg=p.hg_fit(x,w);f=dcca_identity(p,x,w);np.testing.assert_allclose(hg(y)['discordant'],f(y),rtol=0,atol=1e-10)
    f,_=features_fit(p,h,x,w);a=f(x);b=f(y);assert all(a[k].shape[1]==2 for k in TWO)
    for k,z in a.items():
        c=x[:,[2]];m=h.ridge_model(c,z,x[:,0]+rng.normal(size=len(x)),w);assert np.isfinite(m(y[:,[2]],b[k])).all()
    raw=pd.DataFrame({'SEQN':[1,2],'RIDAGEYR':[20,30],'RIAGENDR':[1,2],'RIDEXPRG':[np.nan,3],'eligible':[True,True],'complete':[True,True],'strict_unknown_excluded':[False,True],'expected_weight':[1.,2.],'SDMVSTRA':[1.,1.],'SDMVPSU':[1.,2.],'expected_stratum':['0506_1']*2,'expected_psu':['0506_1_1','0506_1_2'],**{k:[1.,2.] for k in RAW}})
    arc=raw.copy();arc['CYCLE']='0506';arc['PERIOD']='2005-2008';arc['SURVEY_WT']=raw.expected_weight;arc['STRATUM']=raw.expected_stratum;arc['PSU']=raw.expected_psu;arc['ELIGIBLE_ADULT_NONPREG']=1
    row,_=check_frame('0506',raw,arc,'test');assert row['design_pass'] and row['marker_pass'];arc.loc[0,'SURVEY_WT']=2;row,_=check_frame('0506',raw,arc,'test');assert not row['design_pass']
    print('128 synthetic self-test passed (canonical identity, model fits, design mismatch detection).')
    return 0

if __name__=='__main__':sys.exit(main())
