#!/usr/bin/env python3
"""129 | Reference reconciliation and eligibility sensitivity.
Place alongside Scripts 123, 125, 127, 128 in EDA. No R/Torch or overwrites.
Reconcile literal-positive versus original >1e-12 pooled weights against raw
CDC data and archived 63/72/73 records. Exclude known RIDEXPRG=3 in a separate
sensitivity, without treating missing pregnancy status as confirmed negative.
Output aggregate tables only; no participant identifiers exported.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, importlib.util, sys, json, traceback, hashlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits

EDA=Path(__file__).resolve().parent
R=EDA/'Results'
FLOOR=1e-12
EARLY=('0506','0708')
RAW=['LBXHGB','LBXRBCSI','LBXMCVSI','LBXRDW','LBXGH','LBXGLU','LOG_IN']

def load(stem):
    p=EDA/stem; name='s129_'+stem[:3]
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def save(out,name,rows):
    (rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)).to_csv(out/f'129_{name}.csv',index=False)

def dump(out,name,obj):
    (out/f'129_{name}.json').write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def read63(path,s):
    d=pd.read_csv(path,low_memory=False);d.CYCLE=s.cycle_string(d.CYCLE)
    for k in ['SEQN','SURVEY_WT','ELIGIBLE_ADULT_NONPREG']+RAW:d[k]=pd.to_numeric(d[k],errors='coerce')
    if d.SEQN.isna().any() or d.SEQN.duplicated().any():raise ValueError(f'Invalid IDs: {path}')
    return d

def complete(d,weight='SURVEY_WT',floor=FLOOR):
    return d.ELIGIBLE_ADULT_NONPREG.eq(1)&d[weight].gt(floor)&np.isfinite(d[RAW]).all(axis=1)

def counts(raw,a63,a72,a73,cy,s):
    q=raw[raw.complete];valid=q[q.expected_weight.gt(FLOOR)];tiny=q[q.expected_weight.le(FLOOR)]
    ids=set(valid.SEQN);r={'cycle':cy,'literal_positive_complete7':len(q),'weight_at_or_below_floor':len(tiny),'original_floor_complete7':len(valid),'tiny_weight_max':tiny.expected_weight.max(),'tiny_weight_sum':tiny.expected_weight.sum(),'valid_weight_sum':valid.expected_weight.sum(),'indeterminate_complete7_literal':int(q.RIDEXPRG.eq(3).sum()),'indeterminate_complete7_original_floor':int(valid.RIDEXPRG.eq(3).sum())}
    for label,d in [('63',a63),('72',a72),('73',a73)]:
        a=d[d.CYCLE.eq(cy)];mask=complete(a)
        if label in ('72','73'):mask=mask&np.isfinite(a[s.Z]).all(axis=1)
        ai=set(a.loc[mask,'SEQN']);r[f'n_{label}']=len(ai);r[f'missing_{label}']=len(ids-ai);r[f'extra_{label}']=len(ai-ids)
    return r

def fit_reference(x,w):
    # Same canonical construction as Script 72, with fixed rank 2.
    w=np.asarray(w,float);w=w/w.sum();mu=w@x;sd=np.sqrt(w@((x-mu)**2));t=(x-mu)/sd;h,g=t[:,:4],t[:,4:]
    def inv(c):
        v,u=np.linalg.eigh((c+c.T)/2);return (u*np.maximum(v,1e-10)**-.5)@u.T
    a=inv((h*w[:,None]).T@h);b=inv((g*w[:,None]).T@g)
    u,r,vt=np.linalg.svd(a@(h*w[:,None]).T@g@b,full_matrices=False)
    _,_,last=np.linalg.svd(u.T,full_matrices=True)
    return dict(A_mean=mu[:4],A_sd=sd[:4],G_mean=mu[4:],G_sd=sd[4:],WA=a@np.column_stack([u,last[-1]]),WG=b@vt.T,rho=r)

def compare_maps(s,reference,models,out):
    rows=[]
    for label,q in reference.items():
        x=q[RAW].to_numpy(float);w=q.SURVEY_WT.to_numpy(float);base=s.frozen_scores(x,models['archived'])
        for name,m in models.items():
            z=s.frozen_scores(x,m)
            for group,ix in s.GROUPS.items():
                rows.append(dict(evaluation=label,model=name,group=group,n=len(q),similarity=s.similarity(base[:,ix],z[:,ix],w),max_rho_delta=float(np.max(np.abs(m['rho']-models['archived']['rho']))),rho1=m['rho'][0],rho2=m['rho'][1],rho3=m['rho'][2]))
    save(out,'map_similarity',rows);return pd.DataFrame(rows)

def conditional_sensitivity(s,samples,frames,models,out):
    rows=[]
    for name,m in tqdm(models.items(),desc='129B | Laboratory associations'):
        for (anchor,cy),q0 in samples.items():
            q=q0.copy()
            if name=='exclude_indeterminate_all':q=q[q.RIDEXPRG.ne(3)].copy()
            model=models['exclude_indeterminate_ref'] if name=='exclude_indeterminate_all' else m
            q[s.Z]=s.frozen_scores(q[RAW].to_numpy(float),model)
            tests,_=s.conditional_test(q,frames[cy],anchor,cy,name)
            rows+=tests
    d=pd.DataFrame(rows);d['p_Holm']=np.nan
    for name in d.model.unique():
        ix=d.model.eq(name);d.loc[ix,'p_Holm']=s.holm(d.loc[ix,'p'].to_numpy())
    save(out,'laboratory_sensitivity',d)
    old=d[d.model.eq('archived')][['anchor','cycle','group','partial_R2','p_Holm','n']]
    diff=d.merge(old,on=['anchor','cycle','group'],suffixes=('','_archived'),validate='many_to_one');diff['delta_partial_R2']=diff.partial_R2-diff.partial_R2_archived;diff['significance_changed']=(diff.p_Holm<.05)!=(diff.p_Holm_archived<.05)
    save(out,'laboratory_sensitivity_deltas',diff);return d,diff

def hgi_sensitivity(s,h,p,samples,ref,out):
    rows=[]
    for mode in tqdm(['archived_protocol','exclude_indeterminate_ref','exclude_indeterminate_all'],desc='129C | Frozen HGI sensitivity'):
        a=ref if mode=='archived_protocol' else ref[ref.RIDEXPRG.ne(3)]
        x=a[RAW].to_numpy(float);w=a.SURVEY_WT.to_numpy(float);hg=p.hg_fit(x,w);res,_,_=h.fit_hgi(x,w)
        def features(q):
            xx=q[RAW].to_numpy(float);hh=res(xx);zz=hg(xx);return {'Discordant2':zz['discordant'],'HGI_FPG2':np.c_[hh[:,0],xx[:,5]],'DirectionalResidual2':hh,'Raw7':xx}
        for anchor in s.ANCHORS:
            tr=samples[anchor,'0506'];tr=tr if mode!='exclude_indeterminate_all' else tr[tr.RIDEXPRG.ne(3)]
            c=s.covariates(tr,anchor);y=tr.Y.to_numpy(float);w=tr.W.to_numpy(float)
            fits={k:h.ridge_model(c,z,y,w) for k,z in features(tr).items()}
            for cy in ('0708','2123'):
                if (anchor,cy) not in samples:continue
                q=samples[anchor,cy];q=q if mode!='exclude_indeterminate_all' else q[q.RIDEXPRG.ne(3)]
                c=s.covariates(q,anchor);y=q.Y.to_numpy(float);w=q.W.to_numpy(float)
                for k,z in features(q).items():rows.append(dict(mode=mode,anchor=anchor,cycle=cy,method=k,n=len(q),reference_n=len(a),**h.metrics(y,fits[k](c,z),w)))
    d=pd.DataFrame(rows);save(out,'hgi_pregnancy_sensitivity',d)
    pivot=d.pivot(index=['mode','anchor','cycle','n','reference_n'],columns='method',values='MSE').reset_index()
    for k in ('HGI_FPG2','DirectionalResidual2','Raw7'):pivot[k+'_minus_D']=pivot[k]-pivot.Discordant2
    save(out,'hgi_pregnancy_contrasts',pivot)
    return d

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--fetch-missing',action='store_true');ap.add_argument('--out',type=Path,default=R/'129_reference_reconciliation');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:return self_test()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Use a new empty --out directory; archived outputs preserved')
    status={};hashes={};fail=[]
    try:
        s=load('123_biological_specificity_and_leakage_audit.py');v=load('128_final_validation.py');p=load('125_adversarial_comparative_validation.py');h=load('127_reviewer_directed_validation.py')
        paths=[R/'63_deep_ddx_input.csv',R/'72_shared_private_model_input.csv',R/'73_frozen_phenotype_map_input.csv',R/'72_shared_private_frozen_transform.npz']
        dump(out,'protocol',dict(weight_floor=FLOOR,age_minimum=20,original_pregnancy_rule='exclude code 1; code 3 retained',sensitivity_rule='exclude code 3 in reference alone or reference and evaluation; structural missing is not imputed',input_hashes={str(q):v.sha(q) for q in paths+[Path(__file__),Path(v.__file__),Path(s.__file__),Path(p.__file__),Path(h.__file__)]},fixed_rank=2,limits='Exploratory conditional sensitivities, not new external validation. No automatic rebuild. HGI prediction sensitivity descriptive.'))
        a63=read63(paths[0],s);a72=s.load_scaffold(paths[1]);a73=s.load_scaffold(paths[2]);frozen=dict(np.load(paths[3]));s.OUT=out
        oldcycles=s.CYCLES.copy();rr=[];raws=[];audits=[];markers=[]
        try:
            for cy in tqdm(v.CYCLES,desc='129A | Reference reconciliation'):
                raw,_=v.raw_frame(s,cy,args,hashes);raw['CYCLE']=cy;raws.append(raw)
                rr.append(counts(raw,a63,a72,a73,cy,s))
                for label,d in [('73',a73)]+([('63',a63)] if cy in EARLY else []):
                    row,ms=v.check_frame(cy,raw,d,label);audits.append(row);markers+=ms
        finally:s.CYCLES=oldcycles
        save(out,'reference_counts',rr);save(out,'scoped_source_audit',audits);save(out,'scoped_marker_audit',markers);dump(out,'raw_hashes',hashes)
        allraw=pd.concat(raws,ignore_index=True);rawref=allraw[allraw.CYCLE.isin(EARLY)&allraw.complete&allraw.expected_weight.gt(FLOOR)].copy();rawref['SURVEY_WT']=rawref.expected_weight
        old=a72[a72.CYCLE.isin(EARLY)&s.eligible(a72)].copy();orig63=a63[a63.CYCLE.isin(EARLY)&complete(a63)].copy();orig73=a73[a73.CYCLE.isin(EARLY)&s.eligible(a73)].copy()
        ids=set(old.SEQN);same_ids=(ids==set(rawref.SEQN)==set(orig63.SEQN)==set(orig73.SEQN))
        # Exact participant-level match in memory, never export IDs.
        score_error=float(np.max(np.abs(s.frozen_scores(old[RAW].to_numpy(float),frozen)-old[s.Z].to_numpy(float))))
        full=a73[s.eligible(a73)].copy();all_score_error=float(np.max(np.abs(s.frozen_scores(full[RAW].to_numpy(float),frozen)-full[s.Z].to_numpy(float))))
        early=[r for r in rr if r['cycle'] in EARLY];n_literal=sum(r['literal_positive_complete7'] for r in early);n_tiny=sum(r['weight_at_or_below_floor'] for r in early)
        strict=rawref[rawref.RIDEXPRG.ne(3)].copy()
        models={'archived':frozen,'raw_refit':fit_reference(rawref[RAW].to_numpy(float),rawref.SURVEY_WT.to_numpy(float)),'exclude_indeterminate_ref':fit_reference(strict[RAW].to_numpy(float),strict.SURVEY_WT.to_numpy(float))}
        sim=compare_maps(s,{'original_reference':old,'recent':full[full.CYCLE.eq('2123')]},models,out)
        eq=sim[sim.model.eq('raw_refit')];rho_delta=float(eq.max_rho_delta.max());min_sim=float(eq.similarity.min())
        # Verify all intended source memberships, not later 63/72 marker coverage.
        all73=all(r['missing_73']==r['extra_73']==0 for r in rr)
        design_marker_ok=all(r['design_pass'] and r['marker_pass'] for r in audits)
        gate=bool(same_ids and len(old)==4088 and n_literal-n_tiny==len(old) and all73 and design_marker_ok and score_error<1e-7 and all_score_error<1e-7 and rho_delta<1e-7 and min_sim>1-1e-8)
        status['A_reference']=dict(complete=gate,literal_positive_reference_n=n_literal,excluded_by_original_weight_floor=n_tiny,original_reference_n=len(old),raw_reference_n=len(rawref),same_reference_participant_ids=same_ids,scoped_source_checks_pass=design_marker_ok,all_73_memberships_match=all73,reference_score_error=score_error,all_73_score_error=all_score_error,max_rho_delta=rho_delta,min_refit_subspace_similarity=min_sim,known_indeterminate_reference_n=len(rawref)-len(strict),known_indeterminate_all_complete_n=int((allraw.complete&allraw.expected_weight.gt(FLOOR)&allraw.RIDEXPRG.eq(3)).sum()),interpretation='Tiny positive weights are counted by literal >0 but excluded by original >1e-12 rule; only verified if counts and IDs match. Later 63 marker absence is out of scope for reference fitting.')
        dump(out,'completion',status)
        # Reuse original assay sample construction; add pregnancy field by ID.
        samples,frames,_=s.build_samples(a73,args.fetch_missing);preg=allraw[['SEQN','RIDEXPRG']].copy()
        if preg.SEQN.duplicated().any():raise ValueError('Duplicate raw IDs')
        samples={k:q.drop(columns=['RIDEXPRG'],errors='ignore').merge(preg,on='SEQN',how='left',validate='one_to_one') for k,q in samples.items()}
        if len(samples)!=11:raise ValueError('Expected eleven assay-cycle samples')
        mods={**models,'exclude_indeterminate_all':models['exclude_indeterminate_ref']}
        d,delta=conditional_sensitivity(s,samples,frames,mods,out)
        status['B_pregnancy']=dict(complete=len(d)==176 and d.p_Holm.notna().all(),holm_significant={k:int((q.p_Holm<.05).sum()) for k,q in d.groupby('model')},max_partial_R2_change={k:float(q.delta_partial_R2.abs().max()) for k,q in delta.groupby('model')},changed_significance_count={k:int(q.significance_changed.sum()) for k,q in delta.groupby('model')},min_strict_subspace_similarity=float(sim[sim.model.eq('exclude_indeterminate_ref')].similarity.min()),note='Sensitivity, not a prespecified equivalence test. Review all changes before submission.')
        ref=rawref[rawref.CYCLE.eq('0506')];d=hgi_sensitivity(s,h,p,samples,ref,out)
        status['C_HGI']=dict(complete=len(d)==84,rows=len(d),note='Descriptive sensitivity of frozen HGI comparisons; original 2005-06 reference uses official weights up to constant scaling.')
        status['D_gate']=dict(reference_rebuild_required=not gate,status='reference_reconciled_review_sensitivity' if gate else 'unresolved_reference_rebuild_assessment_required',automatic_rebuild=False,all_other_analyses_reexecuted=False)
    except Exception:
        fail.append(traceback.format_exc());print(fail[-1],flush=True)
    dump(out,'completion',dict(modules=status,failures=fail));lines=['# Script 129 | Reference reconciliation','',*['## '+k+'\n\n'+'\n'.join('- '+str(a)+': '+str(b) for a,b in val.items())+'\n' for k,val in status.items()]]
    if fail:lines+=['## Failures',*fail]
    (out/'129_READOUT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines));print('Outputs:',out)
    return int(bool(fail) or not status.get('A_reference',{}).get('complete',False) or not status.get('B_pregnancy',{}).get('complete',False) or not status.get('C_HGI',{}).get('complete',False))

def self_test():
    rng=np.random.default_rng(129);x=rng.normal(size=(500,7));x[:,4:6]+=.5*x[:,:2];w=rng.uniform(.5,2,500);m=fit_reference(x,w)
    assert m['WA'].shape==(4,4) and m['WG'].shape==(3,3) and np.all(m['rho']<1)
    d=pd.DataFrame({**{k:np.ones(3) for k in RAW},'ELIGIBLE_ADULT_NONPREG':[1,1,1],'SURVEY_WT':[1.,1e-79,0.]});assert complete(d,floor=0).sum()==2 and complete(d).sum()==1
    print('129 numerical and weight-threshold self-test passed.');return 0

if __name__=='__main__':sys.exit(main())
