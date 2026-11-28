#!/usr/bin/env python3
"""125 | Adversarial comparative validation. Place alongside Scripts 114 and 123 in EDA.

Default: run numerical controls, 6 misspecified simulation settings and independent
assay holdouts. Optional public-code comparators require local clones (see --help).
No outcome or assay is used to fit the representations. The real-data reference
is 2005–06 ONLY; this is a new refit, not the manuscript's original 2005–08 HG7.
Temporal cohorts have previously been inspected. No causal or clinical claims.
"""
from pathlib import Path
import argparse, contextlib, hashlib, importlib.util, io, json, platform, shutil, sys, traceback
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

EDA=Path(__file__).resolve().parent
SEED=125
GROUPS={'shared':2,'discordant':2,'H_private':2,'G_private':1}
SCENARIOS=['asymmetric_mixing','nonlinear_coupling','correlated_factors','heteroskedastic','contamination','independent_blocks']
BASE=['HG7','PCA7','White7','Raw7','JIVE7','AJIVE7_adaptive','DCCA','FiGuRO_fixed','FiGuRO_adaptive']

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x): p.write_text(json.dumps(x,indent=2,default=str),encoding='utf-8')
def weights(w,n):
    w=np.ones(n) if w is None else np.asarray(w,float)
    if len(w)!=n or not np.isfinite(w).all() or np.any(w<=0): raise ValueError('Invalid weights')
    return w/w.sum()
def std(x,w=None):
    w=weights(w,len(x)); mu=w@x; sd=np.sqrt(w@((x-mu)**2))
    if np.any(sd<1e-12): raise ValueError('Constant input')
    return mu,sd

def invsqrt(c):
    l,v=np.linalg.eigh(c)
    if l.min()<=1e-10: raise ValueError('Singular covariance')
    return (v*l**-.5)@v.T

def centered_whiten(x,w=None):
    w=weights(w,len(x)); mu=w@x; c=x-mu
    return mu,invsqrt((c*w[:,None]).T@c)

def affine(x,y,w=None):
    w=weights(w,len(x)); a=np.c_[np.ones(len(x)),x]; s=np.sqrt(w)[:,None]
    return np.linalg.lstsq(a*s,np.asarray(y)*s,rcond=None)[0]
def pred(x,b): return np.c_[np.ones(len(x)),x]@b

def group_map(z):
    return {'all':z,'shared':z[:,:2],'discordant':z[:,2:4],'H_private':z[:,4:6],'G_private':z[:,6:7]}

def hg_fit(x,w=None):
    w=weights(w,len(x)); mu,sd=std(x,w); t=(x-mu)/sd; h,g=t[:,:4],t[:,4:]
    a=invsqrt((h*w[:,None]).T@h); b=invsqrt((g*w[:,None]).T@g)
    p,r,q=np.linalg.svd(a@(h*w[:,None]).T@g@b,full_matrices=True)
    if np.any(r[:2]>=1-1e-10): raise ValueError('Near-singular canonical pair')
    def transform(new):
        u=((new[:,:4]-mu[:4])/sd[:4])@a@p
        v=((new[:,4:]-mu[4:])/sd[4:])@b@q.T
        z=np.c_[(u[:,:2]+v[:,:2])/np.sqrt(2*(1+r[:2])),(u[:,:2]-v[:,:2])/np.sqrt(2*(1-r[:2])),u[:,2:],v[:,2]]
        return group_map(z)
    return transform

def linear_controls(x,w=None):
    w=weights(w,len(x)); mu,sd=std(x,w); t=(x-mu)/sd
    c=(t*w[:,None]).T@t; l,v=np.linalg.eigh(c); v=v[:,::-1]; white=invsqrt(c)
    return {'Raw7':lambda y:{'all':(y-mu)/sd},'PCA7':lambda y:{'all':((y-mu)/sd)@v},'White7':lambda y:{'all':((y-mu)/sd)@white}}

def rank(x): return np.linalg.matrix_rank(x-x.mean(0),tol=1e-7*max(1,np.linalg.norm(x-x.mean(0),2)))

def normalize_groups(d):
    out={}
    for k,v in d.items():
        k={'Shared':'shared','Discordant':'discordant'}.get(k,k); a=np.asarray(v,float)
        if a.ndim!=2 or not np.isfinite(a).all(): raise ValueError(f'Invalid group {k}')
        if a.shape[1] and rank(a)==a.shape[1]: out[k]=a
    return out

def dcca_fit(x,module):
    # Official D-CCA with predeclared full block ranks and common rank 2.
    # Original code is transductive; frozen affine adapter validated on training
    # outputs. Group bases are native output-matrix PC scores, not raw D-CCA factors.
    mu,sd=std(x); t=(x-mu)/sd
    result=module.dCCA(t[:,:4].T,t[:,4:].T,r_1=4,r_2=3,r_12=2)
    _,_,ch,cg,dh,dg,*_=result
    native={'shared':np.c_[ch.T,cg.T],'H_private':dh.T,'G_private':dg.T}
    maps={}
    for name,z in native.items():
        u,s,v=np.linalg.svd(z-z.mean(0),full_matrices=False); k=int(np.sum(s>s[0]*1e-8)) if len(s) and s[0]>0 else 0
        if k==0: continue
        scores=(z-z.mean(0))@v[:k].T; coef=affine(t,scores)
        if np.linalg.norm(pred(t,coef)-scores)/np.linalg.norm(scores)>1e-7: raise ValueError('D-CCA adapter not affine')
        maps[name]=coef
    def transform(y):
        a={k:pred((y-mu)/sd,b) for k,b in maps.items()}
        if a: a['all']=np.column_stack(list(a.values()))
        return a
    return transform

def figuro_fit(x,module,args,seed,adaptive):
    import torch
    torch.manual_seed(seed); torch.set_num_threads(args.torch_threads)
    mu,sd=std(x); t=(x-mu)/sd
    # Official module, raw standardized 4/3 marker blocks as embeddings;
    # use its reverse transformation layers to support arbitrary block sizes.
    dims=[3,4,3] if adaptive else [2,2,1]
    model=module.FiGuRO(n_modalities=2,latent_dims=[4,3],decomp_dims=dims,
                        transformation_layers=True,rank_reduction_frequency=10,distortion_metric='R2')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    # Disjoint internal validation rows used only for adaptive rank tracking.
    # No phenotype, assay or external data is involved.
    ids=np.random.default_rng(seed).permutation(len(t)); cut=int(.8*len(t)); tr,va=t[ids[:cut]],t[ids[cut:]]
    def blocks(a): return [torch.tensor(a[:,:4],dtype=torch.float32),torch.tensor(a[:,4:],dtype=torch.float32)]
    trb,vab=blocks(tr),blocks(va)
    if adaptive: model.initialize_tracking(epochs=args.epochs,warmup=min(40,args.epochs//3))
    for epoch in range(args.epochs):
        model.train(); opt.zero_grad(); rec=model(trb); loss=sum((a-b).square().mean() for a,b in zip(rec,trb))/2
        if not torch.isfinite(loss): raise ValueError('Nonfinite FiGuRO loss')
        loss.backward(); opt.step()
        if adaptive:
            model.eval()
            with torch.no_grad(): model.step(epoch,model(vab),vab)
    model.eval()
    def transform(y):
        with torch.no_grad():
            sh,private=model.encode(blocks((y-mu)/sd)); dims=model.active_dims
            vals=[sh,*private]; d={k:v[:,:int(n)].cpu().numpy().astype(float) for k,v,n in zip(['shared','H_private','G_private'],vals,dims)}
            d['all']=np.column_stack(list(d.values())); return d
    return transform

class Methods:
    def __init__(self,args):
        self.args=args; self.modules={}; self.status=[]
        self.selected=args.methods
        for name,path in [('DCCA',args.dcca),('FiGuRO',args.figuro)]:
            if path:
                try: self.modules[name]=load(path,'external_'+name)
                except Exception as e: self.status.append({'method':name,'status':'unavailable','reason':repr(e)})
        if any(m in self.selected for m in ['JIVE7','AJIVE7_adaptive']):
            try:
                names=['114_structure_recovery.py','103_jive_ajive_benchmark.py','112_finish_benchmarks.py']
                self.s114,self.s103,self.s112=[load(EDA/n,'s125_'+n[:3]) for n in names]
                self.helpers=self.s114.cca_functions(EDA/'72_shared_private_decomposition_and_phenotype.py')
                self.rscript=Path(args.rscript or shutil.which('Rscript') or Path(sys.prefix)/'Lib/R/bin/x64/Rscript.exe')
            except Exception as e: self.status.append({'method':'JIVE/AJIVE','status':'unavailable','reason':repr(e)})
    def fit(self,name,x,seed,w=None):
        if name=='HG7': return hg_fit(x,w)
        if name in ['Raw7','White7','PCA7']: return linear_controls(x,w)[name]
        if name=='DCCA':
            if 'DCCA' not in self.modules: raise RuntimeError('Pass --dcca path/to/official/dcca.py')
            return dcca_fit(x,self.modules['DCCA'])
        if name.startswith('FiGuRO'):
            if 'FiGuRO' not in self.modules: raise RuntimeError('Pass --figuro path/to/official/src/figuro.py')
            return figuro_fit(x,self.modules['FiGuRO'],self.args,seed,name.endswith('adaptive'))
        if name=='JIVE7' and not self.rscript.is_file(): raise RuntimeError('Rscript unavailable')
        return self.s114.fit_method(name,x,self.helpers,self.s103,self.s112,self.rscript,self.args.timeout)

def generator(rng,n,scenario,rot):
    s,d,h,g=np.split(rng.normal(size=(n,7)),[2,4,6],axis=1)
    if scenario=='correlated_factors': d=.65*s+np.sqrt(1-.65**2)*d; h=.55*s+np.sqrt(1-.55**2)*h
    # Deliberately unequal and mixed loadings, not exact pairwise sums/differences.
    a=np.c_[s[:,0]+.25*d[:,1],.4*s[:,0]+.8*s[:,1]-.4*d[:,0],h[:,0]+.3*d[:,0],h[:,1]+.3*s[:,1]]
    b=np.c_[.45*s[:,0]+.85*s[:,1]+.5*d[:,0],.75*s[:,0]-.65*d[:,1]+.2*g[:,0],g[:,0]-.3*d[:,0]]
    if scenario=='nonlinear_coupling':
        a[:,0]=np.tanh(2*s[:,0])+d[:,1]*.3; b[:,0]=s[:,0]**2-.8+s[:,1]*.4; b[:,1]+=s[:,1]*d[:,0]*.5
    if scenario=='independent_blocks': a=rng.normal(size=(n,4)); b=rng.normal(size=(n,3))
    noise=.2 if scenario!='heteroskedastic' else (.1+.6*np.abs(s[:,0]))[:,None]
    a=a@rot[0]+rng.normal(size=a.shape)*noise; b=b@rot[1]+rng.normal(size=b.shape)*noise
    if scenario=='contamination':
        mask=rng.random(n)<.05; a[mask]+=rng.normal(0,5,(mask.sum(),4)); b[mask]+=rng.normal(0,3,(mask.sum(),3))
    return np.c_[a,b],{'shared':s,'discordant':d,'H_private':h,'G_private':g}

def r2(y,p,mu=None,w=None):
    w=weights(w,len(y)); mu=w@y if mu is None else mu
    return float(1-np.sum(w[:,None]*(y-p)**2)/max(np.sum(w[:,None]*(y-mu)**2),1e-20))

def fixed_groups(train,test):
    # Native groups (first native directions, no outcome-informed selection).
    # PCA/whitening prefixes are neutral dimension controls, not semantic groups.
    for group,k in GROUPS.items():
        if group in train and group in test and train[group].shape[1]>=k:
            yield group,train[group][:,:k],test[group][:,:k], 'native' if train[group].shape[1]==k else 'native_truncated'
        if 'all' in train and 'all' in test and train['all'].shape[1]>=k:
            yield group,train['all'][:,:k],test['all'][:,:k],'prefix_control'

def sim_run(args,methods,out,errors):
    rows=[]; metadata=[]; rng0=np.random.default_rng(args.seed)
    for si,scenario in enumerate(SCENARIOS):
        for rep in tqdm(range(args.reps),desc=scenario):
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,si,rep])); rot=[np.linalg.qr(rng.normal(size=(k,k)))[0] for k in (4,3)]
            x,t=generator(rng,args.n_train,scenario,rot); y,u=generator(rng,args.n_test,scenario,rot)
            for method in methods.selected:
                try:
                    f=methods.fit(method,x,int(rng0.integers(2**31-1))); a,b=normalize_groups(f(x)),normalize_groups(f(y))
                    dim=a.get('all',np.empty((len(x),0))).shape[1]
                    metadata.append(dict(scenario=scenario,rep=rep,method=method,total_dim=dim,**{k+'_dim':a.get(k,np.empty((len(x),0))).shape[1] for k in GROUPS}))
                    if 'all' in a and 'all' in b:
                        rec=r2(y,pred(b['all'],affine(a['all'],x)),x.mean(0))
                        for target in t:
                            rows.append(dict(scenario=scenario,rep=rep,method=method,group='all',target=target,kind='all',dim=dim,R2=r2(u[target],pred(b['all'],affine(a['all'],t[target])),t[target].mean(0)),reconstruction_R2=rec))
                    for group,z,v,kind in fixed_groups(a,b):
                        for target in t:
                            rows.append(dict(scenario=scenario,rep=rep,method=method,group=group,target=target,kind=kind,dim=z.shape[1],R2=r2(u[target],pred(v,affine(z,t[target])),t[target].mean(0))))
                except Exception as e: errors.append(dict(stage='simulation',scenario=scenario,rep=rep,method=method,reason=repr(e)))
        pd.DataFrame(rows).to_csv(out/'125_sim_recovery.csv',index=False)
    df=pd.DataFrame(rows); pd.DataFrame(metadata).to_csv(out/'125_sim_dimensions.csv',index=False)
    if df.empty:return
    # Native designated signal comparisons; also compare neutral prefixes only
    # for PCA/White/Raw, not competitor's arbitrary group ordering.
    d=df[df.group.eq(df.target) & df.scenario.ne('independent_blocks')].copy()
    h=d[d.method.eq('HG7') & d.kind.eq('native')]
    comp=d[~d.method.eq('HG7') & (d.kind.ne('prefix_control') | d.method.isin(['PCA7','Raw7','White7']))]
    comp=comp.merge(h[['scenario','rep','group','dim','R2']],on=['scenario','rep','group','dim'],suffixes=('','_HG'))
    pairs=[]
    for key,q in comp.groupby(['scenario','method','group','kind','dim']):
        v=(q.R2_HG-q.R2).to_numpy(); n=len(v)
        if n<5:continue
        rng=np.random.default_rng(args.seed+1); boot=v[rng.integers(n,size=(args.boot,n))].mean(1)
        p=float(wilcoxon(v).pvalue) if np.any(v) else 1.
        pairs.append(dict(zip(['scenario','method','group','kind','dim'],key),n=n,delta_R2=v.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),p_signed_rank=p))
    pairs=pd.DataFrame(pairs); adjust(pairs,'p_signed_rank'); pairs.to_csv(out/'125_sim_paired.csv',index=False)
    # Selectivity: intended target recovery minus strongest off-target recovery.
    # Negative values and correlated-factor settings are retained, not suppressed.
    keys=['scenario','rep','method','group','kind','dim']
    group_rows=df[df.group.ne('all')].copy()
    on=group_rows[group_rows.group.eq(group_rows.target)][keys+['R2']].rename(columns={'R2':'on_R2'})
    off=group_rows[group_rows.group.ne(group_rows.target)].groupby(keys,as_index=False).R2.max().rename(columns={'R2':'max_off_R2'})
    sel=on.merge(off,on=keys);sel['selectivity_margin']=sel.on_R2-sel.max_off_R2
    sel.to_csv(out/'125_sim_selectivity.csv',index=False)
    # Equal total dimensionality only. No comparison of lossy against 7D as if matched.
    full=df[df.group.eq('all')]; hg=full[full.method.eq('HG7')]
    equal=full[~full.method.eq('HG7')].merge(hg[['scenario','rep','target','dim','R2']],on=['scenario','rep','target','dim'],suffixes=('','_HG'))
    equal['delta_R2']=equal.R2_HG-equal.R2
    equal.to_csv(out/'125_sim_equal_total_dimension.csv',index=False)

def adjust(df,col):
    if col not in df:return
    df[col+'_holm']=np.nan; ix=df.index[np.isfinite(df[col])]; ix=df.loc[ix,col].sort_values().index
    if len(ix):df.loc[ix,col+'_holm']=np.minimum(1,np.maximum.accumulate(df.loc[ix,col].to_numpy()*np.arange(len(ix),0,-1)))

def numerical(x,w,methods,out):
    # Weighted covariance conditioning, using equal sample and standardized design.
    # Full-rank affine predictions compared via an arbitrary deterministic target.
    y=np.sin(x[:,0])+x[:,4]; rows=[]
    for m in ['Raw7','White7','PCA7','HG7']:
        z=methods.fit(m,x,125,w=w)(x)['all']; mu,sd=std(z,w); t=(z-mu)/sd; wn=weights(w,len(z)); c=(t*wn[:,None]).T@t
        rows.append(dict(method=m,n=len(x),max_VIF=np.diag(np.linalg.inv(c)).max(),condition_number=np.linalg.cond(t*np.sqrt(wn[:,None])),reconstruction_R2=r2(x,pred(z,affine(z,x,w)),w=wn),max_prediction_diff=np.max(np.abs(pred(z,affine(z,y[:,None],w))-pred(x,affine(x,y[:,None],w))))))
    pd.DataFrame(rows).to_csv(out/'125_numerical.csv',index=False)

def biology(args,methods,out,errors):
    s=load(EDA/'123_biological_specificity_and_leakage_audit.py','s123_for125'); s.OUT=out
    d=s.load_scaffold(EDA/'Results/73_frozen_phenotype_map_input.csv'); ref=d[d.CYCLE.eq('0506') & s.eligible(d)].copy()
    x=ref[s.RAW].to_numpy(float); w=ref.SURVEY_WT.to_numpy(float); numerical(x,w,methods,out)
    samples,frames,coverage=s.build_samples(d,args.fetch_missing)
    maps={}; dimensions=[]
    for method in tqdm(methods.selected,desc='Reference fits'):
        try: maps[method]=methods.fit(method,x,args.seed)
        except Exception as e:errors.append(dict(stage='assay_reference',method=method,reason=repr(e)))
    # Add survey-weighted HG as explicitly different sensitivity, and random
    # whitened rotations. Random controls are not labelled physiological groups.
    maps['HG7_weighted']=hg_fit(x,w)
    mu,wh=centered_whiten(x); rng=np.random.default_rng(args.seed+3)
    for i in range(args.rotations):
        q,r=np.linalg.qr(rng.normal(size=(7,7))); q=q*np.where(np.diag(r)<0,-1,1)
        maps[f'Random{i:03d}']=lambda y,q=q:{'all':(y-mu)@wh@q}
    assoc=[]; performance=[]; contrasts=[]; random_rows=[]
    for anchor in tqdm(s.ANCHORS,desc='Assays'):
        if (anchor,'0506') not in samples:continue
        tr=samples[anchor,'0506']; wt=tr.W.to_numpy(float); yt=tr.Y.to_numpy(float); ct=s.covariates(tr,anchor); cm,cs=std(ct,wt); ct=(ct-cm)/cs
        # Cache training groups and learn affine whitening on training anchor sample only.
        fitted={}
        for method,f in maps.items():
            try:
                z=normalize_groups(f(tr[s.RAW].to_numpy(float)))
                for group,a,_,kind in fixed_groups(z,z):
                    if kind=='prefix_control' and not (method in ['Raw7','PCA7','White7'] or method.startswith('Random')):continue
                    if method.startswith('Random') and group in ['discordant','H_private']:continue # same 2D prefix, avoid duplicate controls
                    zm,zw=centered_whiten(a,wt); v=(a-zm)@zw
                    reg=Ridge(alpha=10).fit(np.c_[ct,v],yt,sample_weight=wt/wt.mean())
                    fitted[method,group,kind]=(zm,zw,reg)
            except Exception as e: errors.append(dict(stage='assay_train',method=method,anchor=anchor,reason=repr(e)))
        base=Ridge(alpha=10).fit(ct,yt,sample_weight=wt/wt.mean())
        for cycle in ['0708','2123']:
            if (anchor,cycle) not in samples:continue
            q=samples[anchor,cycle]; ww=q.W.to_numpy(float); yy=q.Y.to_numpy(float); c=(s.covariates(q,anchor)-cm)/cs
            loss0=(yy-base.predict(c))**2; losses={}; groups_cache={}
            for (method,group,kind),(zm,zw,reg) in fitted.items():
                try:
                    if method not in groups_cache: groups_cache[method]=normalize_groups(maps[method](q[s.RAW].to_numpy(float)))
                    zs=groups_cache[method]; a=zs[group if kind!='prefix_control' else 'all'][:,:GROUPS[group]]
                    z=(a-zm)@zw; pred_y=reg.predict(np.c_[c,z]); loss=(yy-pred_y)**2
                    key=(method,group,kind); losses[key]=loss
                    row=dict(anchor=anchor,cycle=cycle,method=method,group=group,kind=kind,dim=GROUPS[group],n=len(q),MSE=float(np.average(loss,weights=ww)),delta_cov_MSE=float(np.average(loss0-loss,weights=ww)))
                    if method.startswith('Random'):random_rows.append(row);continue
                    performance.append(row)
                    # Independent association beyond covariates, NOT conditional on
                    # remaining coordinates. This estimand permits lossy competitors.
                    des=np.c_[np.ones(len(q)),c,z]; b,vc,df=s.survey_wls(des,yy,ww,q,frames[cycle]); F,p,k=s.wald(b,vc,list(range(des.shape[1]-z.shape[1],des.shape[1])),df)
                    red=des[:,:-z.shape[1]]; mse0=s.mse(yy,red@s.fit_wls(red,yy,ww),ww); mse1=s.mse(yy,des@b,ww)
                    assoc.append(dict(**row,partial_R2=(mse0-mse1)/mse0,F=F,p=p,design_df=df))
                except Exception as e:errors.append(dict(stage='assay_test',method=method,anchor=anchor,cycle=cycle,group=group,reason=repr(e)))
            for group in GROUPS:
                h=losses.get(('HG7',group,'native'))
                if h is None:continue
                comp=[(k,v) for k,v in losses.items() if k[1]==group and k[0]!='HG7' and not k[0].startswith('Random')]
                if comp:
                    point,lo,hi=s.bootstrap_contrasts(q,frames[cycle],np.column_stack([v-h for _,v in comp]),args.boot,np.random.default_rng(args.seed+5))
                    for i,(k,_) in enumerate(comp):contrasts.append(dict(anchor=anchor,cycle=cycle,group=group,comparator=k[0],kind=k[2],dim=GROUPS[group],delta_MSE=float(point[i]),lo=float(lo[i]),hi=float(hi[i])))
                kgroup='shared' if GROUPS[group]==2 else 'G_private'
                vals=[np.average(v,weights=ww) for k,v in losses.items() if k[0].startswith('Random') and k[1]==kgroup]
                if vals:dimensions.append(dict(anchor=anchor,cycle=cycle,group=group,dim=GROUPS[group],n_random=len(vals),HG_percentile_better=100*np.mean(np.array(vals)>np.average(h,weights=ww))))
    for name,rows in [('assay_performance',performance),('assay_paired',contrasts),('random_controls',random_rows),('random_percentiles',dimensions),('assay_associations',assoc)]:
        df=pd.DataFrame(rows)
        if name=='assay_associations':adjust(df,'p')
        df.to_csv(out/f'125_{name}.csv',index=False)

def self_test():
    rng=np.random.default_rng(125); x=rng.normal(size=(300,7)); x[:,4]+=.4*x[:,0]; w=rng.uniform(.2,2,len(x)); z=hg_fit(x,w)(x)['all']
    assert rank(z)==7; assert np.max(np.abs(pred(z,affine(z,x,w))-x))<1e-9
    for sc in SCENARIOS:
        a,t=generator(rng,100,sc,[np.eye(4),np.eye(3)]); assert a.shape==(100,7) and np.isfinite(a).all()
        assert all(v.shape[0]==100 for v in t.values())
    print('PASS: weighted HG7 inverse, synthetic generators, full-rank controls')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',choices=BASE,default=BASE); p.add_argument('--figuro',type=Path,help='Official FiGuRO clone: src/figuro.py'); p.add_argument('--dcca',type=Path,help='Official D-CCA clone: dcca.py')
    p.add_argument('--reps',type=int,default=50);p.add_argument('--n-train',type=int,default=500);p.add_argument('--n-test',type=int,default=1500);p.add_argument('--epochs',type=int,default=300);p.add_argument('--torch-threads',type=int,default=1);p.add_argument('--boot',type=int,default=999);p.add_argument('--rotations',type=int,default=199);p.add_argument('--seed',type=int,default=SEED);p.add_argument('--rscript',type=Path);p.add_argument('--timeout',type=int,default=1200);p.add_argument('--run-name',default='125_adversarial_validation');p.add_argument('--fetch-missing',action='store_true');p.add_argument('--simulation-only',action='store_true');p.add_argument('--biology-only',action='store_true');p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:return self_test()
    if args.simulation_only and args.biology_only:p.error('Select only one phase')
    if args.reps<5 or min(args.n_train,args.n_test)<50 or args.boot<199 or args.epochs<20 or args.rotations<99:p.error('Minimum reps 5, samples 50, boot 199, epochs 20, rotations 99')
    if 'HG7' not in args.methods:p.error('HG7 required')
    if not args.run_name.replace('_','').replace('-','').isalnum():p.error('Invalid run name')
    out=EDA/'Results'/args.run_name;out.mkdir(parents=True,exist_ok=True)
    if (out/'125_protocol.json').exists():raise FileExistsError('Choose a new --run-name; existing results preserved')
    methods=Methods(args); errors=list(methods.status)
    # Skip absent external implementations once, rather than generating hundreds of identical failures.
    for m in list(methods.selected):
        missing=(m=='DCCA' and 'DCCA' not in methods.modules) or (m.startswith('FiGuRO') and 'FiGuRO' not in methods.modules) or (m.startswith(('JIVE','AJIVE')) and not hasattr(methods,'helpers')) or (m=='JIVE7' and not getattr(methods,'rscript',Path('.')).is_file())
        if missing:errors.append(dict(stage='setup',method=m,reason='Official implementation/dependency unavailable; not evaluated'));methods.selected=[k for k in methods.selected if k!=m]
    files=[Path(__file__),args.figuro,args.dcca,*[EDA/n for n in ['114_structure_recovery.py','103_jive_ajive_benchmark.py','112_finish_benchmarks.py','72_shared_private_decomposition_and_phenotype.py','123_biological_specificity_and_leakage_audit.py','Results/73_frozen_phenotype_map_input.csv']]]
    write_json(out/'125_protocol.json',dict(arguments=vars(args),methods=methods.selected,scenarios=SCENARIOS,python=platform.python_version(),numpy=np.__version__,source_hashes={str(f):digest(f) for f in files if f and f.is_file()},limits=['Unweighted matched representations; weighted HG sensitivity','FiGuRO fixed budget 2/2/1 and adaptive capacity 3/4/3 are distinct; not full paper reproduction','D-CCA fixed full block ranks/common rank 2, native output matrices converted to frozen affine scores; not exact original out-of-sample method','No native discordance group for FiGuRO, D-CCA, JIVE or AJIVE; no semantic superiority claim on unmatched groups','Prefix controls are neutral and not physiologically labelled','Simulated generators are illustrative; independent-block truth serves null recovery control','Paired signed-rank tests and Holm; bootstrap intervals for mean difference are descriptive and unadjusted','Assay partial R2 beyond covariates only; Holm across computed association tests; PSU contrast intervals conditional on fitted models and unadjusted','Random percentiles descriptive; no assay is asserted a biological negative control','Previously inspected NHANES periods; no independent clinical or causal validation','BFHD official key_code.py has unfilled TrainConfig assignments and cannot be imported as released; HCL not evaluated (no verified integration); PULSE not comparable on cross-sectional design','Single neural seed per simulation replicate and per real-data reference fit; no neural hyperparameter search; seed sensitivity remains untested']))
    completed=[]
    for label,fn,skip in [('simulation',sim_run,args.biology_only),('biology',biology,args.simulation_only)]:
        if skip:continue
        try:fn(args,methods,out,errors);completed.append(label)
        except Exception:errors.append(dict(stage=label,reason=traceback.format_exc()))
        pd.DataFrame(errors).to_csv(out/'125_failures.csv',index=False)
    write_json(out/'125_completion.json',dict(completed_phases=completed,requested_methods=args.methods,attempted_methods=methods.selected,recorded_issues=len(errors),outputs=sorted(f.name for f in out.glob('*.csv'))))
    lines=['# Script 125 readout','','Completed phases: '+', '.join(completed),'','Attempted methods: '+', '.join(methods.selected),'',f'Recorded issues: {len(errors)}. See 125_failures.csv.','', 'No superiority is implied by completion. Inspect matched effect estimates, rank eligibility, negative controls and all adverse findings.','']
    for f in ['125_numerical.csv','125_sim_paired.csv','125_assay_paired.csv','125_assay_associations.csv','125_random_percentiles.csv']:
        if (out/f).exists():lines.append('- '+f)
    (out/'125_READOUT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines));print('Output:',out)

if __name__=='__main__':main()
