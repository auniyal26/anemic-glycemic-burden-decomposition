#!/usr/bin/env python3
"""Script 97 | Can phenotype Y identify one reversible code for seven raw X measurements?

Run from repository root:
    python EDA/97_X_code_from_Y_identifiability.py
    python EDA/97_X_code_from_Y_identifiability.py --self-test

Requires existing EDA/96_Y_code_cryptanalysis_audit.py and REAL NHANES source
files used by Script 96. No synthetic data fallback and no frozen-model edits.
Train ONLY 2005-06; held-out 2007-08, 2009-18, 2021-23.

Separation of questions:
  1. X <-> code_X is mathematically reversible at fixed 0.001 precision.
  2. Is Y -> code_X identifiable? Audit collisions/conditional ambiguity.
  3. Can learned phenotype frequencies retrieve/decode a plausible X code?
  4. How accurate are decoded continuous X and A/G quantile signatures?

CAVEATS: No causal identification, no patient-level diagnostic inference.
Codebook recovery is limited to REAL 2005-06 training observations. An
oracle top-k score uses test X for EVALUATION ONLY and is not deployable.
"""
from __future__ import annotations
import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'Results'
PRIOR = HERE / '96_Y_code_cryptanalysis_audit.py'
SEED = 97
PERIODS = ('2007-2008', '2009-2018', '2021-2023')


def import_96():
    if not PRIOR.is_file():
        raise FileNotFoundError(f'Missing tested Script 96, expected {PRIOR}')
    spec = importlib.util.spec_from_file_location('audit96_for_97', PRIOR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # Redirect the Script-96 loader's diagnostic to a new subfolder,
    # avoiding modifications to existing Script-96 output files.
    m.OUTPUT = OUT / '97_loader_diagnostics'
    m.OUTPUT.mkdir(parents=True, exist_ok=True)
    return m


def weights(d):
    w = d['SURVEY_WT'].to_numpy(float)
    if not (np.isfinite(w).all() and np.all(w > 0)):
        raise ValueError('Invalid survey weights')
    return w / w.mean()


def quantize(X):
    if not np.isfinite(X).all():
        raise ValueError('Missing or nonfinite X is not permissible in the paired experiment')
    if float(np.max(np.abs(X))) > 1e9:
        raise OverflowError('Physiology out of safe quantization range')
    return np.rint(X * 1000).astype(np.int64)


def natrow(q):
    return tuple(2 * int(v) if v >= 0 else -2 * int(v) - 1 for v in q)


def codes_for(Q, codec, width, max_bits):
    return [codec.encode(natrow(row), 'decimal_interleave', width, max_bits)
            for row in Q]


def decode_code(code, codec, width):
    unsigned = codec.decode(code, 'decimal_interleave', 7, width)
    return np.array([codec.signed(int(v)) / 1000 for v in unsigned], dtype=float)


def codec_audit(d, codec, max_bits, n_per_period):
    rng = np.random.default_rng(SEED)
    r = []
    for p, section in d.groupby('SPLIT', sort=False):
        sample = section.iloc[np.sort(rng.choice(len(section), size=min(len(section), n_per_period), replace=False))]
        X = sample[codec.X].to_numpy(float)
        Q = quantize(X)
        U = [natrow(q) for q in Q]
        widths = {name: max(1, max(len(str(x)) for vals in U for x in vals))
                  for name in ('X_only', 'X_plus_total_Y')}
        for domain in ('X_only', 'X_plus_total_Y'):
            vals_list = U if domain == 'X_only' else [t + (int(y),) for t, y in zip(U, sample.PHQ9_TOTAL)]
            n = len(vals_list[0])
            width = widths[domain]
            if domain == 'X_plus_total_Y':
                width = max(width, 2)
            for method in codec.METHODS:
                status = 'ok'; codes = []; failures = 0; size_max = 0
                if method == 'two_adic' and n != 2:
                    status = 'not_defined_for_full_vector'
                else:
                    try:
                        for vals in vals_list:
                            code = codec.encode(vals, method, width, max_bits)
                            recovered = codec.decode(code, method, n, width)
                            failures += (recovered != vals)
                            codes.append(code)
                            size_max = max(size_max, len(code) if isinstance(code, str) else code.bit_length())
                    except (OverflowError, ValueError, MemoryError) as e:
                        status = 'not_feasible:' + str(e)
                r.append(dict(period=p, domain=domain, n_dimensions=n, method=method,
                              tested=len(vals_list), status=status, roundtrip_failures=failures if status == 'ok' else None,
                              distinct_input_states=len(set(vals_list)),
                              distinct_codes=len(set(codes)) if status == 'ok' else None,
                              collisions=(len(set(vals_list)) - len(set(codes))) if status == 'ok' else None,
                              max_code_digits_or_bits=size_max if status == 'ok' else None))
    return pd.DataFrame(r)


def ykeys(d, feature, yitems):
    if feature == 'total':
        return [tuple(row) for row in d[['PHQ9_TOTAL']].to_numpy(int)]
    if feature == 'pair':
        return [tuple(row) for row in d[['SOMATIC_SCORE','COGAFF_SUM']].to_numpy(int)]
    if feature == 'items':
        return [tuple(row) for row in d[yitems].to_numpy(int)]
    raise ValueError(feature)


def identifiability(d, full_codes, yitems):
    # These are descriptive statistics and IN-SAMPLE combinatorial upper bounds,
    # not empirical accuracy on unseen individuals.
    rows = []
    for period, dd in d.groupby('SPLIT', sort=False):
        observed_codes = np.array(full_codes[period], dtype=object)
        for feature in ('total', 'pair', 'items'):
            ys = ykeys(dd, feature, yitems)
            groups = {}
            for k, code in zip(ys, observed_codes):
                groups.setdefault(k, []).append(code)
            ambiguous = sum(len(v) for v in groups.values() if len(set(v)) > 1)
            max_correct = sum(max(pd.Series(v).value_counts().to_numpy()) for v in groups.values())
            repeated = sum(len(v) for v in groups.values() if len(v) > 1)
            largest = max(len(v) for v in groups.values())
            rows.append(dict(period=period, phenotype=feature, n_people=len(dd),
                n_distinct_Y_keys=len(groups), n_distinct_X_codes=len(set(observed_codes)),
                n_people_in_repeated_Y_groups=repeated,
                n_people_with_ambiguous_Y_to_X=ambiguous,
                fraction_ambiguous=ambiguous/len(dd),
                largest_Y_group=largest,
                best_possible_in_sample_deterministic_exact_recovery=max_correct/len(dd)))
    return pd.DataFrame(rows)


def tokens(row, order):
    # Position-anchored analog of letters, bigrams, and trigrams.
    out = []
    for k in range(1, order + 1):
        for inds in itertools.combinations(range(9), k):
            out.append('N'+str(k)+'_'+'_'.join(str(i)+'v'+str(int(row[i])) for i in inds))
    return ' '.join(out)


def make_candidate_models(tr, items):
    yt = tr[items].to_numpy(int)
    texts2 = [tokens(row, 2) for row in yt]
    texts3 = [tokens(row, 3) for row in yt]
    models = {}
    for order, texts in [(2,texts2),(3,texts3)]:
        tf = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                             dtype=np.float32, norm='l2')
        mat = tf.fit_transform(texts)
        models[f'cryptanalysis_{order}grams'] = (tf, mat)
    return models


def make_ridge_models(tr, Xcols, items):
    targ = tr[Xcols].to_numpy(float)
    w = weights(tr)
    return {feat: make_pipeline(StandardScaler(), Ridge(alpha=10.)).fit(
        tr[cols].to_numpy(float), targ, ridge__sample_weight=w)
        for feat, cols in [('total',['PHQ9_TOTAL']),
                           ('pair',['SOMATIC_SCORE','COGAFF_SUM']),
                           ('items',items)]}


def get_coarse_bins(X, cutoffs):
    return np.column_stack([np.searchsorted(cutoffs[j], X[:,j], side='left')
                             for j in range(X.shape[1])]).astype(np.int8)


def predicted_scores(te, tr, tf_models, items, method, i, stop):
    # Only PHQ data enter the similarity scores; never held-out X or Z.
    a = te.iloc[i:stop]
    if method == 'prior_weighted':
        return np.zeros((len(a),len(tr)),dtype=np.float32)
    if method == 'total':
        x = np.abs(a.PHQ9_TOTAL.to_numpy(float)[:,None] - tr.PHQ9_TOTAL.to_numpy(float)[None,:])
        return (-x).astype(np.float32)
    if method == 'pair':
        u = a[['SOMATIC_SCORE','COGAFF_SUM']].to_numpy(float)
        v = tr[['SOMATIC_SCORE','COGAFF_SUM']].to_numpy(float)
        return (-np.sum(np.abs(u[:,None,:]-v[None,:,:]),axis=2)).astype(np.float32)
    if method == 'items':
        u = a[items].to_numpy(np.int8)
        v = tr[items].to_numpy(np.int8)
        return (-np.count_nonzero(u[:,None,:]!=v[None,:,:],axis=2)).astype(np.float32)
    if method.startswith('cryptanalysis_'):
        tf, base = tf_models[method]
        order = int(method.split('_')[1][0])
        other = tf.transform([tokens(row,order) for row in a[items].to_numpy(int)])
        return (other @ base.T).toarray().astype(np.float32)
    raise ValueError(method)


def softmax_top(scores):
    s = scores.astype(float) - np.max(scores)
    w = np.exp(s)
    return w / w.sum()


def eval_retrieval(te, tr, tf_models, items, train_codes, test_codes,
                   tr_bins, te_bins, codec, width, max_k, mu, sd, batch):
    Xtr = tr[codec.X].to_numpy(float)
    Xte = te[codec.X].to_numpy(float)
    qtr = quantize(Xtr)
    qte = quantize(Xte)
    wtr = weights(tr)
    wte = weights(te)
    priorbias = 0.025 * np.log(np.maximum(wtr,1e-12))
    methods = ['prior_weighted','total','pair','items','cryptanalysis_2grams','cryptanalysis_3grams']
    rec = []
    for method in methods:
        hits = np.zeros(len(te),dtype=float)
        top_k_code = np.zeros(len(te),dtype=float)
        top1_coarse = np.zeros(len(te),dtype=float)
        topk_coarse = np.zeros(len(te),dtype=float)
        coarse_per_variable = np.zeros(len(te),dtype=float)
        oracle_dist = np.zeros(len(te),dtype=float)
        pred_decode = np.zeros_like(Xte)
        pred_mean = np.zeros_like(Xte)
        for start in range(0,len(te),batch):
            end = min(start+batch,len(te))
            scores = predicted_scores(te,tr,tf_models,items,method,start,end)
            scores += priorbias[None,:]
            # Deterministic tie-breaking for reproducible ranks.
            inds = np.argsort(-scores,axis=1,kind='stable')[:,:max_k]
            v = scores[np.arange(len(inds))[:,None],inds]
            for b, ix in enumerate(inds):
                row = start+b
                top = int(ix[0]); decoded = decode_code(train_codes[top],codec,width)
                if not np.array_equal(quantize(decoded[None,:])[0],qtr[top]):
                    raise AssertionError('Codebook retrieval decoded to a different physiological state')
                pred_decode[row] = decoded
                # The candidate average is NOT itself decoded from one integer.
                ws = softmax_top(v[b])
                pred_mean[row] = ws @ Xtr[ix]
                hits[row] = float(train_codes[top] == test_codes[row])
                top_k_code[row] = float(test_codes[row] in {train_codes[int(t)] for t in ix})
                top1_coarse[row] = float(np.array_equal(tr_bins[top],te_bins[row]))
                topk_coarse[row] = float(np.any(np.all(tr_bins[ix]==te_bins[row][None,:],axis=1)))
                coarse_per_variable[row] = float(np.mean(tr_bins[top] == te_bins[row]))
                oracle_dist[row] = float(np.min(np.linalg.norm((Xtr[ix]-Xte[row][None,:])/sd[None,:],axis=1)/math.sqrt(7)))
        for kind, pred in [('decoded_top1',pred_decode),('topk_mean_NOT_single_code',pred_mean)]:
            sr2, smse = codec.standardized_score(Xte,pred,wte,mu,sd)
            rec.append(dict(period=te.SPLIT.iloc[0],model=method,output=kind,n=len(te),
                mean_weighted_R2=float(sr2.mean()),A_mean_R2=float(sr2[:4].mean()),G_mean_R2=float(sr2[4:].mean()),
                mean_standardized_MSE=float(smse.mean()),
                exact_code_top1=float(np.average(hits,weights=wte)),
                exact_code_hit_at_k=float(np.average(top_k_code,weights=wte)),
                all7_coarse_exact_top1=float(np.average(top1_coarse,weights=wte)),
                all7_coarse_hit_at_k=float(np.average(topk_coarse,weights=wte)),
                mean_single_biomarker_coarse_accuracy=float(np.average(coarse_per_variable,weights=wte)),
                oracle_min_distance_topk_standardized=float(np.average(oracle_dist,weights=wte)),
                k=max_k,status='ok'))
        print(f'    {te.SPLIT.iloc[0]} {method}: top1 decoded X R2={rec[-2]["mean_weighted_R2"]:.4f}; all7-bin={rec[-2]["all7_coarse_exact_top1"]:.3f}')
    return rec


def eval_ridge(te,tr,models,items,codec,mu,sd):
    truth=te[codec.X].to_numpy(float); ww=weights(te)
    records=[]
    for name, cols in [('total',['PHQ9_TOTAL']),('pair',['SOMATIC_SCORE','COGAFF_SUM']),('items',items)]:
        p=models[name].predict(te[cols].to_numpy(float))
        r,m=codec.standardized_score(truth,p,ww,mu,sd)
        records.append(dict(period=te.SPLIT.iloc[0],model='ridge_'+name,output='continuous_prediction_NOT_code',
            n=len(te),mean_weighted_R2=float(r.mean()),A_mean_R2=float(r[:4].mean()),
            G_mean_R2=float(r[4:].mean()),mean_standardized_MSE=float(m.mean()),status='ok'))
    p=np.tile(mu,(len(te),1))
    r,m=codec.standardized_score(truth,p,ww,mu,sd)
    records.append(dict(period=te.SPLIT.iloc[0],model='prior_train_mean',output='continuous_prediction_NOT_code',
            n=len(te),mean_weighted_R2=float(r.mean()),A_mean_R2=float(r[:4].mean()),
            G_mean_R2=float(r[4:].mean()),mean_standardized_MSE=float(m.mean()),status='ok'))
    return records


def run(args):
    codec=import_96()
    if args.self_test:
        self_test(codec)
        return
    print('SCRIPT 97 | X-code recovery from Y (no test physiology available to decoder)')
    df=codec.load()
    if not set(codec.X).issubset(df.columns):
        raise ValueError('Source file lacks raw biomarkers')
    df=df.loc[df[codec.X].notna().all(axis=1)].copy()
    groups={k:g.reset_index(drop=True) for k,g in df.groupby('SPLIT',sort=False)}
    if set(['train_2005-2006',*PERIODS]) != set(groups):
        raise RuntimeError(f'Missing a period: {sorted(groups)}')
    train = groups['train_2005-2006']
    print('COMPLETE REAL PAIRED OBSERVATIONS:',{p:len(g) for p,g in groups.items()})
    Q={p:quantize(g[codec.X].to_numpy(float)) for p,g in groups.items()}
    # Declared encoding width is fixed in advance, NEVER inferred from held-out X.
    # X is bounded to 1e9 before quantization, so unsigned integers fit in 16 digits.
    width=16
    allcodes={p:codes_for(Q[p],codec,width,args.max_bits) for p in groups}
    for p in groups:
        for i in list(range(min(10,len(groups[p])))):
            if not np.array_equal(quantize(decode_code(allcodes[p][i],codec,width)[None,:])[0],Q[p][i]):
                raise AssertionError('Exact physiological codec failed')
    codec_df=codec_audit(df,codec,args.max_bits,args.codec_audit_rows)
    codec_df.to_csv(OUT/'97_joint_codec_feasibility.csv',index=False)
    identities=identifiability(df,allcodes,codec.YITEMS)
    identities.to_csv(OUT/'97_identifiability.csv',index=False)
    print('\nPHENOTYPE AMBIGUITY:')
    print(identities[identities.period=='train_2005-2006'][['phenotype','n_distinct_Y_keys','n_distinct_X_codes','fraction_ambiguous','best_possible_in_sample_deterministic_exact_recovery']].to_string(index=False))
    Xtr=train[codec.X].to_numpy(float); wtr=weights(train)
    mu=np.average(Xtr,axis=0,weights=wtr)
    sd=np.sqrt(np.average((Xtr-mu)**2,axis=0,weights=wtr)).clip(1e-8)
    # Discovery-only quantile boundaries for coarse physiological signature checks.
    cutoffs=[np.quantile(Xtr[:,i],[1/3,2/3]) for i in range(7)]
    bins={p:get_coarse_bins(g[codec.X].to_numpy(float),cutoffs) for p,g in groups.items()}
    tf_models=make_candidate_models(train,codec.YITEMS)
    reg_models=make_ridge_models(train,codec.X,codec.YITEMS)
    metrics=[]
    for p in PERIODS:
        te=groups[p]
        print(f'\nPERIOD {p} n={len(te)}')
        metrics.extend(eval_ridge(te,train,reg_models,codec.YITEMS,codec,mu,sd))
        metrics.extend(eval_retrieval(te,train,tf_models,codec.YITEMS,
              allcodes['train_2005-2006'],allcodes[p],bins['train_2005-2006'],bins[p],
              codec,width,args.k,mu,sd,args.batch))
    result=pd.DataFrame(metrics)
    result.to_csv(OUT/'97_reverse_X_code_metrics.csv',index=False)
    manifest=dict(script=Path(__file__).name, input='REAL Script 73 + original DPQ as in 96',
        train='2005-2006 only', heldout=list(PERIODS),cohorts={p:len(g) for p,g in groups.items()},
        source_X=codec.X, quantization_decimals=3,
        decimal_width=width,code_dimension=7,k=args.k,
        note='Candidate code is always encoded from observed TRAIN X; no test X in decoding. Test X used only for evaluation.',
        codecs=list(codec.METHODS),
        comparisons=['prior','ridge_total','ridge_pair','ridge_items','score_match','item_hamming','cryptanalytic_ngrams'],
        limits=['A single score cannot uniquely identify X if several X correspond to that score.',
                'Candidate top-k oracle uses test X only for evaluation.',
                'Quantile bins are exploratory surrogate outcomes, not clinical thresholds.',
                'Zero holdout exact-code hits can reflect unseen continuous states; compare coarse and distance metrics.',
                'No causality or patient-level clinical claims.',
                'Survey-weighted point metrics; no complex-design uncertainty or multiplicity corrections.',
                'Code length can be extremely large; coding cannot manufacture information absent from Y.'])
    (OUT/'97_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    summary=result.pivot_table(index=['period','model'],columns='output',values='mean_weighted_R2',aggfunc='first')
    report=['# Script 97: Can Y recover a seven-biomarker encoded physiological state?',
        '', '## What this experiment establishes',
        '- Mathematical exactness of the X code is tested independently of statistical reverse inference.',
        '- Identical phenotypes with different X codes prove a unique Y-to-X inverse does not exist for those observed cases.',
        '- Learned frequency and token-motif models rank TRAIN physiology codes using Y only; exact decode then returns a real X candidate.',
        '', '## Cohorts', '```json',json.dumps(manifest['cohorts'],indent=2),'```',
        '', '## Mean held-out weighted R2', '```',summary.to_string(), '```',
        '', '## Interpretation',
        'Candidate top-1 is an exactly decoded TRAIN participant code, not proof of matching the held-out patient.',
        'The top-k mean is a regression-style average, NOT the decoding of one encoded state.',
        'Any gains need comparison with ridge/prior and replication; no causal inference follows.']
    (OUT/'97_readout.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\nOUTPUTS: 97_joint_codec_feasibility.csv, 97_identifiability.csv, 97_reverse_X_code_metrics.csv, 97_manifest.json, 97_readout.md')
    print('COMPLETE. No frozen models changed.')


def self_test(codec):
    # Toy values for SOFTWARE CHECKS ONLY, not empirical physiology.
    for v in [(1,2,3,4,5,6,7), (0,0,0,0,0,0,0),(-1,0,2,-3,4,5,0)]:
        u=natrow(v)
        enc=codec.encode(u,'decimal_interleave',3,4096)
        dec=codec.decode(enc,'decimal_interleave',7,3)
        assert dec==u
        assert tuple(codec.signed(q) for q in dec)==v
    # Same Y, distinct X: impossible to decode both from Y alone.
    toy=pd.DataFrame({'PHQ9_TOTAL':[10,10,10],
      'SOMATIC_SCORE':[4,4,3],'COGAFF_SUM':[6,6,7],
      **{f'DPQ0{i}0':[0,0,0] for i in range(1,10)}})
    toy.loc[0,'DPQ010']=1
    toy.loc[1,'DPQ010']=1
    assert ykeys(toy,'total',codec.YITEMS)[0]==ykeys(toy,'total',codec.YITEMS)[2]
    assert (len(tokens([0]*9,2).split()) == 9+36)
    assert (len(tokens([0]*9,3).split()) == 9+36+84)
    print('PASS: Script 97 mathematical roundtrip, inverse-ambiguity, and motif construction checks (toy only).')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true',help='Codec and data-structure checks only; no NHANES claims.')
    parser.add_argument('--k',type=int,default=10,help='Top-k physiological codes returned per phenotype')
    parser.add_argument('--batch',type=int,default=200,help='Number of test queries processed together')
    parser.add_argument('--codec-audit-rows',type=int,default=128,help='Number of genuine sampled rows per period for complete-vector codecs')
    parser.add_argument('--max-bits',type=int,default=4096,help='Stop exact integer codes that exceed this feasibility bound')
    args=parser.parse_args()
    if args.k < 1 or args.batch < 1 or args.codec_audit_rows < 1 or args.max_bits < 256:
        parser.error('k,batch,codec-audit-rows must be positive and max-bits >= 256')
    run(args)
