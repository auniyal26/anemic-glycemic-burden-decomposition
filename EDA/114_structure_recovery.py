#!/usr/bin/env python3
"""Exploratory simulation benchmark; run manually from the repository.
Same input, requested 5D/7D budgets, train-only scaling/adapters/decoders.
AJIVE may reduce joint rank: actual dimensions are reported separately. PCA first two PCs are a variance
baseline, not PCA's claim to identify shared factors. Refit stability is
simulation bootstrap stability, NOT survey or temporal validation.
No biological identification, clinical utility or novelty is established here.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, ast, hashlib, importlib.util, inspect, json, subprocess, sys, tempfile, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ('shared_low_noise', 'private_dominant', 'discordant', 'no_sharing')
METHODS = ('HG5', 'PCA5', 'JIVE5', 'AJIVE5_adaptive', 'HG7', 'PCA7', 'JIVE7', 'AJIVE7_adaptive')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def cca_functions(path):
    # Script72 has top-level analysis: extract ONLY these pure functions.
    names = {'wnorm', 'weighted_mean_sd', 'wcov', 'invsqrt',
             'fit_weighted_cca', 'transform_block'}
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise RuntimeError('Script72 helper definitions changed')
    ns = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns

def affine(x, y):
    return np.linalg.lstsq(np.c_[np.ones(len(x)), x], y, rcond=1e-12)[0]

def predict(x, coef):
    return np.c_[np.ones(len(x)), x] @ coef

def basis(x):
    u, s, _ = np.linalg.svd(x-x.mean(0), full_matrices=False)
    return u[:, s > (s[0]*1e-10 if len(s) else 0)]

def stability(a, b):
    if a.shape[1] == 0 or a.shape[1] != b.shape[1]:
        return float('nan')
    qa, qb = basis(a), basis(b)
    if qa.shape[1] != a.shape[1] or qb.shape[1] != b.shape[1]:
        return float('nan')
    return float(np.linalg.svd(qa.T @ qb, compute_uv=False).min()**2)

def generate(rng, n, scenario, rotations):
    s = rng.normal(size=(n, 2)); d = rng.normal(size=(n, 2))
    ah = rng.normal(size=(n, 2)); ag = rng.normal(size=(n, 1))
    if scenario == 'no_sharing':
        hpair = rng.normal(size=(n, 2)); gpair = rng.normal(size=(n, 2))
    else:
        delta = .6 if scenario == 'discordant' else 0.
        hpair = s + delta*d; gpair = s - delta*d
    private = 4. if scenario == 'private_dominant' else 1.
    h = np.c_[hpair, private*ah] @ rotations[0]
    g = np.c_[gpair, private*ag] @ rotations[1]
    x = np.c_[h, g] + rng.normal(scale=.2, size=(n, 7))
    targets = {'shared': s, 'H_private': ah, 'G_private': ag}
    if scenario == 'discordant': targets['discordant'] = d
    # In no_sharing, s is independent negative-control truth.
    return x, targets

def ajive_scores(x, budget, native):
    am = native.ajive_module()
    original_check, original_svd = am.reconsider_joint_components, am.svd_wrapper
    # Local compatibility correction only: repeated removal is idempotent.
    # Preserve package threshold logic (including its per-block break).
    source = inspect.getsource(original_check)
    if 'to_keep.remove(j)' not in source:
        raise RuntimeError('AJIVE source changed: review compatibility correction')
    ns = dict(am.__dict__)
    exec(compile(source.replace('to_keep.remove(j)', 'to_keep.discard(j)'),
                 '<114 AJIVE idempotent-removal correction>', 'exec'), ns)
    am.reconsider_joint_components = ns['reconsider_joint_components']
    am.svd_wrapper = native.exact_svd
    try:
        fit = am.AJIVE(init_signal_ranks={'H':3, 'G':2}, joint_rank=2,
            indiv_ranks={'H':2 if budget == 5 else 3, 'G':1 if budget == 5 else 2},
            center=False, reconsider_joint_components=True, n_jobs=1, store_full=True)
        fit.fit({'H':x[:,:4], 'G':x[:,4:]})
        r = fit.results_dict()
        common = r['common']['scores']
        q = np.asarray(common) if common is not None else np.empty((len(x),0))
        pieces = [q]; individual = {}
        for name, block in [('H', x[:,:4]), ('G', x[:,4:])]:
            actual = fit.blocks[name].joint.full_
            if actual is None: actual = np.zeros_like(block)
            expected = q @ (q.T @ block)
            if np.max(abs(actual-expected)) > 1e-8:
                raise RuntimeError('AJIVE retained-projector check failed')
            part = r[name]['individual']['scores']
            individual[name + '_private'] = np.asarray(part) if part is not None else np.empty((len(x),0))
            if part is not None: pieces.append(np.asarray(part))
        return np.column_stack(pieces), q, individual
    finally:
        am.reconsider_joint_components, am.svd_wrapper = original_check, original_svd


def fit_method(method, raw, helpers, b, native, rscript, timeout):
    budget = 7 if '7' in method else 5
    scaler = StandardScaler().fit(raw); x = scaler.transform(raw)
    extra = {}
    if method.startswith('HG'):
        f = helpers['fit_weighted_cca'](x[:, :4], x[:, 4:], np.ones(len(x)))
        frame, _, _ = helpers['transform_block'](x[:, :4], x[:, 4:], f, 2)
        z = frame[b.Z_COLS].to_numpy()
        selected = b.select_z5_from_physiology(z, x, np.ones(len(x))) if budget == 5 else list(range(7))
        scores = z[:, selected]; shared = z[:, :2]
        # Only evaluate coordinates actually retained in this representation.
        extra = {label:z[:, [j for j in indices if j in selected]]
                 for label, indices in [('discordant', [2,3]), ('H_private', [4,5]), ('G_private', [6])]}
    elif method.startswith('PCA'):
        scores = b.WeightedPCA(budget).fit(x).transform(x); shared = scores[:, :2]
    elif method.startswith('AJIVE'):
        scores, shared, extra = ajive_scores(x, budget, native)
    else:
        env = os.environ.copy(); env['R_HOME'] = str(native.ENV/'Lib/R')
        env['PATH'] = os.pathsep.join([str(native.ENV/'Library/bin'), str(native.ENV/'Lib/R/bin/x64'), env['PATH']])
        with tempfile.TemporaryDirectory(prefix='114_jive_', dir=ROOT/'EDA'/'Results') as temp:
            td = Path(temp); pd.DataFrame(x).to_csv(td/'x.csv', index=False)
            code = native.R_CODE.replace('for(k in c(5,7))', f'for(k in c({budget}))').replace('showProgress=FALSE', 'showProgress=TRUE')
            (td/'run.R').write_text(code, encoding='utf-8')
            subprocess.run([str(rscript), '--vanilla', str(td/'run.R'), str(td/'x.csv'), str(td)], env=env, check=True, timeout=timeout)
            scores = pd.read_csv(td/f'jive{budget}.csv').to_numpy(); shared = scores[:, :2]
            extra = {'H_private':scores[:, 2:4 if budget == 5 else 5],
                     'G_private':scores[:, 4 if budget == 5 else 5:]}
    if np.linalg.matrix_rank(scores) != scores.shape[1]:
        raise RuntimeError('Numerically dependent score columns; cannot claim a valid representation')
    if not method.startswith('AJIVE') and scores.shape[1] != budget:
        raise RuntimeError('Requested score budget not met')
    maps = {}
    for label, z in dict(all=scores, shared=shared, **extra).items():
        coef = affine(x, z)
        err = np.linalg.norm(predict(x, coef)-z)/max(np.linalg.norm(z-z.mean(0)), 1e-12)
        if err > 1e-7: raise RuntimeError(f'Affine adapter error {err}')
        maps[label] = coef
    print(f'  {method}: {scores.shape[1]} total, {shared.shape[1]} shared scores', flush=True)
    return lambda new: {label:predict(scaler.transform(new), cf) for label, cf in maps.items()}

def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--reps', type=int, default=20)
    pa.add_argument('--run-name', default='114_structure_recovery_v2')
    pa.add_argument('--n-train', type=int, default=300)
    pa.add_argument('--n-test', type=int, default=1000)
    pa.add_argument('--seed', type=int, default=114)
    pa.add_argument('--timeout', type=int, default=1200)
    pa.add_argument('--rscript', type=Path, default=Path(r'C:\Users\auniy\anaconda3\envs\physiodecomp\Lib\R\bin\x64\Rscript.exe'))
    args = pa.parse_args()
    if args.reps < 2 or min(args.n_train, args.n_test) < 30: pa.error('Use >=2 replicates and >=30 participants per split')
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        pa.error('run-name must contain only letters, digits, underscores or hyphens')
    out = ROOT/'EDA/Results'/args.run_name; out.mkdir(parents=True, exist_ok=True)
    if (out/'114_protocol.json').exists():
        raise FileExistsError('114 output already exists. Preserve/review it before starting another run.')
    paths = [ROOT/'EDA'/f for f in ('72_shared_private_decomposition_and_phenotype.py', '103_jive_ajive_benchmark.py', '112_finish_benchmarks.py')]
    protocol = dict(arguments=vars(args), scenarios=SCENARIOS, methods=METHODS,
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + [Path(__file__)]},
        limits=['Exploratory simulations; no clinical or biological claim',
                'Equal weights; requested 5D and 7D budgets; AJIVE joint-rank reductions reported, never padded',
                'AJIVE compatibility correction: local remove-to-discard only; installed package unchanged',
                'Compare recovery only within equal actual dimensions; report all rank reductions and failures',
                '7D total recovery is an information-preservation control, not superiority',
                'Named-component recovery and off-target recovery measure separation, not clinical interpretability',
                'HG rank remains fixed at two even in no-sharing; this does not test false-positive rank selection',
                'HG5 uses the existing target-free reconstruction selection',
                'All-score recovery measures information recoverability, not semantic separation',
                'Shared2 compares designated shared scores with first-two-PC baseline',
                'No-sharing shared target is an independent negative control',
                'Stability is simulation row-bootstrap, not real survey stability',
                'Recovery decoder learns synthetic truth on training rows only',
                'Replicate intervals are Monte Carlo descriptive intervals, no multiplicity-adjusted superiority test'])
    (out/'114_protocol.json').write_text(json.dumps(protocol, indent=2, default=str))
    helpers = cca_functions(paths[0]); b = load_module('source103_for114', paths[1]); native = load_module('source112_for114', paths[2])
    rows, failures = [], []; started = time.monotonic()
    total = len(SCENARIOS)*args.reps*len(METHODS); done = 0
    for scenario_id, scenario in enumerate(SCENARIOS):
        for rep in range(args.reps):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, scenario_id, rep]))
            rotations = [np.linalg.qr(rng.normal(size=(k,k)))[0] for k in (4,3)]
            train, truth = generate(rng, args.n_train, scenario, rotations)
            test, test_truth = generate(rng, args.n_test, scenario, rotations)
            boot = rng.integers(0, len(train), len(train))
            for method in METHODS:
                done += 1
                print(f'[{done}/{total}] {scenario}, replicate {rep+1}, {method}: fitting + bootstrap refit ({time.monotonic()-started:.0f}s elapsed)', flush=True)
                try:
                    transform = fit_method(method, train, helpers, b, native, args.rscript, args.timeout)
                    refit = fit_method(method, train[boot], helpers, b, native, args.rscript, args.timeout)
                    ztrain, ztest, zboot = transform(train), transform(test), refit(test)
                    pending = []
                    for scope in ztrain:
                        st = stability(ztest[scope], zboot[scope])
                        for target, y in truth.items():
                            yp = predict(ztest[scope], affine(ztrain[scope], y))
                            yt = test_truth[target]
                            r2 = 1-np.sum((yt-yp)**2)/np.sum((yt-yt.mean(0))**2)
                            pending.append(dict(scenario=scenario, replicate=rep, method=method, scope=scope, dimensions=ztest[scope].shape[1], bootstrap_dimensions=zboot[scope].shape[1], requested_budget=7 if '7' in method else 5, target=target, heldout_R2=float(r2), refit_min_cosine_squared=st))
                    rows.extend(pending)
                except (Exception,) as exc:
                    failures.append(dict(scenario=scenario, replicate=rep, method=method, error=traceback.format_exc()))
                    print(f'FAILED (recorded, not replaced): {exc}', flush=True)
                pd.DataFrame(rows).to_csv(out/'114_recovery.csv', index=False)
                pd.DataFrame(failures, columns=['scenario','replicate','method','error']).to_csv(out/'114_failures.csv', index=False)
    if rows:
        frame = pd.DataFrame(rows)
        summary = frame.groupby(['scenario','method','scope','dimensions','bootstrap_dimensions','target']).agg(successful_reps=('heldout_R2','count'), mean_R2=('heldout_R2','mean'), sd_R2=('heldout_R2','std'), mean_refit_stability=('refit_min_cosine_squared','mean')).reset_index()
        summary.to_csv(out/'114_summary.csv', index=False)
    print(f'COMPLETE: {len(failures)} failed fits. Inspect failures alongside recovery; no automatic winner declared. Results: {out}', flush=True)

if __name__ == '__main__':
    main()
