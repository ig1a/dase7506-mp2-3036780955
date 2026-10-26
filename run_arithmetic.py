"""One pre-specified linear pairwise fit; train/validation only, no test access."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import hashlib
import json
import sys
import time
import warnings
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / 'upstream/code'
sys.path.insert(0, str(OFFICIAL))
from common import load_inputs, read_jsonl, parse_answer
sys.path.insert(0, str(ROOT))
import submission_arithmetic as submission
import submission as baseline_submission

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def labels(bank, split):
    assert split in ('train', 'validation')
    gold = read_jsonl(OFFICIAL / 'data' / f'gsm8k_{split}.jsonl')
    assert [b['id'] for b in bank] == [g['id'] for g in gold]
    answers = submission.parsed_answers(bank)
    valid = np.array([[a is not None for a in aa] for aa in answers])
    y = np.array([[a is not None and a == parse_answer(g['answer']) for a in aa] for aa, g in zip(answers, gold)])
    hard = np.array([len(set(a for a in aa if a is not None)) >= 2 for aa in answers])
    assert y.any(axis=1).all()
    return y, valid, hard, answers

def accuracy(y, hard, choices):
    ok = y[np.arange(len(y)), choices]
    return {'correct': int(ok.sum()), 'n': len(y), 'accuracy': float(100 * ok.mean()),
            'hard_correct': int(ok[hard].sum()), 'hard_n': int(hard.sum()),
            'hard_accuracy': float(100 * ok[hard].mean())}, ok

def main():
    out = ROOT / 'results/arithmetic-20261026'
    assert out.exists()
    assert not (out / 'summary.json').exists()
    diagnostic = json.loads((out/'diagnostic.json').read_text())
    audit = json.loads((out/'manual-audit.json').read_text())
    assert diagnostic['quantitative_gate_passed'] and audit['gate_passed']
    assert digest(ROOT/'arithmetic_evidence.py') == diagnostic['parser_sha256']
    checkpoint = ROOT / 'artifacts/arithmetic-C001-20261026.npz'
    assert not checkpoint.exists()
    original = ROOT / 'artifacts/pairwise-C001-20261026.npz'
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in (original, ROOT / 'submission.py', ROOT / 'run_round.py', ROOT / 'arithmetic_evidence.py', ROOT / 'submission_arithmetic.py')}
    plan = {'feature_set': 'arithmetic', 'C': 0.01, 'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-6,
            'fit_intercept': False, 'seed': 17, 'fits': 1, 'threads': 4,
            'pairs': 'Every valid correct/incorrect pair within each training question; no positive-positive pairs',
            'weighting': 'Each informative question total weight 1; each pair represented by +/- difference, each with weight 1/(2 * pairs_in_question)',
            'objective': 'sum_q mean_pairs softplus(-(s_correct-s_wrong)) + ||w||^2/(2*C)',
            'scaler': 'StandardScaler fit on all train candidates, same as previous round',
            'selection': 'One configuration only; compare complete validation and hard subset; no test scoring',
            'frozen_baseline': frozen}
    write(out / 'FIT_PLAN.json', plan)
    b, a = load_inputs('train'); y, valid, hard, _ = labels(b, 'train')
    x = submission.make_features(b, a, 'arithmetic')
    scaler = StandardScaler().fit(x.reshape(-1, x.shape[-1]))
    with np.load(original) as old:
        np.testing.assert_allclose(scaler.mean_[:1028], old['mean'], atol=1e-12)
        np.testing.assert_allclose(scaler.scale_[:1028], old['scale'], atol=1e-12)
    z = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    pairs, weights, informative = [], [], 0
    for q in range(len(b)):
        pos = np.flatnonzero(y[q]); neg = np.flatnonzero(valid[q] & ~y[q])
        count = len(pos) * len(neg)
        if not count:
            continue
        informative += 1
        for i in pos:
            for j in neg:
                pairs.append(z[q, i] - z[q, j]); weights.append(1 / (2 * count))
    d = np.asarray(pairs); weight = np.asarray(weights)
    assert len(d) == 6516 and informative == 2114
    np.testing.assert_allclose(2 * weight.sum(), informative)
    train_x = np.concatenate([d, -d]); train_y = np.r_[np.ones(len(d)), np.zeros(len(d))]
    sample_weight = np.r_[weight, weight]
    model = LogisticRegression(C=0.01, fit_intercept=False, solver='lbfgs', max_iter=2000, tol=1e-6, random_state=17)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(train_x, train_y, sample_weight=sample_weight)
    fit_seconds = time.perf_counter() - started
    assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), 'Non-converged; do not retry/search'
    np.savez(checkpoint, feature_set=np.array('arithmetic'), mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_[0], intercept=np.array(0.0))
    train_metrics, _ = accuracy(y, hard, submission.predict(b, a, checkpoint))
    vb, va = load_inputs('validation'); vy, vv, vh, answers = labels(vb, 'validation')
    vx = submission.make_features(vb, va, 'arithmetic')
    expected = model.decision_function(scaler.transform(vx.reshape(-1, vx.shape[-1]))).reshape(-1, 4)
    exported = submission.decision_scores(vb, va, checkpoint)
    np.testing.assert_allclose(expected, exported, atol=1e-10, rtol=1e-10)
    # Check equivalent pair-margin expression and candidate permutation with the actual fitted asset.
    np.testing.assert_allclose(d[:50] @ model.coef_[0], model.decision_function(d[:50]), atol=1e-10)
    perm = [2, 0, 3, 1]
    shuffled = [dict(row, candidates=[dict(row['candidates'][i], index=j) for j,i in enumerate(perm)]) for row in vb[:20]]
    perm_scores = submission.decision_scores(shuffled, {'embeddings': va['embeddings'][:20, perm]}, checkpoint)
    np.testing.assert_allclose(perm_scores, exported[:20, perm], atol=1e-10, rtol=1e-10)
    predictions = ROOT / 'results/predictions/arithmetic-C001-20261026.json'
    commands = [
        [sys.executable, '-B', str(OFFICIAL/'predict.py'), '--method', str(ROOT/'submission_arithmetic.py'), '--checkpoint', str(checkpoint), '--split', 'validation', '--output', str(predictions)],
        [sys.executable, '-B', str(OFFICIAL/'score_predictions.py'), '--split', 'validation', '--predictions', str(predictions), '--output', str(out/'official-score.json')],
    ]
    processes = []
    for command in commands:
        tick = time.perf_counter()
        result = subprocess.run(command, cwd=OFFICIAL, text=True, capture_output=True, check=True)
        processes.append({'command': command, 'seconds': time.perf_counter()-tick, 'stdout': result.stdout, 'stderr': result.stderr})
    choices = np.array(json.loads(predictions.read_text())['selected'])
    metrics, ok = accuracy(vy, vh, choices)
    old_choices = np.array(baseline_submission.predict(vb, va, original)); old_metrics, old_ok = accuracy(vy, vh, old_choices)
    assert json.loads((out/'official-score.json').read_text())['correct'] == metrics['correct']
    assert metrics['correct'] == int(vy[np.arange(len(vy)), np.array(submission.choose(vb, expected))].sum())
    by_pattern = {}; wrong_modes = {'old': 0, 'new': 0}
    for n, aa in enumerate(answers):
        if not vh[n]: continue
        counts = Counter(v for v in aa if v is not None)
        pattern = ','.join(map(str, sorted(counts.values(), reverse=True)))
        row = by_pattern.setdefault(pattern, {'n': 0, 'old_correct': 0, 'new_correct': 0})
        row['n'] += 1; row['old_correct'] += int(old_ok[n]); row['new_correct'] += int(ok[n])
        modes = [v for v,c in counts.items() if c==max(counts.values())]
        for name, choice, correct in [('old', old_choices[n], old_ok[n]), ('new', choices[n], ok[n])]:
            wrong_modes[name] += int(not correct and len(modes)==1 and aa[choice]==modes[0])
    cases=[]
    for category, mask in [('recovered', ~old_ok & ok), ('regressed', old_ok & ~ok)]:
        for n in np.flatnonzero(mask)[:5]:
            cases.append({'category':category, 'id':vb[n]['id'], 'question':vb[n]['question'], 'answers':answers[n],
                          'correct_indices':np.flatnonzero(vy[n]).tolist(), 'old_choice':int(old_choices[n]), 'new_choice':int(choices[n]),
                          'candidates':[c['text'] for c in vb[n]['candidates']]})
    payload = sum(z.file_size for z in zipfile.ZipFile(checkpoint).infolist())
    assert frozen == {p:digest(ROOT/p) for p in frozen}
    summary={'plan':plan, 'train':train_metrics, 'baseline_validation':old_metrics, 'pairwise_validation':metrics,
             'paired':{'recovered':int((~old_ok & ok).sum()), 'regressed':int((old_ok & ~ok).sum()), 'both_wrong':int((~old_ok & ~ok).sum())},
             'wrong_unique_plurality_chosen':wrong_modes, 'by_answer_pattern':by_pattern,
             'fit_seconds':fit_seconds, 'iterations':int(model.n_iter_[0]), 'pair_count':len(d), 'informative_questions':informative,
             'fresh_prediction_and_scoring_seconds':sum(p['seconds'] for p in processes), 'processes':processes,
             'asset_bytes':checkpoint.stat().st_size, 'asset_uncompressed_bytes':payload, 'checkpoint_sha256':digest(checkpoint),
             'numpy_export_max_error':float(np.max(np.abs(expected-exported))), 'permutation_check':'passed', 'baseline_unchanged':True,
             'hardware':'Apple M5; not reference Xeon qualification', 'test_accessed':False,
             'dependencies':{'numpy':np.__version__, 'sklearn':__import__('sklearn').__version__}}
    summary['arithmetic_feature_coefficients_standardized'] = dict(zip(__import__('arithmetic_evidence').FEATURE_NAMES, model.coef_[0][-4:].tolist()))
    summary['diagnostic'] = diagnostic
    summary['selection'] = 'retain baseline on tie/loss; no extra fits'
    write(out/'summary.json',summary); write(out/'error-cases.json',cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
