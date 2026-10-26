"""Run exactly six fixed logistic fits and the majority baseline on validation."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'

from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings
import zipfile

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / 'upstream/code'
sys.path.insert(0, str(OFFICIAL))
from common import load_inputs, parse_answer, read_jsonl
sys.path.insert(0, str(ROOT))
import majority
import submission


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def labels_and_stats(bank, split):
    gold = read_jsonl(OFFICIAL / 'data' / f'gsm8k_{split}.jsonl')
    assert [q['id'] for q in gold] == [q['id'] for q in bank]
    answers = submission.parsed_answers(bank)
    y = np.array([[a is not None and a == parse_answer(g['answer']) for a in aa]
                  for aa, g in zip(answers, gold)], dtype=int)
    assert y.any(axis=1).all()
    hard = np.array([len(set(a for a in aa if a is not None)) >= 2 for aa in answers])
    stats = {
        'questions': len(bank), 'candidates': 4 * len(bank),
        'parseable_candidates': sum(a is not None for aa in answers for a in aa),
        'parseable_candidate_rate': sum(a is not None for aa in answers for a in aa) / (4 * len(bank)),
        'correct_candidates_per_question': dict(sorted(Counter(map(int, y.sum(axis=1))).items())),
        'distinct_valid_answers_per_question': dict(sorted(Counter(len(set(a for a in aa if a is not None)) for aa in answers).items())),
        'questions_with_duplicate_valid_answers': sum(len([a for a in aa if a is not None]) > len(set(a for a in aa if a is not None)) for aa in answers),
        'hard_subset_questions': int(hard.sum()),
    }
    return y, hard, stats


def evaluate(name, method, checkpoint, labels, hard):
    predictions = ROOT / 'results/predictions' / f'{name}.json'
    score_file = ROOT / 'results/scores' / f'{name}.json'
    predict_cmd = [sys.executable, str(OFFICIAL / 'predict.py'), '--method', str(method),
                   '--split', 'validation', '--output', str(predictions)]
    if checkpoint is not None:
        predict_cmd += ['--checkpoint', str(checkpoint)]
    score_cmd = [sys.executable, str(OFFICIAL / 'score_predictions.py'), '--split', 'validation',
                 '--predictions', str(predictions), '--output', str(score_file)]
    started = time.perf_counter()
    pred = subprocess.run(predict_cmd, cwd=OFFICIAL, capture_output=True, text=True, check=True)
    predict_seconds = time.perf_counter() - started
    started = time.perf_counter()
    score_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(score_cmd, cwd=OFFICIAL, capture_output=True, text=True, check=True)
    scoring_seconds = time.perf_counter() - started
    choices = np.array(json.loads(predictions.read_text())['selected'])
    correctness = labels[np.arange(len(labels)), choices].astype(bool)
    official = json.loads(score_file.read_text())
    assert int(correctness.sum()) == official['correct']
    file_bytes = checkpoint.stat().st_size if checkpoint is not None else 0
    payload_bytes = sum(z.file_size for z in zipfile.ZipFile(checkpoint).infolist()) if checkpoint else 0
    result = {
        'name': name, 'correct': official['correct'], 'n': official['n'], 'accuracy': official['accuracy'],
        'hard_correct': int(correctness[hard].sum()), 'hard_n': int(hard.sum()),
        'hard_accuracy': float(100 * correctness[hard].mean()),
        'fresh_prediction_process_seconds': predict_seconds,
        'fresh_scoring_process_seconds': scoring_seconds,
        'total_process_seconds': predict_seconds + scoring_seconds,
        'runner_prediction_seconds': json.loads(pred.stdout)['prediction_seconds'],
        'asset_file_bytes': file_bytes, 'asset_archive_uncompressed_bytes': payload_bytes,
        'counted_asset_bytes': max(file_bytes, payload_bytes),
        'prediction_command': predict_cmd, 'score_command': score_cmd,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
    }
    assert result['counted_asset_bytes'] <= 16 * 2**20
    return result, choices, correctness


def main():
    results = ROOT / 'results'
    assert not (results / 'summary.json').exists(), 'Round already completed; do not rerun silently'
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    started = time.perf_counter()
    manifest = json.loads((OFFICIAL / 'PACKAGE_MANIFEST.json').read_text())
    for relative, expected in manifest['files'].items():
        p = ROOT / 'upstream' / relative
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, relative
    check = subprocess.run([sys.executable, str(OFFICIAL / 'check_data.py')], cwd=OFFICIAL,
                           capture_output=True, text=True, check=True)
    write(results / 'data-check.json', {'package_hashes_passed': len(manifest['files']),
                                      'official_check': json.loads(check.stdout)})
    train_bank, train_arrays = load_inputs('train')
    val_bank, val_arrays = load_inputs('validation')
    train_y, _, train_stats = labels_and_stats(train_bank, 'train')
    val_y, hard, val_stats = labels_and_stats(val_bank, 'validation')
    write(results / 'data-stats.json', {'train': train_stats, 'validation': val_stats})
    versions = {p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl', 'narwhals', 'cloudpickle')}
    metadata = {
        'python': sys.version, 'platform': platform.platform(), 'hardware': 'Apple M5; 10 CPU cores; 16 GiB RAM',
        'dependencies': versions, 'threads': {k: os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')},
        'training_threadpools': threadpool_info(), 'source': json.loads((ROOT / 'SOURCE.json').read_text()),
        'seed': 17, 'C_values': [0.1, 1.0, 10.0], 'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-6,
        'formal_test_scored': False,
    }
    write(results / 'environment.json', metadata)
    rows = []
    predictions = {}
    correct = {}
    row, predictions['majority'], correct['majority'] = evaluate('majority', ROOT / 'majority.py', None, val_y, hard)
    row.update({'feature_set': 'majority', 'C': None, 'fit_seconds': 0.0})
    rows.append(row)
    fit_count = 0
    for kind in ('embedding', 'context'):
        train_x = submission.make_features(train_bank, train_arrays, kind).reshape(4 * len(train_bank), -1)
        val_x = submission.make_features(val_bank, val_arrays, kind)
        scaler = StandardScaler().fit(train_x)
        train_z = scaler.transform(train_x)
        for regularization in (0.1, 1.0, 10.0):
            name = f'{kind}-C{regularization:g}'
            model = LogisticRegression(C=regularization, solver='lbfgs', max_iter=2000, tol=1e-6, random_state=17)
            tick = time.perf_counter()
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                model.fit(train_z, train_y.reshape(-1))
            fit_seconds = time.perf_counter() - tick
            fit_count += 1
            if any(issubclass(w.category, ConvergenceWarning) for w in observed):
                raise RuntimeError(f'{name} did not converge within the fixed budget')
            checkpoint = artifacts / f'{name}.npz'
            np.savez(checkpoint, feature_set=np.array(kind), mean=scaler.mean_, scale=scaler.scale_,
                     coef=model.coef_[0], intercept=model.intercept_[0])
            expected = model.decision_function(scaler.transform(val_x.reshape(4 * len(val_bank), -1))).reshape(-1, 4)
            exported = submission.decision_scores(val_bank, val_arrays, checkpoint)
            np.testing.assert_allclose(exported, expected, atol=1e-9, rtol=1e-10)
            assert submission.choose(val_bank, expected) == submission.predict(val_bank, val_arrays, checkpoint)
            row, predictions[name], correct[name] = evaluate(name, ROOT / 'submission.py', checkpoint, val_y, hard)
            train_choices = submission.choose(train_bank, model.decision_function(train_z).reshape(-1, 4))
            row.update({'feature_set': kind, 'C': regularization, 'feature_dimensions': train_x.shape[1],
                        'fit_seconds': fit_seconds, 'iterations': int(model.n_iter_[0]),
                        'train_accuracy': float(100 * train_y[np.arange(len(train_bank)), train_choices].mean()),
                        'export_max_absolute_score_difference': float(np.max(np.abs(exported - expected))),
                        'warnings': [str(w.message) for w in observed]})
            rows.append(row)
    assert fit_count == 6
    ranking = lambda r: (-r['correct'], {'majority': 0, 'embedding': 1, 'context': 2}[r['feature_set']], r['C'] or 0)
    chosen = sorted(rows, key=ranking)[0]
    winners = {kind: sorted([r for r in rows if r['feature_set'] == kind], key=ranking)[0]
               for kind in ('majority', 'embedding', 'context')}
    emb = winners['embedding']['name']
    ctx = winners['context']['name']
    categories = [
        ('context_regression', correct[emb] & ~correct[ctx]),
        ('context_recovery', ~correct[emb] & correct[ctx]),
        ('majority_only_correct', correct['majority'] & ~correct[emb] & ~correct[ctx]),
        ('all_three_wrong', ~correct['majority'] & ~correct[emb] & ~correct[ctx]),
    ]
    gold = read_jsonl(OFFICIAL / 'data/gsm8k_validation.jsonl')
    cases, seen = [], set()
    for category, mask in categories:
        allowance = 3 if category in ('context_regression', 'context_recovery') else 2
        for n in np.flatnonzero(mask)[:allowance]:
            n = int(n)
            if n in seen:
                continue
            seen.add(n)
            cases.append({'category': category, 'id': val_bank[n]['id'], 'question': val_bank[n]['question'],
                          'reference_answer': parse_answer(gold[n]['answer']),
                          'parsed_candidate_answers': submission.parsed_answers([val_bank[n]])[0],
                          'selected': {k: int(predictions[v['name']][n]) for k, v in winners.items()},
                          'candidates': [c['text'] for c in val_bank[n]['candidates']]})
    write(results / 'error-cases.json', cases)
    paired = {}
    for name, a, b in [('embedding_vs_majority', 'majority', emb), ('context_vs_embedding', emb, ctx)]:
        paired[name] = {'recovered': int((~correct[a] & correct[b]).sum()),
                        'regressed': int((correct[a] & ~correct[b]).sum()),
                        'both_wrong': int((~correct[a] & ~correct[b]).sum())}
    summary = {
        'rows': rows, 'family_winners': winners, 'selected': chosen, 'paired': paired,
        'fits_completed': fit_count, 'total_fit_seconds': sum(r['fit_seconds'] for r in rows),
        'round_wall_seconds': time.perf_counter() - started,
        'formal_test_scored': False, 'resource_qualification': 'Local Apple M5 validation measurement only; reference Xeon not tested',
    }
    write(results / 'summary.json', summary)
    print(json.dumps({'status': 'complete', 'fits': fit_count,
                      'family_accuracy': {k: v['accuracy'] for k, v in winners.items()},
                      'selected': chosen['name'], 'wall_seconds': summary['round_wall_seconds']}))


if __name__ == '__main__':
    main()
