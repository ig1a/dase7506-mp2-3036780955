"""Train-only coverage and selection headroom, before any validation access."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root, args.out
    out.mkdir(exist_ok=True, parents=True)
    sys.path.insert(0, str(root/'upstream/code'))
    sys.path.insert(0, str(root))
    from common import load_inputs
    from run_pairwise_c001 import labels, digest
    import submission
    from arithmetic_evidence import inspect_text, FEATURE_NAMES
    started = time.perf_counter()
    bank, arrays = load_inputs('train')
    y, valid, hard, _ = labels(bank, 'train')
    checks = [[inspect_text(c['text']) for c in row['candidates']] for row in bank]
    n = np.array([[len(cs) for cs in row] for row in checks])
    bad = np.array([[sum(c['inconsistent'] for c in cs) for cs in row] for row in checks])
    covered, flagged = n > 0, bad > 0
    features = np.stack((np.log1p(n), np.log1p(bad), bad/np.maximum(n, 1), 1-covered.astype(float)), axis=-1)
    baseline = root/'artifacts/pairwise-C001-20261026.npz'
    choices = np.array(submission.predict(bank, arrays, baseline))
    idx = np.arange(len(bank))
    correct = y[idx, choices]
    has_unflagged_correct = (y & ~flagged).any(axis=1)
    headroom = ~correct & flagged[idx, choices] & has_unflagged_correct
    # This is an oracle opportunity count, not an achieved accuracy improvement.
    risk = correct & flagged[idx, choices] & (valid & ~y & ~flagged).any(axis=1)
    variation = np.ptp(features, axis=1).max(axis=1) > 0
    wrong = valid & ~y
    stats = {
        'train_questions': len(bank), 'train_candidates': int(y.size),
        'checked_candidate_count': int(covered.sum()), 'checked_candidate_fraction': float(covered.mean()),
        'checked_equations': int(n.sum()), 'inconsistent_equations': int(bad.sum()),
        'flagged_candidates': int(flagged.sum()),
        'final_correct_candidates': int(y.sum()), 'final_correct_flagged': int((flagged & y).sum()),
        'final_wrong_valid_candidates': int(wrong.sum()), 'final_wrong_flagged': int((flagged & wrong).sum()),
        'flag_rate_given_correct_final': float(flagged[y].mean()),
        'flag_rate_given_wrong_final': float(flagged[wrong].mean()),
        'wrong_final_fraction_among_flagged_valid': float(wrong[flagged & valid].mean()),
        'hard_questions_with_feature_variation': int((variation & hard).sum()),
        'baseline_hard_correct': int((correct & hard).sum()),
        'oracle_correction_opportunities': int(headroom.sum()),
        'correct_choices_flagged_with_unflagged_wrong_alternative': int(risk.sum()),
        'diagnostic_seconds': time.perf_counter()-started,
        'baseline_sha256': digest(baseline),
        'parser_sha256': digest(Path(__import__('arithmetic_evidence').__file__)),
        'feature_names': FEATURE_NAMES,
        'test_accessed': False, 'validation_accessed': False,
    }
    gate = {
        'coverage_ge_20_percent': stats['checked_candidate_fraction'] >= .2,
        'variation_ge_50_hard_questions': stats['hard_questions_with_feature_variation'] >= 50,
        'wrong_more_flagged_than_correct': stats['flag_rate_given_wrong_final'] > stats['flag_rate_given_correct_final'],
        'oracle_opportunities_ge_10': stats['oracle_correction_opportunities'] >= 10,
    }
    stats['quantitative_gate'] = gate
    stats['quantitative_gate_passed'] = all(gate.values())
    rng = np.random.default_rng(17)
    audit = []
    for category, mask in [('correct_final', flagged & y), ('wrong_final', flagged & wrong)]:
        positions = np.argwhere(mask)
        positions = positions[rng.permutation(len(positions))[:10]]
        for q, i in positions:
            audit.append({'category': category, 'question_index': int(q), 'candidate_index': int(i),
                          'id': bank[q]['id'], 'question': bank[q]['question'],
                          'text': bank[q]['candidates'][i]['text'], 'checks': checks[q][i]})
    examples = []
    for category, mask in [('opportunity', headroom), ('risk', risk)]:
        for q in np.flatnonzero(mask)[:5]:
            examples.append({'category': category, 'id': bank[q]['id'], 'question': bank[q]['question'],
                             'choice': int(choices[q]), 'correct_indices': np.flatnonzero(y[q]).tolist(),
                             'candidates': bank[q]['candidates'], 'checks': checks[q]})
    for name, value in [('diagnostic.json', stats), ('manual-audit-sample.json', audit), ('train-opportunity-risk.json', examples)]:
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__': main()
