"""NumPy-only MP2 selector; reference labels are never opened here."""
from collections import Counter
from pathlib import Path

import numpy as np

from common import parse_answer


def parsed_answers(bank):
    return [[parse_answer(c['text']) for c in row['candidates']] for row in bank]


def make_features(bank, arrays, feature_set):
    raw = np.asarray(arrays['embeddings'], dtype=np.float64)
    if raw.shape != (len(bank), 4, 1024):
        raise ValueError('Expected [N, 4, 1024] embeddings')
    unit = raw / np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
    if feature_set == 'embedding':
        return unit
    if feature_set != 'context':
        raise ValueError('Unknown feature set')
    answers = parsed_answers(bank)
    numeric = np.zeros((len(bank), 4, 4), dtype=np.float64)
    similarity = np.einsum('nid,njd->nij', unit, unit)
    for n, row in enumerate(bank):
        counts = Counter(a for a in answers[n] if a is not None)
        for i, candidate in enumerate(row['candidates']):
            others = [j for j in range(4) if j != i]
            numeric[n, i] = (
                counts[answers[n][i]] / 4 if answers[n][i] is not None else 0,
                np.log1p(len(candidate['text'])),
                float(similarity[n, i, others].mean()),
                float(similarity[n, i, others].max()),
            )
    return np.concatenate((unit, numeric), axis=-1)


def choose(bank, scores):
    scores = np.array(scores, dtype=np.float64, copy=True)
    if scores.shape != (len(bank), 4) or not np.isfinite(scores).all():
        raise ValueError('Expected finite scores with shape [N, 4]')
    for n, answers in enumerate(parsed_answers(bank)):
        valid = np.array([a is not None for a in answers])
        if valid.any():
            scores[n, ~valid] = -np.inf
        else:
            scores[n] = 0
    # NumPy argmax selects the lowest index on exact ties, including all-invalid.
    return [int(i) for i in scores.argmax(axis=1)]


def decision_scores(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('A trained NumPy checkpoint is required')
    with np.load(Path(checkpoint), allow_pickle=False) as model:
        features = make_features(bank, arrays, str(model['feature_set'].item()))
        return ((features - model['mean']) / model['scale']) @ model['coef'] + model['intercept'].item()


def predict(bank, arrays, checkpoint):
    return choose(bank, decision_scores(bank, arrays, checkpoint))
