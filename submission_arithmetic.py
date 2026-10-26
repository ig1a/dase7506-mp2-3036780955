"""NumPy-only ranker with conservative arithmetic-evidence features."""
from pathlib import Path
import numpy as np
import submission as base
from arithmetic_evidence import make_arithmetic_features

parsed_answers = base.parsed_answers
choose = base.choose


def make_features(bank, arrays, feature_set='arithmetic'):
    if feature_set != 'arithmetic':
        raise ValueError('Expected arithmetic feature set')
    return np.concatenate((base.make_features(bank, arrays, 'context'), make_arithmetic_features(bank)), axis=-1)


def decision_scores(bank, arrays, checkpoint):
    with np.load(Path(checkpoint), allow_pickle=False) as model:
        features = make_features(bank, arrays, str(model['feature_set'].item()))
        return ((features-model['mean']) / model['scale']) @ model['coef'] + model['intercept'].item()


def predict(bank, arrays, checkpoint):
    return choose(bank, decision_scores(bank, arrays, checkpoint))
