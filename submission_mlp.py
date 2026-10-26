"""Small shared nonlinear ranker. NumPy inference; no reference access."""
import numpy as np
from submission import make_features, choose


def network(z, params):
    shape = z.shape[:-1]
    flat = z.reshape(-1, z.shape[-1])
    hidden = np.tanh(flat @ params['w1'] + params['b1'])
    return (hidden @ params['w2']).reshape(shape)


def decision_scores(bank, arrays, checkpoint):
    with np.load(checkpoint, allow_pickle=False) as model:
        x = make_features(bank, arrays, 'context')
        z = ((x - model['mean']) / model['scale']).astype(np.float32)
        return network(z, model)


def predict(bank, arrays, checkpoint):
    return choose(bank, decision_scores(bank, arrays, checkpoint))
