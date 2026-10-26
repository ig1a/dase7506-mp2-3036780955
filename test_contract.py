"""Small, discriminative interface tests; no public test-bank scoring."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'upstream/code'))
from common import parse_answer, validate_choices
sys.path.insert(0, str(ROOT))
import majority
import submission


def bank(texts):
    return [{'id': 'synthetic', 'question': 'Synthetic contract check',
             'candidates': [{'index': i, 'text': t} for i, t in enumerate(texts)]}]


class Contract(unittest.TestCase):
    def setUp(self):
        self.arrays = {'embeddings': np.random.default_rng(17).normal(size=(1, 4, 1024)).astype('float32')}

    def test_fraction_and_decimal_votes(self):
        b = bank(['#### 1/2', '#### 0.5', 'no numeric ending', '#### 2'])
        self.assertEqual(parse_answer(b[0]['candidates'][0]['text']), '1/2')
        self.assertEqual(majority.predict(b, self.arrays, None), [0])

    def test_missing_answer_mask_and_empty_fallback(self):
        b = bank(['no ending', '#### 2', '#### 3', '#### 4'])
        self.assertEqual(submission.choose(b, [[100, 1, 2, 3]]), [3])
        empty = bank(['a', 'b', 'c', 'd'])
        self.assertEqual(majority.predict(empty, self.arrays, None), [0])
        self.assertEqual(submission.choose(empty, [[1, 2, 3, 4]]), [0])

    def test_exact_tie_rule(self):
        b = bank(['#### 3', '#### 2', '#### 2', '#### 3'])
        self.assertEqual(majority.predict(b, self.arrays, None), [0])
        self.assertEqual(submission.choose(b, [[0, 0, 0, 0]]), [0])

    def test_multiple_correct_candidates(self):
        b = bank(['#### 1/2', '#### 0.5', '#### 2', '#### 4'])
        correct = [parse_answer(c['text']) == '1/2' for c in b[0]['candidates']]
        self.assertEqual(correct, [True, True, False, False])
        for selected in ([0], [1]):
            validate_choices(b, ['synthetic'], selected)
            self.assertTrue(correct[selected[0]])

    def test_candidate_permutation_and_numpy_checkpoint(self):
        b = bank(['#### 1/2', '#### 0.5', '#### 2', 'unparsed'])
        perm = np.array([2, 0, 3, 1])
        shuffled = copy.deepcopy(b)
        shuffled[0]['candidates'] = [dict(b[0]['candidates'][i], index=j) for j, i in enumerate(perm)]
        rearranged = {'embeddings': self.arrays['embeddings'][:, perm]}
        for kind, dimensions in [('embedding', 1024), ('context', 1028)]:
            features = submission.make_features(b, self.arrays, kind)
            np.testing.assert_allclose(submission.make_features(shuffled, rearranged, kind), features[:, perm], atol=1e-12)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'model.npz'
                np.savez(path, feature_set=kind, mean=np.zeros(dimensions), scale=np.ones(dimensions),
                         coef=np.random.default_rng(29).normal(size=dimensions), intercept=np.array(0.1))
                scores = submission.decision_scores(b, self.arrays, path)
                chosen = submission.predict(b, self.arrays, path)
                moved = submission.predict(shuffled, rearranged, path)
                np.testing.assert_allclose(submission.decision_scores(shuffled, rearranged, path), scores[:, perm], atol=1e-12)
                self.assertEqual(chosen, [int(perm[moved[0]])])
                validate_choices(b, ['synthetic'], chosen)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    print(json.dumps({'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
                      'passed': result.wasSuccessful()}))
    sys.exit(0 if result.wasSuccessful() else 1)
