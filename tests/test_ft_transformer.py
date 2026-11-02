"""Pranay: train-only preprocessing, missing inputs, and deterministic training."""
import unittest
import numpy as np
import pandas as pd
from src.models.ft_transformer import FTTransformer
from sklearn.metrics import average_precision_score


class FTTransformerTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({'n': [1., 2., np.nan, 4., 5., 6., 7., 8.],
                                   'c': ['a', 'unknown', 'b', 'a', 'b', 'a', 'b', 'a']})
        self.val = pd.DataFrame({'n': [1000., np.nan, 3., 2.],
                                 'c': ['validation_only', None, 'unknown', 'a']})
        self.y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    def make(self):
        return FTTransformer(['n'], ['c'], seed=42, n_blocks=1, d_block=16,
                             max_epochs=2, patience=1, batch_size=4, num_threads=1)

    def test_training_missingness_and_reproducibility(self):
        a = self.make().fit(self.train, self.y, self.val, np.array([0, 1, 1, 0]))
        self.assertEqual(a.imputer_.statistics_[0], 5.)
        self.assertNotIn('validation_only', a.vocabs_['c'])
        self.assertGreater(a.vocabs_['c']['unknown'], 0)
        self.assertEqual(a._transform(self.val)[1][:, 0].tolist()[:2], [0, 0])
        missing = pd.DataFrame({'n': [np.nan, np.nan], 'c': [None, 'new']})
        p = a.predict_proba(missing)
        self.assertEqual(p.shape, (2,))
        self.assertTrue(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
        self.assertEqual(len(a.predict_proba(missing.iloc[:0])), 0)
        np.testing.assert_array_equal(a.model_.cat_embeddings.embeddings[0].weight[0].detach().numpy(), 0)
        b = self.make().fit(self.train, self.y, self.val, np.array([0, 1, 1, 0]))
        np.testing.assert_array_equal(p, b.predict_proba(missing))
        self.assertAlmostEqual(a.best_val_ap_, max(h['val_ap'] for h in a.history_))
        self.assertAlmostEqual(a.best_val_ap_, average_precision_score([0, 1, 1, 0], a.predict_proba(self.val)))

    def test_invalid_usage(self):
        with self.assertRaises(RuntimeError):
            self.make().predict_proba(self.val)
        with self.assertRaises(ValueError):
            self.make().fit(self.train, self.y)
        with self.assertRaises(ValueError):
            FTTransformer(['n'], ['c'], typo=1)
        with self.assertRaises(ValueError):
            FTTransformer(['n'], ['c'], d_block=17)


if __name__ == '__main__':
    unittest.main()
