import unittest
import numpy as np
import pandas as pd
from experiments.runtime import summarize_runtime


class RuntimeTests(unittest.TestCase):
    def test_masks_are_not_independent_fits(self):
        df = pd.DataFrame([dict(split='random', model='ftt', seed=s, mask_rate=m,
                               fit_s=float(s), infer_ms_per_row=float(s)/10)
                           for s in [1, 3] for m in [0., .1, .2, .3]])
        result = summarize_runtime(df).iloc[0]
        self.assertEqual(result.n_seeds, 2)
        self.assertAlmostEqual(result.fit_s_mean, 2.)
        self.assertAlmostEqual(result.fit_s_std, np.sqrt(2))
        one = summarize_runtime(df[df.seed == 1]).iloc[0]
        self.assertTrue(pd.isna(one.fit_s_std))
        with self.assertRaises(ValueError):
            summarize_runtime(pd.concat([df,df]))


if __name__ == '__main__':
    unittest.main()
