"""Regression checks for descriptive comparison displays."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'comparability'))
from diagnostic_comparisons import reference_strata, validate_source


class ComparisonDisplays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / 'comparability/results/PAIR_SEASON_SUMMARY.csv')

    def test_complete_source(self):
        validate_source(self.data)

    def test_missing_object_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_source(self.data.iloc[1:])

    def test_duplicate_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_source(pd.concat([self.data, self.data.iloc[:1]]))

    def test_undefined_ordering_is_not_zero(self):
        altered = self.data.copy()
        zero = altered.zero_reference.astype(bool)
        altered.loc[zero, 'sync_reversal_rate'] = 0.
        with self.assertRaises(ValueError):
            validate_source(altered)

    def test_strata_use_unique_reference_objects(self):
        for endpoint in ['median', 'heat_weighted']:
            boundaries, labels, assign = reference_strata(self.data, 'PreDist', endpoint)
            repeated = pd.concat([self.data, self.data[self.data.duration_days.eq(7)]])
            actual, _, _ = reference_strata(repeated, 'PreDist', endpoint)
            np.testing.assert_array_equal(boundaries, actual)
            np.testing.assert_array_equal(assign(np.array([0., -2., 2.])),
                                          np.array([0, assign(np.array([2.]))[0], assign(np.array([2.]))[0]]))
            self.assertEqual(len(labels), len(boundaries) + 2)

    def test_changing_reference_identity_is_rejected(self):
        altered = self.data.copy()
        altered.loc[0, 'reference_contrast_c'] += .01
        with self.assertRaises(ValueError):
            reference_strata(altered, altered.loc[0, 'provider'], altered.loc[0, 'endpoint'])


if __name__ == '__main__':
    unittest.main()
