import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import (campaign_starts, contrast_arrays, daily_complement,
                      endpoint, node_estimate, normalize_frozen_hourly,
                      pair_season_means, safe_functions, validate_spec)


class ScientificContracts(unittest.TestCase):
    def test_common_mode_cartesian_and_unchanged_marginals(self):
        a = np.array([11., 13., 15.])
        b = np.array([9., 11., 13.])
        r = contrast_arrays(a, b, 2.)
        self.assertEqual(r['error'].shape, (3, 3))
        np.testing.assert_array_equal(np.diag(r['error']), 0.)
        self.assertGreater(np.abs(r['error']).mean(), 0.)
        self.assertEqual(len(r['error'].ravel()), 9)
        self.assertEqual(np.abs(a - 13).mean(),
                         np.broadcast_to(np.abs(a - 13)[:, None], (3, 3)).mean())
        self.assertEqual(np.abs(b - 11).mean(),
                         np.broadcast_to(np.abs(b - 11)[None, :], (3, 3)).mean())

    def test_zero_reference_undefined_and_campaign_ties_separate(self):
        r = contrast_arrays(np.array([1., 2.]), np.array([1., 2.]), 0.)
        self.assertTrue(np.isnan(r['reversal']).all())
        r = contrast_arrays(np.array([-1., 0., 1.]), np.array([0.]), 1.)
        np.testing.assert_array_equal(r['reversal'].ravel(), [1., 0., 0.])
        np.testing.assert_array_equal(r['tie'].ravel(), [0., 1., 0.])

    def test_repairing_preserves_signed_mean_not_absolute_error(self):
        a = np.array([2., 6., 9.])
        b = np.array([1., 5., 12.])
        error = contrast_arrays(a, b, 0.75)['error']
        self.assertAlmostEqual(np.diag(error).mean(), error.mean())
        self.assertAlmostEqual(error.mean(), a.mean() - b.mean() - 0.75)

    def test_synchronization_can_improve_worsen_or_leave_mae_unchanged(self):
        examples = [([-1., 1.], [-1., 1.], 0., 1.),
                    ([-1., 1.], [1., -1.], 2., 1.),
                    ([3., 3.], [1., 1.], 2., 2.)]
        for a, b, synchronous, asynchronous in examples:
            with self.subTest(a=a, b=b):
                error = contrast_arrays(a, b, 0.)['absolute_error']
                self.assertEqual(np.diag(error).mean(), synchronous)
                self.assertEqual(error.mean(), asynchronous)

    def test_nonzero_signed_error_does_not_cancel_under_alignment(self):
        error = contrast_arrays([4., 6.], [1., 3.], 1.)['error']
        self.assertEqual(np.diag(error).mean(), 2.)
        self.assertEqual(error.mean(), 2.)

    def test_original_resolution_and_positive_heat_weights(self):
        values = np.array([0., 0., 0., 100.])
        self.assertEqual(endpoint(values, np.ones(4), 'median'), 0.)
        self.assertEqual(endpoint(values, np.array([1., -1., np.nan, 3.]),
                                  'heat_weighted'), 75.)
        self.assertTrue(np.isnan(endpoint(values, np.zeros(4), 'heat_weighted')))
        self.assertNotEqual(endpoint(values, np.ones(4), 'median'),
                            np.median([0., 100.]))

    def test_weekly_starts_common_to_all_durations_and_80_percent(self):
        dates = pd.date_range('2022-10-01', '2023-04-30', tz='UTC')
        starts, counts = campaign_starts(dates, 2023)
        self.assertEqual(starts[0], pd.Timestamp('2022-10-01', tz='UTC'))
        self.assertTrue(all((s - starts[0]).days % 7 == 0 for s in starts))
        self.assertTrue(all(s + pd.Timedelta(days=90) <=
                            pd.Timestamp('2023-05-01', tz='UTC') for s in starts))
        sparse = dates.difference(dates[:2])
        starts2, counts2 = campaign_starts(sparse, 2023)
        self.assertNotIn(starts[0], starts2)
        self.assertEqual(counts2[0]['count_7'], 5)
        self.assertEqual(counts[0]['count_90'], 90)

    def test_disjoint_envelope_not_duration_specific(self):
        dates = pd.date_range('2022-10-01', periods=170, tz='UTC')
        keep = daily_complement(dates, dates[7])
        self.assertEqual(keep.sum(), 80)
        self.assertFalse(keep[7:97].any())
        self.assertTrue(keep[:7].all())

    def test_node_hierarchy_not_pair_equal_and_duplicate_self_excluded(self):
        matrix = np.array([[np.nan, 0., 12.], [0., np.nan, np.nan],
                           [12., np.nan, np.nan]])
        self.assertEqual(node_estimate(matrix, np.ones(3)), 6.)
        # Multiplicity [2, 1, 0] creates two copies of A, never A-A pairs.
        self.assertEqual(node_estimate(matrix, np.array([2., 1., 0.])), 0.)
        self.assertTrue(np.isnan(node_estimate(matrix, np.array([3., 0., 0.]))))
        irregular = np.array([[np.nan, 0., 0., np.nan],
                              [0., np.nan, 12., np.nan],
                              [0., 12., np.nan, 24.],
                              [np.nan, np.nan, 24., np.nan]])
        self.assertAlmostEqual(node_estimate(irregular, np.ones(4)), 10.5)
        self.assertEqual(np.nanmean(irregular), 9.)
        multiplicity = np.array([2., 1., 1., 0.])
        self.assertAlmostEqual(node_estimate(irregular, multiplicity), 2.)

    def test_safe_ast_does_not_import_log_function_or_top_level_write(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'reader.py'
            p.write_text('from pathlib import Path\n'
                         'Path("forbidden").write_text("x")\n'
                         'def log_access():\n    raise RuntimeError("write")\n'
                         'def read():\n    return 4\n')
            namespace = safe_functions(p, ['read'], {})
            self.assertEqual(namespace['read'](), 4)
            self.assertNotIn('log_access', namespace)
            self.assertFalse((Path.cwd() / 'forbidden').exists())

    def test_spec_rejects_changed_threshold_or_seed(self):
        import json
        spec = json.loads((Path(__file__).parents[1] / 'analysis_spec.json').read_text())
        validate_spec(spec)
        spec['minimum_shared_dates'] = 149
        with self.assertRaises(ValueError):
            validate_spec(spec)

    def test_frozen_column_and_utc_adapter_changes_no_numeric_values(self):
        raw = pd.DataFrame({'hour':pd.to_datetime(['2022-10-01 04:00']),
                            'substation':['unit'], 'return_c':[35.], 'heat_kwh':[2.]})
        adapted = normalize_frozen_hourly(raw)
        self.assertEqual(adapted.system.iloc[0], 'unit')
        self.assertEqual(str(adapted.hour.dtype), 'datetime64[ns, UTC]')
        self.assertEqual(adapted.return_c.iloc[0], 35.)
        self.assertEqual(adapted.heat_kwh.iloc[0], 2.)

    def test_seasons_average_before_counterparts(self):
        seasons = pd.DataFrame({'system_i':['A','A','A','A'],
                                'system_j':['B','B','B','C'],
                                'heating_year':[1,2,3,1], 'mae':[0.,0.,0.,12.]})
        pair = pair_season_means(seasons,['mae'])
        self.assertEqual(pair.mae.tolist(), [0.,12.])
        focal_a = pair.mae.mean()
        self.assertEqual(focal_a,6.)
        self.assertNotEqual(focal_a,seasons.mae.mean())


if __name__ == '__main__':
    unittest.main()
