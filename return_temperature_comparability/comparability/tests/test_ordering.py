"""Contracts for the displayed strict-reversal estimates."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_comparability import load_tables
from render_ordering import ordering_records, validate_records


class Ordering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_tables(Path(__file__).resolve().parents[1] / 'results')
        cls.rows = ordering_records(cls.tables)

    def test_forty_registered_points(self):
        self.assertEqual(len(self.rows), 40)
        validate_records(self.rows)

    def test_xai_bounds_are_descriptive(self):
        self.assertEqual({r['interval'] for r in self.rows if r['provider'] == 'XAI4HEAT'},
                         {'leave_one_system_range'})

    def test_missing_object_rejected(self):
        with self.assertRaises(ValueError):
            validate_records(self.rows[:-1])

    def test_wrong_interval_rejected(self):
        rows = copy.deepcopy(self.rows)
        next(r for r in rows if r['provider'] == 'XAI4HEAT')['interval'] = '95% CI'
        with self.assertRaises(ValueError):
            validate_records(rows)

    def test_wrong_rate_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['estimate'] = 1.2
        with self.assertRaises(ValueError):
            validate_records(rows)


if __name__ == '__main__':
    unittest.main()
