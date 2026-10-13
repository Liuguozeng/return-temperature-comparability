"""Positive and negative contracts for presentation-derived diagnostic maps."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostic_facets import bin_edges, derive_maps, validate_maps
from render_comparability import load_tables


class DiagnosticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = load_tables(Path(__file__).resolve().parents[1]/"results")["PAIR_SEASON_SUMMARY.csv"]
        cls.mapping, cls.grid, cls.facets = derive_maps(cls.pairs)

    def test_complete_mapping(self):
        validate_maps(self.pairs, self.mapping, self.grid, self.facets)
        self.assertEqual(len(self.mapping), 165000)
        self.assertEqual(int((~self.mapping.plotted).sum()), 320)

    def test_edge_tail_and_zero(self):
        edges = bin_edges(1., .5)
        self.assertGreater(edges[-1], 1.)
        counts, _ = np.histogram([0, .5, 1.], edges)
        self.assertEqual(counts.sum(), 3)

    def test_exact_scatter_and_ties(self):
        mapping = self.mapping
        xai = self.facets.loc[self.facets.provider.eq("XAI4HEAT")]
        self.assertTrue(xai.plotted_rows.eq(14).all())
        for design in ("sync", "async"):
            for figure in ("S1", "S2"):
                values = mapping.loc[mapping.figure.eq(figure) & mapping.design.eq(design)].sort_values("source_csv_line")
                self.assertTrue(np.array_equal(values.campaign_ties, self.pairs[design+"_ties"]))

    def test_negative_missing_mapping(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_maps(self.pairs, self.mapping.iloc[1:], self.grid, self.facets)

    def test_negative_frequency_mutation(self):
        grid = self.grid.copy()
        grid.loc[0, "count"] += 1
        with self.assertRaisesRegex(ValueError, "Frequency count"):
            validate_maps(self.pairs, self.mapping, grid, self.facets)

    def test_negative_axis_mutation(self):
        facets = self.facets.copy()
        facets.loc[0, "x_max"] += .5
        with self.assertRaisesRegex(ValueError, "axis drift"):
            validate_maps(self.pairs, self.mapping, self.grid, facets)

    def test_negative_undefined_becomes_zero(self):
        mapping = self.mapping.copy()
        index = mapping.index[~mapping.plotted][0]
        mapping.loc[index, ["plotted", "display_value"]] = [True, 0.]
        with self.assertRaises(ValueError):
            validate_maps(self.pairs, mapping, self.grid, self.facets)

    def test_negative_exact_coordinate_mutation(self):
        mapping = self.mapping.copy()
        index = mapping.index[mapping.provider.eq("XAI4HEAT")][0]
        mapping.loc[index, "absolute_reference_contrast_c"] += .0001
        with self.assertRaisesRegex(ValueError, "coordinate drift"):
            validate_maps(self.pairs, mapping, self.grid, self.facets)

    def test_negative_tie_denominator_mutation(self):
        mapping = self.mapping.copy()
        mapping.loc[0, "campaign_ties"] += 1
        with self.assertRaisesRegex(ValueError, "denominator drift"):
            validate_maps(self.pairs, mapping, self.grid, self.facets)


if __name__ == "__main__":
    unittest.main(verbosity=2)
