"""Pair-season facets with source-row and frequency maps."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DURATIONS = [7, 14, 30, 60, 90]
COLUMNS = [("median", "sync"), ("median", "async"),
           ("heat_weighted", "sync"), ("heat_weighted", "async")]
BLUE = "#2879AD"
PINK = "#C97E9C"
GRAY = "#607F92"

from diagnostic_comparisons import CAPTIONS
S1_CAPTION = CAPTIONS['S1']
S2_CAPTION = CAPTIONS['S2']

def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def bin_edges(maximum, width):
    # Extend beyond the maximum so even a boundary-valued tail has a full bin.
    return np.arange(int(np.floor(maximum / width)) + 2, dtype=float) * width


def derive_maps(pairs):
    """Count display cells without changing scientific summaries or denominators."""
    maps, grids, facets = [], [], []
    for metric, figure in [("error", "S1"), ("reversal", "S2")]:
        for provider in ["PreDist", "XAI4HEAT"]:
            local = pairs.loc[pairs.provider.eq(provider)]
            xedges = bin_edges(local.reference_contrast_c.abs().max(), .5)
            ymax = max(local.sync_mae_c.max(), local.async_mae_c.max()) if metric == "error" else 100
            yedges = bin_edges(ymax, .5) if metric == "error" else np.arange(0, 105, 5.)
            for days in DURATIONS:
                for endpoint, design in COLUMNS:
                    rows = local.loc[local.endpoint.eq(endpoint) & local.duration_days.eq(days)]
                    x = rows.reference_contrast_c.abs().to_numpy()
                    y = rows[design + ("_mae_c" if metric == "error" else "_reversal_rate")].to_numpy()
                    if metric == "reversal":
                        y = y * 100
                    defined = np.isfinite(y)
                    require(np.isfinite(x).all(), "Nonfinite reference contrast")
                    require(np.isfinite(y).all() if metric == "error" else
                            np.array_equal(~defined, rows.zero_reference.to_numpy().astype(bool)),
                            "Invalid error or undefined-ordering identity")
                    facet_id = f"{figure}_{provider}_{endpoint}_{days}_{design}"
                    if provider == "PreDist":
                        counts, _, _ = np.histogram2d(x[defined], y[defined], bins=[xedges, yedges])
                        require(int(counts.sum()) == int(defined.sum()), "Count grid lost a tail")
                        for ix, iy in np.ndindex(counts.shape):
                            count = int(counts[ix, iy])
                            grids.append(dict(facet_id=facet_id, figure=figure, provider=provider,
                                endpoint=endpoint, duration_days=days, design=design, metric=metric,
                                x_bin=ix, y_bin=iy, x_lower_c=xedges[ix], x_upper_c=xedges[ix+1],
                                y_lower=yedges[iy], y_upper=yedges[iy+1], count=count,
                                log1p_count=float(np.log1p(count))))
                    for index, xv, yv, ok in zip(rows.index, x, y, defined):
                        row = rows.loc[index]
                        ix = min(int(np.searchsorted(xedges, xv, side="right") - 1), len(xedges)-2)
                        iy = min(int(np.searchsorted(yedges, yv, side="right") - 1), len(yedges)-2) if ok else None
                        maps.append(dict(source_csv="PAIR_SEASON_SUMMARY.csv", source_csv_line=int(index)+2,
                            pair_season_id=row.pair_season_id, figure=figure, facet_id=facet_id,
                            provider=provider, endpoint=endpoint, duration_days=days, design=design,
                            metric=metric, absolute_reference_contrast_c=float(xv), display_value=float(yv),
                            plotted=bool(ok), excluded_reason="" if ok else "undefined_zero_reference_ordering",
                            zero_reference=int(row.zero_reference), campaign_ties=int(row[design+"_ties"]),
                            order_defined=int(row[design+"_order_defined"]),
                            x_bin=ix if provider == "PreDist" else None,
                            y_bin=iy if provider == "PreDist" else None))
                    facets.append(dict(facet_id=facet_id, figure=figure, provider=provider,
                        endpoint=endpoint, duration_days=days, design=design, metric=metric,
                        source_rows=len(rows), plotted_rows=int(defined.sum()), excluded_rows=int((~defined).sum()),
                        zero_reference_rows=int(rows.zero_reference.sum()), x_min=0., x_max=float(xedges[-1]),
                        y_min=0., y_max=float(yedges[-1]), x_bin_width_c=.5,
                        axis_x_min=0. if provider == "PreDist" else -.02*float(xedges[-1]),
                        axis_x_max=float(xedges[-1])*(1. if provider == "PreDist" else 1.02),
                        axis_y_min=0. if provider == "PreDist" else -.04*float(yedges[-1]),
                        axis_y_max=float(np.ceil(yedges[-1])) if provider == "PreDist" else float(yedges[-1])*1.04,
                        y_bin_width=.5 if metric == "error" else 5.,
                        representation="frequency_grid" if provider == "PreDist" else "exact_scatter"))
    return pd.DataFrame(maps), pd.DataFrame(grids), pd.DataFrame(facets)


def validate_maps(pairs, mapping, grid, facets):
    require(len(mapping) == len(pairs)*4 and len(facets) == 80, "Incomplete source/facet coverage")
    require(not mapping.duplicated(["source_csv_line", "figure", "design"]).any(), "Duplicate source mapping")
    require(mapping.groupby("source_csv_line").size().eq(4).all(), "Missing source mappings")
    for figure, metric in [("S1", "_mae_c"), ("S2", "_reversal_rate")]:
        for design in ["sync", "async"]:
            rows = mapping.loc[mapping.figure.eq(figure) & mapping.design.eq(design)].sort_values("source_csv_line")
            require(np.array_equal(rows.source_csv_line.to_numpy(), pairs.index.to_numpy()+2), "Source line drift")
            require(np.array_equal(rows.absolute_reference_contrast_c.to_numpy(), pairs.reference_contrast_c.abs().to_numpy()), "Source coordinate drift")
            expected = pairs[design+metric].to_numpy() * (100 if figure == "S2" else 1)
            require(np.array_equal(rows.display_value.to_numpy(), expected, equal_nan=True), "Source value drift")
            for column, source in [("campaign_ties", design+"_ties"), ("order_defined", design+"_order_defined"),
                                   ("zero_reference", "zero_reference"), ("pair_season_id", "pair_season_id"),
                                   ("provider", "provider"), ("endpoint", "endpoint"), ("duration_days", "duration_days")]:
                require(np.array_equal(rows[column].to_numpy(), pairs[source].to_numpy()), "Source identity or denominator drift")
    for facet in facets.itertuples():
        mapped = mapping.loc[mapping.facet_id.eq(facet.facet_id)]
        require(len(mapped) == facet.source_rows and int(mapped.plotted.sum()) == facet.plotted_rows,
                "Source/facet count mismatch")
        if facet.provider == "PreDist":
            bins = grid.loc[grid.facet_id.eq(facet.facet_id)]
            require(int(bins["count"].sum()) == facet.plotted_rows, "Frequency count mismatch")
            actual = mapped.loc[mapped.plotted].groupby(["x_bin", "y_bin"]).size()
            expected = bins.set_index(["x_bin", "y_bin"])["count"]
            require(all(int(expected.loc[key]) == count for key, count in actual.items()) and
                    int(expected.sum()) == int(actual.sum()), "Source-to-bin mismatch")
            require(np.allclose(bins.log1p_count, np.log1p(bins["count"])), "Wrong log frequency")
    for (_, _), group in facets.groupby(["provider", "metric"]):
        require(group[["x_min", "x_max", "y_min", "y_max", "axis_x_min", "axis_x_max", "axis_y_min", "axis_y_max"]].nunique().eq(1).all(), "Facet axis drift")
    errors = mapping.loc[mapping.figure.eq("S1")]
    require(errors.plotted.all(), "Zero-reference or other error row lost")
    reversals = mapping.loc[mapping.figure.eq("S2")]
    require(np.array_equal(~reversals.plotted.to_numpy(), reversals.zero_reference.eq(1).to_numpy()),
            "Undefined ordering became a plotted zero")


def build_diagnostics(pairs, output, plt, save_figure):
    from diagnostic_comparisons import build
    return build(pairs, output, plt, save_figure)
