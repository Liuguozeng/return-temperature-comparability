# Paired return-temperature comparisons

This directory contains paired campaign results, source/count maps and
implementation for comparing temporal alignment designs. The package supports
derived-result verification, original-source replay and figure rendering as
separate operations. Release version 1.0.0 is prepared, not published.

## Requirements

Python 3.12 with NumPy 2.3.5 and pandas 2.2.3 was used. A compatible local Python
with NumPy and pandas is required for derived verification. The runtime guard
uses the Unix resource and signal modules on macOS/Linux. Install rendering
dependencies separately for the figure commands below.
All relative commands below run from the extracted `comparability/` directory. Replace
`python3` with the path to the local interpreter when necessary.

## Offline Verification

```sh
python3 -B -m unittest discover -s tests -v
python3 -B offline_check.py --output validation
```

`validation` must not exist. The second command disables network access and
blocks Python file access outside the extracted package, selected interpreter
and operating-system resources. It verifies SHA256_MANIFEST.json, runs unit
tests, and reconstructs all saved Cartesian error values, the daily-disjoint
windows, provider/focal/pair statistics, saved node-draw identities and XAI
leave-one-system estimates. Its success status is PASS_PACKAGED_DERIVED_VERIFICATION_ONLY and its
raw replay field remains NOT_RUN_EXTERNAL_ARCHIVES_NOT_INCLUDED. Runtime and
memory checks stop before the registered 3,600 s / 4 GiB limits.

The nested FULL_CARTESIAN_VALUES.zip has one NPZ per qualified pair-season and
contains both endpoint families and all five durations, including every
asynchronous diagonal. NODE_BOOTSTRAP_DRAWS.zip retains all 5,000 physical-node
multiplicity draws per case. NODE_BOOTSTRAP_ESTIMATES.csv.gz retains all metric
replicates. Verification reconstructs all point estimates and intervals from
these retained data and checks eight literal repeated-node replicates per
metric. The unequal-degree unit test distinguishes
the registered estimator from pair pooling. Identity/exclusion and start-count
tables permit independent reconstruction without relying on the overview.

## Optional Raw Replay

Obtain the original archives lawfully from the cited providers. Make a local
manifest from input_manifest.example.json; preserve the
archive hashes and point the first two entries to your local files. Relative
input paths resolve relative to the manifest's directory. Readers and the
selected valid-day carrier are included and do not need historical source
directories. Do not publish a manifest containing private local paths.

```sh
python3 -B replay_raw.py --input-manifest input_manifest.example.json --analysis-spec analysis_spec.json --output raw_replay
```

Without the two external archives this command exits 3, writes
NOT_RUN_MISSING_EXTERNAL_INPUTS, and does NOT report PASS. With matching archives
it runs the exact AST-preserved scientific functions under the locked bounds;
it omits figures and historical log writing. Successful computation alone is
reported as RAW_COMPUTATION_COMPLETE_NOT_YET_INDEPENDENTLY_VERIFIED, not as
raw-replay validation. Compare regenerated derived values against this package
before making any fresh raw-replay verification claim.

## Estimator And Boundaries

The order is windows/combinations within pair-season, seasons within pair,
equal eligible counterparts within focal system, then equal focal systems
within provider. Node multiplicities weight both counterpart and focal nodes,
exclude original-node self pairs even after repetition, and preserve reuse of
both endpoints. Shared dates do not require identical active hours. Primary
medians use frozen hourly observation resolution; positive heat weights are
unchanged. Daily-median date-disjoint results target a separate estimand.
XAI has five physical systems, point estimates and all omissions, no CIs.

SCOPE_MAP.json lists every included file and its role. DATA_DICTIONARY.json
documents table fields and array layouts. PROVENANCE.json records source,
transformation and scientific-function identities.
SOURCE_PROVENANCE.json, ATTRIBUTION.md and licenses/ preserve original dataset
terms and attribution. Source records retain the available acquisition dates.
License scopes and dataset-specific terms are documented in the accompanying
license matrices and attribution records.

SHA256_MANIFEST.json hashes every packaged file except itself (explicit
self-exclusion). results/HASH_MANIFEST.json is a package-relative adapter for
the exact independent verifier; it excludes itself and SHA256_MANIFEST.json to
avoid a circular hash dependency. Archive-level checksums and execution
receipts are recorded separately from numerical inputs.

## Current Manuscript Evidence Figure

The visual-only renderer `render_comparability.py` uses the unchanged packaged source
values. It writes main Figure 2, two-page comparison-error Figure S1,
two-page strict-reversal Figure S2, four provider-specific PNG/PDF pages,
and source/frequency/facet maps into an isolated output directory. The supplied diagnostics are
included under `diagnostics/`. Python with the locked NumPy/pandas and
Matplotlib dependencies is required for this optional rendering layer.

```sh
python3 -B render_comparability.py test --results results --output figure_contracts.json
python3 -B render_comparability.py figures --results results --output reproduced_figures
```

The execution receipt identifies the figure outputs separately from
derived-result verification and original-source replay.

Exact rendering dependencies are recorded in requirements-figures.txt. The selected Times New Roman font must be lawfully installed as a system font; no proprietary font is redistributed. Font absence is reported by the renderer, not silently certified as equivalent output.

## Current manuscript interface

Temporal alignment shapes return-temperature comparability between heating systems

This package renders current main Figure 2 and the four-page SI diagnostics.
Stable input identities retain their original paths; display numbering is explicit
in MANUSCRIPT_INTERFACE.json.

`python3 -B render_comparability.py figures --results results --output NEW_EMPTY_OUTPUT`

Retained provenance identifies the source evidence; MANUSCRIPT_INTERFACE.json
gives its current display numbering.

The exact source basename is `PAIR_SEASON_SUMMARY.csv`; the similarly suffixed
date-disjoint file is a separate estimand and is not a diagnostic input. Figure S1
uses 0.5 degree Celsius error bins; Figure S2 uses 5 percentage-point reversal
bins, both with 0.5 degree Celsius horizontal bins for PreDist. A single neutral
blue-gray frequency scale within each metric reports natural `log(1+count)`,
not probability density. All facets use common axes within provider/metric and
include the full tails. Blue `#2879AD` and pink `#C97E9C` identify design headers.
XAI4HEAT displays all 14 exact pair-season values per facet, without smoothing,
jitter or translucent point overlap. Small symmetric axis padding prevents
marker clipping without changing data. True zero-reference contrasts remain
in S1; undefined reference orderings remain in the source map but are excluded
from S2 with per-facet counts. Campaign ties and hierarchical rates are unchanged.
The count/source maps describe dependent pair-season observations. Field
definitions are in `diagnostics/DATA_DICTIONARY.json`.

```sh
python3 -B tests/test_diagnostic_facets.py
```

In Figure 2, the endpoint shapes apply to panel f only. Panel e uses circles
with endpoint-duration row labels. Focal-system timing effects average eligible
partners under each pair's shared-date reference. The paired effect intervals
come from the supplied paired estimates. The contract tests verify that
re-pairing preserves the signed mean but can improve, worsen or leave absolute
comparison error unchanged.
