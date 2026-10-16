# Paired return-temperature comparisons

This directory contains paired campaign results, source/count maps and
implementation for comparing temporal alignment designs. The package supports
derived-result verification, original-source replay and figure rendering as
separate operations.

## Requirements

Python 3.12 with NumPy 2.3.5 and pandas 2.2.3 was used. A compatible local Python
with NumPy and pandas is required for derived verification. The runtime guard
uses the Unix resource and signal modules on macOS/Linux. Install rendering
dependencies separately for the figure commands below.
For dependencies installed in a separate local runtime directory, the offline
command accepts `--dependency /path/to/runtime`; it otherwise uses the selected
interpreter's installed packages.
All relative commands below run from the extracted `comparability/` directory. Replace
`python3` with the path to the local interpreter when necessary.

## Offline Verification

```sh
python3 -B -m unittest discover -s tests -v
python3 -B offline_check.py --output /absolute/new-validation
```

The validation directory must not exist. Choose a directory outside the package
to keep generated checks separate from input files. The second command disables network access and
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

The renderer `render_comparability.py` uses the packaged source values.
It writes the main comparison figure and three pages each for Figures S1
and S2: the PreDist empirical distribution, PreDist reference-contrast
strata and exact XAI4HEAT pairs. Source-object and stratum maps accompany
the six pages. The package-root `reproduce.py figures` command assigns
the manuscript figure numbers. Earlier frequency maps in `diagnostics/`
describe the same source objects in fixed grids. Python with NumPy/pandas and
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

The comparison renderer writes main Figure 2 and the six-page SI diagnostics.
The package-root `reproduce.py figures` command also writes strict-reversal
Figure 3 and the retained duration and reference-target figures.
Stable input identities retain their original paths; display numbering is explicit
in MANUSCRIPT_INTERFACE.json.

`python3 -B render_comparability.py figures --results results --output NEW_EMPTY_OUTPUT`

Retained provenance identifies the source evidence; MANUSCRIPT_INTERFACE.json
gives its current display numbering.

The exact source basename is `PAIR_SEASON_SUMMARY.csv`; the similarly suffixed
date-disjoint file is a separate estimand and is not a diagnostic input. Figure S1
uses comparison MAE; Figure S2 uses strict-reversal percentages. Each has
an empirical-distribution page and a reference-contrast-stratum page for
PreDist, followed by exact paired values for XAI4HEAT. Nonzero absolute reference-contrast
quartiles are defined from unique pair-seasons and stay fixed across durations
and designs; boxes show medians, quartiles and complete ranges. Blue
`#2879AD` and pink `#C97E9C` identify synchronous and asynchronous designs.
XAI4HEAT displays all 14 exact pair-season values per endpoint and duration.
Symmetric axis padding keeps low-reference markers clear of the axes without
moving their scientific coordinates. Zero-reference contrasts remain in S1;
undefined reference orderings are excluded from S2 and identified in the
source map. Campaign ties retain their defined-order denominator. Earlier
frequency maps use 0.5 degree Celsius reference/error bins and 5 percentage-point
reversal bins. Source maps describe dependent pair-season observations. Field
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
