# Return temperature comparability

Release version: 1.0.0.

This package contains the derived results, source values and implementation
for evaluating seasonal return-temperature records and between-system
comparisons. Extract the code and data ZIPs into the same empty directory;
their disjoint members form one `return_temperature_comparability/` root.
Install the dependencies listed in the accompanying requirements files.

## Run

```bash
python3 reproduce.py verify --package-root .
python3 reproduce.py figures --package-root . --output NEW_EMPTY_OUTPUT
```

`verify` checks the complete package member list and file identities. `figures`
renders data figures from their source values into a new output directory.
The six supplementary diagnostic pages and two combined PDFs show empirical
distributions, reference-contrast strata and exact XAI4HEAT paired values.
They are generated from `comparability/results/PAIR_SEASON_SUMMARY.csv`;
object mappings and stratum summaries accompany the output. Figure exports are
generated outputs rather than package inputs.

This entry point accepts the return-temperature release only. Numerical checks
of the supplied tables are separate:

```bash
python3 -B validation/verify_reported_values.py
python3 -B validation/verify_release_interfaces.py
python3 -B validation/verify_supporting_tables.py
```

In Figures 3 and 5, AAU diamonds denote single-system estimates. Solid whiskers
denote confidence intervals and dashed whiskers denote two-system descriptive
ranges. Figure source values and dictionaries accompany the plotting code.


## Inputs and provenance

Original-source replay uses separately obtained lawful archives and an explicit
input manifest. HEATWISE, temporal-sensitivity and paired PreDist/XAI4HEAT
workflows have separate input contracts in their subdirectories. Required
missing inputs are reported as `NOT_EXECUTED`.

Source terms and redistribution scope are recorded in `LICENSE_MATRIX.csv`
and the source-attribution records. The release contains permitted derived
values; raw-provider archives are obtained from their cited source records.

Development comparisons use the fixed organization and period groups recorded
in `supporting_tables/S6_PROVENANCE.json`. That record includes the eight
held-out identifiers, selection method and original evidence hashes.
