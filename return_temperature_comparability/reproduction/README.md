## Portable lawful-input replay

This workflow computes HEATWISE hourly-to-window and provider summaries,
and PreDist/XAI4HEAT/HEATWISE temporal sensitivities. Earlier development and
Aalborg provider summaries enter as identified inputs.

Use Python 3.9.6 with the versions in `reproduction/requirements.txt` for
source replay. Copy `reproduction/input_manifest.example.json` outside the
package and supply the carriers listed in `reproduction/INPUT_REQUIREMENTS.json`.

```sh
python validation/verify_reported_values.py
python validation/verify_presentation_tables.py
python validation/rerun_submission_results.py --input-manifest input_manifest.json --output-dir replay_output
```

The first two commands verify supplied results. The replay command uses an
isolated temporary directory and can save output outside the package and
input carriers. `--source-workspace` is an alternative locator for inputs;
choose it or `--input-manifest`. Missing required inputs return `NOT EXECUTED`
(exit 2); changed identities, missing outputs and unequal results fail.
`rerun_primary_entrypoint.py` computes the HEATWISE primary outputs and
combines the earlier-provider summaries. Temporal sensitivities use the
submission-results entry point. Implementation and dependency provenance
are in `reproduction/code` and `CODE_PROVENANCE.json`.

Numerical replay checks CSV values, discrete identities and categories.
Compression headers can vary while decompressed cells agree. Current
manuscript results are under `results/`; historical reference inputs carry
their own identities. Obtain raw carriers under the cited source terms.
