# Temporal alignment shapes return-temperature comparability between heating systems

Code and derived evidence for seasonal return-temperature records and paired heating-system comparisons.

## Reproduce

Clone this repository, then download `return_temperature_comparability_data.zip` from
[release 1.0.4](https://github.com/Liuguozeng/return-temperature-comparability/releases/tag/v1.0.4) and extract it in the repository
root. Its members complete the existing `return_temperature_comparability/` directory.

```bash
git clone https://github.com/Liuguozeng/return-temperature-comparability.git
cd return-temperature-comparability
unzip /path/to/return_temperature_comparability_data.zip
cd return_temperature_comparability
python3 reproduce.py verify --package-root .
python3 reproduce.py figures --package-root . --output ../generated_figures
```

Install the requirements listed inside the package before running the figure
or numerical workflows. The [package README](return_temperature_comparability/README.md) describes the
available commands and input requirements. Figure files are generated outputs.

The release also provides a code ZIP for use without Git. Extract both ZIPs
into the same empty directory; their disjoint members form one package root.

## License and Sources

Original analysis and plotting code is available under the MIT license.
Source data, upstream material and derived data retain the terms documented in
`LICENSE_MATRIX.csv` in the data archive and the package attribution
records. The MIT license does not replace those source-specific terms.
Original source archives are obtained from the cited official records.
