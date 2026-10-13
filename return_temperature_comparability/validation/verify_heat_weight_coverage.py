'Check supplied qualification counts without reading restricted source records.'
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = ['CAMPAIGN_HEAT_WEIGHT_COVERAGE.csv', 'REFERENCE_HEAT_WEIGHT_COVERAGE.csv']
checks = []
summary = {}
for filename in files:
    with (ROOT / 'supporting_tables' / filename).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    keys = []
    for r in rows:
        keys.append(tuple(r[x] for x in ['provider', 'system', 'heating_year'] +
                          (['duration_days', 'window_start'] if filename.startswith('CAMPAIGN') else [])))
        nh, ph, nd, pd = [int(r[x]) for x in ['temperature_hours', 'positive_weight_hours', 'temperature_days', 'positive_weight_days']]
        checks.append(0 < ph <= nh and 0 < pd <= nd)
        checks.append(math.isfinite(float(r['total_weight'])) and float(r['total_weight']) > 0)
        checks.append(abs(float(r['positive_hour_fraction']) - ph / nh) <= 1e-14)
        checks.append(abs(float(r['positive_day_fraction']) - pd / nd) <= 1e-14)
    checks.append(len(set(keys)) == len(keys))
    summary[filename] = {'rows': len(rows), 'providers': sorted({r['provider'] for r in rows})}
checks.append(summary[files[0]]['rows'] == 25071)
checks.append(summary[files[1]]['rows'] == 213)
print(json.dumps({'scope': 'supplied heat-weight qualification counts and fractions, not raw-input replay',
                  'summary': summary, 'failed_checks': len(checks) - sum(checks)}, indent=2))
raise SystemExit(0 if all(checks) else 1)
