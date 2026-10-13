'Portable checks of additional SI tables against packaged frozen results.'
from pathlib import Path
import csv
import gzip
import math
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def table(path):
    with (gzip.open(path,'rt',newline='') if path.suffix=='.gz' else path.open(newline='')) as f:
        return list(csv.DictReader(f))

def keyed(rows, keys):
    out={}
    for r in rows:
        key=tuple(str(r[k]) for k in keys)
        if key in out:raise ValueError('duplicate table identity')
        out[key]=r
    return out

def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12):
        raise ValueError('table value differs from frozen reduction')

def verify_development_groups():
    record=json.loads((ROOT/'supporting_tables/S6_PROVENANCE.json').read_text())
    selection=record['organization_holdout']
    expected=['AH','AP','AR','AV','BA','BE','J','K']
    if selection['selected_identifiers']!=expected or selection['eligible_organizations']!=31:
        raise ValueError('fixed organization holdout differs')
    if record['period_groups']!={'discovery':109,'later_period_holdout':32,'unseen_organization_holdout':127}:
        raise ValueError('fixed development period groups differ')
    if selection['selection_specification_sha256']!='62cc8cefbdb9d7f1789203ba3a7ce577439eeb82194f347e0168ee7ee8537c97':
        raise ValueError('selection evidence identity differs')
    if 'salt' in selection or 'salt' in record:
        raise ValueError('unexpected holdout configuration field')
    return dict(heldout_organizations=len(expected),development_periods=sum(record['period_groups'].values()))

def return_temperature():
    raw=pd.read_csv(ROOT/'results/NONOVERLAP_WINDOWS.csv.gz')
    identity=['provider','system','heating_year','window_start','duration_days']
    if raw.duplicated(identity).any():raise ValueError('duplicate window')
    if not raw.groupby(identity[:-1]).duration_days.agg(set).map(lambda s:s=={7,14,30,60,90}).all():
        raise ValueError('missing common-start duration')
    period=raw.groupby(['provider','system','heating_year','duration_days']).absolute_error_c.mean()
    system=period.groupby(['provider','system','duration_days']).mean().rename('mae_c').reset_index()
    provider=system.groupby(['provider','duration_days']).mae_c.mean().reset_index()
    expected=keyed(provider.astype({'duration_days':str}).to_dict('records'),['provider','duration_days'])
    actual=keyed(table(ROOT/'supporting_tables/S8C_NONOVERLAP_PROVIDER_ABSOLUTE_MAE.csv'),['provider','duration_days'])
    if set(actual)!=set(expected):raise ValueError('provider table membership')
    for key in expected:close(actual[key]['mae_c'],expected[key]['mae_c'])
    wide=system[system.provider.eq('XAI4HEAT')].pivot(index='system',columns='duration_days',values='mae_c')
    units=keyed(table(ROOT/'supporting_tables/S8D_NONOVERLAP_XAI_UNIT_MAE_CONTRASTS.csv'),['system'])
    if {k[0] for k in units}!=set(wide.index):raise ValueError('XAI unit table membership')
    for name,r in wide.iterrows():
        for duration in [7,14,30,60,90]:close(units[(name,)][str(duration)],r[duration])
        for short in [7,14]:close(units[(name,)][f'contrast_{short}_minus_90_c'],r[short]-r[90])
    return dict(provider_duration_rows=len(actual),physical_units=len(units),windows=len(raw))


if __name__=='__main__':
    result=return_temperature()
    result.update(verify_development_groups())
    print(json.dumps(dict(status='PASS',**result)))
