'Frozen reader functions only; source hashes in PROVENANCE.json.'
from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

def _hourly_from_predist_member(archive: ZipFile, name: str) -> pd.DataFrame:
    manufacturer = 'manufacturer_1' if name.startswith('manufacturer 1/') else 'manufacturer_2'
    needed = {'timestamp', 'outdoor_temperature', 'p_net_return_temperature', 'p_net_meter_heat_power', 'p_net_meter_flow'}
    with archive.open(name) as handle:
        raw = pd.read_csv(handle, sep=';', usecols=lambda value: value in needed, low_memory=False)
    for column in needed - {'timestamp'}:
        if column not in raw:
            raw[column] = np.nan
        raw[column] = pd.to_numeric(raw[column], errors='coerce')
    power_kw = raw['p_net_meter_heat_power'].copy()
    flow = raw['p_net_meter_flow'].copy()
    if manufacturer == 'manufacturer_2':
        power_kw /= 1000.0
        flow /= 1000.0
    active = power_kw.gt(0) | (power_kw.isna() & flow.gt(0))
    time = pd.to_datetime(raw['timestamp'], errors='coerce')
    valid = active & time.notna() & raw['p_net_return_temperature'].between(0, 130) & raw['outdoor_temperature'].between(-50, 60)
    frame = pd.DataFrame(
        {
            'datetime': time.loc[valid],
            'return_c': raw.loc[valid, 'p_net_return_temperature'],
            'outdoor_c': raw.loc[valid, 'outdoor_temperature'],
            'heat_kwh': power_kw.loc[valid].where(power_kw.loc[valid].gt(0)),
        }
    )
    frame['hour'] = frame['datetime'].dt.floor('h')
    hourly = frame.groupby('hour', as_index=False, observed=True).agg(return_c=('return_c', 'median'), outdoor_c=('outdoor_c', 'median'), heat_kwh=('heat_kwh', 'mean'))
    hourly['provider'] = 'PreDist'
    hourly['manufacturer'] = manufacturer
    hourly['substation'] = f"{manufacturer}:{Path(name).stem}"
    return hourly

def load_xai_hourly() -> pd.DataFrame:
    frames = []
    with ZipFile(XAI) as archive:
        members = sorted(name for name in archive.namelist() if 'scada_' in name and name.endswith('_processed.csv'))
        for name in members:
            with archive.open(name) as handle:
                raw = pd.read_csv(handle)
            required = ['datetime', 't_amb', 't_ret_prim', 'delta_e']
            if any(column not in raw for column in required):
                raise RuntimeError(f"XAI4HEAT schema mismatch in {name}")
            for column in required[1:]:
                raw[column] = pd.to_numeric(raw[column], errors='coerce')
            time = pd.to_datetime(raw['datetime'], errors='coerce')
            valid = time.notna() & raw['delta_e'].gt(0) & raw['t_ret_prim'].between(0, 130) & raw['t_amb'].between(-50, 60)
            local = pd.DataFrame({'hour': time.loc[valid].dt.floor('h'), 'return_c': raw.loc[valid, 't_ret_prim'], 'outdoor_c': raw.loc[valid, 't_amb'], 'heat_kwh': raw.loc[valid, 'delta_e']})
            local = local.groupby('hour', as_index=False, observed=True).agg(return_c=('return_c', 'median'), outdoor_c=('outdoor_c', 'median'), heat_kwh=('heat_kwh', 'sum'))
            local['provider'] = 'XAI4HEAT'
            local['manufacturer'] = 'not_applicable'
            local['substation'] = Path(name).stem.replace('xai4heat_scada_', '').replace('_processed', '')
            frames.append(local)
    if len(frames) != 5:
        raise RuntimeError('Expected five XAI4HEAT substations')
    return pd.concat(frames, ignore_index=True)
