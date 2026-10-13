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
    active = power_kw.gt(0) | power_kw.isna() & flow.gt(0)
    time = pd.to_datetime(raw['timestamp'], errors='coerce', utc=True)
    valid = active & time.notna() & raw['p_net_return_temperature'].between(0, 130) & raw['outdoor_temperature'].between(-50, 60)
    frame = pd.DataFrame({'hour': time.loc[valid].dt.floor('h'), 'return_c': raw.loc[valid, 'p_net_return_temperature']})
    hourly = frame.groupby('hour', as_index=False, observed=True).agg(return_c=('return_c', 'median'))
    hourly['provider'] = 'PreDist'
    hourly['system'] = f'{manufacturer}:{Path(name).stem}'
    return hourly

def load_xai_hourly() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with ZipFile(XAI) as archive:
        members = sorted((name for name in archive.namelist() if 'scada_' in name and name.endswith('_processed.csv')))
        for name in members:
            with archive.open(name) as handle:
                raw = pd.read_csv(handle)
            required = ['datetime', 't_amb', 't_ret_prim', 'delta_e']
            if any((column not in raw for column in required)):
                raise RuntimeError(f'XAI4HEAT schema mismatch in {name}')
            for column in required[1:]:
                raw[column] = pd.to_numeric(raw[column], errors='coerce')
            time = pd.to_datetime(raw['datetime'], errors='coerce', utc=True)
            valid = time.notna() & raw['delta_e'].gt(0) & raw['t_ret_prim'].between(0, 130) & raw['t_amb'].between(-50, 60)
            frame = pd.DataFrame({'hour': time.loc[valid].dt.floor('h'), 'return_c': raw.loc[valid, 't_ret_prim']})
            frame = frame.groupby('hour', as_index=False, observed=True).agg(return_c=('return_c', 'median'))
            frame['provider'] = 'XAI4HEAT'
            frame['system'] = Path(name).stem.replace('xai4heat_scada_', '').replace('_processed', '')
            frames.append(frame)
    if len(frames) != 5:
        raise RuntimeError(f'Expected five XAI4HEAT systems, found {len(frames)}')
    return pd.concat(frames, ignore_index=True)

def heating_year(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, utc=True)
    return parsed.dt.year + parsed.dt.month.ge(10).astype(int)

def build_daily_medians(hourly: pd.DataFrame) -> pd.DataFrame:
    local = hourly.copy()
    local['hour'] = pd.to_datetime(local['hour'], errors='raise', utc=True)
    local['date'] = local['hour'].dt.floor('D')
    local = local.loc[local['date'].dt.month.isin([10, 11, 12, 1, 2, 3, 4])].copy()
    local['heating_year'] = heating_year(local['date']).astype(int)
    day = local.groupby(['provider', 'system', 'heating_year', 'date'], observed=True).agg(daily_median_return_c=('return_c', 'median'), valid_hours=('hour', 'nunique')).reset_index()
    day = day.loc[day['valid_hours'].ge(4)].copy()
    periods = day.groupby(['provider', 'system', 'heating_year'], observed=True)['date'].nunique().rename('valid_days').reset_index()
    eligible = periods.loc[periods['valid_days'].ge(150), ['provider', 'system', 'heating_year']]
    day = day.merge(eligible, on=['provider', 'system', 'heating_year'], validate='many_to_one')
    return day.sort_values(['provider', 'system', 'heating_year', 'date']).reset_index(drop=True)
