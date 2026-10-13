from __future__ import annotations
import math
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
DURATIONS = (7, 14, 30, 60, 90)
BOOTSTRAPS = 5000
SEED_RETURN_TEMPERATURE = 20260830

def nonoverlap_eligible_starts(group: pd.DataFrame, *, heating_year: int, minimum_valid_day_fraction: float=0.8, minimum_complement_days: int=60) -> list[pd.Timestamp]:
    dates = pd.to_datetime(group['date'], utc=True)
    start = pd.Timestamp(heating_year - 1, 10, 1, tz='UTC')
    latest = pd.Timestamp(heating_year, 5, 1, tz='UTC') - pd.Timedelta(days=90)
    starts: list[pd.Timestamp] = []
    while start <= latest:
        all_windows_valid = True
        for duration in DURATIONS:
            end = start + pd.Timedelta(days=duration)
            count = int((dates.ge(start) & dates.lt(end)).sum())
            if count < math.ceil(0.8 * duration):
                all_windows_valid = False
                break
        envelope_end = start + pd.Timedelta(days=90)
        complement = int((dates.lt(start) | dates.ge(envelope_end)).sum())
        if all_windows_valid and complement >= minimum_complement_days:
            starts.append(start)
        start += pd.Timedelta(days=7)
    return starts

def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator='\n', float_format='%.15g')

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

def load_predist_hourly() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with ZipFile(PREDIST) as archive:
        members = sorted((name for name in archive.namelist() if '/operational_data/substation_' in name and name.endswith('.csv')))
        for index, name in enumerate(members, start=1):
            frame = _hourly_from_predist_member(archive, name)
            if not frame.empty:
                frames.append(frame)
            if index % 20 == 0:
                print(f'PreDist members processed: {index}/{len(members)}', flush=True)
    if not frames:
        raise RuntimeError('No qualifying PreDist hourly records')
    return pd.concat(frames, ignore_index=True)

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

def load_heatwise_hourly() -> pd.DataFrame:
    frame = pd.read_csv(HEATWISE_HOURLY, usecols=['hour', 'return_c', 'provider', 'system'])
    frame = frame.loc[frame['provider'].isin(['AAU', 'Empa/NEST'])].copy()
    frame['hour'] = pd.to_datetime(frame['hour'], errors='raise', utc=True)
    return frame

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

def build_equal_day_windows(daily: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (provider, system, year), group in daily.groupby(['provider', 'system', 'heating_year'], observed=True, sort=True):
        dates = pd.to_datetime(group['date'], utc=True)
        reference = float(group['daily_median_return_c'].median())
        start = pd.Timestamp(int(year) - 1, 10, 1, tz='UTC')
        final = pd.Timestamp(int(year), 5, 1, tz='UTC')
        for duration in DURATIONS:
            local_start = start
            latest = final - pd.Timedelta(days=duration)
            while local_start <= latest:
                end = local_start + pd.Timedelta(days=duration)
                mask = dates.ge(local_start) & dates.lt(end)
                count = int(mask.sum())
                if count >= math.ceil(0.8 * duration):
                    endpoint = float(group.loc[mask, 'daily_median_return_c'].median())
                    records.append({'provider': provider, 'system': system, 'heating_year': int(year), 'duration_days': int(duration), 'window_start': local_start, 'valid_days': count, 'campaign_equal_day_median_c': endpoint, 'reference_equal_day_median_c': reference, 'absolute_error_c': abs(endpoint - reference)})
                local_start += pd.Timedelta(days=7)
    if not records:
        raise RuntimeError('No registered equal-day windows were eligible')
    return pd.DataFrame(records).sort_values(['provider', 'system', 'heating_year', 'duration_days', 'window_start']).reset_index(drop=True)

def system_duration_rows(windows: pd.DataFrame, value: str='absolute_error_c') -> pd.DataFrame:
    period = windows.groupby(['provider', 'system', 'heating_year', 'duration_days'], observed=True)[value].mean().rename('period_mean').reset_index()
    return period.groupby(['provider', 'system', 'duration_days'], observed=True)['period_mean'].mean().rename('estimate').reset_index()

def bootstrap_system_mean(values: pd.DataFrame, value: str, seed: int) -> np.ndarray:
    systems = sorted(values['system'].astype(str).unique())
    by_system = values.set_index('system')[value].to_dict()
    rng = np.random.default_rng(seed)
    draws = rng.choice(systems, size=(BOOTSTRAPS, len(systems)), replace=True)
    matrix = np.vectorize(by_system.__getitem__)(draws).astype(float)
    return matrix.mean(axis=1)

def summarize_equal_day(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    system_rows = system_duration_rows(windows)
    duration_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    for provider, provider_rows in system_rows.groupby('provider', observed=True, sort=True):
        system_count = int(provider_rows['system'].nunique())
        formal = provider in {'PreDist', 'XAI4HEAT'}
        for duration, group in provider_rows.groupby('duration_days', observed=True, sort=True):
            estimate = float(group['estimate'].mean())
            row: dict[str, object] = {'provider': provider, 'duration_days': int(duration), 'physical_systems': system_count, 'estimate_absolute_error_c': estimate, 'uncertainty_type': 'cluster_bootstrap_ci' if formal else 'point_only' if system_count == 1 else 'descriptive_range', 'lower': np.nan, 'upper': np.nan}
            if formal:
                boot = bootstrap_system_mean(group[['system', 'estimate']], 'estimate', SEED_RETURN_TEMPERATURE + int(duration))
                row['lower'], row['upper'] = np.quantile(boot, [0.025, 0.975])
            elif system_count > 1:
                row['lower'] = float(group['estimate'].min())
                row['upper'] = float(group['estimate'].max())
            duration_rows.append(row)
        wide = provider_rows.pivot(index='system', columns='duration_days', values='estimate')
        for short in (7, 14):
            values = (wide[short] - wide[90]).rename('contrast_c').reset_index()
            estimate = float(values['contrast_c'].mean())
            row = {'provider': provider, 'short_days': short, 'long_days': 90, 'physical_systems': system_count, 'contrast_c': estimate, 'uncertainty_type': 'cluster_bootstrap_ci' if formal else 'point_only' if system_count == 1 else 'descriptive_range', 'ci95_lower': np.nan, 'ci95_upper': np.nan}
            if formal:
                boot = bootstrap_system_mean(values, 'contrast_c', SEED_RETURN_TEMPERATURE + 1000 + short)
                row['ci95_lower'], row['ci95_upper'] = np.quantile(boot, [0.025, 0.975])
            elif system_count > 1:
                row['ci95_lower'] = float(values['contrast_c'].min())
                row['ci95_upper'] = float(values['contrast_c'].max())
            contrast_rows.append(row)
    return (pd.DataFrame(duration_rows), pd.DataFrame(contrast_rows))

def structural_nonoverlap_gate(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    for (provider, system, year), group in daily.groupby(['provider', 'system', 'heating_year'], observed=True, sort=True):
        starts = nonoverlap_eligible_starts(group, heating_year=int(year))
        rows.append({'provider': provider, 'system': system, 'heating_year': int(year), 'eligible_common_starts': len(starts)})
    audit = pd.DataFrame(rows)
    summary = audit.groupby('provider', observed=True).agg(physical_systems=('system', lambda values: int(pd.Series(values).nunique())), eligible_common_starts=('eligible_common_starts', 'sum')).reset_index()
    summary['structural_gate_pass'] = summary['physical_systems'].ge(5) & summary['eligible_common_starts'].ge(30)
    qualified = summary.loc[summary['structural_gate_pass'], 'provider'].astype(str).tolist()
    return (summary, qualified)

def build_nonoverlap_windows(daily: pd.DataFrame, qualified: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    selected = daily.loc[daily['provider'].isin(qualified)].copy()
    for (provider, system, year), group in selected.groupby(['provider', 'system', 'heating_year'], observed=True, sort=True):
        dates = pd.to_datetime(group['date'], utc=True)
        starts = nonoverlap_eligible_starts(group, heating_year=int(year))
        for start in starts:
            complement = group.loc[dates.lt(start) | dates.ge(start + pd.Timedelta(days=90))]
            reference = float(complement['daily_median_return_c'].median())
            for duration in DURATIONS:
                mask = dates.ge(start) & dates.lt(start + pd.Timedelta(days=duration))
                endpoint = float(group.loc[mask, 'daily_median_return_c'].median())
                records.append({'provider': provider, 'system': system, 'heating_year': int(year), 'window_start': start, 'duration_days': int(duration), 'valid_window_days': int(mask.sum()), 'valid_complement_days': int(len(complement)), 'absolute_error_c': abs(endpoint - reference)})
    return pd.DataFrame(records)

def summarize_nonoverlap(windows: pd.DataFrame) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    system_rows = system_duration_rows(windows)
    rows: list[dict[str, object]] = []
    for provider, group in system_rows.groupby('provider', observed=True, sort=True):
        wide = group.pivot(index='system', columns='duration_days', values='estimate')
        for short in (7, 14):
            values = (wide[short] - wide[90]).rename('contrast_c').reset_index()
            boot = bootstrap_system_mean(values, 'contrast_c', SEED_RETURN_TEMPERATURE + 9000 + short)
            lower, upper = np.quantile(boot, [0.025, 0.975])
            rows.append({'provider': provider, 'short_days': short, 'long_days': 90, 'physical_systems': int(values['system'].nunique()), 'contrast_c': float(values['contrast_c'].mean()), 'ci95_lower': float(lower), 'ci95_upper': float(upper)})
    return pd.DataFrame(rows)
