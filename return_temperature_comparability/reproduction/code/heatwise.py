from __future__ import annotations
import csv
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
EXPECTED_PREREG = 'fa0bec01d9efaa7927a2af889caa3115b64408083ab449c68b4f5b68871e7111'
SEED = 20260827
BOOTSTRAPS = 2000
DURATIONS = [7, 14, 30, 60, 90]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def log_access(carrier: Path, columns: Iterable[str], purpose: str, output_hash: str='pending', note: str='input_processing') -> None:
    signature = f"{carrier.resolve()},{sha256_file(carrier)},{'|'.join(columns)},true,{EXPECTED_PREREG},{purpose}"
    existing = ACCESS_LOG.read_text(encoding='utf-8') if ACCESS_LOG.exists() else ''
    if signature in existing:
        return
    with ACCESS_LOG.open('a', newline='', encoding='utf-8') as handle:
        csv.writer(handle).writerow([datetime.now(timezone.utc).isoformat(), 'return_temperature_comparability', Path(__file__).name, str(carrier.resolve()), sha256_file(carrier), '|'.join(columns), 'true', EXPECTED_PREREG, purpose, output_hash, note])

def read_source_columns(path: Path) -> tuple[list[str], int]:
    'Return the provider columns without changing a malformed source file.'
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        first = handle.readline()
        if first.startswith('"') and (not first.rstrip('\r\n').endswith('"')):
            pieces = [first]
            while True:
                continuation = handle.readline()
                if not continuation:
                    raise ValueError(f'Unterminated quoted header in {path}')
                pieces.append(continuation)
                if continuation.rstrip('\r\n').endswith('"'):
                    break
            joined = ''.join(pieces).strip('\r\n')
            joined = joined[1:-1].replace('\r', '').replace('\n', '')
            return (joined.split(','), len(pieces))
        return (next(csv.reader([first])), 1)

def read_selected_csv(path: Path, selected: list[str]) -> pd.DataFrame:
    columns, skiprows = read_source_columns(path)
    missing = sorted(set(selected).difference(columns))
    if missing:
        raise ValueError(f'Missing columns in {path.name}: {missing}')
    return pd.read_csv(path, names=columns, skiprows=1 if skiprows > 1 else skiprows, usecols=selected, low_memory=False)

def find_column(columns: Iterable[str], required: Iterable[str]) -> str:
    tokens = list(required)
    candidates = [value for value in columns if all((token in value for token in tokens))]
    if len(candidates) != 1:
        raise ValueError(f'Expected one column for {tokens}, found {candidates}')
    return candidates[0]

def harmonize_system(raw: pd.DataFrame, *, provider: str, system: str, time_col: str, return_col: str, power_col: str | None, flow_col: str | None) -> pd.DataFrame:
    time = pd.to_datetime(raw[time_col], errors='coerce', utc=True)
    return_c = pd.to_numeric(raw[return_col], errors='coerce')
    power = pd.to_numeric(raw[power_col], errors='coerce') if power_col is not None else pd.Series(np.nan, index=raw.index, dtype=float)
    flow = pd.to_numeric(raw[flow_col], errors='coerce') if flow_col is not None else pd.Series(np.nan, index=raw.index, dtype=float)
    active = power.gt(0) | power.isna() & flow.gt(0)
    valid = time.notna() & return_c.between(0.0, 130.0) & active
    frame = pd.DataFrame({'hour': time.loc[valid].dt.floor('h'), 'return_c': return_c.loc[valid].astype(float), 'heat_kwh': power.loc[valid].where(power.loc[valid].gt(0)).astype(float)})
    hourly = frame.groupby('hour', as_index=False, observed=True).agg(return_c=('return_c', 'median'), heat_kwh=('heat_kwh', 'mean')).sort_values('hour')
    hourly['provider'] = provider
    hourly['system'] = system
    return hourly

def load_weather(path: Path, time_col: str, outdoor_col: str, provider: str) -> pd.DataFrame:
    log_access(path, [time_col, outdoor_col], f'{provider} hourly outdoor-temperature harmonization')
    raw = read_selected_csv(path, [time_col, outdoor_col])
    time = pd.to_datetime(raw[time_col], errors='coerce', utc=True)
    outdoor = pd.to_numeric(raw[outdoor_col], errors='coerce')
    valid = time.notna() & outdoor.between(-50.0, 60.0)
    return pd.DataFrame({'hour': time.loc[valid].dt.floor('h'), 'outdoor_c': outdoor.loc[valid].astype(float)}).groupby('hour', as_index=False, observed=True).agg(outdoor_c=('outdoor_c', 'median')).sort_values('hour')

def merge_weather(systems: list[pd.DataFrame], weather: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(systems, ignore_index=True)
    return combined.merge(weather, on='hour', how='inner', validate='many_to_one')

def load_aau() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    heating = RAW / 'FO01_2022_1__2024_7.csv'
    outdoor = RAW / 'outdoor_2022_1__2024_7.csv'
    columns, _ = read_source_columns(heating)
    time_col = 'time'
    return_col = find_column(columns, ['__Full__', 'temperature_return_water'])
    power_col = find_column(columns, ['__Full__', 'power_heat'])
    flow_col = find_column(columns, ['__Full__', 'flowrate_water__L2'])
    selected = [time_col, return_col, power_col, flow_col]
    log_access(heating, selected, 'AAU full-building hydronic-system harmonization')
    raw = read_selected_csv(heating, selected)
    system = harmonize_system(raw, provider='AAU', system='AAU:FO01:Full-DH', time_col=time_col, return_col=return_col, power_col=power_col, flow_col=flow_col)
    outdoor_columns, _ = read_source_columns(outdoor)
    outdoor_col = find_column(outdoor_columns, ['outdoor_temperature'])
    weather = load_weather(outdoor, 'time', outdoor_col, 'AAU')
    merged = merge_weather([system], weather)
    audit = [{'provider': 'AAU', 'system': 'AAU:FO01:Full-DH', 'source_rows': int(len(raw)), 'qualifying_hours_before_weather': int(len(system)), 'qualifying_hours_after_weather': int(len(merged)), 'first_hour_utc': merged['hour'].min(), 'last_hour_utc': merged['hour'].max()}]
    return (merged, audit)

def nest_system_specs(path: Path) -> list[tuple[str, str, str, str | None]]:
    columns, _ = read_source_columns(path)
    building = path.stem.split('_')[0].upper()
    networks = ['HTE_Net', 'MTE_Net'] if building == 'UMAR' else ['MTE_Net']
    specs: list[tuple[str, str, str, str | None]] = []
    for network in networks:
        required = ['__Full__', f'__{network}__']
        return_col = find_column(columns, [*required, 'temperature_return_water'])
        power_col = find_column(columns, [*required, 'power_heat'])
        flow_candidates = [value for value in columns if all((token in value for token in required)) and 'flowrate_water' in value]
        flow_col = flow_candidates[0] if len(flow_candidates) == 1 else None
        specs.append((f'Empa/NEST:{building}:{network}', return_col, power_col, flow_col))
    return specs

def load_nest() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    weather_path = RAW / 'outdoor_weather_station.csv'
    weather_columns, _ = read_source_columns(weather_path)
    time_col = weather_columns[0]
    outdoor_col = find_column(weather_columns, ['outdoor_temperature'])
    weather = load_weather(weather_path, time_col, outdoor_col, 'Empa/NEST')
    systems: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for path in [RAW / 'dfab_thermal_meter.csv', RAW / 'hilo_thermal_meter.csv', RAW / 'sprint_thermal_meter.csv', RAW / 'umar_thermal_meter.csv']:
        columns, _ = read_source_columns(path)
        local_time = columns[0]
        for system_name, return_col, power_col, flow_col in nest_system_specs(path):
            selected = [local_time, return_col, power_col]
            if flow_col is not None:
                selected.append(flow_col)
            log_access(path, selected, f'{system_name} hydronic-system harmonization')
            raw = read_selected_csv(path, selected)
            system = harmonize_system(raw, provider='Empa/NEST', system=system_name, time_col=local_time, return_col=return_col, power_col=power_col, flow_col=flow_col)
            systems.append(system)
            audit.append({'provider': 'Empa/NEST', 'system': system_name, 'source_rows': int(len(raw)), 'qualifying_hours_before_weather': int(len(system))})
    merged = merge_weather(systems, weather)
    merged_counts = merged.groupby('system', observed=True).size()
    for record in audit:
        record['qualifying_hours_after_weather'] = int(merged_counts.get(record['system'], 0))
        local = merged.loc[merged['system'].eq(record['system']), 'hour']
        record['first_hour_utc'] = local.min() if len(local) else pd.NaT
        record['last_hour_utc'] = local.max() if len(local) else pd.NaT
    return (merged, audit)

def heating_year(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, utc=True)
    return parsed.dt.year + parsed.dt.month.ge(10).astype(int)

def energy_weighted_mean(values: pd.Series, energy: pd.Series) -> float:
    valid = values.notna() & energy.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(values.loc[valid].to_numpy(float), weights=energy.loc[valid].to_numpy(float)))

def hierarchy_weights(frame: pd.DataFrame, levels: list[str]) -> np.ndarray:
    keys = frame[levels].copy()
    weight = np.ones(len(frame), dtype=float)
    for index, level in enumerate(levels):
        if index == 0:
            weight /= float(keys[level].nunique())
            continue
        parents = levels[:index]
        unique = keys[parents + [level]].drop_duplicates()
        counts = unique.groupby(parents, observed=True)[level].nunique().rename('n_child')
        lookup = keys[parents].merge(counts.reset_index(), on=parents, how='left')['n_child']
        weight /= lookup.to_numpy(float)
    row_counts = keys.groupby(levels, observed=True)[levels[-1]].transform('size').to_numpy(float)
    weight /= row_counts
    return weight / weight.sum()

def weighted_mean(values: pd.Series, weights: np.ndarray | pd.Series) -> float:
    values_array = np.asarray(values, float)
    weight_array = np.asarray(weights, float)
    valid = np.isfinite(values_array) & np.isfinite(weight_array)
    if not valid.any():
        return np.nan
    return float(np.average(values_array[valid], weights=weight_array[valid]))

def external_windows(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = hourly.copy()
    local['date'] = pd.to_datetime(local['hour'], utc=True).dt.floor('D')
    local = local.loc[local['date'].dt.month.isin([10, 11, 12, 1, 2, 3, 4])].copy()
    local['heating_year'] = heating_year(local['date']).astype(int)
    day_count = local.groupby(['provider', 'system', 'heating_year', 'date'], observed=True).size().rename('hours').reset_index()
    valid_days = day_count.loc[day_count['hours'].ge(4), ['provider', 'system', 'heating_year', 'date']]
    local = local.merge(valid_days, on=['provider', 'system', 'heating_year', 'date'], how='inner', validate='many_to_one')
    records: list[dict[str, object]] = []
    periods: list[dict[str, object]] = []
    for (provider, system, year), rows in local.groupby(['provider', 'system', 'heating_year'], observed=True):
        rows = rows.sort_values('hour')
        valid_day_count = int(rows['date'].nunique())
        status = 'eligible' if valid_day_count >= 150 else 'exclude_lt_150_valid_days'
        period_record: dict[str, object] = {'provider': provider, 'system': system, 'heating_year': int(year), 'valid_days': valid_day_count, 'valid_hours': int(rows['hour'].nunique()), 'eligibility_status': status}
        if status != 'eligible':
            periods.append(period_record)
            continue
        reference_median = float(rows['return_c'].median())
        reference_heat_weighted = energy_weighted_mean(rows['return_c'], rows['heat_kwh'])
        period_record['reference_median_return_c'] = reference_median
        period_record['reference_heat_weighted_return_c'] = reference_heat_weighted
        periods.append(period_record)
        period_start = pd.Timestamp(int(year) - 1, 10, 1, tz='UTC')
        period_end = pd.Timestamp(int(year), 5, 1, tz='UTC')
        for duration in DURATIONS:
            start = period_start
            while start <= period_end - pd.Timedelta(days=duration):
                end = start + pd.Timedelta(days=duration)
                window = rows.loc[rows['hour'].ge(start) & rows['hour'].lt(end)]
                valid_window_days = int(window['date'].nunique())
                if valid_window_days >= math.ceil(0.8 * duration):
                    median = float(window['return_c'].median())
                    heat_weighted = energy_weighted_mean(window['return_c'], window['heat_kwh'])
                    records.append({'provider': provider, 'system': system, 'heating_year': int(year), 'duration_days': duration, 'window_start': start, 'window_end': end, 'valid_days': valid_window_days, 'valid_hours': int(window['hour'].nunique()), 'valid_day_fraction': valid_window_days / duration, 'window_median_return_c': median, 'reference_median_return_c': reference_median, 'absolute_temperature_error': abs(median - reference_median), 'window_heat_weighted_return_c': heat_weighted, 'reference_heat_weighted_return_c': reference_heat_weighted, 'energy_weighted_absolute_error': abs(heat_weighted - reference_heat_weighted) if np.isfinite(heat_weighted) and np.isfinite(reference_heat_weighted) else np.nan})
                start += pd.Timedelta(days=7)
    windows = pd.DataFrame(records)
    periods_frame = pd.DataFrame(periods)
    if not windows.empty:
        windows['weight'] = np.nan
        for provider, provider_rows in windows.groupby('provider', observed=True):
            windows.loc[provider_rows.index, 'weight'] = hierarchy_weights(provider_rows, ['system', 'heating_year'])
        windows = windows.sort_values(['provider', 'system', 'heating_year', 'duration_days', 'window_start']).reset_index(drop=True)
    return (windows, periods_frame.sort_values(['provider', 'system', 'heating_year']).reset_index(drop=True))

def provider_duration_evidence(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_rows: list[dict[str, object]] = []
    metrics = {'mean_absolute_error_c': 'absolute_temperature_error', 'mean_energy_weighted_absolute_error_c': 'energy_weighted_absolute_error'}
    for (provider, system, duration), rows in frame.groupby(['provider', 'system', 'duration_days'], observed=True):
        weights = hierarchy_weights(rows, ['heating_year'])
        record: dict[str, object] = {'provider': provider, 'system': system, 'duration_days': int(duration), 'heating_periods': int(rows['heating_year'].nunique()), 'windows': int(len(rows))}
        for metric, source in metrics.items():
            record[metric] = weighted_mean(rows[source], weights)
        cluster_rows.append(record)
    clusters = pd.DataFrame(cluster_rows)
    estimates: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    for provider, provider_clusters in clusters.groupby('provider', observed=True):
        rng = np.random.default_rng(SEED + sum((ord(value) for value in str(provider))))
        for duration, duration_rows in provider_clusters.groupby('duration_days', observed=True):
            for metric in metrics:
                values = duration_rows[metric].to_numpy(float)
                finite = np.isfinite(values)
                valid_values = values[finite]
                if not len(valid_values):
                    continue
                if len(valid_values) >= 2:
                    sampled = rng.integers(0, len(valid_values), size=(BOOTSTRAPS, len(valid_values)))
                    boot = valid_values[sampled].mean(axis=1)
                    ci_low = float(np.quantile(boot, 0.025))
                    ci_high = float(np.quantile(boot, 0.975))
                else:
                    ci_low = np.nan
                    ci_high = np.nan
                estimates.append({'provider': provider, 'record_type': 'duration_estimate', 'metric': metric, 'duration_days': int(duration), 'comparison': '', 'estimate': float(valid_values.mean()), 'ci_low': ci_low, 'ci_high': ci_high, 'systems': int(len(valid_values)), 'heating_periods': int(duration_rows.loc[finite, 'heating_periods'].sum()), 'windows': int(duration_rows.loc[finite, 'windows'].sum())})
        for metric in metrics:
            pivot = provider_clusters.pivot(index='system', columns='duration_days', values=metric)
            for short in [7, 14]:
                paired = pivot[[short, 90]].dropna()
                differences = (paired[short] - paired[90]).to_numpy(float)
                if not len(differences):
                    continue
                if len(differences) >= 2:
                    sampled = rng.integers(0, len(differences), size=(BOOTSTRAPS, len(differences)))
                    boot = differences[sampled].mean(axis=1)
                    ci_low = float(np.quantile(boot, 0.025))
                    ci_high = float(np.quantile(boot, 0.975))
                else:
                    ci_low = np.nan
                    ci_high = np.nan
                contrasts.append({'provider': provider, 'record_type': 'duration_contrast', 'metric': metric, 'duration_days': np.nan, 'comparison': f'{short}_minus_90_days', 'estimate': float(differences.mean()), 'ci_low': ci_low, 'ci_high': ci_high, 'systems': int(len(differences)), 'heating_periods': np.nan, 'windows': np.nan})
    return (pd.DataFrame(estimates), pd.DataFrame(contrasts))

def annotate_endpoint(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output['endpoint'] = np.where(output['metric'].eq('mean_absolute_error_c'), 'complete-period median', 'complete-period heat-weighted mean')
    output['evidence_unit'] = 'physical hydronic system'
    return output

def combine_with_provider_results(estimates: pd.DataFrame, contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    provider_results_path = ROOT.parent / 'original_source_records/RESULTS_TABLES.csv'
    provider_results = pd.read_csv(provider_results_path)
    prior_estimates = provider_results.loc[provider_results['table_id'].eq('duration_estimates')].copy()
    prior_contrasts = provider_results.loc[provider_results['table_id'].eq('short_minus_90_contrasts')].copy()
    shared = ['provider', 'record_type', 'metric', 'duration_days', 'comparison', 'estimate', 'ci_low', 'ci_high', 'heating_periods', 'windows', 'endpoint', 'evidence_unit']
    new_estimates = annotate_endpoint(estimates)
    new_contrasts = annotate_endpoint(contrasts)
    for frame in [new_estimates, new_contrasts]:
        frame['heating_periods'] = frame['heating_periods']
        frame['windows'] = frame['windows']
    combined_estimates = pd.concat([prior_estimates.reindex(columns=shared), new_estimates.reindex(columns=shared)], ignore_index=True)
    combined_contrasts = pd.concat([prior_contrasts.reindex(columns=shared), new_contrasts.reindex(columns=shared)], ignore_index=True)
    combined_estimates['endpoint_family'] = np.where(combined_estimates['endpoint'].str.contains('heat-weighted', na=False), 'heat_weighted', 'median')
    combined_contrasts['endpoint_family'] = np.where(combined_contrasts['endpoint'].str.contains('heat-weighted', na=False), 'heat_weighted', 'median')
    provider_summary: list[dict[str, object]] = []
    for (endpoint, duration), rows in combined_estimates.groupby(['endpoint_family', 'duration_days'], observed=True):
        values = rows['estimate'].dropna().to_numpy(float)
        provider_summary.append({'record_type': 'provider_equal_duration_summary', 'endpoint_family': endpoint, 'duration_days': int(duration), 'comparison': '', 'provider_equal_estimate': float(values.mean()), 'provider_min': float(values.min()), 'provider_max': float(values.max()), 'providers': int(len(values))})
    for (endpoint, comparison), rows in combined_contrasts.groupby(['endpoint_family', 'comparison'], observed=True):
        values = rows['estimate'].dropna().to_numpy(float)
        provider_summary.append({'record_type': 'provider_equal_duration_contrast', 'endpoint_family': endpoint, 'duration_days': np.nan, 'comparison': comparison, 'provider_equal_estimate': float(values.mean()), 'provider_min': float(values.min()), 'provider_max': float(values.max()), 'providers': int(len(values))})
    return (combined_estimates, combined_contrasts, pd.DataFrame(provider_summary))
