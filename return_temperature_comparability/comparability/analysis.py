'Locked, read-only-source return_temperature_comparability pair comparison; fixed analysis specification.'
from __future__ import annotations
import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import resource
import signal
import sys
import threading
import time
from itertools import combinations
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile
ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_variable] = '1'
import numpy as np
import pandas as pd
DURATIONS = (7, 14, 30, 60, 90)
PROVIDERS = ('PreDist', 'XAI4HEAT')
ENDPOINTS = ('median', 'heat_weighted')
METRICS = ('sync_mae_c', 'async_mae_c', 'sync_minus_async_mae_c', 'sync_signed_error_c', 'async_signed_error_c', 'marginal_i_mae_c', 'marginal_j_mae_c', 'sync_reversal_rate', 'async_reversal_rate', 'sync_tie_rate', 'async_tie_rate')

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def json_write(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def validate_spec(spec):
    expected = {'providers': list(PROVIDERS), 'durations': list(DURATIONS), 'minimum_shared_dates': 150, 'minimum_valid_day_hours': 4, 'minimum_campaign_valid_fraction': 0.8, 'weekly_step_days': 7, 'complement_envelope_days': 90, 'minimum_complement_dates': 60, 'bootstrap_replicates': 5000, 'primary_seed': 2026100701, 'sensitivity_seed': 2026100702, 'stage_limit_seconds': 3600, 'checkpoint_seconds': 3500, 'memory_limit_bytes': 4 * 1024 ** 3, 'blas_threads': 1, 'reference': 'each_system_hourly_records_on_shared_valid_dates_not_hour_intersection', 'async': 'full_cartesian_same_common_eligible_start_set_including_diagonal', 'orientation': 'sorted_physical_identifiers_i_minus_j', 'aggregation': 'window_then_year_then_counterpart_then_focal_equal_within_provider', 'disjoint': 'synchronous_daily_median_common_90day_envelope_complement_separate_estimand', 'bootstrap': 'node_multiplicity_both_endpoints_no_duplicate_node_self_pairs'}
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f'Locked specification mismatch: {key}')
    if not 0 < spec['memory_checkpoint_bytes'] < spec['memory_limit_bytes']:
        raise ValueError('Memory checkpoint must precede registered limit')

def safe_functions(path, names, namespace):
    'Compile only named frozen functions, never top-level code or log_access.'
    tree = ast.parse(Path(path).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('Requested frozen function missing')
    for function in selected:
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == 'log_access':
                    raise ValueError('Writing access log forbidden')
    result = dict(namespace)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), 'exec', flags=__future_annotations()), result)
    return result

def __future_annotations():
    import __future__
    return __future__.annotations.compiler_flag

def endpoint(values, weights, family):
    values, weights = (np.asarray(values, float), np.asarray(weights, float))
    if family == 'median':
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if len(finite) else np.nan
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[good], weights=weights[good])) if good.any() else np.nan

def campaign_starts(shared_dates, year):
    dates = pd.DatetimeIndex(shared_dates)
    candidate = pd.date_range(pd.Timestamp(year - 1, 10, 1, tz='UTC'), pd.Timestamp(year, 5, 1, tz='UTC') - pd.Timedelta(days=90), freq='7D')
    accepted, audit = ([], [])
    for start in candidate:
        counts = {f'count_{d}': int(((dates >= start) & (dates < start + pd.Timedelta(days=d))).sum()) for d in DURATIONS}
        good = all((counts[f'count_{d}'] >= math.ceil(0.8 * d) for d in DURATIONS))
        complement = int(daily_complement(dates, start).sum())
        audit.append(dict(window_start=start.isoformat(), eligible=good, complement_dates=complement, disjoint_eligible=good and complement >= 60, **counts))
        if good:
            accepted.append(start)
    return (accepted, audit)

def daily_complement(dates, start):
    return (dates < start) | (dates >= start + pd.Timedelta(days=90))

def contrast_arrays(a, b, reference):
    contrast = np.asarray(a)[:, None] - np.asarray(b)[None, :]
    error = contrast - reference
    good = np.isfinite(contrast) & np.isfinite(reference)
    reversal = np.where(good & (reference != 0), (contrast * reference < 0).astype(float), np.nan)
    tie = np.where(good & (reference != 0), (contrast == 0).astype(float), np.nan)
    return dict(contrast=contrast, error=error, absolute_error=np.abs(error), reversal=reversal, tie=tie)

def mean_finite(values):
    values = np.asarray(values, float)
    good = np.isfinite(values)
    return float(values[good].mean()) if good.any() else np.nan

def normalize_frozen_hourly(frame):
    local = frame.rename(columns={'substation': 'system'}).copy()
    local['hour'] = pd.to_datetime(local.hour, utc=True)
    return local

def pair_season_means(local, metrics):
    return local.groupby(['system_i', 'system_j'], sort=True)[metrics].mean().reset_index()

def node_estimate(matrix, multiplicity):
    'Counterpart multiplicity inside focal mean, focal multiplicity outside.'
    matrix = np.array(matrix, dtype=float, copy=True)
    np.fill_diagonal(matrix, np.nan)
    counts = np.asarray(multiplicity, float)
    good = np.isfinite(matrix)
    denominator = good @ counts
    numerator = np.nan_to_num(matrix, nan=0.0) @ counts
    focal = np.divide(numerator, denominator, out=np.full(len(counts), np.nan), where=denominator > 0)
    retained = np.isfinite(focal) & (counts > 0)
    return float(np.average(focal[retained], weights=counts[retained])) if retained.any() else np.nan

class Guard:

    def __init__(self, output, spec):
        self.output, self.spec = (output, spec)
        self.name, self.start, self.progress = ('initialization', time.monotonic(), {})
        self.stats, self.stop = ([], threading.Event())
        signal.signal(signal.SIGALRM, self.timeout)
        self.monitor = threading.Thread(target=self.watch, daemon=True)
        self.monitor.start()

    def rss(self):
        maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(maximum if sys.platform == 'darwin' else maximum * 1024)

    def checkpoint(self, reason):
        json_write(self.output / 'CHECKPOINT.json', dict(stage=self.name, reason=reason, elapsed_seconds=time.monotonic() - self.start, peak_rss_bytes=self.rss(), progress=self.progress, status='stopped_not_complete'))

    def timeout(self, *_):
        self.checkpoint('stage_checkpoint_time_limit')
        raise TimeoutError('Stage stopped before 3600-second limit')

    def watch(self):
        while not self.stop.wait(0.5):
            if self.rss() >= self.spec['memory_checkpoint_bytes']:
                self.checkpoint('memory_checkpoint_limit')
                os._exit(75)

    def stage(self, name):
        if self.name != 'initialization':
            self.finish_stage()
        self.name, self.start, self.progress = (name, time.monotonic(), {})
        signal.setitimer(signal.ITIMER_REAL, self.spec['checkpoint_seconds'])
        print(f'STAGE {name}', flush=True)

    def finish_stage(self):
        elapsed = time.monotonic() - self.start
        self.stats.append(dict(stage=self.name, elapsed_seconds=elapsed, cumulative_peak_rss_bytes=self.rss()))
        json_write(self.output / 'RESOURCE_STATS.json', self.stats)
        if elapsed >= self.spec['stage_limit_seconds'] or self.rss() >= self.spec['memory_limit_bytes']:
            raise RuntimeError('Registered compute limit exceeded')

    def close(self):
        self.finish_stage()
        signal.setitimer(signal.ITIMER_REAL, 0)
        self.stop.set()
        self.monitor.join(timeout=2)

class CsvSink:

    def __init__(self, path, fields):
        self.handle = gzip.open(path, 'wt', newline='') if str(path).endswith('.gz') else Path(path).open('w', newline='')
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        self.writer.writeheader()

    def row(self, value):
        self.writer.writerow(value)

    def close(self):
        self.handle.close()

def prepare(inputs, output, guard):
    namespace = dict(np=np, pd=pd, Path=Path, ZipFile=ZipFile, PREDIST=inputs['predist'], XAI=inputs['xai'])
    portable = safe_functions(inputs['temporal_reader'], ['_hourly_from_predist_member', 'load_xai_hourly', 'heating_year', 'build_daily_medians'], namespace)
    weighted = safe_functions(inputs['weighted_reader'], ['_hourly_from_predist_member', 'load_xai_hourly'], namespace)
    frames, member_audit = ([], [])
    with ZipFile(inputs['predist']) as archive:
        members = sorted((n for n in archive.namelist() if '/operational_data/substation_' in n and n.endswith('.csv')))
        for index, name in enumerate(members):
            guard.progress = dict(provider='PreDist', members_completed=index, members=len(members))
            original = normalize_frozen_hourly(weighted['_hourly_from_predist_member'](archive, name))
            replay = portable['_hourly_from_predist_member'](archive, name)
            pd.testing.assert_frame_equal(original[replay.columns].reset_index(drop=True), replay.reset_index(drop=True), check_dtype=False)
            member_audit.append(dict(provider='PreDist', member=name, system='manufacturer_1:' + Path(name).stem if name.startswith('manufacturer 1/') else 'manufacturer_2:' + Path(name).stem, active_valid_hours=len(original), portable_replay='exact'))
            if len(original):
                frames.append(original[['hour', 'return_c', 'heat_kwh', 'provider', 'system']])
            if (index + 1) % 10 == 0:
                print(f'PreDist frozen-reader replay {index + 1}/{len(members)}', flush=True)
    xai = normalize_frozen_hourly(weighted['load_xai_hourly']())
    replay = portable['load_xai_hourly']()
    pd.testing.assert_frame_equal(xai[replay.columns].reset_index(drop=True), replay.reset_index(drop=True), check_dtype=False)
    frames.append(xai[['hour', 'return_c', 'heat_kwh', 'provider', 'system']])
    for system, group in xai.groupby('system', sort=True):
        member_audit.append(dict(provider='XAI4HEAT', member='xai4heat_scada_' + system + '_processed.csv', system=system, active_valid_hours=len(group), portable_replay='exact'))
    pd.DataFrame(member_audit).to_csv(output / 'MEMBER_IDENTITIES.csv', index=False)
    hourly = pd.concat(frames, ignore_index=True)
    del frames, xai, replay
    daily = portable['build_daily_medians'](hourly)
    frozen = pd.read_csv(inputs['valid_day_carrier'])
    frozen = frozen.loc[frozen.provider.isin(PROVIDERS)].copy()
    frozen['date'] = pd.to_datetime(frozen.date, utc=True)
    keys = ['provider', 'system', 'heating_year', 'date']
    frozen = frozen.sort_values(keys).reset_index(drop=True)
    daily = daily.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(daily, frozen[daily.columns], check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    json_write(output / 'FROZEN_REPLAY_CHECK.json', dict(valid_day_rows=len(daily), providers={p: int(daily.loc[daily.provider.eq(p), 'system'].nunique()) for p in PROVIDERS}, portable_hourly_replay='exact_all_members', daily_carrier_replay='all_identities_and_values_1e-12'))
    hourly['date'] = pd.to_datetime(hourly.hour, utc=True).dt.floor('D')
    hourly = hourly.loc[hourly.date.dt.month.isin([10, 11, 12, 1, 2, 3, 4])].copy()
    hourly['heating_year'] = portable['heating_year'](hourly.date).astype(int)
    rawday = hourly.groupby(keys, observed=True).agg(valid_hours=('hour', 'nunique')).reset_index()
    period_audit = rawday.groupby(keys[:3], observed=True).agg(recorded_days=('date', 'size'), valid_days=('valid_hours', lambda x: int((x >= 4).sum()))).reset_index()
    period_audit['status'] = np.where(period_audit.valid_days >= 150, 'eligible_system_season', 'exclude_lt_150_valid_dates')
    period_audit.to_csv(output / 'SYSTEM_SEASON_AUDIT.csv', index=False)
    hourly = hourly.merge(daily[keys], on=keys, how='inner', validate='many_to_one')
    daily.to_csv(output / 'VALID_DAY_REPLAY.csv.gz', index=False)
    periods = {}
    for key, group in hourly.groupby(keys[:3], observed=True, sort=True):
        group = group.sort_values('hour')
        day = daily.loc[(daily.provider == key[0]) & (daily.system == key[1]) & (daily.heating_year == key[2])]
        dates = pd.DatetimeIndex(day.date)
        periods[key] = dict(dates=dates, daily=day.daily_median_return_c.to_numpy(float), record_dates=pd.DatetimeIndex(group.date), hours=pd.DatetimeIndex(group.hour), values=group.return_c.to_numpy(float), weights=group.heat_kwh.to_numpy(float))
    print(f'Prepared {len(periods)} eligible system-seasons, {len(hourly)} valid-date hourly records', flush=True)
    return (periods, period_audit)

def pair_metrics(a, b, reference, ai_ref, bj_ref):
    arrays = contrast_arrays(a, b, reference)
    diagonal = np.diag(arrays['error'])
    marginal_i = np.abs(a - ai_ref)
    marginal_j = np.abs(b - bj_ref)
    metrics = dict(sync_mae_c=mean_finite(np.abs(diagonal)), async_mae_c=mean_finite(arrays['absolute_error']), sync_signed_error_c=mean_finite(diagonal), async_signed_error_c=mean_finite(arrays['error']), marginal_i_mae_c=mean_finite(marginal_i), marginal_j_mae_c=mean_finite(marginal_j), sync_reversal_rate=mean_finite(np.diag(arrays['reversal'])), async_reversal_rate=mean_finite(arrays['reversal']), sync_tie_rate=mean_finite(np.diag(arrays['tie'])), async_tie_rate=mean_finite(arrays['tie']))
    metrics['sync_minus_async_mae_c'] = metrics['sync_mae_c'] - metrics['async_mae_c']
    if np.isfinite(marginal_i).all() and np.isfinite(marginal_j).all():
        assert np.isclose(marginal_i.mean(), np.broadcast_to(marginal_i[:, None], arrays['error'].shape).mean())
        assert np.isclose(marginal_j.mean(), np.broadcast_to(marginal_j[None, :], arrays['error'].shape).mean())
        assert np.isclose(metrics['sync_signed_error_c'], metrics['async_signed_error_c'], atol=1e-12)
    counts = dict(sync_combinations=len(a), async_combinations=len(a) * len(b), sync_finite_errors=int(np.isfinite(diagonal).sum()), async_finite_errors=int(np.isfinite(arrays['error']).sum()), sync_reversals=int(np.nansum(np.diag(arrays['reversal']))), async_reversals=int(np.nansum(arrays['reversal'])), sync_order_defined=int(np.isfinite(np.diag(arrays['reversal'])).sum()), async_order_defined=int(np.isfinite(arrays['reversal']).sum()), sync_ties=int(np.nansum(np.diag(arrays['tie']))), async_ties=int(np.nansum(arrays['tie'])), zero_reference=int(reference == 0), missing_i_campaigns=int((~np.isfinite(a)).sum()), missing_j_campaigns=int((~np.isfinite(b)).sum()))
    return (metrics, counts, arrays)

def compare(periods, period_audit, output, guard):
    identity = ['pair_season_id', 'provider', 'system_i', 'system_j', 'heating_year']
    start_sink = CsvSink(output / 'CANDIDATE_START_AUDIT.csv.gz', identity + ['window_start', 'eligible', 'complement_dates', 'disjoint_eligible'] + [f'count_{d}' for d in DURATIONS])
    date_sink = CsvSink(output / 'SHARED_VALID_DATES.csv.gz', identity + ['date'])
    endpoint_sink = CsvSink(output / 'CAMPAIGN_ENDPOINTS.csv.gz', identity + ['endpoint', 'duration_days', 'window_start', 'shared_window_dates', 'hours_i', 'hours_j', 'positive_weight_hours_i', 'positive_weight_hours_j', 'campaign_i_c', 'campaign_j_c', 'reference_i_c', 'reference_j_c'])
    disjoint_sink = CsvSink(output / 'DATE_DISJOINT_WINDOWS.csv.gz', identity + ['duration_days', 'window_start', 'shared_window_dates', 'complement_dates', 'campaign_i_c', 'campaign_j_c', 'reference_i_c', 'reference_j_c', 'reference_contrast_c', 'campaign_contrast_c', 'signed_error_c', 'absolute_error_c', 'marginal_i_error_c', 'marginal_j_error_c', 'reversal', 'campaign_tie', 'zero_reference'])
    audits, summaries, disjoint_summaries = ([], [], [])
    zip_path = output / 'FULL_CARTESIAN_VALUES.zip'
    counter = 0
    with ZipFile(zip_path, 'w', compression=ZIP_STORED, allowZip64=True) as matrix_archive:
        for provider in PROVIDERS:
            years = sorted(set((int(y) for p, s, y in periods if p == provider)))
            recorded = period_audit.loc[period_audit.provider.eq(provider)]
            for year in sorted(recorded.heating_year.unique()):
                systems = sorted(recorded.loc[recorded.heating_year.eq(year), 'system'])
                for system_i, system_j in combinations(systems, 2):
                    counter += 1
                    key = dict(pair_season_id=f'PS{counter:06d}', provider=provider, system_i=system_i, system_j=system_j, heating_year=int(year))
                    guard.progress = key
                    if (provider, system_i, year) not in periods or (provider, system_j, year) not in periods:
                        audits.append(dict(key, status='exclude_individual_lt_150_valid_dates', shared_valid_dates=np.nan, eligible_starts=0, disjoint_starts=0, async_combinations_per_duration=0))
                        continue
                    i, j = (periods[provider, system_i, year], periods[provider, system_j, year])
                    shared = i['dates'].intersection(j['dates']).sort_values()
                    if len(shared) < 150:
                        audits.append(dict(key, status='exclude_lt_150_shared_valid_dates', shared_valid_dates=len(shared), eligible_starts=0, disjoint_starts=0, async_combinations_per_duration=0))
                        continue
                    starts, start_audit = campaign_starts(shared, int(year))
                    for row in start_audit:
                        start_sink.row(dict(key, **row))
                    for date in shared:
                        date_sink.row(dict(key, date=date.isoformat()))
                    disjoint_starts = [s for s in starts if daily_complement(shared, s).sum() >= 60]
                    audits.append(dict(key, status='eligible' if starts else 'exclude_no_common_all_duration_start', shared_valid_dates=len(shared), eligible_starts=len(starts), disjoint_starts=len(disjoint_starts), async_combinations_per_duration=len(starts) ** 2))
                    if not starts:
                        continue
                    records_i = i['record_dates'].isin(shared)
                    records_j = j['record_dates'].isin(shared)
                    dates_i, dates_j = (i['record_dates'][records_i], j['record_dates'][records_j])
                    xi, wi = (i['values'][records_i], i['weights'][records_i])
                    xj, wj = (j['values'][records_j], j['weights'][records_j])
                    references = np.array([[endpoint(xi, wi, e), endpoint(xj, wj, e)] for e in ENDPOINTS])
                    campaigns = np.full((2, 5, len(starts), 2), np.nan)
                    errors = np.full((2, 5, len(starts), len(starts)), np.nan)
                    reversals = np.full_like(errors, np.nan)
                    ties = np.full_like(errors, np.nan)
                    for di, duration in enumerate(DURATIONS):
                        for si, start in enumerate(starts):
                            end = start + pd.Timedelta(days=duration)
                            mi = (dates_i >= start) & (dates_i < end)
                            mj = (dates_j >= start) & (dates_j < end)
                            shared_window = int(((shared >= start) & (shared < end)).sum())
                            for ei, e in enumerate(ENDPOINTS):
                                ai, bj = (endpoint(xi[mi], wi[mi], e), endpoint(xj[mj], wj[mj], e))
                                campaigns[ei, di, si] = [ai, bj]
                                endpoint_sink.row(dict(key, endpoint=e, duration_days=duration, window_start=start.isoformat(), shared_window_dates=shared_window, hours_i=int(mi.sum()), hours_j=int(mj.sum()), positive_weight_hours_i=int((np.isfinite(wi[mi]) & (wi[mi] > 0)).sum()), positive_weight_hours_j=int((np.isfinite(wj[mj]) & (wj[mj] > 0)).sum()), campaign_i_c=ai, campaign_j_c=bj, reference_i_c=references[ei, 0], reference_j_c=references[ei, 1]))
                        for ei, e in enumerate(ENDPOINTS):
                            reference = references[ei, 0] - references[ei, 1]
                            metrics, counts, arrays = pair_metrics(campaigns[ei, di, :, 0], campaigns[ei, di, :, 1], reference, *references[ei])
                            summaries.append(dict(key, estimand='same_season', endpoint=e, duration_days=duration, shared_valid_dates=len(shared), eligible_starts=len(starts), reference_i_c=references[ei, 0], reference_j_c=references[ei, 1], reference_contrast_c=reference, **counts, **metrics))
                            errors[ei, di], reversals[ei, di], ties[ei, di] = (arrays['error'], arrays['reversal'], arrays['tie'])
                    buffer = io.BytesIO()
                    np.savez_compressed(buffer, references=references, campaigns=campaigns, signed_contrast_errors=errors, absolute_contrast_errors=np.abs(errors), reversals=reversals, campaign_ties=ties, starts=np.array([s.isoformat() for s in starts]), shared_dates=np.array([s.isoformat() for s in shared]), durations=np.array(DURATIONS), endpoints=np.array(ENDPOINTS))
                    matrix_archive.writestr(key['pair_season_id'] + '.npz', buffer.getvalue())
                    daily_i = i['daily'][i['dates'].get_indexer(shared)]
                    daily_j = j['daily'][j['dates'].get_indexer(shared)]
                    for duration in DURATIONS:
                        rows = []
                        for start in disjoint_starts:
                            complement = daily_complement(shared, start)
                            window = (shared >= start) & (shared < start + pd.Timedelta(days=duration))
                            ai, bj = (float(np.median(daily_i[window])), float(np.median(daily_j[window])))
                            ri, rj = (float(np.median(daily_i[complement])), float(np.median(daily_j[complement])))
                            ref, con = (ri - rj, ai - bj)
                            row = dict(key, duration_days=duration, window_start=start.isoformat(), shared_window_dates=int(window.sum()), complement_dates=int(complement.sum()), campaign_i_c=ai, campaign_j_c=bj, reference_i_c=ri, reference_j_c=rj, reference_contrast_c=ref, campaign_contrast_c=con, signed_error_c=con - ref, absolute_error_c=abs(con - ref), marginal_i_error_c=abs(ai - ri), marginal_j_error_c=abs(bj - rj), reversal=float(con * ref < 0) if ref != 0 else np.nan, campaign_tie=float(con == 0) if ref != 0 else np.nan, zero_reference=int(ref == 0))
                            disjoint_sink.row(row)
                            rows.append(row)
                        if rows:
                            disjoint_summaries.append(dict(key, estimand='date_disjoint_daily', endpoint='daily_median', duration_days=duration, eligible_starts=len(rows), sync_combinations=len(rows), async_combinations=0, zero_reference=sum((r['zero_reference'] for r in rows)), sync_order_defined=sum((np.isfinite(r['reversal']) for r in rows)), sync_reversals=sum((r['reversal'] for r in rows if np.isfinite(r['reversal']))), sync_ties=sum((r['campaign_tie'] for r in rows if np.isfinite(r['campaign_tie']))), sync_mae_c=mean_finite([r['absolute_error_c'] for r in rows]), sync_signed_error_c=mean_finite([r['signed_error_c'] for r in rows]), marginal_i_mae_c=mean_finite([r['marginal_i_error_c'] for r in rows]), marginal_j_mae_c=mean_finite([r['marginal_j_error_c'] for r in rows]), sync_reversal_rate=mean_finite([r['reversal'] for r in rows]), sync_tie_rate=mean_finite([r['campaign_tie'] for r in rows])))
                    if counter % 100 == 0:
                        print(f'Compared candidate pair-season {counter}; eligible={len(summaries) // 10}', flush=True)
                        json_write(output / 'CHECKPOINT.json', dict(status='running', stage='comparisons', last_completed=key))
    for sink in [start_sink, date_sink, endpoint_sink, disjoint_sink]:
        sink.close()
    audit = pd.DataFrame(audits)
    audit.to_csv(output / 'PAIR_SEASON_AUDIT.csv', index=False)
    main = pd.DataFrame(summaries)
    sensitivity = pd.DataFrame(disjoint_summaries)
    main.to_csv(output / 'PAIR_SEASON_SUMMARY.csv', index=False, float_format='%.17g')
    sensitivity.to_csv(output / 'DATE_DISJOINT_PAIR_SEASON_SUMMARY.csv', index=False, float_format='%.17g')
    return (main, sensitivity, audit)

def summarize(main, sensitivity, output, spec, guard):
    frame = pd.concat([main, sensitivity], ignore_index=True)
    grouping = ['provider', 'estimand', 'endpoint', 'duration_days']
    all_pair, all_system, provider_rows, los_rows = ([], [], [], [])
    draws_sink = CsvSink(output / 'NODE_BOOTSTRAP_ESTIMATES.csv.gz', grouping + ['metric', 'replicate', 'estimate'])
    boot_archive = ZipFile(output / 'NODE_BOOTSTRAP_DRAWS.zip', 'w', compression=ZIP_STORED)
    for group, local in frame.groupby(grouping, sort=True):
        provider, estimand, endpoint_name, duration = group
        identity = dict(zip(grouping, group))
        metrics = [m for m in METRICS if m in local and local[m].notna().any()]
        pair = pair_season_means(local, metrics)
        years = local.groupby(['system_i', 'system_j']).heating_year.nunique().rename('seasons').reset_index()
        pair = pair.merge(years, on=['system_i', 'system_j'], validate='one_to_one')
        all_pair.append(pair.assign(**identity))
        systems = sorted(set(pair.system_i) | set(pair.system_j))
        index = {s: k for k, s in enumerate(systems)}
        matrices = {}
        for metric in metrics + ['marginal_focal_mae_c']:
            matrix = np.full((len(systems), len(systems)), np.nan)
            for row in pair.itertuples(index=False):
                i, j = (index[row.system_i], index[row.system_j])
                if metric == 'marginal_focal_mae_c':
                    matrix[i, j], matrix[j, i] = (row.marginal_i_mae_c, row.marginal_j_mae_c)
                else:
                    matrix[i, j] = matrix[j, i] = getattr(row, metric)
            matrices[metric] = matrix
            good = np.isfinite(matrix)
            counts = good.sum(axis=1)
            means = np.divide(np.nan_to_num(matrix, nan=0).sum(axis=1), counts, out=np.full(len(systems), np.nan), where=counts > 0)
            for system, n, value in zip(systems, counts, means):
                all_system.append(dict(identity, system=system, metric=metric, eligible_counterparts=int(n), estimate=float(value)))
        if provider == 'PreDist':
            seed = spec['primary_seed'] if estimand == 'same_season' else spec['sensitivity_seed']
            rng = np.random.default_rng(seed)
            multiplicities = rng.multinomial(len(systems), np.full(len(systems), 1 / len(systems)), size=5000)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, systems=np.array(systems), multiplicities=multiplicities, seed=np.array(seed))
            draw_id = f'{estimand}_{endpoint_name}_{duration}'
            boot_archive.writestr(draw_id + '_multiplicities.npz', buffer.getvalue())
        for metric, matrix in matrices.items():
            point = node_estimate(matrix, np.ones(len(systems)))
            lower, upper = (np.nan, np.nan)
            valid_replicates = 0
            if provider == 'PreDist':
                finite = np.isfinite(matrix)
                num = multiplicities @ np.nan_to_num(matrix, nan=0).T
                denom = multiplicities @ finite.astype(float).T
                focal = np.divide(num, denom, out=np.full(num.shape, np.nan), where=denom > 0)
                kept = np.isfinite(focal)
                effective = multiplicities * kept
                boot = np.divide((np.nan_to_num(focal, nan=0) * effective).sum(axis=1), effective.sum(axis=1), out=np.full(5000, np.nan), where=effective.sum(axis=1) > 0)
                valid_replicates = int(np.isfinite(boot).sum())
                if valid_replicates:
                    lower, upper = np.quantile(boot[np.isfinite(boot)], [0.025, 0.975])
                for replicate, value in enumerate(boot):
                    draws_sink.row(dict(identity, metric=metric, replicate=replicate, estimate=float(value)))
                for r in [0, 1, 2, 2499, 4999]:
                    np.testing.assert_allclose(boot[r], node_estimate(matrix, multiplicities[r]), atol=1e-12, equal_nan=True)
            else:
                for omitted in systems:
                    multiplicity = np.ones(len(systems))
                    multiplicity[index[omitted]] = 0
                    los_rows.append(dict(identity, metric=metric, omitted_system=omitted, full_estimate=point, leave_one_system_estimate=node_estimate(matrix, multiplicity), retained_systems=len(systems) - 1))
            provider_rows.append(dict(identity, metric=metric, estimate=point, ci95_lower=lower, ci95_upper=upper, physical_systems=len(systems), pairs=len(pair), pair_seasons=len(local), synchronous_combinations=int(local.sync_combinations.sum()), asynchronous_combinations=int(local.async_combinations.sum()), uncertainty='node_bootstrap_percentile_95' if provider == 'PreDist' else 'point_only_no_CI', valid_bootstrap_replicates=valid_replicates))
        guard.progress = identity
        print(f'Aggregated {identity}', flush=True)
    boot_archive.close()
    draws_sink.close()
    pd.concat(all_pair, ignore_index=True).to_csv(output / 'PAIR_SUMMARY.csv', index=False, float_format='%.17g')
    pd.DataFrame(all_system).to_csv(output / 'FOCAL_SYSTEM_SUMMARY.csv', index=False, float_format='%.17g')
    summary = pd.DataFrame(provider_rows)
    summary.to_csv(output / 'PROVIDER_SUMMARY.csv', index=False, float_format='%.17g')
    los = pd.DataFrame(los_rows)
    los.to_csv(output / 'XAI_LEAVE_ONE_SYSTEM.csv', index=False, float_format='%.17g')
    return (summary, los)
