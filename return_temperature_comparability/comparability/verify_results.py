'Independent reconstruction from exported identities, values and node draws.'
import argparse
import gzip
import hashlib
import io
import json
import resource
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
START = time.monotonic()
GROUP = ['provider','estimand','endpoint','duration_days']


def bounded():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == 'darwin' else rss * 1024
    if time.monotonic()-START > 3500 or rss > .95*4*1024**3:
        raise RuntimeError('Independent verification stopped before registered resource limit')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-11,equal_nan=True)


def finite_mean(a):
    a=np.asarray(a,float)
    return a[np.isfinite(a)].mean() if np.isfinite(a).any() else np.nan


def expanded_node_mean(matrix,labels):
    # Literal repeated-node graph: diagonal and any two copies of the same node
    # are excluded. This is independent of the production multiplicity formula.
    repeated=matrix[np.ix_(labels,labels)].copy()
    repeated[np.equal.outer(labels,labels)]=np.nan
    focal=np.array([finite_mean(row) for row in repeated])
    return finite_mean(focal)


def verify(results,output):
    output.mkdir(parents=True,exist_ok=False)
    checks={}
    manifest=json.loads((results/'HASH_MANIFEST.json').read_text())
    for record in manifest['files']:
        if digest(record['path'])!=record['sha256']:
            raise AssertionError(f'Hash mismatch: {record['path']}')
    checks['all_input_code_output_hashes']=len(manifest['files'])
    main=pd.read_csv(results/'PAIR_SEASON_SUMMARY.csv')
    disjoint=pd.read_csv(results/'DATE_DISJOINT_PAIR_SEASON_SUMMARY.csv')
    audit=pd.read_csv(results/'PAIR_SEASON_AUDIT.csv')
    frame=pd.concat([main,disjoint],ignore_index=True)
    provider=pd.read_csv(results/'PROVIDER_SUMMARY.csv')
    exported_pairs=pd.read_csv(results/'PAIR_SUMMARY.csv')
    exported_focal=pd.read_csv(results/'FOCAL_SYSTEM_SUMMARY.csv')
    los=pd.read_csv(results/'XAI_LEAVE_ONE_SYSTEM.csv')
    bootstrap=pd.read_csv(results/'NODE_BOOTSTRAP_ESTIMATES.csv.gz')
    matrix_archive=ZipFile(results/'FULL_CARTESIAN_VALUES.zip')
    indexed=main.set_index(['pair_season_id','endpoint','duration_days'])
    daily=pd.read_csv(results/'VALID_DAY_REPLAY.csv.gz')
    daily.date=pd.to_datetime(daily.date,utc=True)
    daily_lookup={key:rows.set_index('date').daily_median_return_c for key,rows in
                  daily.groupby(['provider','system','heating_year'])}
    windows=pd.read_csv(results/'DATE_DISJOINT_WINDOWS.csv.gz')
    windows_lookup={key:rows for key,rows in windows.groupby('pair_season_id',sort=False)}
    matrix_total=0
    disjoint_total=0
    for member in matrix_archive.namelist():
        bounded()
        identifier=Path(member).stem
        with np.load(io.BytesIO(matrix_archive.read(member)),allow_pickle=False) as saved:
            starts=pd.to_datetime(saved['starts'],utc=True)
            shared=pd.to_datetime(saved['shared_dates'],utc=True)
            assert len(shared)>=150
            assert len(starts)==len(set(starts))
            assert len(saved['endpoints'])==2
            for ei,endpoint in enumerate(saved['endpoints']):
                for di,duration in enumerate(saved['durations']):
                    row=indexed.loc[(identifier,endpoint,duration)]
                    a,b=saved['campaigns'][ei,di,:,0],saved['campaigns'][ei,di,:,1]
                    ri,rj=saved['references'][ei]
                    reference=ri-rj
                    contrast=a[:,None]-b[None,:]
                    errors=contrast-reference
                    good=np.isfinite(errors)
                    reversal=np.where(good & (reference!=0),(contrast*reference<0).astype(float),np.nan)
                    ties=np.where(good & (reference!=0),(contrast==0).astype(float),np.nan)
                    close(errors,saved['signed_contrast_errors'][ei,di])
                    close(np.abs(errors),saved['absolute_contrast_errors'][ei,di])
                    close(reversal,saved['reversals'][ei,di])
                    close(ties,saved['campaign_ties'][ei,di])
                    assert row.async_combinations==len(starts)**2
                    assert row.sync_combinations==len(starts)
                    for metric,value in {
                        'sync_mae_c':finite_mean(np.abs(np.diag(errors))),
                        'async_mae_c':finite_mean(np.abs(errors)),
                        'sync_minus_async_mae_c':finite_mean(np.abs(np.diag(errors)))-finite_mean(np.abs(errors)),
                        'sync_signed_error_c':finite_mean(np.diag(errors)),
                        'async_signed_error_c':finite_mean(errors),
                        'marginal_i_mae_c':finite_mean(np.abs(a-ri)),
                        'marginal_j_mae_c':finite_mean(np.abs(b-rj)),
                        'sync_reversal_rate':finite_mean(np.diag(reversal)),
                        'async_reversal_rate':finite_mean(reversal),
                        'sync_tie_rate':finite_mean(np.diag(ties)),
                        'async_tie_rate':finite_mean(ties),
                    }.items():
                        close(row[metric],value)
                    if good.all():
                        close(row.sync_signed_error_c,row.async_signed_error_c)
                    for start in starts:
                        assert ((shared>=start)&(shared<start+pd.Timedelta(days=int(duration)))).sum()>=np.ceil(.8*duration)
                    matrix_total+=errors.size
            row_identity=indexed.loc[(identifier,'median',7)]
            xi=daily_lookup[(row_identity.provider,row_identity.system_i,row_identity.heating_year)].loc[shared].to_numpy()
            xj=daily_lookup[(row_identity.provider,row_identity.system_j,row_identity.heating_year)].loc[shared].to_numpy()
            local=windows_lookup.get(identifier,pd.DataFrame())
            for row in local.itertuples(index=False):
                start=pd.Timestamp(row.window_start)
                complement=(shared<start)|(shared>=start+pd.Timedelta(days=90))
                window=(shared>=start)&(shared<start+pd.Timedelta(days=row.duration_days))
                assert not (complement & window).any()
                assert complement.sum()==row.complement_dates>=60
                values=[np.median(xi[window]),np.median(xj[window]),np.median(xi[complement]),np.median(xj[complement])]
                close(values,[row.campaign_i_c,row.campaign_j_c,row.reference_i_c,row.reference_j_c])
                err=values[0]-values[1]-values[2]+values[3]
                close(err,row.signed_error_c)
                close(abs(err),row.absolute_error_c)
                disjoint_total+=1
    matrix_archive.close()
    checks['cartesian_combinations_both_endpoints_all_durations']=matrix_total
    checks['date_disjoint_windows_all_values']=disjoint_total
    checks['eligible_pair_seasons']=int(audit.status.eq('eligible').sum())
    assert set(main.pair_season_id)==set(audit.loc[audit.status.eq('eligible'),'pair_season_id'])
    assert main.groupby('pair_season_id').size().eq(10).all()
    # Independent recomputation of daily-disjoint pair-season window means.
    disjoint_window_metrics={'sync_mae_c':'absolute_error_c','sync_signed_error_c':'signed_error_c',
        'marginal_i_mae_c':'marginal_i_error_c','marginal_j_mae_c':'marginal_j_error_c',
        'sync_reversal_rate':'reversal','sync_tie_rate':'campaign_tie'}
    for metric,column in disjoint_window_metrics.items():
        re=windows.groupby(['pair_season_id','duration_days'])[column].mean()
        reported=disjoint.set_index(['pair_season_id','duration_days'])[metric].reindex(re.index)
        close(re.to_numpy(),reported.to_numpy())
    node_archive=ZipFile(results/'NODE_BOOTSTRAP_DRAWS.zip')
    reconstructed=[]
    unequal_degree_differences=[]
    for group,local in frame.groupby(GROUP,sort=True):
        bounded()
        identity=dict(zip(GROUP,group))
        p=provider.loc[np.logical_and.reduce([provider[k].eq(v) for k,v in identity.items()])]
        for metric in p.metric:
            if metric=='marginal_focal_mae_c':
                pair=local.groupby(['system_i','system_j'])[['marginal_i_mae_c','marginal_j_mae_c']].mean()
            else:
                pair=local.groupby(['system_i','system_j'])[metric].mean().to_frame()
                ep=exported_pairs.loc[np.logical_and.reduce([exported_pairs[k].eq(v) for k,v in identity.items()])]
                close(pair[metric].to_numpy(),ep.set_index(['system_i','system_j'])[metric].reindex(pair.index).to_numpy())
            systems=sorted(set(pair.index.get_level_values(0))|set(pair.index.get_level_values(1)))
            indices={s:k for k,s in enumerate(systems)}
            matrix=np.full((len(systems),len(systems)),np.nan)
            for (a,b),values in pair.iterrows():
                if metric=='marginal_focal_mae_c':
                    matrix[indices[a],indices[b]]=values.marginal_i_mae_c
                    matrix[indices[b],indices[a]]=values.marginal_j_mae_c
                else:
                    matrix[indices[a],indices[b]]=matrix[indices[b],indices[a]]=values[metric]
            means=np.array([finite_mean(row) for row in matrix])
            point=finite_mean(means)
            focal=exported_focal.loc[np.logical_and.reduce([exported_focal[k].eq(v) for k,v in identity.items()])&
                                     exported_focal.metric.eq(metric)].set_index('system').reindex(systems)
            close(means,focal.estimate.to_numpy())
            close(np.isfinite(matrix).sum(axis=1),focal.eligible_counterparts.to_numpy())
            reported=p.loc[p.metric.eq(metric)].iloc[0]
            close(point,reported.estimate)
            reconstructed.append(dict(identity,metric=metric,estimate=point))
            unequal_degree_differences.append(abs(point-finite_mean(matrix)))
            if group[0]=='PreDist':
                draw_id=f'{group[1]}_{group[2]}_{group[3]}'
                with np.load(io.BytesIO(node_archive.read(draw_id+'_multiplicities.npz'))) as saved:
                    assert saved['systems'].tolist()==systems
                    counts=saved['multiplicities']
                    assert counts.shape==(5000,len(systems))
                    assert (counts.sum(axis=1)==len(systems)).all()
                    assert int(saved['seed'])==(2026100701 if group[1]=='same_season' else 2026100702)
                    rng=np.random.default_rng(int(saved['seed']))
                    np.testing.assert_array_equal(counts,rng.multinomial(len(systems),np.full(len(systems),1/len(systems)),size=5000))
                exported=bootstrap.loc[np.logical_and.reduce([bootstrap[k].eq(v) for k,v in identity.items()]) &
                                       bootstrap.metric.eq(metric)].sort_values('replicate')
                assert len(exported)==5000
                for replicate in [0,1,2,7,113,1309,2499,4999]:
                    labels=np.repeat(np.arange(len(systems)),counts[replicate])
                    close(expanded_node_mean(matrix,labels),exported.estimate.iloc[replicate])
                ci=np.nanquantile(exported.estimate,[.025,.975])
                close(ci,[reported.ci95_lower,reported.ci95_upper])
            else:
                assert np.isnan(reported.ci95_lower) and np.isnan(reported.ci95_upper)
                assert len(systems)==5
                retained=los.loc[np.logical_and.reduce([los[k].eq(v) for k,v in identity.items()]) & los.metric.eq(metric)]
                assert set(retained.omitted_system)==set(systems)
                for row in retained.itertuples(index=False):
                    labels=np.array([indices[s] for s in systems if s!=row.omitted_system])
                    close(expanded_node_mean(matrix,labels),row.leave_one_system_estimate)
    node_archive.close()
    pd.DataFrame(reconstructed).to_csv(output/'INDEPENDENT_PROVIDER_RECONSTRUCTION.csv',index=False,float_format='%.17g')
    checks['provider_estimates_reconstructed']=len(reconstructed)
    checks['max_node_vs_pair_pooling_difference']=max(unequal_degree_differences)
    # Test records and counts remain descriptive, never new thresholds.
    stats=[]
    for (p,e),local in main.groupby(['provider','endpoint']):
        stats.append(dict(provider=p,endpoint=e,pair_seasons=local.pair_season_id.nunique(),
            systems=len(set(local.system_i)|set(local.system_j)),
            zero_reference_pair_seasons=int(local.loc[local.duration_days.eq(7),'zero_reference'].sum()),
            missing_i_campaigns=int(local.missing_i_campaigns.sum()),
            missing_j_campaigns=int(local.missing_j_campaigns.sum()),
            synchronous_errors=int(local.sync_finite_errors.sum()),asynchronous_errors=int(local.async_finite_errors.sum()),
            synchronous_reversals=int(local.sync_reversals.sum()),asynchronous_reversals=int(local.async_reversals.sum()),
            synchronous_order_defined=int(local.sync_order_defined.sum()),asynchronous_order_defined=int(local.async_order_defined.sum())))
    pd.DataFrame(stats).to_csv(output/'MISSINGNESS_ORDERING_COUNTS.csv',index=False)
    endpoints=pd.read_csv(results/'CAMPAIGN_ENDPOINTS.csv.gz',usecols=['hours_i','hours_j'])
    checks['campaign_endpoint_rows_with_different_active_hour_counts']=int(endpoints.hours_i.ne(endpoints.hours_j).sum())
    checks['campaign_endpoint_rows']=len(endpoints)
    bounded()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    checks['elapsed_seconds']=time.monotonic()-START
    checks['peak_rss_bytes']=int(rss if sys.platform=='darwin' else rss*1024)
    checks['status']='PASS'
    (output/'VERIFICATION.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        raise ValueError('Output outside owned return_temperature directory')
    verify(args.results,args.output)
