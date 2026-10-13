'Read-only inputs and isolated frozen-method replay for this paper only.'
import argparse
import collections
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT/'reproduction'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def csvsave(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(frame,'to_csv'):frame.to_csv(path,index=False)
    elif frame:
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(frame[0]));w.writeheader();w.writerows(frame)

def load(name, injected=None):
    path=REP/'code'/(name+'.py')
    expected=next(x['export_sha256'] for x in read(REP/'CODE_PROVENANCE.json') if x['file']==path.name)
    if digest(path)!=expected:raise ValueError('Implementation hash mismatch: '+name)
    m=types.ModuleType('paper_replay_'+name);m.__file__=str(path)
    m.__dict__.update(injected or {})
    exec(compile(path.read_text(),str(path),'exec'),m.__dict__)
    return m

def required_artifacts(primary_only=False):
    names=read(REP/'RESULT_CONTRACT.json')['required_artifacts']
    if len(names)!=len(set(names)):raise ValueError('Duplicate expected artifact')
    declared={r['artifact'] for r in read(ROOT/'REPRODUCTION_SCOPE_MAP.json') if r['verification_mode']=='recomputed'}
    if set(names)!=declared:raise ValueError('Scope map and replay artifact contract differ')
    if primary_only:
        names=[n for n in names if 'SOURCE_COMPOSITION' not in n and
               ('_' not in n or n.startswith('results/'))]
    for name in names:
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or not name.startswith('results/'):
            raise ValueError('Invalid result path: '+name)
        if not (ROOT/path).is_file():raise FileNotFoundError('Required expected result missing: '+name)
    return names

def resolve_inputs(args):
    requirements=read(REP/'INPUT_REQUIREMENTS.json'); supplied={}
    if args.input_manifest:
        mf=args.input_manifest.resolve();obj=read(mf)
        if obj.get('paper')!=requirements['paper']:raise ValueError('Cross-paper input manifest')
        for row in obj['inputs']:
            if row['id'] in supplied:raise ValueError('Duplicate input identity: '+row['id'])
            p=Path(row['path']);supplied[row['id']]=(mf.parent/p).resolve() if not p.is_absolute() else p
    else:
        base=args.source_workspace.resolve()
        # Compatibility only: discover the original carriers by file identity, never execute old code.
        byname=collections.defaultdict(list)
        for p in base.rglob('*'):
            if p.is_file():byname[p.name].append(p)
        hit=base/'original_source_records/ACTUAL_HITS.csv'
        if hit.is_file():
            for row in csv.DictReader(hit.open()):
                p=Path(row['path'])
                if p.is_file():
                    byname[p.name].append(p)
                    for q in p.parent.parent.rglob('*.csv*'):byname[q.name].append(q)
        for row in requirements['inputs']:
            if 'bundled_path' not in row:
                supplied[row['id']]=next((p for p in byname[row['filename']] if digest(p)==row['sha256']),None)
    allowed={r['id'] for r in requirements['inputs'] if 'bundled_path' not in r}
    if set(supplied)-allowed:raise ValueError('Undeclared external input IDs: '+str(sorted(set(supplied)-allowed)))
    resolved={}
    for row in requirements['inputs']:
        p=ROOT/row['bundled_path'] if 'bundled_path' in row else supplied.get(row['id'])
        if p is None or not p.is_file():raise FileNotFoundError('NOT EXECUTED: missing lawful input '+row['id'])
        if digest(p)!=row['sha256']:raise ValueError('Input version/hash mismatch: '+row['id'])
        resolved[row['id']]=p.resolve()
    return requirements,resolved

def compare_outputs(actual_root,names):
    import pandas as pd
    results=[]
    for name in names:
        expected=ROOT/name;actual=actual_root/Path(name).name
        if not expected.is_file() or not actual.is_file():raise FileNotFoundError('Required artifact absent: '+name)
        if expected.suffix in {'.csv','.gz'}:
            a=pd.read_csv(expected);b=pd.read_csv(actual)
            pd.testing.assert_frame_equal(a,b,check_exact=True)
        elif read(expected)!=read(actual):raise ValueError('JSON result mismatch: '+name)
        results.append({'artifact':name,'expected_sha256':digest(expected),'actual_sha256':digest(actual),'serialized_cells_equal':True})
    return results

def return_temperature(temp,inputs,primary_only=False):
    import pandas as pd
    work=temp/'work';monitoring_analysis=work/'original_source_records/monitoring_analysis';raw=monitoring_analysis/'raw';raw.mkdir(parents=True)
    for key,p in inputs.items():
        if key.startswith('heatwise_'):(raw/key[len('heatwise_'):]).symlink_to(p)
    prior=work/'original_source_records/RESULTS_TABLES.csv'
    prior.parent.mkdir(parents=True);prior.symlink_to(inputs['prior_provider_results'])
    m=load('heatwise',dict(RAW=raw,ROOT=monitoring_analysis,ACCESS_LOG=temp/'read_access.csv'))
    a,aa=m.load_aau();b,bb=m.load_nest()
    hourly=pd.concat([a,b],ignore_index=True).sort_values(['provider','system','hour'])
    windows,periods=m.external_windows(hourly)
    estimates,contrasts=m.provider_duration_evidence(windows)
    combined,contrasts_all,summary=m.combine_with_provider_results(estimates,contrasts)
    out=temp/'results';out.mkdir()
    for name,f in [('HEATWISE_PERIOD_ELIGIBILITY',periods),('HEATWISE_FIELD_AUDIT',pd.DataFrame(aa+bb)),
                   ('HEATWISE_DURATION_EVIDENCE',m.annotate_endpoint(estimates)),('HEATWISE_DURATION_CONTRASTS',m.annotate_endpoint(contrasts)),
                   ('CROSS_ECOSYSTEM_DURATION_SUMMARY',combined),('CROSS_ECOSYSTEM_CONTRASTS',contrasts_all),('PROVIDER_EQUAL_SUMMARY',summary)]:
        csvsave(out/(''+name+'.csv'),f)
    if primary_only:return out,{'scope':'HEATWISE raw replay plus prior-provider summaries','new_model_fits':0}
    hp=temp/'HEATWISE_HOURLY.csv.gz';hourly.to_csv(hp,index=False,compression={'method':'gzip','mtime':0})
    t=load('temporal',dict(PREDIST=inputs['predist'],XAI=inputs['xai'],HEATWISE_HOURLY=hp,RESULTS=out))
    daily=t.build_daily_medians(pd.concat([t.load_predist_hourly(),t.load_xai_hourly(),t.load_heatwise_hourly()],ignore_index=True))
    dp=out/'VALID_DAY_MEDIANS.csv.gz';daily.to_csv(dp,index=False,compression={'method':'gzip','mtime':0})
    windows=t.build_equal_day_windows(daily);windows.to_csv(out/'EQUAL_DAY_WINDOWS.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    duration,contrasts=t.summarize_equal_day(windows)
    t.write_csv(out/'EQUAL_DAY_DURATION_SUMMARY.csv',duration);t.write_csv(out/'EQUAL_DAY_CONTRASTS.csv',contrasts)
    formal=contrasts.loc[contrasts.provider.isin(['PreDist','XAI4HEAT'])]
    if len(formal)!=4 or not formal.contrast_c.ge(0).all():raise ValueError('Original temporal branch no longer holds')
    daily=pd.read_csv(dp);daily['date']=pd.to_datetime(daily.date,errors='raise',utc=True)
    structural,qualified=t.structural_nonoverlap_gate(daily)
    if len(qualified)<2:raise ValueError('Original structural qualification no longer holds')
    t.write_csv(out/'NONOVERLAP_STRUCTURAL_GATE.csv',structural)
    w=t.build_nonoverlap_windows(daily,qualified);w.to_csv(out/'NONOVERLAP_WINDOWS.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    t.write_csv(out/'NONOVERLAP_CONTRASTS.csv',t.summarize_nonoverlap(w))
    return out,{'scope':'HEATWISE raw replay plus prior-provider summary combination and temporal replays','new_model_fits':0}


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input-manifest',type=Path);g.add_argument('--source-workspace',type=Path)
    p.add_argument('--primary-only',action='store_true');p.add_argument('--output-dir',type=Path)
    args=p.parse_args();names=required_artifacts(args.primary_only);req,inputs=resolve_inputs(args)
    if req['paper']!='return_temperature_comparability':raise ValueError('Cross-paper replay configuration')
    import importlib.metadata
    actual={k:importlib.metadata.version(k) for k in req['software'] if k!='python'}
    if any(actual[k]!=v for k,v in req['software'].items() if k!='python'):
        raise RuntimeError('Use the locked software versions: '+json.dumps({'expected':req['software'],'actual':actual}))
    if '.'.join(map(str,sys.version_info[:3]))!=req['software']['python']:
        raise RuntimeError('Use the locked Python version: '+req['software']['python'])
    if args.output_dir and (args.output_dir.exists() and any(args.output_dir.iterdir())):raise ValueError('Output directory must be empty')
    if args.output_dir:
        target=args.output_dir.resolve()
        if ROOT==target or ROOT in target.parents or any(p==target or target in p.parents for p in inputs.values()):
            raise ValueError('Output must be separate from the package and input carriers')
    before={str(p):digest(p) for p in inputs.values()}
    with tempfile.TemporaryDirectory(prefix='paper_portable_') as name:
        temp=Path(name)
        out,diagnostics=return_temperature(temp,inputs,args.primary_only)
        comparisons=compare_outputs(out,names)
        if before!={str(p):digest(p) for p in inputs.values()}:raise ValueError('Input changed during replay')
        report={'paper':req['paper'],'status':'PASS','comparisons':comparisons,'diagnostics':diagnostics,
                'inputs_read_only':True,'required_artifact_count':len(names)}
        if args.output_dir:
            args.output_dir.mkdir(parents=True,exist_ok=True)
            for n in names:shutil.copyfile(out/Path(n).name,args.output_dir/Path(n).name)
            save(args.output_dir/'REPLAY_RESULT.json',report)
        print(json.dumps(report,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'status':'NOT_EXECUTED' if isinstance(exc,FileNotFoundError) else 'FAIL','reason':str(exc)},indent=2),file=sys.stderr)
        raise SystemExit(2)
