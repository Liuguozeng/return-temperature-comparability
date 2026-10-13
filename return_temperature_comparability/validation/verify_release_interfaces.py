'Check the exposed paired-MAE arithmetic against retained relative results.'
from pathlib import Path
import csv
import json
import math
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def verify(root=ROOT):
    added=rows(root/'supporting_tables/S6B_PAIRED_ABSOLUTE_MAE.csv')
    frozen=rows(root/'supporting_tables/S6_SAME_PERIOD_RELATIVE_CONTRASTS.csv')
    key=lambda r:(r['split'],r['short_days'],r['long_days'])
    original={key(r):r for r in frozen}
    assert len(added)==len({key(r) for r in added})==4
    assert {key(r) for r in added}==set(original)
    for r in added:
        assert float(r['long_mae_C'])>0
        rebuilt=100*(float(r['short_mae_C'])-float(r['long_mae_C']))/float(r['long_mae_C'])
        assert math.isclose(rebuilt,float(r['frozen_percent']),rel_tol=0,abs_tol=1e-12)
        assert math.isclose(rebuilt,100*float(original[key(r)]['relative_temperature_mae_difference_point']),rel_tol=0,abs_tol=1e-12)
        assert float(r['ratio_difference'])==0
        assert int(r['paired_building_periods'])==(32 if r['split']=='chronology_holdout' else 126)
    index=root/'supporting_tables/ACTUAL_INCLUDED_SEASON_INDEX.csv'
    seasons=rows(index)
    assert len(seasons)==len({(r['provider'],r['unit'],r['season']) for r in seasons})==6503
    assert Counter(r['provider'] for r in seasons)==Counter({'Development':268,'PreDist':199,'XAI4HEAT':10,'Aalborg':6022,'AAU':2,'Empa/NEST':2})
    assert all('/Volumes/' not in r['source_artifact'] and '/Users/' not in r['source_artifact'] for r in seasons)
    return {'status':'PASS','paired_contrasts':4,'arithmetic_tolerance':1e-12,
            'unit_season_index_rows':len(rows(index)),'not_new_confidence_intervals':True}


if __name__=='__main__':print(json.dumps(verify(),indent=2))
