import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from analysis import ROOT, normalize_frozen_hourly, safe_functions


class VendoredReaders(unittest.TestCase):
    def test_ast_exact_original_and_no_module_side_effects(self):
        provenance=json.loads((ROOT/'vendor'/'PROVENANCE.json').read_text())
        for row in provenance['readers']:
            snapshot=ast.parse((ROOT/row['packaged_path']).read_text())
            for n in snapshot.body:
                if isinstance(n,ast.FunctionDef):
                    canonical=ast.dump(n,include_attributes=False)
                    self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(),
                                     row['canonical_function_ast_sha256'][n.name])
                    self.assertNotEqual(n.name,'log_access')
                else:
                    self.assertTrue(isinstance(n,(ast.Import,ast.ImportFrom)) or
                                    isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant))

    def test_predist_parsing_units_activity_and_hourly_weights(self):
        text=('timestamp;outdoor_temperature;p_net_return_temperature;p_net_meter_heat_power;p_net_meter_flow\n'
              '2022-10-01 01:01:00;10;30;1000;1000\n'
              '2022-10-01 01:40:00;10;50;3000;1000\n'
              '2022-10-01 02:00:00;10;60;bad;1000\n'
              '2022-10-01 03:00:00;10;70;0;1000\n'
              'invalid;10;50;2000;1000\n'
              '2022-10-01 04:00:00;10;140;2000;1000\n')
        namespace=dict(np=np,pd=pd,Path=Path,ZipFile=ZipFile)
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            path=Path(directory)/'raw.zip'
            name='manufacturer 2/operational_data/substation_1.csv'
            with ZipFile(path,'w') as archive:
                archive.writestr(name,text)
            with ZipFile(path) as archive:
                reader=safe_functions(ROOT/'vendor'/'weighted.py',['_hourly_from_predist_member'],namespace)
                rows=normalize_frozen_hourly(reader['_hourly_from_predist_member'](archive,name))
            self.assertEqual(rows.return_c.tolist(),[40.,60.])
            self.assertEqual(rows.heat_kwh.iloc[0],2.)
            self.assertTrue(np.isnan(rows.heat_kwh.iloc[1]))
            self.assertEqual(rows.system.iloc[0],'manufacturer_2:substation_1')

    def test_xai_positive_delta_e_hourly_sum_and_schema(self):
        text=('datetime,t_amb,t_ret_prim,delta_e\n'
              '2022-10-01 01:00:00,10,30,1\n'
              '2022-10-01 01:30:00,10,50,3\n'
              '2022-10-01 02:00:00,10,80,0\n'
              'invalid,10,80,2\n')
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            path=Path(directory)/'raw.zip'
            with ZipFile(path,'w') as archive:
                for name in ['L4','L8','L12','L17','L22']:
                    archive.writestr('xai4heat_scada_'+name+'_processed.csv',text)
            namespace=dict(np=np,pd=pd,Path=Path,ZipFile=ZipFile,XAI=path)
            reader=safe_functions(ROOT/'vendor'/'weighted.py',['load_xai_hourly'],namespace)
            rows=normalize_frozen_hourly(reader['load_xai_hourly']())
            self.assertEqual(len(rows),5)
            self.assertTrue(rows.return_c.eq(40).all())
            self.assertTrue(rows.heat_kwh.eq(4).all())
            self.assertEqual(set(rows.system),{'L4','L8','L12','L17','L22'})
            with ZipFile(path,'w') as archive:
                archive.writestr('xai4heat_scada_L4_processed.csv','datetime,t_ret_prim\n')
            with self.assertRaises(RuntimeError):
                reader['load_xai_hourly']()


if __name__=='__main__':
    unittest.main()
