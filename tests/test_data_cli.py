import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from waternilm.__main__ import main
from waternilm.data import load_series, prepare_ampds, write_rows


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, header, rows):
        path = self.root / name
        with path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_align_labels_by_timestamp_not_row_offset(self):
        e = self.write('e.csv', ['TIMESTAMP','I','S'], [[60,0,0],[120,1,20],[180,0,0]])
        w = self.write('w.csv', ['unix_ts','avg_rate'], [[60,0],[120,1],[180,0]])
        labels = self.write('labels.csv', ['unix_ts','avg_rate'], [[0,0],[60,0],[120,.5],[180,0]])
        result = prepare_ampds(e,w,boundaries=[.5], start=120,end=180,labels_path=labels)
        self.assertEqual(result, [{'timestamp':120, 'electrical_state':1, 'water_lpm':1., 'active':1, 'truth_lpm':.5}])
        write_rows(self.root/'prepared.csv',result)
        self.assertEqual(load_series(self.root/'prepared.csv').truth,[.5])

    def test_timestamp_and_value_rejections(self):
        header=['timestamp','electrical_state','water_lpm','active']
        for rows in ([[0,0,0,0],[120,1,.5,1]], [[0,0,0,0],[0,1,.5,1]], [[0,1,.5,0]], [[0,0,'nan',0]], [[0,.5,0,1]], [[0,0,0,2]]):
            path=self.write('bad.csv',header,rows)
            with self.assertRaises(ValueError):
                load_series(path)
        path=self.write('offgrid.csv',header,[[0,1,.4,1]])
        with self.assertRaises(ValueError):
            load_series(path).caps(.5)

    def test_missing_meter_timestamp_rejected(self):
        e=self.write('e.csv',['timestamp','i','s'],[[0,1,10],[60,1,10]])
        w=self.write('w.csv',['timestamp','avg_rate'],[[60,.5]])
        with self.assertRaisesRegex(ValueError,'missing'):
            prepare_ampds(e,w,boundaries=[.5])

    def test_cli_roundtrip_metrics_and_leakage_guard(self):
        examples=Path(__file__).resolve().parents[1]/'examples'
        model=self.root/'model.json'
        result=self.root/'prediction.csv'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['train',str(examples/'train.csv'),'--model',str(model),'--num-observed','3']),0)
        for decoder in ('exact','legacy'):
            output=self.root/f'{decoder}.csv'
            report=io.StringIO()
            with contextlib.redirect_stdout(report):
                self.assertEqual(main(['predict',str(examples/'test.csv'),'--model',str(model),'--output',str(output),'--decoder',decoder]),0)
            self.assertEqual(json.loads(report.getvalue())['all_minutes']['mse_lpm2'],0)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['predict',str(examples/'train.csv'),'--model',str(model),'--output',str(result)]),2)
            self.assertEqual(main(['train',str(examples/'train.csv'),'--model',str(model),'--num-observed','3']),2)
        self.assertFalse(result.exists())


if __name__ == '__main__':
    unittest.main()
