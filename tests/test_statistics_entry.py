"""Small public tests of statistics and historical summary compatibility."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import summarize_calibration_results as stats

class StatisticsEntryTests(unittest.TestCase):
    def test_historical_saved_report(self):
        report=ROOT/'src/research_runs/fixed_degree_up_exploration_20261123/report.json'
        groups,_=stats.normalize([report]); result=stats.summarize(groups)
        self.assertEqual(len(result['aggregates']),144)
        self.assertEqual(sum(x['candidate']=='fixed_1_1' for x in result['paired_differences']),120)
        item=next(x for x in result['paired_differences'] if x['dataset']=='digg' and x['policy']=='val_auroc'
                  and x['candidate']=='fixed_1_1' and x['baseline']=='original' and x['metric']=='AUROC')
        self.assertAlmostEqual(item['mean'],.063857,places=6)

    def test_single_seed_missing_comparison_and_duplicate(self):
        context={'dataset':'synthetic','run_id':'run','format':'test'}
        value={k:.5 for k in stats.METRICS}
        groups={'synthetic':{'context':context,'sources':set(),'rows':{
            ('val_auroc',0):{'provenance':{},'metrics':{'fixed_1_1':value}}}}}
        result=stats.summarize(groups,policies=['val_auroc'],methods=['fixed_1_1','none'])
        self.assertIsNone(result['aggregates'][0]['sample_sd'])
        self.assertEqual(next(x for x in result['paired_differences'] if x['baseline']=='none')['status'],'METHOD_OR_POLICY_MISSING')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text('{"dataset":"a","dataset":"b"}')
            with self.assertRaises(ValueError):stats.normalize([p])

if __name__=='__main__':unittest.main()
