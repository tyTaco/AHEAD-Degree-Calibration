import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_preprocessed_calibration as api
from degree_development import batch_completion_times, chronological_development_split
from train_research import file_sha256


class CalibrationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ahead-calibration-test-')
        self.project = Path(self.temp.name)
        for name in set(api.ALGORITHM_FILES + api.VERIFIER_FILES):
            shutil.copyfile(ROOT / name, self.project / name)
        self.root = self.project / 'research_runs' / 'newrun'
        self.dev = self.root / 'degree_development'
        self.dev.mkdir(parents=True)
        self.data = self.project / 'processed_data' / 'mooc'
        self.data.mkdir(parents=True)
        self.frame = pd.DataFrame({'u':[1]*100, 'i':[2]*100, 'ts':np.arange(1.,101.),
                                   'label':[i%2 for i in range(100)], 'idx':np.arange(1,101)})
        self.frame.to_csv(self.data / 'ml_mooc.csv', index=False)
        (self.root / 'provenance.json').write_text('{}')
        (self.root / 'selection_complete.json').write_text('{}')
        self.parts = chronological_development_split(self.frame.ts.to_numpy())
        self.protocol = {'dataset':'mooc','run_id':'newrun','batch_size':2,
                         'input_provenance_sha256':file_sha256(self.root / 'provenance.json'),
                         'selection_seal_sha256':file_sha256(self.root / 'selection_complete.json'),
                         'segments':{key:[s.start,s.stop] for key,s in self.parts.items()}, 'development_code_sha256':{}}
        (self.dev / 'development_started.json').write_text(json.dumps(self.protocol))
        self.artifacts = {}
        self.policies = {}
        for policy in api.POLICIES:
            self.policies[policy] = {'checkpoint_sha256':f'checkpoint-{policy}', 'epoch':1}
            metric = self.dev / f'{policy}_seed0_metrics.json'
            metric.write_text(json.dumps({'seed':0,'policy':policy,'selected_epoch':1,
                                          'checkpoint_sha256':f'checkpoint-{policy}'}))
            self.artifacts[metric.name] = file_sha256(metric)
            c, d = self.parts['reference'], self.parts['development']
            rc, rd = self.frame.iloc[c], self.frame.iloc[d]
            scores = np.linspace(.1,.9,len(rd))
            degrees = np.arange(len(rd)) % 5
            path = self.dev / f'{policy}_seed0_scores.npz'
            np.savez(path, reference_times=rc.ts.to_numpy(), reference_event_ids=rc.idx.to_numpy(),
                     reference_score_available_at=batch_completion_times(rc.ts.to_numpy(),2),
                     reference_scores=np.linspace(.1,.9,len(rc)), reference_degrees=np.arange(len(rc))%5,
                     development_times=rd.ts.to_numpy(), development_event_ids=rd.idx.to_numpy(),
                     development_score_available_at=batch_completion_times(rd.ts.to_numpy(),2),
                     none=scores, development_degrees=degrees, original=scores*np.where(degrees<3,.9,1.),
                     continuous=scores*(1-.1*np.exp(-degrees/3.)))
            self.artifacts[path.name] = file_sha256(path)
        self.update_seal()
        self.options={'fixed_multiplier':1.1,'block_size':2,'replicates':20,'bootstrap_seed':7}
        self.runs=[{'seed':0,'policies':self.policies}]
        self.prov={'dataset':'mooc'}

    def update_seal(self):
        p=self.dev/'development_complete.json'
        p.write_text(json.dumps({'num_runs':1, 'artifact_sha256':self.artifacts,
                                 'protocol_sha256':file_sha256(self.dev/'development_started.json')}))
        self.seal=file_sha256(p)

    def tearDown(self):
        self.temp.cleanup()

    def run_eval(self, methods=api.METHODS, policies=api.POLICIES, **kwargs):
        with mock.patch.object(api, '_verify_run', return_value=(self.root,self.prov,self.runs)), \
             mock.patch.object(api, 'verified_integration', return_value=None):
            return api.evaluate(self.project,'newrun','mooc',[0],policies,methods,'custom',
                                self.options,self.seal,**kwargs)

    def test_all_methods_both_policies_and_event_ids(self):
        result=self.run_eval()
        self.assertEqual(len(result['rows']),2)
        for row in result['rows']:
            self.assertEqual(set(row['metrics']),set(api.METHODS))
            self.assertEqual(row['alignment'],'EVENT_IDS_MATCH')
            self.assertEqual(set(row['metrics']['none']), {'AUROC','AP','PR_AUC'})
            self.assertIsNotNone(row['c_pairwise_selection'])
            self.assertIsNotNone(row['c_guarded_selection'])
        one=self.run_eval(('none',), ('train_loss',))
        self.assertEqual(set(one['rows'][0]['metrics']),{'none'})
        self.assertIsNone(one['rows'][0]['c_pairwise_selection'])
        multi=self.run_eval(('original','fixed_1_1'), ('val_auroc',))
        self.assertEqual(set(multi['rows'][0]['metrics']),{'original','fixed_1_1'})

    def test_methods_and_selection_tie_and_guard_fallback(self):
        score=np.array([.1,.2,.3,.4]); deg=np.array([3,3,3,3]); labels=np.array([0,1,0,1])
        pair,guard=api.choose_methods(score,deg,labels,api.METHODS,self.options)
        self.assertEqual(pair['factor'],1.)
        self.assertEqual(guard['factor'],1.)
        values=api.apply_methods(score,deg,api.METHODS,pair,guard,1.1)
        for a in values.values(): np.testing.assert_allclose(a, score if a is not values['continuous'] else score*(1-.1*np.exp(-deg/3.)))
        low=np.array([0,1,3,4]); values=api.apply_methods(score,low,api.METHODS,pair,guard,1.2)
        np.testing.assert_allclose(values['fixed_1_1'],score*np.where(low<3,1.2,1.))
        guard=api.choose_guarded_factor(score,low,np.zeros(4),block_size=2,replicates=20,seed=7)
        self.assertEqual(guard['reason'],'single_class_fallback')

    def test_tamper_and_missing_event_identity(self):
        p=self.dev/'val_auroc_seed0_scores.npz'
        with p.open('ab') as out: out.write(b'X')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): self.run_eval()
        p.unlink()
        with self.assertRaisesRegex(ValueError,'Missing'): self.run_eval()

    def test_wrong_ids_rejected_even_if_seal_updated(self):
        p=self.dev/'val_auroc_seed0_scores.npz'
        with np.load(p) as z: values={name:z[name] for name in z.files}
        values['reference_event_ids']=values['reference_event_ids'][::-1]
        np.savez(p,**values); self.artifacts[p.name]=file_sha256(p); self.update_seal()
        with self.assertRaisesRegex(ValueError,'identifiers'): self.run_eval()

    def test_legacy_requires_explicit_opt_in(self):
        for policy in api.POLICIES:
            p=self.dev/f'{policy}_seed0_scores.npz'
            with np.load(p) as z: values={name:z[name] for name in z.files if not name.endswith('_event_ids')}
            np.savez(p,**values); self.artifacts[p.name]=file_sha256(p)
        self.update_seal()
        with self.assertRaisesRegex(ValueError,'lack event IDs'): self.run_eval()
        result=self.run_eval(allow_legacy=True)
        self.assertEqual(result['rows'][0]['alignment'],'LEGACY_TIMESTAMPS_ONLY_EVENT_ID_UNVERIFIED')

    def test_dry_run_cannot_be_completed(self):
        project=self.project/'project'; project.mkdir()
        (self.project/'integration_start.json').write_text(json.dumps({'status':'DRY-RUN'}))
        with self.assertRaisesRegex(ValueError,'Missing'): api.verified_integration(project,'newrun',True)

    def test_phase31_integration_seal_mock(self):
        workspace = self.project / 'project'
        destination = workspace / 'processed_data/mooc'
        destination.mkdir(parents=True)
        training = workspace / 'research_runs/newrun'
        training.mkdir(parents=True)
        seal = training / 'selection_complete.json'
        seal.write_text('{}')
        source = self.project / 'raw.csv'
        source.write_text('raw')
        data = destination / 'ml_mooc.csv'
        data.write_text('synthetic')
        manifest = destination / 'reproducibility_manifest.json'
        record = {'dataset':'mooc','mode':'custom','profile_sha256':'profile', 'source_sha256':'source',
                  'artifacts':{'ml_mooc.csv':file_sha256(data)}, 'status':'CUSTOM-VALID'}
        manifest.write_text(json.dumps(record))
        start = self.project / 'integration_start.json'
        info = {'status':'RUNNING','run_id':'newrun','workspace':str(workspace), 'dataset':'mooc',
                'data_mode':'custom','profile_sha256':'profile','source_sha256':'source',
                'inputs':record['artifacts'],'source':str(source),'manifest_sha256':file_sha256(manifest), 'code_sha256':{}}
        start.write_text(json.dumps(info))
        (self.project / 'integration_complete.json').write_text(json.dumps({'status':'COMPLETED',
            'start_sha256':file_sha256(start),'selection_sha256':file_sha256(seal)}))
        with mock.patch.object(api.prep,'verify_manifest',return_value=record) as verify:
            self.assertEqual(api.verified_integration(workspace,'newrun',True),info)
            verify.assert_called_once()
            data.write_text('changed')
            with self.assertRaisesRegex(ValueError,'data hash mismatch'):
                api.verified_integration(workspace,'newrun',True)

    def test_cli_writes_one_seed_without_training(self):
        output=self.project/'new_result.json'
        args=['--project-root',str(self.project),'--run-id','newrun','--dataset','mooc','--seeds','0',
              '--policies','val_auroc','--methods','none','original','--development-seal-sha256',self.seal,
              '--output',str(output)]
        with mock.patch.object(api,'_verify_run',return_value=(self.root,self.prov,self.runs)), \
             mock.patch.object(api,'verified_integration',return_value=None):
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(api.main(args),0)
        result=json.loads(output.read_text())
        self.assertEqual(result['rows'][0]['seed'],0)
        self.assertEqual(set(result['rows'][0]['metrics']),{'none','original'})
        self.assertEqual(result['status'],'COMPLETED_SCORES_ONLY')

    def test_seal_and_duplicate_output_protection(self):
        with mock.patch.object(api,'_verify_run',return_value=(self.root,self.prov,self.runs)), \
             mock.patch.object(api,'verified_integration',return_value=None):
            with self.assertRaisesRegex(ValueError,'seal SHA-256'):
                api.evaluate(self.project,'newrun','mooc',[0],['val_auroc'],['none'],'historical',
                             self.options,'0'*64)
        existing=self.project/'result.json'; existing.write_text('original')
        args=['--project-root',str(self.project),'--run-id','newrun','--dataset','mooc','--seeds','0',
              '--policies','val_auroc','--methods','none','--development-seal-sha256',self.seal,'--output',str(existing)]
        with contextlib.redirect_stderr(io.StringIO()): self.assertEqual(api.main(args),1)
        self.assertEqual(existing.read_text(),'original')

if __name__ == '__main__': unittest.main()
