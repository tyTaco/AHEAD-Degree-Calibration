import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import score_preprocessed_research as scoring
import evaluate_preprocessed_calibration as evaluation
from degree_development import chronological_development_split
from train_research import file_sha256


class FakeModel:
    calls=[]
    def __init__(self): self.memory_generator=SimpleNamespace(grad_mask_dict='before')
    def to(self, device): return self
    def set_neighbor_sampler(self, sampler): pass
    def eval(self): return self
    def load_state_dict(self, state, strict=True): pass
    def _get_temporal_degree_from_sampler(self, node_ids, node_interact_times, num_neighbors, time_gap):
        return np.asarray(node_ids)%4
    def compute_anomaly_score(self, src_node_ids, dst_node_ids, node_interact_times, edge_ids, num_neighbors, time_gap):
        self.calls.append(np.array(edge_ids,copy=True))
        self._get_temporal_degree_from_sampler(src_node_ids,node_interact_times,num_neighbors,time_gap)
        return torch.as_tensor(np.asarray(edge_ids,dtype=np.float32)/100.)


class ScoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='ahead-score-integration-')
        self.project=Path(self.tmp.name)/'project';self.project.mkdir()
        files=set(evaluation.ALGORITHM_FILES+evaluation.VERIFIER_FILES+('run_degree_development.py','degree_ecdf_prototype.py'))
        for name in files: shutil.copyfile(ROOT/name,self.project/name)
        self.root=self.project/'research_runs'/'newrun';self.root.mkdir(parents=True)
        self.data=self.project/'processed_data'/'mooc';self.data.mkdir(parents=True)
        self.frame=pd.DataFrame({'u':np.arange(100)%3+1,'i':np.arange(100)%2+4,
            'ts':np.repeat(np.arange(1.,51.),2),'label':np.arange(100)%2,'idx':np.arange(1,101)})
        self.frame.to_csv(self.data/'ml_mooc.csv',index=False)
        np.save(self.data/'ml_mooc.npy',np.zeros((101,172),dtype=np.float32))
        np.save(self.data/'ml_mooc_node.npy',np.zeros((6,172),dtype=np.float32))
        (self.root/'provenance.json').write_text('{}')
        (self.root/'selection_complete.json').write_text('{}')
        folder=self.root/'seed0';folder.mkdir()
        policies={}
        for name in evaluation.POLICIES:
            ck=folder/f'{name}_epoch0001.pkl';ck.write_bytes(b'fake checkpoint for dispatch testing')
            policies[name]={'checkpoint':ck.name,'checkpoint_sha256':file_sha256(ck),'epoch':1}
        self.runs=[{'seed':0,'policies':policies}]
        self.config={'dataset_name':'mooc','val_ratio':.05,'test_ratio':.5,'num_neighbors':20,
            'sample_neighbor_strategy':'recent','batch_size':2,'time_gap':2000,'time_scaling_factor':1e-6}
        self.prov={'dataset':'mooc','configuration':self.config,'calibration_mode':'original'}
        self.integration={'dataset':'mooc','configuration':self.config,'calibration':'original',
            'data_mode':'custom','experiment_mode':'custom','manifest_sha256':'fake'}
        FakeModel.calls=[]

    def tearDown(self): self.tmp.cleanup()

    def test_mock_cpu_scoring_seal_event_ids_and_six_method_integration(self):
        with mock.patch.object(scoring,'verified_integration',return_value=self.integration), \
             mock.patch.object(evaluation,'verified_integration',return_value=self.integration), \
             mock.patch.object(scoring,'_verify_run',return_value=(self.root,self.prov,self.runs)), \
             mock.patch.object(evaluation,'_verify_run',return_value=(self.root,self.prov,self.runs)), \
             mock.patch.object(scoring,'get_neighbor_sampler',return_value=object()), \
             mock.patch.object(scoring,'build_research_model',side_effect=lambda *a:FakeModel()), \
             mock.patch.object(scoring.torch,'load',return_value={}):
            seal=scoring.score(self.project,'newrun',[0],evaluation.POLICIES,confirmed=True)
            opts={'fixed_multiplier':1.1,'block_size':2,'replicates':20,'bootstrap_seed':7}
            result=evaluation.evaluate(self.project,'newrun','mooc',[0],evaluation.POLICIES,
                                       evaluation.METHODS,'custom',opts,seal,require_integration=True)
            self.assertEqual(len(result['rows']),2)
            self.assertTrue(all(r['alignment']=='EVENT_IDS_MATCH' for r in result['rows']))
            self.assertTrue(all(set(r['metrics'])==set(evaluation.METHODS) for r in result['rows']))
            parts=chronological_development_split(self.frame.ts.to_numpy())
            b,c,d=(parts[name] for name in ('selection','reference','development'))
            calls=np.concatenate(FakeModel.calls)
            expected=self.frame.idx.to_numpy()[b.start:d.stop]
            np.testing.assert_array_equal(calls,np.tile(expected,2))
            for policy in evaluation.POLICIES:
                with np.load(self.root/'degree_development'/f'{policy}_seed0_scores.npz') as z:
                    np.testing.assert_array_equal(z['reference_event_ids'],self.frame.idx.to_numpy()[c])
                    np.testing.assert_array_equal(z['development_event_ids'],self.frame.idx.to_numpy()[d])
                    self.assertLess(len(np.unique(z['reference_times'])),len(z['reference_times']))
            with self.assertRaises(FileExistsError): scoring.score(self.project,'newrun',[0],evaluation.POLICIES,confirmed=True)

    def test_missing_confirmation_and_original_project_rejected(self):
        with self.assertRaisesRegex(ValueError,'confirm-scores'):
            scoring.score(self.project,'newrun',[0],['val_auroc'])
        with self.assertRaisesRegex(ValueError,'original research'):
            scoring.score(ROOT,'newrun',[0],['val_auroc'],confirmed=True)

if __name__=='__main__': unittest.main()
