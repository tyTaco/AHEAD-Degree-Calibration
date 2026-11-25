import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare_reproducible as prep


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ahead-preprocess-test-')
        self.base = Path(self.temp.name)
        self.source = self.base / 'events.csv'
        self.source.write_text('user,item,timestamp,state_label,features\n'
                               'u1,i1,1,0,0.5,1.5\n'
                               'u2,i1,2,1,0.6,1.6\n'
                               'u1,i2,2,0,0.7,1.7\n', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def run_custom(self, name, edge='F'):
        return prep.convert('mooc', 'custom', self.source, self.base / name, edge, 'C')

    def synthetic_profile(self, manifest):
        profile = copy.deepcopy(prep.load_profiles()['mooc'])
        profile.update(source_sha256=prep.sha256(self.source), expected_sha256=manifest['artifacts'],
                       events=3, nodes=4, positives=1, edge_dim=2, node_dim=172)
        return {'mooc': profile}

    def test_profile_loading_and_blocked(self):
        profiles = prep.load_profiles()
        self.assertEqual(set(profiles), {'digg', 'bitcoinalpha', 'wikipedia', 'mooc'})
        self.assertEqual(profiles['digg']['status'], 'BLOCKED')
        with self.assertRaisesRegex(ValueError, 'BLOCKED'):
            prep.convert('digg', 'historical', self.source, self.base / 'blocked')
        self.assertFalse((self.base / 'blocked').exists())

    def test_custom_order_and_manifest(self):
        c = self.run_custom('c', 'C')
        f = self.run_custom('f', 'F')
        self.assertEqual(c['status'], 'CUSTOM-VALID')
        self.assertEqual(c['historical_reconstruction_status'], 'NOT-TESTED')
        self.assertEqual(c['parameters']['edge_order'], 'C')
        self.assertEqual(f['parameters']['edge_order'], 'F')
        a = np.load(self.base / 'c/ml_mooc.npy', allow_pickle=False)
        b = np.load(self.base / 'f/ml_mooc.npy', allow_pickle=False)
        self.assertTrue(a.flags.c_contiguous and b.flags.f_contiguous)
        np.testing.assert_array_equal(a, b)
        self.assertNotEqual(c['artifacts']['ml_mooc.npy'], f['artifacts']['ml_mooc.npy'])
        self.assertEqual(json.loads((self.base / 'f/reproducibility_manifest.json').read_text())['artifacts'], f['artifacts'])

    def test_historical_success_and_no_overwrite(self):
        c = self.run_custom('reference', 'F')
        with patch.object(prep, 'load_profiles', return_value=self.synthetic_profile(c)):
            result = prep.convert('mooc', 'historical', self.source, self.base / 'historical')
            self.assertEqual(result['status'], 'PASS-BITWISE')
            with self.assertRaises(FileExistsError):
                prep.convert('mooc', 'historical', self.source, self.base / 'historical')
            with self.assertRaisesRegex(ValueError, 'overrides'):
                prep.convert('mooc', 'historical', self.source, self.base / 'override', 'C')

    def test_hash_failures(self):
        c = self.run_custom('reference', 'F')
        profiles = self.synthetic_profile(c)
        profiles['mooc']['source_sha256'] = '0' * 64
        with patch.object(prep, 'load_profiles', return_value=profiles):
            with self.assertRaisesRegex(ValueError, 'source SHA-256 mismatch'):
                prep.convert('mooc', 'historical', self.source, self.base / 'badsource')
        self.assertFalse((self.base / 'badsource').exists())
        profiles = self.synthetic_profile(c)
        profiles['mooc']['expected_sha256']['ml_mooc.npy'] = '0' * 64
        with patch.object(prep, 'load_profiles', return_value=profiles):
            with self.assertRaisesRegex(ValueError, 'output mismatch'):
                prep.convert('mooc', 'historical', self.source, self.base / 'badoutput')
        result = json.loads((self.base / 'badoutput/reproducibility_manifest.json').read_text())
        self.assertEqual(result['status'], 'MISMATCH')

    def test_cli_errors_nonzero(self):
        cmd = [sys.executable, '-B', str(ROOT / 'prepare_reproducible.py'), '--dataset', 'digg',
               '--mode', 'historical', '--source', str(self.source), '--destination', str(self.base / 'out')]
        result = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('BLOCKED', result.stderr)

    def test_verify_manifest_detects_tampering(self):
        self.run_custom('verified', 'F')
        dest = self.base / 'verified'
        self.assertEqual(prep.verify_manifest('mooc', 'custom', self.source, dest)['status'], 'CUSTOM-VALID')
        with (dest / 'ml_mooc.csv').open('a', encoding='utf-8') as stream:
            stream.write('1,2,3,0,4\n')
        with self.assertRaisesRegex(ValueError, 'Artifact SHA-256 mismatch'):
            prep.verify_manifest('mooc', 'custom', self.source, dest)

    def test_generated_synthetic_data_loader(self):
        from utils.DataLoader import get_node_classification_data
        self.run_custom('generated', 'F')
        target = self.base / 'processed_data/mooc'
        target.mkdir(parents=True)
        for name in ('ml_mooc.csv', 'ml_mooc.npy', 'ml_mooc_node.npy'):
            shutil.copyfile(self.base / 'generated' / name, target / name)
        previous = Path.cwd()
        try:
            os.chdir(self.base)
            nodes, edges, full, _, _, _ = get_node_classification_data('mooc', .1, .3)
        finally:
            os.chdir(previous)
        self.assertEqual(len(full.src_node_ids), 3)
        self.assertEqual(nodes.shape, (5, 172))
        self.assertEqual(edges.shape, (4, 172))


if __name__ == '__main__':
    unittest.main()
