import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare_reproducible as prep
import train_preprocessed_research as bridge


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ahead-training-bridge-')
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.csv'
        self.source.write_text('user,item,timestamp,state_label,features\n' + ''.join(
            f'u{i%3},v{i%2},{i},{i%2},0.1,0.2\n' for i in range(1, 31)))
        self.data = self.root / 'input'
        prep.convert('mooc', 'custom', self.source, self.data, 'F', 'C')
        self.manifest = prep.sha256(self.data / 'reproducibility_manifest.json')
        self.parent = self.root / 'workspaces'
        self.parent.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def argv(self, *extra):
        return ['--dataset', 'mooc', '--data-mode', 'custom', '--source', str(self.source),
                '--data-dir', str(self.data), '--manifest-sha256', self.manifest,
                '--experiment-mode', 'custom', '--run-id', 'sample',
                '--workspace-parent', str(self.parent), *extra]

    def run_cli(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return bridge.main(self.argv(*extra))

    def test_verify_and_strategy(self):
        self.assertEqual(self.run_cli('--verify-only', '--num-epochs', '2'), 0)
        self.assertEqual(bridge.configuration('mooc', 'cpu', {})[0], 'segmented')
        self.assertEqual(bridge.configuration('wikipedia', 'cpu', {})[0], 'shared')
        self.assertEqual(bridge.configuration('bitcoinalpha', 'cpu', {})[0], 'shared')
        config = bridge.configuration('mooc', 'cpu', {})[2]
        self.assertEqual((config['num_runs'], config['num_epochs'], config['batch_size'],
                          config['num_neighbors'], config['time_gap'], config['optimizer'],
                          config['learning_rate'], config['val_ratio'], config['test_ratio']),
                         (5, 100, 200, 20, 2000, 'Adam', 0.0001, 0.05, 0.5))
        self.assertEqual(self.run_cli('--verify-only', '--batch-size', '0'), 1)
        self.assertEqual(self.run_cli('--verify-only', '--learning-rate', 'nan'), 1)

    def test_dry_run_isolated_no_trainer_and_data_loader(self):
        with mock.patch.object(bridge, 'invoke', side_effect=AssertionError('training called')):
            self.assertEqual(self.run_cli('--dry-run', '--num-epochs', '2'), 0)
        run = self.parent / 'sample'
        info = json.loads((run / 'integration_start.json').read_text())
        self.assertEqual(info['status'], 'DRY-RUN')
        self.assertEqual(info['configuration']['num_epochs'], 2)
        self.assertEqual(info['data_status'], 'CUSTOM-VALID')
        self.assertEqual(info['selection_policies'], ['val_auroc', 'train_loss'])
        self.assertFalse((run / 'project/research_runs').exists())
        self.assertFalse((run / 'project/processed_data/mooc/ml_mooc.csv').is_symlink())
        self.assertEqual(self.run_cli('--dry-run'), 1)
        from utils.DataLoader import get_node_classification_data
        old = Path.cwd()
        try:
            os.chdir(run / 'project')
            with contextlib.redirect_stdout(io.StringIO()):
                node, edge, full, train, val, test = get_node_classification_data('mooc', 0.2, 0.2)
            self.assertEqual(full.num_interactions, 30)
            self.assertEqual(edge.shape[0], 31)
        finally:
            os.chdir(old)

    def test_data_and_manifest_tampering_and_missing(self):
        p = self.data / 'ml_mooc.csv'
        p.write_text(p.read_text() + '\n')
        self.assertEqual(self.run_cli('--verify-only'), 1)
        p.unlink()
        self.assertEqual(self.run_cli('--verify-only'), 1)
        m = self.data / 'reproducibility_manifest.json'
        m.write_text(m.read_text() + ' ')
        self.assertEqual(self.run_cli('--verify-only'), 1)

    def test_legacy_rejects_and_digg_blocked(self):
        (self.data / 'conversion_manifest.json').write_text('{}')
        self.assertEqual(self.run_cli('--verify-only'), 1)
        with self.assertRaisesRegex(ValueError, 'BLOCKED'):
            bridge.validate('digg', 'historical', self.source, self.data, self.manifest)

    def test_historical_overrides_and_execute_require_confirmation(self):
        args = self.argv('--verify-only', '--num-epochs', '1')
        args[args.index('--experiment-mode') + 1] = 'historical'
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bridge.main(args), 1)
        with mock.patch.dict(os.environ, {'AHEAD_INTEGRATION_TRAINING_CONFIRM': ''}):
            self.assertEqual(self.run_cli('--execute'), 1)
        self.assertFalse((self.parent / 'sample').exists())

    def test_legacy_matching_manifest(self):
        names = [f'ml_mooc{x}' for x in ('.csv', '.npy', '_node.npy')]
        legacy = {'schema_version': 1, 'converter_sha256': prep.sha256(ROOT / 'prepare_jodie_research.py'),
                  'artifacts': {name: prep.sha256(self.data / name) for name in names}}
        (self.data / 'conversion_manifest.json').write_text(json.dumps(legacy))
        self.assertEqual(bridge.check_legacy(self.data, 'mooc'), 'VERIFIED')
        self.assertEqual(self.run_cli('--verify-only'), 0)

    def test_worker_dispatch_with_fake_trainers(self):
        (self.root / 'train_shared_research.py').write_text(
            "import json, os\ndef main():\n a=get_node_classification_args()\n open('shared.json','w').write(json.dumps({'dataset':a.dataset_name,'run':os.environ['AHEAD_RESEARCH_RUN_ID'],'confirm':os.environ['AHEAD_SHARED_TRAINING_CONFIRM']}))\n")
        (self.root / 'train_segmented_research.py').write_text(
            "import json\ndef run(project,args,run_id,mode,confirmed):\n open('segmented.json','w').write(json.dumps({'dataset':args.dataset_name,'run':run_id,'mode':mode,'confirm':confirmed}))\n")
        config = bridge.configuration('mooc', 'cpu', {'num_epochs': 1})[2]
        bridge.invoke(self.root, 'shared', config, 'mockshared', 'none')
        bridge.invoke(self.root, 'segmented', config, 'mocksegment', 'original')
        shared = json.loads((self.root / 'shared.json').read_text())
        segmented = json.loads((self.root / 'segmented.json').read_text())
        self.assertEqual((shared['run'], shared['confirm']), ('mockshared', '1'))
        self.assertEqual((segmented['run'], segmented['mode'], segmented['confirm']), ('mocksegment', 'original', True))

    def test_worker_is_constructed_without_running(self):
        config = bridge.configuration('mooc', 'cpu', {'num_epochs': 1})[2]
        with mock.patch.object(bridge.subprocess, 'run') as call:
            call.return_value.returncode = 0
            command = bridge.invoke(self.root, 'segmented', config, 'test', 'original')
        compile(command[3], '<worker>', 'exec')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')


if __name__ == '__main__':
    unittest.main()
