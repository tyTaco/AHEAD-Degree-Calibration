"""Synthetic checks for retrospective, label-informed development only."""
import unittest
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

from retrospective_pairwise_development import (choose_factor, chronological_cut,
                                                evaluate_frozen, fixed_scores)


class RetrospectivePairwiseTests(unittest.TestCase):
    def test_earlier_selection_and_later_labels_are_separate(self):
        times = np.arange(1, 9, dtype=np.float64)
        available = np.repeat(np.array([2., 4., 6., 8.]), 2)
        cut = chronological_cut(times, available, batch_size=2)
        self.assertEqual(cut, 4)
        early_scores = np.array([1., 1.05, 1., 1.05], dtype=np.float32)
        early_degrees, early_y = [1, 3, 1, 3], [1, 0, 1, 0]
        selection = choose_factor(early_scores, early_degrees, early_y)
        self.assertEqual(selection['factor'], 1.1)
        scores = fixed_scores([1., 1.05, 1.1, 1.], [1, 3, 1, 3], selection)
        unchanged = {key: value.copy() for key, value in scores.items()}
        result_a = evaluate_frozen(scores, [1, 0, 1, 0])
        result_b = evaluate_frozen(scores, [0, 1, 0, 1])
        self.assertEqual(selection['factor'], 1.1)
        for key in scores:
            np.testing.assert_array_equal(scores[key], unchanged[key])
            self.assertAlmostEqual(result_a[key]['AUROC'] + result_b[key]['AUROC'], 1.)
        np.testing.assert_allclose(scores['original'], [0.9, 1.05, .99, 1.])

    def test_single_class_explicit_fallback_and_bad_inputs(self):
        selected = choose_factor([1., 2.], [1, 3], [0, 0])
        self.assertEqual(selected, {'factor': 1., 'reason': 'single_class_fallback', 'historical_auroc': {}})
        values = fixed_scores([1., 2.], [1, 3], selected)
        np.testing.assert_array_equal(values['none'], values['pairwise'])
        for labels in ([0, 2], [0], [0, np.nan], None):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                choose_factor([1., 2.], [1, 3], labels)
        with self.assertRaises(ValueError):
            fixed_scores([1., 2.], [1, 3], {'factor': .9, 'reason': 'single_class_fallback'})
        with self.assertRaises(ValueError):
            evaluate_frozen(values, [0, 0])

    def test_boundary_and_score_availability(self):
        with self.assertRaisesRegex(ValueError, 'completion'):
            chronological_cut(np.array([1., 2., 3., 4.]), np.array([1., 2., 3., 4.]), batch_size=2)
        with self.assertRaisesRegex(ValueError, 'boundary'):
            chronological_cut(np.ones(4, np.float64), np.ones(4, np.float64), batch_size=2)
        with self.assertRaises(ValueError):
            chronological_cut(np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32), batch_size=1)
        with self.assertRaises(ValueError):
            choose_factor([2**53, 2**53+1], [1, 3], [0, 1])

    def test_report_guards_test_boundary_and_overwrite(self):
        import run_retrospective_pairwise as runner
        from train_research import file_sha256

        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory(prefix='ahead_pairwise_report_') as directory:
                project = Path(directory)
                root = project / 'research_runs' / runner.RUN_ID
                dev = root / 'degree_development'
                dev.mkdir(parents=True)
                data = project / 'processed_data' / 'digg'
                data.mkdir(parents=True)
                (data / 'ml_digg.csv').write_text('ts,label\n' + ''.join(
                    f'{i},{i % 2}\n' for i in range(1, 801)) + '801,UNAVAILABLE_TEST_LABEL\n')
                for name in ('run_retrospective_pairwise.py', 'run_retrospective_pairwise_datasets.py',
                             'retrospective_pairwise_development.py',
                             'label_gate_development.py'):
                    (project / name).write_text('synthetic provenance\n')
                (root / 'provenance.json').write_text('{}')
                (root / 'selection_complete.json').write_text('{}')
                protocol = {'run_id': runner.RUN_ID, 'dataset': 'digg', 'batch_size': 200,
                    'input_provenance_sha256': file_sha256(root / 'provenance.json'),
                    'selection_seal_sha256': file_sha256(root / 'selection_complete.json'),
                    'segments': {'development': [0, 800], 'old_test': [800, 801]}}
                (dev / 'development_started.json').write_text(json.dumps(protocol))
                artifacts, runs = {}, []
                times = np.arange(1, 801, dtype=np.float64)
                available = np.repeat([200., 400., 600., 800.], 200)
                raw = np.where(np.arange(800) % 2, 1.05, 1.).astype(np.float32)
                degree = np.where(np.arange(800) % 2, 3, 1).astype(np.int32)
                original = raw * np.where(degree < 3, .9, 1.)
                continuous = raw * (1. - .1 * np.exp(-degree / 3.))
                for seed in range(5):
                    runs.append({'seed': seed, 'policies': {p: {'checkpoint_sha256': 'synthetic'}
                                                            for p in ('val_auroc', 'train_loss')}})
                    for policy in ('val_auroc', 'train_loss'):
                        name = f'{policy}_seed{seed}_scores.npz'
                        np.savez(dev / name, none=raw, original=original, continuous=continuous,
                                 development_degrees=degree, development_times=times,
                                 development_score_available_at=available)
                        artifacts[name] = file_sha256(dev / name)
                (dev / 'development_complete.json').write_text(json.dumps({'num_runs': 5,
                    'protocol_sha256': file_sha256(dev / 'development_started.json'),
                    'artifact_sha256': artifacts}))
                os.chdir(project)
                with patch.object(runner, '_verify_run', return_value=(root, {'dataset': 'digg'}, runs)):
                    with self.assertRaisesRegex(ValueError, 'opt-in'):
                        runner.run(project)
                    protocol['segments']['old_test'][0] = 799
                    (dev / 'development_started.json').write_text(json.dumps(protocol))
                    bad_seal = {'num_runs': 5, 'protocol_sha256': file_sha256(dev / 'development_started.json'),
                                'artifact_sha256': artifacts}
                    (dev / 'development_complete.json').write_text(json.dumps(bad_seal))
                    with self.assertRaisesRegex(ValueError, 'old Test'):
                        runner.run(project, confirmed=True)
                    protocol['segments']['old_test'][0] = 800
                    (dev / 'development_started.json').write_text(json.dumps(protocol))
                    bad_seal['protocol_sha256'] = file_sha256(dev / 'development_started.json')
                    (dev / 'development_complete.json').write_text(json.dumps(bad_seal))
                    report = runner.run(project, confirmed=True)
                    self.assertEqual(report['selection_cut_index'], 400)
                    self.assertEqual(report['label_release_time'], 'unknown')
                    self.assertEqual(len(report['results']['train_loss']), 5)
                    with self.assertRaises(FileExistsError):
                        runner.run(project, confirmed=True)
                import run_retrospective_pairwise_datasets as multi
                with self.assertRaisesRegex(ValueError, 'allowed'):
                    multi.run(project, 'bitcoinotc', confirmed=True)
                root2 = project / 'second_synthetic_run'
                root2.mkdir()
                shutil.copytree(dev, root2 / 'degree_development')
                shutil.copy2(root / 'provenance.json', root2 / 'provenance.json')
                shutil.copy2(root / 'selection_complete.json', root2 / 'selection_complete.json')
                with patch.dict(multi.RUNS, {'digg': runner.RUN_ID}), \
                     patch.object(multi, '_verify_run', return_value=(root2, {'dataset': 'digg'}, runs)):
                    second = multi.run(project, 'digg', confirmed=True)
                    self.assertEqual(second['summary'], report['summary'])
                    with self.assertRaises(FileExistsError):
                        multi.run(project, 'digg', confirmed=True)
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
