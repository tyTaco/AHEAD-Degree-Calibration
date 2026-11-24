"""Small CPU tests: malformed summaries and read-only failure paths."""
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
REPORT = PACKAGE / 'src/research_runs/fixed_degree_up_exploration_20261123/report.json'
PRIOR = PACKAGE / 'src/research_runs/c_reference_pairwise_exploration_20261123/report.json'
VERIFY = PACKAGE / 'scripts/verify_scores.py'
SPEC = importlib.util.spec_from_file_location('show_results_test', PACKAGE / 'scripts/show_results.py')
show = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(show)


class ReproToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REPORT.read_text(encoding='utf-8'))

    def test_summary_count_and_tampered_pair(self):
        rows = show.summarize(self.report)
        self.assertEqual(sum(r[0] == 'method' for r in rows), 144)
        self.assertEqual(sum(r[0] == 'paired' for r in rows), 120)
        bad = copy.deepcopy(self.report)
        bad['datasets']['digg']['summary']['val_auroc']['none']['AUROC']['mean'] += .1
        with self.assertRaisesRegex(ValueError, '不一致'):
            show.summarize(bad)

    def test_missing_root_and_hash_fail_without_output(self):
        with tempfile.TemporaryDirectory(prefix='ahead_repro_test_') as temp:
            root = Path(temp)
            (root / 'research_runs').mkdir()
            (root / 'processed_data').mkdir()
            before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
            cmd = [sys.executable, '-B', str(VERIFY), '--assets-root', str(root), '--dataset', 'digg']
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('缺少檔案', result.stderr)
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob('*')))
            dest = root / 'research_runs/fixed_degree_up_exploration_20261123/report.json'
            dest.parent.mkdir()
            dest.write_text('{}', encoding='utf-8')
            before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('hash 不一致', result.stderr)
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob('*')))

    def test_altered_external_code_fails(self):
        with tempfile.TemporaryDirectory(prefix='ahead_repro_hash_') as temp:
            root = Path(temp)
            for rel, src in [('research_runs/fixed_degree_up_exploration_20261123/report.json', REPORT),
                             ('research_runs/c_reference_pairwise_exploration_20261123/report.json', PRIOR)]:
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())
            (root / 'processed_data').mkdir()
            (root / 'run_fixed_degree_up_development.py').write_text('# altered\n', encoding='utf-8')
            before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
            result = subprocess.run([sys.executable, '-B', str(VERIFY), '--assets-root', str(root)],
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=''),
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('程式 hash', result.stderr)
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob('*')))


if __name__ == '__main__':
    unittest.main()
