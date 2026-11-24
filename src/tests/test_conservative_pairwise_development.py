"""Synthetic checks for the exploratory historical-label bootstrap guard."""
import unittest
import numpy as np

from conservative_pairwise_development import choose_guarded_factor
from retrospective_pairwise_development import evaluate_frozen, fixed_scores


class ConservativePairwiseTests(unittest.TestCase):
    def test_strong_historical_signal_and_later_labels_do_not_select(self):
        raw = np.tile([1., 1.05], 4)
        degree = np.tile([1, 3], 4)
        y = np.tile([1, 0], 4)
        chosen = choose_guarded_factor(raw, degree, y, block_size=2, replicates=20, seed=7)
        self.assertEqual(chosen['factor'], 1.1)
        self.assertEqual(chosen['reason'], 'bootstrap_guard_passed')
        scores = fixed_scores([1., 1.05], [1, 3], {'factor': chosen['factor'], 'reason': 'historical_auroc'})
        before = scores['pairwise'].copy()
        for future_labels in ([0, 1], [1, 0]):
            metrics = evaluate_frozen(scores, future_labels)
            np.testing.assert_array_equal(scores['pairwise'], before)
            self.assertEqual(chosen['factor'], 1.1)
            if future_labels == [0, 1]:
                self.assertEqual(metrics['none']['AUROC'], 1.)
                self.assertEqual(metrics['pairwise']['AUROC'], 0.)

    def test_single_class_tie_short_window_and_invalid(self):
        result = choose_guarded_factor([1., 2.], [1, 3], [0, 0], block_size=1, replicates=20)
        self.assertEqual((result['factor'], result['reason']), (1., 'single_class_fallback'))
        result = choose_guarded_factor([1., 2.], [1, 3], [0, 1], block_size=2, replicates=20)
        self.assertEqual((result['factor'], result['reason']), (1., 'insufficient_blocks'))
        result = choose_guarded_factor([1., 2., 3., 4.], [3, 3, 3, 3], [0, 1, 0, 1],
                                       block_size=2, replicates=20)
        self.assertEqual((result['factor'], result['reason']), (1., 'no_candidate_passed'))
        result = choose_guarded_factor([1., 1.05, 1.05, 1.05], [1, 3, 3, 3], [1, 0, 0, 0],
                                       block_size=1, replicates=100, seed=7)
        self.assertEqual((result['factor'], result['reason']), (1., 'insufficient_valid_bootstraps'))
        for bad in ([0, 2], [0], None):
            with self.subTest(labels=bad), self.assertRaises(ValueError):
                choose_guarded_factor([1., 2.], [1, 3], bad, block_size=1, replicates=20)
        with self.assertRaises(ValueError):
            choose_guarded_factor([1., 2.], [1, 3], [0, 1], block_size=True, replicates=20)


if __name__ == '__main__':
    unittest.main()
