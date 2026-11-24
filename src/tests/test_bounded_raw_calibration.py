"""Synthetic safety properties for raw-score bounded calibration."""
import unittest
import numpy as np
from bounded_raw_calibration import BoundedRawCalibration, BoundedMedianDegreeCalibration


class BoundedMedianDegreeTests(unittest.TestCase):
    def test_shift_is_reference_only_bounded_and_can_hurt_auc(self):
        from sklearn.metrics import roc_auc_score
        reference = np.array([1., 1.1, 2., 2.1])
        times = np.arange(1, 5, dtype=np.int64)
        model = BoundedMedianDegreeCalibration(epsilon=.1, min_group=2).fit(
            reference, [1, 1, 3, 3], times, model_ready_time=np.int64(0))
        self.assertEqual(model.reason, 'median_alignment')
        self.assertEqual(model.reference_counts, (2, 2))
        self.assertGreater(model.shift, 0)
        self.assertLessEqual(abs(model.shift), .1 * np.subtract(*np.quantile(reference, [.75, .25])))
        raw = np.array([1., 1.04])
        out = model.transform(raw, [1, 3], np.array([5, 5], dtype=np.int64))
        self.assertEqual(roc_auc_score([0, 1], raw), 1.)
        self.assertEqual(roc_auc_score([0, 1], out), 0.)
        self.assertEqual(out[1], raw[1])
        with self.assertRaises(RuntimeError):
            model.fit(reference, [1, 1, 3, 3], times, model_ready_time=np.int64(0))

    def test_insufficient_or_flat_reference_falls_back(self):
        times = np.arange(1, 5, dtype=np.int64)
        for reference, degrees, min_group, reason in (
                ([1., 2., 3., 4.], [1, 1, 1, 3], 2, 'insufficient_reference_group'),
                ([1., 1., 1., 1.], [1, 1, 3, 3], 2, 'zero_reference_iqr')):
            model = BoundedMedianDegreeCalibration(epsilon=.1, min_group=min_group).fit(
                reference, degrees, times, model_ready_time=np.int64(0))
            self.assertEqual(model.reason, reason)
            np.testing.assert_array_equal(model.transform([1., 2.], [1, 3], np.array([5, 5], dtype=np.int64)), [1., 2.])

    def test_invalid_inputs_and_early_query_are_rejected(self):
        times = np.arange(1, 5, dtype=np.int64)
        with self.assertRaises(ValueError):
            BoundedMedianDegreeCalibration(epsilon=-1)
        with self.assertRaises(ValueError):
            BoundedMedianDegreeCalibration(min_group=True)
        model = BoundedMedianDegreeCalibration(epsilon=.1, min_group=2)
        with self.assertRaises(ValueError):
            model.fit([1., 2., 3., 4.], [1, 1, 3, 21], times, model_ready_time=np.int64(0))
        model.fit([1., 2., 3., 4.], [1, 1, 3, 3], times, model_ready_time=np.int64(0))
        with self.assertRaises(ValueError):
            model.transform([1.], [1], np.array([4], dtype=np.int64))
        with self.assertRaises(ValueError):
            model.transform([1.], [1.5], np.array([5], dtype=np.int64))


class BoundedRawCalibrationTests(unittest.TestCase):
    def test_zero_preserves_exact_scores_and_positive_bound(self):
        reference = np.array([0., 1., 2., 3.])
        times = np.array([1, 2, 3, 4], dtype=np.int64)
        raw = np.array([1., 2., 5.])
        candidate = np.array([-50., 80., 0.])
        qtime = np.array([5, 6, 7], dtype=np.int64)
        zero = BoundedRawCalibration(epsilon=0).fit(reference, times, model_ready_time=np.int64(0))
        np.testing.assert_array_equal(zero.transform(raw, candidate, qtime), raw)
        bounded = BoundedRawCalibration(epsilon=.1).fit(reference, times, model_ready_time=np.int64(0))
        out = bounded.transform(raw, candidate, qtime)
        limit = .1 * (np.quantile(reference, .75) - np.quantile(reference, .25))
        self.assertTrue(np.all(np.abs(out - raw) <= limit + 1e-14))
        for i in range(len(raw)):
            for j in range(len(raw)):
                if raw[i] > raw[j] + 2 * limit:
                    self.assertGreater(out[i], out[j])

    def test_flat_reference_returns_raw_and_early_query_fails(self):
        times = np.array([1, 2, 3, 4], dtype=np.int64)
        model = BoundedRawCalibration(epsilon=.5).fit(np.ones(4), times, model_ready_time=np.int64(0))
        np.testing.assert_array_equal(model.transform([1., 2.], [8., 9.], np.array([5, 6], dtype=np.int64)), [1., 2.])
        with self.assertRaisesRegex(ValueError, 'strictly after'):
            model.transform([1.], [2.], np.array([4], dtype=np.int64))
        with self.assertRaises(RuntimeError):
            model.fit(np.ones(4), times, model_ready_time=np.int64(0))

    def test_bounded_decrease_can_still_reverse_a_near_pair(self):
        from sklearn.metrics import roc_auc_score
        model = BoundedRawCalibration(epsilon=.1).fit(
            np.array([0., 1., 2., 3.]), np.array([1, 2, 3, 4], dtype=np.int64),
            model_ready_time=np.int64(0))
        raw, candidate = np.array([1., 1.1]), np.array([1., .8])
        adjusted = model.transform(raw, candidate, np.array([5, 6], dtype=np.int64))
        self.assertEqual(roc_auc_score([0, 1], raw), 1.)
        self.assertEqual(roc_auc_score([0, 1], adjusted), 0.)

    def test_labeled_auc_bound_covers_random_and_tied_pairs(self):
        from sklearn.metrics import roc_auc_score
        reference = np.array([0., 1., 2., 3.])
        model = BoundedRawCalibration(epsilon=.1).fit(
            reference, np.array([1, 2, 3, 4], dtype=np.int64), model_ready_time=np.int64(0))
        generator = np.random.default_rng(731)
        for _ in range(50):
            raw = np.round(generator.normal(size=40), 1)
            labels = np.arange(40) % 2
            generator.shuffle(labels)
            candidate = raw + generator.normal(size=40)
            adjusted = model.transform(raw, candidate, np.full(40, 5, dtype=np.int64))
            actual = abs(roc_auc_score(labels, adjusted) - roc_auc_score(labels, raw))
            bound = model.auc_change_upper_bound(raw, labels)
            limit = .1 * (np.quantile(reference, .75) - np.quantile(reference, .25))
            brute = np.mean(np.abs(raw[labels == 1, None] - raw[None, labels == 0]) <= 2 * limit)
            self.assertGreaterEqual(bound + 1e-12, brute)
            self.assertLessEqual(actual, bound + 1e-12)
            self.assertLessEqual(bound, 1.)
        with self.assertRaisesRegex(ValueError, 'both binary'):
            model.auc_change_upper_bound([1., 2.], [0, 0])
        with self.assertRaises(RuntimeError):
            BoundedRawCalibration(epsilon=.1).auc_change_upper_bound([1., 2.], [0, 1])

    def test_auc_bound_is_conservative_at_float_boundaries(self):
        from fractions import Fraction

        cases = [
            (np.array([0., 0., 1., 1.]), 0.),
            (np.array([0., np.nextafter(0., 1.), np.nextafter(0., -1.), 1.]), 0.),
            (np.array([1., np.nextafter(1., 2.), np.nextafter(1., 0.), 2.]), 1e-16),
            (np.array([-np.finfo(np.float64).max, -1., 1., np.finfo(np.float64).max]), 1e308),
        ]
        labels = np.array([0, 1, 0, 1])
        for scores, limit in cases:
            with self.subTest(limit=limit, scores=scores):
                model = BoundedRawCalibration(epsilon=limit / 1.5).fit(
                    np.array([0., 1., 2., 3.]), np.array([1, 2, 3, 4], dtype=np.int64),
                    model_ready_time=np.int64(0))
                exact_limit = Fraction.from_float(model._limit)
                positives = [Fraction.from_float(float(x)) for x in scores[labels == 1]]
                negatives = [Fraction.from_float(float(x)) for x in scores[labels == 0]]
                close = sum(abs(pos - neg) <= 2 * exact_limit for pos in positives for neg in negatives)
                bound = model.auc_change_upper_bound(scores, labels)
                self.assertGreaterEqual(bound, close / (len(positives) * len(negatives)))
                self.assertLessEqual(bound, 1.)


if __name__ == '__main__':
    unittest.main()
