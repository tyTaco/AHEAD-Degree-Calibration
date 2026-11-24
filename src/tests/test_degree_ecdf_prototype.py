"""CPU-only tests on invented scores, degrees and timestamps."""
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from degree_ecdf_prototype import DegreeConditionalECDF


class DegreeECDFTests(unittest.TestCase):
    def test_reference_is_unlabeled_and_scoring_requires_strictly_later_time(self):
        model = DegreeConditionalECDF(degree_cap=20, prior_strength=10)
        self.assertNotIn('labels', inspect.signature(model.fit).parameters)
        model.fit([1., 2., 3.], [0, 1, 20], np.array([10, 20, 20], dtype=np.int64), model_ready_time=9)
        for times in (np.array([19], dtype=np.int64), np.array([20], dtype=np.int64)):
            with self.assertRaisesRegex(ValueError, 'strictly later'):
                model.transform([2.], [0], times)
        with self.assertRaisesRegex(ValueError, 'same dtype'):
            model.transform([2.], [0], np.array([21.], dtype=np.float64))
        self.assertEqual(model.transform([2.], [0], np.array([21], dtype=np.int64)).shape, (1,))
        with self.assertRaisesRegex(RuntimeError, 'silently replaced'):
            model.fit([1.], [0], np.array([1], dtype=np.int64), model_ready_time=0)

    def test_midrank_ties_bounds_and_same_group_monotonicity(self):
        model = DegreeConditionalECDF(degree_cap=20, prior_strength=0)
        model.fit([1., 2., 2., 9.], [0, 0, 0, 0], np.arange(4, dtype=np.int64), model_ready_time=-1)
        values = model.transform([0., 2., 2., 100.], [0, 0, 0, 0], np.full(4, 10, dtype=np.int64))
        self.assertAlmostEqual(values[1], .5)
        self.assertEqual(values[1], values[2])
        self.assertTrue(np.all(np.diff(values) >= 0))
        self.assertTrue(np.all((values > 0) & (values < 1)))

    def test_missing_group_uses_pooled_and_references_are_copied(self):
        scores = np.array([1., 2., 9.])
        model = DegreeConditionalECDF(degree_cap=20, prior_strength=5).fit(
            scores, [0, 0, 0], np.array([1, 2, 3], dtype=np.int64), model_ready_time=0)
        before = model.transform([2.], [20], np.array([4], dtype=np.int64))
        scores[:] = 1000
        after = model.transform([2.], [20], np.array([4], dtype=np.int64))
        np.testing.assert_array_equal(before, after)
        self.assertAlmostEqual(before.item(), .5)

    def test_group_percentiles_can_reverse_cross_group_raw_score_order(self):
        model = DegreeConditionalECDF(degree_cap=20, prior_strength=0).fit(
            [.1, .2, 10., 20.], [0, 0, 4, 4], np.array([1, 2, 3, 4], dtype=np.int64), model_ready_time=0)
        calibrated = model.transform([1., 15.], [0, 4], np.array([5, 5], dtype=np.int64))
        self.assertLess(1., 15.)
        self.assertGreater(calibrated[0], calibrated[1])

    def test_extreme_outliers_tie_and_shrinkage_approaches_pooled(self):
        scores, degrees = [1., 2., 9., 10.], [0, 0, 4, 4]
        times = np.arange(4, dtype=np.int64)
        unshrunk = DegreeConditionalECDF(degree_cap=20, prior_strength=0).fit(scores, degrees, times, model_ready_time=-1)
        shrunk = DegreeConditionalECDF(degree_cap=20, prior_strength=100).fit(scores, degrees, times, model_ready_time=-1)
        query_time = np.array([5, 5], dtype=np.int64)
        tails = unshrunk.transform([100., 10000.], [0, 0], query_time)
        self.assertEqual(tails[0], tails[1])
        pure_group = unshrunk.transform([5.], [0], np.array([5], dtype=np.int64)).item()
        near_pool = shrunk.transform([5.], [0], np.array([5], dtype=np.int64)).item()
        self.assertGreater(pure_group, near_pool)
        self.assertGreater(near_pool, .5)

    def test_invalid_or_unsafe_inputs_rejected(self):
        for cap, strength in ((3, 1), (20, -1), (20, float('nan'))):
            with self.subTest(cap=cap, strength=strength), self.assertRaises(ValueError):
                DegreeConditionalECDF(degree_cap=cap, prior_strength=strength)
        model = DegreeConditionalECDF(degree_cap=20, prior_strength=1)
        with self.assertRaisesRegex(RuntimeError, 'Fit a reference'):
            model.transform([1.], [0], np.array([2], dtype=np.int64))
        for scores, degree, times in (([float('nan')], [0], np.array([1], dtype=np.int64)),
                                      ([1.], [21], np.array([1], dtype=np.int64)),
                                      ([1.], [.5], np.array([1], dtype=np.int64)),
                                      ([1.], [0], np.array([1], dtype=np.float32)),
                                      ([1., 2.], [0], np.array([1], dtype=np.int64))):
            with self.subTest(scores=scores, degree=degree, times=times), self.assertRaises(ValueError):
                model.fit(scores, degree, times, model_ready_time=0)

    def test_reference_cannot_precede_the_declared_frozen_model(self):
        model = DegreeConditionalECDF(degree_cap=20, prior_strength=1)
        times = np.array([10, 20], dtype=np.int64)
        for cutoff in (10, 15, 20):
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(ValueError, 'strictly later'):
                model.fit([1., 2.], [0, 1], times, model_ready_time=cutoff)
        with self.assertRaisesRegex(ValueError, 'timestamp dtype'):
            model.fit([1., 2.], [0, 1], times, model_ready_time=9.)
        model.fit([1., 2.], [0, 1], times, model_ready_time=9)


if __name__ == '__main__':
    unittest.main()
