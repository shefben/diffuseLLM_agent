import unittest
from pathlib import Path
import math # For isnan, isinf, log1p

from src.profiler.sample_weigher import calculate_sample_weights, InputCodeSampleForWeighing

class TestSampleWeigher(unittest.TestCase):

    def setUp(self):
        self.dummy_fp = {"key": "value"}
        self.sample1 = InputCodeSampleForWeighing(Path("f1.py"), self.dummy_fp, 100.0, 1700.0) # Large, New
        self.sample2 = InputCodeSampleForWeighing(Path("f2.py"), self.dummy_fp, 10.0,  1600.0) # Small, Old
        self.sample3 = InputCodeSampleForWeighing(Path("f3.py"), self.dummy_fp, 50.0,  1650.0) # Mid, Mid
        self.samples_basic = [self.sample1, self.sample2, self.sample3]

    def test_calculate_sample_weights_empty_input(self):
        self.assertEqual(calculate_sample_weights([]), [])

    def test_calculate_sample_weights_missing_metadata_or_fingerprint(self):
        samples_mixed = [
            self.sample1, # Valid
            InputCodeSampleForWeighing(Path("no_meta.py"), self.dummy_fp, None, None),
            InputCodeSampleForWeighing(Path("no_fp.py"), None, 10.0, 1600.0),
            InputCodeSampleForWeighing(Path("valid_again.py"), self.dummy_fp, 20.0, 1620.0)
        ]
        weighted = calculate_sample_weights(samples_mixed)
        # Expect 3 items in output: sample1, valid_again, and no_meta (with default weight)
        self.assertEqual(len(weighted), 3)

        paths_in_result = {item['file_path'] for item in weighted}
        self.assertIn("f1.py", paths_in_result)
        self.assertIn("valid_again.py", paths_in_result)
        self.assertIn("no_meta.py", paths_in_result) # Gets default weight
        self.assertNotIn("no_fp.py", paths_in_result)  # Skipped as no fingerprint

        for item in weighted:
            if item['file_path'] == "no_meta.py":
                self.assertEqual(item['weight'], 0.5) # Default weight

    def test_calculate_sample_weights_normalization_and_coeffs(self):
        weighted = calculate_sample_weights(self.samples_basic, size_weight_coeff=0.7, recency_weight_coeff=0.3)
        self.assertEqual(len(weighted), 3)

        # sample1 (Large, New): size_score=1, recency_score=1. Weight = 0.7*1 + 0.3*1 = 1.0
        # sample2 (Small, Old): size_score=0, recency_score=0. Weight = 0.7*0 + 0.3*0 = 0.0
        # sample3 (Mid, Mid):
        #   size: (50-10)/(100-10) = 40/90 = 0.444...
        #   recency: (1650-1600)/(1700-1600) = 50/100 = 0.5
        #   Weight = 0.7 * 0.4444 + 0.3 * 0.5 = 0.31108 + 0.15 = 0.46108 -> rounded to 0.4611

        weights = {item['file_path']: item['weight'] for item in weighted}
        self.assertAlmostEqual(weights["f1.py"], 1.0)
        self.assertAlmostEqual(weights["f2.py"], 0.0)
        self.assertAlmostEqual(weights["f3.py"], 0.4611, places=4)

    def test_calculate_sample_weights_log_scale_size(self):
        # size1_log = log1p(100) ~= 4.61512
        # size2_log = log1p(10)  ~= 2.39790
        # size3_log = log1p(50)  ~= 3.93183
        # min_log_s = 2.39790, max_log_s = 4.61512
        # sample1_norm_log_s = 1.0
        # sample2_norm_log_s = 0.0
        # sample3_norm_log_s = (3.93183 - 2.39790) / (4.61512 - 2.39790) = 1.53393 / 2.21722 ~= 0.69182
        # recency scores are the same as before (1.0, 0.0, 0.5)
        # Weights (0.7 size, 0.3 recency):
        # S1: 0.7*1 + 0.3*1 = 1.0
        # S2: 0.7*0 + 0.3*0 = 0.0
        # S3: 0.7*0.69182 + 0.3*0.5 = 0.484274 + 0.15 = 0.634274 -> 0.6343

        weighted = calculate_sample_weights(self.samples_basic, use_log_scale_for_size=True)
        weights = {item['file_path']: item['weight'] for item in weighted}
        self.assertAlmostEqual(weights["f1.py"], 1.0)
        self.assertAlmostEqual(weights["f2.py"], 0.0)
        self.assertAlmostEqual(weights["f3.py"], 0.6343, places=4)

    def test_calculate_sample_weights_single_sample_or_all_same_metadata(self):
        # Single sample
        weighted_single = calculate_sample_weights([self.sample1])
        self.assertEqual(len(weighted_single), 1)
        # norm_size_score = 1.0 (as min_size == max_size, and size > 0)
        # norm_recency_score = 1.0 (as min_ts == max_ts)
        # Weight = 0.7*1 + 0.3*1 = 1.0
        self.assertAlmostEqual(weighted_single[0]['weight'], 1.0)

        # All same metadata
        same_meta_samples = [
            InputCodeSampleForWeighing(Path("s1.py"), self.dummy_fp, 50.0, 1600.0),
            InputCodeSampleForWeighing(Path("s2.py"), self.dummy_fp, 50.0, 1600.0),
        ]
        weighted_same = calculate_sample_weights(same_meta_samples)
        self.assertEqual(len(weighted_same), 2)
        # norm_size_score = 1.0 (as min=max > 0), norm_recency_score = 1.0 (as min=max)
        # All weights should be 1.0
        self.assertTrue(all(item['weight'] == 1.0 for item in weighted_same))

    def test_calculate_sample_weights_zero_size_files(self):
        zero_size_samples = [
            InputCodeSampleForWeighing(Path("z1.py"), self.dummy_fp, 0.0, 1700.0), # New, zero size
            InputCodeSampleForWeighing(Path("z2.py"), self.dummy_fp, 0.0, 1600.0)  # Old, zero size
        ]
        weighted = calculate_sample_weights(zero_size_samples)
        # Size score for both is 0.0 because min_size=max_size=0 (special case in norm gives 0.0)
        # Recency: z1 gets 1.0, z2 gets 0.0
        # Weights (0.7 size, 0.3 recency):
        # z1: 0.7*0 + 0.3*1 = 0.3
        # z2: 0.7*0 + 0.3*0 = 0.0
        weights = {item['file_path']: item['weight'] for item in weighted}
        self.assertAlmostEqual(weights["z1.py"], 0.3)
        self.assertAlmostEqual(weights["z2.py"], 0.0)


if __name__ == '__main__':
    unittest.main()
