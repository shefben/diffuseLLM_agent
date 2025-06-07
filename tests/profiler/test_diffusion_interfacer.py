import unittest
import random
from typing import Dict, List, Any, Union, Optional

from src.profiler.diffusion_interfacer import unify_fingerprints_with_diffusion, UnifiedStyleProfile

# For testing, we might need a simplified SingleSampleFingerprint dict structure
# as input, similar to what llm_interfacer.get_style_fingerprint_from_llm would produce.
import json
from pathlib import Path
from collections import Counter, defaultdict # Added defaultdict for fallback tests
# Import torch and transformers for type hinting and checking Nones, not for direct use in tests
import src.profiler.diffusion_interfacer # To allow patching its torch/transformers globals
from unittest.mock import call # Added call


class TestDiffusionInterfacer(unittest.TestCase):

    def setUp(self):
        self.model_path_example = "dummy/divot5_model_dir"

        # Sample fingerprints (content of the 'fingerprint' sub-dictionary)
        self.fp_style_A = {"indent_width": 4, "preferred_quotes": "single", "docstyle": "google", "max_line_length": 88, "snake_pct": 0.8, "camel_pct": 0.1, "screaming_pct": 0.05} # Note: key for screaming_pct is screaming_pct in sample, maps to UPPER_SNAKE_CASE_pct in profile
        self.fp_style_B = {"indent_width": 2, "preferred_quotes": "double", "docstyle": "numpy", "max_line_length": 79, "snake_pct": 0.2, "camel_pct": 0.7, "screaming_pct": 0.03}
        self.fp_style_C = {"indent_width": 4, "preferred_quotes": "single", "docstyle": "google", "max_line_length": 100, "snake_pct": 0.7, "camel_pct": 0.2, "screaming_pct": 0.1}

        # Input structure for unify_fingerprints_with_diffusion (list of dicts)
        self.per_sample_fingerprints_input_weighted = [
            {"fingerprint": self.fp_style_A, "file_path": "src/module_a/file1.py", "weight": 0.7},
            {"fingerprint": self.fp_style_B, "file_path": "src/legacy/file2.py", "weight": 0.2},
            {"fingerprint": self.fp_style_C, "file_path": "src/module_a/file3.py", "weight": 0.1}
        ]

        self.expected_unified_profile_keys = UnifiedStyleProfile().to_dict().keys()

        self.mock_divot5_success_output_str = json.dumps({
            "indent_width": 4, "preferred_quotes": "single", "docstyle": "google", "max_line_length": 88,
            "identifier_snake_case_pct": 0.75, "identifier_camelCase_pct": 0.15,
            "identifier_UPPER_SNAKE_CASE_pct": 0.05,
            "directory_overrides": {"src/legacy/": {"indent_width": 2}},
            "confidence_score": 0.9
        })

    @patch('src.profiler.diffusion_interfacer.torch')
    @patch('src.profiler.diffusion_interfacer.T5TokenizerFast')
    @patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration')
    def test_unify_fingerprints_success_flow_with_weights_in_prompt(self, MockModel, MockTokenizer, MockTorch):
        """Test successful DivoT5 flow, ensuring weights are in the prompt."""
        MockTorch.cuda.is_available.return_value = False

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = MagicMock(input_ids="dummy_input_ids")
        mock_tokenizer_instance.decode.return_value = self.mock_divot5_success_output_str
        MockTokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = ["dummy_output_tensor"]
        MockModel.from_pretrained.return_value = mock_model_instance

        result_dict = unify_fingerprints_with_diffusion(
            self.per_sample_fingerprints_input_weighted, self.model_path_example
        )

        # Check that the input text to T5 tokenizer includes the weights in serialized JSON
        args_tokenizer_call, _ = mock_tokenizer_instance.call_args
        input_text_to_t5 = args_tokenizer_call[0]

        self.assertIn("SYSTEM:", input_text_to_t5)
        self.assertIn("Pay more attention to samples with higher 'weight'", input_text_to_t5) # Check for new prompt part
        self.assertIn("USER_FINGERPRINT_SAMPLES_START:", input_text_to_t5)
        # Verify that the serialized JSON passed to T5 contains the weights
        # json.dumps on the input list will include the 'weight' field.
        self.assertIn(json.dumps(self.per_sample_fingerprints_input_weighted, indent=2), input_text_to_t5)
        self.assertIn("PROJECT_STYLE_PROFILE_JSON_START:", input_text_to_t5)

        expected_dict = json.loads(self.mock_divot5_success_output_str)
        self.assertEqual(result_dict, expected_dict)

    # --- Tests for Weighted Fallback Logic ---
    def _run_fallback_test(self, patch_target, patch_value=None, side_effect_value=None):
        """Helper to run tests that should trigger the fallback logic."""
        # This helper might need adjustment if patch_target is a module-level var vs. a class method
        if side_effect_value:
            patcher = patch(f'src.profiler.diffusion_interfacer.{patch_target}', side_effect=side_effect_value)
        else:
            patcher = patch(f'src.profiler.diffusion_interfacer.{patch_target}', patch_value)

        with patcher:
            result = unify_fingerprints_with_diffusion(
                self.per_sample_fingerprints_input_weighted, self.model_path_example
            )
        self.assertEqual(result.get("confidence_score"), 0.5, f"Fallback not triggered as expected for {patch_target}")
        return result

    def test_fallback_transformers_not_available(self):
        original_t5_gen = src.profiler.diffusion_interfacer.T5ForConditionalGeneration
        src.profiler.diffusion_interfacer.T5ForConditionalGeneration = None
        try:
            result = unify_fingerprints_with_diffusion(
                self.per_sample_fingerprints_input_weighted, self.model_path_example
            )
            self.assertEqual(result.get("confidence_score"), 0.5)
            # Expected weighted mode for indent: sample1 (0.7*4=2.8) + sample3 (0.1*4=0.4) Total for 4 = 3.2. sample2 (0.2*2=0.4) for indent 2. So, indent 4.
            self.assertEqual(result.get("indent_width"), 4)
            # Expected weighted average for linelen: (88*0.7 + 79*0.2 + 100*0.1) / (0.7+0.2+0.1)
            # = (61.6 + 15.8 + 10) / 1.0 = 87.4 -> rounded to 87
            self.assertEqual(result.get("max_line_length"), 87)
        finally:
            src.profiler.diffusion_interfacer.T5ForConditionalGeneration = original_t5_gen


    @patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration.from_pretrained', side_effect=Exception("Model load failed"))
    def test_fallback_model_load_exception(self, mock_model_load):
        # Ensure torch.cuda.is_available is also patched if it's called before the exception
        with patch('src.profiler.diffusion_interfacer.torch.cuda.is_available', return_value=False):
            result = unify_fingerprints_with_diffusion(self.per_sample_fingerprints_input_weighted, self.model_path_example)
        self.assertEqual(result.get("confidence_score"), 0.5)
        self.assertEqual(result.get("indent_width"), 4)


    @patch('src.profiler.diffusion_interfacer.torch')
    @patch('src.profiler.diffusion_interfacer.T5TokenizerFast')
    @patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration')
    def test_fallback_model_generate_exception(self, MockModel, MockTokenizer, MockTorch):
        MockTorch.cuda.is_available.return_value = False
        mock_model_instance = MagicMock()
        mock_model_instance.generate.side_effect = Exception("Generate failed")
        MockModel.from_pretrained.return_value = mock_model_instance
        MockTokenizer.from_pretrained.return_value = MagicMock() # Basic mock for tokenizer

        result = unify_fingerprints_with_diffusion(
            self.per_sample_fingerprints_input_weighted, self.model_path_example
        )
        self.assertEqual(result.get("confidence_score"), 0.5)
        self.assertEqual(result.get("indent_width"), 4)


    @patch('src.profiler.diffusion_interfacer.torch')
    @patch('src.profiler.diffusion_interfacer.T5TokenizerFast')
    @patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration')
    def test_fallback_json_decode_error(self, MockModel, MockTokenizer, MockTorch):
        MockTorch.cuda.is_available.return_value = False
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.decode.return_value = "This is not JSON" # Invalid JSON output
        MockTokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = ["dummy_tensor"]
        MockModel.from_pretrained.return_value = mock_model_instance

        result = unify_fingerprints_with_diffusion(
            self.per_sample_fingerprints_input_weighted, self.model_path_example
        )
        self.assertEqual(result.get("confidence_score"), 0.5) # Fallback triggered
        self.assertEqual(result.get("indent_width"), 4) # Check weighted mode from fallback
        # max_line_length = (88*0.7 + 79*0.2 + 100*0.1) / 1.0 = 87.4 -> 87
        self.assertEqual(result.get("max_line_length"), 87)
        # snake_pct from samples: A=0.8, B=0.2, C=0.7. Weights: A=0.7, B=0.2, C=0.1
        # snake_pct_avg = (0.8*0.7 + 0.2*0.2 + 0.7*0.1) / (0.7+0.2+0.1) = (0.56 + 0.04 + 0.07) / 1.0 = 0.67
        self.assertAlmostEqual(result.get("identifier_snake_case_pct"), 0.67)


    def test_fallback_with_empty_fingerprint_list(self):
        original_t5_gen = src.profiler.diffusion_interfacer.T5ForConditionalGeneration
        src.profiler.diffusion_interfacer.T5ForConditionalGeneration = None
        try:
            result = unify_fingerprints_with_diffusion([], self.model_path_example)
            default_profile = UnifiedStyleProfile().to_dict()
            self.assertEqual(result, default_profile)
        finally:
            src.profiler.diffusion_interfacer.T5ForConditionalGeneration = original_t5_gen


    def test_fallback_with_fingerprints_missing_data_for_aggregation(self):
        samples_missing_keys = [
            {"fingerprint": {"indent_width": 4}, "file_path": "a.py", "weight": 0.5},
            {"fingerprint": {"preferred_quotes": "single"}, "file_path": "b.py", "weight": 0.5}
        ]
        original_t5_gen = src.profiler.diffusion_interfacer.T5ForConditionalGeneration
        src.profiler.diffusion_interfacer.T5ForConditionalGeneration = None
        try:
            result = unify_fingerprints_with_diffusion(samples_missing_keys, self.model_path_example)
            self.assertEqual(result.get("confidence_score"), 0.5)
            self.assertEqual(result.get("indent_width"), 4)
            self.assertEqual(result.get("preferred_quotes"), "single")
            self.assertIsNone(result.get("docstyle"))
        finally:
            src.profiler.diffusion_interfacer.T5ForConditionalGeneration = original_t5_gen

    @patch('src.profiler.diffusion_interfacer.torch')
    @patch('src.profiler.diffusion_interfacer.T5TokenizerFast')
    @patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration')
    def test_unify_fingerprints_output_missing_keys_warning_adapted(self, MockModel, MockTokenizer, MockTorch):
        MockTorch.cuda.is_available.return_value = False
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = MagicMock(input_ids="dummy_ids")
        incomplete_json_str = json.dumps({"indent_width": 4, "preferred_quotes": "single"})
        mock_tokenizer_instance.decode.return_value = incomplete_json_str
        MockTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        MockModel.from_pretrained.return_value = MagicMock(generate=MagicMock(return_value=["dummy_tensor"]))

        with patch('builtins.print') as mock_print:
            result = unify_fingerprints_with_diffusion(self.per_sample_fingerprints_input_weighted, self.model_path_example)
            self.assertEqual(result.get("indent_width"), 4)
            printed_warnings = "".join(str(c.args[0]) for c in mock_print.call_args_list)
            self.assertIn("Warning: DivoT5 output JSON missing expected key 'docstyle'", printed_warnings)
            self.assertIn("Warning: DivoT5 output JSON missing expected key 'max_line_length'", printed_warnings)


if __name__ == '__main__':
    unittest.main()
