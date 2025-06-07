import unittest
import random
from typing import Dict, List, Any, Union, Optional

from src.profiler.diffusion_interfacer import unify_fingerprints_with_diffusion, UnifiedStyleProfile

# For testing, we might need a simplified SingleSampleFingerprint dict structure
# as input, similar to what llm_interfacer.get_style_fingerprint_from_llm would produce.
import json # For creating mock JSON strings and parsing
from pathlib import Path # For model_path if used directly
from collections import Counter # For checking fallback logic
# Import torch and transformers for type hinting and checking Nones, not for direct use in tests
import src.profiler.diffusion_interfacer # To allow patching its torch/transformers globals
from unittest.mock import call # Added call


class TestDiffusionInterfacer(unittest.TestCase):

    def setUp(self):
        self.model_path_example = "dummy/divot5_model_dir"
        self.sample_fp_1 = {"indent": 4, "quotes": "single", "linelen": 88, "docstyle": "google", "snake_pct": 0.8, "camel_pct": 0.1, "UPPER_SNAKE_CASE_pct": 0.05}
        self.sample_fp_2 = {"indent": 2, "quotes": "double", "linelen": 79, "docstyle": "numpy", "snake_pct": 0.2, "camel_pct": 0.7, "UPPER_SNAKE_CASE_pct": 0.03}

        self.per_sample_fingerprints_input = [
            {"fingerprint": self.sample_fp_1, "file_path": "src/module_a/file1.py"},
            {"fingerprint": self.sample_fp_2, "file_path": "src/legacy/file2.py"}
        ]

        self.expected_unified_profile_keys = UnifiedStyleProfile().to_dict().keys()

        # A typical successful JSON output string from DivoT5
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
    def test_unify_fingerprints_success_flow(self, MockModel, MockTokenizer, MockTorch):
        """Test successful unification flow with DivoT5 model interaction."""
        MockTorch.cuda.is_available.return_value = False # Test CPU path

        mock_tokenizer_instance = MagicMock()
        # Simulate tokenizer returning a dict with input_ids (or any non-None object)
        mock_tokenizer_instance.return_value = MagicMock(input_ids="dummy_input_ids")
        mock_tokenizer_instance.decode.return_value = self.mock_divot5_success_output_str
        MockTokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        # Simulate model.generate returning a list of tensors (or just a list with one item)
        mock_model_instance.generate.return_value = ["dummy_output_tensor"]
        MockModel.from_pretrained.return_value = mock_model_instance

        result_dict = unify_fingerprints_with_diffusion(
            self.per_sample_fingerprints_input, self.model_path_example
        )

        MockTokenizer.from_pretrained.assert_called_once_with(self.model_path_example)
        MockModel.from_pretrained.assert_called_once_with(self.model_path_example)
        mock_model_instance.to.assert_called_once_with("cpu")
        mock_model_instance.eval.assert_called_once()

        # Check input to tokenizer (which forms the input to model.generate)
        args_tokenizer_call, kwargs_tokenizer_call = mock_tokenizer_instance.call_args
        input_text_to_t5 = args_tokenizer_call[0]
        self.assertIn("SYSTEM:", input_text_to_t5)
        self.assertIn("USER_FINGERPRINT_SAMPLES_START:", input_text_to_t5)
        self.assertIn(json.dumps(self.per_sample_fingerprints_input, indent=2), input_text_to_t5)
        self.assertIn("PROJECT_STYLE_PROFILE_JSON_START:", input_text_to_t5)
        self.assertEqual(kwargs_tokenizer_call.get("max_length"), 4096) # Check max_input_length

        mock_model_instance.generate.assert_called_once_with(
            "dummy_input_ids", # This comes from mock_tokenizer_instance().input_ids
            max_length=1024, num_beams=4, early_stopping=True
        )
        mock_tokenizer_instance.decode.assert_called_once_with("dummy_output_tensor", skip_special_tokens=True)

        # Check the final parsed dictionary
        expected_dict = json.loads(self.mock_divot5_success_output_str)
        self.assertEqual(result_dict, expected_dict)
        for key in self.expected_unified_profile_keys:
             self.assertIn(key, result_dict, f"Expected key {key} not in successful result.")


    def test_unify_fingerprints_empty_input_list_fallback(self):
        """Test with an empty list of per_sample_fingerprints - should use fallback."""
        # Mock transformers to be None to easily trigger fallback for empty list
        with patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration', None):
            result = unify_fingerprints_with_diffusion([], self.model_path_example)

        self.assertIsInstance(result, dict)
        default_profile = UnifiedStyleProfile().to_dict()
        self.assertEqual(result, default_profile) # Fallback for empty list returns default profile

    @patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration', None) # Simulate Transformers not installed
    def test_unify_fingerprints_transformers_not_available_fallback(self):
        result = unify_fingerprints_with_diffusion(
            self.per_sample_fingerprints_input, self.model_path_example
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("confidence_score"), 0.5) # Fallback sets this
        # Check if some aggregation happened (e.g. indent might be mode)
        self.assertIn(result.get("indent_width"), {4, 2}) # Mode of 4,2

    @patch('src.profiler.diffusion_interfacer.torch') # Mock torch to control cuda.is_available
    @patch('src.profiler.diffusion_interfacer.T5TokenizerFast')
    @patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration')
    def test_unify_fingerprints_model_load_exception_fallback(self, MockModel, MockTokenizer, MockTorch):
        MockModel.from_pretrained.side_effect = Exception("Model load failed")
        MockTorch.cuda.is_available.return_value = False

        result = unify_fingerprints_with_diffusion(
            self.per_sample_fingerprints_input, self.model_path_example
        )
        self.assertEqual(result.get("confidence_score"), 0.5) # Fallback was triggered
        self.assertIn(result.get("indent_width"), {4, 2})

    @patch('src.profiler.diffusion_interfacer.torch')
    @patch('src.profiler.diffusion_interfacer.T5TokenizerFast')
    @patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration')
    def test_unify_fingerprints_model_generate_exception_fallback(self, MockModel, MockTokenizer, MockTorch):
        MockTorch.cuda.is_available.return_value = False
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = MagicMock(input_ids="dummy_ids")
        MockTokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.generate.side_effect = Exception("Generate failed")
        MockModel.from_pretrained.return_value = mock_model_instance

        result = unify_fingerprints_with_diffusion(
            self.per_sample_fingerprints_input, self.model_path_example
        )
        self.assertEqual(result.get("confidence_score"), 0.5)
        self.assertIn(result.get("indent_width"), {4, 2})

    @patch('src.profiler.diffusion_interfacer.torch')
    @patch('src.profiler.diffusion_interfacer.T5TokenizerFast')
    @patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration')
    def test_unify_fingerprints_json_decode_error_fallback(self, MockModel, MockTokenizer, MockTorch):
        MockTorch.cuda.is_available.return_value = False
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = MagicMock(input_ids="dummy_ids")
        mock_tokenizer_instance.decode.return_value = "This is not JSON { not really" # Invalid JSON output
        MockTokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = ["dummy_tensor"]
        MockModel.from_pretrained.return_value = mock_model_instance

        result = unify_fingerprints_with_diffusion(
            self.per_sample_fingerprints_input, self.model_path_example
        )
        self.assertEqual(result.get("confidence_score"), 0.5) # Fallback triggered
        self.assertIn(result.get("indent_width"), {4, 2})

    @patch('src.profiler.diffusion_interfacer.torch')
    @patch('src.profiler.diffusion_interfacer.T5TokenizerFast')
    @patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration')
    def test_unify_fingerprints_output_missing_keys_warning(self, MockModel, MockTokenizer, MockTorch):
        """Test DivoT5 output is valid JSON but missing some expected UnifiedProfile keys."""
        MockTorch.cuda.is_available.return_value = False
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = MagicMock(input_ids="dummy_ids")
        # Valid JSON, but missing e.g. 'docstring_style' and 'directory_overrides'
        incomplete_json_str = json.dumps({
            "indent_width": 4, "preferred_quotes": "single", "max_line_length": 80,
            "identifier_snake_case_pct": 0.9
        })
        mock_tokenizer_instance.decode.return_value = incomplete_json_str
        MockTokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = ["dummy_tensor"]
        MockModel.from_pretrained.return_value = mock_model_instance

        with patch('builtins.print') as mock_print:
            result = unify_fingerprints_with_diffusion(
                self.per_sample_fingerprints_input, self.model_path_example
            )
            self.assertEqual(result.get("indent_width"), 4) # Parsed correctly
            self.assertIsNone(result.get("docstring_style")) # Key was missing from DivoT5 output

            # Check if warnings for missing keys were printed
            missing_key_warnings_found = 0
            expected_missing = ["docstring_style", "identifier_camelCase_pct", "identifier_UPPER_SNAKE_CASE_pct", "directory_overrides"]
            for print_arg_tuple in mock_print.call_args_list:
                print_arg = print_arg_tuple[0][0] # First argument of print call
                if "Warning: DivoT5 output JSON missing expected key" in print_arg:
                    for em_key in expected_missing:
                        if f"'{em_key}'" in print_arg:
                             missing_key_warnings_found +=1
                             break
            self.assertGreaterEqual(missing_key_warnings_found, len(expected_missing)-2, "Expected warnings for several missing keys") # Allow some flexibility if not all are warned in one go

    @patch('src.profiler.diffusion_interfacer.torch')
    @patch('src.profiler.diffusion_interfacer.T5TokenizerFast')
    @patch('src.profiler.diffusion_interfacer.T5ForConditionalGeneration')
    def test_unify_fingerprints_output_with_prefix_stripping(self, MockModel, MockTokenizer, MockTorch):
        """Test if 'PROJECT_STYLE_PROFILE_JSON_START:' prefix is stripped from DivoT5 output."""
        MockTorch.cuda.is_available.return_value = False
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = MagicMock(input_ids="dummy_ids")
        output_with_prefix = "PROJECT_STYLE_PROFILE_JSON_START:" + self.mock_divot5_success_output_str
        mock_tokenizer_instance.decode.return_value = output_with_prefix
        MockTokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = ["dummy_tensor"]
        MockModel.from_pretrained.return_value = mock_model_instance

        result_dict = unify_fingerprints_with_diffusion(
            self.per_sample_fingerprints_input, self.model_path_example
        )
        expected_dict = json.loads(self.mock_divot5_success_output_str)
        self.assertEqual(result_dict, expected_dict)


if __name__ == '__main__':
    unittest.main()
