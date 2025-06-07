import unittest
from unittest.mock import patch, MagicMock
import ast
from pathlib import Path
import tempfile
import shutil
import random

# Assuming StyleSampler and CodeSample are in src.profiler.style_sampler
# Adjust import path if your project structure is different or if already installed.
from src.profiler.style_sampler import StyleSampler, CodeSample
import json # For creating mock AI fingerprint dicts
from unittest.mock import call # Added call

class TestStyleSampler(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to simulate a repo
        self.test_repo_dir = Path(tempfile.mkdtemp(prefix="test_repo_"))
        # Provide dummy model paths required by StyleSampler constructor
        self.dummy_deepseek_path = "dummy/deepseek.gguf"
        self.dummy_divot5_path = "dummy/divot5_model"

        # Instantiate StyleSampler with new required arguments
        self.sampler = StyleSampler(
            repo_path=str(self.test_repo_dir),
            deepseek_model_path=self.dummy_deepseek_path,
            divot5_model_path=self.dummy_divot5_path,
            random_seed=42
        )

        # A mock successful AI fingerprint
        self.mock_successful_ai_fp = {
            "indent": 4, "quotes": "single", "linelen": 88,
            "snake_pct": 0.8, "camel_pct": 0.1, "screaming_pct": 0.05,
            "docstyle": "google", "validation_status": "passed"
        }
        # A mock AI fingerprint with validation errors
        self.mock_failed_ai_fp = {
            "validation_status": "failed_sanity_checks",
            "validation_errors": ["Invalid 'indent' value: 5"]
        }


    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_repo_dir)

    def _create_dummy_file(self, path_from_repo_root: str, content: str = ""):
        full_path = self.test_repo_dir / path_from_repo_root
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return full_path

    def test_discover_python_files_simple(self):
        self._create_dummy_file("file1.py")
        self._create_dummy_file("subdir/file2.py")
        self._create_dummy_file("file3.txt") # Non-python file

        self.sampler._discover_python_files()

        discovered_paths = sorted([p.relative_to(self.test_repo_dir) for p in self.sampler._all_py_files])
        expected_paths = sorted([Path("file1.py"), Path("subdir/file2.py")])

        self.assertEqual(discovered_paths, expected_paths)

    def test_discover_python_files_with_ignored_dirs(self):
        self._create_dummy_file("main.py")
        self._create_dummy_file(".venv/ignored_in_venv.py")
        self._create_dummy_file("src/.git/ignored_in_git.py")
        self._create_dummy_file("env/another_ignored.py")
        self._create_dummy_file("project/actual_code.py")

        self.sampler._discover_python_files()

        discovered_paths = sorted([p.relative_to(self.test_repo_dir) for p in self.sampler._all_py_files])
        # Updated expected paths to match the change in test_discover_python_files_with_ignored_dirs
        expected_paths = sorted([Path("main.py")])

        self.assertEqual(discovered_paths, expected_paths)

    def test_discover_python_files_empty_repo(self):
        self.sampler._discover_python_files()
        self.assertEqual(len(self.sampler._all_py_files), 0)


    @patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample')
    def test_extract_elements_from_file_with_ai_fingerprinting(self, mock_get_ai_fp):
        mock_get_ai_fp.return_value = self.mock_successful_ai_fp

        content = """
class MyClass: # Class definition is one element
    def my_method(self): # Method definition is another
        pass
def my_function(): # Function definition
    pass
""" # Module docstring not present here for simplicity of counting elements.
        file_path = self._create_dummy_file("test_ai_module.py", content)
        elements = self.sampler._extract_elements_from_file(file_path)

        # Expected elements: MyClass, my_method, my_function (3 elements)
        # Each will call get_ai_style_fingerprint_for_sample
        self.assertEqual(len(elements), 3)
        self.assertEqual(mock_get_ai_fp.call_count, 3)

        for i, el in enumerate(elements):
            self.assertEqual(el.file_path, file_path)
            self.assertIsNotNone(el.ai_fingerprint)
            self.assertEqual(el.ai_fingerprint.get("validation_status"), "passed")
            self.assertIsNone(el.error_while_fingerprinting)

            # Verify get_ai_style_fingerprint_for_sample was called with correct snippet
            # and model paths from self.sampler
            args, kwargs = mock_get_ai_fp.call_args_list[i]
            self.assertEqual(args[0], el.code_snippet) # First arg is the snippet
            self.assertEqual(kwargs['deepseek_model_path'], self.dummy_deepseek_path)
            self.assertEqual(kwargs['divot5_model_path'], self.dummy_divot5_path)
            self.assertEqual(kwargs['deepseek_n_gpu_layers'], -1) # Default from sampler init

    @patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample')
    def test_extract_elements_ai_fingerprinting_failure(self, mock_get_ai_fp):
        # Simulate AI fingerprinting failing with an exception
        mock_get_ai_fp.side_effect = Exception("AI model exploded")

        content = "def error_func(): pass"
        file_path = self._create_dummy_file("error_module.py", content)
        elements = self.sampler._extract_elements_from_file(file_path)

        self.assertEqual(len(elements), 1)
        self.assertEqual(mock_get_ai_fp.call_count, 1)

        el = elements[0]
        self.assertIsNone(el.ai_fingerprint)
        self.assertIsNotNone(el.error_while_fingerprinting)
        self.assertIn("Exception during AI fingerprinting: AI model exploded", el.error_while_fingerprinting)

    @patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample')
    def test_extract_elements_ai_fingerprint_validation_failed(self, mock_get_ai_fp):
        # Simulate AI fingerprinting returning a dict with validation_status != "passed"
        mock_get_ai_fp.return_value = self.mock_failed_ai_fp

        content = "def validation_fail_func(): pass"
        file_path = self._create_dummy_file("validation_fail_module.py", content)
        elements = self.sampler._extract_elements_from_file(file_path)

        self.assertEqual(len(elements), 1)
        self.assertEqual(mock_get_ai_fp.call_count, 1)

        el = elements[0]
        self.assertIsNotNone(el.ai_fingerprint)
        self.assertEqual(el.ai_fingerprint.get("validation_status"), "failed_sanity_checks")
        self.assertIsNotNone(el.error_while_fingerprinting) # Error message should be populated
        self.assertIn("AI fingerprinting validation failed: failed_sanity_checks", el.error_while_fingerprinting)


    def test_extract_elements_from_file_syntax_error_no_ai_call(self):
        # No need to mock get_ai_style_fingerprint_for_sample if parsing fails first
        file_path = self._create_dummy_file("syntax_error.py", "def func( :")
        elements = self.sampler._extract_elements_from_file(file_path)
        self.assertEqual(elements, [])


    # Test collect_all_elements with mocked AI fingerprinting
    @patch('src.profiler.style_sampler.StyleSampler._discover_python_files') # Keep this simple
    @patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample')
    def test_collect_all_elements_with_ai(self, mock_get_ai_fp, mock_discover):
        # Setup _discover_python_files mock
        file1_path = self._create_dummy_file("file1_for_collect.py", "def func1(): pass")
        file2_path = self._create_dummy_file("file2_for_collect.py", "class ClassA: pass")
        # We need to set self.sampler._all_py_files because _discover_python_files is mocked
        # and doesn't run its original logic to populate it.
        self.sampler._all_py_files = [file1_path, file2_path]
        mock_discover.return_value = None # _discover_python_files modifies in place

        # Setup get_ai_style_fingerprint_for_sample mock
        mock_get_ai_fp.return_value = self.mock_successful_ai_fp

        self.sampler.collect_all_elements()

        mock_discover.assert_called_once()
        # Expect get_ai_style_fingerprint_for_sample to be called for each element in each file
        # file1 has 1 element (func1), file2 has 1 element (ClassA)
        self.assertEqual(mock_get_ai_fp.call_count, 2)

        self.assertEqual(len(self.sampler._collected_elements), 2)
        for el in self.sampler._collected_elements:
            self.assertEqual(el.ai_fingerprint, self.mock_successful_ai_fp)


    # Tests for sample_elements can remain similar, but they now sample CodeSample objects
    # that potentially have AI fingerprints.
    def test_sample_elements_fewer_than_target_with_ai_data(self):
        self.sampler.target_samples = 10
        # Manually populate _collected_elements with CodeSamples having AI data
        self.sampler._collected_elements = [
            CodeSample(Path("f.py"), f"item{i}", "function", "pass", 1,1, ai_fingerprint=self.mock_successful_ai_fp)
            for i in range(5)
        ]
        with patch.object(self.sampler, 'collect_all_elements', MagicMock()): # Prevent actual collection
            samples = self.sampler.sample_elements()
            self.assertEqual(len(samples), 5)
            self.assertTrue(all(s.ai_fingerprint == self.mock_successful_ai_fp for s in samples))

    # (The existing test_sample_elements_more_than_target_random_sampling and
    # test_sample_elements_with_seed_for_deterministic_random_sample are still relevant
    # for the sampling logic itself, assuming _collected_elements is populated correctly
    # with CodeSample objects that may or may not have AI fingerprints yet, or have errors.)
    # They might need slight adjustment if the structure of CodeSample objects they create for testing
    # needs to align with the new definition (i.e. add None for ai_fingerprint & error fields).

    def test_sample_elements_more_than_target_random_sampling_updated(self):
        self.sampler.target_samples = 5
        self.sampler._collected_elements = [
             CodeSample(Path("f.py"), f"item{i}", "function", "pass", 1,1, ai_fingerprint=None, error_while_fingerprinting=None) for i in range(10)
        ]
        with patch.object(self.sampler, 'collect_all_elements', MagicMock()):
            samples = self.sampler.sample_elements()
            self.assertEqual(len(samples), 5)

    def test_sample_elements_with_seed_for_deterministic_random_sample(self):
        # This test relies on the current mock behavior of random.sample in sample_elements
        self.sampler.target_samples = 3
        # Populate with new CodeSample structure
        elements_for_seed_test = [
             CodeSample(Path("f.py"), f"item{i}", "function", "pass", 1,1,
                        ai_fingerprint=None, error_while_fingerprinting=None)
             for i in range(5)
        ]

        # First run
        # Create new samplers for this test to ensure fresh seeding if StyleSampler's __init__ handles it
        sampler1 = StyleSampler(
            repo_path=str(self.test_repo_dir),
            deepseek_model_path=self.dummy_deepseek_path,
            divot5_model_path=self.dummy_divot5_path,
            target_samples=3, random_seed=123
        )
        sampler1._collected_elements = elements_for_seed_test[:]
        with patch.object(sampler1, 'collect_all_elements', MagicMock()):
            samples1 = sampler1.sample_elements()

        # Second run with the same seed
        sampler2 = StyleSampler(
            repo_path=str(self.test_repo_dir),
            deepseek_model_path=self.dummy_deepseek_path,
            divot5_model_path=self.dummy_divot5_path,
            target_samples=3, random_seed=123
        )
        sampler2._collected_elements = elements_for_seed_test[:]
        with patch.object(sampler2, 'collect_all_elements', MagicMock()):
            samples2 = sampler2.sample_elements()

        self.assertEqual(samples1, samples2, "Samples should be deterministic with the same seed.")

        # Third run with a different seed
        sampler3 = StyleSampler(
            repo_path=str(self.test_repo_dir),
            deepseek_model_path=self.dummy_deepseek_path,
            divot5_model_path=self.dummy_divot5_path,
            target_samples=3, random_seed=456
        )
        sampler3._collected_elements = elements_for_seed_test[:]
        with patch.object(sampler3, 'collect_all_elements', MagicMock()):
            samples3 = sampler3.sample_elements()

        if len(elements_for_seed_test) > sampler3.target_samples :
             self.assertNotEqual(samples1, samples3, "Samples should differ with different seeds if sampling occurs.")


if __name__ == '__main__':
    unittest.main()
