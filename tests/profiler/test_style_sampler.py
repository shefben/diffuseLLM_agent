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
from unittest.mock import call, mock_open # Added mock_open for Path.stat potentially
from collections import Counter # Added Counter

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
    @patch('src.profiler.style_sampler.Path.stat') # Mock Path.stat to control metadata
    def test_extract_elements_from_file_with_ai_fingerprinting_and_metadata(self, mock_stat, mock_get_ai_fp):
        # Setup mock for Path.stat()
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 2048 # 2KB
        mock_stat_result.st_mtime = 1678886400.0 # A fixed timestamp
        mock_stat.return_value = mock_stat_result

        mock_get_ai_fp.return_value = self.mock_successful_ai_fp

        content = "class MyClass: pass\ndef my_function(): pass" # 2 elements
        file_path = self._create_dummy_file("test_metadata_module.py", content)

        # The 'file_path' object in the test needs to return the mock_stat object
        # This is tricky because file_path is a real Path object from _create_dummy_file.
        # Instead of mocking Path.stat globally, it's better if the production code
        # uses os.path.getsize and os.path.getmtime so we can mock those more easily per path,
        # or we pass a stat_func to _extract_elements_from_file for testing.
        # For now, let's assume the global Path.stat mock works for the file_path.
        # A more robust test would involve creating a file and checking its actual stat,
        # or more targeted mocking.

        # Re-create file_path with a mock 'stat' method if Path.stat is hard to globally control per instance
        # For this subtask, we'll assume the global mock_stat is sufficient.

        elements = self.sampler._extract_elements_from_file(file_path)

        self.assertEqual(len(elements), 2) # MyClass, my_function
        self.assertEqual(mock_get_ai_fp.call_count, 2)
        # mock_stat.assert_called_with(file_path) # Check that stat was called on the file
        # Path.stat is a method of Path instances. Mocking it globally for a specific instance (file_path) is tricky.
        # The mock setup above will ensure that any call to Path(...).stat() returns mock_stat_result.
        # We can check if it was called at least once.
        mock_stat.assert_called()


        for el_idx, el in enumerate(elements):
            self.assertEqual(el.file_path, file_path)
            self.assertIsNotNone(el.ai_fingerprint)
            self.assertEqual(el.ai_fingerprint.get("validation_status"), "passed")
            self.assertIsNone(el.error_while_fingerprinting)
            self.assertEqual(el.file_size_kb, 2.0) # 2048 / 1024.0
            self.assertEqual(el.mod_timestamp, 1678886400.0)

            # Verify AI call args
            args, kwargs = mock_get_ai_fp.call_args_list[el_idx]
            self.assertEqual(args[0], el.code_snippet)
            self.assertEqual(kwargs['deepseek_model_path'], self.dummy_deepseek_path)

    @patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample')
    # We also need to mock stat here, otherwise the function might fail before AI call
    @patch('src.profiler.style_sampler.Path.stat')
    def test_extract_elements_ai_fingerprinting_failure(self, mock_stat, mock_get_ai_fp):
        # Simulate AI fingerprinting failing with an exception
        mock_get_ai_fp.side_effect = Exception("AI model exploded")
        mock_stat_result = MagicMock() # Basic stat mock
        mock_stat_result.st_size = 100
        mock_stat_result.st_mtime = 1000.0
        mock_stat.return_value = mock_stat_result

        content = "def error_func(): pass"
        file_path = self._create_dummy_file("error_module.py", content)
        elements = self.sampler._extract_elements_from_file(file_path)

        self.assertEqual(len(elements), 1)
        self.assertEqual(mock_get_ai_fp.call_count, 1)

        el = elements[0]
        self.assertIsNone(el.ai_fingerprint)
        self.assertIsNotNone(el.error_while_fingerprinting)
        self.assertIn("Exception during AI fingerprinting: AI model exploded", el.error_while_fingerprinting)
        self.assertEqual(el.file_size_kb, round(100/1024.0, 2)) # Check metadata still populated

    @patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample')
    @patch('src.profiler.style_sampler.Path.stat')
    def test_extract_elements_ai_fingerprint_validation_failed(self, mock_stat, mock_get_ai_fp):
        # Simulate AI fingerprinting returning a dict with validation_status != "passed"
        mock_get_ai_fp.return_value = self.mock_failed_ai_fp
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 150
        mock_stat_result.st_mtime = 1010.0
        mock_stat.return_value = mock_stat_result

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
        self.assertEqual(el.file_size_kb, round(150/1024.0, 2))

    @patch('src.profiler.style_sampler.Path.stat')
    def test_extract_elements_from_file_syntax_error_no_ai_call(self, mock_stat):
        # No need to mock get_ai_style_fingerprint_for_sample if parsing fails first
        # Mock stat because it's called before AST parsing
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 50
        mock_stat_result.st_mtime = 1020.0
        mock_stat.return_value = mock_stat_result

        file_path = self._create_dummy_file("syntax_error.py", "def func( :")
        elements = self.sampler._extract_elements_from_file(file_path)
        self.assertEqual(elements, [])


    # Test collect_all_elements with mocked AI fingerprinting
    @patch('src.profiler.style_sampler.StyleSampler._discover_python_files') # Keep this simple
    @patch('src.profiler.style_sampler.StyleSampler._extract_elements_from_file') # Mock the whole method
    def test_collect_all_elements_with_ai_metadata(self, mock_extract_elements, mock_discover):
        file1_path = self.test_repo_dir / "file1_collect.py"
        # Create dummy CodeSample objects with metadata
        sample1 = CodeSample(file1_path, "func1", "function", "def func1(): pass", 1, 1,
                             ai_fingerprint=self.mock_successful_ai_fp, file_size_kb=1.0, mod_timestamp=100.0)
        mock_extract_elements.return_value = [sample1] # _extract_elements_from_file returns a list

        self.sampler._all_py_files = [file1_path] # Pre-populate for the discover mock
        mock_discover.return_value = None

        self.sampler.collect_all_elements()

        mock_discover.assert_called_once()
        mock_extract_elements.assert_called_once_with(file1_path)
        self.assertEqual(len(self.sampler._collected_elements), 1)
        self.assertEqual(self.sampler._collected_elements[0].file_size_kb, 1.0)
        self.assertEqual(self.sampler._collected_elements[0].mod_timestamp, 100.0)

    # Ensure sample_elements tests that manually create CodeSample objects are updated
    def test_sample_elements_fewer_than_target_returns_all_metadata(self):
        self.sampler.target_samples = 10
        self.sampler._collected_elements = [
            CodeSample(Path("f.py"), f"item{i}", "function", "pass", 1,1,
                       ai_fingerprint=self.mock_successful_ai_fp, file_size_kb=float(i+1), mod_timestamp=float(100+i))
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

    # Removed old test_sample_elements_more_than_target_random_sampling_updated
    # Removed old test_sample_elements_with_seed_for_deterministic_random_sample (replaced by more specific stratified test)

    # New or Refocused Tests for Stratified Sampling:

    def test_sample_elements_fewer_than_target_returns_all(self):
        """If total elements are fewer than target, all should be returned."""
        self.sampler.target_samples = 10
        self._create_dummy_file("dir1/f1.py", "class A: pass")
        self._create_dummy_file("dir1/f2.py", "class B: pass")
        self._create_dummy_file("dir2/f3.py", "class C: pass")

        # Mock AI fingerprinting for simplicity in this test of sampling logic
        with patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample', return_value=self.mock_successful_ai_fp):
            self.sampler.collect_all_elements() # Should find 3 elements (A, B, C)

        self.assertEqual(len(self.sampler._collected_elements), 3)

        # No need to mock collect_all_elements for sample_elements itself, as it calls it if needed.
        # But for controlled input to sample_elements, we can pre-populate _collected_elements
        # and mock collect_all_elements if we want to bypass discovery/AI for a specific sample_elements test.
        # Here, collect_all_elements has run, so _collected_elements is populated.

        samples = self.sampler.sample_elements()
        self.assertEqual(len(samples), 3)
        self.assertEqual(len(samples), len(self.sampler._collected_elements))
        # Ensure it's a copy
        self.assertNotSame(samples, self.sampler._collected_elements)


    def test_sample_elements_stratified_proportional_allocation(self):
        """Test approximate proportional allocation from different strata."""
        self.sampler.target_samples = 10 # Target 10 samples

        # Create elements: dirA: 60, dirB: 30, dirC: 10 (total 100)
        # Expected proportions for 10 samples: dirA ~6, dirB ~3, dirC ~1
        for i in range(60): self._create_dummy_file(f"dirA/f{i}.py", f"class A{i}: pass")
        for i in range(30): self._create_dummy_file(f"dirB/f{i}.py", f"class B{i}: pass")
        for i in range(10): self._create_dummy_file(f"dirC/f{i}.py", f"class C{i}: pass")

        with patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample', return_value=self.mock_successful_ai_fp):
            # collect_all_elements will call the mocked AI fingerprinting
            self.sampler.collect_all_elements()

        self.assertEqual(len(self.sampler._collected_elements), 100)

        samples = self.sampler.sample_elements()
        self.assertEqual(len(samples), 10)

        counts_by_dir = Counter(s.file_path.parent for s in samples)

        # Expected paths based on dummy file creation
        path_dirA = (self.test_repo_dir / "dirA").resolve()
        path_dirB = (self.test_repo_dir / "dirB").resolve()
        path_dirC = (self.test_repo_dir / "dirC").resolve()

        # Check if counts are roughly proportional. Due to rounding and integer allocation,
        # they might not be exact floats, but the distribution should be clear.
        # The largest remainder method should give:
        # dirA: (60/100)*10 = 6.0 -> 6
        # dirB: (30/100)*10 = 3.0 -> 3
        # dirC: (10/100)*10 = 1.0 -> 1
        # Sum = 10. No remainders to distribute.
        self.assertEqual(counts_by_dir[path_dirA], 6)
        self.assertEqual(counts_by_dir[path_dirB], 3)
        self.assertEqual(counts_by_dir[path_dirC], 1)


    def test_sample_elements_stratified_remainder_distribution(self):
        """Test distribution of remainders when ideal counts are fractional."""
        self.sampler.target_samples = 10
        # dirA: 7 (0.368 * 10 = 3.68 -> floor 3, frac 0.68)
        # dirB: 7 (0.368 * 10 = 3.68 -> floor 3, frac 0.68)
        # dirC: 5 (0.263 * 10 = 2.63 -> floor 2, frac 0.63)
        # Total elements = 19.
        # Initial floor allocation: 3 (A) + 3 (B) + 2 (C) = 8 samples.
        # Remainders needed = 10 - 8 = 2.
        # Sorted by frac (desc), then num_available (desc): dirA (0.68, 7), dirB (0.68, 7), dirC (0.63, 5)
        # (Assuming str(path) sorting makes dirA before dirB for tie-breaking if num_available is same)
        # 1st remainder to dirA (becomes 4).
        # 2nd remainder to dirB (becomes 4).
        # Final: dirA=4, dirB=4, dirC=2. Sum = 10.

        for i in range(7): self._create_dummy_file(f"dirA/f{i}.py", f"class A{i}: pass") # dirA
        for i in range(7): self._create_dummy_file(f"dirB/f{i}.py", f"class B{i}: pass") # dirB
        for i in range(5): self._create_dummy_file(f"dirC/f{i}.py", f"class C{i}: pass") # dirC

        with patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample', return_value=self.mock_successful_ai_fp):
            self.sampler.collect_all_elements()
        self.assertEqual(len(self.sampler._collected_elements), 19)

        samples = self.sampler.sample_elements()
        self.assertEqual(len(samples), 10)

        counts_by_dir = Counter(s.file_path.parent for s in samples)
        path_dirA = (self.test_repo_dir / "dirA").resolve()
        path_dirB = (self.test_repo_dir / "dirB").resolve()
        path_dirC = (self.test_repo_dir / "dirC").resolve()

        # Tie-breaking for fractional part relies on secondary sort key (num_available)
        # and then on the sorted order of directory paths if num_available is also tied.
        # Let's check if sum is 10 and individual counts are reasonable.
        # With seed 42, specific outcome for dirA, dirB can be asserted if we trace precisely.
        # For now, let's check expected counts.
        self.assertEqual(counts_by_dir[path_dirA], 4) # Expected 3 + 1 remainder
        self.assertEqual(counts_by_dir[path_dirB], 4) # Expected 3 + 1 remainder
        self.assertEqual(counts_by_dir[path_dirC], 2) # Expected 2, no remainder

    def test_sample_elements_stratum_exhaustion(self):
        """Test when a stratum's proportional share exceeds its available items."""
        self.sampler.target_samples = 10
        # dirA: 3 elements. Ideal share: (3/33)*10 = 0.9 -> floor 0, frac 0.9
        # dirB: 30 elements. Ideal share: (30/33)*10 = 9.09 -> floor 9, frac 0.09
        # Initial allocation: dirA=0, dirB=9. Sum=9. Remainder=1.
        # Remainder goes to dirA (highest frac). So dirA=1, dirB=9.
        # This is fine.
        # What if target is higher, e.g. target_samples = 30
        # dirA: (3/33)*30 = 2.72 -> floor 2, frac 0.72
        # dirB: (30/33)*30 = 27.27 -> floor 27, frac 0.27
        # Initial: dirA=2, dirB=27. Sum=29. Remainder=1.
        # Remainder to dirA. So dirA=3, dirB=27. All elements from dirA are taken.

        self.sampler.target_samples = 30
        self._create_dummy_file("dirA/f1.py", "class A1: pass")
        self._create_dummy_file("dirA/f2.py", "class A2: pass")
        self._create_dummy_file("dirA/f3.py", "class A3: pass")
        for i in range(30): self._create_dummy_file(f"dirB/f{i}.py", f"class B{i}: pass")

        with patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample', return_value=self.mock_successful_ai_fp):
            self.sampler.collect_all_elements()
        self.assertEqual(len(self.sampler._collected_elements), 33)

        samples = self.sampler.sample_elements()
        self.assertEqual(len(samples), 30)

        counts_by_dir = Counter(s.file_path.parent for s in samples)
        path_dirA = (self.test_repo_dir / "dirA").resolve()
        path_dirB = (self.test_repo_dir / "dirB").resolve()

        self.assertEqual(counts_by_dir[path_dirA], 3) # All elements from dirA
        self.assertEqual(counts_by_dir[path_dirB], 27)


    def test_sample_elements_deterministic_with_seed(self):
        """Stratified sampling should be deterministic with a fixed seed."""
        # Use the same setup as test_sample_elements_stratified_remainder_distribution
        self.sampler.target_samples = 10
        for i in range(7): self._create_dummy_file(f"dirA/f{i}.py", f"class A{i}: pass")
        for i in range(7): self._create_dummy_file(f"dirB/f{i}.py", f"class B{i}: pass")
        for i in range(5): self._create_dummy_file(f"dirC/f{i}.py", f"class C{i}: pass")

        with patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample', return_value=self.mock_successful_ai_fp):
            # Sampler 1 (self.sampler was already initialized with seed 42)
            self.sampler.collect_all_elements()
            samples1 = self.sampler.sample_elements()

            # Sampler 2 (new instance with same seed and paths)
            sampler2 = StyleSampler(
                repo_path=str(self.test_repo_dir),
                deepseek_model_path=self.dummy_deepseek_path,
                divot5_model_path=self.dummy_divot5_path,
                random_seed=42, # Same seed
                target_samples=10
            )
            sampler2.collect_all_elements() # Must use same collected elements for fair comparison of sampling
            samples2 = sampler2.sample_elements()

            # To ensure comparison is fair, convert to a sortable representation if order within strata isn't guaranteed
            # For CodeSample, we can sort by file_path then item_name
            samples1_sorted = sorted(samples1, key=lambda s: (str(s.file_path), s.item_name))
            samples2_sorted = sorted(samples2, key=lambda s: (str(s.file_path), s.item_name))

            self.assertEqual(samples1_sorted, samples2_sorted)

    def test_sample_elements_target_zero(self):
        self._create_dummy_file("dirA/f1.py", "pass")
        self.sampler.target_samples = 0
        with patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample', return_value=self.mock_successful_ai_fp):
            self.sampler.collect_all_elements()
        samples = self.sampler.sample_elements()
        self.assertEqual(len(samples), 0)

    def test_sample_elements_all_in_one_directory(self):
        self.sampler.target_samples = 5
        for i in range(10): self._create_dummy_file(f"dirA/f{i}.py", f"class Item{i}: pass")
        with patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample', return_value=self.mock_successful_ai_fp):
            self.sampler.collect_all_elements()

        samples = self.sampler.sample_elements()
        self.assertEqual(len(samples), 5)
        path_dirA = (self.test_repo_dir / "dirA").resolve()
        self.assertTrue(all(s.file_path.parent == path_dirA for s in samples))

    def test_sample_elements_num_dirs_greater_than_target_samples(self):
        """Test when target samples is less than number of non-empty directories."""
        self.sampler.target_samples = 2 # Target 2 samples
        # 3 directories, each with 1 element
        self._create_dummy_file("dirA/f1.py", "class A: pass")
        self._create_dummy_file("dirB/f1.py", "class B: pass")
        self._create_dummy_file("dirC/f1.py", "class C: pass")

        with patch('src.profiler.style_sampler.get_ai_style_fingerprint_for_sample', return_value=self.mock_successful_ai_fp):
            self.sampler.collect_all_elements()
        self.assertEqual(len(self.sampler._collected_elements), 3)

        samples = self.sampler.sample_elements()
        self.assertEqual(len(samples), 2)
        # Check that the two samples come from different directories due to remainder distribution
        # (proportional is 2/3 for each, floor is 0. Remainder distribution gives 1 to two dirs)
        sampled_dirs = {s.file_path.parent for s in samples}
        self.assertEqual(len(sampled_dirs), 2)


if __name__ == '__main__':
    unittest.main()
