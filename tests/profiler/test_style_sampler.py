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

class TestStyleSampler(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to simulate a repo
        self.test_repo_dir = Path(tempfile.mkdtemp(prefix="test_repo_"))
        self.sampler = StyleSampler(str(self.test_repo_dir), random_seed=42)

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
        expected_paths = sorted([Path("main.py"), Path("project/actual_code.py")])

        self.assertEqual(discovered_paths, expected_paths)

    def test_discover_python_files_empty_repo(self):
        self.sampler._discover_python_files()
        self.assertEqual(len(self.sampler._all_py_files), 0)

    def test_extract_elements_from_file_basic(self):
        content = """
"""Module docstring."""
class MyClass:
    """Class docstring."""
    def my_method(self, x):
        """Method docstring."""
        return x
def my_function():
    """Function docstring."""
    pass
"""
        file_path = self._create_dummy_file("test_module.py", content)
        elements = self.sampler._extract_elements_from_file(file_path)

        self.assertTrue(len(elements) >= 4) # Module doc, class, class doc, method, method doc, func, func doc -> could be 7

        item_names = {el.item_name for el in elements}
        self.assertIn("test_module.py::module_docstring", item_names)
        self.assertIn("MyClass", item_names)
        self.assertIn("MyClass::docstring", item_names)
        self.assertIn("my_method", item_names) # FunctionDef node name
        self.assertIn("my_method::docstring", item_names) # Its docstring
        self.assertIn("my_function", item_names)
        self.assertIn("my_function::docstring", item_names)

        for el in elements:
            self.assertEqual(el.file_path, file_path)
            if el.item_name == "MyClass":
                self.assertEqual(el.item_type, "class")
                self.assertIn("class MyClass:", el.code_snippet)
            elif el.item_name == "my_method":
                self.assertEqual(el.item_type, "function") # AST treats methods as FunctionDef
                self.assertIn("def my_method(self, x):", el.code_snippet)
            elif el.item_name == "MyClass::docstring":
                 self.assertEqual(el.item_type, "docstring")
                 self.assertEqual(el.code_snippet, "Class docstring.")


    def test_extract_elements_from_file_no_docstrings(self):
        content = """
class NoDocClass:
    def no_doc_method(self):
        pass
def no_doc_func():
    pass
"""
        file_path = self._create_dummy_file("no_docs_module.py", content)
        elements = self.sampler._extract_elements_from_file(file_path)

        # Expected: class, method, function. No separate docstring elements.
        # Module docstring is also None.
        docstring_elements = [el for el in elements if el.item_type == "docstring"]
        # The current _extract_elements_from_file might still create a "module_docstring" CodeSample with snippet=None
        # if ast.get_docstring(tree) returns None. Let's check this behavior.
        # Based on current style_sampler, it only adds docstring if `module_docstring` is truthy.
        self.assertEqual(len(docstring_elements), 0, f"Found docstring elements: {[el.item_name for el in docstring_elements]}")

        item_names = {el.item_name for el in elements}
        self.assertIn("NoDocClass", item_names)
        self.assertIn("no_doc_method", item_names)
        self.assertIn("no_doc_func", item_names)
        self.assertNotIn("no_docs_module.py::module_docstring", item_names)


    def test_extract_elements_from_file_syntax_error(self):
        file_path = self._create_dummy_file("syntax_error.py", "def func( :")
        elements = self.sampler._extract_elements_from_file(file_path)
        self.assertEqual(elements, []) # Expect graceful failure (empty list)

    def test_extract_elements_from_file_empty_file(self):
        file_path = self._create_dummy_file("empty.py", "")
        elements = self.sampler._extract_elements_from_file(file_path)
        self.assertEqual(elements, [])


    @patch('src.profiler.style_sampler.StyleSampler._extract_elements_from_file')
    @patch('src.profiler.style_sampler.StyleSampler._discover_python_files')
    def test_collect_all_elements(self, mock_discover, mock_extract):
        # Setup mock for _discover_python_files
        file1_path = self.test_repo_dir / "file1.py"
        file2_path = self.test_repo_dir / "file2.py"
        self.sampler._all_py_files = [file1_path, file2_path] # Pre-populate for the mock
        mock_discover.return_value = None # _discover_python_files modifies self._all_py_files

        # Setup mock for _extract_elements_from_file
        sample1 = CodeSample(file1_path, "func1", "function", "def func1(): pass", 1, 1)
        sample2 = CodeSample(file2_path, "ClassA", "class", "class ClassA: pass", 1, 1)
        mock_extract.side_effect = [[sample1], [sample2]]

        self.sampler.collect_all_elements()

        mock_discover.assert_called_once()
        self.assertEqual(mock_extract.call_count, 2)
        mock_extract.assert_any_call(file1_path)
        mock_extract.assert_any_call(file2_path)

        self.assertEqual(self.sampler._collected_elements, [sample1, sample2])

    def test_sample_elements_fewer_than_target(self):
        self.sampler.target_samples = 10
        # Manually populate _collected_elements for this test
        self.sampler._collected_elements = [
            CodeSample(Path("f.py"), f"item{i}", "function", "pass", 1,1) for i in range(5)
        ]
        # Mock collect_all_elements to prevent it from running and overriding our manual population
        with patch.object(self.sampler, 'collect_all_elements', MagicMock()):
            samples = self.sampler.sample_elements()
            self.assertEqual(len(samples), 5)
            self.assertEqual(samples, self.sampler._collected_elements)


    def test_sample_elements_more_than_target_random_sampling(self):
        self.sampler.target_samples = 5
        self.sampler._collected_elements = [
             CodeSample(Path("f.py"), f"item{i}", "function", "pass", 1,1) for i in range(10)
        ]
        with patch.object(self.sampler, 'collect_all_elements', MagicMock()):
            samples = self.sampler.sample_elements()
            self.assertEqual(len(samples), 5)
            # Check if all sampled items are from the original list (subset)
            for sample in samples:
                self.assertIn(sample, self.sampler._collected_elements)

    def test_sample_elements_with_seed_for_deterministic_random_sample(self):
        # This test relies on the current mock behavior of random.sample in sample_elements
        self.sampler.target_samples = 3
        self.sampler._collected_elements = [
             CodeSample(Path("f.py"), f"item{i}", "function", "pass", 1,1) for i in range(5)
        ]

        # First run
        with patch.object(self.sampler, 'collect_all_elements', MagicMock()):
            sampler1 = StyleSampler(str(self.test_repo_dir), target_samples=3, random_seed=123)
            sampler1._collected_elements = self.sampler._collected_elements[:] # Use same collected elements
            samples1 = sampler1.sample_elements()

            # Second run with the same seed
            sampler2 = StyleSampler(str(self.test_repo_dir), target_samples=3, random_seed=123)
            sampler2._collected_elements = self.sampler._collected_elements[:]
            samples2 = sampler2.sample_elements()

            self.assertEqual(samples1, samples2, "Samples should be deterministic with the same seed.")

            # Third run with a different seed (or no seed if StyleSampler re-seeds globally)
            # Our StyleSampler sets random.seed() in __init__ if seed is provided.
            # So, a new instance with a different seed will produce different results.
            sampler3 = StyleSampler(str(self.test_repo_dir), target_samples=3, random_seed=456)
            sampler3._collected_elements = self.sampler._collected_elements[:]
            samples3 = sampler3.sample_elements()

            if len(self.sampler._collected_elements) > sampler3.target_samples : # only if actual sampling happens
                 self.assertNotEqual(samples1, samples3, "Samples should differ with different seeds if sampling occurs.")


if __name__ == '__main__':
    unittest.main()
