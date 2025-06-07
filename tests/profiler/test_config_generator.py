import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import toml # For creating expected TOML structures in tests
from pathlib import Path
import tempfile
import shutil

# Adjust import path as necessary
from src.profiler.config_generator import (
    generate_black_config_in_pyproject,
    generate_ruff_config_in_pyproject,
    generate_docstring_template_file,
    DEFAULT_PYPROJECT_PATH, # Though tests will use custom paths
    DEFAULT_DOCSTRING_TEMPLATE_PATH
)

class TestConfigGenerator(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_config_gen_"))
        self.dummy_pyproject_path = self.test_dir / "pyproject.toml"
        self.dummy_doc_template_path = self.test_dir / "docstring_template.py"

        self.profile_default = {
            "max_line_length": 88,
            "preferred_quotes": "single",
            "docstring_style": "google",
        }
        self.profile_strict = {
            "max_line_length": 79,
            "preferred_quotes": "double",
            "docstring_style": "numpy",
        }
        self.profile_mixed_quotes = { # For black's skip-string-normalization
            "max_line_length": 90,
            "preferred_quotes": None,
            "docstring_style": "pep257",
        }


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # --- Tests for generate_black_config_in_pyproject ---

    @patch('src.profiler.config_generator.toml.dump')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    @patch('src.profiler.config_generator.Path.exists', return_value=False) # Simulate file not existing
    def test_generate_black_new_pyproject(self, mock_exists, mock_file_open, mock_toml_dump):
        generate_black_config_in_pyproject(self.profile_default, self.dummy_pyproject_path)

        mock_exists.assert_called_once()
        mock_file_open.assert_called_once_with(self.dummy_pyproject_path, "w", encoding="utf-8")

        # Check the data passed to toml.dump
        args, _ = mock_toml_dump.call_args
        dumped_data = args[0]
        self.assertIn("tool", dumped_data)
        self.assertIn("black", dumped_data["tool"])
        self.assertEqual(dumped_data["tool"]["black"]["line_length"], 88)
        self.assertNotIn("skip-string-normalization", dumped_data["tool"]["black"]) # Default is false

    @patch('src.profiler.config_generator.toml.dump')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    @patch('src.profiler.config_generator.Path.exists', return_value=False)
    def test_generate_black_skip_string_normalization(self, mock_exists, mock_file_open, mock_toml_dump):
        generate_black_config_in_pyproject(self.profile_mixed_quotes, self.dummy_pyproject_path)
        args, _ = mock_toml_dump.call_args
        dumped_data = args[0]
        self.assertTrue(dumped_data["tool"]["black"]["skip-string-normalization"])


    @patch('src.profiler.config_generator.toml.dump')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    @patch('src.profiler.config_generator.Path.exists', return_value=True) # Simulate file existing
    @patch('src.profiler.config_generator.toml.load') # Mock toml.load
    def test_generate_black_update_existing_pyproject(self, mock_toml_load, mock_exists, mock_file_open, mock_toml_dump):
        # Simulate existing data in pyproject.toml
        existing_data = {"project": {"name": "test"}, "tool": {"other_tool": {}}}
        mock_toml_load.return_value = existing_data

        generate_black_config_in_pyproject(self.profile_default, self.dummy_pyproject_path)

        mock_toml_load.assert_called_once()
        mock_file_open.assert_called_once_with(self.dummy_pyproject_path, "w", encoding="utf-8")

        args, _ = mock_toml_dump.call_args
        dumped_data = args[0]
        self.assertEqual(dumped_data["project"]["name"], "test") # Preserved
        self.assertIn("black", dumped_data["tool"])
        self.assertEqual(dumped_data["tool"]["black"]["line_length"], 88)


    # --- Tests for generate_ruff_config_in_pyproject ---

    @patch('src.profiler.config_generator.toml.dump')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    @patch('src.profiler.config_generator.Path.exists', return_value=False)
    def test_generate_ruff_new_pyproject(self, mock_exists, mock_file_open, mock_toml_dump):
        generate_ruff_config_in_pyproject(self.profile_default, self.dummy_pyproject_path)

        mock_exists.assert_called_once()
        mock_file_open.assert_called_once_with(self.dummy_pyproject_path, "w", encoding="utf-8")

        args, _ = mock_toml_dump.call_args
        dumped_data = args[0]
        self.assertIn("tool", dumped_data)
        self.assertIn("ruff", dumped_data["tool"])
        ruff_conf = dumped_data["tool"]["ruff"]
        self.assertEqual(ruff_conf["line-length"], 88)
        self.assertEqual(sorted(ruff_conf["lint"]["select"]), sorted(["E", "F", "W", "D", "Q"])) # D for pydocstyle, Q for quotes
        self.assertEqual(ruff_conf["lint"]["pydocstyle"]["convention"], "google")
        self.assertEqual(ruff_conf["format"]["quote-style"], "single")
        self.assertEqual(ruff_conf["lint"]["flake8-quotes"]["docstring-quotes"], "double")


    @patch('src.profiler.config_generator.toml.dump')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    @patch('src.profiler.config_generator.Path.exists', return_value=True)
    @patch('src.profiler.config_generator.toml.load')
    def test_generate_ruff_update_existing_pyproject(self, mock_toml_load, mock_exists, mock_file_open, mock_toml_dump):
        existing_data = {"project": {"name": "test-ruff"}, "tool": {"black": {"line_length": 100}}}
        mock_toml_load.return_value = existing_data

        generate_ruff_config_in_pyproject(self.profile_strict, self.dummy_pyproject_path)

        args, _ = mock_toml_dump.call_args
        dumped_data = args[0]
        self.assertEqual(dumped_data["project"]["name"], "test-ruff") # Preserved
        self.assertEqual(dumped_data["tool"]["black"]["line_length"], 100) # Preserved Black settings
        self.assertIn("ruff", dumped_data["tool"])
        ruff_conf = dumped_data["tool"]["ruff"]
        self.assertEqual(ruff_conf["line-length"], 79)
        self.assertEqual(ruff_conf["lint"]["pydocstyle"]["convention"], "numpy")
        self.assertEqual(ruff_conf["format"]["quote-style"], "double")

    # --- Tests for generate_docstring_template_file ---

    @patch('src.profiler.config_generator.Path.mkdir')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    def test_generate_docstring_template_google(self, mock_file_open, mock_mkdir):
        generate_docstring_template_file(self.profile_default, self.dummy_doc_template_path)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_file_open.assert_called_once_with(self.dummy_doc_template_path, "w", encoding="utf-8")

        # Check content written to file
        handle = mock_file_open()
        written_content = handle.write.call_args[0][0]
        self.assertIn("Docstring Template: Google Style", written_content)
        self.assertIn("Args:", written_content)
        self.assertIn("Returns:", written_content)
        self.assertIn("from typing import Optional", written_content)


    @patch('src.profiler.config_generator.Path.mkdir')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    def test_generate_docstring_template_numpy(self, mock_file_open, mock_mkdir):
        generate_docstring_template_file(self.profile_strict, self.dummy_doc_template_path)
        handle = mock_file_open()
        written_content = handle.write.call_args[0][0]
        self.assertIn("Docstring Template: Numpy Style", written_content)
        self.assertIn("Parameters", written_content)
        self.assertIn("----------", written_content)
        self.assertIn("Returns", written_content)
        self.assertIn("-------", written_content)

    @patch('src.profiler.config_generator.Path.mkdir')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    def test_generate_docstring_template_pep257(self, mock_file_open, mock_mkdir):
        profile_pep257 = {"docstring_style": "pep257"}
        generate_docstring_template_file(profile_pep257, self.dummy_doc_template_path)
        handle = mock_file_open()
        written_content = handle.write.call_args[0][0]
        self.assertIn("Docstring Template: PEP 257 / reStructuredText Style", written_content)
        self.assertIn(":param ", written_content)
        self.assertIn(":type ", written_content)
        self.assertIn(":returns:", written_content)
        self.assertIn(":rtype:", written_content)

    @patch('src.profiler.config_generator.Path.mkdir')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    def test_generate_docstring_template_fallback(self, mock_file_open, mock_mkdir):
        profile_other = {"docstring_style": "other_style"}
        generate_docstring_template_file(profile_other, self.dummy_doc_template_path)
        handle = mock_file_open()
        written_content = handle.write.call_args[0][0]
        self.assertIn("Docstring Template: Other_style Style", written_content) # Capitalized
        self.assertIn("not explicitly templated", written_content)


    # --- Integration-like tests that use the file system ---
    def test_black_config_generation_integration(self):
        generate_black_config_in_pyproject(self.profile_default, self.dummy_pyproject_path)
        self.assertTrue(self.dummy_pyproject_path.exists())
        with open(self.dummy_pyproject_path, "r") as f:
            data = toml.load(f)
        self.assertEqual(data["tool"]["black"]["line_length"], 88)

    def test_ruff_config_generation_integration(self):
        # Start with a pyproject.toml that has black settings
        generate_black_config_in_pyproject(self.profile_default, self.dummy_pyproject_path)
        generate_ruff_config_in_pyproject(self.profile_default, self.dummy_pyproject_path)

        self.assertTrue(self.dummy_pyproject_path.exists())
        with open(self.dummy_pyproject_path, "r") as f:
            data = toml.load(f)
        self.assertEqual(data["tool"]["ruff"]["line-length"], 88)
        self.assertEqual(data["tool"]["ruff"]["lint"]["pydocstyle"]["convention"], "google")
        self.assertEqual(data["tool"]["black"]["line_length"], 88) # Ensure Black settings preserved

    def test_docstring_template_generation_integration(self):
        generate_docstring_template_file(self.profile_default, self.dummy_doc_template_path)
        self.assertTrue(self.dummy_doc_template_path.exists())
        with open(self.dummy_doc_template_path, "r") as f:
            content = f.read()
        self.assertIn("Docstring Template: Google Style", content)


if __name__ == '__main__':
    unittest.main()
