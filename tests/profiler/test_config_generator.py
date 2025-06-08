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
# Add sqlite3 and any other necessary types
import sqlite3
from typing import Dict, Any, List, Tuple # Ensure List, Tuple are imported

# (Existing imports for functions from config_generator, database_setup, naming_conventions)
from src.profiler.database_setup import create_naming_conventions_db
from src.profiler.naming_conventions import NAMING_CONVENTIONS_REGEX # For inserting regexes into mock DB


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

    # Helper to set up a mock DB with specific active rules for a test
    def _setup_mock_db_with_rules(self, db_path: Path, active_rules: Dict[str, str]):
        # active_rules = {"function": "CAMEL_CASE", "variable": "SNAKE_CASE", ...}
        create_naming_conventions_db(db_path=db_path) # Ensure schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        # Deactivate all first for a clean slate for this test's active rules
        # Assuming IDENTIFIER_TYPES is accessible or redefined here for test setup
        # from src.profiler.naming_conventions import IDENTIFIER_TYPES
        # For now, using a hardcoded list, or ensure IDENTIFIER_TYPES is imported.
        # It's better to import it if it's part of the public API of naming_conventions.
        # Let's assume it's imported or defined in this test file for helper.
        temp_identifier_types = ["constant", "class", "function", "variable", "test_function", "test_class"]
        for id_type in temp_identifier_types:
            cursor.execute("UPDATE naming_rules SET is_active = FALSE WHERE identifier_type = ?", (id_type,))

        for id_type, convention_name in active_rules.items():
            if convention_name in NAMING_CONVENTIONS_REGEX:
                regex_pattern = NAMING_CONVENTIONS_REGEX[convention_name]
                # INSERT OR REPLACE to ensure the rule exists, then update its active status
                cursor.execute("""
                    INSERT OR REPLACE INTO naming_rules
                    (identifier_type, convention_name, regex_pattern, description, is_active)
                    VALUES (?, ?, ?, ?, ?)
                """, (id_type, convention_name, regex_pattern, f"Mock active rule for {id_type}", False)) # Insert/replace as inactive first
                cursor.execute("""
                    UPDATE naming_rules SET is_active = TRUE
                    WHERE identifier_type = ? AND convention_name = ?
                """, (id_type, convention_name))
            else:
                # If a test specifies a convention not in NAMING_CONVENTIONS_REGEX, it's a test setup error
                raise ValueError(f"Convention {convention_name} not in NAMING_CONVENTIONS_REGEX for testing.")
        conn.commit()
        conn.close()

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


    # --- Update existing tests or add new ones for generate_ruff_config_in_pyproject ---

    @patch('src.profiler.config_generator.toml.dump')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    @patch('src.profiler.config_generator.Path.exists', return_value=False) # New pyproject
    def test_generate_ruff_new_pyproject_with_db_snake_dominant(self, mock_exists, mock_file_open, mock_toml_dump):
        """Test Ruff config with DB: snake_case dominant for func/var, N rules NOT ignored."""
        mock_db_path = self.test_dir / "mock_rules_snake.db"
        active_rules = {
            "function": "SNAKE_CASE", "variable": "SNAKE_CASE",
            "class": "PASCAL_CASE", "constant": "UPPER_SNAKE_CASE"
        }
        self._setup_mock_db_with_rules(mock_db_path, active_rules)

        generate_ruff_config_in_pyproject(self.profile_default, self.dummy_pyproject_path, db_path=mock_db_path)

        args, _ = mock_toml_dump.call_args
        dumped_data = args[0]
        ruff_lint_conf = dumped_data.get("tool", {}).get("ruff", {}).get("lint", {})

        self.assertIn("N", ruff_lint_conf.get("select", []))
        # Since dominant styles match PEP8, no N-rules should be specifically ignored due to conflict
        self.assertNotIn("N801", ruff_lint_conf.get("ignore", [])) # Class PascalCase is fine
        self.assertNotIn("N802", ruff_lint_conf.get("ignore", [])) # Function snake_case is fine
        self.assertNotIn("N803", ruff_lint_conf.get("ignore", [])) # Variable (for arg) snake_case is fine
        self.assertNotIn("N806", ruff_lint_conf.get("ignore", [])) # Variable (in func) snake_case is fine
        self.assertNotIn("N815", ruff_lint_conf.get("ignore", [])) # Variable (class scope) snake_case is fine for not being mixed
        self.assertNotIn("N816", ruff_lint_conf.get("ignore", [])) # Constant UPPER_SNAKE_CASE is fine for not being mixed global

    @patch('src.profiler.config_generator.toml.dump')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    @patch('src.profiler.config_generator.Path.exists', return_value=False) # New pyproject
    def test_generate_ruff_new_pyproject_with_db_camel_dominant_func_var(self, mock_exists, mock_file_open, mock_toml_dump):
        """Test Ruff config with DB: camelCase dominant for func/var, relevant N rules ignored."""
        mock_db_path = self.test_dir / "mock_rules_camel.db"
        active_rules = {
            "function": "CAMEL_CASE", "variable": "CAMEL_CASE",
            "class": "PASCAL_CASE", "constant": "UPPER_SNAKE_CASE" # Constants are still PEP8
        }
        self._setup_mock_db_with_rules(mock_db_path, active_rules)

        # Profile for other settings (line length, quotes, etc.)
        profile_for_camel_test = {
            "max_line_length": 80, "preferred_quotes": "double", "docstring_style": "numpy"
        }
        generate_ruff_config_in_pyproject(profile_for_camel_test, self.dummy_pyproject_path, db_path=mock_db_path)

        args, _ = mock_toml_dump.call_args
        dumped_data = args[0]
        ruff_lint_conf = dumped_data.get("tool", {}).get("ruff", {}).get("lint", {})

        self.assertIn("N", ruff_lint_conf.get("select", []))
        ignored_rules = set(ruff_lint_conf.get("ignore", []))

        self.assertIn("N802", ignored_rules) # Function name (expects snake_case)
        self.assertIn("N803", ignored_rules) # Argument name (expects snake_case)
        self.assertIn("N806", ignored_rules) # Variable in function (expects snake_case)
        self.assertIn("N815", ignored_rules) # MixedCase var in class scope (camelCase is mixed)
        self.assertIn("N816", ignored_rules) # MixedCase var in global scope (camelCase is mixed, but this rule might also catch non-UPPER_SNAKE_CASE constants if they were camelCase)
                                             # Our constant is UPPER_SNAKE_CASE, so N816 is ignored because vars are camelCase.
        self.assertNotIn("N801", ignored_rules) # Class PascalCase is fine

    @patch('src.profiler.config_generator.toml.dump')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    @patch('src.profiler.config_generator.Path.exists', return_value=False)
    def test_generate_ruff_db_non_pep8_constants(self, mock_exists, mock_file_open, mock_toml_dump):
        """Test Ruff config when constants are, e.g., PascalCase."""
        mock_db_path = self.test_dir / "mock_rules_pascal_const.db"
        active_rules = {
            "function": "SNAKE_CASE", "variable": "SNAKE_CASE",
            "class": "PASCAL_CASE", "constant": "PASCAL_CASE" # Non-PEP8 constants
        }
        self._setup_mock_db_with_rules(mock_db_path, active_rules)
        generate_ruff_config_in_pyproject(self.profile_default, self.dummy_pyproject_path, db_path=mock_db_path)

        args, _ = mock_toml_dump.call_args
        dumped_data = args[0]
        ruff_lint_conf = dumped_data.get("tool", {}).get("ruff", {}).get("lint", {})
        ignored_rules = set(ruff_lint_conf.get("ignore", []))

        # N816 flags mixedCase in global scope. If constants are PascalCase (a form of mixedCase),
        # then N816 should be ignored.
        self.assertIn("N816", ignored_rules)


    @patch('src.profiler.config_generator.toml.dump')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    @patch('src.profiler.config_generator.Path.exists', return_value=True) # Existing pyproject
    @patch('src.profiler.config_generator.toml.load')
    def test_generate_ruff_update_existing_with_db(self, mock_toml_load, mock_exists, mock_file_open, mock_toml_dump):
        """Test updating an existing pyproject.toml with DB-driven N-rule ignores."""
        existing_data = {
            "project": {"name": "test-ruff-db"},
            "tool": {
                "ruff": {
                    "lint": {"select": ["E", "F"], "ignore": ["D100"]}
                }
            }
        }
        mock_toml_load.return_value = existing_data

        mock_db_path = self.test_dir / "mock_rules_camel_update.db"
        active_rules = {"function": "CAMEL_CASE", "variable": "CAMEL_CASE"} # Others default or not set
        self._setup_mock_db_with_rules(mock_db_path, active_rules)

        generate_ruff_config_in_pyproject(self.profile_default, self.dummy_pyproject_path, db_path=mock_db_path)

        args, _ = mock_toml_dump.call_args
        dumped_data = args[0]
        ruff_lint_conf = dumped_data.get("tool", {}).get("ruff", {}).get("lint", {})

        self.assertIn("N", ruff_lint_conf.get("select", []))
        self.assertIn("E", ruff_lint_conf.get("select", [])) # Original E preserved

        expected_ignores = {"D100", "N802", "N803", "N806", "N815", "N816"} # N801 (class) and const rules not ignored as per active_rules
        self.assertEqual(set(ruff_lint_conf.get("ignore", [])), expected_ignores)


    @patch('src.profiler.config_generator.toml.dump')
    @patch('src.profiler.config_generator.open', new_callable=mock_open)
    @patch('src.profiler.config_generator.Path.exists', return_value=False)
    def test_generate_ruff_no_db_path_provided(self, mock_exists, mock_file_open, mock_toml_dump):
        """Test Ruff config when no db_path is provided, N rules use defaults."""
        generate_ruff_config_in_pyproject(self.profile_default, self.dummy_pyproject_path, db_path=None)

        args, _ = mock_toml_dump.call_args
        dumped_data = args[0]
        ruff_lint_conf = dumped_data.get("tool", {}).get("ruff", {}).get("lint", {})

        self.assertIn("N", ruff_lint_conf.get("select", [])) # N should still be selected by default
        # If db_path is None, rules_to_potentially_ignore remains empty, so current_ignore will be empty unless already populated.
        # The test for new pyproject, lint.ignore should not be present if current_ignore is empty.
        self.assertNotIn("ignore", ruff_lint_conf)

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
