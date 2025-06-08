import unittest
from unittest.mock import patch, MagicMock, call, ANY
import sqlite3
from pathlib import Path
import tempfile
import shutil
import subprocess # For CompletedProcess
import os # For os.path.join for creating subdirs in temp

# Adjust imports
from src.style_scorer import score_style, AstIdentifierExtractor, get_active_naming_rules # get_active_naming_rules for test setup
from src.profiler.database_setup import create_naming_conventions_db
from src.profiler.naming_conventions import NAMING_CONVENTIONS_REGEX, populate_naming_rules_from_profile

class TestScoreStyle(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_score_style_"))
        self.sample_file_path = self.test_dir / "sample.py"
        self.db_path = self.test_dir / "naming.db"

        # Create schema for every test, populate as needed
        create_naming_conventions_db(self.db_path)

        self.default_profile = {
            "max_line_length": 80,
            "preferred_quotes": "single",
            "docstring_style": "google", # Means docstrings are expected for public items
            "directory_overrides": {},
            # Identifier percentages are not directly used by score_style, but by DB population
        }
        # Populate DB with some default "good" rules for a baseline
        populate_naming_rules_from_profile(self.db_path, {
            "identifier_snake_case_pct": 0.9,
            "identifier_camelCase_pct": 0.05,
            "identifier_UPPER_SNAKE_CASE_pct": 0.9
        }) # This sets snake_case for func/var, Pascal for class, UPPER for const

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write_sample_code(self, code: str):
        with open(self.sample_file_path, "w", encoding="utf-8") as f:
            f.write(code)

    @patch('src.style_scorer.shutil.which')
    @patch('src.style_scorer.subprocess.run')
    def test_score_perfectly_conforming_code(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.return_value = "/fake/path/to/tool" # Both tools found
        # Black --check returns 0 (no changes), Ruff check returns 0 (no issues)
        mock_subprocess_run.side_effect = [
            subprocess.CompletedProcess(args=ANY, returncode=0, stdout="", stderr=""), # Black
            subprocess.CompletedProcess(args=ANY, returncode=0, stdout="", stderr="")  # Ruff
        ]
        code = """
\"\"\"Module docstring.\"\"\"
A_CONSTANT = 10
def my_function(param_a: int) -> str:
    \"\"\"This is a good docstring.

    Args:
        param_a (int): An integer.

    Returns:
        str: A string.
    \"\"\"
    my_variable = param_a
    return str(my_variable)

class MyClass:
    \"\"\"A good class docstring.\"\"\"
    def __init__(self):
        \"\"\"Init docstring.\"\"\"
        pass
"""
        self._write_sample_code(code)
        score = score_style(self.sample_file_path, self.default_profile, self.db_path)
        self.assertAlmostEqual(score, 0.0, places=2, msg=f"Score was {score}, expected ~0.0")

    @patch('src.style_scorer.shutil.which')
    @patch('src.style_scorer.subprocess.run')
    def test_score_black_ruff_formatting_issues(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.return_value = "/fake/path/to/tool"
        # Black --check returns 1 (changes needed), Ruff check returns 1 (issues found)
        mock_subprocess_run.side_effect = [
            subprocess.CompletedProcess(args=ANY, returncode=1, stdout="diff...", stderr=""), # Black finds issues
            subprocess.CompletedProcess(args=ANY, returncode=1, stdout="issues...", stderr="")  # Ruff finds issues
        ]
        self._write_sample_code("def foo (): pass") # Content doesn't matter as much as mock returns

        # Test with only formatting weights active
        score = score_style(self.sample_file_path, self.default_profile, self.db_path,
                              w_black_diff=0.5, w_ruff_diff=0.5, # These are the only active weights
                              w_naming=0, w_linelen=0, w_quotes=0, w_docstrings=0)
        # Normalization: max_penalty = w_black (0.5) + w_ruff (0.5) = 1.0
        # Total penalty = 0.5 (black) + 0.5 (ruff) = 1.0
        # Score = 1.0 / 1.0 = 1.0
        self.assertAlmostEqual(score, 1.0, places=2, msg=f"Score was {score}")


    @patch('src.style_scorer.shutil.which')
    @patch('src.style_scorer.subprocess.run')
    def test_score_naming_violations(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.return_value = "/fake/path/to/tool"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=ANY, returncode=0) # Black/Ruff OK

        code = "def MyFunction(): pass\nclass my_class: pass\nMY_VAR = 1"
        # MyFunction (FAIL vs snake_case), my_class (FAIL vs PascalCase)
        # MY_VAR (OK for const if rule is UPPER_SNAKE_CASE and heuristic works)
        # AstIdentifierExtractor: [('MyFunction', 'function'), ('my_class', 'class'), ('MY_VAR', 'variable')]
        # Heuristic: MY_VAR matches UPPER_SNAKE_CASE, so it's checked against 'constant' rule (which is UPPER_SNAKE_CASE here). -> OK
        # Violations: MyFunction, my_class. Total relevant: MyFunction, my_class, MY_VAR. Rate = 2/3.
        self._write_sample_code(code)

        score = score_style(self.sample_file_path, self.default_profile, self.db_path,
                              w_black_diff=0, w_ruff_diff=0, w_linelen=0, w_quotes=0, w_docstrings=0, w_naming=1.0)
        expected_score = (2/3) * 1.0
        self.assertAlmostEqual(score, expected_score, places=2, msg=f"Score was {score}")


    @patch('src.style_scorer.shutil.which')
    @patch('src.style_scorer.subprocess.run')
    def test_score_line_length_violations(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.return_value = "/fake/path/to/tool"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=ANY, returncode=0)

        profile = self.default_profile.copy()
        profile["max_line_length"] = 10
        code = "short\nthis_is_a_very_long_line\nshort_again" # 1 of 3 lines is overlong
        self._write_sample_code(code)

        score = score_style(self.sample_file_path, profile, self.db_path,
                              w_black_diff=0, w_ruff_diff=0, w_naming=0, w_quotes=0, w_docstrings=0, w_linelen=1.0)
        expected_score = (1/3) * 1.0
        self.assertAlmostEqual(score, expected_score, places=2, msg=f"Score was {score}")

    @patch('src.style_scorer.shutil.which')
    @patch('src.style_scorer.subprocess.run')
    def test_score_quote_violations(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.return_value = "/fake/path/to/tool"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=ANY, returncode=0)

        profile = self.default_profile.copy()
        profile["preferred_quotes"] = "single"
        code = "s1 = 'ok'\nd1 = \"NOT OK\"\ns2 = 'ok'\nd2 = \"NOT OK EITHER\""
        # single_q_count = 2, double_q_count = 2. total_quotes = 4.
        # Preferred "single". Penalty logic: if double_q_count > single_q_count (false here), add penalty.
        # This logic needs to be proportional to non-preferred quotes.
        # Assuming corrected proportional logic: (double_q_count / total_quotes) = 2/4 = 0.5
        self._write_sample_code(code)

        score = score_style(self.sample_file_path, profile, self.db_path,
                              w_black_diff=0, w_ruff_diff=0, w_naming=0, w_linelen=0, w_docstrings=0, w_quotes=1.0)
        expected_score = (2/4) * 1.0 # Based on 2 double quotes out of 4 total being non-preferred
        self.assertAlmostEqual(score, expected_score, places=2, msg=f"Score was {score}")


    @patch('src.style_scorer.shutil.which')
    @patch('src.style_scorer.subprocess.run')
    def test_score_missing_docstrings(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.return_value = "/fake/path/to/tool"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=ANY, returncode=0)

        code = "\"\"\"Module doc OK.\"\"\"\ndef public_func_no_doc(): pass\nclass PublicClassNoDoc: pass\nclass _Private: pass"
        # Module (1 ok), func (1 missing), class (1 missing), _Private (1 ignored). Total public definable = 3. Missing = 2.
        self._write_sample_code(code)
        profile = self.default_profile.copy()
        profile["docstring_style"] = "google"

        score = score_style(self.sample_file_path, profile, self.db_path,
                              w_black_diff=0, w_ruff_diff=0, w_naming=0, w_linelen=0, w_quotes=0, w_docstrings=1.0)
        expected_score = (2/3) * 1.0
        self.assertAlmostEqual(score, expected_score, places=2, msg=f"Score was {score}")


    @patch('src.style_scorer.shutil.which')
    @patch('src.style_scorer.subprocess.run')
    def test_score_directory_override(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.return_value = "/fake/path/to/tool"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=ANY, returncode=0)

        profile_with_override = {
            "max_line_length": 80, "preferred_quotes": "single", "docstring_style": "google",
            "directory_overrides": {
                str(self.test_dir): {"max_line_length": 10} # Override for test_dir
            }
        }
        code = "this_is_a_very_long_line_for_override" # len > 10, but < 80
        self._write_sample_code(code)

        score = score_style(self.sample_file_path, profile_with_override, self.db_path,
                              w_black_diff=0,w_ruff_diff=0,w_naming=0,w_quotes=0,w_docstrings=0,w_linelen=1.0)
        self.assertAlmostEqual(score, 1.0, places=2, msg=f"Score was {score}, expected override to apply.")


    def test_score_file_not_found(self):
        score = score_style(self.test_dir / "nonexistent.py", self.default_profile, self.db_path)
        self.assertEqual(score, 1.0)

    def test_score_unparseable_code(self):
        self._write_sample_code("def func( : invalid syntax")
        with patch('src.style_scorer.shutil.which', return_value="/tool"), \
             patch('src.style_scorer.subprocess.run', return_value=subprocess.CompletedProcess(args=ANY, returncode=0)):
            score = score_style(self.sample_file_path, self.default_profile, self.db_path)

        # AST related weights: w_naming (0.2), w_linelen (0.1), w_quotes (0.1), w_docstrings (0.1)
        # Expected penalty from AST error = 0.2 + 0.1 + 0.1 + 0.1 = 0.5 (if all these checks were active based on profile)
        # Max possible penalty with all checks active = 0.25(black) + 0.25(ruff) + 0.2(name) + 0.1(line) + 0.1(quote) + 0.1(doc) = 1.0
        # Expected score = 0.5 / 1.0 = 0.5
        # The score_style's max_penalty calculation needs to be robust for this.
        # With current logic: Black=0, Ruff=0. AST fails.
        # total_penalty = w_naming + w_linelen + w_quotes + w_docstrings = 0.2+0.1+0.1+0.1 = 0.5
        # max_possible = w_black + w_ruff + w_naming + w_linelen + w_quotes + w_docstrings = 1.0 (assuming all conditions met)
        # Score = 0.5 / 1.0 = 0.5
        self.assertAlmostEqual(score, 0.50, places=2, msg=f"Score was {score}")


if __name__ == '__main__':
    unittest.main()
