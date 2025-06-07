import unittest
from unittest.mock import patch, MagicMock, call, ANY
from pathlib import Path
import tempfile
import shutil
import subprocess # For CompletedProcess

# Adjust import path as necessary
from src.formatter import format_code

class TestFormatter(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_formatter_"))
        self.dummy_file_path = self.test_dir / "test_file.py"
        with open(self.dummy_file_path, "w") as f:
            f.write("print('hello')")

        self.dummy_profile = {"max_line_length": 88} # Profile not directly used by format_code yet
        self.dummy_pyproject_path = self.test_dir / "pyproject.toml" # Also not directly used yet by format_code's core logic

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    def test_format_code_success_all_tools_ok(self, mock_subprocess_run, mock_shutil_which):
        """Test format_code when Black and Ruff run successfully."""
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}" # Simulate tools found

        # Simulate successful runs for Black, Ruff Format, Ruff Lint
        mock_subprocess_run.side_effect = [
            subprocess.CompletedProcess(args=['black', str(self.dummy_file_path)], returncode=0, stdout="", stderr=""), # Black OK
            subprocess.CompletedProcess(args=['ruff', 'format', str(self.dummy_file_path)], returncode=0, stdout="", stderr=""), # Ruff Format OK
            subprocess.CompletedProcess(args=['ruff', 'check', '--fix', '--exit-zero-even-if-changed', str(self.dummy_file_path)], returncode=0, stdout="", stderr="")  # Ruff Lint OK
        ]

        with patch('builtins.print') as mock_print: # To check for placeholder print
            success = format_code(self.dummy_file_path, self.dummy_profile, self.dummy_pyproject_path)

        self.assertTrue(success)
        self.assertEqual(mock_subprocess_run.call_count, 3)
        mock_shutil_which.assert_has_calls([call("black"), call("ruff")], any_order=True)

        # Check if the LibCST placeholder print was called
        found_libcst_placeholder_print = False
        for print_call in mock_print.call_args_list:
            if "Placeholder: Identifier renaming pass (LibCST)" in print_call.args[0]:
                found_libcst_placeholder_print = True
                break
        self.assertTrue(found_libcst_placeholder_print)


    def test_format_code_file_not_found(self):
        non_existent_file = self.test_dir / "no_such_file.py"
        success = format_code(non_existent_file, self.dummy_profile, self.dummy_pyproject_path)
        self.assertFalse(success)

    @patch('src.formatter.shutil.which', return_value=None) # Simulate tool not found
    def test_format_code_black_not_found(self, mock_shutil_which):
        success = format_code(self.dummy_file_path, self.dummy_profile, self.dummy_pyproject_path)
        self.assertFalse(success)
        mock_shutil_which.assert_called_once_with("black")

    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    def test_format_code_black_fails_syntax_error(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=['black', str(self.dummy_file_path)], returncode=123, stdout="", stderr="Syntax error details"
        ) # Black syntax error

        success = format_code(self.dummy_file_path, self.dummy_profile, self.dummy_pyproject_path)
        self.assertFalse(success)
        mock_subprocess_run.assert_called_once() # Black fails, Ruff not called

    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    def test_format_code_black_fails_other_error(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=['black', str(self.dummy_file_path)], returncode=1, stdout="", stderr="Some other Black error"
        ) # Black internal error

        success = format_code(self.dummy_file_path, self.dummy_profile, self.dummy_pyproject_path)
        self.assertFalse(success)
        mock_subprocess_run.assert_called_once()

    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    def test_format_code_ruff_format_fails(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}"
        # Simulate Black OK, Ruff Format fails, Ruff Lint also indicates failure or doesn't run effectively
        mock_subprocess_run.side_effect = [
            subprocess.CompletedProcess(args=['black', str(self.dummy_file_path)], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=['ruff', 'format', str(self.dummy_file_path)], returncode=1, stdout="", stderr="Ruff format error"),
            subprocess.CompletedProcess(args=['ruff', 'check', '--fix', '--exit-zero-even-if-changed', str(self.dummy_file_path)], returncode=2, stdout="", stderr="Ruff lint error after format error") # Ruff Lint might also fail or report issues
        ]

        success = format_code(self.dummy_file_path, self.dummy_profile, self.dummy_pyproject_path)
        self.assertFalse(success)
        # Expect Black, Ruff Format, and Ruff Lint to be called.
        # The current formatter.py tries all steps.
        self.assertEqual(mock_subprocess_run.call_count, 3)


    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    def test_format_code_ruff_lint_unfixable_issues(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}"
        mock_subprocess_run.side_effect = [
            subprocess.CompletedProcess(args=['black', str(self.dummy_file_path)], returncode=0, stdout="", stderr=""), # Black OK
            subprocess.CompletedProcess(args=['ruff', 'format', str(self.dummy_file_path)], returncode=0, stdout="", stderr=""), # Ruff Format OK
            subprocess.CompletedProcess(args=['ruff', 'check', '--fix', '--exit-zero-even-if-changed', str(self.dummy_file_path)], returncode=1, stdout="Unfixable lint issues...", stderr="")  # Ruff Lint finds unfixable issues
        ]

        success = format_code(self.dummy_file_path, self.dummy_profile, self.dummy_pyproject_path)
        self.assertFalse(success) # Should fail if unfixable issues remain
        self.assertEqual(mock_subprocess_run.call_count, 3)

    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    def test_format_code_ruff_lint_internal_error(self, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}"
        mock_subprocess_run.side_effect = [
            subprocess.CompletedProcess(args=['black', str(self.dummy_file_path)], returncode=0, stdout="", stderr=""), # Black OK
            subprocess.CompletedProcess(args=['ruff', 'format', str(self.dummy_file_path)], returncode=0, stdout="", stderr=""), # Ruff Format OK
            subprocess.CompletedProcess(args=['ruff', 'check', '--fix', '--exit-zero-even-if-changed', str(self.dummy_file_path)], returncode=2, stdout="", stderr="Ruff internal error")  # Ruff Lint internal error
        ]

        success = format_code(self.dummy_file_path, self.dummy_profile, self.dummy_pyproject_path)
        self.assertFalse(success)
        self.assertEqual(mock_subprocess_run.call_count, 3)

if __name__ == '__main__':
    unittest.main()
