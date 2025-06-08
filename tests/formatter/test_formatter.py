import unittest
from unittest.mock import patch, MagicMock, call, ANY
from pathlib import Path
import tempfile
import shutil
import subprocess # For CompletedProcess

# Adjust import path for format_code
from src.formatter import format_code
# We will mock 'src.formatter.rename_identifiers_in_code_placeholder' or
# 'src.transformer.identifier_renamer.rename_identifiers_in_code' depending on what format_code imports.
# format_code currently uses a placeholder by default.

class TestFormatter(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_formatter_"))
        self.dummy_file_path = self.test_dir / "test_file.py"
        self.original_content = "import os,sys\ndef MyFunc():\n  print('hello Original')" # Content that black/ruff will change
        with open(self.dummy_file_path, "w", encoding="utf-8") as f:
            f.write(self.original_content)

        self.dummy_profile = {"max_line_length": 88}
        self.dummy_pyproject_path = self.test_dir / "pyproject.toml"
        # Create a dummy pyproject.toml so Black/Ruff don't complain loudly or use global configs
        with open(self.dummy_pyproject_path, "w") as f:
            f.write("[tool.black]\nline_length=88\n[tool.ruff]\nline-length=88\n")

        self.dummy_db_path = self.test_dir / "naming.db" # For tests involving db_path
        # Create a dummy db file to simulate it existing for some tests
        with open(self.dummy_db_path, "w") as f:
            f.write("dummy_db_content")


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # Test the main success path, including mocking the renamer
    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    @patch('src.formatter.rename_identifiers_in_code_placeholder') # Mock the renamer function used by format_code
    def test_format_code_success_with_renaming(
        self, mock_renamer, mock_subprocess_run, mock_shutil_which
    ):
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}"

        # Simulate Black and Ruff outputs
        black_formatted_content = "import os\nimport sys\n\ndef MyFunc():\n    print(\"hello Original\")\n"
        ruff_further_formatted_content = "import os\nimport sys\n\ndef MyFunc():\n    print('hello Original')\n" # e.g. Ruff fixes quotes if Black didn't

        mock_subprocess_run.side_effect = [
            subprocess.CompletedProcess(args=['black', ANY], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=['ruff', 'format', ANY], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=['ruff', 'check', ANY], returncode=0, stdout="", stderr="")
        ]

        # Simulate renamer making a change
        renamed_content_after_ruff = "import os\nimport sys\n\ndef my_func():\n    print('hello Original')\n"
        mock_renamer.return_value = renamed_content_after_ruff

        # Mock file read/write to check intermediate and final content
        # We need to control what file_path.read_text() returns after Black/Ruff
        # and verify file_path.write_text() is called with renamer's output.

        # This mock setup is getting complex. A simpler way for this test:
        # 1. Let Black/Ruff run (they modify the dummy_file_path).
        # 2. Read content after Black/Ruff.
        # 3. Mock renamer to take this content and return something new.
        # 4. Verify final file content.
        # This requires actual Black/Ruff to be installed and runnable.
        # For pure unit test with mocks:

        mock_file_content_after_black_ruff = ruff_further_formatted_content

        with patch.object(Path, 'read_text', return_value=mock_file_content_after_black_ruff) as mock_read, \
             patch.object(Path, 'write_text') as mock_write:

            success = format_code(self.dummy_file_path, self.dummy_profile,
                                  pyproject_path=self.dummy_pyproject_path, db_path=self.dummy_db_path)

        self.assertTrue(success)
        self.assertEqual(mock_subprocess_run.call_count, 3) # Black, Ruff format, Ruff lint

        # Check that read_text was called after Black/Ruff and before renamer
        # This specific call to read_text is inside format_code before calling renamer
        mock_read.assert_called_once_with(encoding="utf-8")

        # Check renamer call
        mock_renamer.assert_called_once_with(mock_file_content_after_black_ruff, self.dummy_db_path)

        # Check that write_text was called with the output from the renamer
        mock_write.assert_called_once_with(renamed_content_after_ruff, encoding="utf-8")


    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    @patch('src.formatter.rename_identifiers_in_code_placeholder')
    def test_format_code_renamer_makes_no_changes(
        self, mock_renamer, mock_subprocess_run, mock_shutil_which
    ):
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}"
        ruff_formatted_content = "import os\nimport sys\n\ndef my_func():\n    print('hello Original')\n"

        mock_subprocess_run.side_effect = [
            subprocess.CompletedProcess(args=['black', ANY], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=['ruff', 'format', ANY], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=['ruff', 'check', ANY], returncode=0, stdout="", stderr="")
        ]
        mock_renamer.return_value = ruff_formatted_content # Renamer returns same content

        with patch.object(Path, 'read_text', return_value=ruff_formatted_content) as mock_read, \
             patch.object(Path, 'write_text') as mock_write:
            success = format_code(self.dummy_file_path, self.dummy_profile, db_path=self.dummy_db_path)

        self.assertTrue(success)
        mock_renamer.assert_called_once_with(ruff_formatted_content, self.dummy_db_path)
        # write_text should NOT be called if content is unchanged by renamer
        mock_write.assert_not_called()


    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    @patch('src.formatter.rename_identifiers_in_code_placeholder') # Mock the renamer
    def test_format_code_no_db_path_skips_renamer(
        self, mock_renamer, mock_subprocess_run, mock_shutil_which
    ):
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[ANY], returncode=0) # All tools succeed

        success = format_code(self.dummy_file_path, self.dummy_profile, db_path=None) # No db_path

        self.assertTrue(success)
        mock_renamer.assert_not_called() # Renamer should not be called

    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    @patch('src.formatter.rename_identifiers_in_code_placeholder')
    def test_format_code_db_not_exists_skips_renamer(
        self, mock_renamer, mock_subprocess_run, mock_shutil_which
    ):
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[ANY], returncode=0)

        non_existent_db = self.test_dir / "no_db_here.db"
        # Ensure it doesn't exist for real, though format_code checks Path.exists()
        if non_existent_db.exists(): non_existent_db.unlink()

        success = format_code(self.dummy_file_path, self.dummy_profile, db_path=non_existent_db)

        self.assertTrue(success)
        mock_renamer.assert_not_called()


    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    @patch('src.formatter.rename_identifiers_in_code_placeholder', side_effect=Exception("Renamer boom!"))
    def test_format_code_renamer_exception_is_warning(
        self, mock_renamer, mock_subprocess_run, mock_shutil_which
    ):
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[ANY], returncode=0) # Black/Ruff OK

        with patch('builtins.print') as mock_print:
            # Current format_code logs a warning and returns True if renamer fails but Black/Ruff OK'd
            success = format_code(self.dummy_file_path, self.dummy_profile, db_path=self.dummy_db_path)

        self.assertTrue(success) # As per current logic, renamer error is not fatal
        mock_renamer.assert_called_once()
        self.assertTrue(any("An error occurred during identifier renaming" in str(c.args) for c in mock_print.call_args_list))


    # Keep existing tests for Black/Ruff failures, just add db_path=None to their calls
    # Example of adapting one such test:
    @patch('src.formatter.shutil.which')
    @patch('src.formatter.subprocess.run')
    @patch('src.formatter.rename_identifiers_in_code_placeholder') # Must mock renamer too
    def test_format_code_black_fails_syntax_error_adapted(self, mock_renamer, mock_subprocess_run, mock_shutil_which):
        mock_shutil_which.side_effect = lambda cmd: f"/fake/path/to/{cmd}"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=['black', str(self.dummy_file_path)], returncode=123, stderr="Syntax error"
        )

        success = format_code(self.dummy_file_path, self.dummy_profile, db_path=self.dummy_db_path) # Pass db_path
        self.assertFalse(success)
        mock_subprocess_run.assert_called_once() # Black fails, Ruff & Renamer not called
        mock_renamer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
        self.assertFalse(success)
        self.assertEqual(mock_subprocess_run.call_count, 3)

if __name__ == '__main__':
    unittest.main()
