import unittest
import os
import subprocess # For checking if ruff is available
from src.profiler.ruff_analyzer import analyze_files_with_ruff

class TestRuffAnalyzer(unittest.TestCase):

    def setUp(self):
        self.test_files_dir = "test_ruff_analyzer_files"
        os.makedirs(self.test_files_dir, exist_ok=True)
        self.clean_file = os.path.join(self.test_files_dir, "clean.py")
        self.issues_file = os.path.join(self.test_files_dir, "issues.py")
        self.non_existent_file = os.path.join(self.test_files_dir, "non_existent_ruff.py")

        with open(self.clean_file, "w") as f:
            f.write("import os\nprint(os.getcwd())\n") # Should be clean by default

        with open(self.issues_file, "w") as f:
            f.write("import os,sys\nprint(   os.getcwd()   )\n") # Ruff should find issues

        # Check if ruff is installed
        try:
            subprocess.run(["ruff", "--version"], capture_output=True, check=True)
            self.ruff_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.ruff_available = False


    def tearDown(self):
        if os.path.exists(self.clean_file):
            os.remove(self.clean_file)
        if os.path.exists(self.issues_file):
            os.remove(self.issues_file)
        if os.path.exists(self.test_files_dir):
            os.rmdir(self.test_files_dir)

    @unittest.skipUnless(TestRuffAnalyzer.ruff_available, "Ruff CLI not found or not working")
    def test_analyze_clean_file(self):
        diff_output, error_output = analyze_files_with_ruff([self.clean_file])
        # Ruff with --diff might still output the filename if no issues, or nothing.
        # Let's check that no actual diff markers like '+++' or '---' are present
        # if it's truly clean for default rules.
        # A more robust check would be specific to ruff's output for "no changes".
        # Often, it's just empty stdout for --diff if no changes.
        is_diff_present = "--- " in diff_output and "+++ " in diff_output
        self.assertFalse(is_diff_present, f"Expected no diff, got: {diff_output}")
        # stderr might contain info, so not strictly checking it for emptiness unless errors are expected.

    @unittest.skipUnless(TestRuffAnalyzer.ruff_available, "Ruff CLI not found or not working")
    def test_analyze_issues_file(self):
        diff_output, error_output = analyze_files_with_ruff([self.issues_file])
        self.assertIn("--- Diff for", diff_output) # Expect some diff
        self.assertIn("import os,sys", diff_output) # Part of the issue

    def test_analyze_non_existent_file(self):
        # This test doesn't strictly need ruff to be installed, as it tests the script's error handling.
        diff_output, error_output = analyze_files_with_ruff([self.non_existent_file])
        # The current ruff_analyzer.py using subprocess will have Ruff itself report file not found to its stderr.
        # The script then wraps this.
        self.assertIn(f"Ruff stderr for {self.non_existent_file}", error_output)
        self.assertIn("No such file or directory", error_output) # Ruff's typical message

    def test_analyze_empty_list(self):
        diff_output, error_output = analyze_files_with_ruff([])
        self.assertEqual(diff_output.strip(), "")
        self.assertEqual(error_output.strip(), "")

if __name__ == '__main__':
    # Need to manually set ruff_available for direct execution if not using unittest discovery
    # This is a bit of a hack for direct script running; test runners handle this better.
    try:
        subprocess.run(["ruff", "--version"], capture_output=True, check=True, text=True)
        TestRuffAnalyzer.ruff_available = True
    except:
        TestRuffAnalyzer.ruff_available = False
        print("Warning: Ruff CLI not found or not working, some tests will be skipped.")
    unittest.main()
