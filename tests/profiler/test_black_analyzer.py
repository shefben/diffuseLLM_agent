import unittest
import os
from src.profiler.black_analyzer import analyze_files_with_black

class TestBlackAnalyzer(unittest.TestCase):

    def setUp(self):
        self.test_files_dir = "test_black_analyzer_files"
        os.makedirs(self.test_files_dir, exist_ok=True)
        self.well_formatted_file = os.path.join(self.test_files_dir, "well_formatted.py")
        self.needs_formatting_file = os.path.join(self.test_files_dir, "needs_formatting.py")
        self.non_existent_file = os.path.join(self.test_files_dir, "non_existent.py")

        with open(self.well_formatted_file, "w") as f:
            f.write('print("hello world")\n') # Already black-formatted

        with open(self.needs_formatting_file, "w") as f:
            f.write("print('hello world')\n") # Needs reformatting (quotes)

    def tearDown(self):
        if os.path.exists(self.well_formatted_file):
            os.remove(self.well_formatted_file)
        if os.path.exists(self.needs_formatting_file):
            os.remove(self.needs_formatting_file)
        if os.path.exists(self.test_files_dir):
            os.rmdir(self.test_files_dir)

    def test_analyze_well_formatted_file(self):
        # This test assumes 'black' is installed and accessible.
        # It's more of an integration test.
        diff_output, error_output = analyze_files_with_black([self.well_formatted_file])
        self.assertEqual(diff_output.strip(), "") # No diff expected
        # Error output might contain version info or other non-critical messages from black
        # For a true unit test, black itself would be mocked.
        # For now, we primarily check that it runs and produces no diff.

    def test_analyze_needs_formatting_file(self):
        diff_output, error_output = analyze_files_with_black([self.needs_formatting_file])
        self.assertIn("--- Diff for", diff_output) # Expect some diff
        self.assertIn("print('hello world')", diff_output)
        self.assertIn('print("hello world")', diff_output)


    def test_analyze_non_existent_file(self):
        diff_output, error_output = analyze_files_with_black([self.non_existent_file])
        self.assertEqual(diff_output.strip(), "") # No diff for non-existent file
        self.assertIn(f"File not found: {self.non_existent_file}", error_output)

    def test_analyze_empty_list(self):
        diff_output, error_output = analyze_files_with_black([])
        self.assertEqual(diff_output.strip(), "")
        self.assertEqual(error_output.strip(), "")


if __name__ == '__main__':
    unittest.main()
