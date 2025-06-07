import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import shutil
import random

# Adjust import path as necessary
from src.style_scorer import score_style

class TestStyleScorer(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_scorer_"))
        self.dummy_sample_file = self.test_dir / "sample.py"
        with open(self.dummy_sample_file, "w") as f:
            f.write("def foo():\n    pass\n") # Minimal valid Python

        self.dummy_project_profile = {"indent_width": 4} # Profile not used by mock yet

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_score_style_returns_float_in_range(self):
        """Test that score_style (mock) returns a float in the expected range."""
        # Seed random for predictable "random" value if needed for a specific check,
        # or just check type and general range for the current mock.
        # The current mock returns random.uniform(0.0, 0.3).

        with patch('src.style_scorer.random.uniform', return_value=0.15) as mock_uniform:
            score = score_style(self.dummy_sample_file, project_profile=self.dummy_project_profile)
            self.assertIsInstance(score, float)
            self.assertTrue(0.0 <= score <= 1.0, "Score should be between 0.0 and 1.0")
            # More specific to current mock's random range if we didn't mock random.uniform itself:
            # self.assertTrue(0.0 <= score <= 0.3, "Mock score should be between 0.0 and 0.3")
            mock_uniform.assert_called_once_with(0.0, 0.3)


    def test_score_style_file_not_found(self):
        """Test score_style when the sample file does not exist."""
        non_existent_file = self.test_dir / "no_such_file.py"
        score = score_style(non_existent_file, project_profile=self.dummy_project_profile)
        self.assertEqual(score, 1.0, "Score should be 1.0 (max deviation) for non-existent file.")

    @patch('src.style_scorer.random.uniform') # Mock random to control its output
    def test_score_style_mock_behavior_with_project_profile(self, mock_random_uniform):
        """Test the current mock passes through the project_profile (though unused by mock)."""
        mock_random_uniform.return_value = 0.25 # A fixed "random" value

        score = score_style(self.dummy_sample_file, project_profile=self.dummy_project_profile)
        self.assertEqual(score, 0.25)
        mock_random_uniform.assert_called_once_with(0.0, 0.3)
        # This test mainly confirms the function runs with the profile argument.

    def test_score_style_multiple_calls_check_print_message(self):
        """Check the placeholder print message is present."""
        with patch('builtins.print') as mock_print:
            score_style(self.dummy_sample_file, project_profile=self.dummy_project_profile)

            found_placeholder_print = False
            for print_call in mock_print.call_args_list:
                if "Placeholder: score_style for" in print_call.args[0] and "would perform detailed analysis" in print_call.args[0]:
                    found_placeholder_print = True
                    break
            self.assertTrue(found_placeholder_print, "Expected placeholder print message not found.")


if __name__ == '__main__':
    unittest.main()
