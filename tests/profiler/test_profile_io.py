import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
from pathlib import Path
import tempfile
import shutil

from src.profiler.profile_io import save_style_profile, load_style_profile, DEFAULT_FINGERPRINT_PATH

class TestProfileIO(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for file operations if needed for integration-like tests
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_profile_io_"))
        self.dummy_profile_data = {
            "indent_width": 4, "preferred_quotes": "single", "max_line_length": 88,
            "docstring_style": "google", "confidence_score": 0.9
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('src.profiler.profile_io.Path.mkdir')
    @patch('src.profiler.profile_io.open', new_callable=mock_open)
    @patch('src.profiler.profile_io.json.dump')
    def test_save_style_profile_success(self, mock_json_dump, mock_file_open, mock_mkdir):
        """Test successful saving of a style profile."""
        test_path = Path("config/test_fingerprint.json")
        save_style_profile(self.dummy_profile_data, output_path=test_path)

        # Check that the parent directory of test_path was created
        # The actual path passed to mkdir would be test_path.parent
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        mock_file_open.assert_called_once_with(test_path, "w", encoding="utf-8")
        mock_json_dump.assert_called_once_with(self.dummy_profile_data, mock_file_open(), indent=4)

    @patch('src.profiler.profile_io.Path.mkdir') # Mock mkdir to avoid actual creation
    @patch('src.profiler.profile_io.open', new_callable=mock_open)
    @patch('src.profiler.profile_io.json.dump', side_effect=TypeError("Not serializable"))
    def test_save_style_profile_type_error(self, mock_json_dump, mock_file_open, mock_mkdir):
        """Test save_style_profile handling of TypeError during json.dump."""
        test_path = Path("config/type_error_test.json")
        with self.assertRaises(TypeError):
            save_style_profile({"data": set()}, output_path=test_path) # set is not JSON serializable by default
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


    @patch('src.profiler.profile_io.Path.exists', return_value=True)
    @patch('src.profiler.profile_io.open', new_callable=mock_open, read_data=json.dumps({"key": "value"}))
    def test_load_style_profile_success(self, mock_file_open_read, mock_path_exists):
        """Test successful loading of a style profile."""
        test_path = Path("dummy_load.json")
        profile = load_style_profile(input_path=test_path)
        self.assertEqual(profile, {"key": "value"})
        mock_path_exists.assert_called_once_with()
        mock_file_open_read.assert_called_once_with(test_path, "r", encoding="utf-8")


    @patch('src.profiler.profile_io.Path.exists', return_value=False)
    def test_load_style_profile_file_not_found(self, mock_exists):
        """Test load_style_profile when the file does not exist."""
        test_path = Path("non_existent.json")
        profile = load_style_profile(input_path=test_path)
        self.assertIsNone(profile)
        mock_exists.assert_called_once_with()

    @patch('src.profiler.profile_io.Path.exists', return_value=True)
    @patch('src.profiler.profile_io.open', new_callable=mock_open, read_data="invalid json data")
    def test_load_style_profile_json_decode_error(self, mock_file_open_invalid, mock_path_exists):
        """Test load_style_profile with invalid JSON content."""
        test_path = Path("invalid_json.json")
        profile = load_style_profile(input_path=test_path)
        self.assertIsNone(profile)
        mock_path_exists.assert_called_once_with()
        mock_file_open_invalid.assert_called_once_with(test_path, "r", encoding="utf-8")


    @patch('src.profiler.profile_io.Path.exists', return_value=True)
    @patch('src.profiler.profile_io.open', side_effect=IOError("File read error"))
    def test_load_style_profile_io_error(self, mock_open_io_error, mock_path_exists):
        """Test load_style_profile handling of IOError during file read."""
        test_path = Path("io_error_test.json")
        profile = load_style_profile(input_path=test_path)
        self.assertIsNone(profile)
        mock_path_exists.assert_called_once_with()
        mock_open_io_error.assert_called_once_with(test_path, "r", encoding="utf-8")


    def test_save_and_load_integration(self):
        """Test saving and then loading a profile using the actual file system (integration)."""
        test_file_path = self.test_dir / "integrated_profile.json"

        save_style_profile(self.dummy_profile_data, output_path=test_file_path)
        self.assertTrue(test_file_path.exists())

        loaded_data = load_style_profile(input_path=test_file_path)
        self.assertEqual(self.dummy_profile_data, loaded_data)

        # Test loading a non-existent file via integration path
        non_existent_path = self.test_dir / "non_existent_integration.json"
        self.assertIsNone(load_style_profile(input_path=non_existent_path))


if __name__ == '__main__':
    unittest.main()
