import unittest
from unittest.mock import patch, MagicMock, mock_open, call, ANY
from pathlib import Path
import tempfile
import shutil
import libcst as cst
# tree_sitter types might not be directly available for isinstance if library is only partially mocked
# from tree_sitter import Tree as TreeSitterTree

# Adjust import path
from src.digester.repository_digester import RepositoryDigester, ParsedFileResult

class TestRepositoryDigester(unittest.TestCase):

    def setUp(self):
        self.test_repo_root = Path(tempfile.mkdtemp(prefix="test_repo_digest_"))
        # We need to be able to control tree-sitter parser initialization for some tests
        # Patching at the module level where RepositoryDigester imports it.
        self.patcher_get_parser = patch('src.digester.repository_digester.get_parser')
        self.MockGetParser = self.patcher_get_parser.start()

        # Simulate successful parser acquisition by default
        self.mock_ts_parser_instance = MagicMock()
        self.MockGetParser.return_value = self.mock_ts_parser_instance

    def tearDown(self):
        shutil.rmtree(self.test_repo_root)
        self.patcher_get_parser.stop()


    def _create_dummy_file(self, path_from_repo_root: str, content: str = ""):
        full_path = self.test_repo_root / path_from_repo_root
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return full_path

    # --- Tests for discover_python_files ---
    def test_discover_files_default_ignores(self):
        self._create_dummy_file("file1.py")
        self._create_dummy_file("subdir/file2.py")
        self._create_dummy_file("tests/test_app.py") # Default ignored dir
        self._create_dummy_file("setup.py") # Default ignored file
        self._create_dummy_file(".venv/lib/lib.py") # Default ignored dir

        digester = RepositoryDigester(self.test_repo_root)
        digester.discover_python_files()

        discovered_paths_rel = sorted([p.relative_to(self.test_repo_root) for p in digester._all_py_files])
        expected_paths_rel = sorted([Path("file1.py"), Path("subdir/file2.py")])
        self.assertEqual(discovered_paths_rel, expected_paths_rel)

    def test_discover_files_custom_ignores(self):
        self._create_dummy_file("core_logic.py")
        self._create_dummy_file("scripts/my_script.py")
        self._create_dummy_file("temp_generated/generated.py")
        self._create_dummy_file("main_app.py")

        digester = RepositoryDigester(self.test_repo_root)
        digester.discover_python_files(ignored_dirs=["scripts"], ignored_files=["main_app.py"])

        discovered_paths_rel = sorted([p.relative_to(self.test_repo_root) for p in digester._all_py_files])
        expected_paths_rel = sorted([Path("core_logic.py"), Path("temp_generated/generated.py")])
        self.assertEqual(discovered_paths_rel, expected_paths_rel)

    # --- Tests for parse_file ---
    def test_parse_file_success(self):
        code = "def foo():\n    return 42\n"
        file_path = self._create_dummy_file("good_code.py", code)

        # Mock Tree-sitter's parse method and root_node.has_error
        mock_ts_tree = MagicMock()
        mock_ts_tree.root_node.has_error = False
        self.mock_ts_parser_instance.parse.return_value = mock_ts_tree

        digester = RepositoryDigester(self.test_repo_root) # Re-init to use mocked get_parser
        result = digester.parse_file(file_path)

        self.assertEqual(result.file_path, file_path)
        self.assertEqual(result.source_code, code)
        self.assertIsNotNone(result.libcst_module)
        self.assertIsInstance(result.libcst_module, cst.Module)
        self.assertIsNone(result.libcst_error)

        self.mock_ts_parser_instance.parse.assert_called_once_with(bytes(code, "utf8"))
        self.assertEqual(result.treesitter_tree, mock_ts_tree)
        self.assertFalse(result.treesitter_has_errors)
        self.assertIsNone(result.treesitter_error_message)

    def test_parse_file_syntax_errors(self):
        code = "def foo():\n  bar baz oops\n" # Syntax error
        file_path = self._create_dummy_file("bad_code.py", code)

        mock_ts_tree_with_error = MagicMock()
        mock_ts_tree_with_error.root_node.has_error = True
        self.mock_ts_parser_instance.parse.return_value = mock_ts_tree_with_error

        digester = RepositoryDigester(self.test_repo_root)
        result = digester.parse_file(file_path)

        self.assertIsNone(result.libcst_module) # LibCST fails hard on syntax errors
        self.assertIsNotNone(result.libcst_error)
        self.assertIn("LibCST ParserSyntaxError", result.libcst_error)

        self.assertEqual(result.treesitter_tree, mock_ts_tree_with_error) # Tree-sitter returns a tree
        self.assertTrue(result.treesitter_has_errors)
        self.assertIsNotNone(result.treesitter_error_message)
        self.assertIn("Tree-sitter found syntax errors", result.treesitter_error_message)

    def test_parse_file_empty_file(self):
        file_path = self._create_dummy_file("empty.py", "")

        mock_ts_tree_empty = MagicMock()
        mock_ts_tree_empty.root_node.has_error = False # Empty file is valid
        self.mock_ts_parser_instance.parse.return_value = mock_ts_tree_empty

        digester = RepositoryDigester(self.test_repo_root)
        result = digester.parse_file(file_path)

        self.assertIsNotNone(result.libcst_module) # LibCST parses "" to an empty Module
        self.assertIsNone(result.libcst_error)
        self.assertEqual(result.treesitter_tree, mock_ts_tree_empty)
        self.assertFalse(result.treesitter_has_errors)

    @patch('builtins.open', side_effect=IOError("Failed to read"))
    def test_parse_file_read_error(self, mock_open_fail):
        # File path doesn't really matter as open is mocked to fail
        file_path = self.test_repo_root / "cant_read.py"

        digester = RepositoryDigester(self.test_repo_root)
        result = digester.parse_file(file_path)

        self.assertIsNone(result.libcst_module)
        self.assertIn("File read error: Failed to read", result.libcst_error)
        self.assertIsNone(result.treesitter_tree)
        self.assertTrue(result.treesitter_has_errors)
        self.assertIn("File read error: Failed to read", result.treesitter_error_message)

    def test_parse_file_treesitter_parser_init_fails(self):
        self.MockGetParser.return_value = None # Simulate get_parser failing in __init__
        # Or self.MockGetParser.side_effect = Exception("TS Lang load fail")
        # For this to work, RepositoryDigester must be instantiated *after* this patch state is set

        code = "def test(): pass"
        file_path = self._create_dummy_file("ts_fail_init.py", code)

        # This digester instance will have self.ts_parser = None
        digester_no_ts = RepositoryDigester(self.test_repo_root)

        result = digester_no_ts.parse_file(file_path)

        self.assertIsNotNone(result.libcst_module) # LibCST should still work
        self.assertIsNone(result.libcst_error)

        self.assertIsNone(result.treesitter_tree)
        self.assertTrue(result.treesitter_has_errors)
        self.assertIn("Tree-sitter parser not initialized", result.treesitter_error_message)

    # --- Tests for digest_repository orchestration ---
    @patch('src.digester.repository_digester.RepositoryDigester.parse_file')
    def test_digest_repository_orchestration(self, mock_parse_file):
        # Create some dummy files for discovery
        file1 = self._create_dummy_file("fileA.py", "a=1")
        file2 = self._create_dummy_file("subdir/fileB.py", "b=2")

        # Mock return value for parse_file
        mock_result_A = ParsedFileResult(file_path=file1, source_code="a=1", libcst_module=MagicMock(), treesitter_tree=MagicMock())
        mock_result_B = ParsedFileResult(file_path=file2, source_code="b=2", libcst_error="LibCST Error on B")
        mock_parse_file.side_effect = [mock_result_A, mock_result_B]

        digester = RepositoryDigester(self.test_repo_root)
        with patch('builtins.print') as mock_print: # Suppress prints or check them
            digester.digest_repository()

        self.assertEqual(mock_parse_file.call_count, 2)
        # Order of calls might not be guaranteed by rglob, so check with any_order or specific list
        mock_parse_file.assert_any_call(file1)
        mock_parse_file.assert_any_call(file2)

        self.assertEqual(len(digester.digested_files), 2)
        self.assertEqual(digester.digested_files[file1], mock_result_A)
        self.assertEqual(digester.digested_files[file2], mock_result_B)

        # Check summary print (example)
        # This part depends on the exact print statements in digest_repository
        # For instance, check if it printed about LibCST Error on B
        found_summary_print = False
        for p_call in mock_print.call_args_list:
            if "LibCST Error on B" in str(p_call.args): # Check if the error message was part of any print call
                # This check might be too broad if the error message is printed for other reasons.
                # A more specific check would target the exact summary print line.
                # For now, this confirms the error was logged.
                found_summary_print = True
                break
        self.assertTrue(found_summary_print, "Expected summary print about LibCST error not found.")


if __name__ == '__main__':
    unittest.main()
