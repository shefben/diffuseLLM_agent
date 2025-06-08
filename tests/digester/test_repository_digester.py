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
# Import pyanalyze components for type checking mocks and patching globals
import src.digester.repository_digester # To patch its global 'pyanalyze'
# Add sys for sys.path testing
import sys
import os # For os.pathsep
import ast # For test_infer_types_sys_path_management_and_placeholder_return


class TestRepositoryDigester(unittest.TestCase):

    def setUp(self):
        self.test_repo_root = Path(tempfile.mkdtemp(prefix="test_repo_digest_")).resolve()

        # Default mock for tree-sitter's get_parser
        self.patcher_get_parser = patch('src.digester.repository_digester.get_parser')
        self.MockGetParser = self.patcher_get_parser.start()
        self.mock_ts_parser_instance = MagicMock()
        self.MockGetParser.return_value = self.mock_ts_parser_instance

        # Default mock for pyanalyze's Checker and related components
        # These are patched at the module level where RepositoryDigester imports them.
        self.patcher_pyanalyze_checker = patch('src.digester.repository_digester.Checker')
        self.MockPyanalyzeChecker = self.patcher_pyanalyze_checker.start()
        self.mock_pyanalyze_checker_instance = MagicMock()
        self.MockPyanalyzeChecker.return_value = self.mock_pyanalyze_checker_instance

        self.patcher_pyanalyze_config = patch('src.digester.repository_digester.Config')
        self.MockPyanalyzeConfig = self.patcher_pyanalyze_config.start()
        self.patcher_pyanalyze_options = patch('src.digester.repository_digester.Options')
        self.MockPyanalyzeOptions = self.patcher_pyanalyze_options.start()

        # Ensure pyanalyze itself is seen as imported for most tests
        # Also mock its sub-attributes that are checked in _infer_types_with_pyanalyze
        self.mock_pyanalyze_module = MagicMock()
        # Ensure ast_annotator and annotate_code exist on the mock to avoid AttributeError
        # if pyanalyze module itself is mocked but not its attributes.
        if hasattr(self.mock_pyanalyze_module, 'ast_annotator'):
            self.mock_pyanalyze_module.ast_annotator.annotate_code = MagicMock()
        else: # If ast_annotator itself needs to be a MagicMock
            self.mock_pyanalyze_module.ast_annotator = MagicMock(annotate_code=MagicMock())

        self.patcher_pyanalyze_module = patch('src.digester.repository_digester.pyanalyze', self.mock_pyanalyze_module)
        self.MockPyanalyzeModule_obj = self.patcher_pyanalyze_module.start() # Renamed to avoid conflict

        self.patcher_pyanalyze_dump_value = patch('src.digester.repository_digester.pyanalyze_dump_value', MagicMock())
        self.MockPyanalyzeDumpValue = self.patcher_pyanalyze_dump_value.start()


    def tearDown(self):
        shutil.rmtree(self.test_repo_root)
        self.patcher_get_parser.stop()
        self.patcher_pyanalyze_checker.stop()
        self.patcher_pyanalyze_config.stop()
        self.patcher_pyanalyze_options.stop()
        self.patcher_pyanalyze_module.stop()
        self.patcher_pyanalyze_dump_value.stop()


    def _create_dummy_file(self, path_from_repo_root: str, content: str = ""):
        # Ensure path_from_repo_root is relative for consistent joining
        if Path(path_from_repo_root).is_absolute():
            raise ValueError("Use relative path for _create_dummy_file")
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

        mock_ts_tree = MagicMock()
        mock_ts_tree.root_node.has_error = False
        self.mock_ts_parser_instance.parse.return_value = mock_ts_tree

        # Mock Pyanalyze's annotate_code to return a simple AST node for the visitor
        # and the visitor to return some dummy type info
        self.mock_pyanalyze_module.ast_annotator.annotate_code.return_value = ast.parse(code)

        digester = RepositoryDigester(self.test_repo_root)
        with patch.object(digester, '_infer_types_with_pyanalyze', return_value={"foo.return": "int"}) as mock_infer_types_method:
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

            mock_infer_types_method.assert_called_once() # Check it was called
            self.assertEqual(result.type_info, {"foo.return": "int"})


    def test_parse_file_syntax_errors(self):
        code = "def foo():\n  bar baz oops\n"
        file_path = self._create_dummy_file("bad_code.py", code)

        mock_ts_tree_with_error = MagicMock()
        mock_ts_tree_with_error.root_node.has_error = True
        self.mock_ts_parser_instance.parse.return_value = mock_ts_tree_with_error

        digester = RepositoryDigester(self.test_repo_root)
        result = digester.parse_file(file_path)

        self.assertIsNone(result.libcst_module)
        self.assertIsNotNone(result.libcst_error)
        self.assertIn("LibCST ParserSyntaxError", result.libcst_error)

        self.assertEqual(result.treesitter_tree, mock_ts_tree_with_error)
        self.assertTrue(result.treesitter_has_errors)
        self.assertIsNotNone(result.treesitter_error_message)
        self.assertIn("Tree-sitter found syntax errors", result.treesitter_error_message)
        self.assertEqual(result.type_info, {"info": "Type inference skipped due to LibCST parsing errors."})


    def test_parse_file_empty_file(self):
        file_path = self._create_dummy_file("empty.py", "")

        mock_ts_tree_empty = MagicMock()
        mock_ts_tree_empty.root_node.has_error = False
        self.mock_ts_parser_instance.parse.return_value = mock_ts_tree_empty

        digester = RepositoryDigester(self.test_repo_root)
        result = digester.parse_file(file_path)

        self.assertIsNotNone(result.libcst_module)
        self.assertIsNone(result.libcst_error)
        self.assertEqual(result.treesitter_tree, mock_ts_tree_empty)
        self.assertFalse(result.treesitter_has_errors)
        self.assertEqual(result.type_info, {"info": "Type inference skipped for empty file."})


    @patch('builtins.open', side_effect=IOError("Failed to read"))
    def test_parse_file_read_error(self, mock_open_fail):
        file_path = self.test_repo_root / "cant_read.py"

        digester = RepositoryDigester(self.test_repo_root)
        result = digester.parse_file(file_path)

        self.assertIsNone(result.libcst_module)
        self.assertIn("File read error: Failed to read", result.libcst_error)
        self.assertIsNone(result.treesitter_tree)
        self.assertTrue(result.treesitter_has_errors)
        self.assertIn("File read error: Failed to read", result.treesitter_error_message)
        self.assertEqual(result.type_info, {"error": "File read error: Failed to read"})


    def test_parse_file_treesitter_parser_init_fails(self):
        self.MockGetParser.return_value = None

        code = "def test(): pass"
        file_path = self._create_dummy_file("ts_fail_init.py", code)

        digester_no_ts = RepositoryDigester(self.test_repo_root)

        with patch.object(digester_no_ts, '_infer_types_with_pyanalyze', return_value={"info":"Pyanalyze placeholder"}) as mock_infer:
            result = digester_no_ts.parse_file(file_path)

            self.assertIsNotNone(result.libcst_module)
            self.assertIsNone(result.libcst_error)

            self.assertIsNone(result.treesitter_tree)
            self.assertTrue(result.treesitter_has_errors)
            self.assertIn("Tree-sitter parser not initialized", result.treesitter_error_message)
            mock_infer.assert_called_once() # Type inference should still run


    # --- Test __init__ for Pyanalyze setup ---
    def test_init_pyanalyze_checker_setup_success(self):
        self.MockPyanalyzeOptions.reset_mock()
        self.MockPyanalyzeConfig.from_options.reset_mock()
        self.MockPyanalyzeChecker.reset_mock()

        digester = RepositoryDigester(self.test_repo_root)
        self.MockPyanalyzeOptions.assert_called_once_with(paths=[str(self.test_repo_root)])
        self.MockPyanalyzeConfig.from_options.assert_called_once()
        self.MockPyanalyzeChecker.assert_called_once()
        self.assertIsNotNone(digester.pyanalyze_checker)

    @patch('src.digester.repository_digester.pyanalyze', None)
    def test_init_pyanalyze_module_not_available(self, mock_pyanalyze_import_none):
        with patch('builtins.print') as mock_print:
            digester = RepositoryDigester(self.test_repo_root)
            self.assertIsNone(digester.pyanalyze_checker)
            self.assertTrue(any("pyanalyze library components not found" in str(c.args) for c in mock_print.call_args_list))

    def test_init_pyanalyze_checker_setup_exception(self):
        self.MockPyanalyzeChecker.side_effect = Exception("Checker init failed")
        with patch('builtins.print') as mock_print:
            digester = RepositoryDigester(self.test_repo_root)
            self.assertIsNone(digester.pyanalyze_checker)
            self.assertTrue(any("Failed to initialize Pyanalyze Checker: Checker init failed" in str(c.args) for c in mock_print.call_args_list))
        self.MockPyanalyzeChecker.side_effect = None


    # --- Tests for _infer_types_with_pyanalyze (current placeholder behavior) ---
    def test_infer_types_pyanalyze_checker_is_none(self):
        digester = RepositoryDigester(self.test_repo_root)
        digester.pyanalyze_checker = None
        file_path = self._create_dummy_file("some_code.py", "x = 1")

        with patch('builtins.print') as mock_print:
            result = digester._infer_types_with_pyanalyze(file_path, "x = 1")

        expected_info = {"info": "Pyanalyze components unavailable."}
        self.assertEqual(result, expected_info)
        self.assertTrue(any("Pyanalyze (or ast_annotator/dump_value) not available" in str(c.args) for c in mock_print.call_args_list))


    def test_infer_types_sys_path_management_and_placeholder_return(self):
        digester = RepositoryDigester(self.test_repo_root)
        if digester.pyanalyze_checker is None:
            digester.pyanalyze_checker = self.mock_pyanalyze_checker_instance

        file_rel_path = "subdir/my_module.py"
        file_abs_path = self._create_dummy_file(file_rel_path, "x=1")

        initial_sys_path = list(sys.path)
        self.mock_pyanalyze_module.ast_annotator.annotate_code.return_value = ast.parse("x=1")

        try:
            result = digester._infer_types_with_pyanalyze(file_abs_path, "x=1")
            self.assertEqual(sys.path, initial_sys_path, "sys.path was not restored correctly.")
            self.assertIsNotNone(result)
            self.assertIn("info", result)
            self.assertEqual(result, {"info": "Pyanalyze ran but no types extracted by visitor."})
        finally:
            sys.path = initial_sys_path


    # --- Tests for parse_file integration of type inference ---
    @patch('src.digester.repository_digester.RepositoryDigester._infer_types_with_pyanalyze')
    def test_parse_file_calls_infer_types_on_success(self, mock_infer_types):
        mock_infer_types.return_value = {"var_x": "int"}
        code = "x = 1"
        file_path = self._create_dummy_file("type_integ.py", code)

        mock_ts_tree = MagicMock(); mock_ts_tree.root_node.has_error = False
        self.mock_ts_parser_instance.parse.return_value = mock_ts_tree

        digester = RepositoryDigester(self.test_repo_root)
        result = digester.parse_file(file_path)

        mock_infer_types.assert_called_once_with(file_path, code)
        self.assertEqual(result.type_info, {"var_x": "int"})
        self.assertIsNone(result.libcst_error)

    @patch('src.digester.repository_digester.RepositoryDigester._infer_types_with_pyanalyze')
    def test_parse_file_skips_infer_types_on_libcst_error(self, mock_infer_types):
        code = "def func( :"
        file_path = self._create_dummy_file("type_integ_libcst_err.py", code)
        self.mock_ts_parser_instance.parse.return_value = MagicMock(root_node=MagicMock(has_error=True))

        digester = RepositoryDigester(self.test_repo_root)
        result = digester.parse_file(file_path)

        mock_infer_types.assert_not_called()
        self.assertIsNotNone(result.libcst_error)
        self.assertIsNotNone(result.type_info)
        self.assertEqual(result.type_info, {"info": "Type inference skipped due to LibCST parsing errors."})

    @patch('src.digester.repository_digester.RepositoryDigester._infer_types_with_pyanalyze')
    def test_parse_file_skips_infer_types_on_empty_code(self, mock_infer_types):
        code = ""
        file_path = self._create_dummy_file("type_integ_empty.py", code)
        # Pyanalyze's annotate_code is called by _infer_types_with_pyanalyze,
        # and _infer_types_with_pyanalyze itself handles empty string.
        # parse_file calls _infer_types_with_pyanalyze if no libcst error AND code is not empty.
        # So, _infer_types_with_pyanalyze IS NOT CALLED by parse_file if code is empty.

        self.mock_ts_parser_instance.parse.return_value = MagicMock(root_node=MagicMock(has_error=False))

        digester = RepositoryDigester(self.test_repo_root)
        result = digester.parse_file(file_path)

        mock_infer_types.assert_not_called()
        self.assertIsNotNone(result.type_info)
        self.assertEqual(result.type_info, {"info": "Type inference skipped for empty file."})


    # --- Test digest_repository summary for type_info ---
    @patch('src.digester.repository_digester.RepositoryDigester.parse_file')
    def test_digest_repository_summary_includes_type_info_count_updated(self, mock_parse_file):
        file1 = self._create_dummy_file("fileTA.py", "a=1")
        file2 = self._create_dummy_file("subdir/fileTB.py", "b=2")
        file3 = self._create_dummy_file("fileTC.py", "c=3")

        mock_result_A = ParsedFileResult(file_path=file1, source_code="a=1", libcst_module=MagicMock(), treesitter_tree=MagicMock(), type_info={"a": "int"})
        mock_result_B = ParsedFileResult(file_path=file2, source_code="b=2", libcst_module=MagicMock(), treesitter_tree=MagicMock(), type_info={"error": "failed"})
        mock_result_C_no_info = ParsedFileResult(file_path=file3, source_code="c=3", libcst_module=MagicMock(), treesitter_tree=MagicMock(), type_info=None)

        mock_parse_file.side_effect = [mock_result_A, mock_result_B, mock_result_C_no_info]

        digester = RepositoryDigester(self.test_repo_root)
        digester._all_py_files = [file1, file2, file3]

        with patch('builtins.print') as mock_print:
            digester.digest_repository()

        self.assertEqual(mock_parse_file.call_count, 3)

        summary_output = "".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)

        self.assertIn("Files with some type info extracted: 1", summary_output)


if __name__ == '__main__':
    unittest.main()
