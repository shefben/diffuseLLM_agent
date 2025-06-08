import unittest
from unittest.mock import patch, MagicMock, mock_open, call, ANY
from pathlib import Path
import tempfile
import shutil
import libcst as cst
import ast # Needed for constructing ASTs for mocking
import sys
import os # For os.pathsep

# Adjust import path
from src.digester.repository_digester import RepositoryDigester, ParsedFileResult
# Import pyanalyze components for type checking mocks and patching globals
import src.digester.repository_digester # To patch its global 'pyanalyze'

# Helper to create a mock 'inferred_value' object for attaching to AST nodes
def make_mock_inferred_value(type_name_str: str):
    mv = MagicMock(name=f"inferred_value_for_{type_name_str}")
    # Store the type_name on the mock so dump_value can identify it
    mv.mock_type_name = type_name_str
    return mv

class TestRepositoryDigester(unittest.TestCase):

    def setUp(self):
        self.test_repo_root = Path(tempfile.mkdtemp(prefix="test_repo_digest_")).resolve()

        # Default mock for tree-sitter's get_parser
        self.patcher_get_parser = patch('src.digester.repository_digester.get_parser')
        self.MockGetParser = self.patcher_get_parser.start()
        self.mock_ts_parser_instance = MagicMock()
        self.MockGetParser.return_value = self.mock_ts_parser_instance

        # Default mock for pyanalyze's Checker and related components
        self.patcher_pyanalyze_checker = patch('src.digester.repository_digester.Checker')
        self.MockPyanalyzeChecker = self.patcher_pyanalyze_checker.start()
        self.mock_pyanalyze_checker_instance = MagicMock()
        self.MockPyanalyzeChecker.return_value = self.mock_pyanalyze_checker_instance

        self.patcher_pyanalyze_config = patch('src.digester.repository_digester.Config')
        self.MockPyanalyzeConfig = self.patcher_pyanalyze_config.start()
        self.patcher_pyanalyze_options = patch('src.digester.repository_digester.Options')
        self.MockPyanalyzeOptions = self.patcher_pyanalyze_options.start()

        self.mock_pyanalyze_module = MagicMock()
        if hasattr(self.mock_pyanalyze_module, 'ast_annotator'):
            self.mock_pyanalyze_module.ast_annotator.annotate_code = MagicMock()
        else:
            self.mock_pyanalyze_module.ast_annotator = MagicMock(annotate_code=MagicMock())
        self.patcher_pyanalyze_module = patch('src.digester.repository_digester.pyanalyze', self.mock_pyanalyze_module)
        self.MockPyanalyzeModule_obj = self.patcher_pyanalyze_module.start()

        self.mock_dump_value_func = MagicMock()
        self.mock_dump_value_func.side_effect = lambda val: getattr(val, 'mock_type_name', 'UnknownFromDumpValue')
        self.patcher_pyanalyze_dump_value = patch('src.digester.repository_digester.pyanalyze_dump_value', self.mock_dump_value_func)
        self.MockPyanalyzeDumpValue_obj = self.patcher_pyanalyze_dump_value.start()


    def tearDown(self):
        shutil.rmtree(self.test_repo_root)
        self.patcher_get_parser.stop()
        self.patcher_pyanalyze_checker.stop()
        self.patcher_pyanalyze_config.stop()
        self.patcher_pyanalyze_options.stop()
        self.patcher_pyanalyze_module.stop()
        self.patcher_pyanalyze_dump_value.stop()


    def _create_dummy_file(self, path_from_repo_root: str, content: str = ""):
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
        self._create_dummy_file("tests/test_app.py")
        self._create_dummy_file("setup.py")
        self._create_dummy_file(".venv/lib/lib.py")

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


    # --- Tests for _infer_types_with_pyanalyze ---
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

    def test_infer_types_variable_definition(self):
        code = "x = 10\ny: str = 'hello'"
        file_path = self._create_dummy_file("vars.py", code)
        parsed_ast = ast.parse(code)
        setattr(parsed_ast.body[0].targets[0], 'inferred_value', make_mock_inferred_value("int"))
        setattr(parsed_ast.body[1].target, 'inferred_value', make_mock_inferred_value("str"))
        setattr(parsed_ast.body[1].annotation, 'inferred_value', make_mock_inferred_value("Type[str]"))
        self.mock_pyanalyze_module.ast_annotator.annotate_code.return_value = parsed_ast

        digester = RepositoryDigester(self.test_repo_root)
        type_info = digester._infer_types_with_pyanalyze(file_path, code)

        self.assertIsNotNone(type_info)
        self.assertEqual(type_info.get("x:variable_definition:1:0"), "int")
        self.assertEqual(type_info.get("y:variable_definition:2:0"), "str")
        self.assertEqual(type_info.get("y_annotation:type_annotation_itself:2:3"), "Type[str]")
        self.mock_pyanalyze_module.ast_annotator.annotate_code.assert_called_once_with(code, filename=str(file_path), show_errors=False, verbose=False)

    def test_infer_types_function_params_and_return(self):
        code = "def foo(a: int, b) -> str:\n  return str(a + b)"
        file_path = self._create_dummy_file("func.py", code)
        parsed_ast = ast.parse(code)
        func_node = parsed_ast.body[0]
        setattr(func_node.args.args[0], 'inferred_value', make_mock_inferred_value("int"))
        setattr(func_node.args.args[1], 'inferred_value', make_mock_inferred_value("Any"))
        setattr(func_node.returns, 'inferred_value', make_mock_inferred_value("Type[str]"))
        setattr(func_node, 'inferred_value', make_mock_inferred_value("Callable[..., str]"))
        self.mock_pyanalyze_module.ast_annotator.annotate_code.return_value = parsed_ast
        digester = RepositoryDigester(self.test_repo_root)
        type_info = digester._infer_types_with_pyanalyze(file_path, code)
        self.assertIsNotNone(type_info)
        self.assertEqual(type_info.get("foo:function_callable_type:1:0"), "Callable[..., str]")
        self.assertEqual(type_info.get("foo.a:parameter:1:8"), "int")
        self.assertEqual(type_info.get("foo.b:parameter:1:17"), "Any")
        self.assertEqual(type_info.get("foo:function_return_annotation_type:1:24"), "Type[str]")

    def test_infer_types_class_vars_and_methods(self):
        code = """
class MyClass:
    cls_var: bool = True
    def __init__(self, val):
        self.inst_var = val
    def get_val(self) -> int:
        return self.inst_var
"""
        file_path = self._create_dummy_file("cls.py", code)
        parsed_ast = ast.parse(code)
        class_node = parsed_ast.body[0]
        cls_var_ann_assign_node = class_node.body[0]
        init_method_node = class_node.body[1]
        inst_var_assign_node = init_method_node.body[0].targets[0]
        get_val_method_node = class_node.body[2]
        setattr(cls_var_ann_assign_node.target, 'inferred_value', make_mock_inferred_value("bool"))
        setattr(inst_var_assign_node, 'inferred_value', make_mock_inferred_value("Any"))
        setattr(init_method_node.args.args[1], 'inferred_value', make_mock_inferred_value("Any"))
        setattr(get_val_method_node.returns, 'inferred_value', make_mock_inferred_value("Type[int]"))
        setattr(get_val_method_node, 'inferred_value', make_mock_inferred_value("Callable[..., int]"))
        self.mock_pyanalyze_module.ast_annotator.annotate_code.return_value = parsed_ast
        digester = RepositoryDigester(self.test_repo_root)
        type_info = digester._infer_types_with_pyanalyze(file_path, code)
        self.assertIsNotNone(type_info)
        self.assertEqual(type_info.get("MyClass.cls_var:class_variable_definition:3:4"), "bool")
        self.assertEqual(type_info.get("MyClass.__init__.val:parameter:4:23"), "Any")
        self.assertEqual(type_info.get("self.inst_var:attribute_definition:5:8"), "Any")
        self.assertEqual(type_info.get("MyClass.get_val:method_return_annotation_type:6:25"), "Type[int]")
        self.assertEqual(type_info.get("MyClass.get_val:method_callable_type:6:4"), "Callable[..., int]")

    def test_infer_types_call_result(self):
        code = "res = int('10')"
        file_path = self._create_dummy_file("call.py", code)
        parsed_ast = ast.parse(code)
        assign_node = parsed_ast.body[0]
        call_node = assign_node.value
        setattr(call_node, 'inferred_value', make_mock_inferred_value("int"))
        setattr(assign_node.targets[0], 'inferred_value', make_mock_inferred_value("int"))
        self.mock_pyanalyze_module.ast_annotator.annotate_code.return_value = parsed_ast
        digester = RepositoryDigester(self.test_repo_root)
        type_info = digester._infer_types_with_pyanalyze(file_path, code)
        self.assertIsNotNone(type_info)
        self.assertEqual(type_info.get("call_to_int:call_result_type:1:6"), "int")
        self.assertEqual(type_info.get("res:variable_definition:1:0"), "int")

    # --- Tests for parse_file integration of type inference ---
    @patch('src.digester.repository_digester.RepositoryDigester._infer_types_with_pyanalyze')
    def test_parse_file_calls_infer_types_on_success_adapted(self, mock_infer_types):
        mock_infer_types.return_value = {"var_x": "int_from_pyanalyze_mock"}
        code = "x = 1"
        file_path = self._create_dummy_file("type_integ_success.py", code)
        mock_ts_tree = MagicMock(); mock_ts_tree.root_node.has_error = False
        self.mock_ts_parser_instance.parse.return_value = mock_ts_tree

        digester = RepositoryDigester(self.test_repo_root)
        result = digester.parse_file(file_path)

        mock_infer_types.assert_called_once_with(file_path, code)
        self.assertEqual(result.type_info, {"var_x": "int_from_pyanalyze_mock"})
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
        self.mock_ts_parser_instance.parse.return_value = MagicMock(root_node=MagicMock(has_error=False))

        digester = RepositoryDigester(self.test_repo_root)
        result = digester.parse_file(file_path)

        # _infer_types_with_pyanalyze itself handles empty string and returns an info dict.
        # parse_file calls _infer_types_with_pyanalyze if no libcst error AND code is not empty (this was the old logic).
        # New logic: parse_file calls _infer_types if no libcst error. _infer_types handles empty.
        # The prompt for _infer_types_with_pyanalyze was:
        # `if not source_code.strip(): return {"info": "Empty source code, Pyanalyze not run."}`
        # The prompt for parse_file was:
        # `if not libcst_error_str and source_code.strip() : # Only run type inference if code is syntactically valid enough for ast.parse`
        # This means for empty code, `source_code.strip()` is false, so `_infer_types_with_pyanalyze` is NOT called by `parse_file`.
        # Instead, `parse_file` sets `file_type_info = {"info": "Type inference skipped for empty file."}`.

        mock_infer_types.assert_not_called() # Based on `and source_code.strip()` in parse_file
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
