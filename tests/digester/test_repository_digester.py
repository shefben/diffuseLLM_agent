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
from src.digester.repository_digester import RepositoryDigester, ParsedFileResult, _create_ast_node_id
from src.digester.graph_structures import NodeID, CallGraph, ControlDependenceGraph, DataDependenceGraph
from src.digester.signature_trie import generate_function_signature_string # For signature trie tests
from typing import Dict, Set, Optional, Any, Callable # Ensure Any is imported for NumpyNdarray fallback
import numpy as np

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

        # Mocks for SentenceTransformer and FAISS
        self.mock_embedding_model_instance = MagicMock()
        self.mock_embedding_model_instance.get_sentence_embedding_dimension.return_value = 384
        self.mock_embedding_model_instance.encode.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

        self.patcher_sentence_transformer = patch('src.digester.repository_digester.SentenceTransformer', return_value=self.mock_embedding_model_instance)
        self.MockSentenceTransformer = self.patcher_sentence_transformer.start()

        self.mock_faiss_index_instance = MagicMock()
        self.mock_faiss_index_instance.ntotal = 0
        def faiss_add_side_effect(embeddings_array):
            self.mock_faiss_index_instance.ntotal += len(embeddings_array)
        self.mock_faiss_index_instance.add.side_effect = faiss_add_side_effect

        # Patch where faiss.IndexFlatL2 is looked up
        self.patcher_faiss_index_constructor = patch('src.digester.repository_digester.faiss.IndexFlatL2', return_value=self.mock_faiss_index_instance)
        self.MockFaissIndexFlatL2 = self.patcher_faiss_index_constructor.start()

        # Mock the faiss module itself if it's checked directly (e.g., 'if faiss:')
        self.patcher_faiss_module = patch('src.digester.repository_digester.faiss', MagicMock(IndexFlatL2=self.MockFaissIndexFlatL2))
        self.MockFaissModule_obj = self.patcher_faiss_module.start()

        # Patch numpy if repository_digester.py imports it as 'np'
        self.patcher_numpy_module = patch('src.digester.repository_digester.np', np)
        self.MockNumpyModule_obj = self.patcher_numpy_module.start()

    def tearDown(self):
        shutil.rmtree(self.test_repo_root)
        self.patcher_get_parser.stop()
        self.patcher_pyanalyze_checker.stop()
        self.patcher_pyanalyze_config.stop()
        self.patcher_pyanalyze_options.stop()
        self.patcher_pyanalyze_module.stop()
        self.patcher_pyanalyze_dump_value.stop()
        self.patcher_sentence_transformer.stop()
        self.patcher_faiss_index_constructor.stop()
        self.patcher_faiss_module.stop()
        self.patcher_numpy_module.stop()

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

    # --- Tests for FQN Helper Methods ---
    def test_get_module_qname_from_path(self):
        # Test cases: (file_rel_path_str, expected_qname)
        # project_root for these tests will be self.test_repo_root

        test_cases = [
            ("module.py", "module"),
            ("pkg/module.py", "pkg.module"),
            ("pkg/subpkg/module.py", "pkg.subpkg.module"),
            # Current behavior based on: ".".join(relative_path.with_suffix("").parts)
            ("pkg/subpkg/__init__.py", "pkg.subpkg.__init__"),
            ("__init__.py", "__init__"),
            ("file_with_no_py_suffix", "file_with_no_py_suffix")
        ]

        for file_rel_str, expected_qname in test_cases:
            # The file doesn't actually need to exist for this static method.
            file_path = self.test_repo_root / file_rel_str

            with self.subTest(file_path=str(file_path), expected=expected_qname): # Use str forsubTest
                qname = RepositoryDigester._get_module_qname_from_path(file_path, self.test_repo_root)
                self.assertEqual(qname, expected_qname)

    def test_get_module_qname_from_path_outside_root(self):
        """Test behavior when file_path is not under project_root."""
        # Construct a path that is intentionally outside the test_repo_root
        # Using Path.joinpath or / operator for robust path construction
        outside_path = Path(tempfile.gettempdir()) / "some_other_project_temp" / "another_module.py"

        # Ensure the directory exists for a more realistic scenario if needed, though not strictly by method
        # For this test, it's more about how Path objects behave.
        # outside_path.parent.mkdir(parents=True, exist_ok=True) # Not strictly needed for this method

        qname = RepositoryDigester._get_module_qname_from_path(outside_path, self.test_repo_root)
        # Fallback behavior: expects file_path.stem due to ValueError from relative_to
        self.assertEqual(qname, "another_module")

        # Clean up if we created the directory structure (optional, but good practice if we did)
        # if outside_path.exists():
        #     shutil.rmtree(Path(tempfile.gettempdir()) / "some_other_project_temp", ignore_errors=True)


    def test_get_module_qname_from_path_same_as_root(self):
        """Test behavior when file_path is the same as project_root (edge case)."""
        # This scenario is unlikely for a .py file but tests path logic.
        # file_path = self.test_repo_root (which is a directory)
        # relative_to will raise ValueError if file_path == project_root.
        # Current _get_module_qname_from_path catches ValueError and returns file_path.stem (which is dir name)

        qname = RepositoryDigester._get_module_qname_from_path(self.test_repo_root, self.test_repo_root)
        self.assertEqual(qname, self.test_repo_root.name)

    # --- Tests for Call Graph Construction ---

    def _run_call_graph_test(self, code_content: str, file_name: str = "test_module.py",
                             mock_type_info: Optional[Dict[str, str]] = None) -> CallGraph:
        file_path = self._create_dummy_file(file_name, code_content)

        # Ensure tree-sitter parsing is mocked as successful for parse_file to proceed
        mock_ts_tree = MagicMock()
        mock_ts_tree.root_node.has_error = False
        self.mock_ts_parser_instance.parse.return_value = mock_ts_tree

        digester = RepositoryDigester(self.test_repo_root) # Re-init for clean graph etc.

        # Mock _infer_types_with_pyanalyze to control the type_info provided to CallGraphVisitor
        # This method is called by digester.parse_file()
        with patch.object(digester, '_infer_types_with_pyanalyze', return_value=mock_type_info if mock_type_info else {}) as mock_infer_types:
            # Patch _all_py_files to ensure digest_repository only processes our test file
            with patch.object(digester, '_all_py_files', [file_path]):
                 digester.digest_repository() # This calls parse_file, then _build_call_graph_for_file

        return digester.project_call_graph


    def test_cg_direct_intra_module_call(self):
        code = """
def callee(): pass
def caller(): callee()
"""
        call_graph = self._run_call_graph_test(code, "module_a.py")

        caller_fqn = NodeID("module_a.caller")
        callee_fqn = NodeID("module_a.callee")

        self.assertIn(caller_fqn, call_graph)
        self.assertIn(callee_fqn, call_graph[caller_fqn])

    def test_cg_method_call_on_self(self):
        code = """
class MyClass:
    def method_b(self): pass
    def method_a(self): self.method_b()
"""
        call_graph = self._run_call_graph_test(code, "module_b.py")

        caller_fqn = NodeID("module_b.MyClass.method_a")
        callee_fqn = NodeID("module_b.MyClass.method_b")

        self.assertIn(caller_fqn, call_graph)
        self.assertIn(callee_fqn, call_graph[caller_fqn])

    def test_cg_method_call_on_typed_object(self):
        code = """
class OtherClass:
    def target_method(self): pass

def main_func():
    obj = OtherClass()
    obj.target_method() # Call to resolve
"""
        # Key for 'obj' definition: "obj:variable_definition:LINE:COL"
        # PyanalyzeTypeExtractionVisitor uses f"{name}:{category}:{node.lineno}:{node.col_offset}"
        # In this code: obj is ast.Name node, lineno=5, col_offset=4
        # So key is "obj:variable_definition:5:4"
        mock_types = {
            "obj:variable_definition:5:4": "module_c.OtherClass"
        }
        call_graph = self._run_call_graph_test(code, "module_c.py", mock_type_info=mock_types)

        caller_fqn = NodeID("module_c.main_func")
        callee_fqn = NodeID("module_c.OtherClass.target_method")

        self.assertIn(caller_fqn, call_graph)
        self.assertIn(callee_fqn, call_graph[caller_fqn])

    def test_cg_class_instantiation_as_call(self):
        code = """
class MyTargetClass:
    def __init__(self): pass

def creator():
    x = MyTargetClass() # This is a call
"""
        # Type info for 'MyTargetClass' when used as a callable.
        # The CallGraphVisitor's _resolve_callee_fqn has a heuristic:
        # if re.match(r"^[A-Z]", callee_name): return NodeID(f"{self.module_qname}.{callee_name}.__init__")
        # This heuristic will be used if type info is not specific enough or not found.
        # For this test, let's rely on the heuristic for MyTargetClass.
        call_graph = self._run_call_graph_test(code, "module_d.py", mock_type_info=None) # No specific type info needed for this one due to heuristic

        caller_fqn = NodeID("module_d.creator")
        callee_fqn = NodeID("module_d.MyTargetClass.__init__")

        self.assertIn(caller_fqn, call_graph)
        self.assertIn(callee_fqn, call_graph[caller_fqn])

    def test_cg_no_calls_in_file(self):
        code = "def no_calls():\n  x = 1 + 2"
        call_graph = self._run_call_graph_test(code, "module_e.py")
        self.assertEqual(len(call_graph), 0)

    def test_cg_skip_file_with_libcst_error(self):
        file_path = self._create_dummy_file("bad_syntax_for_cg.py", "def func( : oops")

        digester = RepositoryDigester(self.test_repo_root)
        # Mock _infer_types_with_pyanalyze because it's called by parse_file
        with patch.object(digester, '_infer_types_with_pyanalyze', return_value={}) as mock_infer_types:
             # Patch _all_py_files to ensure digest_repository only processes our test file
            with patch.object(digester, '_all_py_files', [file_path]):
                with patch.object(digester, '_build_call_graph_for_file', wraps=digester._build_call_graph_for_file) as spy_build_cg:
                    digester.digest_repository()
                    # _build_call_graph_for_file IS called by digest_repository.
                    # Inside _build_call_graph_for_file, it should return early due to no LibCST module.
                    spy_build_cg.assert_called_once()
                    self.assertEqual(len(digester.project_call_graph), 0, "Call graph should be empty for file with LibCST errors.")

    def test_cg_imported_function_call_simplified(self):
        code = """
from my_other_lib import utility_func
def main_caller():
    utility_func()
"""
        # CallGraphVisitor's _resolve_callee_fqn uses heuristics if type info is missing.
        # For an ast.Name like 'utility_func', it might become "importer_module.utility_func".
        # To test proper resolution with type info:
        # Key for 'utility_func' usage (ast.Name): "utility_func:variable_usage:LINE:COL"
        # If PyanalyzeTypeExtractionVisitor was more detailed, it would provide this.
        # For now, CallGraphVisitor's _resolve_callee_fqn for cst.Name doesn't use type_info yet.
        # It assumes local or uses heuristics.
        # The heuristic for cst.Name is: NodeID(f"{self.module_qname}.{callee_name}")
        # So, this test will reflect that current limitation/behavior.

        # If we wanted to test type_info driven resolution for imports, CallGraphVisitor._resolve_callee_fqn
        # would need enhancement for cst.Name nodes using type_info.
        # For now, this tests the existing heuristic.
        call_graph = self._run_call_graph_test(code, "importer_module.py", mock_type_info=None)

        caller_fqn = NodeID("importer_module.main_caller")
        # Expected based on current heuristic: module_qname + callee_name
        callee_fqn_heuristic = NodeID("importer_module.utility_func")

        # If _resolve_callee_fqn were enhanced to use type_info for cst.Name:
        # mock_types = { "utility_func:name_usage_key:3:4": "my_other_lib.utility_func" }
        # callee_fqn_ideal = NodeID("my_other_lib.utility_func")
        # For now, assert current behavior:
        self.assertIn(caller_fqn, call_graph)
        self.assertIn(callee_fqn_heuristic, call_graph[caller_fqn])

    # --- Tests for _create_ast_node_id ---
    def test_create_ast_node_id_uniqueness_and_format(self):
        file_id = "test_file.py"

        # Simple name node
        name_node = ast.Name(id="my_var", ctx=ast.Store(), lineno=1, col_offset=0, end_lineno=1, end_col_offset=6)
        node_id_name = _create_ast_node_id(file_id, name_node, "definition")
        self.assertEqual(node_id_name, NodeID(f"{file_id}:1:0:Name:my_var:definition"))

        # FunctionDef node
        func_node = ast.FunctionDef(name="my_func", args=MagicMock(), body=[], decorator_list=[], lineno=3, col_offset=0, end_lineno=3, end_col_offset=15)
        node_id_func = _create_ast_node_id(file_id, func_node, "declaration")
        self.assertEqual(node_id_func, NodeID(f"{file_id}:3:0:FunctionDef:my_func:declaration"))

        # If node (test part) - ast.Compare has no 'name' or 'id'
        if_node_test = ast.Compare(left=ast.Name(id='x', ctx=ast.Load(), lineno=5, col_offset=3, end_lineno=5, end_col_offset=8), ops=[ast.Eq()], comparators=[ast.Constant(value=1, lineno=5, col_offset=9, end_lineno=5, end_col_offset=10)], lineno=5, col_offset=3, end_lineno=5, end_col_offset=10)
        node_id_if_test = _create_ast_node_id(file_id, if_node_test, "if_condition")
        self.assertEqual(node_id_if_test, NodeID(f"{file_id}:5:3:Compare:if_condition"))


    # --- Tests for Control Dependence Graph Construction ---
    def _run_cdg_test(self, code_content: str, file_name: str = "cdg_test_module.py") -> ControlDependenceGraph:
        file_path = self._create_dummy_file(file_name, code_content)

        # For CDG, _build_control_dependencies_for_file primarily needs source_code.
        # Other fields of ParsedFileResult are not essential for this specific unit of testing.
        parsed_result = ParsedFileResult(
            file_path=file_path,
            source_code=code_content,
            libcst_module=None,
            treesitter_tree=None,
            type_info=None
        )

        digester = RepositoryDigester(self.test_repo_root)
        # We are testing _build_control_dependencies_for_file in isolation.
        # It updates digester.project_control_dependence_graph
        digester._build_control_dependencies_for_file(file_path, parsed_result)
        return digester.project_control_dependence_graph

    def test_cdg_if_statement(self):
        code = """
if x == 1: # line 2
    y = 1   # line 3
else:
    y = 2   # line 5
"""
        file_name = "if_stmt.py"
        cdg = self._run_cdg_test(code, file_name)

        parsed_ast = ast.parse(code)
        if_node = parsed_ast.body[0]
        condition_node_id = _create_ast_node_id(file_name, if_node.test, "if_condition")
        body_stmt_node_id = _create_ast_node_id(file_name, if_node.body[0], "statement")
        orelse_stmt_node_id = _create_ast_node_id(file_name, if_node.orelse[0], "statement")

        self.assertIn(condition_node_id, cdg)
        self.assertIn(body_stmt_node_id, cdg[condition_node_id])
        self.assertIn(orelse_stmt_node_id, cdg[condition_node_id])
        self.assertEqual(len(cdg[condition_node_id]), 2)

    def test_cdg_for_loop(self):
        code = """
for i in range(3): # line 2
    print(i)       # line 3
else:
    print("done")  # line 5
"""
        file_name = "for_loop.py"
        cdg = self._run_cdg_test(code, file_name)

        parsed_ast = ast.parse(code)
        for_node = parsed_ast.body[0]
        controller_id = _create_ast_node_id(file_name, for_node.iter, "for_iterable")
        body_stmt_id = _create_ast_node_id(file_name, for_node.body[0], "statement")
        orelse_stmt_id = _create_ast_node_id(file_name, for_node.orelse[0], "statement")

        self.assertIn(controller_id, cdg)
        self.assertIn(body_stmt_id, cdg[controller_id])
        self.assertIn(orelse_stmt_id, cdg[controller_id])

    def test_cdg_while_loop(self):
        code = """
while x < 5: # line 2
    x += 1   # line 3
"""
        file_name = "while_loop.py"
        cdg = self._run_cdg_test(code, file_name)
        parsed_ast = ast.parse(code)
        while_node = parsed_ast.body[0]
        controller_id = _create_ast_node_id(file_name, while_node.test, "while_condition")
        body_stmt_id = _create_ast_node_id(file_name, while_node.body[0], "statement")

        self.assertIn(controller_id, cdg)
        self.assertIn(body_stmt_id, cdg[controller_id])

    def test_cdg_try_except_finally(self):
        code = """
try:         # line 2
    x = 1/0  # line 3
except ZeroDivisionError as e: # line 4
    print(e) # line 5
finally:
    print("cleanup") # line 7
"""
        file_name = "try_stmt.py"
        cdg = self._run_cdg_test(code, file_name)
        parsed_ast = ast.parse(code)
        try_node = parsed_ast.body[0]

        try_construct_id = _create_ast_node_id(file_name, try_node, "try_construct")
        try_body_stmt_id = _create_ast_node_id(file_name, try_node.body[0], "statement")

        handler_node = try_node.handlers[0]
        # Controller for handler body is the handler's type node (or handler itself if type is None)
        handler_condition_id = _create_ast_node_id(file_name, handler_node.type, "except_handler_condition")
        handler_body_stmt_id = _create_ast_node_id(file_name, handler_node.body[0], "statement")

        # Statements in finalbody depend on the try_construct_id
        finally_body_stmt_id = _create_ast_node_id(file_name, try_node.finalbody[0], "finally_statement")

        self.assertIn(try_construct_id, cdg)
        self.assertIn(try_body_stmt_id, cdg[try_construct_id])
        self.assertIn(finally_body_stmt_id, cdg[try_construct_id])

        self.assertIn(handler_condition_id, cdg)
        self.assertIn(handler_body_stmt_id, cdg[handler_condition_id])

    def test_cdg_with_statement(self):
        code = """
with open("f.txt") as f: # line 2
    content = f.read()   # line 3
"""
        file_name = "with_stmt.py"
        cdg = self._run_cdg_test(code, file_name)
        parsed_ast = ast.parse(code)
        with_node = parsed_ast.body[0]

        # The ControlDependenceVisitor uses the With node itself as the controller for its body
        controller_id = _create_ast_node_id(file_name, with_node, "with_block")
        body_stmt_id = _create_ast_node_id(file_name, with_node.body[0], "statement")

        self.assertIn(controller_id, cdg)
        self.assertIn(body_stmt_id, cdg[controller_id])

    def test_cdg_nested_controls(self):
        code = """
if a:      # L2 IfA
    for x in y: # L3 ForX
        if b:   # L4 IfB
            c() # L5 Ccall
"""
        file_name = "nested.py"
        cdg = self._run_cdg_test(code, file_name)
        parsed_ast = ast.parse(code)
        if_a_node = parsed_ast.body[0]
        for_x_node = if_a_node.body[0]
        if_b_node = for_x_node.body[0]
        c_call_expr_node = if_b_node.body[0] # ast.Expr node

        if_a_cond_id = _create_ast_node_id(file_name, if_a_node.test, "if_condition")
        # The for_x_node (as a whole statement) depends on if_a_cond
        for_x_stmt_id = _create_ast_node_id(file_name, for_x_node, "statement")

        for_x_iter_id = _create_ast_node_id(file_name, for_x_node.iter, "for_iterable")
        # The if_b_node (as a whole statement) depends on for_x_iter
        if_b_stmt_id = _create_ast_node_id(file_name, if_b_node, "statement")

        if_b_cond_id = _create_ast_node_id(file_name, if_b_node.test, "if_condition")
        # The c_call_expr_node (ast.Expr statement) depends on if_b_cond
        c_call_stmt_id = _create_ast_node_id(file_name, c_call_expr_node, "statement")

        self.assertIn(if_a_cond_id, cdg)
        self.assertIn(for_x_stmt_id, cdg[if_a_cond_id])

        self.assertIn(for_x_iter_id, cdg)
        self.assertIn(if_b_stmt_id, cdg[for_x_iter_id])

        self.assertIn(if_b_cond_id, cdg)
        self.assertIn(c_call_stmt_id, cdg[if_b_cond_id])

    def test_cdg_build_handles_syntax_error_in_ast_parse(self):
        file_path = self._create_dummy_file("syntax_err_cdg.py", "def func( : oops")
        # Source code that will fail ast.parse
        parsed_result_with_error = ParsedFileResult(
            file_path=file_path, source_code="def func( : oops",
            libcst_module=None, treesitter_tree=None
        )
        digester = RepositoryDigester(self.test_repo_root)
        with patch('builtins.print') as mock_print:
            digester._build_control_dependencies_for_file(file_path, parsed_result_with_error)

        self.assertEqual(len(digester.project_control_dependence_graph), 0)
        self.assertTrue(any("SyntaxError parsing" in str(c.args) for c in mock_print.call_args_list))

    # --- Helper for DDG Tests ---
    def _run_ddg_test(self, code_content: str, file_name: str = "ddg_test_module.py") -> DataDependenceGraph:
        file_path = self._create_dummy_file(file_name, code_content)

        parsed_result = ParsedFileResult(
            file_path=file_path,
            source_code=code_content,
            # Other fields (libcst, treesitter, type_info) are not directly used by current DDG logic
        )

        # Create a new digester instance for each DDG test to have a clean graph
        digester = RepositoryDigester(self.test_repo_root)
        digester._build_data_dependencies_for_file(file_path, parsed_result)
        return digester.project_data_dependence_graph

    # --- Tests for Data Dependence Graph Construction ---

    def test_ddg_simple_def_use(self):
        code = "x = 1\ny = x" # x def at L1, x use at L2
        file_name = "simple_def_use.py"
        ddg = self._run_ddg_test(code, file_name)

        parsed_ast = ast.parse(code)
        x_def_node = parsed_ast.body[0].targets[0] # ast.Name 'x' in Store ctx
        y_assign_node = parsed_ast.body[1]       # ast.Assign for y = x
        x_use_node = y_assign_node.value         # ast.Name 'x' in Load ctx

        x_def_id = _create_ast_node_id(file_name, x_def_node, "x:def")
        x_use_id_in_y = _create_ast_node_id(file_name, x_use_node, "x:use")

        self.assertIn(x_use_id_in_y, ddg)
        self.assertIn(x_def_id, ddg[x_use_id_in_y])

    def test_ddg_redefinition(self):
        code = "x = 1\nx = 2\ny = x" # y = x should depend on the second def of x
        file_name = "redef.py"
        ddg = self._run_ddg_test(code, file_name)

        parsed_ast = ast.parse(code)
        # x_def1_node = parsed_ast.body[0].targets[0] # First def of x
        x_def2_node = parsed_ast.body[1].targets[0] # Second def of x
        x_use_node = parsed_ast.body[2].value       # Use of x

        # x_def1_id = _create_ast_node_id(file_name, x_def1_node, "x:def")
        x_def2_id = _create_ast_node_id(file_name, x_def2_node, "x:def")
        x_use_id = _create_ast_node_id(file_name, x_use_node, "x:use")

        self.assertIn(x_use_id, ddg)
        self.assertIn(x_def2_id, ddg[x_use_id])
        # self.assertNotIn(x_def1_id, ddg[x_use_id]) # Current logic (overwrite def) ensures this

    def test_ddg_function_parameter_use(self):
        code = "def foo(a):\n  b = a" # Use of 'a' in b=a depends on param 'a'
        file_name = "func_param.py"
        ddg = self._run_ddg_test(code, file_name)

        parsed_ast = ast.parse(code)
        func_node = parsed_ast.body[0]
        param_a_node = func_node.args.args[0] # ast.arg 'a'
        assign_b_node = func_node.body[0]     # ast.Assign b = a
        use_a_node = assign_b_node.value      # ast.Name 'a' (Load)

        param_a_def_id = _create_ast_node_id(file_name, param_a_node, "a:def")
        use_a_id = _create_ast_node_id(file_name, use_a_node, "a:use")

        self.assertIn(use_a_id, ddg)
        self.assertIn(param_a_def_id, ddg[use_a_id])

    def test_ddg_scope_shadowing(self):
        code = """
x = 1        # Module x (def1)
def f():
    x = 2    # Function f's x (def2)
    y = x    # Use of f.x (depends on def2)
f()
z = x        # Use of module x (depends on def1)
"""
        file_name = "scope.py"
        ddg = self._run_ddg_test(code, file_name)
        parsed_ast = ast.parse(code)

        x_def1_node = parsed_ast.body[0].targets[0] # Module x
        func_f_node = parsed_ast.body[1]            # Function f
        x_def2_node = func_f_node.body[0].targets[0] # f.x
        y_assign_node = func_f_node.body[1]         # y = x in f
        x_use_in_y_node = y_assign_node.value
        z_assign_node = parsed_ast.body[3]          # z = x in module (index 2 is f() call)
        x_use_in_z_node = z_assign_node.value

        x_def1_id = _create_ast_node_id(file_name, x_def1_node, "x:def")
        x_def2_id = _create_ast_node_id(file_name, x_def2_node, "x:def")
        x_use_in_y_id = _create_ast_node_id(file_name, x_use_in_y_node, "x:use")
        x_use_in_z_id = _create_ast_node_id(file_name, x_use_in_z_node, "x:use")

        self.assertIn(x_use_in_y_id, ddg, "Use of x in y not found in DDG")
        self.assertIn(x_def2_id, ddg[x_use_in_y_id], "y=x should depend on inner x def")
        self.assertNotIn(x_def1_id, ddg[x_use_in_y_id], "y=x should NOT depend on outer x def")

        self.assertIn(x_use_in_z_id, ddg, "Use of x in z not found in DDG")
        self.assertIn(x_def1_id, ddg[x_use_in_z_id], "z=x should depend on outer x def")

    def test_ddg_loop_variable_def_use(self):
        code = "for i in range(3):\n  print(i)" # print(i) uses 'i' defined by for
        file_name = "ddg_for.py"
        ddg = self._run_ddg_test(code, file_name)
        parsed_ast = ast.parse(code)
        for_node = parsed_ast.body[0]
        loop_var_i_target_node = for_node.target # ast.Name 'i' (Store)
        print_call_node = for_node.body[0].value # ast.Call print(i)
        loop_var_i_use_node = print_call_node.args[0] # ast.Name 'i' (Load)

        loop_var_i_def_id = _create_ast_node_id(file_name, loop_var_i_target_node, "i:def")
        loop_var_i_use_id = _create_ast_node_id(file_name, loop_var_i_use_node, "i:use")

        self.assertIn(loop_var_i_use_id, ddg)
        self.assertIn(loop_var_i_def_id, ddg[loop_var_i_use_id])

    def test_ddg_use_in_if_condition(self):
        code = "x=0\nif x > 0:\n  pass" # use of x in condition
        file_name = "ddg_if.py"
        ddg = self._run_ddg_test(code, file_name)
        parsed_ast = ast.parse(code)
        x_def_node = parsed_ast.body[0].targets[0]
        if_node = parsed_ast.body[1]
        x_use_in_cond_node = if_node.test.left # x from 'x > 0'

        x_def_id = _create_ast_node_id(file_name, x_def_node, "x:def")
        x_use_id = _create_ast_node_id(file_name, x_use_in_cond_node, "x:use")

        self.assertIn(x_use_id, ddg)
        self.assertIn(x_def_id, ddg[x_use_id])

    def test_ddg_build_handles_syntax_error_gracefully(self):
        """Ensures DDG isn't built and doesn't crash if ast.parse fails."""
        file_path = self._create_dummy_file("syntax_err_ddg.py", "def func( : oops")
        # Source code that will fail ast.parse
        parsed_result_with_error = ParsedFileResult(file_path=file_path, source_code="def func( : oops")

        digester = RepositoryDigester(self.test_repo_root)
        with patch('builtins.print') as mock_print:
            digester._build_data_dependencies_for_file(file_path, parsed_result_with_error)

        self.assertEqual(len(digester.project_data_dependence_graph), 0)
        printed_output = "".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn("SyntaxError parsing syntax_err_ddg.py with ast for data dependence", printed_output)

    # --- Tests for Embedding and FAISS Initialization in __init__ ---
    def test_init_embedding_model_and_faiss_success(self):
        # Reset mocks that might have been called during RepositoryDigester instantiation in other tests' helpers
        self.MockSentenceTransformer.reset_mock()
        self.MockFaissIndexFlatL2.reset_mock()

        digester = RepositoryDigester(self.test_repo_root, embedding_model_name="test-model")

        self.MockSentenceTransformer.assert_called_once_with("test-model")
        self.assertIsNotNone(digester.embedding_model)
        self.assertEqual(digester.embedding_dimension, 384)
        self.mock_embedding_model_instance.get_sentence_embedding_dimension.assert_called_once()
        self.MockFaissIndexFlatL2.assert_called_once_with(384)
        self.assertIsNotNone(digester.faiss_index)

    @patch('src.digester.repository_digester.SentenceTransformer', None)
    def test_init_sentence_transformer_not_available(self, mock_st_none):
        # This test needs to ensure that the global SentenceTransformer symbol in repository_digester
        # is None when __init__ is called.
        with patch('builtins.print') as mock_print:
            digester = RepositoryDigester(self.test_repo_root) # Re-instantiate to trigger __init__ with patch
        self.assertIsNone(digester.embedding_model)
        self.assertIsNone(digester.embedding_dimension)
        self.assertIsNone(digester.faiss_index)
        self.assertTrue(any("sentence-transformers library not available" in str(c.args) for c in mock_print.call_args_list))

    @patch('src.digester.repository_digester.faiss', None)
    def test_init_faiss_not_available(self, mock_faiss_none):
        self.MockSentenceTransformer.reset_mock() # Ensure ST model loads
        with patch('builtins.print') as mock_print:
            digester = RepositoryDigester(self.test_repo_root) # Re-instantiate
        self.assertIsNotNone(digester.embedding_model)
        self.assertIsNone(digester.faiss_index)
        self.assertTrue(any("faiss library not available" in str(c.args) for c in mock_print.call_args_list))

    # --- Tests for Embedding Generation and FAISS population in digest_repository ---
    @patch('src.digester.repository_digester.RepositoryDigester.parse_file')
    def test_digest_repository_embedding_and_faiss_flow(self, mock_parse_file):
        file1_path = self._create_dummy_file("file1_embed.py", "def f1(): pass")

        symbol1_f1_code = {"fqn": "file1_embed.f1", "item_type": "function_code", "content": "def f1(): pass", "file_path": str(file1_path), "start_line":1, "end_line":1}
        symbol2_f1_doc = {"fqn": "file1_embed.f1", "item_type": "docstring_for_function", "content": "doc for f1", "file_path": str(file1_path), "start_line":1, "end_line":1}

        parsed_result_file1 = ParsedFileResult(
            file_path=file1_path, source_code="def f1(): pass",
            extracted_symbols=[symbol1_f1_code, symbol2_f1_doc],
            libcst_module=MagicMock() # Needs a truthy LibCST module for graph builders to be called
        )
        mock_parse_file.return_value = parsed_result_file1

        dummy_embedding_dim = 384 # Must match what get_sentence_embedding_dimension returns
        dummy_embedding1 = np.random.rand(dummy_embedding_dim).astype(np.float32)
        dummy_embedding2 = np.random.rand(dummy_embedding_dim).astype(np.float32)
        self.mock_embedding_model_instance.encode.return_value = [dummy_embedding1, dummy_embedding2]

        digester = RepositoryDigester(self.test_repo_root)
        digester._all_py_files = [file1_path]

        with patch('src.digester.repository_digester.np.vstack') as mock_vstack:
            mock_vstack.return_value = np.array([dummy_embedding1, dummy_embedding2])
            digester.digest_repository()

        self.mock_embedding_model_instance.encode.assert_called_once_with(
            ["def f1(): pass", "doc for f1"], show_progress_bar=False
        )
        self.assertIsNotNone(symbol1_f1_code.get("embedding"))
        self.assertTrue(np.array_equal(symbol1_f1_code["embedding"], dummy_embedding1))
        self.assertIsNotNone(symbol2_f1_doc.get("embedding"))
        self.assertTrue(np.array_equal(symbol2_f1_doc["embedding"], dummy_embedding2))

        mock_vstack.assert_called_once()
        args_vstack, _ = mock_vstack.call_args
        self.assertEqual(len(args_vstack[0]), 2)
        self.assertTrue(np.array_equal(args_vstack[0][0], dummy_embedding1))

        self.mock_faiss_index_instance.add.assert_called_once()
        args_faiss_add, _ = self.mock_faiss_index_instance.add.call_args
        self.assertTrue(isinstance(args_faiss_add[0], np.ndarray))
        self.assertEqual(args_faiss_add[0].dtype, np.float32)
        self.assertEqual(self.mock_faiss_index_instance.ntotal, 2)


        self.assertEqual(len(digester.faiss_id_to_metadata), 2)
        self.assertEqual(digester.faiss_id_to_metadata[0]["fqn"], "file1_embed.f1")
        self.assertEqual(digester.faiss_id_to_metadata[0]["item_type"], "function_code")
        self.assertEqual(digester.faiss_id_to_metadata[1]["fqn"], "file1_embed.f1")
        self.assertEqual(digester.faiss_id_to_metadata[1]["item_type"], "docstring_for_function")

    @patch('src.digester.repository_digester.RepositoryDigester.parse_file')
    def test_digest_repository_no_embeddings_generated(self, mock_parse_file):
        file1_path = self._create_dummy_file("file_no_embed_content.py", "def f1(): pass")
        symbol1_f1_code = {"fqn": "file1.f1", "item_type": "function_code", "content": "  ", "file_path": str(file1_path)}
        parsed_result_file1 = ParsedFileResult(
            file_path=file1_path, source_code="def f1(): pass",
            extracted_symbols=[symbol1_f1_code],
            libcst_module=MagicMock() # For graph builders
        )
        mock_parse_file.return_value = parsed_result_file1

        digester = RepositoryDigester(self.test_repo_root)
        digester._all_py_files = [file1_path]

        # Reset encode mock for this specific test as it might be called by other tests' __init__
        self.mock_embedding_model_instance.encode = MagicMock()

        digester.digest_repository()

        self.mock_embedding_model_instance.encode.assert_not_called()
        self.mock_faiss_index_instance.add.assert_not_called()
        self.assertEqual(len(digester.faiss_id_to_metadata), 0)
        self.assertEqual(self.mock_faiss_index_instance.ntotal, 0)

    # --- Tests for Signature Trie Population ---
    def test_extract_symbols_populates_signature_trie(self):
        """
        Tests that _extract_symbols_and_docstrings_from_ast (when called, e.g., by parse_file)
        correctly generates signatures and inserts them into the trie.
        """
        code = """
def func_one(a: int, b: str) -> bool:
    pass

class MyClass:
    def method_one(self, c: float) -> None:
        pass
"""
        file_name = "sig_trie_test.py"
        file_path = self._create_dummy_file(file_name, code)
        module_qname = "sig_trie_test"

        # Mock type_info that _extract_symbols_and_docstrings_from_ast's visitor would use
        # Keys are based on SymbolAndSignatureExtractorVisitor._local_type_resolver's logic
        # For parameters: "FunctionName.ParamName:parameter:Line:Col"
        # For return: "FunctionName.<return_annotation>:function_return_annotation:Line:Col"
        mock_type_info_for_resolver = {
            f"func_one.a:parameter:2:{len('def func_one(')}": "int",
            f"func_one.b:parameter:2:{len('def func_one(a: int, ')}": "str",
            f"func_one.<return_annotation>:function_return_annotation:2:{len('def func_one(a: int, b: str) -> ')}": "bool",

            f"MyClass.method_one.self:parameter:6:{len('    def method_one(')}": "sig_trie_test.MyClass",
            f"MyClass.method_one.c:parameter:6:{len('    def method_one(self, ')}": "float",
            f"MyClass.method_one.<return_annotation>:function_return_annotation:6:{len('    def method_one(self, c: float) -> ')}": "None",
        }

        py_ast_module = ast.parse(code)
        source_code_lines = code.splitlines()

        digester = RepositoryDigester(self.test_repo_root)

        with patch.object(digester.signature_trie, 'insert') as mock_trie_insert:
            # Note: _extract_symbols_and_docstrings_from_ast expects Dict[str,str] for type_info
            # but PyanalyzeTypeExtractionVisitor produces Dict[str,Any] (can be error dicts).
            # The logic in parse_file filters this. So, pass a correctly filtered map.
            digester._extract_symbols_and_docstrings_from_ast(
                py_ast_module, module_qname, file_path, source_code_lines, mock_type_info_for_resolver
            )

        self.assertGreaterEqual(mock_trie_insert.call_count, 2)

        # Expected signature for func_one
        func_one_node = py_ast_module.body[0]
        # This local resolver mimics the one inside SymbolAndSignatureExtractorVisitor
        def local_resolver(node: ast.AST, hint: str) -> Optional[str]:
            key_name_part = hint.split(":")[0]
            key_category_part = hint.split(":")[-1]
            if key_category_part == "return_annotation":
                key_to_lookup = f"{key_name_part}.<return_annotation>:function_return_annotation:{node.lineno}:{node.col_offset}"
            elif key_category_part.startswith("parameter"):
                key_to_lookup = f"{key_name_part}:parameter:{node.lineno}:{node.col_offset}"
            else:
                key_to_lookup = f"{key_name_part}:{key_category_part}:{node.lineno}:{node.col_offset}"
            return mock_type_info_for_resolver.get(key_to_lookup)

        expected_sig_func_one = generate_function_signature_string(func_one_node, local_resolver) # type: ignore
        expected_fqn_func_one = f"{module_qname}.func_one"

        class_node = py_ast_module.body[1]
        method_one_node = class_node.body[0]
        expected_sig_method_one = generate_function_signature_string(method_one_node, local_resolver) # type: ignore
        expected_fqn_method_one = f"{module_qname}.MyClass.method_one"

        mock_trie_insert.assert_any_call(expected_sig_func_one, expected_fqn_func_one)
        mock_trie_insert.assert_any_call(expected_sig_method_one, expected_fqn_method_one)


    @patch('src.digester.repository_digester.RepositoryDigester._infer_types_with_pyanalyze')
    @patch('src.digester.repository_digester.RepositoryDigester._build_call_graph_for_file')
    @patch('src.digester.repository_digester.RepositoryDigester._build_control_dependencies_for_file')
    @patch('src.digester.repository_digester.RepositoryDigester._build_data_dependencies_for_file')
    # No need to mock SymbolAndSignatureExtractorVisitor.visit, we mock trie.insert directly
    def test_digest_repository_populates_signature_trie(
            self, mock_ddg, mock_cdg, mock_cg, mock_infer_types):

        code = "def main_func(p1: int) -> str: pass"
        file_name = "main_for_trie.py"
        file_path = self._create_dummy_file(file_name, code)
        module_qname = "main_for_trie"

        # Mock Pyanalyze to return type_info for the function
        # Keys need to match what _local_type_resolver in SymbolAndSignatureExtractorVisitor expects
        func_node_for_lines = ast.parse(code).body[0] # type: ignore
        param_p1_node = func_node_for_lines.args.args[0]
        return_node = func_node_for_lines.returns

        mock_type_info = {
            f"{module_qname}.main_func.p1:parameter:{param_p1_node.lineno}:{param_p1_node.col_offset}": "int",
            f"{module_qname}.main_func.<return_annotation>:function_return_annotation:{return_node.lineno}:{return_node.col_offset}": "str"
        }
        mock_infer_types.return_value = mock_type_info

        mock_ts_tree = MagicMock(); mock_ts_tree.root_node.has_error = False
        self.mock_ts_parser_instance.parse.return_value = mock_ts_tree

        digester = RepositoryDigester(self.test_repo_root)
        # Mock the insert method of the trie instance that the digester created
        digester.signature_trie.insert = MagicMock()

        digester.digest_repository()

        expected_signature = "main_func(int)->str"
        expected_fqn = f"{module_qname}.main_func"
        digester.signature_trie.insert.assert_called_once_with(expected_signature, expected_fqn)

    # --- Tests for Incremental Update Methods ---

    def test_clear_data_for_file(self):
        digester = RepositoryDigester(self.test_repo_root)
        file_a_rel = "file_a.py"
        file_b_rel = "file_b.py"
        # Resolve paths as they would be stored in digested_files keys
        file_a_abs = (self.test_repo_root / file_a_rel).resolve()
        file_b_abs = (self.test_repo_root / file_b_rel).resolve()

        # For NodeIDs, the prefix is based on relative path to repo_root or filename
        # In _clear_data_for_file, it's `str(file_path.relative_to(self.repo_path))`
        # So, for file_a_abs, the prefix will be file_a_rel

        mock_parsed_a = ParsedFileResult(file_path=file_a_abs, source_code="def fa(): pass", extracted_symbols=[{"fqn":"file_a.fa", "file_path":str(file_a_abs)}])
        mock_parsed_b = ParsedFileResult(file_path=file_b_abs, source_code="def fb(): pass", extracted_symbols=[{"fqn":"file_b.fb", "file_path":str(file_b_abs)}])
        digester.digested_files = {file_a_abs: mock_parsed_a, file_b_abs: mock_parsed_b}

        node_a1 = NodeID(f"{file_a_rel}:1:0:FunctionDef:fa:def")
        node_a2 = NodeID(f"{file_a_rel}:2:0:Name:x:use") # Conceptual node from file_a
        node_b1 = NodeID(f"{file_b_rel}:1:0:FunctionDef:fb:def")
        node_b2 = NodeID(f"{file_b_rel}:2:0:Name:y:use") # Conceptual node from file_b

        digester.project_call_graph = {node_a1: {node_b1}, node_b1: {node_a2, node_b2}}

        digester.faiss_id_to_metadata = [
            {"file_path": str(file_a_abs), "fqn": "file_a.fa"},
            {"file_path": str(file_b_abs), "fqn": "file_b.fb"},
            {"file_path": str(file_a_abs), "fqn": "file_a.fa_doc"}
        ]
        # Mock signature trie (simplified: assume no real removal for this placeholder test)
        digester.signature_trie.delete = MagicMock()

        digester._clear_data_for_file(file_a_abs)

        self.assertNotIn(file_a_abs, digester.digested_files)
        self.assertIn(file_b_abs, digester.digested_files)

        self.assertNotIn(node_a1, digester.project_call_graph)
        self.assertIn(node_b1, digester.project_call_graph)
        self.assertNotIn(node_a2, digester.project_call_graph.get(node_b1, set()))
        self.assertIn(node_b2, digester.project_call_graph.get(node_b1, set()))

        # FAISS metadata handled by _rebuild_full_faiss_index, _clear_data_for_file doesn't directly modify it now
        # The placeholder version of _clear_data_for_file in the prompt *did* modify it.
        # The version in the SUT *does* modify self.faiss_id_to_metadata
        self.assertEqual(len(digester.faiss_id_to_metadata), 1) # Based on SUT's _clear_data_for_file logic
        if digester.faiss_id_to_metadata: # Check if list is not empty
            self.assertEqual(digester.faiss_id_to_metadata[0]["fqn"], "file_b.fb")


    def test_rebuild_full_faiss_index(self):
        digester = RepositoryDigester(self.test_repo_root)
        if not digester.embedding_model or not digester.embedding_dimension or not src.digester.repository_digester.faiss or not src.digester.repository_digester.np:
            self.skipTest("Dependencies for FAISS/embedding (faiss, numpy, SentenceTransformer) not available or model not loaded.")

        file1 = self._create_dummy_file("file1.py", "def f1(): pass")
        file2 = self._create_dummy_file("file2.py", "def f2(): pass")

        emb_dim = digester.embedding_dimension
        emb1 = np.random.rand(emb_dim).astype(np.float32)
        emb2 = np.random.rand(emb_dim).astype(np.float32)
        emb3 = np.random.rand(emb_dim).astype(np.float32)

        digester.digested_files[file1.resolve()] = ParsedFileResult(file1.resolve(), "", extracted_symbols=[
            {"fqn":"f1","item_type":"function_code","content":"c1","file_path":str(file1.resolve()),"embedding":emb1},
            {"fqn":"f1d","item_type":"docstring","content":"d1","file_path":str(file1.resolve()),"embedding":emb2}
        ])
        digester.digested_files[file2.resolve()] = ParsedFileResult(file2.resolve(), "", extracted_symbols=[
            {"fqn":"f2","item_type":"function_code","content":"c2","file_path":str(file2.resolve()),"embedding":emb3}
        ])

        # Reset the mocked faiss_index instance that was created in setUp for the digester instance
        # This ensures we are testing the re-initialization within _rebuild_full_faiss_index
        digester.faiss_index = MagicMock(spec=src.digester.repository_digester.faiss.IndexFlatL2 if src.digester.repository_digester.faiss else MagicMock())
        digester.faiss_index.ntotal = 0
        def add_side_effect_rebuild(embeddings_array): digester.faiss_index.ntotal += len(embeddings_array)
        digester.faiss_index.add.side_effect = add_side_effect_rebuild

        # We need to ensure that when _rebuild_full_faiss_index calls `faiss.IndexFlatL2`,
        # it gets our new mock, not the one from global setUp.
        with patch('src.digester.repository_digester.faiss.IndexFlatL2', return_value=digester.faiss_index) as mock_faiss_constructor_in_rebuild:
            digester._rebuild_full_faiss_index()

        mock_faiss_constructor_in_rebuild.assert_called_once_with(emb_dim)
        self.assertEqual(digester.faiss_index.add.call_count, 1)
        args_add, _ = digester.faiss_index.add.call_args
        self.assertEqual(args_add[0].shape[0], 3)
        self.assertEqual(len(digester.faiss_id_to_metadata), 3)
        self.assertEqual(digester.faiss_id_to_metadata[0]["fqn"], "f1")
        self.assertEqual(digester.faiss_id_to_metadata[2]["fqn"], "f2")

    @patch('src.digester.repository_digester.RepositoryDigester._rebuild_full_faiss_index')
    @patch('src.digester.repository_digester.RepositoryDigester._build_data_dependencies_for_file')
    @patch('src.digester.repository_digester.RepositoryDigester._build_control_dependencies_for_file')
    @patch('src.digester.repository_digester.RepositoryDigester._build_call_graph_for_file')
    @patch('src.digester.repository_digester.RepositoryDigester.parse_file')
    @patch('src.digester.repository_digester.RepositoryDigester._clear_data_for_file')
    def test_update_or_add_file(self, mock_clear, mock_parse, mock_build_cg, mock_build_cdg, mock_build_ddg, mock_rebuild_faiss):
        file_path_abs = self.test_repo_root / "updated.py"
        self._create_dummy_file("updated.py", "new content")

        mock_parsed_result = ParsedFileResult(file_path_abs, "new content", libcst_module=MagicMock(), extracted_symbols=[{"content":"sym"}])
        mock_parse.return_value = mock_parsed_result

        digester = RepositoryDigester(self.test_repo_root)
        # Ensure necessary components are mocked if not fully available/reliable from setUp
        if not digester.embedding_model: digester.embedding_model = MagicMock()
        if not digester.faiss_index: digester.faiss_index = MagicMock()
        if not src.digester.repository_digester.np: src.digester.repository_digester.np = np


        digester._update_or_add_file(file_path_abs)

        mock_clear.assert_called_once_with(file_path_abs)
        mock_parse.assert_called_once_with(file_path_abs)
        self.assertEqual(digester.digested_files[file_path_abs], mock_parsed_result)
        mock_build_cg.assert_called_once_with(file_path_abs, mock_parsed_result)
        mock_build_cdg.assert_called_once_with(file_path_abs, mock_parsed_result)
        mock_build_ddg.assert_called_once_with(file_path_abs, mock_parsed_result)
        mock_rebuild_faiss.assert_called_once()

    @patch('src.digester.repository_digester.RepositoryDigester._clear_data_for_file')
    @patch('src.digester.repository_digester.RepositoryDigester._rebuild_full_faiss_index')
    def test_remove_file_data(self, mock_rebuild_faiss, mock_clear):
        file_rel_path = "to_remove.py"
        file_abs_path = self.test_repo_root / file_rel_path
        self._create_dummy_file(file_rel_path)

        digester = RepositoryDigester(self.test_repo_root)
        digester._all_py_files = [file_abs_path]
        digester.digested_files[file_abs_path] = MagicMock()

        digester._remove_file_data(file_abs_path)

        mock_clear.assert_called_once_with(file_abs_path)
        mock_rebuild_faiss.assert_called_once()
        self.assertNotIn(file_abs_path, digester._all_py_files)
        # _clear_data_for_file is mocked, so it won't remove from digested_files unless we assert its internal behavior
        # or let it run (but it's complex). For this test, mock_clear is enough.
        # If we want to check digested_files, mock_clear should have a side_effect or not be mocked.
        # For now, assume mock_clear implies removal from digested_files as per its contract.


    @patch('src.digester.repository_digester.RepositoryDigester._update_or_add_file')
    @patch('src.digester.repository_digester.RepositoryDigester._remove_file_data')
    def test_handle_file_event_dispatch(self, mock_remove, mock_update_add):
        digester = RepositoryDigester(self.test_repo_root)
        src_rel_path = "src.py"
        dest_rel_path = "dest.py"
        src_abs_path = self._create_dummy_file(src_rel_path) # Create for path resolution
        dest_abs_path = self.test_repo_root / dest_rel_path # Does not need to exist for this test logic

        digester.handle_file_event("created", src_abs_path)
        mock_update_add.assert_called_once_with(src_abs_path)
        mock_update_add.reset_mock()

        digester.handle_file_event("modified", src_abs_path)
        mock_update_add.assert_called_once_with(src_abs_path)
        mock_update_add.reset_mock()

        digester.handle_file_event("deleted", src_abs_path)
        mock_remove.assert_called_once_with(src_abs_path)
        mock_remove.reset_mock()

        digester.handle_file_event("moved", src_abs_path, dest_abs_path)
        mock_remove.assert_called_once_with(src_abs_path)
        mock_update_add.assert_called_once_with(dest_abs_path)

    def test_handle_file_event_path_outside_repo(self):
        digester = RepositoryDigester(self.test_repo_root)
        # Create a path that's genuinely outside the temp repo_root
        with tempfile.NamedTemporaryFile(suffix=".py", delete=True) as tmp_file:
            outside_path = Path(tmp_file.name)

        with patch('builtins.print') as mock_print:
            digester.handle_file_event("created", outside_path)

        # Ensure the warning about path being outside is printed
        # The exact path string might vary (e.g. /tmp/ vs /var/folders/) so check for substring
        self.assertTrue(any(f"Warning: Event path {outside_path.resolve()} is outside configured repo root" in str(c.args) for c in mock_print.call_args_list))


if __name__ == '__main__':
    unittest.main()
