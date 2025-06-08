from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import libcst as cst
from dataclasses import dataclass, field
import sys # For manipulating sys.path if needed for pyanalyze
import os  # For path manipulation
import ast # For Pyanalyze integration

# Tree-sitter imports
try:
    from tree_sitter import Parser, Language
    from tree_sitter_languages import get_language, get_parser
except ImportError: # Fallback definitions
    Parser, Language, get_language, get_parser = None, None, None, None
    # Warning for tree-sitter is now in __init__ if it fails there
TreeSitterTree = Any
if Language: # Check if Language was successfully imported
    from tree_sitter import Tree as TreeSitterTreeTypeActual # type: ignore
    TreeSitterTree = TreeSitterTreeTypeActual


# Pyanalyze imports (guarded)
try:
    import pyanalyze
    from pyanalyze import name_check_visitor # Though not directly used in this placeholder, it's key for real impl.
    from pyanalyze.value import Value, KnownValue, TypedValue, dump_value as pyanalyze_dump_value
    from pyanalyze.checker import Checker
    from pyanalyze.config import Config
    from pyanalyze.options import Options
    from pyanalyze.error_code import ErrorCode # Not used, but good to have for ref
except ImportError:
    pyanalyze = None # type: ignore
    name_check_visitor = None # type: ignore
    Checker = None # type: ignore
    pyanalyze_dump_value = None # type: ignore
    Config = None # type: ignore
    Options = None # type: ignore
    # Warning for pyanalyze is now in __init__

@dataclass
class ParsedFileResult:
    file_path: Path
    source_code: str
    libcst_module: Optional[cst.Module] = None
    treesitter_tree: Optional[TreeSitterTree] = None
    libcst_error: Optional[str] = None
    treesitter_has_errors: bool = False
    treesitter_error_message: Optional[str] = None
    type_info: Optional[Dict[str, Any]] = None # Field for type inference data

class PyanalyzeTypeExtractionVisitor(ast.NodeVisitor):
    def __init__(self, file_path_str: str):
        self.file_path_str = file_path_str # For creating unique keys
        self.type_info_map: Dict[str, str] = {}
        # pyanalyze_dump_value should be available from the global scope of repository_digester
        self.dump_value_func = pyanalyze_dump_value

    def _add_type_info(self, node: ast.AST, name: str, category: str = "variable"):
        if hasattr(node, 'inferred_value') and self.dump_value_func:
            inferred_val = getattr(node, 'inferred_value')
            if inferred_val is not None:
                type_str = self.dump_value_func(inferred_val)
                node_key = f"{name}:{category}:{node.lineno}:{node.col_offset}"
                self.type_info_map[node_key] = type_str

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self._add_type_info(node, node.id, "variable_def")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.returns:
            self._add_type_info(node.returns, node.name, "function_return_annotation")
        for arg_node in node.args.args:
            self._add_type_info(arg_node, f"{node.name}.{arg_node.arg}", "parameter")
        if node.args.vararg:
            self._add_type_info(node.args.vararg, f"{node.name}.*{node.args.vararg.arg}", "parameter_vararg")
        if node.args.kwarg:
            self._add_type_info(node.args.kwarg, f"{node.name}.**{node.args.kwarg.arg}", "parameter_kwarg")
        for kwarg_node_only in node.args.kwonlyargs:
             self._add_type_info(kwarg_node_only, f"{node.name}.{kwarg_node_only.arg}", "parameter_kwonly")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                self._add_type_info(child.target, f"{node.name}.{child.target.id}", "class_variable")
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                         self._add_type_info(target, f"{node.name}.{target.id}", "class_variable")
        self.generic_visit(node)

class RepositoryDigester:
    def __init__(self, repo_path: Union[str, Path]):
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.is_dir():
            raise ValueError(f"Repository path {self.repo_path} is not a valid directory.")

        self._all_py_files: List[Path] = []
        self.digested_files: Dict[Path, ParsedFileResult] = {}

        self.ts_parser: Optional[Parser] = None
        if Parser and get_language:
            try:
                if get_parser: self.ts_parser = get_parser('python')
                else:
                    PYTHON_LANGUAGE = get_language('python')
                    self.ts_parser = Parser()
                    self.ts_parser.set_language(PYTHON_LANGUAGE)
            except Exception as e:
                print(f"Warning: Failed to initialize Tree-sitter Python parser: {e}.")
                self.ts_parser = None
        else:
            print("Warning: tree-sitter or tree-sitter-languages not found during RepositoryDigester init. Tree-sitter parsing disabled.")
            self.ts_parser = None

        self.pyanalyze_checker: Optional[Checker] = None
        if pyanalyze and Checker and Config and Options:
            try:
                pyanalyze_options = Options(paths=[str(self.repo_path)])
                pyanalyze_config = Config.from_options(pyanalyze_options)
                self.pyanalyze_checker = Checker(config=pyanalyze_config)
                print("Pyanalyze Checker initialized.")
            except Exception as e:
                print(f"Warning: Failed to initialize Pyanalyze Checker: {e}. Type inference will be disabled.")
                self.pyanalyze_checker = None
        else:
            if not pyanalyze:
                 print("Warning: pyanalyze library components not found. Type inference via Pyanalyze will be disabled.")
            self.pyanalyze_checker = None

    def discover_python_files(self, ignored_dirs: Optional[List[str]] = None, ignored_files: Optional[List[str]] = None):
        self._all_py_files = []
        default_ignored_dirs = {".venv", "venv", ".env", "env", "node_modules", ".git", "__pycache__", "docs", "examples", "tests", "test"}
        current_ignored_dirs = set(ignored_dirs) if ignored_dirs is not None else default_ignored_dirs
        default_ignored_files = {"setup.py"}
        current_ignored_files = set(ignored_files) if ignored_files is not None else default_ignored_files
        for py_file in self.repo_path.rglob("*.py"):
            try:
                if py_file.name in current_ignored_files: continue
                if any(part in current_ignored_dirs for part in py_file.relative_to(self.repo_path).parts[:-1]): continue
                self._all_py_files.append(py_file)
            except Exception as e: print(f"Warning: Could not process path {py_file} during discovery: {e}")
        print(f"RepositoryDigester: Discovered {len(self._all_py_files)} Python files.")

    def _infer_types_with_pyanalyze(self, file_path: Path, source_code: str) -> Optional[Dict[str, Any]]:
        if not (pyanalyze and hasattr(pyanalyze, 'ast_annotator') and hasattr(pyanalyze.ast_annotator, 'annotate_code') and pyanalyze_dump_value):
            print(f"Pyanalyze (or ast_annotator/dump_value) not available, skipping type inference for {file_path.name}.")
            return {"info": "Pyanalyze components unavailable."}

        original_sys_path = list(sys.path)
        file_dir_str = str(file_path.parent.resolve())
        repo_path_str = str(self.repo_path.resolve())

        paths_to_add = []
        if file_dir_str not in sys.path: paths_to_add.append(file_dir_str)
        if repo_path_str not in sys.path and repo_path_str != file_dir_str: paths_to_add.append(repo_path_str)

        for p_add in reversed(paths_to_add):
            sys.path.insert(0, p_add)

        type_info_map: Dict[str, Any] = {}
        try:
            print(f"Pyanalyze: Annotating AST for {file_path.name}...")

            if not source_code.strip():
                 return {"info": "Empty source code, Pyanalyze not run."}

            annotated_ast = pyanalyze.ast_annotator.annotate_code(source_code, str(file_path), show_errors=False, verbose=False)

            if annotated_ast:
                print(f"Pyanalyze: Extracting types from annotated AST for {file_path.name}...")
                visitor = PyanalyzeTypeExtractionVisitor(str(file_path))
                visitor.visit(annotated_ast)
                type_info_map = visitor.type_info_map
                if not type_info_map:
                    type_info_map = {"info": "Pyanalyze ran but no types extracted by visitor."}
            else:
                type_info_map = {"error": "Pyanalyze annotate_code returned None."}

        except Exception as e:
            print(f"Error during Pyanalyze processing for {file_path.name}: {e}")
            type_info_map = {"error": f"Pyanalyze processing failed: {e}"}
        finally:
            new_sys_path = []
            current_sys_path = list(sys.path)
            for p in current_sys_path:
                if p not in paths_to_add:
                    new_sys_path.append(p)
            sys.path = new_sys_path

        return type_info_map if type_info_map else None

    def parse_file(self, file_path: Path) -> ParsedFileResult:
        source_code = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f: source_code = f.read()
        except Exception as e:
            return ParsedFileResult(file_path=file_path, source_code="",
                                    libcst_error=f"File read error: {e}",
                                    treesitter_has_errors=True, treesitter_error_message=f"File read error: {e}",
                                    type_info={"error": f"File read error: {e}"})

        libcst_module_obj: Optional[cst.Module] = None
        libcst_error_str: Optional[str] = None
        try:
            libcst_module_obj = cst.parse_module(source_code)
        except cst.ParserSyntaxError as e_libcst_syntax:
            libcst_error_str = f"LibCST ParserSyntaxError: {e_libcst_syntax.message} (lines {e_libcst_syntax.lines})"
        except Exception as e_libcst_general:
            libcst_error_str = f"LibCST general error: {e_libcst_general}"

        treesitter_tree_obj: Optional[TreeSitterTree] = None
        treesitter_errors_flag: bool = False
        treesitter_error_msg: Optional[str] = None
        if self.ts_parser:
            try:
                treesitter_tree_obj = self.ts_parser.parse(bytes(source_code, "utf8"))
                if treesitter_tree_obj.root_node.has_error:
                    treesitter_errors_flag = True
                    treesitter_error_msg = "Tree-sitter found syntax errors in the file."
            except Exception as e_ts:
                 treesitter_errors_flag = True; treesitter_error_msg = f"Tree-sitter parsing exception: {e_ts}"
        else:
            treesitter_errors_flag = True; treesitter_error_msg = "Tree-sitter parser not initialized."

        file_type_info: Optional[Dict[str, Any]] = None
        if not libcst_error_str and source_code.strip() :
            try:
                file_type_info = self._infer_types_with_pyanalyze(file_path, source_code)
            except Exception as e_typeinf:
                print(f"Error calling type inference for {file_path.name}: {e_typeinf}")
                file_type_info = {"error": f"Type inference call failed: {e_typeinf}"}
        elif not source_code.strip():
             file_type_info = {"info": "Type inference skipped for empty file."}
        else:
            file_type_info = {"info": "Type inference skipped due to LibCST parsing errors."}

        return ParsedFileResult(
            file_path=file_path, source_code=source_code,
            libcst_module=libcst_module_obj, treesitter_tree=treesitter_tree_obj,
            libcst_error=libcst_error_str,
            treesitter_has_errors=treesitter_errors_flag, treesitter_error_message=treesitter_error_msg,
            type_info=file_type_info
        )

    def digest_repository(self):
        self.discover_python_files()
        if not self._all_py_files:
            print("RepositoryDigester: No Python files to digest.")
            return
        num_total_files = len(self._all_py_files)
        print(f"RepositoryDigester: Starting digestion of {num_total_files} Python files...")

        for i, py_file in enumerate(self._all_py_files):
            print(f"RepositoryDigester: Processing file {i+1}/{num_total_files}: {py_file.name}...")
            if py_file not in self.digested_files:
                parse_result = self.parse_file(py_file)
                self.digested_files[py_file] = parse_result
        print(f"RepositoryDigester: Digestion complete.")
        files_fully_ok_both_parsers = sum(1 for res in self.digested_files.values() if res.libcst_module and not res.libcst_error and res.treesitter_tree and not res.treesitter_has_errors)
        files_with_any_libcst_error = sum(1 for res in self.digested_files.values() if res.libcst_error)
        files_with_any_treesitter_issue = sum(1 for res in self.digested_files.values() if res.treesitter_has_errors)
        files_with_type_info = sum(1 for res in self.digested_files.values() if res.type_info and not res.type_info.get("error") and not res.type_info.get("info"))

        print(f"  Total Python files found: {len(self._all_py_files)}")
        print(f"  Files for which parsing was attempted: {len(self.digested_files)}")
        print(f"  Files successfully parsed by LibCST & Tree-sitter (no errors): {files_fully_ok_both_parsers}")
        print(f"  Files with LibCST parsing errors: {files_with_any_libcst_error}")
        print(f"  Files with Tree-sitter parsing issues: {files_with_any_treesitter_issue}")
        print(f"  Files with some type info extracted: {files_with_type_info}")

if __name__ == '__main__':
    current_script_dir = Path(__file__).parent
    dummy_repo = current_script_dir / "_temp_dummy_repo_for_digester_"
    dummy_repo.mkdir(exist_ok=True)
    (dummy_repo / "file1.py").write_text("def foo(x: int) -> str:\n    return str(x)\n")
    (dummy_repo / "subdir").mkdir(exist_ok=True)
    (dummy_repo / "subdir" / "file2.py").write_text("class Bar:\n    y: list[str] = []\n")
    (dummy_repo / "tests").mkdir(exist_ok=True)
    (dummy_repo / "tests" / "test_file.py").write_text("assert True")

    print(f"Digesting dummy repo at: {dummy_repo.resolve()}")
    original_sys_path_main = list(sys.path)
    if str(dummy_repo.resolve()) not in sys.path:
        sys.path.insert(0, str(dummy_repo.resolve()))

    digester = RepositoryDigester(str(dummy_repo))
    digester.digest_repository()

    print("\nDigested file results (summary):")
    for file_path, result in digester.digested_files.items():
        print(f"  {file_path.name}:")
        print(f"    LibCST Error: {result.libcst_error}")
        print(f"    TreeSitter Has Errors: {result.treesitter_has_errors}")
        print(f"    Type Info: {result.type_info}")

    # Restore sys.path carefully
    paths_to_remove_main = []
    if str(dummy_repo.resolve()) in sys.path and str(dummy_repo.resolve()) not in original_sys_path_main :
         paths_to_remove_main.append(str(dummy_repo.resolve()))

    cleaned_sys_path = [p for p in sys.path if p not in paths_to_remove_main]
    # If original_sys_path had paths that are not in cleaned_sys_path now (e.g. due to other manipulations)
    # this simple removal might not be perfect. A more robust way is to restore to original_sys_path_main
    # if we are sure no other thread/part of program expects changes to sys.path.
    # For this script's __main__, direct restoration is likely fine.
    sys.path = original_sys_path_main

    import shutil
    shutil.rmtree(dummy_repo)
    print(f"Cleaned up dummy repo: {dummy_repo}")
