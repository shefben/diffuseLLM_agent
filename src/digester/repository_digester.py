from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import libcst as cst
from dataclasses import dataclass, field

# Import for Tree-sitter
try:
    from tree_sitter import Parser, Language
    # tree_sitter_languages simplifies finding/loading grammars
    from tree_sitter_languages import get_language, get_parser
except ImportError:
    Parser = None # type: ignore
    Language = None # type: ignore
    get_language = None # type: ignore
    get_parser = None # type: ignore
    print("Warning: tree-sitter or tree-sitter-languages not found. Tree-sitter parsing will be disabled.")

# Define actual Tree-sitter Tree type if Language is available
TreeSitterTree = Any
if Language:
    from tree_sitter import Tree as TreeSitterTreeTypeActual # type: ignore
    TreeSitterTree = TreeSitterTreeTypeActual


@dataclass
class ParsedFileResult:
    file_path: Path
    source_code: str
    libcst_module: Optional[cst.Module] = None
    treesitter_tree: Optional[TreeSitterTree] = None # Updated type hint potential
    libcst_error: Optional[str] = None
    treesitter_has_errors: bool = False
    treesitter_error_message: Optional[str] = None # Specific message for TS errors


class RepositoryDigester:
    def __init__(self, repo_path: Union[str, Path]):
        self.repo_path = Path(repo_path)
        if not self.repo_path.is_dir():
            raise ValueError(f"Repository path {repo_path} is not a valid directory.")

        self._all_py_files: List[Path] = []
        self.digested_files: Dict[Path, ParsedFileResult] = {}

        self.ts_parser: Optional[Parser] = None
        if Parser and get_language: # Check if base tree_sitter and tree_sitter_languages are imported
            try:
                # Prefer get_parser if available as it's simpler
                if get_parser:
                    self.ts_parser = get_parser('python')
                else: # Fallback to manual Language loading if get_parser is not found (older tree_sitter_languages?)
                    PYTHON_LANGUAGE = get_language('python')
                    self.ts_parser = Parser()
                    self.ts_parser.set_language(PYTHON_LANGUAGE)
            except Exception as e:
                print(f"Warning: Failed to initialize Tree-sitter Python parser: {e}. Tree-sitter parsing will be disabled.")
                self.ts_parser = None


    def discover_python_files(self, ignored_dirs: Optional[List[str]] = None, ignored_files: Optional[List[str]] = None):
        """
        Discovers all Python files in the repository path, respecting ignores.
        Similar to StyleSampler's discovery but can be customized.
        """
        self._all_py_files = [] # Reset if called multiple times

        default_ignored_dirs = {".venv", "venv", ".env", "env", "node_modules", ".git", "__pycache__", "docs", "examples", "tests", "test"}
        # Allow overriding or extending default ignored dirs
        if ignored_dirs is not None:
            current_ignored_dirs = set(ignored_dirs)
        else:
            current_ignored_dirs = default_ignored_dirs

        default_ignored_files = {"setup.py"} # Example
        if ignored_files is not None:
            current_ignored_files = set(ignored_files)
        else:
            current_ignored_files = default_ignored_files

        for py_file in self.repo_path.rglob("*.py"):
            try:
                if py_file.name in current_ignored_files:
                    continue
                # Check if any part of the relative path is in current_ignored_dirs
                # Ensure py_file is relative to repo_path for this check
                relative_path_parts = py_file.relative_to(self.repo_path).parts
                if any(part in current_ignored_dirs for part in relative_path_parts[:-1]): # Check parent directories
                    continue
                self._all_py_files.append(py_file)
            except Exception as e:
                print(f"Warning: Could not process path {py_file} during discovery: {e}")

        print(f"RepositoryDigester: Discovered {len(self._all_py_files)} Python files.")


    def parse_file(self, file_path: Path) -> ParsedFileResult:
        """
        Parses a single Python file using both LibCST and Tree-sitter.
        (Tree-sitter and LibCST parsing logic will be implemented in subsequent steps)

        Args:
            file_path: Path to the Python file.

        Returns:
            A ParsedFileResult object containing parsing results and/or errors.
        """
        print(f"RepositoryDigester: Parsing {file_path.name}...")
        source_code = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ParsedFileResult(
                file_path=file_path,
                source_code="", # No source if read failed
                libcst_error=f"File read error: {e}",
                treesitter_has_errors=True, treesitter_error_message=f"File read error: {e}"
            )

        # LibCST parsing
        libcst_module_obj: Optional[cst.Module] = None
        libcst_error_str: Optional[str] = None
        try:
            if source_code.strip(): # Avoid parsing empty or whitespace-only strings with LibCST if it causes issues
                libcst_module_obj = cst.parse_module(source_code)
            else: # Handle empty files explicitly for LibCST if needed
                libcst_module_obj = cst.parse_module("") # LibCST can parse empty string into an empty Module
        except cst.ParserSyntaxError as e_libcst_syntax:
            libcst_error_str = f"LibCST ParserSyntaxError: {e_libcst_syntax.message} (lines {e_libcst_syntax.lines})"
            # Additional details if available:
            # print(f"  Raw lines: {e_libcst_syntax.raw_lines}")
            # print(f"  Line: {e_libcst_syntax.lines}")
            # print(f"  Column: {e_libcst_syntax.column}")
            print(f"Warning: LibCST parsing of {file_path.name} failed with syntax error.")
        except Exception as e_libcst_general:
            libcst_error_str = f"LibCST general error: {e_libcst_general}"
            print(f"Warning: LibCST parsing of {file_path.name} failed with general error: {e_libcst_general}")

        # Tree-sitter parsing (existing logic from previous step)
        treesitter_tree_obj: Optional[TreeSitterTree] = None
        treesitter_errors_flag: bool = False
        treesitter_error_msg: Optional[str] = None

        if self.ts_parser:
            try:
                tree_bytes = bytes(source_code, "utf8")
                treesitter_tree_obj = self.ts_parser.parse(tree_bytes)
                if treesitter_tree_obj.root_node.has_error:
                    treesitter_errors_flag = True
                    treesitter_error_msg = "Tree-sitter found syntax errors in the file."
                    print(f"Warning: Tree-sitter parsing of {file_path.name} completed with errors.")
            except Exception as e_ts:
                 treesitter_errors_flag = True
                 treesitter_error_msg = f"Tree-sitter parsing exception: {e_ts}"
                 print(f"Error during Tree-sitter parsing of {file_path.name}: {e_ts}")
        else:
            treesitter_errors_flag = True
            treesitter_error_msg = "Tree-sitter parser not initialized or language not found."

        return ParsedFileResult(
            file_path=file_path,
            source_code=source_code,
            libcst_module=libcst_module_obj,
            treesitter_tree=treesitter_tree_obj,
            libcst_error=libcst_error_str,
            treesitter_has_errors=treesitter_errors_flag,
            treesitter_error_message=treesitter_error_msg
        )

    # (digest_repository method remains the same for this subtask)
    def digest_repository(self):
        """
        Discovers all Python files and parses them using Tree-sitter and LibCST,
        storing the results.
        """
        self.discover_python_files() # Uses default ignores for now

        if not self._all_py_files:
            print("RepositoryDigester: No Python files to digest.")
            return

        num_total_files = len(self._all_py_files)
        # These counters are removed as the summary is now generated from self.digested_files at the end
        # num_processed_ok = 0
        # num_libcst_errors = 0
        # num_treesitter_errors = 0

        print(f"RepositoryDigester: Starting digestion of {num_total_files} Python files...")

        for i, py_file in enumerate(self._all_py_files):
            print(f"RepositoryDigester: Processing file {i+1}/{num_total_files}: {py_file.name}...")
            if py_file not in self.digested_files: # Avoid re-digesting if called multiple times
                parse_result = self.parse_file(py_file)
                self.digested_files[py_file] = parse_result

                # Log specific errors if they occurred for this file
                if parse_result.libcst_error:
                    print(f"  LibCST Error for {py_file.name}: {parse_result.libcst_error[:100]}...")

                if parse_result.treesitter_has_errors:
                    if parse_result.treesitter_error_message:
                        print(f"  Tree-sitter Issues for {py_file.name}: {parse_result.treesitter_error_message[:100]}...")
                    else:
                        print(f"  Tree-sitter Issues for {py_file.name}: Errors present in tree (no specific message).")
            # No 'else' needed here for summary counters, as summary is from final dict.

        print(f"RepositoryDigester: Digestion complete.")
        print(f"  Total Python files initially discovered: {num_total_files}")
        print(f"  Files for which parsing was attempted: {len(self.digested_files)}")

        files_fully_ok_both_parsers = sum(
            1 for res in self.digested_files.values()
            if res.libcst_module and not res.libcst_error and res.treesitter_tree and not res.treesitter_has_errors
        )
        files_with_any_libcst_error = sum(1 for res in self.digested_files.values() if res.libcst_error)
        files_with_any_treesitter_issue = sum(1 for res in self.digested_files.values() if res.treesitter_has_errors)

        print(f"  Files successfully parsed by LibCST & Tree-sitter (no errors/issues): {files_fully_ok_both_parsers}")
        print(f"  Files with LibCST parsing errors: {files_with_any_libcst_error}")
        print(f"  Files with Tree-sitter parsing issues: {files_with_any_treesitter_issue}")

        # (Later, this method will trigger graph building, embeddings, etc. using self.digested_files)

if __name__ == '__main__':
    # Example usage (assuming a dummy repo path)
    # Create a dummy repo for testing
    current_script_dir = Path(__file__).parent
    dummy_repo = current_script_dir / "_temp_dummy_repo_for_digester_"
    dummy_repo.mkdir(exist_ok=True)

    (dummy_repo / "file1.py").write_text("def foo():\n    pass\n")
    (dummy_repo / "subdir").mkdir(exist_ok=True)
    (dummy_repo / "subdir" / "file2.py").write_text("class Bar:\n    x = 1\n")
    (dummy_repo / "tests").mkdir(exist_ok=True) # Ignored by default
    (dummy_repo / "tests" / "test_file.py").write_text("assert True")

    print(f"Digesting dummy repo at: {dummy_repo.resolve()}")
    digester = RepositoryDigester(str(dummy_repo))
    digester.digest_repository()

    print("\nDigested file results (summary):")
    for file_path, result in digester.digested_files.items():
        print(f"  {file_path.name}:")
        print(f"    LibCST Error: {result.libcst_error}")
        print(f"    TreeSitter Has Errors: {result.treesitter_has_errors}")
        # print(f"    LibCST Module: {'Present' if result.libcst_module else 'Absent'}")
        # print(f"    TreeSitter Tree: {'Present' if result.treesitter_tree else 'Absent'}")

    # Clean up dummy repo
    import shutil
    shutil.rmtree(dummy_repo)
    print(f"Cleaned up dummy repo: {dummy_repo}")
