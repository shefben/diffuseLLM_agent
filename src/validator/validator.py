# src/validator/validator.py
from typing import Optional, Tuple, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
import re # Import re for regular expressions

if TYPE_CHECKING:
    from src.digester.repository_digester import RepositoryDigester
    # from src.planner.phase_model import Phase # If phase_ctx were needed by validation tools

class Validator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Validator.
        Args:
            config: Optional dictionary for validator configurations (e.g., tool paths).
        """
        self.config = config if config else {}
        self.common_stdlib_modules = [
            "os", "sys", "math", "re", "json", "collections", "pathlib",
            "datetime", "time", "argparse", "logging", "subprocess", "multiprocessing",
            "threading", "socket", "ssl", "http", "urllib", "tempfile", "shutil",
            "glob", "io", "pickle", "base64", "hashlib", "hmac", "uuid", "functools", "itertools",
            "operator", "typing", "dataclasses", "enum", "inspect", "gc", "weakref"
        ]
        # Example: self.ruff_path = self.config.get("ruff_path", "ruff")
        print(f"Validator initialized (mock tools). Config: {self.config}. Known stdlib modules: {len(self.common_stdlib_modules)}")

    def attempt_heuristic_fixes(
        self,
        error_traceback: str,
        original_patch_script_str: Optional[str], # This is the LibCST script string from the agent
        target_file_path_str: str, # The path of the file the script targets
        # project_root: Path, # Might be needed for context in real fixes
        # digester: 'RepositoryDigester' # Might be needed for context in real fixes
    ) -> Optional[str]:
        """
        Attempts to apply simple heuristic fixes to a LibCST patch script string
        based on common error patterns in a traceback.
        """
        if not original_patch_script_str:
            print("Validator: Original patch script is None or empty, cannot attempt heuristic fixes.")
            return None

        print(f"Validator: Attempting heuristic fixes for error (first 100 chars): {error_traceback[:100]}...")

        modified_script = original_patch_script_str

        # Heuristic 1: Add Missing Import for NameError
        name_error_match = re.search(r"NameError: name '(\w+)' is not defined", error_traceback)
        if name_error_match:
            module_name = name_error_match.group(1)
            if module_name in self.common_stdlib_modules:
                import_statement = f"import {module_name}\n"
                # Check if import (or from module import) already exists - very basic check
                if f"import {module_name}" not in original_patch_script_str and \
                   f"from {module_name}" not in original_patch_script_str:
                    # Simplistic prepend. A real version would try to place it better within the CST script.
                    modified_script = import_statement + modified_script
                    print(f"Validator: Heuristic fix applied - prepended import for '{module_name}'.")
                    return modified_script # Return after first successful fix
            else:
                print(f"Validator: NameError for '{module_name}' found, but not in common stdlib list for auto-import.")

        # Heuristic 2: Add Trailing Newline (if Black-diff indicated it)
        # Ensure this matches the exact error string from _run_black_diff mock
        # The mock error is f"Black-diff: Would add trailing newline to {file_path.name}"
        # So we search for the core part of the message.
        if "Black-diff: Would add trailing newline" in error_traceback:
            if not modified_script.endswith("\n"):
                modified_script += "\n"
                print("Validator: Heuristic fix applied - added trailing newline to the script content.")
                return modified_script # Return after first successful fix

        # If no heuristics were applied, or if a heuristic made a change but didn't return early
        if modified_script == original_patch_script_str:
            print(f"Validator: No applicable heuristic fixes successfully applied for target '{target_file_path_str}'.")
            return None # No changes made by heuristics
        else:
            # This case should ideally not be reached if fixes return early.
            # If it is, means a fix was made but didn't return.
            print(f"Validator: Heuristic fix made but did not return early (unexpected). Returning modified script for {target_file_path_str}.")
            return modified_script


    def _apply_patch_script(self, original_content: str, patch_script_str: str) -> str:
        """
        Placeholder for applying a LibCST patch script to original content.
        For now, it simulates a modification. In a real scenario, this would involve
        parsing the script, instantiating the CodemodCommand, and applying it.
        """
        print(f"Validator._apply_patch_script: Applying mock patch. Original len: {len(original_content)}, Script len: {len(patch_script_str)}")
        # For this mock, assume the patch_script_str itself isn't the full new content,
        # but a script that would generate some additions/changes.
        # A more realistic mock for validation would be if patch_script_str IS the new content.
        # However, current agent flow produces a CST script.
        # Let's simulate the script adding its own content as a comment block for now.
        return original_content + f"\n\n# --- Mock Patch Applied by Validator ---\n# Script content was:\n# {patch_script_str.replacechr(10,chr(10)+'# ')}\n# --- End Mock Patch ---"

    def _run_ruff(self, modified_content: str, file_path: Path) -> Optional[str]:
        """Mock for running Ruff linter."""
        print(f"Validator._run_ruff: Running mock Ruff on content for {file_path.name} (len: {len(modified_content)}).")
        if "# FIXME_RUFF" in modified_content:
            return f"Ruff error: Found FIXME_RUFF in {file_path.name}"
        return None

    def _run_pyright(self, modified_content: str, file_path: Path, project_root: Path) -> Optional[str]:
        """Mock for running Pyright type checker."""
        print(f"Validator._run_pyright: Running mock Pyright on content for {file_path.name} (len: {len(modified_content)}) in project {project_root}.")
        # In a real scenario, Pyright would need the content written to a temporary file
        # or use its language server capabilities.
        if "# FIXME_PYRIGHT" in modified_content:
            return f"Pyright error: Found FIXME_PYRIGHT in {file_path.name}"
        return None

    def _run_black_diff(self, modified_content: str, file_path: Path) -> Optional[str]:
        """Mock for running Black formatter check (diff)."""
        print(f"Validator._run_black_diff: Running mock Black-diff on content for {file_path.name} (len: {len(modified_content)}).")
        # A simple check: does it end with a newline if it has content?
        if modified_content and not modified_content.endswith("\n"):
            return f"Black-diff: Would add trailing newline to {file_path.name}"
        if "  unformatted_code = True" in modified_content: # Look for some obviously unformatted code
             return f"Black-diff: Code requires reformatting in {file_path.name}"
        return None

    def _run_pytest(self, target_file_path: Path, project_root: Path, digester: 'RepositoryDigester') -> Optional[str]:
        """
        Mock for running Pytest.
        This mock assumes that if the target_file_path is a test file itself,
        we check its content for a fail marker. Otherwise, it might try to find
        related tests (which is too complex for this mock).
        """
        print(f"Validator._run_pytest: Running mock Pytest for target {target_file_path.name} within project {project_root}.")
        # This mock is very simplified. A real version would:
        # 1. Identify relevant tests to run based on the target_file_path and project structure (using digester?).
        # 2. Execute pytest in a subprocess.
        # 3. Parse pytest output for failures.

        # For now, if the modified file is a test file, check its content.
        if target_file_path.name.startswith("test_") or target_file_path.name.endswith("_test.py"):
            # We need the content of the test file. The `validate_patch` method has the modified content
            # if target_file_path is the file being patched. If it's a *different* test file,
            # this mock would need to fetch its current (potentially modified by a previous phase) content.
            # This detail is skipped for this mock's simplicity.
            # Let's assume for now this method is called with the content of the test file itself if it's the target.
            # The current signature doesn't pass content directly, which is a limitation.

            # To make this mock testable with current signature, we'd need to get content via digester
            # if target_file_path is the one being validated.
            # However, validate_patch calls this with the *target_file_path* of the patch,
            # not necessarily the test file path.
            # This mock will assume if a *production* file is changed, some generic test might fail if a magic string exists.
            # This is not realistic but makes the mock function.

            # A better mock: if the *patched file* (whose content led to this pytest call)
            # contains a specific marker, assume a related test fails.
            # This requires passing modified_content to _run_pytest, or making it part of the class state,
            # which is not ideal. Let's stick to the current signature and simplify the mock's trigger.

            # Simplification: We don't have the *modified_content* of the *test file* here easily
            # unless target_file_path *is* the test file.
            # For a generic mock, let's assume if the target_file_path (the file being patched)
            # implies a feature that has a test, that test could fail.
            if "FEATURE_WITH_FAILING_TEST" in digester.get_file_content(target_file_path) if digester and target_file_path else "":
                 return f"Pytest error: Mock test failed for changes in {target_file_path.name} (due to FEATURE_WITH_FAILING_TEST marker)"

        # Or, a more direct mock if the target file *is* a test file (less common for patches unless tests themselves are patched)
        if target_file_path.name.startswith("test_") and "# TEST_FAILS" in (digester.get_file_content(target_file_path) or ""):
             return f"Pytest error: Mock test directly failed in {target_file_path.name}"
        return None

    def validate_patch(
        self,
        patch_script_str: Optional[str],
        target_file_path_str: str,
        digester: 'RepositoryDigester',
        project_root: Path
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates a patch script by applying it to the original content and running mock tools.
        Args:
            patch_script_str: The LibCST script string representing the patch.
            target_file_path_str: String path of the file to be patched.
            digester: RepositoryDigester instance to get original file content.
            project_root: Path to the project root for context (e.g., for Pyright).
        Returns:
            A tuple: (is_valid: bool, error_message: Optional[str])
        """
        print(f"\nValidator.validate_patch: Validating script for {target_file_path_str}.")
        if patch_script_str is None:
            print("  Validation failed: Patch script is None.")
            return False, "Patch script is None."

        target_file_path = project_root / target_file_path_str # Assume target_file_path_str is relative to project_root
        if not target_file_path.is_absolute():
             target_file_path = target_file_path.resolve()


        original_content = digester.get_file_content(target_file_path)
        if original_content is None:
            # If the file doesn't exist, it might be a new file patch.
            # For this mock, let's assume if original content is None, it's a new file.
            # And the patch_script_str should then be treated as the full new content.
            print(f"  Original content for {target_file_path} not found. Assuming new file scenario.")
            original_content = "" # Treat as new file

        # In a real scenario, _apply_patch_script would execute the LibCST script.
        # The current mock _apply_patch_script just appends the script content as a comment.
        # For validation tools to be meaningful, they need to run on the *result* of the script.
        # Let's adjust the thinking for mock: if patch_script_str is a full file content (dev-provided),
        # then _apply_patch_script should just return that.
        # If patch_script_str is a LibCST script (agent-generated), then _apply_patch_script
        # should ideally execute it.
        # FOR THIS MOCK: We'll assume patch_script_str from agent is a full file content for simplicity of validation.
        # This contradicts the agent's output of a "LibCST script string".
        # This is a known tension in the current mock design.
        # Let's proceed with the idea that validator needs the *intended final content*.
        # The _apply_patch_script mock will be updated to reflect this more directly for now.

        # If the patch_script_str is a LibCST script, this step is too simple.
        # A real version would run the script using LibCST against original_content.
        # For now, let's assume the "patch_script_str" IS the modified content FOR VALIDATION PURPOSES.
        # This is a temporary simplification to make the mock validator work on some content.
        # This means CollaborativeAgentGroup currently passes a LibCST script string here.
        # Validator's _apply_patch_script mock needs to be smart or this needs adjustment.

        # Let's refine _apply_patch_script's mock behavior slightly:
        # If patch_script_str seems like a full Python module (e.g. starts with "import" or "def" or "class")
        # then treat IT as the modified content. Otherwise, use the previous append behavior.
        # This is still a heuristic.

        # For the purpose of this subtask, we assume the 'patch_script_str' IS the new content.
        # This is a simplification that will need to be addressed when LibCST execution is implemented.
        # modified_content = self._apply_patch_script(original_content, patch_script_str)

        # REVISED MOCK APPROACH for validate_patch:
        # The `patch_script_str` is a LibCST Python *script*.
        # The `_apply_patch_script` should *simulate* running this script.
        # The tools then run on this *simulated output*.
        modified_content = self._apply_patch_script(original_content, patch_script_str)


        all_errors: List[str] = []

        # Run Tools (Placeholders)
        ruff_errors = self._run_ruff(modified_content, target_file_path)
        if ruff_errors: all_errors.append(ruff_errors)

        pyright_errors = self._run_pyright(modified_content, target_file_path, project_root)
        if pyright_errors: all_errors.append(pyright_errors)

        black_diff_errors = self._run_black_diff(modified_content, target_file_path)
        if black_diff_errors: all_errors.append(black_diff_errors)

        # Pytest is more complex: it runs on the project state *after* the patch is (conceptually) applied.
        # The modified_content is of the single target file. Pytest might test interactions.
        # The mock _run_pytest takes target_file_path and digester to potentially access other files or project info.
        pytest_errors = self._run_pytest(target_file_path, project_root, digester)
        if pytest_errors: all_errors.append(pytest_errors)

        if not all_errors:
            print(f"  Validation successful for {target_file_path_str}.")
            return True, None
        else:
            error_summary = "\n".join(all_errors)
            print(f"  Validation failed for {target_file_path_str}:\n{error_summary}")
            return False, error_summary

if __name__ == '__main__':
    print("--- Validator Example Usage (Conceptual) ---")

    # Mock Digester for Validator testing
    class MockDigesterForValidator:
        def __init__(self, files: Dict[str, str]):
            self.files_content = {Path(p).resolve(): c for p, c in files.items()}
            print(f"MockDigesterForValidator initialized with files: {list(self.files_content.keys())}")

        def get_file_content(self, file_path: Path) -> Optional[str]:
            resolved_path = file_path.resolve()
            print(f"MockDigesterForValidator.get_file_content for: {resolved_path}")
            return self.files_content.get(resolved_path)

        def get_code_snippets_for_phase(self, phase_ctx: Any) -> Dict[str, str]: return {} # Mock
        def get_pdg_slice_for_phase(self, phase_ctx: Any) -> Dict[str, Any]: return {} # Mock


    # Setup
    mock_project_root = Path("_temp_mock_validator_project").resolve()
    mock_project_root.mkdir(exist_ok=True)

    mock_file_path_str_relative = "module/example.py"
    mock_file_path_abs = mock_project_root / mock_file_path_str_relative
    mock_file_path_abs.parent.mkdir(parents=True, exist_ok=True)

    original_code = """
def hello_world():
    print("Hello, world!")
# Add # FIXME_RUFF for testing ruff
# Add # FIXME_PYRIGHT for testing pyright
# Add   unformatted_code = True to test black
"""
    mock_file_path_abs.write_text(original_code)

    mock_digester = MockDigesterForValidator({str(mock_file_path_abs): original_code})

    validator_instance = Validator()

    # Test case 1: Clean patch (script doesn't introduce new linting/typing errors beyond the mock _apply_patch behavior)
    print("\n--- Test Case 1: Clean Patch Script ---")
    clean_patch_script = "pass # No new issues introduced by this mock script"
    is_valid, errors = validator_instance.validate_patch(
        clean_patch_script,
        mock_file_path_str_relative, # Pass relative path as per typical phase_ctx
        mock_digester, # type: ignore
        mock_project_root
    )
    print(f"Test Case 1 - Valid: {is_valid}, Errors: {errors}")
    assert not is_valid # Because _apply_patch_script adds the script content, which can trigger black formatting
    assert "Black-diff" in errors if errors else False


    # Test case 2: Patch script that would cause Ruff error
    print("\n--- Test Case 2: Patch Script with Ruff FIXME ---")
    ruff_fixme_patch_script = "# FIXME_RUFF this is a problem"
    # The _apply_patch_script will append this.
    # To make it more direct for the mock, let's assume the script *is* the content.
    # This means our _apply_patch_script mock needs to be smarter or the test needs to adapt.
    # For now, the current _apply_patch_script appends the script.
    # So, the modified_content will be original_code + "\n\n# --- Mock Patch Applied... # FIXME_RUFF..."
    # This means original_code's content can also trigger errors.
    # Let's ensure original_code is clean first for a more isolated test of the patch script's effect.

    clean_original_code = "def main():\n    pass\n"
    mock_file_path_abs.write_text(clean_original_code)
    mock_digester_clean = MockDigesterForValidator({str(mock_file_path_abs): clean_original_code})

    is_valid_ruff, errors_ruff = validator_instance.validate_patch(
        ruff_fixme_patch_script, mock_file_path_str_relative, mock_digester_clean, mock_project_root # type: ignore
    )
    print(f"Test Case 2 - Valid: {is_valid_ruff}, Errors: {errors_ruff}")
    assert not is_valid_ruff
    assert "Ruff error" in errors_ruff if errors_ruff else False

    # Test case 3: Patch script that would cause Pyright error
    print("\n--- Test Case 3: Patch Script with Pyright FIXME ---")
    pyright_fixme_patch_script = "# FIXME_PYRIGHT problem here"
    is_valid_pyright, errors_pyright = validator_instance.validate_patch(
        pyright_fixme_patch_script, mock_file_path_str_relative, mock_digester_clean, mock_project_root # type: ignore
    )
    print(f"Test Case 3 - Valid: {is_valid_pyright}, Errors: {errors_pyright}")
    assert not is_valid_pyright
    assert "Pyright error" in errors_pyright if errors_pyright else False

    # Test case 4: Patch script that would cause Black error (no trailing newline)
    # The _apply_patch_script currently adds newlines, so this specific Black error is hard to trigger
    # unless the original content itself is problematic or _apply_patch_script is made more sophisticated.
    # Let's test the "unformatted_code = True" part of the Black mock.
    print("\n--- Test Case 4: Patch Script with Black unformatted code ---")
    black_error_patch_script = "  unformatted_code = True"
    is_valid_black, errors_black = validator_instance.validate_patch(
        black_error_patch_script, mock_file_path_str_relative, mock_digester_clean, mock_project_root # type: ignore
    )
    print(f"Test Case 4 - Valid: {is_valid_black}, Errors: {errors_black}")
    assert not is_valid_black
    assert "Black-diff: Code requires reformatting" in errors_black if errors_black else False


    # Test case 5: Pytest error simulation
    print("\n--- Test Case 5: Pytest error ---")
    # To trigger the pytest mock, the target file itself needs to be a "test_" file
    # or contain "FEATURE_WITH_FAILING_TEST"
    test_file_rel_path = "test_example_feature.py"
    test_file_abs_path = mock_project_root / test_file_rel_path
    # Original content for the main feature file that implies a failing test
    feature_file_code_triggering_test_fail = "FEATURE_WITH_FAILING_TEST = True"
    mock_digester_pytest = MockDigesterForValidator({
        str(test_file_abs_path): "# Test file, content not directly checked by this mock path",
        str(mock_file_path_abs): feature_file_code_triggering_test_fail # This is the target file for the patch
    })
    # Patch script can be anything for this, as the error is triggered by content of mock_file_path_abs
    some_patch_for_feature = "x = 10 # some change to feature file"
    is_valid_pytest, errors_pytest = validator_instance.validate_patch(
        some_patch_for_feature, mock_file_path_str_relative, mock_digester_pytest, mock_project_root # type: ignore
    )
    print(f"Test Case 5 - Valid: {is_valid_pytest}, Errors: {errors_pytest}")
    # This assertion depends on how the _run_pytest mock is triggered.
    # If it's based on the content of target_file_path (mock_file_path_str_relative), it should find "FEATURE_WITH_FAILING_TEST"
    assert not is_valid_pytest
    assert "Pytest error: Mock test failed for changes in example.py" in errors_pytest if errors_pytest else False

    # Cleanup
    import shutil
    shutil.rmtree(mock_project_root)
    print(f"\nCleaned up mock project root: {mock_project_root}")
    print("--- Validator Example Usage Done ---")
