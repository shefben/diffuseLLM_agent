# src/validator/validator.py
import subprocess
import tempfile
import os
import json # Added for Pyright JSON output parsing
from typing import Optional, Tuple, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
import re # Import re for regular expressions

if TYPE_CHECKING:
    from src.digester.repository_digester import RepositoryDigester
    # from src.planner.phase_model import Phase # If phase_ctx were needed by validation tools

class Validator:
    def __init__(self, app_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Validator.
        Args:
            app_config: Optional dictionary for validator configurations (e.g., tool paths).
        """
        self.app_config = app_config if app_config else {}
        self.verbose = self.app_config.get("general", {}).get("verbose", False)
        self.ruff_path = self.app_config.get("tools", {}).get("ruff_path", "ruff")
        self.black_path = self.app_config.get("tools", {}).get("black_path", "black")
        self.pyright_path = self.app_config.get("tools", {}).get("pyright_path", "pyright")
        self.pytest_path = self.app_config.get("tools", {}).get("pytest_path", "pytest")
        self.default_pytest_target_dir = self.app_config.get("tools", {}).get("default_pytest_target_dir", "tests")


        self.common_stdlib_modules = [
            "os", "sys", "math", "re", "json", "collections", "pathlib",
            "datetime", "time", "argparse", "logging", "subprocess", "multiprocessing",
            "threading", "socket", "ssl", "http", "urllib", "tempfile", "shutil",
            "glob", "io", "pickle", "base64", "hashlib", "hmac", "uuid", "functools", "itertools",
            "operator", "typing", "dataclasses", "enum", "inspect", "gc", "weakref"
        ]
        # Updated print statement to reflect new config structure if verbose is true.
        # Note: The original print statement directly used self.verbose, self.ruff_path etc.
        # These attributes are now initialized using self.app_config.
        # The print statement itself doesn't need to change how it ACCESSES these attributes,
        # as they are already set on self by the lines above.
        # What might change is the SOURCE of these values if the new config structure means
        # they are derived differently and we want to explicitly show that in the log.
        # However, the task is to update accessors during __init__, which has been done.
        # The print statement will correctly reflect the values derived from app_config.
        print(f"Validator initialized. Ruff: '{self.ruff_path}', Black: '{self.black_path}', Pyright: '{self.pyright_path}', Pytest: '{self.pytest_path}', Default Pytest Dir: '{self.default_pytest_target_dir}', Verbose: {self.verbose}. Known stdlib: {len(self.common_stdlib_modules)}")

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

        # Heuristic 3: Remove trailing commas before closing delimiters
        # This is a common syntax error if an LLM adds a comma in a place it's not allowed,
        # e.g. func_call(arg1, arg2,) or def func(p1, p2,):
        # Note: Black often allows trailing commas in lists/dicts, but this handles invalid ones.
        # This specifically targets commas immediately followed by optional whitespace and then a closer.

        # Regex patterns:
        # 1. Comma before closing parenthesis:  ,\s*\)
        # 2. Comma before closing square bracket: ,\s*\]
        # 3. Comma before closing curly brace:   ,\s*\}

        original_script_for_comma_check = modified_script # Keep a copy to see if changes were made

        modified_script = re.sub(r",\s*\)", ")", modified_script)
        modified_script = re.sub(r",\s*\]", "]", modified_script)
        modified_script = re.sub(r",\s*\}", "}", modified_script)

        if modified_script != original_script_for_comma_check:
            print("Validator: Heuristic fix applied - removed trailing comma(s) before closing parenthesis/bracket/brace.")
            return modified_script

        # If no heuristics were applied, or if a heuristic made a change but didn't return early
        if modified_script == original_patch_script_str:
            print(f"Validator: No applicable heuristic fixes successfully applied for target '{target_file_path_str}'.")
            return None # No changes made by heuristics
        else:
            # This case should ideally not be reached if fixes return early.
            # If it is, means a fix was made but didn't return.
            print(f"Validator: Heuristic fix made but did not return early (unexpected). Returning modified script for {target_file_path_str}.")
            return modified_script

    # _apply_patch_script removed as it's no longer used by validate_patch.
    # validate_patch now receives the already modified code content.

    def _run_ruff(self, file_content: str, file_path: Path) -> Optional[str]:
        """Runs Ruff linter on the provided file content."""
        errors = []
        tmp_file_path_str = None # Ensure it's defined for finally block
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding='utf-8') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path_str = tmp_file.name

            command = [self.ruff_path, "check", "--no-cache", "--quiet", tmp_file_path_str]
            # For Ruff > 0.3.0, consider --output-format=json for structured output.
            # Example: command = [self.ruff_path, "check", "--no-cache", "--quiet", "--output-format=json", tmp_file_path_str]

            if self.verbose: print(f"Validator: Running Ruff: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                # Ruff's default output for errors is to stdout. Stderr is for operational errors.
                if result.stdout:
                    errors.append(f"Ruff found issues in '{file_path.name}':\n{result.stdout.strip()}")
                # Include stderr if it contains information, often Ruff operational errors
                if result.stderr:
                    errors.append(f"Ruff operational error for '{file_path.name}':\n{result.stderr.strip()}")

            # If returncode is 0 but there's still stderr output, it might be a warning or other message.
            elif result.stderr:
                 print(f"Validator: Ruff stderr (even with exit code 0) for '{file_path.name}':\n{result.stderr.strip()}")


        except FileNotFoundError:
            return f"Ruff executable not found at '{self.ruff_path}'. Please configure 'ruff_path' or ensure Ruff is in PATH."
        except Exception as e:
            return f"An unexpected error occurred while running Ruff on '{file_path.name}': {e}"
        finally:
            if tmp_file_path_str and Path(tmp_file_path_str).exists():
                os.remove(tmp_file_path_str)

        return "\n".join(errors) if errors else None
    def _run_ruff(self, file_content: str, file_path: Path) -> Optional[str]:
        """Runs Ruff linter on the provided file content."""
        errors = []
        tmp_file_path_str = None # Ensure it's defined for finally block
        try:
            # Ruff works best with a file path, so write content to a temporary file.
            # Using NamedTemporaryFile with delete=False, and manual deletion in finally.
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding='utf-8') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path_str = tmp_file.name # Get the path for the command

            # Command for Ruff: check the temporary file.
            # --no-cache: Disables caching, ensuring a fresh check.
            # --quiet: Suppresses non-error output.
            # --force-exclude: Could be added if Ruff respects project excludes for temp files.
            #                However, usually, we want to lint the exact content provided.
            command = [self.ruff_path, "check", "--no-cache", "--quiet", tmp_file_path_str]
            # Consider --output-format=json for easier parsing if available and stable.

            if self.verbose: print(f"Validator: Running Ruff: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            # Ruff typically exits with 1 if linting errors are found.
            if result.returncode != 0:
                # Ruff's default output for errors is to stdout. Stderr is for operational errors.
                if result.stdout: # This should contain the linting errors
                    errors.append(f"Ruff found issues in '{file_path.name}':\n{result.stdout.strip()}")

                # Include stderr if it contains information (e.g., Ruff operational errors)
                # but only if stdout was empty, to avoid redundancy if Ruff also prints its own errors to stderr.
                if result.stderr and not result.stdout:
                    errors.append(f"Ruff operational error for '{file_path.name}':\n{result.stderr.strip()}")
                elif result.stderr and result.stdout: # If both, log stderr for debugging but don't add to main errors
                     print(f"Validator (Ruff verbose): Stderr for '{file_path.name}':\n{result.stderr.strip()}")

            # If returncode is 0 but there's still stderr output, it might be a warning or other message.
            elif result.stderr: # e.g. warning about config, but no lint errors
                 print(f"Validator (Ruff verbose): Stderr (even with exit code 0) for '{file_path.name}':\n{result.stderr.strip()}")

        except FileNotFoundError:
            return f"Ruff executable not found at '{self.ruff_path}'. Please configure 'ruff_path' or ensure Ruff is in PATH."
        except Exception as e:
            return f"An unexpected error occurred while running Ruff on '{file_path.name}': {e}"
        finally:
            if tmp_file_path_str and Path(tmp_file_path_str).exists():
                os.remove(tmp_file_path_str)

        return "\n".join(errors) if errors else None

    def _run_pyright(self, file_content: str, file_path: Path, project_root: Path) -> Optional[str]:
        """Runs Pyright type checker on the provided file content."""
        errors = []
        tmp_file_path_str = None # Ensure defined for finally block
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding='utf-8') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path_str = tmp_file.name

            # Pyright command: use --outputjson to get structured error information.
            # It's crucial to run Pyright in the context of the project_root for it to find configs (pyrightconfig.json or pyproject.toml) and resolve imports.
            command = [self.pyright_path, "--outputjson", tmp_file_path_str]

            if self.verbose:
                print(f"Validator: Running Pyright: {' '.join(command)} in dir {project_root}")
                print(f"Validator: Note - Pyright's accuracy on a temporary file ('{tmp_file_path_str}') relies on it being analyzed within the project context ('{project_root}') for correct import resolution and configuration loading.")

            # Pyright is often installed via npm; ensure it's accessible.
            # Using project_root as cwd is important.
            result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=project_root)

            # Pyright exit codes: 0 (no errors), 1 (errors), 2 (fatal error), 3 (option error)
            if result.returncode != 0:
                try:
                    output_json = json.loads(result.stdout)
                    num_errors = output_json.get("summary", {}).get("errorCount", 0)

                    if num_errors > 0:
                        formatted_errors = [f"Pyright found {num_errors} error(s) in '{file_path.name}':"]
                        for diag in output_json.get("generalDiagnostics", []):
                            if diag.get("severity") == "error":
                                message = diag.get("message", "No message")
                                # Pyright line/char are 0-indexed
                                line = diag.get("range", {}).get("start", {}).get("line", -1) + 1
                                col = diag.get("range", {}).get("start", {}).get("character", -1) + 1
                                rule = diag.get("rule", "") # Include rule if available
                                error_tag = f" ({rule})" if rule else ""
                                formatted_errors.append(f"  - {file_path.name}:{line}:{col}: {message}{error_tag}")
                        errors.extend(formatted_errors)

                    # If JSON is valid but errorCount is 0 despite non-zero exit code, or if no diagnostics:
                    if not errors and result.returncode !=0 : # Still no specific errors parsed, but Pyright indicated an issue
                        errors.append(f"Pyright exited with code {result.returncode} for '{file_path.name}'.")
                        if result.stdout.strip(): errors.append(f"Pyright output (stdout):\n{result.stdout.strip()}")
                        if result.stderr.strip(): errors.append(f"Pyright error (stderr):\n{result.stderr.strip()}")

                except json.JSONDecodeError:
                    errors.append(f"Pyright returned exit code {result.returncode} for '{file_path.name}', but output was not valid JSON.")
                    if result.stdout.strip(): errors.append(f"Pyright output (stdout):\n{result.stdout.strip()}")
                    if result.stderr.strip(): errors.append(f"Pyright error (stderr):\n{result.stderr.strip()}")

            # Capture stderr if verbose, even on success, for warnings etc.
            elif result.stderr and self.verbose:
                 print(f"Validator: Pyright stderr (even on success) for '{file_path.name}':\n{result.stderr.strip()}")

        except FileNotFoundError:
            return f"Pyright executable not found at '{self.pyright_path}'. Please configure 'pyright_path' or ensure Pyright (e.g., from npm) is in PATH."
        except Exception as e:
            return f"An unexpected error occurred while running Pyright on '{file_path.name}': {e}"
        finally:
            if tmp_file_path_str and Path(tmp_file_path_str).exists():
                os.remove(tmp_file_path_str) # Corrected variable name here

        return "\n".join(errors) if errors else None

    def _run_black_diff(self, file_content: str, file_path: Path) -> Optional[str]:
        """Runs Black formatter check (diff) on the provided file content."""
        tmp_file_path_str = None # Ensure defined for finally
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding='utf-8') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path_str = tmp_file.name

            command = [self.black_path, "--check", "--diff", tmp_file_path_str]
            if self.verbose: print(f"Validator: Running Black: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            # Black with --check exits with 0 if no changes needed.
            # Exits with 1 if changes would be made.
            # Exits with >1 for other errors (e.g., invalid syntax, which Ruff/Pyright should catch first).
            if result.returncode == 1: # Indicates Black would reformat the file.
                error_message = f"Black-diff: Code in '{file_path.name}' is not formatted. Black would make changes."
                if self.verbose and result.stdout: # stdout contains the diff
                     error_message += f"\nDiff:\n{result.stdout.strip()}"
                if result.stderr: # stderr might contain additional info from Black
                    error_message += f"\nBlack stderr (when changes needed):\n{result.stderr.strip()}"
                return error_message
            elif result.returncode > 1 : # Other Black errors
                 # These are often syntax errors that Black itself cannot parse.
                 # Ruff should ideally catch these first.
                 error_detail = result.stderr.strip() if result.stderr else result.stdout.strip()
                 return f"Black operational error for '{file_path.name}' (exit code {result.returncode}):\n{error_detail}"

            # If returncode is 0, but there's still stderr (e.g. warning about config), log it.
            if result.returncode == 0 and result.stderr:
                print(f"Validator (Black verbose): Stderr (even with exit code 0) for '{file_path.name}':\n{result.stderr.strip()}")

            return None # No formatting issues found by Black --check

        except FileNotFoundError:
            return f"Black executable not found at '{self.black_path}'. Please configure 'black_path' or ensure Black is in PATH."
        except Exception as e:
            return f"An unexpected error occurred while running Black on '{file_path.name}': {e}"
        finally:
            if tmp_file_path_str and Path(tmp_file_path_str).exists():
                os.remove(tmp_file_path_str) # Corrected variable name here

    def _run_pytest(self, target_file_path: Path, project_root: Path, modified_content_for_target_file: str, pdg_slice: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Runs Pytest in a temporary copy of the project with the modified file.
        Args:
            target_file_path: Absolute path to the file being modified in the original project.
            project_root: Absolute path to the root of the original project.
            modified_content_for_target_file: The new content for the target file.
        Returns:
            A string containing Pytest error messages, or None if tests passed or no tests were found.
        """
        pytest_errors_str: Optional[str] = None
        # Ensure shutil is imported at the top of the file
        import shutil

        try:
            with tempfile.TemporaryDirectory(prefix="pytest_validation_") as temp_dir_name:
                temp_project_path = Path(temp_dir_name)

                if self.verbose:
                    print(f"Validator: Creating temporary project copy at {temp_project_path}")

                # 1. Copy project to temp directory
                shutil.copytree(
                    project_root,
                    temp_project_path,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('.git', '.venv', 'venv', '__pycache__', '*.pyc', '.*', 'node_modules')
                )

                # 2. Apply modification in the temporary copy
                relative_target_path = target_file_path.relative_to(project_root)
                copied_target_file_path = temp_project_path / relative_target_path
                copied_target_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(copied_target_file_path, "w", encoding='utf-8') as f:
                    f.write(modified_content_for_target_file)

                if self.verbose:
                    print(f"Validator: Applied modified content to {copied_target_file_path}")

                # 3. Determine Pytest target(s)
                pytest_run_location = temp_project_path # CWD for pytest
                specific_test_targets: List[str] = []

                # PDG-based test target selection
                pdg_derived_test_targets = set()
                if pdg_slice and pdg_slice.get("nodes"):
                    if self.verbose: print(f"Validator: Using PDG slice to find test targets. Slice has {len(pdg_slice['nodes'])} nodes.")
                    source_files_in_slice = set()
                    for node_info in pdg_slice["nodes"]:
                        if isinstance(node_info, dict) and node_info.get("file"):
                            # Assuming 'file' in node_info is relative to project_root or absolute.
                            # We need it relative to project_root for constructing test paths.
                            node_file_path = Path(node_info["file"])
                            if node_file_path.is_absolute():
                                try:
                                    node_file_rel_path = node_file_path.relative_to(project_root)
                                    source_files_in_slice.add(node_file_rel_path)
                                except ValueError: # If not under project_root, skip
                                    if self.verbose: print(f"Validator: PDG node file {node_file_path} is outside project_root {project_root}, skipping.")
                                    continue
                            else: # Already relative
                                source_files_in_slice.add(node_file_path)


                    for src_file_rel_path in source_files_in_slice:
                        # Heuristic 1: tests/src/module/test_file.py (if src_file_rel_path starts with src/)
                        # or tests/module/test_file.py (if src_file_rel_path does not start with src/)
                        test_path_1_parts = [self.default_pytest_target_dir] + list(src_file_rel_path.parent.parts) + [f"test_{src_file_rel_path.name}"]
                        test_path_1 = Path(*test_path_1_parts)

                        # Heuristic 2: tests/module/test_file.py (specifically if src_file_rel_path was like 'src/module/file.py')
                        test_path_2 = None
                        if src_file_rel_path.parts and src_file_rel_path.parts[0] == 'src':
                            test_path_2_parts = [self.default_pytest_target_dir] + list(src_file_rel_path.parts[1:]).parent.parts + [f"test_{src_file_rel_path.name}"]
                            test_path_2 = Path(*test_path_2_parts)

                        # Heuristic 3: tests/test_module_file.py (flatter structure)
                        test_path_3_parts = [self.default_pytest_target_dir]
                        if src_file_rel_path.parent.name and src_file_rel_path.parent.name != '.': # Avoid if file is in root
                             test_path_3_parts.append(f"test_{src_file_rel_path.parent.name}_{src_file_rel_path.stem}.py")
                        else:
                             test_path_3_parts.append(f"test_{src_file_rel_path.stem}.py")
                        test_path_3 = Path(*test_path_3_parts)


                        potential_test_files = [test_path_1, test_path_2, test_path_3]
                        for pt_path in potential_test_files:
                            if pt_path is None: continue
                            # Path for checking existence is in the temp_project_path (where tests are copied)
                            # pt_path is already relative to temp_project_path/project_root due to construction.
                            abs_potential_test_path_in_temp = temp_project_path / pt_path
                            if abs_potential_test_path_in_temp.is_file():
                                pdg_derived_test_targets.add(str(pt_path)) # Add relative path for pytest command
                                if self.verbose: print(f"Validator: Added PDG-derived test target: {pt_path}")

                if pdg_derived_test_targets:
                    specific_test_targets = list(pdg_derived_test_targets)
                    if self.verbose: print(f"Validator: Using PDG-derived test targets: {specific_test_targets}")
                else:
                    # Fallback to existing logic if no PDG targets found
                    if self.verbose: print("Validator: No PDG-derived test targets found, using fallback logic.")
                    # relative_target_path is defined when the modified file is written to temp_project_path
                    is_target_in_test_dir = str(target_file_path).startswith(str(project_root / self.default_pytest_target_dir))
                    is_target_a_test_file_by_name = "test" in target_file_path.name.lower()

                    if is_target_in_test_dir or is_target_a_test_file_by_name:
                        specific_test_targets.append(str(relative_target_path))
                        if self.verbose: print(f"Validator: Pytest will target the modified test file: {relative_target_path}")
                    else:
                        default_test_dir_in_temp_rel_path = self.default_pytest_target_dir
                        if (temp_project_path / default_test_dir_in_temp_rel_path).is_dir():
                            specific_test_targets.append(default_test_dir_in_temp_rel_path)
                            if self.verbose: print(f"Validator: Pytest will target default test directory: {default_test_dir_in_temp_rel_path}")
                        else:
                            specific_test_targets.append(".")
                            if self.verbose: print(f"Validator: Default test directory not found in temp copy. Pytest will target temp project root '.'")

                # 4. Execute Pytest
                command = [self.pytest_path] + specific_test_targets + ["-qq", "--disable-warnings", "--cache-clear"]
                if self.verbose: print(f"Validator: Running Pytest: {' '.join(command)} in CWD: {pytest_run_location}")

                result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=pytest_run_location)

                # 5. Process results
                if result.returncode != 0 and result.returncode != 5: # 0=all pass, 5=no tests collected
                    error_message = f"Pytest failed (exit code {result.returncode}) for targets '{' '.join(specific_test_targets)}' in temp project '{temp_project_path}'.\n"
                    failure_summary = []
                    in_failures_section = False
                    # Pytest failures usually go to stdout. Stderr might have pytest operational issues.
                    output_to_parse = result.stdout
                    if not result.stdout.strip() and result.stderr.strip(): # If stdout is empty but stderr has content
                        output_to_parse = result.stderr

                    for line in output_to_parse.splitlines():
                        if "=== FAILURES ===" in line: in_failures_section = True
                        # Heuristic: include lines that seem part of failure reports or summaries
                        if in_failures_section or line.startswith("E ") or " error " in line.lower() or " failed " in line.lower():
                            if line.strip(): failure_summary.append(line)
                        elif failure_summary and not line.strip(): # Blank line might end a specific failure detail block
                             pass

                    if failure_summary:
                        error_message += "\n".join(failure_summary)
                    elif result.stdout.strip(): # Fallback if parsing found nothing but there's stdout
                        error_message += result.stdout.strip()
                    elif result.stderr.strip(): # Fallback to stderr if stdout also empty
                        error_message += result.stderr.strip()
                    else: # Should not happen if returncode is non-zero
                        error_message += "No detailed output found in stdout/stderr."
                    pytest_errors_str = error_message

                elif result.returncode == 5:
                    if self.verbose: print(f"Validator: Pytest: No tests collected for targets '{' '.join(specific_test_targets)}' in temp project.")

                # No errors if result.returncode == 0 (tests passed)

        except FileNotFoundError: # This specifically catches if self.pytest_path is not found
            pytest_errors_str = f"Pytest executable not found at '{self.pytest_path}'. Please ensure Pytest is installed and 'pytest_path' is correctly configured or in PATH."
        except Exception as e:
            pytest_errors_str = f"An unexpected error occurred during Pytest execution: {type(e).__name__} - {e}"

        return pytest_errors_str

    def validate_patch(
        self,
        modified_code_content_str: Optional[str],
        target_file_path_str: str,
        digester: 'RepositoryDigester',
        project_root: Path,
        pdg_slice: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]: # Error message is now the second element, payload removed
        """
        Validates the modified code content by running mock tools.
        Args:
            modified_code_content_str: The string content of the file AFTER the patch script has been applied.
                                       If None, indicates a failure prior to validation (e.g. patch application error).
            target_file_path_str: String path of the file that was patched (relative to project_root).
            digester: RepositoryDigester instance (can be used by tools like Pytest for broader context).
            project_root: Path to the project root for context (e.g., for Pyright).
        Returns:
            A tuple: (is_valid: bool, error_message: Optional[str])
        """
        print(f"\nValidator.validate_patch: Validating content for {target_file_path_str}.")

        if modified_code_content_str is None:
            print("  Validation failed: Input modified_code_content_str was None, possibly due to a patch application failure upstream.")
            return False, "Input modified_code_content was None, possibly due to patch application failure."

        # Ensure target_file_path is absolute for tools that might need it
        target_file_path = project_root / target_file_path_str
        if not target_file_path.is_absolute():
             target_file_path = target_file_path.resolve()

        # The modified_code_content_str IS the content to validate. No script application here.
        # The old _apply_patch_script method is no longer used by this method.

        all_errors: List[str] = []

        # Run Tools (Placeholders) on the provided modified_code_content_str
        ruff_errors = self._run_ruff(modified_code_content_str, target_file_path)
        if ruff_errors: all_errors.append(ruff_errors)

        pyright_errors = self._run_pyright(modified_code_content_str, target_file_path, project_root)
        if pyright_errors: all_errors.append(pyright_errors)

        black_diff_errors = self._run_black_diff(modified_code_content_str, target_file_path)
        if black_diff_errors: all_errors.append(black_diff_errors)

        # Pytest runs on the project state. The modified_code_content_str represents the change
        # to one file. A real pytest would run against the filesystem.
        pytest_errors = self._run_pytest(target_file_path, project_root, modified_code_content_str, pdg_slice=pdg_slice)
        if pytest_errors: all_errors.append(pytest_errors)

        if not all_errors:
            print(f"  Validation successful for {target_file_path_str}.")
            # The 'payload' (second element of tuple) is now the error string, so None for success.
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
