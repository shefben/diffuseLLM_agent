import subprocess
from pathlib import Path
from typing import Dict, Any, Union # Added Union for profile type hint
import shutil # For finding executables

# Assuming these functions are available to be called to ensure configs are fresh
# These would typically be imported if this were part of a larger application structure.
# For now, we are defining format_code. How it gets these functions or ensures
# configs are up-to-date is part of a larger integration.
# from src.profiler.config_generator import generate_black_config_in_pyproject, generate_ruff_config_in_pyproject
# from src.profiler.profile_io import load_style_profile # If profile needs to be loaded by name

# Placeholder for StyleProfile type if we define a dataclass for it formally
StyleProfileType = Dict[str, Any]

def format_code(
    file_path: Path,
    profile: StyleProfileType,
    pyproject_path: Path = Path("pyproject.toml") # Assume pyproject.toml is in current dir or specified
) -> bool:
    """
    Formats a given Python file using Black and Ruff, based on a style profile.
    It also includes a placeholder for future identifier renaming.

    Args:
        file_path: The Path to the Python file to format.
        profile: A dictionary representing the unified style profile.
        pyproject_path: Path to the pyproject.toml where Black/Ruff configs are managed.

    Returns:
        True if all formatting steps were successful (or no changes needed),
        False if any step failed.
    """
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        return False

    # --- Step 1: Ensure Black and Ruff configurations reflect the profile ---
    # In a real application, this might be handled by ensuring pyproject.toml is
    # generated/updated *before* calling format_code for a batch of files.
    # For this helper, we could optionally call the generators here,
    # but it might be inefficient if formatting many files with the same profile.
    # For now, let's assume the pyproject.toml is already configured according to the profile.
    # If not, one would call:
    # generate_black_config_in_pyproject(profile, pyproject_path)
    # generate_ruff_config_in_pyproject(profile, pyproject_path)
    # print(f"Ensured Black/Ruff configurations in {pyproject_path} are based on the profile (conceptual).")

    # --- Step 2: Run Black to format the file ---
    # Black will use the configuration from pyproject.toml if present.
    black_executable = shutil.which("black")
    if not black_executable:
        print("Error: Black executable not found in PATH.")
        return False

    try:
        print(f"Running Black on {file_path}...")
        # Black modifies the file in place by default.
        # Exit codes: 0 if no changes or successful reformatting.
        #             1 if internal error.
        #             123 if syntax error in input.
        black_process = subprocess.run(
            [black_executable, str(file_path)],
            capture_output=True,
            text=True,
            check=False # Don't raise for non-zero exit code immediately
        )
        if black_process.returncode == 0:
            print(f"Black formatting successful for {file_path}.")
        elif black_process.returncode == 123:
            print(f"Black error: Syntax error in {file_path}.")
            print(black_process.stderr)
            return False # Syntax error is a failure for formatting.
        elif black_process.returncode != 0 : # Other Black errors
            print(f"Black failed for {file_path} with exit code {black_process.returncode}.")
            print("Black stderr:")
            print(black_process.stderr)
            print("Black stdout:")
            print(black_process.stdout)
            return False

    except Exception as e:
        print(f"An unexpected error occurred while running Black on {file_path}: {e}")
        return False

    # --- Step 3: Run Ruff to format the file and fix lint errors ---
    # Ruff can also use pyproject.toml. `ruff format` and `ruff check --fix`.
    # We'll run both, format first, then lint fixing.
    ruff_executable = shutil.which("ruff")
    if not ruff_executable:
        print("Error: Ruff executable not found in PATH.")
        return False

    try:
        print(f"Running Ruff Formatter on {file_path}...")
        # Ruff format modifies the file in place.
        ruff_format_process = subprocess.run(
            [ruff_executable, "format", str(file_path)],
            capture_output=True,
            text=True,
            check=False
        )
        if ruff_format_process.returncode == 0:
            print(f"Ruff formatting successful for {file_path}.")
        elif ruff_format_process.returncode != 0:
            # Ruff format might return non-zero for errors like unparseable files.
            print(f"Ruff format failed for {file_path} with exit code {ruff_format_process.returncode}.")
            print("Ruff format stderr:")
            print(ruff_format_process.stderr)
            print("Ruff format stdout:")
            print(ruff_format_process.stdout)
            # Continue to lint fixing, as some lint issues might still be fixable.
            # However, if formatting failed due to syntax, fixing might also fail.

        print(f"Running Ruff Lint (--fix) on {file_path}...")
        # Ruff check --fix modifies the file in place.
        # Exit codes: 0 if no errors or all fixable errors fixed.
        #             1 if unfixable errors remain.
        #             2 if Ruff itself encounters an error.
        ruff_lint_process = subprocess.run(
            [ruff_executable, "check", "--fix", "--exit-zero-even-if-changed", str(file_path)],
            capture_output=True,
            text=True,
            check=False
        )
        # --exit-zero-even-if-changed makes it exit 0 if it made fixes, otherwise 1 if errors remain.
        # We care if errors *remain* after fixing.
        if ruff_lint_process.returncode == 0: # No errors remained, or all were fixed.
            print(f"Ruff lint (--fix) successful for {file_path}.")
        elif ruff_lint_process.returncode == 1: # Unfixable errors remain
            print(f"Ruff lint found unfixable issues in {file_path} (after attempting fixes).")
            print("Ruff lint stdout (includes remaining issues):")
            print(ruff_lint_process.stdout)
            # Depending on strictness, this could be a failure. For now, let's say it is.
            return False
        elif ruff_lint_process.returncode != 0: # Ruff internal error
            print(f"Ruff lint (--fix) failed for {file_path} with exit code {ruff_lint_process.returncode}.")
            print("Ruff lint stderr:")
            print(ruff_lint_process.stderr)
            return False

    except Exception as e:
        print(f"An unexpected error occurred while running Ruff on {file_path}: {e}")
        return False

    # --- Step 4: Apply an identifier-renaming pass (LibCST) ---
    # This is a placeholder for a future, complex implementation.
    # It would involve:
    # 1. Loading the naming_conventions.db (or relevant rules from profile).
    # 2. Parsing the file_path with LibCST.
    # 3. Identifying out-of-profile symbols.
    # 4. Generating new names that conform to the rules.
    # 5. Applying these changes using a LibCST Concrete Syntax Tree (CST) visitor/transformer.
    # 6. Writing the modified CST back to the file.
    print(f"Placeholder: Identifier renaming pass (LibCST) for {file_path} would occur here.")
    # For now, assume this step is successful or does nothing.

    print(f"Formatting pipeline completed for {file_path}.")
    return True


if __name__ == '__main__':
    # Example Usage:
    # Create a dummy pyproject.toml with some settings (or use existing one if formatter assumes it's set up)
    # For this test, we'll assume format_code relies on an existing pyproject.toml
    # that has been configured by generate_black_config_in_pyproject and generate_ruff_config_in_pyproject.

    test_dir = Path("temp_formatter_test_files")
    test_dir.mkdir(exist_ok=True)

    dummy_pyproject_path = test_dir / "pyproject.toml"
    dummy_profile = {
        "max_line_length": 79,
        "preferred_quotes": "single",
        "docstring_style": "google",
        # Add other fields expected by config generators if they were called directly
    }

    # Create a dummy pyproject.toml using our generators
    # (Need to import them for the __main__ block)
    try:
        # Adjust import path for direct execution vs. package execution
        if __package__ is None or __package__ == '':
            from src.profiler.config_generator import generate_black_config_in_pyproject, generate_ruff_config_in_pyproject
        else:
            from .profiler.config_generator import generate_black_config_in_pyproject, generate_ruff_config_in_pyproject

        generate_black_config_in_pyproject(dummy_profile, dummy_pyproject_path)
        generate_ruff_config_in_pyproject(dummy_profile, dummy_pyproject_path)
        print(f"Created dummy pyproject.toml at {dummy_pyproject_path}")
    except ImportError as e:
        print(f"Skipping pyproject.toml generation for example as config_generator not found (Error: {e}). Using manual minimal config.")
        # Create a minimal pyproject.toml manually for the test to proceed somewhat
        with open(dummy_pyproject_path, "w") as f:
            f.write("""
[tool.black]
line_length = 79

[tool.ruff]
line-length = 79
[tool.ruff.lint.flake8-quotes]
inline-quotes = "single"
docstring-quotes = "double"
[tool.ruff.format]
quote-style = "single"
            """)


    # Create a dummy Python file that needs formatting
    test_file_path = test_dir / "test_sample.py"
    original_content = """
import os, sys # Needs reformatting by Ruff (isort)

def very_long_function_name_that_will_certainly_exceed_the_line_length_limit_of_seventy_nine_characters(param1, param2):
    """This is a docstring that uses "double quotes" but profile prefers single."""
    my_variable_with_very_long_name = param1 + param2; # Semicolon, spacing
    print (f'Result: {my_variable_with_very_long_name}') # Extra space before (
    return my_variable_with_very_long_name
"""
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(original_content)

    print(f"\n--- Formatting {test_file_path.name} ---")
    success = format_code(test_file_path, dummy_profile, pyproject_path=dummy_pyproject_path)

    if success:
        print(f"Formatting function reported success for {test_file_path.name}.")
        with open(test_file_path, "r", encoding="utf-8") as f:
            formatted_content = f.read()
        print("\n--- Original Content: ---")
        print(original_content)
        print("\n--- Formatted Content: ---")
        print(formatted_content)
        if original_content == formatted_content:
            print("\nNote: Formatted content is the same as original. Tools might need specific rules enabled in pyproject.toml to make changes.")
        else:
            print("\nNote: Content has been modified by formatting tools.")
    else:
        print(f"Formatting function reported failure for {test_file_path.name}.")

    # Clean up
    # shutil.rmtree(test_dir) # Comment out to inspect files
    print(f"Test files are in {test_dir.resolve()}. Remove manually if needed.")
