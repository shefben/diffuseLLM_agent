import subprocess
from pathlib import Path
from typing import List, Tuple
import shutil # Added for shutil.which

# Removed:
# import black
# import io
# import sys

def analyze_files_with_black(file_paths: List[str]) -> Tuple[str, str]:
    """
    Analyzes a list of Python files using Black via subprocess and returns diffs and errors.

    Args:
        file_paths: A list of strings, where each string is a path to a Python file.

    Returns:
        A tuple containing:
        - A string of all diffs concatenated.
        - A string of all errors concatenated.
    """
    all_diffs: List[str] = []
    all_errors: List[str] = []

    black_executable_path = shutil.which("black")
    if not black_executable_path:
        all_errors.append("Error: Black command not found. Is it installed and in PATH?")
        return "\n".join(all_diffs), "\n".join(all_errors)

    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        if not file_path.exists():
            all_errors.append(f"Error: File not found: {file_path_str}")
            continue
        if not file_path.is_file():
            all_errors.append(f"Error: Not a file: {file_path_str}")
            continue

        try:
            command = [black_executable_path, "--check", "--diff", str(file_path)]
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False  # Do not raise exception for non-zero exit codes
            )

            if process.stdout:
                all_diffs.append(f"--- Diff for {file_path_str} ---\n{process.stdout}")

            if process.stderr:
                # Black often prints "error: cannot format <file>: <reason>" to stderr for syntax errors
                # and also "Oh no! <internal_error_details>" for internal errors.
                all_errors.append(f"--- Errors/stderr for {file_path_str} ---\n{process.stderr}")

            # Black exit codes:
            # 0: no changes needed / success
            # 1: files would be reformatted
            # >1: error (e.g., 123 for internal errors, or syntax errors in file)
            if process.returncode > 1:
                all_errors.append(f"Error: Black exited with code {process.returncode} for {file_path_str}.")

        except FileNotFoundError: # Should be caught by shutil.which, but as a safeguard for the loop
            all_errors.append(f"Error: Black command not found during processing of {file_path_str}. Is it installed and in PATH?")
            # If black was initially found but removed mid-process, this might trigger.
            # Or if black_executable_path was somehow invalid despite shutil.which.
            break # Stop processing if black is suddenly not found
        except Exception as e:
            all_errors.append(f"An unexpected error occurred while processing {file_path_str} with Black: {e}")

    return "\n".join(all_diffs), "\n".join(all_errors)

if __name__ == '__main__':
    # Example usage:
    # Create a dummy file to test
    dummy_unformatted_path = Path("dummy_unformatted_file.py")
    with open(dummy_unformatted_path, "w") as f:
        f.write("print('hello world')\n") # Needs reformatting for quotes

    # Create another dummy file that is well-formatted
    formatted_file_path = Path("formatted_test_file.py")
    with open(formatted_file_path, "w") as f:
        f.write('print("hello world")\n')

    # Create a file with invalid syntax
    invalid_syntax_path = Path("invalid_syntax_file.py")
    with open(invalid_syntax_path, "w") as f:
        f.write("print 'hello world'\n") # Python 2 syntax

    files_to_check = [
        str(dummy_unformatted_path),
        str(formatted_file_path),
        str(invalid_syntax_path),
        "non_existent_file.py"
    ]
    diff_output, error_output = analyze_files_with_black(files_to_check)

    print("--- Black Diff Output ---")
    if diff_output.strip():
        print(diff_output)
    else:
        print("No diffs from Black.")

    print("\n--- Black Error Output ---")
    if error_output.strip():
        print(error_output)
    else:
        print("No errors from Black.")

    # Clean up dummy files
    import os
    os.remove(dummy_unformatted_path)
    os.remove(formatted_file_path)
    os.remove(invalid_syntax_path)
