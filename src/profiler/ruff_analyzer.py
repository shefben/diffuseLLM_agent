import subprocess
from typing import List, Tuple

def analyze_files_with_ruff(file_paths: List[str]) -> Tuple[str, str]:
    """
    Analyzes a list of Python files using Ruff and returns the diff.

    Args:
        file_paths: A list of strings, where each string is a path to a Python file.

    Returns:
        A tuple containing:
        - stdout string (captures diff if any modifications are suggested)
        - stderr string (captures any errors from Ruff)
    """
    all_diffs = []
    all_errors = []

    # Ensure ruff is installed if this were a real environment.
    # For now, we assume it is.

    for file_path in file_paths:
        try:
            # Ruff's '--diff' flag causes it to print diffs to stdout.
            # Exit code 0: no issues found.
            # Exit code 1: issues found, and Ruff would make changes (or suggests them with --diff).
            # Exit code 2: Ruff encountered an error (e.g., invalid config, unparsable file).
            process = subprocess.run(
                ["ruff", "check", "--diff", file_path],
                capture_output=True,
                text=True,
                check=False # Do not raise exception for non-zero exit codes
            )

            if process.stdout:
                all_diffs.append(f"--- Diff for {file_path} ---\n{process.stdout}")

            # Ruff often prints informational messages or errors to stderr,
            # even if stdout also contains a diff.
            if process.stderr:
                # We might want to distinguish between "real" errors and informational messages.
                # For now, capture all stderr.
                all_errors.append(f"--- Ruff stderr for {file_path} ---\n{process.stderr}")

            if process.returncode > 1 : # Typically, 0 means no issues, 1 means linting issues found
                all_errors.append(f"Ruff exited with error code {process.returncode} for {file_path}.")

        except FileNotFoundError:
            all_errors.append(f"Error: Ruff command not found for {file_path}. Is it installed and in PATH?")
        except Exception as e:
            all_errors.append(f"An unexpected error occurred while processing {file_path} with Ruff: {e}")

    return "\n".join(all_diffs), "\n".join(all_errors)

if __name__ == '__main__':
    # Example usage:
    # Create a dummy file to test
    dummy_file_path = "dummy_ruff_test_file.py"
    with open(dummy_file_path, "w") as f:
        f.write("import os, sys\n") # Ruff will suggest combining imports
        f.write("my_var = 1 # comment\n") # Ruff might have opinions on spacing around comments or inline comments

    # Create another dummy file that is clean according to default ruff rules
    clean_file_path = "clean_ruff_test_file.py"
    with open(clean_file_path, "w") as f:
        f.write("import os\nimport sys\n\nmy_var = 1  # Correctly spaced comment\n")


    files_to_check = [dummy_file_path, clean_file_path, "non_existent_ruff_file.py"]
    diff_output, error_output = analyze_files_with_ruff(files_to_check)

    print("--- Ruff Diff Output ---")
    if diff_output:
        print(diff_output)
    else:
        print("No diffs from Ruff.")

    print("\n--- Ruff Error Output ---")
    if error_output:
        print(error_output)
    else:
        print("No errors from Ruff.")

    # Clean up dummy files
    import os
    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)
    if os.path.exists(clean_file_path):
        os.remove(clean_file_path)
