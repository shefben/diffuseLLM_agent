# src/builder/commit_builder.py
from typing import Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
import subprocess # For actual git operations later
import shutil # For file operations or cleanup later

if TYPE_CHECKING:
    from src.planner.spec_model import Spec

class CommitBuilder:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the CommitBuilder.
        Args:
            config: Optional dictionary for configurations (e.g., git username, email, diff tool).
        """
        self.config = config if config else {}
        print(f"CommitBuilder initialized. Config: {self.config}")

    def reformat_patch(
        self,
        file_path: Path, # Path of the file being reformatted
        applied_patch_content: str # String content of the file after patch application
    ) -> str:
        """
        Mocks re-formatting file content using Black and Ruff --fix.
        """
        print(f"\nCommitBuilder: Reformatting content for file: {file_path}")

        # Mock Black Formatting
        black_formatted_content = applied_patch_content
        # Simulate Black's behavior: ensure content is stripped and ends with a single newline.
        black_formatted_content = f"# Mock Black formatting applied by CommitBuilder\n{black_formatted_content.strip()}\n"
        print("  CommitBuilder: (Mock) Black formatting applied.")

        # Mock Ruff --fix Formatting
        ruff_fixed_content = black_formatted_content
        # Simulate a simple Ruff fix, e.g., replacing a hypothetical common pattern.
        # This is highly dependent on what Ruff might actually fix.
        # For a generic mock, let's assume it might fix a quote style or remove unused vars.
        # Example: Replace a magic comment or a specific pattern.
        pattern_to_fix = "some_common_pattern_ruff_might_fix" # Example pattern
        if pattern_to_fix in ruff_fixed_content:
            ruff_fixed_content = ruff_fixed_content.replace(pattern_to_fix, "# FIXED_PATTERN by mock Ruff --fix")
            print(f"  CommitBuilder: (Mock) Ruff --fix applied specific pattern replacement for '{pattern_to_fix}'.")

        ruff_fixed_content = f"# Mock Ruff --fix applied by CommitBuilder (on top of Black's output)\n{ruff_fixed_content}"
        print("  CommitBuilder: (Mock) Ruff --fix general comment added.")

        print(f"CommitBuilder: Reformatting complete for file: {file_path}")
        return ruff_fixed_content

    def generate_changelog_entry(
        self,
        spec: 'Spec',
        diff_summary: Optional[str]
    ) -> str:
        """
        Generates a changelog entry string based on the spec and diff summary.
        """
        print("\nCommitBuilder: Generating changelog entry...")

        entry_lines = []

        # Add issue description
        if hasattr(spec, 'issue_description') and spec.issue_description:
            entry_lines.append(f"Issue: {spec.issue_description}")
        else:
            entry_lines.append("Issue: No description provided in spec.")

        # Add summary of operations
        entry_lines.append("\nSummary of Operations:")
        if hasattr(spec, 'operations') and spec.operations:
            if isinstance(spec.operations, list) and all(isinstance(op, dict) for op in spec.operations):
                for op in spec.operations:
                    op_name = op.get("name", "Unknown Operation")
                    op_target = op.get("target_file", "N/A")
                    # Could add more details from op.get("parameters") if needed
                    entry_lines.append(f"- Operation '{op_name}' on target '{op_target}'.")
            else:
                entry_lines.append("  (Operations format in spec is unexpected.)")
        else:
            entry_lines.append("  (No operations listed in spec.)")

        # Add diff summary
        entry_lines.append("\nDiff Summary:")
        if diff_summary and diff_summary.strip():
            entry_lines.append(diff_summary)
        else:
            entry_lines.append("  (No diff summary provided.)")

        changelog_string = "\n".join(entry_lines)
        print("CommitBuilder: Changelog entry generated.")
        return changelog_string

    def submit_via_git(
        self,
        branch_name: str,
        full_commit_message: str, # Combined commit message
        formatted_content_map: Dict[Path, str], # file_path -> new_content
        project_root: Path
    ) -> None:
        """
        Mocks the process of submitting changes via Git.
        Prints the git commands that would be run.
        """
        print(f"\nCommitBuilder: Attempting (mock) Git submission for branch '{branch_name}' in project '{project_root}'...")

        # 1. Checkout new branch
        # In a real scenario, we'd need to check if branch exists, handle errors, etc.
        # Also, ensure we are in the correct directory (project_root).
        git_command_checkout = ["git", "checkout", "-b", branch_name]
        print(f"  1. Would change CWD to: {project_root}")
        print(f"  2. Would run: {' '.join(git_command_checkout)}")

        # 2. Write files
        print(f"  3. Would write/overwrite {len(formatted_content_map)} files:")
        paths_to_add_relative: List[str] = []
        if not project_root.exists() or not project_root.is_dir():
            print(f"     ERROR: Project root {project_root} does not exist or is not a directory. Skipping file writes.")
        else:
            for file_abs_path, content in formatted_content_map.items():
                # Ensure file_abs_path is within project_root for safety before trying to get relative path
                try:
                    # If file_abs_path is already relative (e.g. from validated_patch_content_map keys),
                    # it needs to be made absolute first for writing, then relative for git add.
                    # Assuming keys in formatted_content_map are absolute or resolvable against project_root.
                    if not file_abs_path.is_absolute():
                        actual_abs_path = (project_root / file_abs_path).resolve()
                    else:
                        actual_abs_path = file_abs_path.resolve()

                    # Ensure the path is truly within the project_root after resolving
                    # This is a basic safety check.
                    actual_abs_path.relative_to(project_root) # This will raise ValueError if not under project_root

                    file_rel_path_for_git = actual_abs_path.relative_to(project_root)
                    paths_to_add_relative.append(str(file_rel_path_for_git))
                    print(f"    - Writing to file: {actual_abs_path} ({len(content)} bytes)")
                    # Mock actual write:
                    # actual_abs_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists
                    # with open(actual_abs_path, 'w', encoding='utf-8') as f:
                    #     f.write(content)
                except ValueError:
                    print(f"     ERROR: File path {file_abs_path} is not within project root {project_root}. Skipping.")
                except Exception as e:
                    print(f"     ERROR: Could not process file {file_abs_path}: {e}. Skipping.")


        # 3. Stage changes
        if paths_to_add_relative:
            git_command_add_files = ["git", "add"] + paths_to_add_relative
            print(f"  4. Would run: {' '.join(git_command_add_files)}")
        else:
            print("  4. No files to stage (or errors occurred).")


        # 4. Commit
        # For subprocess, commit message might need to be passed via temp file or stdin if too long.
        # Using -F <file> or piping to stdin is safer for complex messages.
        # For mock, just show a snippet.
        git_command_commit = ["git", "commit", "-m", full_commit_message] # This is unsafe for real use if message is complex
        print(f"  5. Would run: git commit -m \"{full_commit_message[:100].replace(chr(10), ' ')}...\" (message length: {len(full_commit_message)})")
        # Real example might be: subprocess.run(['git', 'commit', '-F', '-'], input=full_commit_message.encode('utf-8'), ...)

        # 5. Push branch (Optional for mock)
        git_command_push = ["git", "push", "-u", "origin", branch_name]
        print(f"  6. (Optional) Would run: {' '.join(git_command_push)}")

        # 6. Open Pull Request (Conceptual)
        print(f"  7. (Conceptual) Would open a pull request for branch '{branch_name}' against the main branch (e.g., via GitHub CLI or API).")

        print("CommitBuilder: (Mock) Git submission process outlined.")


    def process_and_submit_patch(
        self,
        validated_patch_content_map: Dict[Path, str], # Maps target file path to its new content
        spec: 'Spec',
        diff_summary: str, # Summary of changes from LLM or other source
        validator_results_summary: str, # Summary of validation tool outputs
        branch_name: str,
        commit_title: str,
        project_root: Path
    ) -> None:
        """
        Placeholder for processing validated patches, generating commit messages,
        and performing git operations.
        """
        print(f"\nCommitBuilder: Received request to process and submit for branch '{branch_name}'.")
        print(f"  Commit Title: {commit_title}")
        print(f"  Project Root: {project_root}")
        print(f"  Number of files in patch map: {len(validated_patch_content_map)}")
        if validated_patch_content_map:
            print(f"  Files to be modified/created: {list(validated_patch_content_map.keys())}")

        print(f"  Spec Description (first 100 chars): {spec.issue_description[:100] if hasattr(spec, 'issue_description') else 'N/A'}...")
        print(f"  Diff Summary (first 100 chars): {diff_summary[:100]}...")
        print(f"  Validator Results Summary (first 100 chars): {validator_results_summary[:100]}...")

        # --- Conceptual integration of reformat_patch ---
        print("\nCommitBuilder: Starting patch re-formatting (mock)...")
        formatted_content_map: Dict[Path, str] = {}
        if not validated_patch_content_map:
            print("  No files in patch map to reformat.")
        else:
            for file_p, content in validated_patch_content_map.items():
                # Ensure content is a string, as reformat_patch expects it.
                if isinstance(content, str):
                    formatted_content_map[file_p] = self.reformat_patch(file_p, content)
                else:
                    print(f"  Warning: Content for {file_p} is not a string (type: {type(content)}), skipping reformat.")
                    formatted_content_map[file_p] = str(content) # Or handle error more robustly

        print("CommitBuilder: All files processed for re-formatting (mock).\n")
        # In a real implementation, formatted_content_map would be used for subsequent steps.
        # For example, writing these contents to the actual files in the worktree.
        if formatted_content_map:
            print("  Formatted content map (example of one file):")
            if formatted_content_map: # Ensure not empty before trying to get iter
                example_file, example_content = next(iter(formatted_content_map.items()))
                print(f"    File: {example_file}")
                print(f"    Content (first 200 chars):\n{example_content[:200]}...")
        # --- End conceptual integration ---

        # --- Conceptual integration of generate_changelog_entry ---
        print("\nCommitBuilder: Generating changelog (mock)...")
        changelog_text = self.generate_changelog_entry(spec, diff_summary)
        print(f"CommitBuilder: Generated changelog:\n---\n{changelog_text}\n---")
        # --- End conceptual integration ---

        # --- Construct full commit message and call submit_via_git ---
        print("\nCommitBuilder: Preparing full commit message...")
        # commit_title is already an arg to process_and_submit_patch
        # validator_results_summary is also an arg

        full_commit_msg_parts = [commit_title, "\n\n", changelog_text]
        if validator_results_summary and validator_results_summary.strip():
            full_commit_msg_parts.extend(["\n\nValidator Results:\n", validator_results_summary])
        else:
            full_commit_msg_parts.extend(["\n\nValidator Results: All checks passed or no summary provided."])

        full_commit_message = "".join(full_commit_msg_parts)
        print(f"CommitBuilder: Full commit message prepared (length {len(full_commit_message)}). Preview (first 200 chars):\n{full_commit_message[:200]}...")

        print("\nCommitBuilder: Initiating Git submission (mock)...")
        # formatted_content_map was prepared in the reformatting step
        # Ensure formatted_content_map is used here, not validated_patch_content_map
        self.submit_via_git(
            branch_name,
            full_commit_message,
            formatted_content_map,
            project_root
        )
        print("CommitBuilder: (Mock) Git submission process completed.")
        # --- End Git submission ---

        print("\n  (CommitBuilder Placeholder: Actual git operations were mocked. No real changes made.)")
        pass

if __name__ == '__main__':
    print("--- CommitBuilder Example Usage (Conceptual) ---")

    # Mock Spec (assuming structure from src.planner.spec_model)
    class MockSpec:
        def __init__(self, description="Default mock issue description."):
            self.issue_description = description
            self.target_files = ["src/module/file1.py"] # Example
            # Add other fields if CommitBuilder directly uses them

    mock_spec_instance = MockSpec(description="Fix critical bug in user authentication module.")

    mock_validated_patches = {
        Path("src/module/file1.py"): "def updated_function():\n    return True\n",
        Path("src/module/new_file.py"): "# This is a new file\ndef new_feature():\n    pass\n"
    }
    mock_diff_summary = "Updated function logic in file1.py and added new_feature in new_file.py."
    mock_validator_summary = "Ruff: OK, Pyright: OK, Black-diff: OK, Pytest: OK"
    mock_branch = "feature/auth-bugfix-123"
    mock_title = "Fix: Resolve authentication bypass vulnerability"
    mock_project_root = Path("/tmp/mock_project_for_commitbuilder")

    builder_config = {"git_user_name": "AIAgent", "git_user_email": "ai.agent@example.com"}
    commit_builder = CommitBuilder(config=builder_config)

    commit_builder.process_and_submit_patch(
        validated_patch_content_map=mock_validated_patches,
        spec=mock_spec_instance, # type: ignore
        diff_summary=mock_diff_summary,
        validator_results_summary=mock_validator_summary,
        branch_name=mock_branch,
        commit_title=mock_title,
        project_root=mock_project_root
    )
    print("\n--- CommitBuilder Example Done ---")
