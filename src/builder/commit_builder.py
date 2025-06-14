# src/builder/commit_builder.py
import subprocess
import tempfile
import os
import json # For patch_meta.json
from datetime import datetime # For timestamp in patch_meta.json
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from pathlib import Path
# import shutil

if TYPE_CHECKING:
    from src.planner.spec_model import Spec

class CommitBuilder:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the CommitBuilder.
        Args:
            app_config: Optional dictionary for configurations.
                    Expected keys: "verbose" (bool), "black_path" (str), "ruff_path" (str).
        """
        self.app_config = app_config if app_config else {}
        self.verbose = self.app_config.get("general", {}).get("verbose", False)
        self.black_path = self.app_config.get("tools", {}).get("black_path", "black")
        self.ruff_path = self.app_config.get("tools", {}).get("ruff_path", "ruff")
        self.output_base_dir_config = self.app_config.get("general", {}).get("patches_output_dir") # Store for later use
        # self.output_base_dir is initialized in process_and_submit_patch or can be set here if always needed
        # For now, keeping its logic tied to where it's used or assuming it might be passed explicitly.
        # If it MUST be initialized in __init__, it would be:
        # self.output_base_dir = Path(self.output_base_dir_config) if self.output_base_dir_config else Path.home() / ".autopatch/patches"

        print(f"CommitBuilder initialized. Black: '{self.black_path}', Ruff: '{self.ruff_path}', Verbose: {self.verbose}. Output Base Dir from config: {self.output_base_dir_config}. AppConfig: {self.app_config}")

    def reformat_patch(
        self,
        file_path: Path, # Path of the file being reformatted (used for logging)
        applied_patch_content: str # String content of the file after patch application
    ) -> str:
        """
        Re-formats file content using Black and then Ruff --fix.
        """
        current_content = applied_patch_content
        if self.verbose:
            print(f"\nCommitBuilder: Starting reformatting for file: {file_path} (initial length: {len(current_content)})")

        # --- Black Formatting ---
        tmp_black_file_path_str: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding='utf-8') as tmp_black_file:
                tmp_black_file.write(current_content)
                tmp_black_file_path_str = tmp_black_file.name

            black_command = [self.black_path, tmp_black_file_path_str]
            if self.verbose: print(f"CommitBuilder: Running Black: {' '.join(black_command)}")
            result_black = subprocess.run(black_command, capture_output=True, text=True, check=False)

            if result_black.returncode == 0:
                with open(tmp_black_file_path_str, "r", encoding='utf-8') as f_read:
                    current_content = f_read.read()
                if self.verbose: print(f"CommitBuilder: Black formatting successful for {file_path}.")
            else:
                # Black exit codes: 1 for errors like syntax error, 123 for internal error.
                # It modifies in place and exits 0 if no syntax error and formatting applied/not needed.
                print(f"CommitBuilder Warning: Black exited with code {result_black.returncode} for {file_path}.")
                if result_black.stderr: print(f"Black stderr for {file_path}:\n{result_black.stderr.strip()}")
                if result_black.stdout and self.verbose: print(f"Black stdout for {file_path}:\n{result_black.stdout.strip()}")
                # If Black fails (e.g. syntax error), it doesn't modify the file.
                # We proceed with the content as it was before this Black step (i.e., current_content is unchanged).
        except FileNotFoundError:
            print(f"CommitBuilder Warning: Black executable not found at '{self.black_path}'. Skipping Black formatting for {file_path}.")
        except Exception as e_black:
            print(f"CommitBuilder Warning: Error during Black formatting for {file_path}: {e_black}. Skipping Black formatting.")
        finally:
            if tmp_black_file_path_str and Path(tmp_black_file_path_str).exists():
                os.remove(tmp_black_file_path_str)

        # --- Ruff --fix Formatting (on Black-formatted content) ---
        tmp_ruff_file_path_str: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding='utf-8') as tmp_ruff_file:
                tmp_ruff_file.write(current_content) # Write content potentially modified by Black
                tmp_ruff_file_path_str = tmp_ruff_file.name

            ruff_command = [self.ruff_path, "check", "--fix", "--no-cache", "--quiet", tmp_ruff_file_path_str]
            # Using "check --fix" for lint-fixing. "ruff format" could be an alternative if only formatting is desired.
            # Consider adding specific rules to --select if needed, e.g., "--select=I" for isort.

            if self.verbose: print(f"CommitBuilder: Running Ruff --fix: {' '.join(ruff_command)} for {file_path}")
            result_ruff = subprocess.run(ruff_command, capture_output=True, text=True, check=False)

            # Ruff exit codes: 0 if no errors/no fixes or if fixes were applied without errors.
            # 1 if errors were found (and potentially fixed by --fix, but some might remain or new ones introduced by fix).
            # For --fix, a return code of 0 means either "no issues" or "issues found and fixed successfully".
            # A return code of 1 means "issues found, and after fixing, some issues might still remain".
            # We always read the file content back as Ruff modifies in place with --fix.

            with open(tmp_ruff_file_path_str, "r", encoding='utf-8') as f_read:
                current_content = f_read.read()

            if result_ruff.returncode == 0:
                if self.verbose: print(f"CommitBuilder: Ruff --fix successful (no outstanding errors/lint issues) for {file_path}.")
            elif result_ruff.returncode == 1:
                if self.verbose: print(f"CommitBuilder: Ruff --fix applied changes for {file_path}, but some lint issues might remain or were reported. Output:\n{result_ruff.stdout.strip()}")
                # Even if Ruff reports issues (exit 1), we use the fixed content.
                # The primary purpose here is auto-fixing; validation of remaining issues is separate.
            else: # Other Ruff errors (e.g., config error, internal error)
                print(f"CommitBuilder Warning: Ruff --fix exited with code {result_ruff.returncode} for {file_path}.")
                if result_ruff.stderr: print(f"Ruff stderr for {file_path}:\n{result_ruff.stderr.strip()}")
                if result_ruff.stdout: print(f"Ruff stdout for {file_path}:\n{result_ruff.stdout.strip()}")
                # If Ruff has a more critical error, we proceed with content as it was before this Ruff step.
                # However, current_content was already updated from tmp_ruff_file_path_str. This is acceptable.
        except FileNotFoundError:
            print(f"CommitBuilder Warning: Ruff executable not found at '{self.ruff_path}'. Skipping Ruff --fix for {file_path}.")
        except Exception as e_ruff:
            print(f"CommitBuilder Warning: Error during Ruff --fix for {file_path}: {e_ruff}. Skipping Ruff --fix.")
        finally:
            if tmp_ruff_file_path_str and Path(tmp_ruff_file_path_str).exists():
                os.remove(tmp_ruff_file_path_str)

        if self.verbose:
            print(f"CommitBuilder: Reformatting complete for file: {file_path} (final length: {len(current_content)})")
        return current_content

    def generate_changelog_entry(
        self,
        spec: 'Spec',
        diff_summary: Optional[str]
    ) -> str:
        """
        Generates a changelog entry string based on the spec and diff summary.
        """
        print("\nCommitBuilder: Generating changelog entry...")

        issue_desc = getattr(spec, 'issue_description', "No description provided.")

        ops_summary_parts = ["Summary of Operations:"]
        if hasattr(spec, 'operations') and spec.operations:
            if isinstance(spec.operations, list) and all(isinstance(op, dict) for op in spec.operations):
                for op_idx, op in enumerate(spec.operations):
                    op_name = op.get("name", "Unknown Operation")
                    op_target = op.get("target_file", "N/A")
                    ops_summary_parts.append(f"  - Op {op_idx+1}: '{op_name}' on '{op_target}'.")
            else:
                ops_summary_parts.append("  (Operations format in spec is unexpected.)")
        else:
            ops_summary_parts.append("  (No operations listed in spec.)")
        ops_summary_str = "\n".join(ops_summary_parts)

        diff_summary_str = "Diff Summary:\n" + (diff_summary.strip() if diff_summary and diff_summary.strip() else "  (No diff summary provided.)")

        # Simple paragraph style for the list item
        changelog_markdown = f"- Issue: {issue_desc}\n  {ops_summary_str.replace('\\n', '\\n  ')}\n  {diff_summary_str.replace('\\n', '\\n  ')}\n"

        print("CommitBuilder: Changelog entry (Markdown list item) generated.")
        return changelog_markdown

    def _update_changelog_file(self, project_root: Path, changelog_entry_markdown: str) -> bool:
        """
        Creates or updates CHANGELOG.md with the new entry under an "Unreleased" section.
        """
        changelog_path = project_root / "CHANGELOG.md"
        new_entry = changelog_entry_markdown.rstrip() + "\n" # Ensure single newline at end of entry

        try:
            if not changelog_path.exists():
                if self.verbose: print(f"CommitBuilder: CHANGELOG.md not found at {changelog_path}. Creating new one.")
                content = f"""# Changelog
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
{new_entry}"""
                changelog_path.write_text(content, encoding="utf-8")
                return True

            lines = changelog_path.read_text(encoding="utf-8").splitlines()
            unreleased_header_index = -1

            # Find "## [Unreleased]" section
            for i, line in enumerate(lines):
                if re.match(r"^##\s*\[Unreleased\]", line, re.IGNORECASE):
                    unreleased_header_index = i
                    break

            if unreleased_header_index != -1:
                # Insert new entry after the "[Unreleased]" header
                # Add a blank line before the new entry if the line after header isn't already blank or another list item
                insert_point = unreleased_header_index + 1
                if insert_point < len(lines) and lines[insert_point].strip() and not lines[insert_point].strip().startswith("-"):
                    lines.insert(insert_point, "") # Add a blank line for separation
                    insert_point +=1
                elif insert_point == len(lines) and lines[unreleased_header_index].strip(): # Header is last line
                    lines.append("") # Add blank line after header
                    insert_point +=1

                lines.insert(insert_point, new_entry)
            else:
                # No "[Unreleased]" section, try to add it after main changelog header or preamble
                changelog_header_index = -1
                preamble_end_index = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith("# Changelog"):
                        changelog_header_index = i
                    if "Semantic Versioning" in line and changelog_header_index != -1: # End of common preamble
                        preamble_end_index = i
                        break

                insert_point_for_section = 0
                if preamble_end_index != -1:
                    insert_point_for_section = preamble_end_index + 1
                    lines.insert(insert_point_for_section, "") # Blank line after preamble
                    insert_point_for_section +=1
                elif changelog_header_index != -1:
                    insert_point_for_section = changelog_header_index + 1
                    lines.insert(insert_point_for_section, "") # Blank line after main header
                    insert_point_for_section +=1

                lines.insert(insert_point_for_section, "## [Unreleased]")
                lines.insert(insert_point_for_section + 1, new_entry)

            changelog_path.write_text("\n".join(lines) + "\n", encoding="utf-8") # Ensure trailing newline for file
            if self.verbose: print(f"CommitBuilder: Updated CHANGELOG.md at {changelog_path} with new entry.")
            return True

        except IOError as e:
            print(f"CommitBuilder Error: Failed to read/write {changelog_path}: {e}")
            return False
        except Exception as e_gen:
            print(f"CommitBuilder Error: Unexpected error updating changelog {changelog_path}: {e_gen}")
            return False

    def submit_via_git(
        self,
        branch_name: str,
        full_commit_message: str, # Combined commit message
        formatted_content_map: Dict[Path, str], # file_path -> new_content
        project_root: Path
    ) -> bool:
        """
        Submits changes via Git by checking out a new branch, adding files,
        committing, and optionally pushing.
        Returns:
            True if all Git operations were successful, False otherwise.
        """
        if self.verbose:
            print(f"\nCommitBuilder: Attempting Git submission for branch '{branch_name}' in project '{project_root}'...")

        # 1. Checkout new branch
        if self.verbose: print(f"  1. Checking out new branch '{branch_name}'...")
        checkout_command = ["git", "checkout", "-b", branch_name]
        if not self._run_git_command(checkout_command, project_root):
            # Attempt to checkout existing branch if -b fails (e.g. branch already exists from a previous run)
            if self.verbose: print(f"  Branch creation failed, attempting to checkout existing branch '{branch_name}'...")
            checkout_existing_command = ["git", "checkout", branch_name]
            if not self._run_git_command(checkout_existing_command, project_root):
                print(f"CommitBuilder Error: Failed to checkout new or existing branch '{branch_name}'.")
                return False

        # 2. Write files (This part is not a git command, but file system operations)
        if self.verbose: print(f"  2. Writing/overwriting {len(formatted_content_map)} files to working directory...")
        paths_to_add_relative: List[str] = []
        if not project_root.exists() or not project_root.is_dir():
            print(f"CommitBuilder Error: Project root {project_root} does not exist or is not a directory. Skipping file writes.")
            return False # Cannot proceed without project root

        for file_rel_path_obj, content in formatted_content_map.items(): # Assuming keys are Path objects relative to project_root
            try:
                actual_abs_path = (project_root / file_rel_path_obj).resolve()
                # Security check: Ensure path is within project_root
                actual_abs_path.relative_to(project_root)

                actual_abs_path.parent.mkdir(parents=True, exist_ok=True)
                with open(actual_abs_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                if self.verbose: print(f"    - Wrote to file: {actual_abs_path}")
                paths_to_add_relative.append(str(file_rel_path_obj)) # Add relative path for 'git add'
            except ValueError: # Path outside project_root
                print(f"CommitBuilder Error: File path {file_rel_path_obj} resolved outside project root {project_root}. Skipping.")
                return False # Security risk or misconfiguration
            except Exception as e:
                print(f"CommitBuilder Error: Could not write file {file_rel_path_obj}: {e}")
                return False

        # 3. Stage changes
        if not paths_to_add_relative:
            if self.verbose: print("  3. No files specified to stage for Git. This might be okay if the commit is for other reasons (e.g. merge).")
        else:
            if self.verbose: print(f"  3. Staging {len(paths_to_add_relative)} files...")
            add_command = ["git", "add"] + paths_to_add_relative
            if not self._run_git_command(add_command, project_root):
                print(f"CommitBuilder Error: Failed to stage files: {paths_to_add_relative}")
                return False

        # 4. Commit
        if self.verbose: print(f"  4. Committing changes (message length: {len(full_commit_message)})...")
        # Using -F - to pass commit message via stdin for safety with complex messages
        commit_command = ["git", "commit", "-F", "-"]
        if not self._run_git_command(commit_command, project_root, input_data=full_commit_message):
            # Note: 'git commit' can fail if there's nothing to commit (e.g., if 'git add' didn't actually stage changes)
            # or if pre-commit hooks fail. The _run_git_command helper will print stderr.
            print(f"CommitBuilder Error: Failed to commit changes.")
            # Check if it failed because there was nothing to commit (can happen if files were written but resulted in no diff)
            # This requires inspecting stderr, which _run_git_command already prints.
            # For now, any commit failure is treated as an error for the submission process.
            return False

        # 5. Push branch (Conditional)
        auto_push_config = self.app_config.get("git_submission", {}).get("auto_push", False)
        if auto_push_config:
            if self.verbose: print(f"  5. Auto-push enabled. Pushing branch '{branch_name}' to origin...")
            push_command = ["git", "push", "-u", "origin", branch_name]
            if not self._run_git_command(push_command, project_root):
                print(f"CommitBuilder Error: Failed to push branch '{branch_name}'.")
                return False # Push failure means overall submission failure if auto-push is on
        else:
            if self.verbose: print("  5. Auto-push disabled. Skipping 'git push'.")

        # 6. Open Pull Request (Conceptual - not implemented via subprocess here)
        if self.verbose:
             print(f"  6. (Conceptual) Next step would be to open a pull request for branch '{branch_name}' against the main branch.")

        print("CommitBuilder: Git submission process completed.")
        return True # Assuming all steps including conditional push were successful
        return True # Assuming all steps including conditional push were successful

    def _run_git_command(self, command: List[str], project_root: Path, input_data: Optional[str] = None) -> bool:
        """
        Runs a Git command using subprocess.
        Args:
            command: The Git command and its arguments as a list of strings.
            project_root: The Path to the root of the Git repository (used as cwd).
            input_data: Optional string data to pass to the command's stdin (e.g., for commit messages).
        Returns:
            True if the command was successful (exit code 0), False otherwise.
        """
        if self.verbose:
            print(f"CommitBuilder: Running Git command: {' '.join(command)} in {project_root}")

        try:
            process = subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                input=input_data,
                check=False # Handle non-zero exit codes manually
            )
            if process.returncode == 0:
                if self.verbose and process.stdout:
                    print(f"Git command stdout:\n{process.stdout}")
                if self.verbose and process.stderr: # Some successful git commands print to stderr (e.g. 'git checkout -b')
                    print(f"Git command stderr (info):\n{process.stderr}")
                return True
            else:
                print(f"CommitBuilder Error: Git command failed with exit code {process.returncode}")
                print(f"  Command: {' '.join(command)}")
                if process.stdout: print(f"  Stdout:\n{process.stdout}")
                if process.stderr: print(f"  Stderr:\n{process.stderr}")
                return False
        except FileNotFoundError:
            print(f"CommitBuilder Error: Git command not found (usually 'git'). Ensure Git is installed and in PATH.")
            return False
        except Exception as e:
            print(f"CommitBuilder Error: An unexpected error occurred while running Git command {' '.join(command)}: {e}")
            return False

    def save_patch_to_filesystem(
        self,
        output_base_dir_param: Path,
        patch_set_name: str,
        formatted_content_map: Dict[Path, str],
        changelog_entry: str,
        spec: 'Spec',
        commit_title: Optional[str] = None,
        validator_results_summary: Optional[str] = None,
        patch_source: Optional[str] = None # New parameter
    ) -> Optional[Path]:
        """
        Saves the formatted patch content, changelog, metadata, and other info to the filesystem.
        Args:
            output_base_dir_param: The base directory where the patch set directory will be created.
            patch_set_name: Name for the patch set directory (e.g., derived from branch name).
            formatted_content_map: Dictionary mapping project-relative file paths to their new content.
            changelog_entry: The generated changelog entry string.
            spec: The original Spec object.
        Returns:
            The Path to the created patch_set directory, or None on failure.
        """
        if self.verbose:
            print(f"\nCommitBuilder: Saving patch set '{patch_set_name}' to base directory '{output_base_dir_param}'...")

        patch_output_dir = Path(output_base_dir_param) / patch_set_name

        try:
            patch_output_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose: print(f"CommitBuilder: Ensured output directory exists: {patch_output_dir}")
        except OSError as e_mkdir:
            print(f"CommitBuilder Error: Could not create output directory {patch_output_dir}: {e_mkdir}")
            return None

        # Write Formatted Files
        if self.verbose: print(f"CommitBuilder: Writing {len(formatted_content_map)} formatted file(s) to {patch_output_dir}...")
        for relative_file_path, content_str in formatted_content_map.items():
            output_file_path: Optional[Path] = None # Define for use in exception messages
            try:
                # Ensure relative_file_path is treated as relative to the patch_output_dir
                # It should not be an absolute path.
                if Path(relative_file_path).is_absolute():
                    print(f"CommitBuilder Warning: Received absolute path '{relative_file_path}' in formatted_content_map. This is unexpected. Attempting to use filename only.")
                    # This handling might need refinement based on how these paths are generated.
                    # For safety, place it at the root of patch_output_dir.
                    output_file_path = patch_output_dir / Path(relative_file_path).name
                else:
                    output_file_path = patch_output_dir / relative_file_path

                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file_path, "w", encoding='utf-8') as f:
                    f.write(content_str)
                if self.verbose: print(f"  - Wrote {output_file_path}")
            except IOError as e_io_file:
                print(f"CommitBuilder Error: Could not write file {output_file_path if output_file_path else relative_file_path}: {e_io_file}")
            except Exception as e_gen_file:
                print(f"CommitBuilder Error: Unexpected error writing file {output_file_path if output_file_path else relative_file_path}: {e_gen_file}")

        # Write Changelog File
        changelog_file_path: Optional[Path] = None
        try:
            changelog_file_path = patch_output_dir / "CHANGELOG_PATCH.md"
            with open(changelog_file_path, "w", encoding='utf-8') as f:
                f.write(changelog_entry)
            if self.verbose: print(f"CommitBuilder: Wrote changelog to {changelog_file_path}")
        except IOError as e_io_cl:
            print(f"CommitBuilder Error: Could not write changelog file {changelog_file_path if changelog_file_path else 'CHANGELOG_PATCH.md'}: {e_io_cl}")

        # Write Metadata File (patch_meta.json)
        meta_file_path: Optional[Path] = None
        try:
            meta_file_path = patch_output_dir / "patch_meta.json"
            spec_data = {}
            if hasattr(spec, 'model_dump'):
                spec_data = spec.model_dump(mode='json')
            elif hasattr(spec, 'dict'):
                spec_data = spec.dict()
            else:
                spec_data = {"issue_description": getattr(spec, 'issue_description', "N/A"),
                             "target_files": getattr(spec, 'target_files', []),
                             "operations": getattr(spec, 'operations', []),
                             "acceptance_tests": getattr(spec, 'acceptance_tests', [])}

            metadata_to_save = {
                "patch_set_name": patch_set_name,
                "commit_title": commit_title if commit_title else "N/A",
                "original_spec": spec_data,
                "validator_results_summary": validator_results_summary if validator_results_summary else "N/A",
                "patch_source": patch_source if patch_source else "Unknown", # Added patch_source
                "generation_timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "commit_builder_version": "0.1.0" # Example version
            }
            with open(meta_file_path, "w", encoding='utf-8') as f:
                json.dump(metadata_to_save, f, indent=2)
            if self.verbose: print(f"CommitBuilder: Wrote metadata to {meta_file_path}")
        except IOError as e_io_meta:
            print(f"CommitBuilder Error: Could not write metadata file {meta_file_path if meta_file_path else 'patch_meta.json'}: {e_io_meta}")
        except TypeError as e_type_meta:
             print(f"CommitBuilder Error: Could not serialize metadata for {meta_file_path if meta_file_path else 'patch_meta.json'}: {e_type_meta}")

        if self.verbose: print(f"CommitBuilder: Finished saving patch set '{patch_set_name}'.")
        return patch_output_dir


    def process_and_submit_patch(
        self,
        validated_patch_content_map: Dict[Path, str], # Keys are project-relative paths
        spec: 'Spec',
        diff_summary: str,
        validator_results_summary: str,
        branch_name: str,
        commit_title: str,
        project_root: Path,
        patch_source: Optional[str] = None # New parameter
    ) -> Optional[Path]:
        """
        Processes validated patches, generates changelog and metadata,
        and saves everything to the filesystem.
        """
        if self.verbose:
            print(f"\nCommitBuilder: Received request to process and save patch set '{branch_name}'.")
            print(f"  Commit Title (for metadata): {commit_title}")
            print(f"  Project Root (for default output base): {project_root}")
            print(f"  Number of files in patch map: {len(validated_patch_content_map)}")

        # --- Reformat Patch Content ---
        if self.verbose: print("\nCommitBuilder: Starting patch re-formatting...")
        formatted_content_map: Dict[Path, str] = {}
        if not validated_patch_content_map:
            print("  No files in patch map to reformat.")
        else:
            for file_rel_path, content in validated_patch_content_map.items():
                if isinstance(content, str):
                    # reformat_patch uses file_path for logging purposes only.
                    # The actual path for writing is constructed in save_patch_to_filesystem.
                    formatted_content_map[file_rel_path] = self.reformat_patch(file_rel_path, content)
                else:
                    print(f"  Warning: Content for {file_rel_path} is not a string (type: {type(content)}), skipping reformat.")
                    formatted_content_map[file_rel_path] = str(content)
        if self.verbose: print("CommitBuilder: All files processed for re-formatting.\n")

        # --- Generate Changelog ---
        if self.verbose: print("CommitBuilder: Generating changelog...")
        changelog_text = self.generate_changelog_entry(spec, diff_summary) # This is now Markdown
        if self.verbose: print(f"CommitBuilder: Generated changelog markdown (length {len(changelog_text)} chars).")

        # --- Update CHANGELOG.md file ---
        update_success = self._update_changelog_file(project_root, changelog_text)
        if self.verbose and not update_success:
            print("CommitBuilder Warning: Failed to update CHANGELOG.md file.")
        # Continue with commit process even if CHANGELOG.md update fails, as it's a non-critical failure for the patch itself.

        # --- Construct Full Commit Message ---
        full_commit_message = f"""{commit_title}

{changelog_text}

---
Validator Results:
{validator_results_summary}
---
Patch Source: {patch_source if patch_source else 'Unknown'}
"""
        if self.verbose: print(f"CommitBuilder: Constructed full commit message (length {len(full_commit_message)} chars).")

        # --- Call submit_via_git ---
        if self.verbose: print("CommitBuilder: Attempting Git submission...")
        git_submission_successful = self.submit_via_git(
            branch_name=branch_name,
            full_commit_message=full_commit_message,
            formatted_content_map=formatted_content_map, # Ensure this map has project-relative Path keys
            project_root=project_root
        )

        # --- Determine Output Base Directory & Save Patch Artifacts (Always) ---
        output_base_dir: Path
        if self.output_base_dir_config:
            output_base_dir = Path(self.output_base_dir_config)
        else:
            output_base_dir = project_root / ".autopatches" # Default if not in config

        if self.verbose: print(f"CommitBuilder: Initiating save of patch artifacts to filesystem. Base dir: {output_base_dir}, Patch set name: {branch_name}")
        saved_artifact_path = self.save_patch_to_filesystem(
            output_base_dir_param=output_base_dir,
            patch_set_name=branch_name,
            formatted_content_map=formatted_content_map,
            changelog_entry=changelog_text, # Could also pass full_commit_message or parts if preferred for changelog file
            spec=spec,
            commit_title=commit_title,
            validator_results_summary=validator_results_summary,
            patch_source=patch_source
        )
        if saved_artifact_path:
            if self.verbose: print(f"CommitBuilder: Patch artifacts successfully saved to: {saved_artifact_path}")
        else:
            # This is a warning because the primary goal might be Git submission.
            # However, if saving artifacts is critical, this could influence the overall success.
            print(f"CommitBuilder Warning: Failed to save patch artifacts for '{branch_name}'.")


        # --- Conditional Return Value ---
        if not git_submission_successful:
            print(f"CommitBuilder Error: Git submission failed for branch '{branch_name}'. Patch artifacts might be at {saved_artifact_path if saved_artifact_path else 'N/A'}.")
            return None # Signal overall failure to PhasePlanner

        # If Git submission was successful, return the path to the saved artifacts
        # (which could be None if artifact saving failed, but Git was primary goal)
        return saved_artifact_path

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
