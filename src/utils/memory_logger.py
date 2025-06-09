import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.planner.spec_model import Spec

def log_successful_patch(
    data_directory: Path,
    spec: 'Spec',
    diff_summary: str,
    successful_script_str: str, # The LibCST script string
    patch_source: Optional[str],
    verbose: bool = False # Added verbose flag for controlled printing
) -> bool: # Returns True on success, False on failure
    """
    Logs details of a successfully generated and applied patch to a JSONL file.

    Args:
        data_directory: The directory where the log file will be stored.
        spec: The Spec object that initiated the patch.
        diff_summary: A string summary of the diff (e.g., from difflib).
        successful_script_str: The LibCST script string that was successful.
        patch_source: Information about which agent component generated the patch.
        verbose: If True, prints success/failure messages.

    Returns:
        True if logging was successful, False otherwise.
    """
    try:
        data_directory.mkdir(parents=True, exist_ok=True)
    except OSError as e_mkdir:
        if verbose:
            print(f"MemoryLogger Error: Could not create data directory {data_directory}: {e_mkdir}")
        return False

    log_file_path = data_directory / "success_memory.jsonl"

    # Prepare spec operations summary
    spec_ops_summary = []
    if spec and hasattr(spec, 'operations') and isinstance(spec.operations, list):
        for op in spec.operations:
            if isinstance(op, dict):
                spec_ops_summary.append(op.get("name", "UnknownOperation"))
            elif hasattr(op, 'name'): # If operations are objects with a 'name' attribute
                spec_ops_summary.append(op.name)


    log_entry: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), # Switched to timezone.utc.now()
        "spec_issue_id": getattr(spec, 'issue_id', None), # Assuming Spec might have an issue_id
        "spec_issue_description": getattr(spec, 'issue_description', "N/A"),
        "spec_target_files": getattr(spec, 'target_files', []),
        "spec_operations_summary": spec_ops_summary,
        "successful_diff_summary_snippet": diff_summary[:1000] + ("..." if len(diff_summary) > 1000 else ""), # Store a snippet
        "successful_script_hash": hex(hash(successful_script_str)), # Avoid storing very large scripts directly for now
        # Consider storing full script if required, or a path to it if saved by CommitBuilder
        "patch_source": patch_source if patch_source else "Unknown"
    }

    try:
        with open(log_file_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")
        if verbose:
            print(f"MemoryLogger: Successfully logged patch metadata to {log_file_path}")
        return True
    except IOError as e_io:
        if verbose:
            print(f"MemoryLogger Error: Could not write to log file {log_file_path}: {e_io}")
        return False
    except Exception as e_gen: # Catch other potential errors during logging
        if verbose:
            print(f"MemoryLogger Error: Unexpected error during logging: {e_gen}")
        return False

def load_success_memory(data_directory: Path, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Loads all entries from the success_memory.jsonl log file.

    Args:
        data_directory: The directory where the log file is stored.
        verbose: If True, prints status messages.

    Returns:
        A list of dictionaries, where each dictionary is a logged entry.
        Returns an empty list if the file doesn't exist or an error occurs.
    """
    log_file_path = data_directory / "success_memory.jsonl"
    if not log_file_path.exists():
        if verbose:
            print(f"MemoryLogger: Success memory file not found at {log_file_path}.")
        return []

    entries: List[Dict[str, Any]] = []
    try:
        with open(log_file_path, "r", encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line_content = line.strip()
                if not line_content: # Skip empty lines
                    continue
                try:
                    entries.append(json.loads(line_content))
                except json.JSONDecodeError as e_json:
                    if verbose: # Print even if not fully verbose, as this is a data issue.
                        print(f"MemoryLogger Warning: Skipping malformed JSON line {line_number} in {log_file_path}: {e_json}. Line: '{line_content[:100]}...'")
        if verbose:
            print(f"MemoryLogger: Loaded {len(entries)} entries from success memory at {log_file_path}.")
        return entries
    except IOError as e_io:
        if verbose: # Print even if not fully verbose for IOErrors.
            print(f"MemoryLogger Error: Could not read success memory file {log_file_path}: {e_io}")
        return []
    except Exception as e_gen:
        if verbose:
            print(f"MemoryLogger Error: Unexpected error loading success memory: {e_gen}")
        return []
