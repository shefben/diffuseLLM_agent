import json
from pathlib import Path
from typing import List, Dict, Any

from src.utils.memory_logger import load_success_memory


def build_finetune_dataset(
    data_dir: Path, output_path: Path, verbose: bool = False
) -> bool:
    """Create prompt-completion pairs from success_memory.jsonl."""
    entries = load_success_memory(data_dir, verbose=verbose)
    if not entries:
        if verbose:
            print(f"build_finetune_dataset: no entries found in {data_dir}")
        return False

    dataset: List[Dict[str, Any]] = []
    for entry in entries:
        issue = entry.get("spec_issue_description", "")
        operations = entry.get("spec_operations_details", [])
        diff = entry.get("successful_diff_summary", "")
        source = entry.get("patch_source", "")
        completion = entry.get("successful_script", "")
        prompt = (
            f"Issue: {issue}\n"
            f"Operations: {operations}\n"
            f"Patch Source: {source}\n"
            f"Diff Summary:\n{diff}\n"
            "\n# Response:\n"
        )
        dataset.append({"prompt": prompt, "completion": completion})

    try:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for item in dataset:
                f_out.write(json.dumps(item) + "\n")
        if verbose:
            print(
                f"build_finetune_dataset: wrote {len(dataset)} items to {output_path}"
            )
        return True
    except Exception as e:
        if verbose:
            print(f"build_finetune_dataset: failed to write dataset: {e}")
        return False
