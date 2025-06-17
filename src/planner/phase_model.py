from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class Phase(BaseModel):
    operation_name: str = Field(
        description="The name of the refactor operation to be performed (e.g., 'add_decorator', 'extract_method')."
        " Should match a key in REFACTOR_OPERATION_MAP."
    )
    target_file: Optional[str] = Field(
        None,
        description=(
            "The primary file path this phase will operate on."
            " Deprecated in favour of 'target_files'."
        ),
    )
    target_files: Optional[List[str]] = Field(
        None,
        description=(
            "List of file paths affected by this phase. Use when the change spans multiple files."
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of parameters required by the operation. "
        "The structure of this dictionary is specific to the 'operation_name'.",
    )
    description: Optional[str] = Field(
        None,
        description="A human-readable description of what this phase entails or its purpose within the plan.",
    )
    # Optional: Add fields for estimated complexity, dependencies on other phases, etc.
    # depends_on: List[int] = Field(default_factory=list, description="List of phase indices this phase depends on.")

    class Config:
        extra = "forbid"


# Example Usage (for testing or if run directly)
if __name__ == "__main__":
    example_phase_add_decorator = Phase(
        operation_name="add_decorator",
        target_files=["src/core/utils.py"],
        parameters={
            "target_function": "process_data",
            "decorator_name": "@log_entry_exit",
            "decorator_import_statement": "from ..logging_utils import log_entry_exit",
        },
        description="Add @log_entry_exit decorator to process_data in utils.py and ensure it's imported.",
    )
    print("Example Phase (Add Decorator):")
    try:
        # For Pydantic v2
        print(example_phase_add_decorator.model_dump_json(indent=2))
    except AttributeError:
        # For Pydantic v1
        print(example_phase_add_decorator.json(indent=2))

    example_phase_extract_method = Phase(
        operation_name="extract_method",
        target_files=["src/api/helpers.py"],
        parameters={
            "source_function": "complex_function",
            "start_line": 55,
            "end_line": 72,
            "new_method_name": "extracted_logic_for_xyz",
        },
        description="Extract lines 55-72 from complex_function in helpers.py into a new method called extracted_logic_for_xyz.",
    )
    print("\nExample Phase (Extract Method):")
    try:
        print(example_phase_extract_method.model_dump_json(indent=2))
    except AttributeError:
        print(example_phase_extract_method.json(indent=2))
