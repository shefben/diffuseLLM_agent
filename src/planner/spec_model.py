from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Spec(BaseModel):
    issue_description: str = Field(description="The original issue description in plain English.")
    target_files: List[str] = Field(default_factory=list, description="List of file paths relevant to the issue.")
    operations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="A list of operations to be performed. Each operation is a dictionary, "
                    "e.g., {'name': 'add_decorator', 'target_function': 'my_func', 'decorator': '@log_calls'}."
                    "The structure of these dictionaries will be further refined by specific refactor operations."
    )
    acceptance_tests: List[str] = Field(
        default_factory=list,
        description="A list of human-readable descriptions of acceptance tests that the planned changes should satisfy."
    )
    # Optional: Add a field for the normalized/cleaned spec text from Tiny-T5 if needed later.
    # normalized_spec_text: Optional[str] = Field(None, description="The cleaned/canonicalized spec text from the diffusion model.")
    plan_prefix_summary: Optional[List[str]] = Field(
        default_factory=list, # Initialize with an empty list if not provided
        description="A summary of operations already included in the current partial plan leading up to the current state. Used by LLMs for context."
    )

    class Config:
        extra = "forbid" # To prevent unexpected fields

# Example Usage (for testing or if run directly)
if __name__ == "__main__":
    example_spec = Spec(
        issue_description="The system should log calls to critical functions.",
        target_files=["src/core/utils.py", "src/api/endpoints.py"],
        operations=[
            {
                "name": "add_decorator",
                "target_file": "src/core/utils.py",
                "target_function": "process_data",
                "decorator_name": "@log_entry_exit"
            },
            {
                "name": "add_import",
                "target_file": "src/core/utils.py",
                "import_statement": "from ..logging_utils import log_entry_exit"
            }
        ],
        acceptance_tests=[
            "When process_data in utils.py is called, its entry and exit should be logged.",
            "The new decorator @log_entry_exit should be correctly imported and applied."
        ],
        plan_prefix_summary=[] # Example with an empty summary
    )
    print("Example Spec:")
    try:
        # For Pydantic v2
        print(example_spec.model_dump_json(indent=2))
    except AttributeError:
        # For Pydantic v1
        print(example_spec.json(indent=2))

    # Example of a minimal spec
    minimal_spec = Spec(
        issue_description="Fix typo in README.",
        plan_prefix_summary=["Initial consideration: Check README.md for typos."] # Example with some prefix
    )
    print("\nMinimal Spec:")
    try:
        print(minimal_spec.model_dump_json(indent=2))
    except AttributeError:
        print(minimal_spec.json(indent=2))
