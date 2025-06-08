# src/specs/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Spec(BaseModel):
    task: str = Field(..., description="Free-text summary of the task or issue.")
    target_symbols: List[str] = Field(default_factory=list, description="Fully Qualified Names (FQNs) of symbols relevant to the task, potentially identified from the codebase.")
    operations: List[str] = Field(default_factory=list, description="A list of verbs or operation keywords suggesting the types of changes needed (e.g., add_decorator, edit_test, fix_bug, refactor_method).")
    acceptance: List[str] = Field(default_factory=list, description="Criteria for task completion, often names of tests that must pass or descriptions of behavior to verify.")

class Phase(BaseModel): # Intended for Phase 4 output, but defined here for cohesiveness
    id: str = Field(..., description="A unique identifier for this phase, e.g., 'phase_1', 'phase_2'.")
    operation: str = Field(..., description="The specific operation to be performed in this phase, typically one of the verbs from Spec.operations.")
    target: str = Field(..., description="The primary target of the operation, usually a single Fully Qualified Name (FQN) of a symbol or a file path.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of operation-specific arguments or details needed to execute the phase.")

if __name__ == '__main__':
    # Example Usage and Validation
    try:
        example_spec_data = {
            "task": "add caching to user service get_data method",
            "target_symbols": ["services.user.UserService.get_data", "utils.caching.cache_decorator"],
            "operations": ["add_decorator", "update_test"],
            "acceptance": ["test_user_service::test_get_data_cache_hit", "test_user_service::test_get_data_cache_expiry"]
        }
        spec_instance = Spec(**example_spec_data)
        print("Successfully created Spec instance:")
        print(spec_instance.model_dump_json(indent=2))

        example_spec_minimal = {"task": "fix typo in docs"}
        spec_minimal_instance = Spec(**example_spec_minimal)
        print("\nSuccessfully created minimal Spec instance:")
        print(spec_minimal_instance.model_dump_json(indent=2))
        # Default factory will create empty lists for other fields.

    except Exception as e:
        print(f"Error during Pydantic model example: {e}")

    try:
        example_phase_data = {
            "id": "phase_1_add_decorator",
            "operation": "add_decorator",
            "target": "services.user.UserService.get_data",
            "payload": {"decorator_name": "utils.caching.cache_decorator", "decorator_args": ["ttl=300"]}
        }
        phase_instance = Phase(**example_phase_data)
        print("\nSuccessfully created Phase instance:")
        print(phase_instance.model_dump_json(indent=2))

    except Exception as e:
        print(f"Error during Pydantic Phase model example: {e}")
