# src/specs/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Phase(BaseModel): # Intended for Phase 4 output, but defined here for cohesiveness
    id: str = Field(..., description="A unique identifier for this phase, e.g., 'phase_1', 'phase_2'.")
    operation: str = Field(..., description="The specific operation to be performed in this phase, typically one of the verbs from Spec.operations.")
    target: str = Field(..., description="The primary target of the operation, usually a single Fully Qualified Name (FQN) of a symbol or a file path.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of operation-specific arguments or details needed to execute the phase.")

if __name__ == '__main__':
    # Example Usage and Validation
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
