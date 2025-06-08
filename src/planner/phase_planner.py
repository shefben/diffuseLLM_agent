# src/planner/phase_planner.py
from typing import List, Dict, Any, Union, Optional, TYPE_CHECKING, Type # Added Type
from pathlib import Path
import json
import hashlib
import random # For dummy scorer

# Forward references for type hints if full imports are problematic
if TYPE_CHECKING:
    from ..digester.repository_digester import RepositoryDigester
    # Spec, Phase, BaseRefactorOperation are now directly imported.
    # REFACTOR_OPERATION_INSTANCES is used directly.

# Actual imports
from .spec_model import Spec
from .phase_model import Phase
from .refactor_grammar import BaseRefactorOperation, REFACTOR_OPERATION_INSTANCES
# REFACTOR_OPERATION_CLASSES is not directly used by PhasePlanner logic, REFACTOR_OPERATION_INSTANCES is.

# Fallback for RepositoryDigester if not available (e.g. in isolated subtask)
# This is primarily for type hinting if RepositoryDigester cannot be imported directly.
# The actual instance is passed to __init__.
if TYPE_CHECKING or 'RepositoryDigester' not in globals(): # Keep this for RepositoryDigester hint
    class RepositoryDigester: # type: ignore
        project_call_graph: Dict
        # Add other attributes if _get_graph_statistics accesses them directly.
        pass # Ensure class body is not empty


class PhasePlanner:
    def __init__(
        self,
        style_fingerprint_path: Path,
        digester: 'RepositoryDigester',
        scorer_model_config: Any, # Config for Phi-2 LLM scorer
        refactor_op_map: Optional[Dict[str, BaseRefactorOperation]] = None,
        beam_width: int = 3 # Default beam_width to 3
    ):
        self.style_fingerprint: Dict[str, Any] = {}
        try:
            if style_fingerprint_path.exists():
                with open(style_fingerprint_path, "r", encoding="utf-8") as f:
                    self.style_fingerprint = json.load(f)
                print(f"PhasePlanner: Loaded style fingerprint from {style_fingerprint_path}")
            else:
                print(f"Warning: Style fingerprint file not found at {style_fingerprint_path}. Using empty fingerprint.")
        except json.JSONDecodeError as e:
            print(f"Warning: Error decoding style fingerprint JSON from {style_fingerprint_path}: {e}. Using empty fingerprint.")
        except Exception as e_gen:
             print(f"Warning: Could not load style fingerprint from {style_fingerprint_path} due to: {e_gen}. Using empty fingerprint.")

        self.digester = digester
        self.scorer_model_config = scorer_model_config # Store config
        self.refactor_op_map: Dict[str, BaseRefactorOperation] = refactor_op_map if refactor_op_map is not None else REFACTOR_OPERATION_INSTANCES

        self.beam_width = beam_width
        if self.beam_width <= 0:
            raise ValueError("Beam width must be a positive integer.")

        self.phi2_scorer = None
        if self.scorer_model_config and self.scorer_model_config.get("enabled", True):
            try:
                model_name = self.scorer_model_config.get("model_name", "microsoft/phi-2")
                device_setting = self.scorer_model_config.get("device", "cpu")
                self.phi2_scorer = "mock_phi2_scorer_initialized"
                print(f"Info: LLM scorer mock-initialized for planner. (Config: model='{model_name}', device='{device_setting}')")
            except Exception as e:
                print(f"Warning: Failed to initialize LLM scorer (mock setup): {e}. Falling back to random scoring.")
                self.phi2_scorer = None
        else:
            print("Info: LLM scorer is disabled or no config provided for planner. Falling back to random scoring.")
            self.phi2_scorer = None

        self.plan_cache: Dict[str, List[Phase]] = {}
        # self.phi2_scorer_cache: Dict[str, float] = {} # Optional cache

    def _get_spec_cache_key(self, spec: Spec) -> str:
        # Ensure Spec model is Pydantic v2 compatible for model_dump_json if that's used.
        # Fallback to dict if model_dump_json is not available (e.g. placeholder Spec)
        if hasattr(spec, 'model_dump_json'):
            spec_json_str = spec.model_dump_json(sort_keys=True)
        else: # Fallback for placeholder Spec or Pydantic v1
            spec_json_str = json.dumps(spec.dict(sort_keys=True) if hasattr(spec,'dict') else vars(spec), sort_keys=True)
        return hashlib.md5(spec_json_str.encode('utf-8')).hexdigest()

    def _get_graph_statistics(self) -> Dict[str, Any]:
        graph_stats = {
            "num_call_graph_nodes": 0,
            "num_call_graph_edges": 0,
            "cg_density": 0.0,
            "num_cdg_nodes": 0,
            "num_cdg_edges": 0,
            "cdg_density": 0.0,
            "num_ddg_nodes": 0,
            "num_ddg_edges": 0,
            "ddg_density": 0.0,
        }

        # Call Graph (CG) statistics
        if hasattr(self.digester, 'project_call_graph') and self.digester.project_call_graph is not None:
            cg_nodes = self.digester.project_call_graph
            graph_stats["num_call_graph_nodes"] = len(cg_nodes)
            graph_stats["num_call_graph_edges"] = sum(len(edges) for edges in cg_nodes.values())
            if graph_stats["num_call_graph_nodes"] > 1:
                v = graph_stats["num_call_graph_nodes"]
                e = graph_stats["num_call_graph_edges"]
                graph_stats["cg_density"] = e / (v * (v - 1))
            else:
                graph_stats["cg_density"] = 0.0

        # Control Dependence Graph (CDG) statistics
        if hasattr(self.digester, 'project_control_dependence_graph') and self.digester.project_control_dependence_graph is not None:
            cdg_nodes = self.digester.project_control_dependence_graph
            graph_stats["num_cdg_nodes"] = len(cdg_nodes)
            graph_stats["num_cdg_edges"] = sum(len(edges) for edges in cdg_nodes.values()) # Assuming similar structure to CG
            if graph_stats["num_cdg_nodes"] > 1:
                v = graph_stats["num_cdg_nodes"]
                e = graph_stats["num_cdg_edges"]
                graph_stats["cdg_density"] = e / (v * (v - 1))
            else:
                graph_stats["cdg_density"] = 0.0

        # Data Dependence Graph (DDG) statistics
        if hasattr(self.digester, 'project_data_dependence_graph') and self.digester.project_data_dependence_graph is not None:
            ddg_nodes = self.digester.project_data_dependence_graph
            graph_stats["num_ddg_nodes"] = len(ddg_nodes)
            graph_stats["num_ddg_edges"] = sum(len(edges) for edges in ddg_nodes.values()) # Assuming similar structure to CG
            if graph_stats["num_ddg_nodes"] > 1:
                v = graph_stats["num_ddg_nodes"]
                e = graph_stats["num_ddg_edges"]
                graph_stats["ddg_density"] = e / (v * (v - 1))
            else:
                graph_stats["ddg_density"] = 0.0

        # Ensure all keys are present even if graphs are missing
        final_stats = {
            "num_call_graph_nodes": graph_stats.get("num_call_graph_nodes", 0),
            "num_call_graph_edges": graph_stats.get("num_call_graph_edges", 0),
            "cg_density": graph_stats.get("cg_density", 0.0),
            "num_cdg_nodes": graph_stats.get("num_cdg_nodes", 0),
            "num_cdg_edges": graph_stats.get("num_cdg_edges", 0),
            "cdg_density": graph_stats.get("cdg_density", 0.0),
            "num_ddg_nodes": graph_stats.get("num_ddg_nodes", 0),
            "num_ddg_edges": graph_stats.get("num_ddg_edges", 0),
            "ddg_density": graph_stats.get("ddg_density", 0.0),
        }

        print(f"PhasePlanner: Generated graph stats for scorer: {final_stats}")
        return final_stats

    def _score_candidate_plan_with_llm(self, candidate_plan: List['Phase'], graph_stats: Dict[str, Any]) -> float:
        if not self.phi2_scorer or self.phi2_scorer != "mock_phi2_scorer_initialized":
            return random.uniform(0.2, 0.8)

        if not candidate_plan:
            return 0.05

        plan_str_parts = []
        for i, phase in enumerate(candidate_plan):
            try:
                params_json = json.dumps(phase.parameters)
            except TypeError:
                params_json = str(phase.parameters)

            plan_str_parts.append(
                f"Phase {i + 1} (Operation: {phase.operation_name}):
"
                f"  Target File: {phase.target_file or 'N/A'}
"
                f"  Parameters: {params_json}
"
                f"  Description: {phase.description}"
            )
        plan_formatted_str = "\n\n".join(plan_str_parts)

        try:
            style_fp_str = json.dumps(self.style_fingerprint, indent=2)
        except TypeError:
            style_fp_str = str(self.style_fingerprint)

        try:
            graph_stats_str = json.dumps(graph_stats, indent=2)
        except TypeError:
            graph_stats_str = str(graph_stats)

        prompt = f"""You are an expert software engineering assistant. Your task is to evaluate a proposed refactoring plan for a Python codebase.
A higher score indicates a better plan. Consider the plan's clarity, correctness, potential risks, efficiency, and how well it seems to adhere to common best practices and the described project style.

Project Style Fingerprint:
---
{style_fp_str}
---

Codebase Graph Statistics (for context):
---
{graph_stats_str}
---

Candidate Refactoring Plan (Total phases: {len(candidate_plan)}):
---
{plan_formatted_str}
---

Based on all the above information, score this plan on a continuous scale from 0.0 (very bad) to 1.0 (excellent).
Output only the numerical score as a float (e.g., 0.75).
Score:"""

        mock_llm_response_text = ""
        try:
            score_value = 0.5
            score_value += len(candidate_plan) * 0.03
            if candidate_plan: # Check if plan is not empty
                 score_value -= candidate_plan[0].operation_name.count('error') * 0.1

            if any("extract_method" in p.operation_name for p in candidate_plan):
                score_value += 0.05
            if len(candidate_plan) > 7:
                score_value -= (len(candidate_plan) - 7) * 0.02

            score_value += random.uniform(-0.02, 0.02)
            final_mock_score = max(0.0, min(1.0, score_value))
            mock_llm_response_text = f"{final_mock_score:.2f}"
        except Exception as e:
            print(f"Error during Mock LLM scoring simulation: {e}. Falling back to random score.")
            return random.uniform(0.1, 0.5)

        try:
            match = re.search(r"(0?\.\d+|1\.0|1)", mock_llm_response_text) # Corrected regex
            parsed_score = -1.0
            if match:
                parsed_score = float(match.group(1))
            else:
                general_match = re.search(r"[-+]?\d*\.\d+", mock_llm_response_text)
                if general_match:
                    parsed_score = float(general_match.group(0))

            if 0.0 <= parsed_score <= 1.0:
                return parsed_score
            else:
                print(f"Warning: LLM mock score {parsed_score} out of range [0.0, 1.0] from '{mock_llm_response_text}'. Using default.")
                return 0.1
        except ValueError:
            print(f"Warning: ValueError parsing score from LLM mock response: '{mock_llm_response_text}'. Using default.")
            return 0.1
        except Exception as e:
            print(f"Warning: Unexpected error parsing score: {e}. Raw: '{mock_llm_response_text}'. Using default.")
            return 0.1

    def _beam_search_for_plan(self, spec: Spec, graph_stats: Dict[str, Any]) -> List[Phase]:
        # Initial beam: list of (plan_phases, score)
        beam: List[Tuple[List[Phase], float]] = [([], 0.0)]

        if not spec.operations:
            return []

        # Iterate through each operation defined in the spec
        for op_idx, op_spec_item in enumerate(spec.operations):
            next_beam_candidates: List[Tuple[List[Phase], float]] = []

            op_name = op_spec_item.get("name")
            if not op_name or not isinstance(op_name, str): # Ensure op_name is a string
                print(f"Warning: Spec operation item at index {op_idx} is missing a 'name' or 'name' is not a string. Skipping this item.")
                continue

            operation_instance = self.refactor_op_map.get(op_name)
            if not operation_instance:
                print(f"Warning: Operation '{op_name}' (spec index {op_idx}) not found in refactor grammar. Skipping this item.")
                continue

            phase_parameters = {k: v for k, v in op_spec_item.items() if k not in ["name", "target_file"]}
            target_file_for_phase = op_spec_item.get("target_file")
            if target_file_for_phase is not None and not isinstance(target_file_for_phase, str): # Ensure target_file is str or None
                print(f"Warning: 'target_file' for operation '{op_name}' (spec index {op_idx}) is not a string: {target_file_for_phase}. Treating as None.")
                target_file_for_phase = None


            if not operation_instance.validate_parameters(phase_parameters):
                print(f"Warning: Parameters for operation '{op_name}' (spec index {op_idx}, target: {target_file_for_phase or 'repo-level'}) are invalid. This operation will not be added to plans in the current beam step.")
                continue

            # For each current plan in the beam, try to extend it with the current operation
            for current_plan_phases, current_plan_score in beam:
                new_phase = Phase(
                    operation_name=op_name,
                    target_file=target_file_for_phase,
                    parameters=phase_parameters,
                    description=f"Op {op_idx + 1}/{len(spec.operations)}: {operation_instance.description} for '{op_name}' on '{target_file_for_phase or 'repo-level'}'"
                )

                extended_plan_phases = current_plan_phases + [new_phase]
                extended_score = self._score_candidate_plan_with_llm(extended_plan_phases, graph_stats)
                next_beam_candidates.append((extended_plan_phases, extended_score))

            if not next_beam_candidates:
                if not beam:
                    print(f"Warning: Beam became empty while processing spec operation index {op_idx} ('{op_name}').")
                    return []
                # If beam was not empty, but no candidates for this op_spec_item (e.g. invalid params made us 'continue'),
                # the current 'beam' (from previous step) will be used for the next op_spec_item.
                # This is implicitly handled as 'beam' is only updated if next_beam_candidates is non-empty.
                print(f"Info: No valid new phases generated for spec operation index {op_idx} ('{op_name}'). Current beam carries over.")

            if next_beam_candidates:
                next_beam_candidates.sort(key=lambda x: x[1], reverse=True)
                beam = next_beam_candidates[:self.beam_width]

            if not beam:
                print(f"Warning: Beam search resulted in an empty beam after processing spec operation index {op_idx} ('{op_name}').")
                return []

        if not beam:
            print("Warning: Beam search did not find any valid plan.")
            return []

        best_plan_phases, best_score = beam[0]
        num_phases_in_spec = len(spec.operations)
        num_phases_in_plan = len(best_plan_phases)

        print(f"Beam search completed. Best plan score: {best_score:.4f}. Spec ops: {num_phases_in_spec}, Plan phases: {num_phases_in_plan}.")
        if num_phases_in_plan < num_phases_in_spec:
            print(f"Warning: The generated plan has fewer phases ({num_phases_in_plan}) than specified operations ({num_phases_in_spec}). This might be due to invalid spec items.")

        return best_plan_phases

    def plan_phases(self, spec: Spec) -> List[Phase]:
        print(f"PhasePlanner: Received spec for task: '{spec.issue_description[:50]}...'")
        cache_key = self._get_spec_cache_key(spec)
        if cache_key in self.plan_cache:
            print("PhasePlanner: Returning cached plan.")
            return self.plan_cache[cache_key]

        print("PhasePlanner: No cached plan found. Generating new plan.")
        graph_stats = self._get_graph_statistics()
        best_plan = self._beam_search_for_plan(spec, graph_stats)

        if best_plan:
            self.plan_cache[cache_key] = best_plan
            print("PhasePlanner: Plan generated and cached.")
        else:
            print("PhasePlanner: Failed to generate a plan.")
        return best_plan

if __name__ == '__main__':
    from unittest.mock import MagicMock # For __main__ example
    print("--- PhasePlanner Example Usage (Conceptual) ---")

    class MockDigesterForPlannerMain:
        def __init__(self):
            self.project_call_graph = {"module.func_a": {"module.func_b"}}

    dummy_digester_main = MockDigesterForPlannerMain()

    dummy_style_path_main = Path("_temp_dummy_style_main.json")
    with open(dummy_style_path_main, "w") as f:
        json.dump({"line_length": 99, "preferred_quotes": "double"}, f)

    dummy_scorer_config_main = {"model_path": "path/to/phi2/model_placeholder"}

    try:
        if 'Spec' not in globals() or not hasattr(Spec, 'model_fields') or \
           'Phase' not in globals() or not hasattr(Phase, 'model_fields'):
             raise ImportError("Spec or Phase Pydantic models not correctly defined for __main__.")

        planner = PhasePlanner(
            style_fingerprint_path=dummy_style_path_main,
            digester=dummy_digester_main, # type: ignore
            scorer_model_config=dummy_scorer_config_main, # Pass the config
            beam_width=2
        )

        # Example Spec using the imported spec_model.Spec structure
        example_spec_data_main = {
            "issue_description": "Implement new user registration flow", # Updated field name
            "target_files": ["src/services/user_service.py", "src/db/user_queries.py"], # Updated field name
            "operations": [ # This now matches spec_model.Spec's List[Dict[str,Any]]
                {"name": "add_import", "target_file": "src/services/user_service.py", "import_statement": "from ..db import user_queries"},
                {"name": "extract_method", "target_file": "src/services/user_service.py", "source_function_name": "register_user_v1", "start_line": 10, "end_line": 25, "new_method_name": "_validate_user_input"},
                {"name": "add_decorator", "target_file": "src/services/user_service.py", "target_function_name": "register_user_v2", "decorator_name": "@transactional"}
            ],
            "acceptance_tests": ["test_user_registration_success", "test_user_registration_existing_user"] # Updated field name
        }
        spec_instance_main = Spec(**example_spec_data_main)

        print(f"\nPlanning for spec: {spec_instance_main.issue_description}")
        plan_phases_list_main = planner.plan_phases(spec_instance_main)

        print("\nGenerated Plan Phases:")
        if plan_phases_list_main:
            for p_main in plan_phases_list_main:
                if hasattr(p_main, 'model_dump_json'):
                    print(p_main.model_dump_json(indent=2))
                else:
                    print(f"  Phase ID: {p_main.id}, Op: {p_main.operation}, Target: {p_main.target}, Payload: {p_main.payload}")
        else:
            print("  No plan generated.")

        print(f"\nRequesting plan again for the same spec (should be cached):")
        _ = planner.plan_phases(spec_instance_main)

    except Exception as e_main:
        print(f"Error in PhasePlanner __main__ example: {type(e_main).__name__}: {e_main}")
    finally:
        if dummy_style_path_main.exists():
            dummy_style_path_main.unlink()
    print("\n--- PhasePlanner Example Done ---")
