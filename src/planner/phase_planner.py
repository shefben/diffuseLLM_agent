# src/planner/phase_planner.py
import difflib # For generating diff summaries
from typing import List, Dict, Any, Union, Optional, TYPE_CHECKING, Type, Tuple, Callable
from pathlib import Path
import json
import hashlib
import random # For dummy scorer and branch names

from src.transformer import apply_libcst_codemod_script, PatchApplicationError # For real patch application

# Forward references for type hints if full imports are problematic
if TYPE_CHECKING:
    from ..digester.repository_digester import RepositoryDigester
    # Spec, Phase, BaseRefactorOperation are now directly imported.
    # REFACTOR_OPERATION_INSTANCES is used directly.

# Actual imports
from src.utils.config_loader import DEFAULT_APP_CONFIG # For default model paths
from src.spec_normalizer.spec_normalizer_interface import SpecNormalizerModelInterface
from src.spec_normalizer.t5_client import T5Client as T5ClientForSpecNormalization # Alias if T5Client is imported elsewhere
from src.spec_normalizer.diffusion_spec_normalizer import DiffusionSpecNormalizer
from .spec_model import Spec
from .phase_model import Phase
from .refactor_grammar import BaseRefactorOperation, REFACTOR_OPERATION_INSTANCES

# Child component imports
from src.learning.core_predictor import CorePredictor # Added import
# T5Client is imported for spec normalization above, ensure no conflict if used for other purposes.
# If T5Client is used for other things, the alias T5ClientForSpecNormalization should be used for spec.
# For now, assume T5Client is primarily for spec normalization context here.
from src.profiler.t5_client import T5Client # Assuming direct import is fine - this is potentially the same as T5ClientForSpecNormalization
from src.retriever.symbol_retriever import SymbolRetriever # Assuming direct import
from src.spec_normalizer.spec_fusion import SpecFusion # Assuming direct import
from src.validator.validator import Validator
from src.builder.commit_builder import CommitBuilder
from src.agent_group.collaborative_agent_group import CollaborativeAgentGroup

# Other necessary imports
from src.agent_group.exceptions import PhaseFailure
from src.profiler.llm_interfacer import get_llm_score_for_text
from src.utils.memory_logger import log_successful_patch # For Success Memory Logging

# REFACTOR_OPERATION_CLASSES is not directly used by PhasePlanner logic, REFACTOR_OPERATION_INSTANCES is.

# Fallback for RepositoryDigester if not available (e.g. in isolated subtask)
# This is primarily for type hinting if RepositoryDigester cannot be imported directly.
# The actual instance is passed to __init__.
if TYPE_CHECKING:
    # Ensure Phase is available for type hints if not already.
    from .phase_model import Phase

if TYPE_CHECKING or 'RepositoryDigester' not in globals(): # Keep this for RepositoryDigester hint
    class RepositoryDigester: # type: ignore
        project_call_graph: Dict
        project_control_dependence_graph: Optional[Dict] = None
        project_data_dependence_graph: Optional[Dict] = None
        # Add methods expected by CollaborativeAgentGroup.run context_data population
        def get_project_overview(self) -> Dict[str, Any]:
            print("MockDigester.get_project_overview called")
            return {"mock_overview": True}
        def get_file_content(self, path: Path) -> Optional[str]:
            print(f"MockDigester.get_file_content called for {path}")
            return f"# Mock content for {path}" if path else None
        pass # Ensure class body is not empty


class PhasePlanner:
    # mock_validator_handle is now removed.
    # The actual validator instance's validate_patch method will be used.

    @staticmethod
    def mock_score_style_handle(patch: Any, style_profile: Dict[str, Any]) -> float:
        print(f"PhasePlanner.mock_score_style_handle: Scoring style for patch: {str(patch)[:100]} with profile keys: {list(style_profile.keys())}")
        # Assuming patch is now a script string, this mock might need adjustment if it were to inspect content.
        # For now, it's a simple mock.
        if isinstance(patch, str) and "BAD_STYLE_MARKER_IN_SCRIPT" in patch:
            return 0.3 # Low score for bad style
        return 0.9 # Default high score

    def __init__(
        self,
        project_root_path: Path,
        app_config: Dict[str, Any],
        digester: 'RepositoryDigester',
        refactor_op_map: Optional[Dict[str, BaseRefactorOperation]] = None,
        # beam_width is now primarily from app_config, but can be overridden if passed for testing
        beam_width: Optional[int] = None
    ):
        """
        Initializes the PhasePlanner.

        Args:
            project_root_path: The root path of the project being worked on.
            app_config: The application configuration dictionary.
            digester: An instance of RepositoryDigester for the project.
            refactor_op_map: Optional map of refactoring operations.
            beam_width: Optional override for beam width in plan search.
        """
        self.project_root_path = project_root_path
        self.app_config = app_config
        self.digester = digester
        self.verbose = self.app_config.get("general", {}).get("verbose", False)

        # Load/Determine Paths & Style Fingerprint
        style_fp_path = self.project_root_path / "style_fingerprint.json" # Standard name
        self.style_fingerprint: Dict[str, Any] = {}
        try:
            if style_fp_path.exists():
                with open(style_fp_path, "r", encoding="utf-8") as f:
                    self.style_fingerprint = json.load(f)
                if self.verbose: print(f"PhasePlanner: Loaded style fingerprint from {style_fp_path}")
            else:
                if self.verbose: print(f"PhasePlanner Warning: Style fingerprint file not found at {style_fp_path}. Using empty fingerprint.")
        except json.JSONDecodeError as e:
            print(f"PhasePlanner Warning: Error decoding style fingerprint JSON from {style_fp_path}: {e}. Using empty fingerprint.")
        except Exception as e_gen:
             print(f"PhasePlanner Warning: Could not load style fingerprint from {style_fp_path} due to: {e_gen}. Using empty fingerprint.")

        self.naming_conventions_db_path = self.project_root_path / "naming_conventions.db" # Standard name
        self.data_dir_path = Path(self.app_config.get("general", {}).get("data_dir", str(self.project_root_path / ".agent_data")))
        self.data_dir_path.mkdir(parents=True, exist_ok=True) # Ensure data_dir exists
        if self.verbose: print(f"PhasePlanner: Data directory for logs (e.g. success memory): {self.data_dir_path}")

        # Configure self.scorer_model_config (for internal use by _score_candidate_plan_with_llm)
        llm_params_config = self.app_config.get("llm_params", {})
        scorer_model_p = self.app_config.get("models", {}).get("planner_scorer_gguf", DEFAULT_APP_CONFIG["models"]["planner_scorer_gguf"])

        self.scorer_model_config = {
            "model_path": str(Path(scorer_model_p).resolve()), # Resolve path
            "enabled": True, # Assuming enabled if path is provided; can add specific config key later
            "verbose": self.verbose,
            "n_gpu_layers": llm_params_config.get("n_gpu_layers_default", DEFAULT_APP_CONFIG["llm_params"]["n_gpu_layers_default"]),
            "n_ctx": llm_params_config.get("n_ctx_default", DEFAULT_APP_CONFIG["llm_params"]["n_ctx_default"]),
            "temperature": llm_params_config.get("plan_score_temp", llm_params_config.get("temperature_default", DEFAULT_APP_CONFIG["llm_params"]["temperature_default"])),
            "max_tokens": llm_params_config.get("plan_score_max_tokens", 16) # Renamed from max_tokens_for_score
        }
        if not Path(self.scorer_model_config["model_path"]).is_file():
            print(f"PhasePlanner Warning: Scorer model not found at {self.scorer_model_config['model_path']}. Plan scoring will use fallback.")
            self.scorer_model_config["enabled"] = False
        else:
            if self.verbose: print(f"PhasePlanner: Real LLM scorer configured with model: {self.scorer_model_config['model_path']}")

        self.refactor_op_map: Dict[str, BaseRefactorOperation] = refactor_op_map if refactor_op_map is not None else REFACTOR_OPERATION_INSTANCES
        self.beam_width = beam_width if beam_width is not None else self.app_config.get("planner",{}).get("beam_width", DEFAULT_APP_CONFIG["planner"]["beam_width"])
        if self.beam_width <= 0:
            raise ValueError("Beam width must be a positive integer.")
        if self.verbose: print(f"PhasePlanner: Beam width set to {self.beam_width}")

        # Instantiate Child Components (Passing app_config or derived configs)

        # Spec Normalizer Selection Logic
        spec_normalizer_type = self.app_config.get("spec_normalizer", {}).get("type", "t5") # Default to t5
        spec_model_instance: SpecNormalizerModelInterface

        if spec_normalizer_type.lower() == "diffusion":
            if self.verbose:
                print("PhasePlanner: Using DiffusionSpecNormalizer for spec normalization.")
            spec_model_instance = DiffusionSpecNormalizer(app_config=self.app_config)
        elif spec_normalizer_type.lower() == "t5":
            if self.verbose:
                print("PhasePlanner: Using T5Client for spec normalization.")
            # Ensure T5ClientForSpecNormalization or a T5Client instance is used.
            # Using T5Client directly, assuming it's the one intended for spec normalization.
            spec_model_instance = T5Client(app_config=self.app_config)
        else:
            if self.verbose:
                print(f"PhasePlanner Warning: Unknown spec_normalizer type '{spec_normalizer_type}'. Defaulting to T5Client.")
            spec_model_instance = T5Client(app_config=self.app_config)

        # self.t5_client = T5Client(app_config=self.app_config) # Removed as spec_model_instance replaces its role for SpecFusion
        self.symbol_retriever = SymbolRetriever(digester=self.digester, app_config=self.app_config)
        self.spec_normalizer = SpecFusion(spec_model_interface=spec_model_instance, symbol_retriever=self.symbol_retriever, app_config=self.app_config)
        self.validator = Validator(app_config=self.app_config)
        if self.verbose: print("PhasePlanner: Validator instance initialized.")

        # Agent Group requires specific model paths from app_config
        agent_llm_model_p = self.app_config.get("models", {}).get("agent_llm_gguf", DEFAULT_APP_CONFIG["models"]["agent_llm_gguf"])
        # CollaborativeAgentGroup __init__ currently takes llm_model_path and specific core_configs
        # For now, we pass app_config and let it derive, or pass specific paths as before if its __init__ is not yet updated.
        # Based on previous structure of CollaborativeAgentGroup, it took llm_core_config, diffusion_core_config, and llm_model_path.
        # We will pass app_config to it, and it should become responsible for deriving these.
        # For the purpose of this subtask, we assume CollaborativeAgentGroup will be updated to take app_config.
        self.agent_group: 'CollaborativeAgentGroup' = CollaborativeAgentGroup(
            app_config=self.app_config, # Pass the full app_config
            digester=self.digester, # Pass digester instance
            style_profile=self.style_fingerprint,
            naming_conventions_db_path=self.naming_conventions_db_path,
            validator_instance=self.validator
            # Old parameters like llm_core_config, diffusion_core_config, llm_model_path
            # are assumed to be handled by CollaborativeAgentGroup internally using app_config.
        )
        if self.verbose: print(f"PhasePlanner: CollaborativeAgentGroup initialized.")

        # Store actual validator handle and mock score style handle
        self.validator_handle: Callable[[Optional[str], str, 'RepositoryDigester', Path], Tuple[bool, Optional[str]]] = self.validator.validate_patch
        self.score_style_handle: Callable[[Any, Dict[str, Any]], float] = PhasePlanner.mock_score_style_handle
        if self.verbose: print("PhasePlanner: Validator handle set. Mock style scorer configured.")

        # Initialize CommitBuilder
        self.commit_builder = CommitBuilder(app_config=self.app_config)
        if self.verbose: print("PhasePlanner: CommitBuilder instance initialized.")

        self.plan_cache: Dict[str, List[Phase]] = {}
        # self.phi2_scorer_cache: Dict[str, float] = {} # Optional cache

        # Determine CorePredictor model path from app_config or use a default
        core_predictor_model_path_str = self.app_config.get("models", {}).get(
            "core_predictor_model_path",
            str(self.data_dir_path / "core_predictor.joblib") # Defaulting to data_dir now
        )
        # Ensure the default path is used if the config value is None or empty string
        if not core_predictor_model_path_str: # Handle empty string from config
            core_predictor_model_path_str = str(self.data_dir_path / "core_predictor.joblib")

        core_predictor_model_path = Path(core_predictor_model_path_str)

        self.core_predictor = CorePredictor(
            model_path=core_predictor_model_path,
            verbose=self.verbose
        )
        if self.verbose:
            print(f"PhasePlanner: CorePredictor initialized. Model path: {core_predictor_model_path}, Ready: {self.core_predictor.is_ready}")

    def _suggest_next_candidate_operations(
        self,
        remaining_spec_goals: 'Spec',
        graph_stats_for_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Suggests a list of candidate operation names based on the remaining spec goals.
        Parameters (e.g., target_file, specific arguments for the operation)
        will be inferred in a subsequent step.

        Args:
            remaining_spec_goals: The Spec object representing the work yet to be done.
            graph_stats_for_context: Statistics about the codebase's structure, for LLM context.

        Returns:
            A list of dictionaries, each like {"name": "operation_name"}.
        """
        if self.verbose:
            print(f"PhasePlanner._suggest_next_candidate_operations: Called with goals from spec: '{remaining_spec_goals.issue_description[:50]}...'")

        issue_desc = remaining_spec_goals.issue_description
        acceptance_tests_str = "\n- ".join(remaining_spec_goals.acceptance_tests)
        target_files_str = ", ".join(remaining_spec_goals.target_files)

        plan_prefix_summary_str = ""
        if remaining_spec_goals.plan_prefix_summary and isinstance(remaining_spec_goals.plan_prefix_summary, list):
            prefix_items = "\n- ".join(remaining_spec_goals.plan_prefix_summary)
            if prefix_items: # Ensure not just an empty list leading to "Summary... \n- "
                plan_prefix_summary_str = f"\n**Summary of Operations Already in Current Plan:**\n- {prefix_items}\n"

        available_ops_desc_list = []
        for op_name, op_instance in self.refactor_op_map.items():
            available_ops_desc_list.append(f"- {op_name}: {op_instance.description}")
        available_ops_str = "\n".join(available_ops_desc_list)

        prompt = f"""
Given the following software development goal, codebase statistics, and available operations:

**Goal:**
Issue Description: {issue_desc}
Acceptance Tests:
- {acceptance_tests_str}
Target Files: {target_files_str}
{plan_prefix_summary_str}
**Codebase Statistics:**
{json.dumps(graph_stats_for_context, indent=2)}

**Available Operations (and their descriptions):**
{available_ops_str}

Suggest up to 3 most relevant operation *names* from the 'Available Operations' list that should be considered as the *next logical step* or alternative steps in a plan to achieve these goals.
Focus on suggesting operations that directly address the primary unmet needs described in the goal.
Return your response as a JSON list of strings, where each string is an operation name. For example: ["add_function", "generic_code_edit"]
Ensure the operation names are exactly as provided in the 'Available Operations' list.

JSON List of Suggested Operation Names:
"""
        # Note: Using scorer_model_config for this call. Max_tokens might need adjustment.
        # For now, using existing max_tokens from scorer_model_config (likely small, e.g., 16 for scores).
        # This should be increased for generating a list of strings (e.g., to 100-150).
        # This is a temporary measure; a dedicated config for op suggestion might be better.
        llm_output_str_any = get_llm_score_for_text(
            model_path=self.scorer_model_config["model_path"],
            prompt=prompt,
            verbose=self.scorer_model_config.get("verbose", False), # Use verbose from scorer_config
            n_gpu_layers=self.scorer_model_config["n_gpu_layers"],
            n_ctx=self.scorer_model_config["n_ctx"],
            max_tokens_for_score=self.scorer_model_config.get("max_tokens_for_op_suggestion", 150), # Temp use of max_tokens or new one
            temperature=self.scorer_model_config["temperature"]
        )

        llm_output_str = str(llm_output_str_any) if llm_output_str_any is not None else None

        if llm_output_str:
            try:
                # Attempt to clean and parse the JSON list from the LLM output
                # Common LLM outputs might include markdown ```json ... ``` or just the list.
                match = re.search(r"\[.*?\]", llm_output_str, re.DOTALL)
                if match:
                    json_str_from_llm = match.group(0)
                    suggested_op_names = json.loads(json_str_from_llm)
                    if isinstance(suggested_op_names, list):
                        valid_suggestions = []
                        for name in suggested_op_names:
                            if isinstance(name, str) and name in self.refactor_op_map:
                                valid_suggestions.append({"name": name})

                        if valid_suggestions:
                            if self.verbose: print(f"PhasePlanner._suggest_next_candidate_operations: LLM suggested and validated: {[s['name'] for s in valid_suggestions]}")
                            return valid_suggestions
                        elif self.verbose:
                            print(f"PhasePlanner Warning: LLM suggested ops, but none were valid: {suggested_op_names}")
                    elif self.verbose:
                        print(f"PhasePlanner Warning: LLM output parsed as JSON, but not a list: {suggested_op_names}")
                elif self.verbose:
                    print(f"PhasePlanner Warning: Could not find JSON list in LLM output: '{llm_output_str}'")

            except json.JSONDecodeError as e:
                if self.verbose: print(f"PhasePlanner Warning: Failed to parse LLM output as JSON: {e}. Output: '{llm_output_str}'")
            except Exception as e_parse: # Catch other potential errors during parsing/processing
                 if self.verbose: print(f"PhasePlanner Warning: Error processing LLM output: {e_parse}. Output: '{llm_output_str}'")
        else:
            if self.verbose: print("PhasePlanner Warning: LLM call for operation suggestion returned None or empty string.")

        # Fallback logic
        if self.verbose: print("PhasePlanner: Using fallback logic for operation suggestions.")
        if "generic_code_edit" in self.refactor_op_map:
            return [{"name": "generic_code_edit"}]
        elif self.refactor_op_map:
            return [{"name": list(self.refactor_op_map.keys())[0]}]
        return []

    def _infer_parameters_for_operation(
        self,
        operation_name: str,
        spec: 'Spec',
        graph_stats_for_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Infers parameters for a given operation name based on the spec and context.
        Currently, this is a placeholder. Future versions will use an LLM.

        Args:
            operation_name: The name of the operation for which to infer parameters.
            spec: The overall specification object, providing context.
            graph_stats_for_context: Codebase graph statistics for context.

        Returns:
            A dictionary of inferred parameters, or None if inference fails or
            the operation is not found.
        """
        if self.verbose:
            self.logger.info(f"PhasePlanner._infer_parameters_for_operation: Called for op '{operation_name}' with spec '{spec.issue_description[:50]}...'")

        operation_instance = self.refactor_op_map.get(operation_name)
        if not operation_instance:
            if self.verbose:
                self.logger.warning(f"Operation '{operation_name}' not found in refactor_op_map. Cannot infer parameters.")
            return None

        if not operation_instance.required_parameters:
            if self.verbose:
                self.logger.info(f"Operation '{operation_name}' requires no parameters. Returning empty dict.")
            return {} # No parameters to infer

        issue_desc = spec.issue_description
        acceptance_tests_str = "\n- ".join(spec.acceptance_tests)
        target_files_str = ", ".join(spec.target_files)

        plan_prefix_summary_str = ""
        # spec here is current_remaining_spec_goals from _beam_search_for_plan
        if spec.plan_prefix_summary and isinstance(spec.plan_prefix_summary, list):
            prefix_items = "\n- ".join(spec.plan_prefix_summary)
            if prefix_items:
                plan_prefix_summary_str = f"\n**Summary of Preceding Planned Operations:**\n- {prefix_items}\n"

        # Prepare a string list of required parameters and their descriptions (if available)
        params_desc_list = []
        for param_name in operation_instance.required_parameters:
            # Assuming parameters might have descriptions in the future, e.g. operation_instance.parameter_descriptions[param_name]
            params_desc_list.append(f"- {param_name}") # Placeholder for future richer descriptions
        required_params_str = "\n".join(params_desc_list)

        prompt = f"""
Given the software development task described by the specification and an operation to be performed:

**Overall Task Specification:**
Issue Description: {issue_desc}
Acceptance Tests:
- {acceptance_tests_str}
Target Files: {target_files_str}
{plan_prefix_summary_str}
**Current Operation to Parameterize:**
Operation Name: {operation_name}
Operation Description: {operation_instance.description}
Required Parameters for this Operation:
{required_params_str}

**Codebase Context (Statistics):**
{json.dumps(graph_stats_for_context, indent=2)}

Your goal is to infer appropriate values for ONLY the 'Required Parameters for this Operation'.
Consider the overall task, the operation's purpose, and the codebase context.
Parameter values should be strings or simple JSON-compatible types.
For file paths, provide relative paths from the project root if appropriate, or fully qualified names for symbols.
If a parameter refers to a line number, it should be an integer.

Return your response as a single JSON object where keys are the required parameter names and values are their inferred values.
Example for an 'add_function' operation:
{{
  "target_file": "src/utils/helpers.py",
  "function_name": "calculate_total_price",
  "function_signature": "def calculate_total_price(items: List[Item], tax_rate: float) -> float:",
  "function_body": "  total = sum(item.price for item in items)\\n  return total * (1 + tax_rate)"
}}

JSON Object of Inferred Parameter Values:
"""
        max_tokens_for_param_inference = self.app_config.get("llm_params", {}).get(
            "max_tokens_for_param_inference",
            DEFAULT_APP_CONFIG["llm_params"].get("max_tokens_for_param_inference", 250) # Default from defaults if not in main config
        )

        llm_output_str_any = get_llm_score_for_text( # Reusing this function, 'score' is a misnomer here
            model_path=self.scorer_model_config["model_path"], # Consider dedicated model for this later
            prompt=prompt,
            verbose=self.scorer_model_config.get("verbose", False),
            n_gpu_layers=self.scorer_model_config["n_gpu_layers"],
            n_ctx=self.scorer_model_config["n_ctx"],
            max_tokens_for_score=max_tokens_for_param_inference, # Using the dedicated max_tokens value
            temperature=self.scorer_model_config.get("temperature_for_generation",
                                                   self.scorer_model_config.get("temperature", 0.5)) # Potentially different temp
        )

        llm_output_str = str(llm_output_str_any) if llm_output_str_any is not None else None

        if not llm_output_str:
            if self.verbose:
                self.logger.warning(f"LLM call for parameter inference for '{operation_name}' returned empty. Cannot infer parameters.")
            return None

        try:
            # Try to find JSON object within the LLM output
            match = re.search(r"\{.*\}", llm_output_str, re.DOTALL)
            if not match:
                if self.verbose:
                    self.logger.warning(f"No JSON object found in LLM output for '{operation_name}'. Output: '{llm_output_str}'")
                return None

            json_str_from_llm = match.group(0)
            inferred_params = json.loads(json_str_from_llm)

            if not isinstance(inferred_params, dict):
                if self.verbose:
                    self.logger.warning(f"LLM output for '{operation_name}' parsed as JSON, but not a dictionary: {inferred_params}")
                return None

            # Validate that all required parameters are present in the inferred_params
            missing_params = [
                p_name for p_name in operation_instance.required_parameters if p_name not in inferred_params
            ]
            if missing_params:
                if self.verbose:
                    self.logger.warning(f"LLM output for '{operation_name}' is missing required parameters: {missing_params}. Inferred: {inferred_params}")
                return None

            if self.verbose:
                self.logger.info(f"Successfully inferred parameters for '{operation_name}': {inferred_params}")
            return inferred_params

        except json.JSONDecodeError as e:
            if self.verbose:
                self.logger.error(f"Failed to parse LLM output as JSON for '{operation_name}': {e}. Output: '{llm_output_str}'")
            return None
        except Exception as e_gen:
            if self.verbose:
                self.logger.error(f"Unexpected error during parameter inference for '{operation_name}': {e_gen}. Output: '{llm_output_str}'", exc_info=True)
            return None

    def generate_plan_from_spec(self, spec: Spec) -> List[Phase]:
        """
        Generates a plan (a list of Phase objects) from a given Spec object.
        This method includes caching, graph statistics retrieval, and beam search for plan generation.
        It does not execute the plan.
        """
        print(f"PhasePlanner: Received spec for task: '{spec.issue_description[:50]}...' for plan generation.")
        cache_key = self._get_spec_cache_key(spec)
        if cache_key in self.plan_cache:
            print("PhasePlanner: Returning cached plan.")
            return self.plan_cache[cache_key]

        print("PhasePlanner: No cached plan found. Generating new plan.")
        graph_stats = self._get_graph_statistics()
        best_plan = self._beam_search_for_plan(spec, graph_stats)

        if best_plan:
            self.plan_cache[cache_key] = best_plan # Cache the plan (list of Phase objects)
            print("PhasePlanner: Plan generated and cached.")
        else:
            print("PhasePlanner: Failed to generate a plan (beam search returned empty).")

        return best_plan

    def _get_spec_cache_key(self, spec: Spec) -> str:
        # Ensure Spec model is Pydantic v2 compatible for model_dump_json if that's used.
        # Fallback to dict if model_dump_json is not available (e.g. placeholder Spec)
        if hasattr(spec, 'model_dump_json'):
            spec_json_str = spec.model_dump_json(sort_keys=True)
        else: # Fallback for placeholder Spec or Pydantic v1
            spec_json_str = json.dumps(spec.dict(sort_keys=True) if hasattr(spec,'dict') else vars(spec), sort_keys=True)
        return hashlib.md5(spec_json_str.encode('utf-8')).hexdigest()

    def _get_graph_statistics(self) -> Dict[str, Any]:
        # Initialize stats with defaults
        num_cg_nodes, num_cg_edges, cg_density = 0, 0, 0.0
        num_cdg_nodes, num_cdg_edges, cdg_density = 0, 0, 0.0
        num_ddg_nodes, num_ddg_edges, ddg_density = 0, 0, 0.0

        # Call Graph (CG) statistics
        if hasattr(self.digester, 'project_call_graph') and self.digester.project_call_graph:
            cg = self.digester.project_call_graph
            num_cg_nodes = len(cg)
            num_cg_edges = sum(len(edges) for edges in cg.values())
            if num_cg_nodes > 1:
                cg_density = num_cg_edges / (num_cg_nodes * (num_cg_nodes - 1))

        # Control Dependence Graph (CDG) statistics
        if hasattr(self.digester, 'project_control_dependence_graph') and self.digester.project_control_dependence_graph:
            cdg = self.digester.project_control_dependence_graph
            num_cdg_nodes = len(cdg)
            # Assuming CDG values are lists/sets of dependent nodes (edges)
            num_cdg_edges = sum(len(dependents) for dependents in cdg.values())
            if num_cdg_nodes > 1:
                cdg_density = num_cdg_edges / (num_cdg_nodes * (num_cdg_nodes - 1))

        # Data Dependence Graph (DDG) statistics
        if hasattr(self.digester, 'project_data_dependence_graph') and self.digester.project_data_dependence_graph:
            ddg = self.digester.project_data_dependence_graph
            num_ddg_nodes = len(ddg)
            # Assuming DDG values are lists/sets of dependent nodes (edges)
            num_ddg_edges = sum(len(dependents) for dependents in ddg.values())
            if num_ddg_nodes > 1:
                ddg_density = num_ddg_edges / (num_ddg_nodes * (num_ddg_nodes - 1))

        graph_stats = {
            "num_call_graph_nodes": num_cg_nodes,
            "num_call_graph_edges": num_cg_edges,
            "cg_density": cg_density,
            "num_cdg_nodes": num_cdg_nodes,
            "num_cdg_edges": num_cdg_edges,
            "cdg_density": cdg_density,
            "num_ddg_nodes": num_ddg_nodes,
            "num_ddg_edges": num_ddg_edges,
            "ddg_density": ddg_density,
        }

        print(f"PhasePlanner: Generated graph stats for scorer: {graph_stats}")
        return graph_stats

    def _score_candidate_plan_with_llm(self, candidate_plan: List[Phase], graph_stats: Dict[str, Any]) -> float:
        # Check if LLM scoring is enabled and configured
        if not self.scorer_model_config or not self.scorer_model_config.get("enabled", True):
            print("PhasePlanner Info: Real LLM scorer is disabled in config. Falling back to random scoring.")
            return random.uniform(0.2, 0.8)

        scorer_model_path = self.scorer_model_config.get("model_path")
        # Default to a placeholder if path not provided, get_llm_score_for_text will handle actual existence check.
        if not scorer_model_path:
             scorer_model_path = "./models/placeholder_scorer.gguf"
             print(f"PhasePlanner Warning: No 'model_path' in scorer_model_config. Defaulting to placeholder '{scorer_model_path}' for get_llm_score_for_text.")

        # Check if the placeholder path actually exists if it's the one being used.
        # get_llm_score_for_text also does this, but an early check here can provide clearer context for fallback.
        if scorer_model_path == "./models/placeholder_scorer.gguf" and not Path(scorer_model_path).is_file():
            print(f"PhasePlanner Warning: Default scorer model placeholder '{scorer_model_path}' not found. Falling back to random scoring.")
            return random.uniform(0.2, 0.8)

        if not candidate_plan:
            return 0.05 # Return a low score for an empty plan

        # Prepare prompt components (similar to existing logic)
        plan_str_parts = []
        for i, phase in enumerate(candidate_plan):
            try: params_json = json.dumps(phase.parameters)
            except TypeError: params_json = str(phase.parameters)
            plan_str_parts.append(
                f"Phase {i + 1} (Operation: {phase.operation_name}):\n"
                f"  Target File: {phase.target_file or 'N/A'}\n"
                f"  Parameters: {params_json}\n"
                f"  Description: {phase.description}"
            )
        plan_formatted_str = "\n\n".join(plan_str_parts)

        try: style_fp_str = json.dumps(self.style_fingerprint, indent=2)
        except TypeError: style_fp_str = str(self.style_fingerprint)
        try: graph_stats_str = json.dumps(graph_stats, indent=2)
        except TypeError: graph_stats_str = str(graph_stats)

        prompt = f"""You are an expert software engineering assistant. Your task is to evaluate a proposed refactoring plan for a Python codebase.
A higher score indicates a better plan. Please evaluate the plan based on the following criteria:
- Clarity and Specificity: Are the phase descriptions and parameters detailed and unambiguous enough to be clearly actionable?
- Correctness and Completeness: Does the plan seem to correctly address the underlying goal (implied by the sequence of operations)? Does it cover necessary steps?
- Efficiency and Conciseness: Is the plan direct and to the point? Does it avoid unnecessary steps or complexity? Prefer plans with an appropriate number of impactful phases.
- Risk Assessment: Does the plan introduce any obvious risks? Are changes localized and well-contained where possible?
- Adherence to Conventions: While the style fingerprint provides code style, does the plan itself represent a conventional and logical approach to refactoring or feature implementation?

Consider the plan's overall clarity, correctness, potential risks, efficiency, and how well it seems to adhere to common best practices and the described project style.

Project Style Fingerprint (Code Style):
---
{style_fp_str}
---

Codebase Graph Statistics (for additional context on code structure):
---
{graph_stats_str}
---

Candidate Refactoring Plan (Total phases: {len(candidate_plan)}):
---
{plan_formatted_str}
---

Based on all the above information, score this plan on a continuous scale from 0.0 (very bad) to 1.0 (excellent).
Output ONLY the numerical score as a float (e.g., 0.75).
Score:"""

        # Get LLM parameters from config, with defaults
        llm_verbose = self.scorer_model_config.get("verbose", False)
        llm_n_gpu_layers = self.scorer_model_config.get("n_gpu_layers", -1)
        llm_n_ctx = self.scorer_model_config.get("n_ctx", 2048) # Default from get_llm_score_for_text
        llm_temperature = self.scorer_model_config.get("temperature", 0.1) # Default from get_llm_score_for_text
        max_tokens_score = self.scorer_model_config.get("max_tokens_for_score", 16)

        # Call the new LLM interfacer function
        llm_score = get_llm_score_for_text(
            model_path=scorer_model_path,
            prompt=prompt,
            verbose=llm_verbose,
            n_gpu_layers=llm_n_gpu_layers,
            n_ctx=llm_n_ctx,
            max_tokens_for_score=max_tokens_score,
            temperature=llm_temperature
        )

        if llm_score is not None:
            # Optional: Validate or clamp score to expected range [0.0, 1.0]
            if not (0.0 <= llm_score <= 1.0):
                print(f"PhasePlanner Warning: LLM score {llm_score:.4f} is outside the expected [0.0, 1.0] range. Clamping.")
                llm_score = max(0.0, min(1.0, llm_score))
            print(f"PhasePlanner: LLM Scorer returned score: {llm_score:.4f}")
            return llm_score
        else:
            print("PhasePlanner Warning: Failed to get score from LLM via get_llm_score_for_text. Falling back to default low score for this candidate.")
            return 0.1 # Default low score on failure

    def _convert_op_spec_items_to_phases(self, op_spec_items: List[Dict[str, Any]], spec_description: str) -> List[Phase]:
        """Converts a list of operation spec items (dictionaries) to a list of Phase objects."""
        phases = []
        for i, op_spec_item in enumerate(op_spec_items):
            op_name = op_spec_item.get("name", "unknown_operation")
            target_file = op_spec_item.get("target_file")
            parameters = {k: v for k, v in op_spec_item.items() if k not in ["name", "target_file"]}

            # Try to get description from operation instance, fallback to a generic one
            operation_instance = self.refactor_op_map.get(op_name)
            description = operation_instance.description if operation_instance else f"Execute {op_name}"

            phases.append(Phase(
                operation_name=op_name,
                target_file=target_file,
                parameters=parameters,
                # Using a more generic description here, or could pass more context if needed
                description=f"Phase {i+1}/{len(op_spec_items)} (Op: {op_name}): {description} for spec '{spec_description[:30]}...' on '{target_file or 'repo-level'}'"
            ))
        return phases

    def _score_plan(self, plan_op_specs: List[Dict[str, Any]], original_spec: Spec, graph_stats: Dict[str, Any]) -> float:
        """
        Scores a plan represented by a list of operation spec items.
        Converts op specs to Phase objects before calling the LLM scorer.
        """
        if not plan_op_specs:
            return 0.0 # Or some other low score for an empty plan

        # Convert List[Dict[str, Any]] to List[Phase]
        plan_phases = self._convert_op_spec_items_to_phases(plan_op_specs, original_spec.issue_description)

        # Now call the existing LLM scorer that expects List[Phase]
        return self._score_candidate_plan_with_llm(plan_phases, graph_stats)

    def _beam_search_for_plan(
        self,
        spec: Spec,
        graph_stats: Dict[str, Any],
        initial_plan_tuple: Optional[Tuple[List[Dict[str, Any]], float]] = None,
        max_plan_depth_override: Optional[int] = None
    ) -> List[Phase]: # Returns List[Phase]
        """
        Performs a beam search to find the best sequence of operations (plan)
        to address the given specification. The plan is represented as a list of
        operation spec dictionaries internally during search, and converted to
        List[Phase] at the end.

        Args:
            spec: The specification object for the task.
            graph_stats: Codebase graph statistics for context.
            initial_plan_tuple: Optional starting plan (list of op_spec_items) and its score.
            max_plan_depth_override: Optional override for max_plan_depth for this search.

        Returns:
            The best plan found as a list of Phase objects, or an empty list if no plan is found.
        """
        max_plan_depth = max_plan_depth_override if max_plan_depth_override is not None \
            else self.app_config.get("planner", {}).get("max_plan_depth", DEFAULT_APP_CONFIG["planner"]["max_plan_depth"])

        if self.verbose:
            self.logger.info(f"Starting beam search for spec: '{spec.issue_description[:50]}...'. Beam width: {self.beam_width}, Max depth: {max_plan_depth}")

        # Beam stores items as: {'plan': List[Dict[str, Any]], 'score': float, 'remaining_spec_goals': Spec, 'is_complete': bool}
        beam: List[Dict[str, Any]] = []
        initial_remaining_spec_goals = spec

        if initial_plan_tuple:
            # Assuming initial_plan_tuple does not yet have 'is_complete' flag, calculate it.
            initial_plan_list = initial_plan_tuple[0] if isinstance(initial_plan_tuple[0], list) else []
            initial_score = initial_plan_tuple[1]
            # Convert op_specs to phases for the initial check
            initial_phases_for_check = self._convert_op_spec_items_to_phases(initial_plan_list, initial_remaining_spec_goals.issue_description)
            is_initial_plan_complete = self._are_goals_met(initial_phases_for_check, initial_remaining_spec_goals)
            beam = [{'plan': initial_plan_list, 'score': initial_score, 'remaining_spec_goals': initial_remaining_spec_goals, 'is_complete': is_initial_plan_complete}]
        else:
            # Empty plan is not complete.
            beam = [{'plan': [], 'score': self._score_plan([], spec, graph_stats), 'remaining_spec_goals': initial_remaining_spec_goals, 'is_complete': False}]

        # Beam search main loop
        for depth in range(max_plan_depth):
            new_beam_candidates: List[Dict[str, Any]] = []

            if not beam:
                if self.verbose: self.logger.debug(f"Beam search (depth {depth}): Beam is empty, stopping.")
                break

            if self.verbose: self.logger.debug(f"Beam search (depth {depth}): Expanding {len(beam)} current plans in beam.")

            for item_idx, item in enumerate(beam):
                current_plan_op_specs = item['plan'] # This is List[Dict[str, Any]]
                # current_score = item['score'] # Not directly used for extension, but for selection later
                current_remaining_spec_goals = item['remaining_spec_goals']

                if len(current_plan_op_specs) >= max_plan_depth:
                    new_beam_candidates.append(item) # Keep it as is, cannot extend
                    if self.verbose: self.logger.debug(f"  Plan {item_idx+1} (len {len(current_plan_op_specs)}) reached max depth. Keeping.")
                    continue

                # Suggest next candidate operation *names*
                candidate_op_names_list = self._suggest_next_candidate_operations(
                    remaining_spec_goals=current_remaining_spec_goals, # This should evolve
                    graph_stats_for_context=graph_stats
                )
                if self.verbose:
                    self.logger.debug(f"  For plan {item_idx+1} ({self._plan_to_str(current_plan_op_specs)}), suggested ops: {[op['name'] for op in candidate_op_names_list]}")

                if not candidate_op_names_list:
                    # If no candidates suggested, this path in the beam cannot be extended.
                    # It's kept in new_beam_candidates to be re-sorted with others.
                    new_beam_candidates.append(item)
                    if self.verbose:
                        self.logger.debug(f"  No candidate operations to extend plan {self._plan_to_str(current_plan_op_specs)}. Keeping as is for now.")
                    continue

                for op_name_item in candidate_op_names_list:
                    operation_name = op_name_item["name"]

                    # Infer parameters for this operation_name
                    inferred_params = self._infer_parameters_for_operation(
                        operation_name=operation_name,
                        spec=current_remaining_spec_goals, # Pass current state of spec goals
                        graph_stats_for_context=graph_stats
                    )

                    if inferred_params is None:
                        if self.verbose:
                            self.logger.warning(
                                f"  Could not infer params for op '{operation_name}' (extending {self._plan_to_str(current_plan_op_specs)}). Skipping op."
                            )
                        continue

                    # Construct the full operation spec item (dictionary)
                    op_spec_item = {"name": operation_name, **inferred_params}

                    # Validate parameters for the operation
                    operation_instance = self.refactor_op_map.get(operation_name)
                    if not operation_instance: # Should not happen if _suggest_next_candidate_operations is correct
                        self.logger.warning(f"  Suggested operation '{operation_name}' not found in refactor_op_map. Skipping.")
                        continue
                    try:
                        # BaseRefactorOperation.validate_parameters is a static method
                        type(operation_instance).validate_parameters(op_spec_item)
                    except ValueError as e:
                        self.logger.warning(
                            f"  Parameter validation failed for '{operation_name}' with params {inferred_params}: {e}. Skipping."
                        )
                        continue

                    extended_plan_op_specs = current_plan_op_specs + [op_spec_item]

                    # Update remaining_spec_goals for the next step in this path of the beam
                    if hasattr(current_remaining_spec_goals, 'model_copy'): # Pydantic v2+
                        updated_remaining_spec_goals_after_op = current_remaining_spec_goals.model_copy(deep=True)
                    else: # Pydantic v1 or non-Pydantic Spec model
                        # Assuming .copy(deep=True) is available for older Pydantic or custom Spec
                        updated_remaining_spec_goals_after_op = current_remaining_spec_goals.copy(deep=True)

                    # Ensure plan_prefix_summary is initialized (it should be by default_factory=list)
                    if updated_remaining_spec_goals_after_op.plan_prefix_summary is None:
                        updated_remaining_spec_goals_after_op.plan_prefix_summary = []

                    # Create a summary of the current op_spec_item
                    op_params_summary = {
                        k: v for k, v in op_spec_item.items() if k not in ['name', 'target_file']
                    }
                    op_summary = (
                        f"Operation '{op_spec_item['name']}' "
                        f"on target_file '{op_spec_item.get('target_file', 'N/A')}' "
                        f"with params {op_params_summary} was added to the plan."
                    )
                    updated_remaining_spec_goals_after_op.plan_prefix_summary.append(op_summary)

                    # Score the new plan (list of op_spec_items)
                    # Use original_spec for overall context for scoring, but the updated goals are passed in beam
                    score = self._score_plan(extended_plan_op_specs, spec, graph_stats)
                    if self.verbose:
                        self.logger.debug(f"    Extended plan: {self._plan_to_str(extended_plan_op_specs)}, New Score: {score:.4f}")

                    # Convert extended_plan_op_specs to List[Phase] for the _are_goals_met check
                    # Use current_remaining_spec_goals for context like issue_description for phase descriptions
                    extended_plan_phases_for_check = self._convert_op_spec_items_to_phases(
                        extended_plan_op_specs,
                        current_remaining_spec_goals.issue_description
                    )
                    is_newly_complete = self._are_goals_met(extended_plan_phases_for_check, current_remaining_spec_goals)

                    if self.verbose:
                        self.logger.debug(f"    Plan extended to {len(extended_plan_op_specs)} ops. Is complete: {is_newly_complete}")

                    new_beam_candidates.append({
                        'plan': extended_plan_op_specs,
                        'score': score,
                        'remaining_spec_goals': updated_remaining_spec_goals_after_op,
                        'is_complete': is_newly_complete
                    })

            if not new_beam_candidates: # No new candidates generated from any plan in the beam
                if self.verbose: self.logger.debug(f"Beam search (depth {depth}): No new candidates generated in this iteration. Finalizing beam.")
                break # The current beam is the best we have.

            # Sort all candidates (both extended and non-extended previous beam items) and select top beam_width
            new_beam_candidates.sort(key=lambda x: x['score'], reverse=True)
            beam = new_beam_candidates[:self.beam_width]

            if self.verbose and beam:
                self.logger.debug(f"Beam after depth {depth} (best score: {beam[0]['score']:.4f if beam else 'N/A'}):")
                for i, b_item in enumerate(beam):
                    self.logger.debug(f"  {i+1}. Plan: {self._plan_to_str(b_item['plan'])}, Score: {b_item['score']:.4f}")
            elif self.verbose:
                 self.logger.debug(f"Beam is empty or no new candidates after depth {depth}.")

            # Termination condition: if all plans in beam have reached max_depth
            if not beam or all(len(p['plan']) >= max_plan_depth for p in beam):
                if self.verbose: self.logger.debug("Beam search: All plans reached max depth or beam is empty.")
                break

        # Final selection from the beam
        if not beam:
            if self.verbose: self.logger.info("Beam search completed. Beam is empty. No plan found.")
            return []

        # Prioritize complete plans
        complete_plans_in_beam = [item for item in beam if item['is_complete']]

        selected_item: Optional[Dict[str, Any]] = None

        if complete_plans_in_beam:
            # Sort complete plans by score (descending) - beam should already be sorted, but re-sort to be sure
            complete_plans_in_beam.sort(key=lambda x: x['score'], reverse=True)
            selected_item = complete_plans_in_beam[0]
            if self.verbose:
                self.logger.info(
                    f"Beam search completed. Selected a 'heuristically complete' plan "
                    f"(len: {len(selected_item['plan'])}) with score: {selected_item['score']:.4f}. "
                    f"Plan: {self._plan_to_str(selected_item['plan'])}"
                )
        else:
            # If no complete plans, take the best scoring (potentially partial) plan from the original beam.
            # The beam is already sorted by score, so beam[0] is the best.
            selected_item = beam[0]
            if self.verbose:
                self.logger.info(
                    f"Beam search completed. No 'heuristically complete' plan found. "
                    f"Selected best partial plan (len: {len(selected_item['plan'])}) with score: {selected_item['score']:.4f}. "
                    f"Plan: {self._plan_to_str(selected_item['plan'])}"
                )

        if selected_item and selected_item['plan']: # Ensure a plan exists
            best_plan_op_specs = selected_item['plan']
            # Convert the best plan (List[Dict[str,Any]]) to List[Phase]
            # Use the original spec's issue_description for consistent phase descriptions
            final_plan_phases = self._convert_op_spec_items_to_phases(best_plan_op_specs, spec.issue_description)
            return final_plan_phases
        else:
            # This case should ideally be covered by "if not beam" or if selected_item['plan'] is empty
            if self.verbose:
                self.logger.info("Beam search completed. No valid plan (empty or no plan ops) selected.")
            return []

    def _plan_to_str(self, plan_op_specs: List[Dict[str, Any]], max_phases_to_show: int = 5) -> str:
        """Helper to create a string representation of a plan (list of op_spec_items) for logging."""
        if not plan_op_specs:
            return "[]"

        names = []
        for i, op_spec in enumerate(plan_op_specs):
            if i < max_phases_to_show:
                names.append(op_spec.get("name", "unknown_op"))
            elif i == max_phases_to_show:
                names.append("...")
                break
        return f"[{' -> '.join(names)} (len:{len(plan_op_specs)})]"

    def plan_phases(self, spec: Spec) -> List[Phase]:
        """
        Generates a plan from a spec and then executes each phase of the plan.
        Plan generation is handled by generate_plan_from_spec.
        This method focuses on the execution loop using CollaborativeAgentGroup.
        """
        # generate_plan_from_spec now returns List[Phase] directly
        # It internally calls the new _beam_search_for_plan
        generated_plan_phases = self.generate_plan_from_spec(spec)

        predicted_core = None
        if generated_plan_phases: # Only try to predict if a plan was generated
            if self.core_predictor and self.core_predictor.is_ready:
                # Prepare raw_features for CorePredictor based on the *original* spec.operations,
                # as the generated plan might differ.
                # Or, should it be based on the *generated* plan?
                # For now, let's use the original spec as that's what CorePredictor was trained on.
                # This might need refinement.
                num_ops_for_predictor = len(spec.operations) if spec.operations else len(generated_plan_phases)

                raw_features_for_predictor = {
                    "num_operations": num_ops_for_predictor,
                    "num_target_symbols": len(spec.target_files),
                    "num_input_code_lines": 0,
                    "num_parameters_in_op": 0,
                }
                op_counts = {op_name: 0 for op_name in self.core_predictor.KNOWN_OPERATION_TYPES}

                # If using original spec.operations for features:
                if spec.operations:
                    for op_detail in spec.operations:
                        op_name = op_detail.get("name", "unknown_operation")
                        if op_name in op_counts: op_counts[op_name] += 1
                        else: op_counts["unknown_operation"] += 1
                # Else, if using generated_plan_phases for features (less ideal for current CorePredictor training):
                # else:
                #     for phase_obj in generated_plan_phases:
                #         op_name = phase_obj.operation_name
                #         if op_name in op_counts: op_counts[op_name] += 1
                #         else: op_counts["unknown_operation"] += 1

                raw_features_for_predictor.update(op_counts)

                if self.verbose:
                    self.logger.info(f"PhasePlanner: Raw features for CorePredictor: {raw_features_for_predictor}")

                predicted_core = self.core_predictor.predict(raw_features_for_predictor)
                if self.verbose:
                    self.logger.info(f"PhasePlanner: CorePredictor predicted preferred core: {predicted_core}")
            elif self.verbose:
                self.logger.info("PhasePlanner: CorePredictor not ready or not available. Skipping core prediction.")

        if generated_plan_phases:
            if self.verbose:
                self.logger.info(f"\n--- Running CollaborativeAgentGroup for {len(generated_plan_phases)} phases in the generated plan ---")
            execution_summary = []
            for phase_obj in generated_plan_phases: # Iterate over List[Phase]
                if self.verbose:
                    self.logger.info(f"Executing phase: {phase_obj.operation_name} on {phase_obj.target_file or 'repo-level'}")

                try:
                    validated_patch_script, patch_source = self.agent_group.run(
                        phase_ctx=phase_obj,
                        digester=self.digester,
                        validator_handle=self.validator_handle,
                        score_style_handle=self.score_style_handle,
                        predicted_core=predicted_core
                    )

                    if validated_patch_script:
                        if self.verbose:
                            self.logger.info(f"Phase {phase_obj.operation_name}: Successfully generated patch script (source: {patch_source}). Preview: {str(validated_patch_script)[:100]}...")
                        current_phase_summary = {
                            "phase_operation": phase_obj.operation_name,
                            "target": phase_obj.target_file,
                            "status": "success",
                            "patch_preview": str(validated_patch_script)[:100],
                            "patch_source": patch_source
                        }
                        execution_summary.append(current_phase_summary)

                        target_file_path_str = phase_obj.target_file
                        target_file_path = Path(target_file_path_str) if target_file_path_str else None
                        modified_content: Optional[str] = None
                        diff_summary = "Diff generation skipped: No target file or patch application failed."

                        if not target_file_path:
                            self.logger.warning(f"PhasePlanner: Phase target_file is None for phase '{phase_obj.operation_name}'. Cannot apply patch or proceed with CommitBuilder.")
                            current_phase_summary["status"] = "error"
                            current_phase_summary["error_message"] = "Target file was None for CommitBuilder step"
                            continue

                        abs_target_file_path = self.project_root_path / target_file_path
                        original_content = self.digester.get_file_content(abs_target_file_path)
                        if original_content is None:
                            original_content = ""
                            if self.verbose: self.logger.info(f"PhasePlanner: Target file '{abs_target_file_path}' is new or content not found. Applying patch to empty content.")

                        try:
                            if self.verbose: self.logger.info(f"PhasePlanner: Applying validated LibCST script to '{abs_target_file_path}' (source: {patch_source})...")
                            modified_content = apply_libcst_codemod_script(original_content, validated_patch_script)
                            if self.verbose: self.logger.info(f"PhasePlanner: LibCST script applied successfully to '{target_file_path}'.")

                            diff_lines = list(difflib.unified_diff(
                                original_content.splitlines(keepends=True),
                                modified_content.splitlines(keepends=True),
                                fromfile=f"a/{target_file_path.name}", tofile=f"b/{target_file_path.name}"
                            ))
                            diff_summary = "".join(diff_lines) if diff_lines else f"No textual changes detected for '{target_file_path}'."
                            if self.verbose: self.logger.info(f"PhasePlanner: Diff summary generated for '{target_file_path}' (length {len(diff_summary)}).")

                        except PatchApplicationError as pae:
                            self.logger.error(f"PhasePlanner: Failed to apply validated patch script for {abs_target_file_path}: {pae}")
                            current_phase_summary["status"] = "error"
                            current_phase_summary["error_message"] = f"Patch application failed: {pae}"
                            continue
                        except Exception as e_apply:
                            self.logger.error(f"PhasePlanner: Unexpected error during patch application or diff for {abs_target_file_path}: {e_apply}", exc_info=True)
                            current_phase_summary["status"] = "error"
                            current_phase_summary["error_message"] = f"Unexpected patch application/diff error: {e_apply}"
                            continue

                        if modified_content is None:
                            self.logger.warning(f"PhasePlanner: Skipping CommitBuilder for phase targeting '{target_file_path}' due to no modified content.")
                            current_phase_summary["status"] = "skipped"
                            current_phase_summary["reason"] = "No modified content after application attempt"
                            continue

                        validated_patch_content_map = {target_file_path: modified_content}
                        validator_results_summary = "Validation results summary not available."
                        if self.agent_group.patch_history:
                            try:
                                last_attempt_info = self.agent_group.patch_history[-1]
                                last_error_tb_info = last_attempt_info[3] if len(last_attempt_info) > 3 else "N/A"
                                validator_results_summary = f"Final validation in agent: Valid={last_attempt_info[1]}, Score={last_attempt_info[2]:.2f}, Last Error='{str(last_error_tb_info)[:50]}...'"
                            except Exception as e_hist:
                                self.logger.warning(f"PhasePlanner: Error accessing patch_history for validator summary: {e_hist}")

                        branch_name_suffix = getattr(spec, 'issue_id', None) or spec.issue_description[:20].replace(' ', '-').lower()
                        branch_name = f"feature/auto-patch-{branch_name_suffix}-{random.randint(1000,9999)}"
                        commit_title = f"Auto-apply patch for '{spec.issue_description[:40]}...' (Phase: {phase_obj.operation_name}, Source: {patch_source})"

                        if self.verbose: self.logger.info("PhasePlanner: Calling CommitBuilder.process_and_submit_patch...")
                        saved_patch_set_path = self.commit_builder.process_and_submit_patch(
                            validated_patch_content_map=validated_patch_content_map, spec=spec, diff_summary=diff_summary,
                            validator_results_summary=validator_results_summary, branch_name=branch_name, commit_title=commit_title,
                            project_root=self.project_root_path, patch_source=patch_source
                        )

                        if saved_patch_set_path:
                            self.logger.info(f"PhasePlanner: CommitBuilder successfully saved patch set to: {saved_patch_set_path}")
                            if self.data_dir_path and validated_patch_script:
                                log_success = log_successful_patch(
                                    data_directory=self.data_dir_path, spec=spec, diff_summary=diff_summary,
                                    successful_script_str=validated_patch_script, patch_source=patch_source, verbose=self.verbose
                                )
                                if log_success and self.verbose: self.logger.info(f"PhasePlanner: Logged patch to success memory.")
                            if self.verbose: self.logger.info("PhasePlanner: Breaking after first successful phase patch processing.")
                            break
                        else:
                            self.logger.error(f"PhasePlanner: CommitBuilder failed to save patch set for phase {phase_obj.operation_name}.")
                            current_phase_summary["status"] = "error"
                            current_phase_summary["error_message"] = "CommitBuilder failed to save patch set"

                    else: # validated_patch_script is None
                        self.logger.warning(f"Phase {phase_obj.operation_name}: Failed to generate a patch script. Source info: {patch_source}")
                        execution_summary.append({
                            "phase_operation": phase_obj.operation_name, "target": phase_obj.target_file,
                            "status": "failed_no_patch", "patch_source": patch_source
                        })

                except PhaseFailure as pf_e:
                    self.logger.error(f"PhasePlanner: CollaborativeAgentGroup reported PhaseFailure for phase {phase_obj.operation_name}: {pf_e}")
                    execution_summary.append({"phase_operation": phase_obj.operation_name, "target": phase_obj.target_file, "status": "PhaseFailure", "error_message": str(pf_e)})
                    if self.verbose: self.logger.info("PhasePlanner: Stopping plan execution due to PhaseFailure.")
                    break
                except Exception as e:
                    self.logger.error(f"Error running agent group for phase {phase_obj.operation_name}: {type(e).__name__} - {e}", exc_info=True)
                    execution_summary.append({"phase_operation": phase_obj.operation_name, "target": phase_obj.target_file, "status": "error", "error_message": str(e)})

            if self.verbose:
                self.logger.info("--- CollaborativeAgentGroup execution finished ---")
                self.logger.info("Execution Summary (from planner):")
                for summary_item in execution_summary:
                    self.logger.info(f"  - {summary_item}")
        else:
            self.logger.info("PhasePlanner.plan_phases: No plan generated. Nothing to execute.")

        return generated_plan_phases

    def _are_goals_met(self, current_plan_phases: List[Phase], spec: 'Spec') -> bool:
        """
        Checks if the current plan plausibly meets the goals outlined in the spec.
        This is a simplified heuristic for now.
        """
        if not current_plan_phases: # An empty plan cannot meet goals
            return False

        if self.verbose:
            self.logger.info(f"PhasePlanner._are_goals_met: Checking plan with {len(current_plan_phases)} phase(s) against spec '{spec.issue_description[:50]}...'.")

        # Heuristic 1: If there are acceptance tests, plan length should ideally be related.
        # This is a very rough proxy for goal completion.
        if spec.acceptance_tests:
            # Simple heuristic: plan is "potentially complete" if it has at least as many phases
            # as acceptance tests, or a configurable minimum number of phases if tests are few.
            min_phases_if_tests = self.app_config.get("planner", {}).get("min_phases_for_completion_with_tests", 1)

            is_met = len(current_plan_phases) >= min_phases_if_tests
            if self.verbose:
                self.logger.info(f"PhasePlanner._are_goals_met: Acceptance tests exist ({len(spec.acceptance_tests)}). Plan length {len(current_plan_phases)} >= {min_phases_if_tests} -> {is_met}")
            return is_met
        else:
            # Heuristic 2: If no acceptance tests, any non-empty plan might be considered "meeting goals"
            # for the purpose of this simple check, assuming the plan addresses the issue description.
            # The quality/score of the plan is handled separately.
            if self.verbose:
                self.logger.info(f"PhasePlanner._are_goals_met: No acceptance tests. Non-empty plan considered heuristically complete.")
            return True

if __name__ == '__main__':
    import logging # For __main__ example
    from src.utils.config_loader import load_app_config
    from src.digester.repository_digester import RepositoryDigester

    # Setup basic logging for the __main__ example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    main_logger = logging.getLogger(__name__)
    main_logger.info("--- PhasePlanner Example Usage (Conceptual) ---")

    # Using a more concrete mock for RepositoryDigester
    class MockDigesterForPlannerMain(RepositoryDigester):
        def __init__(self, app_config: Dict[str, Any]): # Accept app_config
            super().__init__(project_root=".", app_config=app_config) # Pass to parent
            self.project_call_graph = {"module.func_a": {"module.func_b"}}
            self.project_control_dependence_graph = {}
            self.project_data_dependence_graph = {}
            self._verbose = app_config.get("general",{}).get("verbose", False) # Use verbose from app_config
            if self._verbose: main_logger.info("MockDigesterForPlannerMain initialized.")

        def get_project_overview(self) -> Dict[str, Any]:
            if self._verbose: main_logger.info("__main__ MockDigester.get_project_overview called")
            return {"files_in_project": 5, "main_language": "python", "mock_overview": True}

        def get_file_content(self, path: Path) -> Optional[str]:
            # Ensure path is absolute for consistent behavior if project_root is "."
            abs_path = Path(self.project_root) / path
            content = f"# __main__ Mock Content for {abs_path}\n# BAD_STYLE example\n# FAIL_VALIDATION example"
            if self._verbose: main_logger.info(f"__main__ MockDigester.get_file_content for {abs_path} returning: '{content[:50]}...'")
            return content # Return content, None if path is problematic (handled by caller)

    dummy_project_root_main = Path("_temp_mock_project_root_main")
    dummy_project_root_main.mkdir(parents=True, exist_ok=True)

    try:
        with open(dummy_project_root_main / "style_fingerprint.json", "w") as f_style:
            json.dump({"line_length": 100, "indent_style": "space", "indent_width": 4}, f_style)
        with open(dummy_project_root_main / "naming_conventions.db", "w") as f_naming:
            f_naming.write("mock_naming_db_content")

        app_cfg_main = load_app_config()
        app_cfg_main["general"]["verbose"] = True
        app_cfg_main["general"]["data_dir"] = str(dummy_project_root_main / ".agent_data_main_test")
        app_cfg_main.setdefault("models", {})
        # Make model paths relative to a known base if needed, or ensure they are absolute
        # For this test, we'll use placeholder names that won't be loaded.
        app_cfg_main["models"]["planner_scorer_gguf"] = "non_existent_scorer.gguf"
        app_cfg_main["models"]["agent_llm_gguf"] = "non_existent_agent_llm.gguf"
        app_cfg_main["models"]["divot5_infill_model_dir"] = "non_existent_divot5_infill"
        # Set a very small beam width and depth for faster testing in __main__
        app_cfg_main.setdefault("planner", {})
        app_cfg_main["planner"]["beam_width"] = 2
        app_cfg_main["planner"]["max_plan_depth"] = 2 # Keep test runs short

        # Pass app_cfg_main to MockDigester
        mock_digester_main = MockDigesterForPlannerMain(app_config=app_cfg_main)

        # Initialize PhasePlanner with the mock digester
        planner = PhasePlanner(
            project_root_path=dummy_project_root_main,
            app_config=app_cfg_main,
            digester=mock_digester_main
        )
        # Inject logger into planner instance for its own logging calls
        planner.logger = main_logger.getChild("PhasePlanner")


        # Example Spec (no operations pre-defined, let beam search create them)
        example_spec_data_main = {
            "issue_description": "Refactor user authentication to use a new JWT library",
            "target_files": ["src/auth/jwt_handler.py", "src/services/auth_service.py"],
            "operations": [], # Start with empty operations, expect beam search to populate
            "acceptance_tests": ["test_jwt_creation", "test_jwt_validation_success", "test_jwt_validation_failure_expired"]
        }
        spec_instance_main = Spec(**example_spec_data_main)

        main_logger.info(f"\nPlanning for spec: {spec_instance_main.issue_description}")
        # This will now use the new _beam_search_for_plan
        plan_phases_list_main = planner.plan_phases(spec_instance_main)

        main_logger.info("\n--- Post-execution: Generated Plan Phases (from planner return) ---")
        if plan_phases_list_main:
            for i, p_main in enumerate(plan_phases_list_main):
                main_logger.info(f"Details for Planned Phase {i+1}:")
                if hasattr(p_main, 'model_dump_json'):
                    main_logger.info(p_main.model_dump_json(indent=2))
                else:
                    main_logger.info(f"  Operation: {getattr(p_main, 'operation_name', 'N/A')}, Target: {getattr(p_main, 'target_file', 'N/A')}, Params: {getattr(p_main, 'parameters', {})}")
        else:
            main_logger.info("  No plan was generated by the planner.")

    except ImportError as e_imp:
        main_logger.error(f"ImportError in PhasePlanner __main__ example: {e_imp}. Check imports.", exc_info=True)
    except Exception as e_main:
        main_logger.error(f"Error in PhasePlanner __main__ example: {type(e_main).__name__}: {e_main}", exc_info=True)
    finally:
        if dummy_project_root_main.exists():
            import shutil
            try:
                shutil.rmtree(dummy_project_root_main)
                if app_cfg_main.get("general",{}).get("verbose",False): main_logger.info(f"Cleaned up dummy project root: {dummy_project_root_main}")
            except OSError as e_ose:
                main_logger.error(f"Error removing directory {dummy_project_root_main}: {e_ose}")

    main_logger.info("\n--- PhasePlanner Example Done ---")
