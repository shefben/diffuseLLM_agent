# src/planner/phase_planner.py
import difflib  # For generating diff summaries
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Tuple, Callable
from pathlib import Path
import json
import hashlib
import random  # For dummy scorer and branch names

from src.transformer import (
    apply_libcst_codemod_script,
    PatchApplicationError,
)  # For real patch application

# Forward references for type hints if full imports are problematic
if TYPE_CHECKING:
    from ..digester.repository_digester import RepositoryDigester

    # Spec, Phase, BaseRefactorOperation are now directly imported.
    # REFACTOR_OPERATION_INSTANCES is used directly.

# Actual imports
from src.utils.config_loader import DEFAULT_APP_CONFIG  # For default model paths
from src.spec_normalizer.spec_normalizer_interface import SpecNormalizerModelInterface
from src.spec_normalizer.t5_client import T5Client
from src.spec_normalizer.diffusion_spec_normalizer import DiffusionSpecNormalizer
from .spec_model import Spec
from .phase_model import Phase
from .refactor_grammar import BaseRefactorOperation, REFACTOR_OPERATION_INSTANCES

# Child component imports
from src.learning.core_predictor import CorePredictor  # Added import

# T5Client is imported for spec normalization above, ensure no conflict if used for other purposes.
# If T5Client is used for other things, the alias T5ClientForSpecNormalization should be used for spec.
# For now, assume T5Client is primarily for spec normalization context here.
from src.retriever.symbol_retriever import SymbolRetriever  # Assuming direct import
from src.spec_normalizer.spec_fusion import SpecFusion  # Assuming direct import
from src.validator.validator import Validator
from src.builder.commit_builder import CommitBuilder
from src.agent_group.collaborative_agent_group import CollaborativeAgentGroup
from src.profiler.style_validator import StyleValidatorCore

# Other necessary imports
from src.agent_group.exceptions import PhaseFailure
from src.profiler.llm_interfacer import (
    get_llm_score_for_text,
    propose_refactor_operations,
)
from src.mcp.mcp_manager import get_mcp_prompt
from src.utils.memory_logger import log_successful_patch  # For Success Memory Logging

# REFACTOR_OPERATION_CLASSES is not directly used by PhasePlanner logic, REFACTOR_OPERATION_INSTANCES is.

# Fallback for RepositoryDigester if not available (e.g. in isolated subtask)
# This is primarily for type hinting if RepositoryDigester cannot be imported directly.
# The actual instance is passed to __init__.
if TYPE_CHECKING:
    # Ensure Phase is available for type hints if not already.
    from .phase_model import Phase

if (
    TYPE_CHECKING or "RepositoryDigester" not in globals()
):  # Keep this for RepositoryDigester hint

    class RepositoryDigester:  # type: ignore
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

        pass  # Ensure class body is not empty


class PhasePlanner:
    # mock_validator_handle is now removed.
    # The actual validator instance's validate_patch method will be used.

    @staticmethod
    def mock_score_style_handle(patch: Any, style_profile: Dict[str, Any]) -> float:
        print(
            f"PhasePlanner.mock_score_style_handle: Scoring style for patch: {str(patch)[:100]} with profile keys: {list(style_profile.keys())}"
        )
        # Assuming patch is now a script string, this mock might need adjustment if it were to inspect content.
        # For now, it's a simple mock.
        if isinstance(patch, str) and "BAD_STYLE_MARKER_IN_SCRIPT" in patch:
            return 0.3  # Low score for bad style
        return 0.9  # Default high score

    def _score_style_with_validator(
        self, patch: Any, _style_profile: Dict[str, Any]
    ) -> float:
        if not isinstance(patch, str):
            return 1.0
        return self.style_validator.score_patch_script_content(
            patch,
            project_root=self.project_root_path,
            db_path=self.naming_conventions_db_path,
        )

    def __init__(
        self,
        project_root_path: Path,
        app_config: Dict[str, Any],
        digester: "RepositoryDigester",
        refactor_op_map: Optional[Dict[str, BaseRefactorOperation]] = None,
        # beam_width is now primarily from app_config, but can be overridden if passed for testing
        beam_width: Optional[int] = None,
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
        style_fp_path = (
            self.project_root_path / "style_fingerprint.json"
        )  # Standard name
        self.style_fingerprint: Dict[str, Any] = {}
        try:
            if style_fp_path.exists():
                with open(style_fp_path, "r", encoding="utf-8") as f:
                    self.style_fingerprint = json.load(f)
                if self.verbose:
                    print(
                        f"PhasePlanner: Loaded style fingerprint from {style_fp_path}"
                    )
            else:
                if self.verbose:
                    print(
                        f"PhasePlanner Warning: Style fingerprint file not found at {style_fp_path}. Using empty fingerprint."
                    )
        except json.JSONDecodeError as e:
            print(
                f"PhasePlanner Warning: Error decoding style fingerprint JSON from {style_fp_path}: {e}. Using empty fingerprint."
            )
        except Exception as e_gen:
            print(
                f"PhasePlanner Warning: Could not load style fingerprint from {style_fp_path} due to: {e_gen}. Using empty fingerprint."
            )

        self.naming_conventions_db_path = (
            self.project_root_path / "naming_conventions.db"
        )  # Standard name
        self.data_dir_path = Path(
            self.app_config.get("general", {}).get(
                "data_dir", str(self.project_root_path / ".agent_data")
            )
        )
        self.data_dir_path.mkdir(parents=True, exist_ok=True)  # Ensure data_dir exists
        if self.verbose:
            print(
                f"PhasePlanner: Data directory for logs (e.g. success memory): {self.data_dir_path}"
            )

        # Style validator for scoring patches against project conventions
        self.style_validator = StyleValidatorCore(
            app_config=self.app_config,
            style_profile=self.style_fingerprint,
        )

        # Default workflow type controls how agents cooperate during execution
        self.workflow_type = self.app_config.get("planner", {}).get(
            "workflow_type", "orchestrator-workers"
        )

        # Configure self.scorer_model_config (for internal use by _score_candidate_plan_with_llm)
        llm_params_config = self.app_config.get("llm_params", {})
        scorer_model_p = self.app_config.get("models", {}).get(
            "planner_scorer_gguf", DEFAULT_APP_CONFIG["models"]["planner_scorer_gguf"]
        )

        self.scorer_model_config = {
            "model_path": str(Path(scorer_model_p).resolve()),  # Resolve path
            "enabled": True,  # Assuming enabled if path is provided; can add specific config key later
            "verbose": self.verbose,
            "n_gpu_layers": llm_params_config.get(
                "n_gpu_layers_default",
                DEFAULT_APP_CONFIG["llm_params"]["n_gpu_layers_default"],
            ),
            "n_ctx": llm_params_config.get(
                "n_ctx_default", DEFAULT_APP_CONFIG["llm_params"]["n_ctx_default"]
            ),
            "temperature": llm_params_config.get(
                "plan_score_temp",
                llm_params_config.get(
                    "temperature_default",
                    DEFAULT_APP_CONFIG["llm_params"]["temperature_default"],
                ),
            ),
            "max_tokens": llm_params_config.get(
                "plan_score_max_tokens", 16
            ),  # Renamed from max_tokens_for_score
        }
        if not Path(self.scorer_model_config["model_path"]).is_file():
            print(
                f"PhasePlanner Warning: Scorer model not found at {self.scorer_model_config['model_path']}. Plan scoring will use fallback."
            )
            self.scorer_model_config["enabled"] = False
        else:
            if self.verbose:
                print(
                    f"PhasePlanner: Real LLM scorer configured with model: {self.scorer_model_config['model_path']}"
                )

        self.refactor_op_map: Dict[str, BaseRefactorOperation] = (
            refactor_op_map
            if refactor_op_map is not None
            else REFACTOR_OPERATION_INSTANCES
        )
        self.beam_width = (
            beam_width
            if beam_width is not None
            else self.app_config.get("planner", {}).get(
                "beam_width", DEFAULT_APP_CONFIG["planner"]["beam_width"]
            )
        )
        if self.beam_width <= 0:
            raise ValueError("Beam width must be a positive integer.")
        if self.verbose:
            print(f"PhasePlanner: Beam width set to {self.beam_width}")

        self.operations_llm_path = str(
            Path(
                self.app_config.get("models", {}).get("operations_llm_gguf", "")
            ).resolve()
        )
        if self.verbose and self.operations_llm_path:
            print(
                f"PhasePlanner: operations LLM path set to {self.operations_llm_path}"
            )

        # Instantiate Child Components (Passing app_config or derived configs)

        # Spec Normalizer Selection Logic
        spec_normalizer_type = self.app_config.get("spec_normalizer", {}).get(
            "type", "t5"
        )  # Default to t5
        spec_model_instance: SpecNormalizerModelInterface

        if spec_normalizer_type.lower() == "diffusion":
            if self.verbose:
                print(
                    "PhasePlanner: Using DiffusionSpecNormalizer for spec normalization."
                )
            spec_model_instance = DiffusionSpecNormalizer(app_config=self.app_config)
        elif spec_normalizer_type.lower() == "t5":
            if self.verbose:
                print("PhasePlanner: Using T5Client for spec normalization.")
            # Ensure T5ClientForSpecNormalization or a T5Client instance is used.
            # Using T5Client directly, assuming it's the one intended for spec normalization.
            spec_model_instance = T5Client(app_config=self.app_config)
        else:
            if self.verbose:
                print(
                    f"PhasePlanner Warning: Unknown spec_normalizer type '{spec_normalizer_type}'. Defaulting to T5Client."
                )
            spec_model_instance = T5Client(app_config=self.app_config)

        # self.t5_client = T5Client(app_config=self.app_config) # Removed as spec_model_instance replaces its role for SpecFusion
        self.symbol_retriever = SymbolRetriever(
            digester=self.digester, app_config=self.app_config
        )
        self.spec_normalizer = SpecFusion(
            spec_model_interface=spec_model_instance,
            symbol_retriever=self.symbol_retriever,
            app_config=self.app_config,
        )
        self.validator = Validator(app_config=self.app_config)
        if self.verbose:
            print("PhasePlanner: Validator instance initialized.")

        # Agent Group requires specific model paths from app_config
        _ = self.app_config.get("models", {}).get(
            "agent_llm_gguf", DEFAULT_APP_CONFIG["models"]["agent_llm_gguf"]
        )
        # CollaborativeAgentGroup __init__ currently takes llm_model_path and specific core_configs
        # For now, we pass app_config and let it derive, or pass specific paths as before if its __init__ is not yet updated.
        # Based on previous structure of CollaborativeAgentGroup, it took llm_core_config, diffusion_core_config, and llm_model_path.
        # We will pass app_config to it, and it should become responsible for deriving these.
        # For the purpose of this subtask, we assume CollaborativeAgentGroup will be updated to take app_config.
        self.agent_group: "CollaborativeAgentGroup" = CollaborativeAgentGroup(
            app_config=self.app_config,  # Pass the full app_config
            digester=self.digester,  # Pass digester instance
            style_profile=self.style_fingerprint,
            naming_conventions_db_path=self.naming_conventions_db_path,
            validator_instance=self.validator,
            # Old parameters like llm_core_config, diffusion_core_config, llm_model_path
            # are assumed to be handled by CollaborativeAgentGroup internally using app_config.
        )
        if self.verbose:
            print("PhasePlanner: CollaborativeAgentGroup initialized.")

        # Store actual validator handle and mock score style handle
        self.validator_handle: Callable[
            [Optional[str], str, "RepositoryDigester", Path], Tuple[bool, Optional[str]]
        ] = self.validator.validate_patch
        self.score_style_handle: Callable[[Any, Dict[str, Any]], float] = (
            self._score_style_with_validator
        )
        if self.verbose:
            print("PhasePlanner: Validator handle set. Style scorer configured.")

        # Initialize CommitBuilder
        self.commit_builder = CommitBuilder(app_config=self.app_config)
        if self.verbose:
            print("PhasePlanner: CommitBuilder instance initialized.")

        self.plan_cache: Dict[str, List[Phase]] = {}

    def set_workflow_type(self, workflow: str) -> None:
        """Update the planner's workflow strategy."""
        valid = {
            "prompt_chaining",
            "routing",
            "parallelization",
            "orchestrator-workers",
            "evaluator-optimizer",
        }
        if workflow in valid:
            self.workflow_type = workflow
        else:
            print(f"Unknown workflow '{workflow}', using default {self.workflow_type}")
        # self.phi2_scorer_cache: Dict[str, float] = {} # Optional cache

        # Determine CorePredictor model path from app_config or use a default
        core_predictor_model_path_str = self.app_config.get("models", {}).get(
            "core_predictor_model_path",
            str(
                self.data_dir_path / "core_predictor.joblib"
            ),  # Defaulting to data_dir now
        )
        # Ensure the default path is used if the config value is None or empty string
        if not core_predictor_model_path_str:  # Handle empty string from config
            core_predictor_model_path_str = str(
                self.data_dir_path / "core_predictor.joblib"
            )

        core_predictor_model_path = Path(core_predictor_model_path_str)

        self.core_predictor = CorePredictor(
            model_path=core_predictor_model_path, verbose=self.verbose
        )
        if self.verbose:
            print(
                f"PhasePlanner: CorePredictor initialized. Model path: {core_predictor_model_path}, Ready: {self.core_predictor.is_ready}"
            )

    def _suggest_alternative_operations(
        self, current_op_spec_item: Dict[str, Any], spec: "Spec"
    ) -> List[Dict[str, Any]]:
        """Return the input operation and reasonable alternatives."""
        alternative_ops = [current_op_spec_item.copy()]
        current_op_name = current_op_spec_item.get("name")

        if current_op_name and current_op_name != "generic_code_edit":
            # Construct edit_description for the generic_code_edit alternative
            original_op_params = current_op_spec_item.get("parameters", {})
            if isinstance(original_op_params, dict):
                original_op_params_str = ", ".join(
                    f"{k}='{v}'" for k, v in original_op_params.items()
                )
            else:  # Fallback if parameters is not a dict (though it should be)
                original_op_params_str = str(original_op_params)

            if (
                not original_op_params_str and original_op_params
            ):  # If params is not empty but str is (e.g. empty dict)
                original_op_params_str = str(original_op_params)

            generic_edit_desc = (
                f"Original operation was '{current_op_name}' with parameters: {original_op_params_str}. "
                f"Consider if a direct code edit can achieve the intended outcome described by the original operation. "
                f"Original spec issue: {spec.issue_description}"
            )

            # Limit length of description
            if len(generic_edit_desc) > 500:
                generic_edit_desc = generic_edit_desc[:497] + "..."

            generic_code_edit_op_spec_item = {
                "name": "generic_code_edit",
                "target_file": current_op_spec_item.get(
                    "target_file"
                ),  # Inherit target_file
                "parameters": self._infer_parameters_for_alternative(
                    "generic_code_edit",
                    current_op_spec_item,
                    spec,
                    {"edit_description": generic_edit_desc},
                ),
            }
            alternative_ops.append(generic_code_edit_op_spec_item)
            if self.verbose:
                print(
                    f"PhasePlanner._suggest_alternative_operations: Suggested 'generic_code_edit' as an alternative for '{current_op_name}'."
                )

            if current_op_name == "add_decorator":
                dec_import = current_op_spec_item.get(
                    "decorator_import_statement_param"
                )
                if not dec_import:
                    dec_import = getattr(
                        REFACTOR_OPERATION_INSTANCES["add_decorator"],
                        "decorator_import_statement",
                        None,
                    )
                if dec_import:
                    add_import_alt = {
                        "name": "add_import",
                        "target_file": current_op_spec_item.get("target_file"),
                        "parameters": self._infer_parameters_for_alternative(
                            "add_import",
                            current_op_spec_item,
                            spec,
                            {"import_statement": dec_import},
                        ),
                    }
                    alternative_ops.append(add_import_alt)
                    if self.verbose:
                        print(
                            "PhasePlanner._suggest_alternative_operations: Added 'add_import' as alternative for missing decorator import."
                        )

            if current_op_name in {"add_function", "modify_function_logic"}:
                doc_alt = {
                    "name": "update_docstring",
                    "target_file": current_op_spec_item.get("target_file"),
                    "parameters": self._infer_parameters_for_alternative(
                        "update_docstring",
                        current_op_spec_item,
                        spec,
                        {},
                    ),
                }
                alternative_ops.append(doc_alt)
                if self.verbose:
                    print(
                        "PhasePlanner._suggest_alternative_operations: Added 'update_docstring' as alternative for function operation."
                    )

        return alternative_ops

    def generate_plan_from_spec(self, spec: Spec) -> List[Phase]:
        """
        Generates a plan (a list of Phase objects) from a given Spec object.
        This method includes caching, graph statistics retrieval, and beam search for plan generation.
        It does not execute the plan.
        """
        print(
            f"PhasePlanner: Received spec for task: '{spec.issue_description[:50]}...' for plan generation."
        )
        cache_key = self._get_spec_cache_key(spec)
        if cache_key in self.plan_cache:
            print("PhasePlanner: Returning cached plan.")
            return self.plan_cache[cache_key]

        print("PhasePlanner: No cached plan found. Generating new plan.")
        graph_stats = self._get_graph_statistics()
        best_plan = self._beam_search_for_plan(spec, graph_stats)

        if best_plan:
            self.plan_cache[cache_key] = (
                best_plan  # Cache the plan (list of Phase objects)
            )
            print("PhasePlanner: Plan generated and cached.")
        else:
            print(
                "PhasePlanner: Failed to generate a plan (beam search returned empty)."
            )

        return best_plan

    def _extract_operations_from_goal(self, goal_text: str) -> List[Dict[str, Any]]:
        """Naively infer operations from free-form text.

        This is a lightweight heuristic used when the spec normalizer returns no
        structured operations.  A real system would delegate to an LLM-based
        planner, but here we map simple keywords to grammar operations.
        """
        ops: List[Dict[str, Any]] = []
        lowered = goal_text.lower()
        if "import" in lowered:
            ops.append({"name": "add_import", "parameters": {"import_statement": ""}})
        if "decorator" in lowered:
            ops.append(
                {
                    "name": "add_decorator",
                    "parameters": {
                        "decorator_name": "@todo",
                        "target_function_name": "func",
                    },
                }
            )
        if "docstring" in lowered:
            ops.append(
                {
                    "name": "update_docstring",
                    "parameters": {"target_name": "func", "new_docstring": ""},
                }
            )
        return ops

    def generate_plan_from_goal(
        self, goal_text: str, workflow_type: Optional[str] = None
    ) -> Tuple[Spec, List[Phase]]:
        """High level entry point: normalize raw goal text and generate a plan."""
        workflow = workflow_type or self.workflow_type
        mcp_spec = get_mcp_prompt(self.app_config, workflow, "SpecNormalizer")
        spec_obj = self.spec_normalizer.normalise_request(goal_text, mcp_spec)
        if spec_obj is None:
            raise ValueError("SpecNormalizer failed to create a Spec from goal text")

        mcp_ops = get_mcp_prompt(self.app_config, workflow, "OperationsModel")
        llm_ops = propose_refactor_operations(
            goal_text,
            model_path=self.operations_llm_path,
            verbose=self.verbose,
            mcp_prompt=mcp_ops,
            use_vllm=self.app_config.get("general", {}).get("use_vllm", False),
        )

        if not llm_ops:
            llm_ops = self._extract_operations_from_goal(goal_text)

        existing_ops = spec_obj.operations or []

        combined_ops: list[dict] = []
        seen_keys: set[tuple[str | None, str | None]] = set()
        for op in existing_ops + llm_ops:
            name = op.get("name")
            target = op.get("target_file")
            key = (str(name), str(target))
            if key not in seen_keys:
                combined_ops.append(op)
                seen_keys.add(key)

        if not combined_ops:
            combined_ops.append(
                {
                    "name": "generic_code_edit",
                    "parameters": {"edit_description": goal_text[:120]},
                }
            )

        spec_obj.operations = combined_ops

        plan_list = self.generate_plan_from_spec(spec_obj)
        return spec_obj, plan_list

    def _get_spec_cache_key(self, spec: Spec) -> str:
        # Ensure Spec model is Pydantic v2 compatible for model_dump_json if that's used.
        # Fallback to dict if model_dump_json is not available (e.g. placeholder Spec)
        if hasattr(spec, "model_dump_json"):
            spec_json_str = spec.model_dump_json(sort_keys=True)
        else:  # Fallback for placeholder Spec or Pydantic v1
            spec_json_str = json.dumps(
                spec.dict(sort_keys=True) if hasattr(spec, "dict") else vars(spec),
                sort_keys=True,
            )
        return hashlib.md5(spec_json_str.encode("utf-8")).hexdigest()

    def _infer_parameters_for_alternative(
        self,
        alt_op_name: str,
        original_op_spec: Dict[str, Any],
        spec: "Spec",
        base_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Heuristic parameter inference for alternative operations.

        This attempts to map fields from the original operation to sensible
        defaults for the alternative one so users do not need to manually fill
        them in when approving a generated plan.
        """
        params = base_params.copy() if base_params else {}

        if alt_op_name == "generic_code_edit":
            target_file = original_op_spec.get("target_file")
            if target_file:
                params.setdefault("target_file", target_file)

        elif alt_op_name == "add_import":
            target_file = original_op_spec.get("target_file")
            if target_file:
                params.setdefault("target_file", target_file)
            import_stmt = original_op_spec.get(
                "decorator_import_statement_param"
            ) or original_op_spec.get("import_statement")
            if not import_stmt and original_op_spec.get("decorator_name"):
                dec = str(original_op_spec["decorator_name"]).lstrip("@")
                import_stmt = f"from . import {dec}"
            if import_stmt:
                params.setdefault("import_statement", import_stmt)

        elif alt_op_name == "update_docstring":
            target = original_op_spec.get(
                "target_function_name"
            ) or original_op_spec.get("target_name")
            if target:
                params.setdefault("target_name", target)
            params.setdefault(
                "new_docstring",
                f"Update docstring to clarify: {spec.issue_description[:40]}...",
            )

        return params

    def _get_graph_statistics(self) -> Dict[str, Any]:
        # Initialize stats with defaults
        num_cg_nodes, num_cg_edges, cg_density = 0, 0, 0.0
        num_cdg_nodes, num_cdg_edges, cdg_density = 0, 0, 0.0
        num_ddg_nodes, num_ddg_edges, ddg_density = 0, 0, 0.0

        # Call Graph (CG) statistics
        if (
            hasattr(self.digester, "project_call_graph")
            and self.digester.project_call_graph
        ):
            cg = self.digester.project_call_graph
            num_cg_nodes = len(cg)
            num_cg_edges = sum(len(edges) for edges in cg.values())
            if num_cg_nodes > 1:
                cg_density = num_cg_edges / (num_cg_nodes * (num_cg_nodes - 1))

        # Control Dependence Graph (CDG) statistics
        if (
            hasattr(self.digester, "project_control_dependence_graph")
            and self.digester.project_control_dependence_graph
        ):
            cdg = self.digester.project_control_dependence_graph
            num_cdg_nodes = len(cdg)
            # Assuming CDG values are lists/sets of dependent nodes (edges)
            num_cdg_edges = sum(len(dependents) for dependents in cdg.values())
            if num_cdg_nodes > 1:
                cdg_density = num_cdg_edges / (num_cdg_nodes * (num_cdg_nodes - 1))

        # Data Dependence Graph (DDG) statistics
        if (
            hasattr(self.digester, "project_data_dependence_graph")
            and self.digester.project_data_dependence_graph
        ):
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

    def _score_candidate_plan_with_llm(
        self, candidate_plan: List[Phase], graph_stats: Dict[str, Any]
    ) -> float:
        # Check if LLM scoring is enabled and configured
        if not self.scorer_model_config or not self.scorer_model_config.get(
            "enabled", True
        ):
            print(
                "PhasePlanner Info: Real LLM scorer is disabled in config. Falling back to random scoring."
            )
            return random.uniform(0.2, 0.8)

        scorer_model_path = self.scorer_model_config.get("model_path")
        # Default to a placeholder if path not provided, get_llm_score_for_text will handle actual existence check.
        if not scorer_model_path:
            scorer_model_path = "./models/placeholder_scorer.gguf"
            print(
                f"PhasePlanner Warning: No 'model_path' in scorer_model_config. Defaulting to placeholder '{scorer_model_path}' for get_llm_score_for_text."
            )

        # Check if the placeholder path actually exists if it's the one being used.
        # get_llm_score_for_text also does this, but an early check here can provide clearer context for fallback.
        if (
            scorer_model_path == "./models/placeholder_scorer.gguf"
            and not Path(scorer_model_path).is_file()
        ):
            print(
                f"PhasePlanner Warning: Default scorer model placeholder '{scorer_model_path}' not found. Falling back to random scoring."
            )
            return random.uniform(0.2, 0.8)

        if not candidate_plan:
            return 0.05  # Return a low score for an empty plan

        # Prepare prompt components (similar to existing logic)
        plan_str_parts = []
        for i, phase in enumerate(candidate_plan):
            try:
                params_json = json.dumps(phase.parameters)
            except TypeError:
                params_json = str(phase.parameters)
            plan_str_parts.append(
                f"Phase {i + 1} (Operation: {phase.operation_name}):\n"
                f"  Target File: {phase.target_file or 'N/A'}\n"
                f"  Parameters: {params_json}\n"
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
        llm_n_ctx = self.scorer_model_config.get(
            "n_ctx", 2048
        )  # Default from get_llm_score_for_text
        llm_temperature = self.scorer_model_config.get(
            "temperature", 0.1
        )  # Default from get_llm_score_for_text
        max_tokens_score = self.scorer_model_config.get("max_tokens_for_score", 16)

        # Call the new LLM interfacer function
        llm_score = get_llm_score_for_text(
            model_path=scorer_model_path,
            prompt=prompt,
            verbose=llm_verbose,
            n_gpu_layers=llm_n_gpu_layers,
            n_ctx=llm_n_ctx,
            max_tokens_for_score=max_tokens_score,
            temperature=llm_temperature,
        )

        if llm_score is not None:
            # Optional: Validate or clamp score to expected range [0.0, 1.0]
            if not (0.0 <= llm_score <= 1.0):
                print(
                    f"PhasePlanner Warning: LLM score {llm_score:.4f} is outside the expected [0.0, 1.0] range. Clamping."
                )
                llm_score = max(0.0, min(1.0, llm_score))
            print(f"PhasePlanner: LLM Scorer returned score: {llm_score:.4f}")
            return llm_score
        else:
            print(
                "PhasePlanner Warning: Failed to get score from LLM via get_llm_score_for_text. Falling back to default low score for this candidate."
            )
            return 0.1  # Default low score on failure

    def _beam_search_for_plan(
        self, spec: Spec, graph_stats: Dict[str, Any]
    ) -> List[Phase]:
        # Initial beam: list of (plan_phases, score)
        beam: List[Tuple[List[Phase], float]] = [([], 0.0)]

        if not spec.operations:
            return []

        # Iterate through each operation defined in the input spec
        for op_idx, op_spec_item_from_input_spec in enumerate(
            spec.operations
        ):  # Renamed for clarity
            next_beam_candidates: List[Tuple[List[Phase], float]] = []

            candidate_op_spec_items = self._suggest_alternative_operations(
                op_spec_item_from_input_spec, spec
            )

            for current_candidate_op_item in candidate_op_spec_items:
                op_name = current_candidate_op_item.get("name")
                if not op_name or not isinstance(
                    op_name, str
                ):  # Ensure op_name is a string
                    print(
                        f"Warning: Candidate operation item (derived from input spec index {op_idx}) is missing a 'name' or 'name' is not a string: {current_candidate_op_item}. Skipping this candidate."
                    )
                    continue

                operation_instance = self.refactor_op_map.get(op_name)
                if not operation_instance:
                    print(
                        f"Warning: Operation '{op_name}' (from candidate derived from input spec index {op_idx}) not found in refactor grammar. Skipping this candidate."
                    )
                    continue

                phase_parameters = {
                    k: v
                    for k, v in current_candidate_op_item.items()
                    if k not in ["name", "target_file"]
                }
                target_file_for_phase = current_candidate_op_item.get("target_file")
                if target_file_for_phase is not None and not isinstance(
                    target_file_for_phase, str
                ):
                    print(
                        f"Warning: 'target_file' for operation '{op_name}' (from candidate derived from input spec index {op_idx}) is not a string: {target_file_for_phase}. Treating as None."
                    )
                    target_file_for_phase = None

                if not operation_instance.validate_parameters(phase_parameters):
                    print(
                        f"Warning: Parameters for operation '{op_name}' (from candidate derived from input spec index {op_idx}, target: {target_file_for_phase or 'repo-level'}) are invalid. This candidate operation will not be added to plans."
                    )
                    continue

                # For each current plan in the beam, try to extend it with the current candidate operation
                for current_plan_phases, current_plan_score in beam:
                    new_phase = Phase(
                        operation_name=op_name,  # From current_candidate_op_item
                        target_file=target_file_for_phase,  # From current_candidate_op_item
                        parameters=phase_parameters,  # From current_candidate_op_item
                        description=f"Op (Input Spec Idx {op_idx + 1}/{len(spec.operations)} - Candidate: '{op_name}'): {operation_instance.description} on '{target_file_for_phase or 'repo-level'}'",
                    )

                    extended_plan_phases = current_plan_phases + [new_phase]
                extended_score = self._score_candidate_plan_with_llm(
                    extended_plan_phases, graph_stats
                )
                next_beam_candidates.append((extended_plan_phases, extended_score))

            if not next_beam_candidates:
                if not beam:
                    print(
                        f"Warning: Beam became empty while processing spec operation index {op_idx} ('{op_name}')."
                    )
                    return []
                # If beam was not empty, but no candidates for this op_spec_item (e.g. invalid params made us 'continue'),
                # the current 'beam' (from previous step) will be used for the next op_spec_item.
                # This is implicitly handled as 'beam' is only updated if next_beam_candidates is non-empty.
                print(
                    f"Info: No valid new phases generated for spec operation index {op_idx} ('{op_name}'). Current beam carries over."
                )

            if next_beam_candidates:
                next_beam_candidates.sort(key=lambda x: x[1], reverse=True)
                beam = next_beam_candidates[: self.beam_width]

            if not beam:
                print(
                    f"Warning: Beam search resulted in an empty beam after processing spec operation index {op_idx} ('{op_name}')."
                )
                return []

        if not beam:
            print("Warning: Beam search did not find any valid plan.")
            return []

        best_plan_phases, best_score = beam[0]
        num_phases_in_spec = len(spec.operations)
        num_phases_in_plan = len(best_plan_phases)

        print(
            f"Beam search completed. Best plan score: {best_score:.4f}. Spec ops: {num_phases_in_spec}, Plan phases: {num_phases_in_plan}."
        )
        if num_phases_in_plan < num_phases_in_spec:
            print(
                f"Warning: The generated plan has fewer phases ({num_phases_in_plan}) than specified operations ({num_phases_in_spec}). This might be due to invalid spec items."
            )

        return best_plan_phases

    def plan_phases(
        self, spec: Spec, workflow_type: Optional[str] = None
    ) -> List[Phase]:
        """Generate a plan and execute it using the chosen workflow."""
        best_plan = self.generate_plan_from_spec(spec)
        workflow = workflow_type or self.workflow_type

        predicted_core = None
        if best_plan and workflow in {
            "routing",
            "orchestrator-workers",
            "evaluator-optimizer",
        }:
            if self.core_predictor and self.core_predictor.is_ready:
                # Prepare raw_features for CorePredictor
                raw_features_for_predictor = {
                    "num_operations": len(spec.operations),
                    "num_target_symbols": len(
                        spec.target_files
                    ),  # Use spec.target_files
                    "num_input_code_lines": 0,  # Placeholder
                    "num_parameters_in_op": 0,  # Placeholder
                }
                op_counts = {
                    op_name: 0 for op_name in self.core_predictor.KNOWN_OPERATION_TYPES
                }
                for (
                    op_detail
                ) in spec.operations:  # spec.operations is List[Dict[str,Any]]
                    op_name = op_detail.get("name", "unknown_operation")
                    if op_name in op_counts:
                        op_counts[op_name] += 1
                    else:
                        op_counts["unknown_operation"] += 1
                raw_features_for_predictor.update(op_counts)

                if self.verbose:
                    print(
                        f"PhasePlanner: Raw features for CorePredictor: {raw_features_for_predictor}"
                    )

                predicted_core = self.core_predictor.predict(raw_features_for_predictor)
                if self.verbose:
                    print(
                        f"PhasePlanner: CorePredictor predicted preferred core: {predicted_core}"
                    )
            elif self.verbose:
                print(
                    "PhasePlanner: CorePredictor not ready or not available. Skipping core prediction."
                )

        if best_plan:
            # self.plan_cache[cache_key] = best_plan # Caching is now done within generate_plan_from_spec
            # print("PhasePlanner: Plan generated and cached.") # Logging also moved

            print(
                "\n--- Running CollaborativeAgentGroup for each phase in the generated plan ---"
            )
            execution_summary = []  # To store results of agent_group.run for each phase
            for phase_obj in best_plan:
                print(
                    f"Executing phase: {phase_obj.operation_name} on {phase_obj.target_file or 'repo-level'}"
                )
                try:
                    workflow_core = (
                        predicted_core
                        if workflow
                        in {"routing", "orchestrator-workers", "evaluator-optimizer"}
                        else None
                    )

                    if workflow == "parallelization":
                        llm_script, llm_src = self.agent_group.run(
                            phase_ctx=phase_obj,
                            digester=self.digester,
                            validator_handle=self.validator_handle,
                            score_style_handle=self.score_style_handle,
                            predicted_core="LLMCore",
                            workflow=workflow,
                        )
                        diff_script, diff_src = self.agent_group.run(
                            phase_ctx=phase_obj,
                            digester=self.digester,
                            validator_handle=self.validator_handle,
                            score_style_handle=self.score_style_handle,
                            predicted_core="DiffusionCore",
                            workflow=workflow,
                        )
                        score_llm = (
                            self.style_validator.score_patch_script_content(
                                llm_script or "",
                                self.project_root_path,
                                self.naming_conventions_db_path,
                            )
                            if llm_script
                            else -1.0
                        )
                        score_diff = (
                            self.style_validator.score_patch_script_content(
                                diff_script or "",
                                self.project_root_path,
                                self.naming_conventions_db_path,
                            )
                            if diff_script
                            else -1.0
                        )
                        if score_diff > score_llm:
                            validated_patch_script, patch_source = diff_script, diff_src
                        else:
                            validated_patch_script, patch_source = llm_script, llm_src
                    else:
                        validated_patch_script, patch_source = self.agent_group.run(
                            phase_ctx=phase_obj,
                            digester=self.digester,
                            validator_handle=self.validator_handle,
                            score_style_handle=self.score_style_handle,
                            predicted_core=workflow_core,
                            workflow=workflow,
                        )

                    if workflow == "evaluator-optimizer" and validated_patch_script:
                        score = get_llm_score_for_text(
                            validated_patch_script,
                            model_path=self.scorer_model_config.get("model_path"),
                            n_ctx=self.scorer_model_config.get("n_ctx"),
                            temperature=self.scorer_model_config.get("temperature"),
                            verbose=self.verbose,
                        )
                        if score is not None and score < 0.5:
                            validated_patch_script = (
                                self.agent_group.llm_agent.polish_patch(
                                    validated_patch_script, {}
                                )
                            )

                    if validated_patch_script:
                        print(
                            f"Phase {phase_obj.operation_name}: Successfully generated patch script (source: {patch_source}). Preview: {str(validated_patch_script)[:100]}..."
                        )
                        execution_summary.append(
                            {
                                "phase_operation": phase_obj.operation_name,
                                "target": phase_obj.target_file,
                                "status": "success",
                                "patch_preview": str(validated_patch_script)[:100],
                                "patch_source": patch_source,
                            }
                        )
                        # --- Start: Real Patch Application & CommitBuilder Integration ---
                        target_file_path_str = phase_obj.target_file
                        # target_file_path is the relative path from project_root
                        target_file_path = (
                            Path(target_file_path_str) if target_file_path_str else None
                        )
                        modified_content: Optional[str] = None
                        diff_summary = "Diff generation skipped: No target file or patch application failed."

                        if not target_file_path:
                            print(
                                f"PhasePlanner Warning: Phase target_file is None for phase '{phase_obj.operation_name}'. Cannot apply patch or proceed with CommitBuilder."
                            )
                            execution_summary.append(
                                {
                                    "phase_operation": phase_obj.operation_name,
                                    "target": "None",
                                    "status": "error",
                                    "error_message": "Target file was None for CommitBuilder step",
                                    "patch_source": patch_source,
                                }
                            )
                            continue  # Skip to next phase_obj

                        abs_target_file_path = self.project_root_path / target_file_path
                        original_content = self.digester.get_file_content(
                            abs_target_file_path
                        )
                        if original_content is None:
                            original_content = ""
                            if self.verbose:
                                print(
                                    f"PhasePlanner: Target file '{abs_target_file_path}' is new or content not found. Applying patch to empty content."
                                )

                        try:
                            if self.verbose:
                                print(
                                    f"PhasePlanner: Applying validated LibCST script to '{abs_target_file_path}' (source: {patch_source})..."
                                )
                            modified_content = apply_libcst_codemod_script(
                                original_content, validated_patch_script
                            )
                            if self.verbose:
                                print(
                                    f"PhasePlanner: LibCST script applied successfully to '{target_file_path}'."
                                )

                            diff_lines = list(
                                difflib.unified_diff(
                                    original_content.splitlines(keepends=True),
                                    modified_content.splitlines(keepends=True),
                                    fromfile=f"a/{target_file_path.name}",
                                    tofile=f"b/{target_file_path.name}",
                                )
                            )
                            if diff_lines:
                                diff_summary = "".join(diff_lines)
                            else:
                                diff_summary = f"No textual changes detected for '{target_file_path}' after patch application."
                            if self.verbose:
                                print(
                                    f"PhasePlanner: Diff summary generated for '{target_file_path}' (length {len(diff_summary)})."
                                )

                        except PatchApplicationError as pae:
                            print(
                                f"PhasePlanner Error: Failed to apply validated patch script for {abs_target_file_path}: {pae}"
                            )
                            # Update execution_summary for this specific failure point
                            last_summary_item = execution_summary[-1]
                            last_summary_item["status"] = "error"
                            last_summary_item["error_message"] = (
                                f"Patch application failed: {pae}"
                            )
                            continue  # Skip CommitBuilder for this phase
                        except Exception as e_apply:
                            print(
                                f"PhasePlanner Error: Unexpected error during patch application or diff for {abs_target_file_path}: {e_apply}"
                            )
                            last_summary_item = execution_summary[-1]
                            last_summary_item["status"] = "error"
                            last_summary_item["error_message"] = (
                                f"Unexpected patch application/diff error: {e_apply}"
                            )
                            continue  # Skip CommitBuilder for this phase

                        if (
                            modified_content is None
                        ):  # Should be caught by exceptions above, but as a safeguard
                            print(
                                f"PhasePlanner: Skipping CommitBuilder for phase targeting '{target_file_path}' due to no modified content."
                            )
                            last_summary_item = execution_summary[-1]
                            last_summary_item["status"] = "skipped"
                            last_summary_item["reason"] = (
                                "No modified content after application attempt"
                            )
                            continue

                        validated_patch_content_map = {
                            target_file_path: modified_content
                        }  # Key is relative path

                        validator_results_summary = "Validation results summary not available (agent history structure might have changed or was empty)."
                        if self.agent_group.patch_history:
                            try:
                                last_attempt_info = self.agent_group.patch_history[-1]
                                last_error_tb_info = (
                                    last_attempt_info[3]
                                    if len(last_attempt_info) > 3
                                    else "N/A"
                                )
                                validator_results_summary = f"Final validation in agent: Valid={last_attempt_info[1]}, Score={last_attempt_info[2]:.2f}, Last Error='{str(last_error_tb_info)[:50]}...'"
                            except Exception as e_hist:
                                print(
                                    f"PhasePlanner: Error accessing patch_history for validator summary: {e_hist}"
                                )

                        branch_name_suffix = (
                            getattr(spec, "issue_id", None)
                            or spec.issue_description[:20].replace(" ", "-").lower()
                        )
                        branch_name = f"feature/auto-patch-{branch_name_suffix}-{random.randint(1000, 9999)}"
                        commit_title = f"Auto-apply patch for '{spec.issue_description[:40]}...' (Source: {patch_source})"

                        if self.verbose:
                            print(
                                "PhasePlanner: Calling CommitBuilder.process_and_submit_patch..."
                            )
                        saved_patch_set_path = (
                            self.commit_builder.process_and_submit_patch(
                                validated_patch_content_map=validated_patch_content_map,
                                spec=spec,
                                diff_summary=diff_summary,
                                validator_results_summary=validator_results_summary,
                                branch_name=branch_name,
                                commit_title=commit_title,
                                project_root=self.project_root_path,
                                patch_source=patch_source,
                            )
                        )

                        if saved_patch_set_path:
                            print(
                                f"PhasePlanner: CommitBuilder successfully saved patch set to: {saved_patch_set_path}"
                            )
                            # --- Log to Success Memory ---
                            if (
                                self.data_dir_path and validated_patch_script
                            ):  # Ensure there's a script and path
                                log_success = log_successful_patch(
                                    data_directory=self.data_dir_path,
                                    spec=spec,  # The overall spec for the plan
                                    diff_summary=diff_summary,
                                    successful_script_str=validated_patch_script,
                                    patch_source=patch_source,
                                    predicted_core=predicted_core,
                                    verbose=self.verbose,
                                )
                                if log_success:
                                    if self.verbose:
                                        print(
                                            f"PhasePlanner: Successfully logged patch to success memory in {self.data_dir_path}."
                                        )
                                elif (
                                    self.verbose
                                ):  # Only print warning if verbose, as it's non-critical
                                    print(
                                        f"PhasePlanner Warning: Failed to log patch to success memory in {self.data_dir_path}."
                                    )
                            elif self.verbose:
                                print(
                                    "PhasePlanner: Success memory logging skipped (data_dir_path not set or no script)."
                                )
                            # --- End Log to Success Memory ---
                            if self.verbose:
                                print(
                                    "PhasePlanner: Breaking after first successful phase patch processing for this subtask."
                                )
                            break  # Process only the first successful phase
                        else:  # CommitBuilder failed to save
                            print(
                                f"PhasePlanner: CommitBuilder failed to save patch set for phase {phase_obj.operation_name}."
                            )
                            # Ensure last_summary_item is correctly referenced if execution_summary was just updated for success
                            if (
                                execution_summary
                                and execution_summary[-1]["status"] == "success"
                            ):
                                execution_summary[-1][
                                    "status"
                                ] = "error"  # Update status
                                execution_summary[-1][
                                    "error_message"
                                ] = "CommitBuilder failed to save patch set"
                            else:  # If summary was not updated for success yet, add new error entry
                                execution_summary.append(
                                    {
                                        "phase_operation": phase_obj.operation_name,
                                        "target": phase_obj.target_file,
                                        "status": "error",
                                        "error_message": "CommitBuilder failed to save patch set after successful patch generation",
                                        "patch_source": patch_source,
                                    }
                                )
                        # --- End: Real Patch Application & CommitBuilder Integration ---
                    else:  # validated_patch_script is None
                        print(
                            f"Phase {phase_obj.operation_name}: Failed to generate a patch script (returned None). Source info: {patch_source}"
                        )
                        execution_summary.append(
                            {
                                "phase_operation": phase_obj.operation_name,
                                "target": phase_obj.target_file,
                                "status": "failed_no_patch",
                                "patch_source": patch_source,
                            }
                        )

                except PhaseFailure as pf_e:
                    print(
                        f"PhasePlanner: CollaborativeAgentGroup reported PhaseFailure for phase {phase_obj.operation_name}: {pf_e}"
                    )
                    execution_summary.append(
                        {
                            "phase_operation": phase_obj.operation_name,
                            "target": phase_obj.target_file,
                            "status": "PhaseFailure",
                            "error_message": str(pf_e),
                        }
                    )
                    print("PhasePlanner: Stopping plan execution due to PhaseFailure.")
                    break
                except Exception as e:
                    print(
                        f"Error running agent group for phase {phase_obj.operation_name}: {type(e).__name__} - {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    execution_summary.append(
                        {
                            "phase_operation": phase_obj.operation_name,
                            "target": phase_obj.target_file,
                            "status": "error",
                            "error_message": str(e),
                        }
                    )

            print(
                "--- CollaborativeAgentGroup execution finished for all phases (or broke early) ---"
            )
            print("Execution Summary (from planner):")
            for summary_item in execution_summary:
                print(f"  - {summary_item}")
            # The method still returns the plan (List[Phase]).
            # The execution_summary is for logging/observation at this stage.
        else:
            # Message moved to generate_plan_from_spec
            # print("PhasePlanner: Failed to generate a plan (beam search returned empty).")
            print(
                "PhasePlanner.plan_phases: No plan generated by generate_plan_from_spec. Nothing to execute."
            )
        return best_plan  # Return the list of Phase objects as per original signature


if __name__ == "__main__":
    # No MagicMock needed now as we are using more concrete (though still mock) classes
    # from unittest.mock import MagicMock
    from src.utils.config_loader import load_app_config  # For __main__
    from src.digester.repository_digester import RepositoryDigester  # For __main__

    print("--- PhasePlanner Example Usage (Conceptual) ---")

    # Using the actual RepositoryDigester mock defined at the class level for TYPE_CHECKING
    # This ensures methods like get_project_overview are available if called by agent_group
    class MockDigesterForPlannerMain(
        RepositoryDigester
    ):  # Inherit from the placeholder to ensure methods
        def __init__(self):
            super().__init__()  # Call super if it has an init
            self.project_call_graph = {"module.func_a": {"module.func_b"}}  # type: ignore
            self.project_control_dependence_graph = {}  # type: ignore
            self.project_data_dependence_graph = {}

        # Override mock methods if more specific behavior is needed for main example
        def get_project_overview(self) -> Dict[str, Any]:
            print("__main__ MockDigester.get_project_overview called")
            return {"files_in_project": 5, "main_language": "python"}

        def get_file_content(self, path: Path) -> Optional[str]:
            content = f"# __main__ Mock Content for {path}\n# BAD_STYLE example\n# FAIL_VALIDATION example"
            print(
                f"__main__ MockDigester.get_file_content for {path} returning: '{content[:50]}...'"
            )
            return content if path else None

    dummy_project_root_main = Path("_temp_mock_project_root")
    dummy_project_root_main.mkdir(parents=True, exist_ok=True)

    # Create dummy style_fingerprint.json and naming_conventions.db for the test
    with open(dummy_project_root_main / "style_fingerprint.json", "w") as f_style:
        json.dump(
            {"line_length": 100, "indent_style": "space", "indent_width": 4}, f_style
        )
    with open(dummy_project_root_main / "naming_conventions.db", "w") as f_naming:
        # In a real scenario, this would be a SQLite DB or similar. For mock, content doesn't matter.
        f_naming.write("mock_naming_db_content")

    app_cfg_main = load_app_config()  # Load default or from existing config.yaml
    app_cfg_main["general"]["verbose"] = True  # Override for testing
    # Ensure data_dir is within our temp project root for cleanup
    app_cfg_main["general"]["data_dir"] = str(
        dummy_project_root_main / ".agent_data_main_test"
    )

    # Mock model paths in app_config if they don't exist, to prevent download attempts
    # but allow PhasePlanner to initialize its scorer_model_config path.
    # These paths won't actually be used in this conceptual __main__ unless scoring is deeply tested.
    app_cfg_main.setdefault("models", {})
    app_cfg_main["models"]["planner_scorer_gguf"] = str(
        dummy_project_root_main / "mock_scorer.gguf"
    )  # Non-existent, will warn
    app_cfg_main["models"]["agent_llm_gguf"] = str(
        dummy_project_root_main / "mock_agent_llm.gguf"
    )  # Non-existent
    app_cfg_main["models"]["divot5_infill_model_dir"] = str(
        dummy_project_root_main / "mock_divot5_infill"
    )  # Non-existent

    # Instantiate RepositoryDigester with app_cfg_main for its verbose setting
    # In a real scenario, digester would also take app_config if its __init__ was updated.
    # For this test, MockDigesterForPlannerMain doesn't use app_config.
    mock_digester_main = MockDigesterForPlannerMain()

    try:
        planner = PhasePlanner(
            project_root_path=dummy_project_root_main,
            app_config=app_cfg_main,
            digester=mock_digester_main,
            # refactor_op_map and beam_width can be passed to override config for testing
        )

        # Example Spec using the imported spec_model.Spec structure
        example_spec_data_main = {
            "issue_description": "Implement new user registration flow",  # Updated field name
            "target_files": [
                "src/services/user_service.py",
                "src/db/user_queries.py",
            ],  # Updated field name
            "operations": [  # This now matches spec_model.Spec's List[Dict[str,Any]]
                {
                    "name": "add_import",
                    "target_file": "src/services/user_service.py",
                    "import_statement": "from ..db import user_queries",
                },
                {
                    "name": "extract_method",
                    "target_file": "src/services/user_service.py",
                    "source_function_name": "register_user_v1",
                    "start_line": 10,
                    "end_line": 25,
                    "new_method_name": "_validate_user_input",
                },
                {
                    "name": "add_decorator",
                    "target_file": "src/services/user_service.py",
                    "target_function_name": "register_user_v2",
                    "decorator_name": "@transactional",
                },
            ],
            "acceptance_tests": [
                "test_user_registration_success",
                "test_user_registration_existing_user",
            ],  # Updated field name
        }
        spec_instance_main = Spec(**example_spec_data_main)

        print(f"\nPlanning for spec: {spec_instance_main.issue_description}")
        plan_phases_list_main = planner.plan_phases(
            spec_instance_main,
            workflow_type=planner.workflow_type,
        )

        print("\n--- Post-execution: Generated Plan Phases (from planner return) ---")
        if plan_phases_list_main:
            for i, p_main in enumerate(plan_phases_list_main):
                print(f"Details for Planned Phase {i + 1}:")
                if hasattr(p_main, "model_dump_json"):
                    print(p_main.model_dump_json(indent=2))
                else:
                    # Fallback for older/non-Pydantic Phase models
                    print(
                        f"  Operation: {getattr(p_main, 'operation_name', 'N/A')}, Target: {getattr(p_main, 'target_file', 'N/A')}, Params: {getattr(p_main, 'parameters', {})}"
                    )
        else:
            print("  No plan was generated by the planner.")

        # print(f"\nRequesting plan again for the same spec (should be cached):")
        # plan_phases_list_cached = planner.plan_phases(spec_instance_main)
        # print(f"Cached plan request returned {len(plan_phases_list_cached) if plan_phases_list_cached else 0} phases.")
        # Note: Agent group execution would re-run unless caching is implemented after agent execution.

    except ImportError as e_imp:
        print(
            f"ImportError in PhasePlanner __main__ example: {e_imp}. Check imports for Spec, Phase, or other dependencies."
        )
    except Exception as e_main:
        print(
            f"Error in PhasePlanner __main__ example: {type(e_main).__name__}: {e_main}"
        )
        import traceback

        traceback.print_exc()
    finally:
        if dummy_project_root_main.exists():
            import shutil  # Import shutil here for cleanup

            try:
                shutil.rmtree(dummy_project_root_main)
                if app_cfg_main["general"]["verbose"]:
                    print(f"Cleaned up dummy project root: {dummy_project_root_main}")
            except OSError as e_ose:
                print(f"Error removing directory {dummy_project_root_main}: {e_ose}")

    print("\n--- PhasePlanner Example Done ---")
