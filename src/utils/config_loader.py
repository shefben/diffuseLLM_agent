import yaml  # from PyYAML
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_CONFIG_FILENAMES = ["config.yaml", "config.yml"]

# Define a more comprehensive default config structure
DEFAULT_APP_CONFIG: Dict[str, Any] = {
    "general": {
        "verbose": False,
        "project_root": None,  # To be dynamically set by the main application/CLI
        "data_dir": ".agent_data",  # Relative to project_root for success memory, etc.
        "patches_output_dir": ".autopatches",  # Relative to project_root for CommitBuilder output
    },
    "models": {
        # Style Profiling
        "deepseek_style_draft_gguf": "./models/placeholder_deepseek.gguf",
        "divot5_style_refiner_dir": "./models/placeholder_divot5_refiner/",
        "deepseek_style_polish_gguf": "./models/placeholder_deepseek.gguf",  # Can reuse
        "divot5_style_unifier_dir": "./models/placeholder_divot5_unifier/",
        # Spec Normalization
        "t5_spec_normalizer_dir": "./models/placeholder_t5_spec_normalizer/",
        # Planner
        "planner_scorer_gguf": "./models/placeholder_scorer.gguf",
        "operations_llm_gguf": "./models/placeholder_operations.gguf",
        # Agent Core LLM (scaffold, polish, repair, infill if GGUF based)
        "agent_llm_gguf": "./models/placeholder_llm_agent.gguf",  # General agent model
        "repair_llm_gguf": "./models/placeholder_repair_model.gguf",  # Specific repair model (can be same as agent_llm_gguf)
        "infill_llm_gguf": "./models/placeholder_llm_agent.gguf",  # Specific infill model (can be same as agent_llm_gguf)
        "divot5_infill_model_dir": "./models/placeholder_divot5_infill/",  # For DivoT5 FIM
        # Embeddings
        "sentence_transformer_model": "all-MiniLM-L6-v2",  # Default to HF Hub name
        # Core Predictor
        "core_predictor_model": "./models/core_predictor.joblib",
    },
    "agent_infill": {
        "type": "gguf",  # Options: "gguf", "divot5"
    },
    "tools": {
        "ruff_path": "ruff",
        "black_path": "black",
        "pyright_path": "pyright",
        "pytest_path": "pytest",
        "default_pytest_target_dir": "tests",
    },
    "llm_params": {
        "n_gpu_layers": -1,  # Common default, applies if not overridden
        "n_ctx": 4096,  # Common default
        "temperature_default": 0.3,
        "max_tokens_default": 2048,
        # Task-specific overrides (examples, actual keys used by components)
        "scorer_temp": 0.1,
        "scorer_max_tokens": 16,
        "agent_scaffold_temp": 0.3,
        "agent_scaffold_max_tokens": 2048,
        "agent_infill_gguf_temp": 0.4,
        "agent_infill_gguf_max_tokens": 512,  # For GGUF infill in DiffusionCore
        "agent_polish_temp": 0.2,
        "agent_polish_max_tokens": 2048,
        "agent_repair_temp": 0.3,
        "agent_repair_max_tokens": 2048,  # For LLMCore repair
    },
    "active_learning": {
        "enabled": False,
        "model_name": "gpt2",
        "output_dir": "./lora_adapters",
        "interval": 3600,
        "predictor_model": "./models/core_predictor.joblib",
    },
    "mcp": {
        "default_tool": "basic",
        "tools": [
            {
                "name": "basic",
                "prompt": "You are a helpful assistant generating high quality Python patches."
            }
        ],
        "workflow_settings": {},
        "agent_defaults": {},
    },
    "webui": {"port": 5001},
    "divot5_fim_params": {  # Parameters specific to DivoT5 Fill-In-Middle models
        "infill_max_length": 256,
        "infill_num_beams": 3,
        "infill_temperature": 0.5,
        "top_p": 0.9,  # Example, common DivoT5 param
        # Add other relevant DivoT5 FIM parameters here
    },
}


def merge_configs(
    base_config: Dict[str, Any], user_config: Dict[str, Any]
) -> Dict[str, Any]:
    merged = base_config.copy()
    for key, value in user_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_app_config(config_file_path: Optional[Path] = None) -> Dict[str, Any]:
    # Start with defaults
    current_config = (
        DEFAULT_APP_CONFIG.copy()
    )  # Use deepcopy if nested dicts are modified in place later

    file_to_load: Optional[Path] = None

    if config_file_path and config_file_path.is_file():
        file_to_load = config_file_path
    else:
        for filename in DEFAULT_CONFIG_FILENAMES:
            default_path = Path.cwd() / filename
            if default_path.is_file():
                file_to_load = default_path
                break

    if file_to_load:
        try:
            with open(file_to_load, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
            if user_config:
                current_config = merge_configs(current_config, user_config)
            print(f"ConfigLoader: Loaded configuration from {file_to_load}")
        except yaml.YAMLError as e_yaml:
            print(
                f"ConfigLoader Warning: Error parsing YAML config file {file_to_load}: {e_yaml}. Using defaults."
            )
        except Exception as e_gen:
            print(
                f"ConfigLoader Warning: Error loading config file {file_to_load}: {e_gen}. Using defaults."
            )
    else:
        print(
            "ConfigLoader Info: No user config file provided or found in default locations. Using built-in defaults."
        )

    # Note: project_root is expected to be set by the calling application (e.g., CLI)
    # and then paths like data_dir can be made absolute.
    # Example:
    # if current_config["general"]["project_root"]:
    #     project_root = Path(current_config["general"]["project_root"]).resolve()
    #     current_config["general"]["project_root"] = str(project_root)
    #     for key in ["data_dir", "patches_output_dir"]:
    #         if current_config["general"].get(key) and not Path(current_config["general"][key]).is_absolute():
    #             current_config["general"][key] = str(project_root / current_config["general"][key])

    return current_config


def get_default_config_yaml_example() -> str:
    # PyYAML typically sorts keys by default, which is fine for an example.
    return yaml.dump(DEFAULT_APP_CONFIG, sort_keys=False, indent=2)


if __name__ == "__main__":  # Basic test
    print("--- Default Config Example ---")
    print(get_default_config_yaml_example())

    print("\n--- Loading Config (no file provided, using defaults) ---")
    loaded_cfg = load_app_config()
    print(f"Verbose from loaded_cfg: {loaded_cfg.get('general', {}).get('verbose')}")
    print(
        f"Default sentence transformer: {loaded_cfg.get('models', {}).get('sentence_transformer_model')}"
    )

    # Create a dummy config.yaml for testing load from file
    dummy_cfg_path = Path("temp_config_test.yaml")
    dummy_user_config = {
        "general": {"verbose": True, "project_root": "/tmp/my_project"},
        "models": {"planner_scorer_gguf": "./custom_models/scorer.gguf"},
        "llm_params": {"agent_repair_temp": 0.45},
    }
    with open(dummy_cfg_path, "w") as f:
        yaml.dump(dummy_user_config, f)

    print(f"\n--- Loading Config (from {dummy_cfg_path}) ---")
    loaded_from_file_cfg = load_app_config(dummy_cfg_path)

    print(
        f"Verbose from file: {loaded_from_file_cfg.get('general', {}).get('verbose')}"
    )
    print(
        f"Project root from file: {loaded_from_file_cfg.get('general', {}).get('project_root')}"
    )
    print(
        f"Scorer model from file: {loaded_from_file_cfg.get('models', {}).get('planner_scorer_gguf')}"
    )
    print(
        f"Agent LLM (default): {loaded_from_file_cfg.get('models', {}).get('agent_llm_gguf')}"
    )  # Should be default
    print(
        f"Agent repair temp from file: {loaded_from_file_cfg.get('llm_params', {}).get('agent_repair_temp')}"
    )  # User
    print(
        f"Agent scaffold temp (default): {loaded_from_file_cfg.get('llm_params', {}).get('agent_scaffold_temp')}"
    )  # Default

    if dummy_cfg_path.exists():
        dummy_cfg_path.unlink()
        print(f"\nCleaned up {dummy_cfg_path}")

    # Test loading from default location
    default_test_cfg_path = Path.cwd() / "config.yaml"
    with open(default_test_cfg_path, "w") as f:
        yaml.dump({"general": {"verbose": "DEBUG_MODE"}}, f)
    print(f"\n--- Loading Config (from default {default_test_cfg_path}) ---")
    loaded_from_default_cfg = load_app_config()  # No path given, should find it
    print(
        f"Verbose from default file: {loaded_from_default_cfg.get('general', {}).get('verbose')}"
    )
    if default_test_cfg_path.exists():
        default_test_cfg_path.unlink()
        print(f"Cleaned up {default_test_cfg_path}")
