# src/spec_normalizer/t5_client.py
from pathlib import Path
from typing import Optional, Dict, Any

from .spec_normalizer_interface import SpecNormalizerModelInterface

# Guarded import for Hugging Face Transformers and PyTorch
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
except ImportError:
    torch = None  # type: ignore
    T5ForConditionalGeneration = None  # type: ignore
    T5TokenizerFast = None  # type: ignore
    print(
        "Warning: PyTorch or Hugging Face Transformers not found. T5Client will not function with real models."
    )


class T5Client(SpecNormalizerModelInterface):
    # NOTE: This T5Client uses a standard T5ForConditionalGeneration model.
    # Phase 3 of the original plan mentions a LoRA-adapted Tiny-T5 diffusion pipeline
    # for the SpecFusion component, which this client serves. The LoRA adaptation
    # and specific diffusion pipeline aspects are not implemented within this client itself.
    def __init__(self, app_config: Dict[str, Any]):
        """
        Initializes the T5Client.

        Args:
            app_config: Application configuration dictionary.
                        Expected keys:
                        - general.verbose (bool, optional)
                        - models.t5_spec_normalizer_dir (str, required)
        """
        self.app_config = app_config
        self.verbose = self.app_config.get("general", {}).get("verbose", False)

        models_config = self.app_config.get("models", {})
        self.model_path_or_name = models_config.get("t5_spec_normalizer_dir")

        if self.model_path_or_name is None:
            print(
                "T5Client Critical Error: 'models.t5_spec_normalizer_dir' not found in app_config!"
            )
            self.model_path_or_name = "./models/ERROR_PATH_NOT_IN_CONFIG"  # Ensure _load_model fails gracefully

        self.model: Optional[T5ForConditionalGeneration] = None
        self.tokenizer: Optional[T5TokenizerFast] = None
        self.device: Optional[str] = None
        self._is_ready: bool = False  # Renamed attribute
        self._load_model()

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _load_model(self) -> None:
        """
        Attempts to load the T5 model and tokenizer from the specified path or Hugging Face Hub.
        Sets `self.is_ready` to True on success, False on failure.
        """
        if (
            T5ForConditionalGeneration is None
            or T5TokenizerFast is None
            or torch is None
        ):
            print(
                "T5Client Error: Required libraries (Transformers/PyTorch) not installed. Model loading aborted."
            )
            self._is_ready = False  # Use renamed attribute
            return

        actual_model_path: str
        default_local_t5_path = Path("./models/placeholder_t5_spec_normalizer/")

        potential_path = Path(self.model_path_or_name)

        if potential_path.is_dir():  # Check if it's an existing directory first
            actual_model_path = str(potential_path.resolve())
            if self.verbose:
                print(
                    f"T5Client Info: Attempting to load model from provided local directory: {actual_model_path}"
                )
        elif (
            self.model_path_or_name == str(default_local_t5_path)
            or self.model_path_or_name.endswith("placeholder_t5_spec_normalizer")
            or self.model_path_or_name == "t5-small"
        ):  # Common default or placeholder names
            if default_local_t5_path.is_dir():
                actual_model_path = str(default_local_t5_path.resolve())
                print(
                    f"T5Client Info: Using default local T5 model directory: {actual_model_path}"
                )
            else:
                # If it was a placeholder name but default local path doesn't exist, try HF Hub with the original name
                actual_model_path = self.model_path_or_name
                print(
                    f"T5Client Warning: Default local path '{default_local_t5_path}' not found. Attempting to load '{actual_model_path}' from Hugging Face Hub."
                )
        else:  # Treat as HF identifier
            actual_model_path = self.model_path_or_name
            if self.verbose:
                print(
                    f"T5Client Info: Attempting to load model '{actual_model_path}' from Hugging Face Hub."
                )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.verbose:
            print(f"T5Client Info: Using device: {self.device}")

        try:
            if self.verbose:
                print(f"T5Client Info: Loading tokenizer from '{actual_model_path}'...")
            self.tokenizer = T5TokenizerFast.from_pretrained(actual_model_path)

            if self.verbose:
                print(
                    f"T5Client Info: Loading model from '{actual_model_path}' to {self.device}..."
                )
            self.model = T5ForConditionalGeneration.from_pretrained(
                actual_model_path
            ).to(self.device)  # type: ignore
            self.model.eval()  # type: ignore

            self._is_ready = True  # Use renamed attribute
            print(
                f"T5Client Info: Model '{actual_model_path}' loaded successfully on {self.device}."
            )
        except Exception as e:
            print(
                f"T5Client Error: Failed to load model/tokenizer from '{actual_model_path}': {e}"
            )
            self._is_ready = False  # Use renamed attribute
            self.model = None
            self.tokenizer = None

    def generate_spec_yaml(
        self,
        raw_issue_text: str,
        context_symbols_string: Optional[str] = None,
        mcp_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generates a YAML specification string from raw issue text and context.
        Returns None if generation fails.

        Args:
            raw_issue_text: The raw text description of the issue.
            context_symbols_string: An optional string containing context symbols like FQNs.

        Returns:
            A string containing the YAML specification, or None if an error occurs or model is not ready.
        """
        if (
            not self.is_ready or self.model is None or self.tokenizer is None
        ):  # Property is_ready used here
            print(
                "T5Client Error: Model not ready or not loaded. Cannot process request."
            )
            return None

        # Enhanced prompt for better YAML spec generation
        prompt = f"""Translate the following software development issue and context into a structured YAML specification.

Your output must be a single YAML block. Do not include any text before or after the YAML.
The YAML should conform to the following structure:
issue_description: "A concise summary of the issue, derived from the input."
target_files:
  - "path/to/relevant/file1.py"
  - "path/to/another/file.py"
  # List source files that likely need changes.
operations:
  - name: "operation_name" # e.g., add_function, modify_class, fix_bug, refactor_method, update_docs
    target_file: "path/to/file_for_this_operation.py" # Optional, can be same as in target_files or different if op is specific
    # Parameters specific to the operation 'name'. Examples:
    # For 'add_function':
    #   function_name: "my_new_function"
    #   parameters: {{"arg1": "int", "arg2": "Optional[str]"}} # Type hints as strings
    #   return_type: "bool"
    #   body: "pass # TODO: Implement actual logic" # Brief placeholder
    # For 'modify_class':
    #   class_name: "MyExistingClass"
    #   changes:
    #     - type: "add_method"
    #       method_name: "new_utility_method"
    #       parameters: {{"data": "List[Dict]"}}
    #       return_type: "None"
    #     - type: "change_attribute"
    #       attribute_name: "some_attribute"
    #       new_value: "new_default_value_if_applicable"
    # For 'fix_bug':
    #   description: "Detailed description of the fix to be applied."
    #   target_element: "function_or_method_name_if_specific" # e.g. "validate_input"
  # Add more operations as needed based on the issue.
acceptance_tests:
  - "Human-readable description of test case 1 to verify the fix."
  - "Description of test case 2, e.g., covering edge cases."
  # These should be understandable criteria for when the task is considered complete.

Context symbols from the codebase (e.g., relevant functions, classes):
{context_symbols_string if context_symbols_string else "None available"}.

Issue Description to process:
{raw_issue_text}

Generate the YAML:
        """
        if mcp_prompt:
            prompt = mcp_prompt + "\n" + prompt

        if self.verbose:
            print(
                f"\nT5Client: Sending prompt to T5 model (length: {len(prompt)}):\n{prompt[:500]}..."
            )

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding="longest",
            ).to(self.device)

            generation_config = {
                "max_length": 512,
                "num_beams": 4,
                "early_stopping": True,
                # Consider adding: "temperature", "top_p", "top_k" for more diverse/controlled generation
            }
            if self.verbose:
                print(f"T5Client Info: Using generation config: {generation_config}")

            outputs = self.model.generate(inputs.input_ids, **generation_config)  # type: ignore

            yaml_spec_string = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()

            if self.verbose:
                print(
                    f"T5Client Info: Raw model output (YAML spec string):\n{yaml_spec_string}"
                )

            return yaml_spec_string
        except Exception as e:
            print(f"T5Client Error: Exception during T5 model inference: {e}")
            return None


if __name__ == "__main__":
    from src.utils.config_loader import (
        load_app_config,
        DEFAULT_APP_CONFIG,
    )  # For __main__

    print("--- T5Client Example Usage ---")

    app_cfg_main = load_app_config()  # Load defaults or from a test file
    app_cfg_main["general"]["verbose"] = True  # Example override

    # Ensure the relevant model path is set in app_cfg for testing.
    # If you have a local placeholder or want to test with "t5-small":
    # Option 1: Use the default from DEFAULT_APP_CONFIG if suitable
    # app_cfg_main["models"]["t5_spec_normalizer_dir"] = DEFAULT_APP_CONFIG["models"]["t5_spec_normalizer_dir"]
    # Option 2: Explicitly set for testing (e.g., if default placeholder doesn't exist and you want to use t5-small)
    if not Path(DEFAULT_APP_CONFIG["models"]["t5_spec_normalizer_dir"]).exists():
        app_cfg_main["models"]["t5_spec_normalizer_dir"] = (
            "t5-small"  # Fallback to HF download for test
        )

    print(
        f"\nAttempting to initialize T5Client with app_config (model: {app_cfg_main['models']['t5_spec_normalizer_dir']})"
    )
    t5_client = T5Client(app_config=app_cfg_main)

    if t5_client.is_ready:
        print("\nT5Client ready. Requesting spec...")
        issue_text = "The user login fails when the password contains special characters like '!@#$%'. Need to update the validation logic in 'src/auth/validation.py' to handle these cases correctly. The function `validate_password` should be checked."
        context_symbols = "src.auth.validation.validate_password, src.user.models.User"

        yaml_output = t5_client.request_spec_from_text(issue_text, context_symbols)

        if yaml_output:
            print("\n--- Generated YAML Spec ---")
            print(yaml_output)
            print("-------------------------")
        else:
            print("\nFailed to generate YAML spec from T5Client.")
    else:
        print(
            "\nT5Client initialization failed or model not ready. Cannot request spec."
        )
        print(
            "This is expected if a real T5 model is not available at the specified/default paths or if Transformers/PyTorch are not installed."
        )

    print("\n--- T5Client Example Done ---")
