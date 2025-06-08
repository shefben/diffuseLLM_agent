# src/spec_normalizer/t5_client.py
from typing import Optional, Dict, Any, Tuple, List # Added Tuple, List for Pydantic fallback
from pathlib import Path
import re

# Guarded imports for major dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase, PreTrainedModel # More specific types
except ImportError:
    torch = None # type: ignore
    AutoTokenizer = None # type: ignore
    AutoModelForSeq2SeqLM = None # type: ignore
    PreTrainedTokenizerBase = None # type: ignore
    PreTrainedModel = None # type: ignore
    print("Warning: PyTorch or Hugging Face Transformers not found. T5 client for spec normalization will be disabled.")

try:
    import yaml
except ImportError:
    yaml = None # type: ignore
    print("Warning: PyYAML not found. YAML parsing for T5 output will be disabled.")

# Assuming Spec is defined in src.specs.schemas and can be imported
# from ..specs.schemas import Spec # Ideal relative import
# Placeholder for subtask if direct relative import is an issue:
try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object # type: ignore
    Field = None # type: ignore
    print("Warning: Pydantic not found. Spec model will not function.")

class Spec(BaseModel): # Fallback Spec definition
    task: str = Field(..., description="Free-text summary of the task or issue.")
    target_symbols: List[str] = Field(default_factory=list, description="FQNs")
    operations: List[str] = Field(default_factory=list, description="Verbs")
    acceptance: List[str] = Field(default_factory=list, description="Tests")
    # Ad-hoc fields for error reporting from normalise_request
    raw_output: Optional[str] = Field(default=None, description="Raw output from T5 model on error.")
    raw_yaml: Optional[str] = Field(default=None, description="Extracted YAML string on parsing error.")
    parsed_dict: Optional[Dict[str, Any]] = Field(default=None, description="Parsed dictionary on validation error.")

    class Config:
        extra = 'allow' # Allow ad-hoc fields like raw_output for error reporting

# End placeholder


# Global cache for models and tokenizers to avoid reloading
_model_cache: Dict[str, PreTrainedModel] = {}
_tokenizer_cache: Dict[str, PreTrainedTokenizerBase] = {}


def _load_t5_model_and_tokenizer(model_path_str: str, device: str) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizerBase]]:
    if model_path_str in _model_cache and model_path_str in _tokenizer_cache:
        print(f"Using cached T5 model and tokenizer for {model_path_str}")
        # Move model to device again in case device changed, though cache usually per device
        # For simplicity, assume model in cache is already on a suitable device or this is handled by caller.
        # A more robust cache would key by (model_path, device).
        return _model_cache[model_path_str].to(device), _tokenizer_cache[model_path_str]

    if not (AutoTokenizer and AutoModelForSeq2SeqLM and torch):
        return None, None

    try:
        print(f"Loading T5 tokenizer from: {model_path_str}")
        tokenizer = AutoTokenizer.from_pretrained(model_path_str)
        print(f"Loading T5 model from: {model_path_str} onto device: {device}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path_str).to(device)
        model.eval()

        _tokenizer_cache[model_path_str] = tokenizer
        _model_cache[model_path_str] = model
        return model, tokenizer
    except Exception as e:
        print(f"Error loading T5 model/tokenizer from {model_path_str}: {e}")
        return None, None


def normalise_request(
    raw_request_text: str,
    symbol_bag_string: str,
    model_path: str,
    device: Optional[str] = None,
    max_input_length: int = 1024,
    max_output_length: int = 512
) -> Spec:
    if not (torch and AutoModelForSeq2SeqLM and AutoTokenizer and yaml and Spec and Field): # Check Field too
        error_msg = "T5 client dependencies (Transformers, PyTorch, PyYAML, Pydantic) not available."
        print(f"Error: {error_msg}")
        return Spec(task=f"Error: {error_msg}")

    selected_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = _load_t5_model_and_tokenizer(model_path, selected_device)

    if model is None or tokenizer is None:
        error_msg = f"Failed to load T5 model/tokenizer from {model_path}."
        return Spec(task=f"Error: {error_msg}")

    input_text = f"<RAW_REQUEST> {raw_request_text} </RAW_REQUEST> <SYMBOL_BAG> {symbol_bag_string} </SYMBOL_BAG>"

    yaml_string_from_model = ""
    try:
        print(f"T5 Client: Normalizing request. Input text length: {len(input_text)}")
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        ).to(selected_device)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                temperature=0.3, # Example: Slightly reduced from default 1.0
                top_p=0.9,       # Example: Nucleus sampling
                max_length=max_output_length,
                num_beams=4,      # Example: Beam search
                early_stopping=True
            )

        if output_sequences is not None and len(output_sequences) > 0:
            yaml_string_from_model = tokenizer.decode(output_sequences[0], skip_special_tokens=True).strip()
            print(f"T5 Client: Raw output from model: '{yaml_string_from_model[:300]}...'")
        else:
            print("Warning: T5 model returned an empty or unexpected response.")
            return Spec(task="Error: T5 model returned empty response.", raw_output=yaml_string_from_model)

    except Exception as e:
        print(f"Error during T5 model inference: {e}")
        return Spec(task=f"Error: T5 inference failed: {e}")

    match = re.search(r"<SPEC_YAML>(.*?)</SPEC_YAML>", yaml_string_from_model, re.DOTALL | re.IGNORECASE)
    if not match:
        print(f"Error: Could not find <SPEC_YAML>...</SPEC_YAML> tags in T5 output. Output was: {yaml_string_from_model[:500]}")
        return Spec(task="Error: Malformed T5 output (missing SPEC_YAML tags).", raw_output=yaml_string_from_model)

    extracted_yaml_str = match.group(1).strip()
    print(f"T5 Client: Extracted YAML string: '{extracted_yaml_str[:300]}...'")

    loaded_yaml_dict: Optional[Dict[str, Any]] = None
    try:
        loaded_yaml_dict = yaml.safe_load(extracted_yaml_str)
        if not isinstance(loaded_yaml_dict, dict):
            raise yaml.YAMLError("YAML content did not parse into a dictionary.")

        spec_object = Spec(**loaded_yaml_dict)
        return spec_object
    except yaml.YAMLError as e_yaml:
        print(f"Error parsing YAML from T5 output: {e_yaml}. YAML was: {extracted_yaml_str[:500]}")
        return Spec(task=f"Error: Failed to parse YAML from T5: {e_yaml}", raw_yaml=extracted_yaml_str)
    except Exception as e_pydantic:
        print(f"Error validating Spec from T5 output: {e_pydantic}. Data was: {loaded_yaml_dict if loaded_yaml_dict is not None else extracted_yaml_str[:500]}")
        return Spec(task=f"Error: Failed to validate Spec from T5: {e_pydantic}", raw_yaml=extracted_yaml_str, parsed_dict=loaded_yaml_dict)


if __name__ == '__main__':
    from unittest.mock import MagicMock # For mocking in example

    if not (torch and AutoModelForSeq2SeqLM and AutoTokenizer and yaml and Spec and Field):
        print("Skipping T5 client __main__ example due to missing dependencies.")
    else:
        print("--- Testing T5 Client for Spec Normalization ---")
        test_model_path = "t5-small"

        raw_req = "add a new method to the user service to update email addresses and refactor the tests"
        sym_bag = "UserService.get_email|UserService.update_profile|test_user_service.test_email_update"

        print(f"Input Raw Request: {raw_req}")
        print(f"Input Symbol Bag: {sym_bag}")

        simulated_model_yaml_output = """<SPEC_YAML>
task: "add user email update functionality"
target_symbols:
  - "UserService.update_profile"
  - "UserService.get_email"
operations:
  - "add_method"
  - "update_existing_method"
  - "refactor_test"
acceptance:
  - "test_user_service.test_email_update_success"
  - "test_user_service.test_email_update_invalid_email"
</SPEC_YAML>"""

        class MockModelInstance: # More complete mock for .to and .eval
            def generate(self, input_ids, attention_mask, temperature, top_p, max_length, num_beams, early_stopping):
                # This needs a real tokenizer to create output_sequences correctly
                # For simplicity, we'll assume the tokenizer used in normalise_request can be used here
                # if we can get an instance of it.
                temp_tokenizer = AutoTokenizer.from_pretrained(test_model_path)
                return temp_tokenizer(simulated_model_yaml_output, return_tensors="pt").input_ids
            def to(self, device): return self
            def eval(self): pass

        # Patch the global _load_t5_model_and_tokenizer to control model and tokenizer instances
        mock_tokenizer_instance = AutoTokenizer.from_pretrained(test_model_path) # Use a real tokenizer for decode

        with patch('src.spec_normalizer.t5_client._load_t5_model_and_tokenizer',
                   return_value=(MockModelInstance(), mock_tokenizer_instance)) as mock_load_fn:

            spec_result = normalise_request(raw_req, sym_bag, model_path=test_model_path)

            print("\n--- Generated Spec Object ---")
            if "Error:" in spec_result.task and spec_result.task.startswith("Error:"): # Check if task field contains an error message
                 print(f"Task: {spec_result.task}")
                 if spec_result.raw_yaml: print(f"Raw YAML: {spec_result.raw_yaml}")
                 if spec_result.parsed_dict: print(f"Parsed Dict: {spec_result.parsed_dict}")
            else:
                print(spec_result.model_dump_json(indent=2))

            assert spec_result.task == "add user email update functionality"
            assert "add_method" in spec_result.operations
        print("\n--- T5 Client Example Done ---")
