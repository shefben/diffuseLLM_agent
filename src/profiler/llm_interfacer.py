import random
from typing import Dict, Literal, Union, Optional
from dataclasses import dataclass, field # Using dataclass for simplicity

# Define the structure for the style fingerprint tuple from a single sample
@dataclass
class SingleSampleFingerprint:
    indent: int
    quotes: Literal["single", "double"]
    linelen: int # Max line length observed or preferred for this sample
    camel_pct: float # Percentage of camelCase identifiers in the sample
    snake_pct: float # Percentage of snake_case identifiers in the sample
    docstyle: Literal["google", "numpy", "epytext", "restructuredtext", "plain", "other"]
    # Add other potential fields based on common style aspects
    # For example, presence of type hints, spacing around operators, etc.
    # These would be derived by the hypothetical LLM from the sample.
    has_type_hints: Optional[bool] = None
    spacing_around_operators: Optional[bool] = None # True if consistent, False if not, None if N/A

    def to_dict(self) -> Dict[str, Union[int, str, float, bool, None]]:
        return {
            "indent": self.indent,
            "quotes": self.quotes,
            "linelen": self.linelen,
            "camel_pct": round(self.camel_pct, 2), # Ensure consistent formatting
            "snake_pct": round(self.snake_pct, 2),
            "docstyle": self.docstyle,
            "has_type_hints": self.has_type_hints,
            "spacing_around_operators": self.spacing_around_operators,
        }

# Added imports for collect_deterministic_stats
import io
import tokenize
import re
import ast
from collections import Counter
# Need to import get_args for Literal type inspection if not already present
# Add this near other typing imports if it's not there
from typing import get_args, List, Any # Ensure List, Any are imported (Dict, Optional, Union already are)
import json # For json.loads in orchestrator
from pathlib import Path # For model paths in __main__

# (Keep existing SingleSampleFingerprint dataclass and get_style_fingerprint_from_llm mock function for now)
# ...

# Define triple quote strings in a way that might be safer for subtask parsing
TRIPLE_SINGLE_QUOTE = chr(39) * 3  # '''
TRIPLE_DOUBLE_QUOTE = chr(34) * 3  # """

def _get_modal_indent(code_snippet: str) -> int:
    indents = []
    for line in code_snippet.splitlines():
        stripped_line = line.lstrip()
        if not stripped_line: continue
        leading_spaces = len(line) - len(stripped_line)
        if leading_spaces > 0: indents.append(leading_spaces)
    if not indents: return 4
    counts = Counter(indents)
    most_common = counts.most_common(1)
    return most_common[0][0] if most_common else 4

def _get_line_length_percentile(code_snippet: str, percentile: float = 0.95) -> int:
    line_lengths = [len(line) for line in code_snippet.splitlines()]
    if not line_lengths: return 88
    line_lengths.sort()
    index = int(len(line_lengths) * percentile) - 1
    index = max(0, min(index, len(line_lengths) - 1))
    return line_lengths[index]

def collect_deterministic_stats(code_snippet: str) -> str:
    stats = {}
    stats['indent_modal'] = _get_modal_indent(code_snippet)

    single_quotes_count = 0
    double_quotes_count = 0
    f_strings_count = 0
    identifiers = []

    try:
        code_bytes = code_snippet.encode('utf-8')
        token_stream = tokenize.tokenize(io.BytesIO(code_bytes).readline)
        for token_info in token_stream:
            if token_info.type == tokenize.STRING:
                token_string = token_info.string
                is_fstring = False
                prefix_len = 0

                lowered_token_string = token_string.lower()
                if lowered_token_string.startswith(("rf", "fr")):
                    is_fstring = True
                    prefix_len = 2
                elif lowered_token_string.startswith(("r", "f", "u", "b")):
                    if lowered_token_string.startswith("f"):
                        is_fstring = True
                    prefix_len = 1

                if is_fstring:
                    f_strings_count += 1

                actual_string_part = token_string[prefix_len:]

                if actual_string_part.startswith(TRIPLE_SINGLE_QUOTE) and actual_string_part.endswith(TRIPLE_SINGLE_QUOTE):
                    single_quotes_count += 1
                elif actual_string_part.startswith(TRIPLE_DOUBLE_QUOTE) and actual_string_part.endswith(TRIPLE_DOUBLE_QUOTE):
                    double_quotes_count += 1
                elif actual_string_part.startswith("'") and actual_string_part.endswith("'"):
                    single_quotes_count += 1
                elif actual_string_part.startswith('"') and actual_string_part.endswith('"'):
                    double_quotes_count += 1

            elif token_info.type == tokenize.NAME:
                identifiers.append(token_info.string)
    except tokenize.TokenError:
        pass

    stats['quotes_single'] = single_quotes_count
    stats['quotes_double'] = double_quotes_count
    stats['f_strings'] = f_strings_count
    stats['line_len_95p'] = _get_line_length_percentile(code_snippet)

    snake_case_count = 0
    camelCase_count = 0
    UPPER_SCREAMING_count = 0
    valid_idents_for_case_analysis = [ident for ident in identifiers if len(ident) > 1 or ident == '_']

    for ident in valid_idents_for_case_analysis:
        if re.fullmatch(r'[a-z0-9_]+', ident) and not any(c.isupper() for c in ident):
            snake_case_count += 1
        elif re.fullmatch(r'[a-z]+[A-Z][a-zA-Z0-9_]*', ident):
            camelCase_count += 1
        elif re.fullmatch(r'[A-Z0-9_]+', ident) and ident.isupper():
            UPPER_SCREAMING_count += 1

    total_countable_idents = len(valid_idents_for_case_analysis)
    stats['snake_pct'] = round(snake_case_count / total_countable_idents, 2) if total_countable_idents > 0 else 0.0
    stats['camel_pct'] = round(camelCase_count / total_countable_idents, 2) if total_countable_idents > 0 else 0.0
    stats['screaming_pct'] = round(UPPER_SCREAMING_count / total_countable_idents, 2) if total_countable_idents > 0 else 0.0

    docstring_markers = []
    try:
        tree = ast.parse(code_snippet)
        for node in ast.walk(tree):
            docstring = ast.get_docstring(node, clean=False)
            if docstring:
                unique_markers_in_doc = set()
                if "Args:" in docstring or "Arguments:" in docstring: unique_markers_in_doc.add("Args:")
                if "Parameters:" in docstring: unique_markers_in_doc.add("Parameters:")
                if "Returns:" in docstring or "Return:" in docstring: unique_markers_in_doc.add("Returns:")
                if ":param" in docstring: unique_markers_in_doc.add(":param")
                docstring_markers.extend(list(unique_markers_in_doc))
        if docstring_markers:
             docstring_markers = sorted(list(set(docstring_markers)))
    except SyntaxError:
        pass
    stats['doc_tokens'] = ", ".join(docstring_markers) if docstring_markers else "none"

    stats_block_lines = ["STATS_START"]
    stats_block_lines.append(f"indent_modal: {stats['indent_modal']}")
    stats_block_lines.append(f"quotes_single: {stats['quotes_single']}")
    stats_block_lines.append(f"quotes_double: {stats['quotes_double']}")
    stats_block_lines.append(f"f_strings: {stats['f_strings']}")
    stats_block_lines.append(f"line_len_95p: {stats['line_len_95p']}")
    stats_block_lines.append(f"snake_pct: {stats['snake_pct']:.2f}")
    stats_block_lines.append(f"camel_pct: {stats['camel_pct']:.2f}")
    stats_block_lines.append(f"screaming_pct: {stats['screaming_pct']:.2f}")
    stats_block_lines.append(f"doc_tokens: {stats['doc_tokens']}")
    stats_block_lines.append("STATS_END")

    return "\n".join(stats_block_lines)

# (Optional __main__ block for testing)
# if __name__ == '__main__':
#     # This main block is for testing collect_deterministic_stats.
#     # Keep it separate from the one testing get_style_fingerprint_from_llm.
#     example_code_for_test = TRIPLE_SINGLE_QUOTE + '''
# Module doc for testing.
# Args:
#     arg1: something
# ''' + TRIPLE_SINGLE_QUOTE + '''
# import os
#
# class MyClassExample:
#     A_CONSTANT = 10
#
#     def __init__(self, my_param_snake: int):
#         self.my_inst_var = my_param_snake
#
#     def anotherMethod(self, another_param: str):
#         # Docstring using TRIPLE_SINGLE_QUOTE
#         pass # body
# ''' # End of example_code_for_test
#     stats_output = collect_deterministic_stats(example_code_for_test)
#     print(stats_output)


# Add LlamaCPP import, guarded for environments where it might not be installed
try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    Llama = None # type: ignore
    LlamaGrammar = None # type: ignore
    print("Warning: llama-cpp-python not found. DeepSeek GGUF model interaction will be disabled.")


def get_deepseek_draft_fingerprint(stats_block: str, model_path: str, n_gpu_layers: int = -1, verbose: bool = False) -> Dict:
    """
    Loads a DeepSeek GGUF model and prompts it with style statistics to get a raw
    style fingerprint.

    Args:
        stats_block: A string containing the deterministic style statistics.
        model_path: Path to the DeepSeek GGUF model file.
        n_gpu_layers: Number of layers to offload to GPU. -1 for all, 0 for none.
        verbose: Whether LlamaCPP should be verbose.

    Returns:
        A dictionary representing the raw style fingerprint from DeepSeek.
        Returns a mock/default dictionary if LlamaCPP is not available or model fails.
    """
    if Llama is None:
        print("Error: llama-cpp-python is not installed. Cannot interact with DeepSeek GGUF model.")
        # Return a mock response that includes potential ambiguities for downstream testing
        return {
            "indent": 4, "quotes": "mixed?", "linelen": 100, "snake_pct": 0.7,
            "camel_pct": 0.2, "screaming_pct": 0.1, "docstyle": "google_or_numpy"
        }

    try:
        # TODO: Make model loading more configurable (n_ctx, etc.)
        # For now, using some defaults. The user might need to adjust n_ctx based on model and prompt length.
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers, # -1 attempts to offload all possible layers
            n_ctx=2048, # Context window, might need adjustment
            verbose=verbose
        )
    except Exception as e:
        # This can catch errors like model file not found, permission issues, or LlamaCPP init errors.
        print(f"Error loading DeepSeek GGUF model from {model_path}: {e}")
        print("Falling back to mock DeepSeek response.")
        return {
            "indent": 4, "quotes": "mixed?", "linelen": 90, "snake_pct": 0.6,
            "camel_pct": 0.3, "screaming_pct": 0.05, "docstyle": "google"
        }

    system_prompt = "Return ONLY a JSON object with keys indent, quotes, linelen, snake_pct, camel_pct, screaming_pct, docstyle."

    # Using a chat completion like format, common for instruction-following models
    # The exact prompt format might need tuning based on the specific DeepSeek GGUF version.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": stats_block}
    ]

    raw_json_output = "{}" # Default to empty JSON string

    try:
        # TODO: Make generation parameters configurable (temp, top_p, etc.)
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.1, # Low temperature for more deterministic, factual output
            max_tokens=256  # Max tokens for the JSON response
        )
        if response and response['choices'] and response['choices'][0]['message']['content']:
            raw_json_output = response['choices'][0]['message']['content'].strip()
        else:
            print("Warning: DeepSeek model returned an empty or unexpected response.")

    except Exception as e:
        print(f"Error during DeepSeek model inference: {e}")
        print("Falling back to mock DeepSeek response due to inference error.")
        return {
            "indent": 4, "quotes": "double?", "linelen": 80, "snake_pct": 0.5,
            "camel_pct": 0.4, "screaming_pct": 0.08, "docstyle": "numpy"
        }

    # Attempt to parse the output as JSON.
    # The model is asked for JSON, but it might not be perfect.
    # Example from user: {"quotes": "mixed?"} - this is not valid JSON due to unquoted "mixed?".
    # The user's example output was: `{"indent": 4, "quotes": "mixed?", ...}` which IS valid JSON if "mixed?" is a string.
    # The problem arises if DeepSeek outputs `mixed?` without quotes.
    # The prompt "Return ONLY a JSON object" should strongly encourage valid JSON.

    fingerprint_dict = {}
    try:
        # A simple attempt to clean common non-JSON issues if model doesn't strictly adhere.
        # E.g., if it uses Python dict-like output with single quotes for strings.
        # This is a heuristic. A more robust solution might involve more advanced parsing
        # or re-prompting if the initial output is not valid JSON.
        # For now, we rely on the prompt and basic json.loads.

        # Try to find the JSON block if there's surrounding text (though prompt asks for ONLY JSON)
        json_match = re.search(r'\{.*\}', raw_json_output, re.DOTALL)
        if json_match:
            raw_json_output = json_match.group(0)

        fingerprint_dict = json.loads(raw_json_output)

        # Ensure all expected keys are present, fill with None if not (or a default ambiguous marker)
        expected_keys = ["indent", "quotes", "linelen", "snake_pct", "camel_pct", "screaming_pct", "docstyle"]
        for key in expected_keys:
            if key not in fingerprint_dict:
                fingerprint_dict[key] = "unknown?" # Mark as unknown if missing

    except json.JSONDecodeError as e:
        print(f"Warning: DeepSeek output was not valid JSON: {raw_json_output}. Error: {e}")
        print("Attempting to create a partial fingerprint from the raw string, or returning a default ambiguous one.")
        # Fallback: try to extract values with regex if JSON parsing fails, or return a default ambiguous dict
        # This part can be made more sophisticated. For now, a simple fallback.
        partial_fingerprint = {}
        for key in ["indent", "linelen"]: # Numerical
            match = re.search(rf'"{key}":\s*(\d+)', raw_json_output)
            if match: partial_fingerprint[key] = int(match.group(1))
        for key in ["quotes", "docstyle"]: # String
            match = re.search(rf'"{key}":\s*"([^"]*)"', raw_json_output) # looking for quoted strings
            if match: partial_fingerprint[key] = match.group(1)
            else: # try to match unquoted problematic values like 'mixed?'
                match_unquoted = re.search(rf'"{key}":\s*([a-zA-Z_?]+)', raw_json_output)
                if match_unquoted : partial_fingerprint[key] = match_unquoted.group(1)

        for key_pct in ["snake_pct", "camel_pct", "screaming_pct"]:
            match = re.search(rf'"{key_pct}":\s*(\d+\.\d+)', raw_json_output)
            if match: partial_fingerprint[key_pct] = float(match.group(1))

        # If very little was parsed, use a more structured ambiguous default
        if not partial_fingerprint or len(partial_fingerprint) < 3:
            fingerprint_dict = {
                "indent": "unknown?", "quotes": "unknown?", "linelen": "unknown?",
                "snake_pct": "unknown?", "camel_pct": "unknown?",
                "screaming_pct": "unknown?", "docstyle": "unknown?"
            }
        else: # Use what was partially parsed, fill missing
            expected_keys = ["indent", "quotes", "linelen", "snake_pct", "camel_pct", "screaming_pct", "docstyle"]
            for key in expected_keys:
                if key not in partial_fingerprint:
                    partial_fingerprint[key] = "unknown?"
            fingerprint_dict = partial_fingerprint

    return fingerprint_dict

# (Optional: Add to the existing __main__ block in llm_interfacer.py or create a new one for testing this)
# if __name__ == '__main__':
#     # This main block would be for testing the whole llm_interfacer.
#     # To test get_deepseek_draft_fingerprint specifically:
#     example_stats = """STATS_START
# indent_modal: 4
# quotes_single: 10
# quotes_double: 2
# f_strings: 3
# line_len_95p: 90
# snake_pct: 0.80
# camel_pct: 0.15
# screaming_pct: 0.05
# doc_tokens: Args:, Returns:
# STATS_END"""
#     # IMPORTANT: Replace with an ACTUAL path to a DeepSeek GGUF model file.
#     # If no model is available, it will use the mock fallback.
#     # For the test environment, we won't have a real model path, so it will always use mock.
#     from pathlib import Path # Ensure Path is imported if this __main__ is uncommented
#     deepseek_model_path = "path/to/your/deepseek-coder-gguf-model.gguf"
#
#     print("\n--- Testing DeepSeek Draft Fingerprint ---")
#     if Llama is None:
#         print("llama-cpp-python not installed, test will use mock response.")
#     # Simple check to see if placeholder path was changed; in CI/test, it won't be.
#     elif deepseek_model_path == "path/to/your/deepseek-coder-gguf-model.gguf" or not Path(deepseek_model_path).exists():
#         print(f"DeepSeek GGUF model not found at '{deepseek_model_path}' (or path is placeholder), test will use mock response if loading fails.")
#
#     draft_fp = get_deepseek_draft_fingerprint(example_stats, deepseek_model_path, verbose=False) # Verbose usually True for debug
#     print("\nDeepSeek Draft Fingerprint Output:")
#     import json # for pretty print
#     print(json.dumps(draft_fp, indent=2))


# Add Hugging Face Transformers import, guarded
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
except ImportError:
    torch = None # type: ignore
    T5ForConditionalGeneration = None # type: ignore
    T5TokenizerFast = None # type: ignore
    print("Warning: PyTorch or Hugging Face Transformers not found. DivoT5 SafeTensor model interaction will be disabled.")


def get_divot5_refined_output(
    code_snippet: str,
    raw_fingerprint_dict: Dict,
    model_path: str,
    num_denoising_steps: int = 10, # User spec: 8-12, used conceptually for generation params
    device: Optional[str] = None
) -> str:
    """
    Loads a DivoT5 SafeTensor model and uses it to refine a raw style fingerprint
    based on the original code snippet.

    Args:
        code_snippet: The original code snippet.
        raw_fingerprint_dict: The raw style fingerprint dictionary from DeepSeek.
        model_path: Path to the DivoT5 model directory (SafeTensor format).
        num_denoising_steps: Conceptually represents generation quality/length.
                             Used to set max_length for T5 generation.
        device: Device to run the model on (e.g., "cpu", "cuda"). Auto-detects if None.

    Returns:
        A string containing the refined key/value pairs for the style fingerprint.
        Returns a mock/default string if Transformers/PyTorch are not available or model fails.
    """
    if T5ForConditionalGeneration is None or T5TokenizerFast is None or torch is None:
        print("Error: Transformers/PyTorch not installed. Cannot interact with DivoT5 model.")
        # Return a mock response that looks like a refined version of a typical raw input
        refined_mock_parts = []
        for key, value in raw_fingerprint_dict.items():
            if isinstance(value, str) and "?" in value: # If ambiguous
                if key == "quotes": refined_mock_parts.append(f'"{key}": "single"') # Example refinement
                elif key == "docstyle": refined_mock_parts.append(f'"{key}": "google"')
                else: refined_mock_parts.append(f'"{key}": "{str(value).replace("?", "")}_refined"')
            elif isinstance(value, float):
                 refined_mock_parts.append(f'"{key}": {value:.2f}') # Ensure formatting
            else:
                refined_mock_parts.append(f'"{key}": "{value}"' if isinstance(value, str) else f'"{key}": {value}')
        return ", ".join(refined_mock_parts)

    if device is None:
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = device

    print(f"Loading DivoT5 model from {model_path} onto device: {selected_device}")

    try:
        tokenizer = T5TokenizerFast.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(selected_device)
        model.eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error loading DivoT5 model from {model_path}: {e}")
        print("Falling back to mock DivoT5 response.")
        # Simulate refinement of the input raw_fingerprint_dict
        refined_mock_parts = []
        for key, value in raw_fingerprint_dict.items():
            if key == "quotes" and value == "mixed?": refined_mock_parts.append(f'"{key}": "single"')
            else: refined_mock_parts.append(f'"{key}": "{value}"' if isinstance(value, str) else f'"{key}": {value}')
        return ", ".join(refined_mock_parts)

    # Construct the input string for DivoT5 as per playbook
    # RAW_FINGERPRINT_START needs key/value pairs, not a dict string literal
    raw_fp_parts = []
    for key, value in raw_fingerprint_dict.items():
        if isinstance(value, str):
            # Ensure internal quotes in string values are escaped if necessary for the model,
            # but typical key-value string format doesn't require this for values.
            # The spec shows simple "key": "value" or "key": number.
            raw_fp_parts.append(f'"{key}": "{value}"')
        else: # numbers, booleans (though not in example)
            raw_fp_parts.append(f'"{key}": {value if isinstance(value, (int, float)) else str(value).lower()}')


    input_text = f"""USER_CODE_START
{code_snippet}
USER_CODE_END
RAW_FINGERPRINT_START
{", ".join(raw_fp_parts)}
RAW_FINGERPRINT_END"""

    refined_output_string = ", ".join(raw_fp_parts) # Default to input if generation fails

    try:
        print(f"DivoT5 input length: {len(input_text)} chars")
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(selected_device) # Max input length for T5

        # "denoising steps" for T5 might translate to generation parameters like num_beams, length_penalty, max_length.
        # The user spec says 8-12 denoising steps. This is more typical of diffusion.
        # For T5, we'll generate a sequence that should be the refined key/value string.
        # Let's use num_denoising_steps to influence max_length of the output.
        # Output is expected to be a string of key/value pairs.
        # Example: "indent": 4, "quotes": "single", ...
        # Max length should be enough for all key-value pairs. ~50 chars per pair * 7 pairs = ~350. Add buffer.
        output_max_length = num_denoising_steps * 40 # Heuristic: 40 chars per "denoising step unit"
        output_max_length = max(200, min(output_max_length, 512)) # Clamp to reasonable range for key/value string

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=output_max_length,
                num_beams=4, # Beam search can produce better quality
                early_stopping=True,
                length_penalty=1.0 # Adjust as needed
            )

        if outputs is not None and len(outputs) > 0:
            refined_output_string = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        else:
            print("Warning: DivoT5 model returned an empty or unexpected response.")

    except Exception as e:
        print(f"Error during DivoT5 model inference: {e}")
        print("Falling back to mock DivoT5 response due to inference error.")
        # Fallback: Simulate some refinement on the input dictionary
        refined_mock_parts = []
        for key, value in raw_fingerprint_dict.items():
            if key == "quotes" and isinstance(value, str) and "?" in value:
                refined_mock_parts.append(f'"{key}": "single"') # Example: 'mixed?' becomes 'single'
            elif key == "docstyle" and isinstance(value, str) and "?" in value:
                refined_mock_parts.append(f'"{key}": "google"') # Example: 'google?' becomes 'google'
            elif isinstance(value, float) :
                 refined_mock_parts.append(f'"{key}": {value:.2f}') # Format float
            else: # numbers or already fine strings
                refined_mock_parts.append(f'"{key}": "{value}"' if isinstance(value, str) else f'"{key}": {value}')
        return ", ".join(refined_mock_parts)

    return refined_output_string

# (Optional: Add to the existing __main__ block in llm_interfacer.py or create a new one for testing this)
# if __name__ == '__main__':
#     # This main block would be for testing the whole llm_interfacer.
#     # To test get_divot5_refined_output specifically:
#
#     # 1. First get a draft fingerprint (mock or real)
#     example_stats = """STATS_START
# indent_modal: 4
# quotes_single: 10
# quotes_double: 8
# f_strings: 1
# line_len_95p: 70
# snake_pct: 0.60
# camel_pct: 0.30
# screaming_pct: 0.10
# doc_tokens: Args:
# STATS_END"""
#     # Use a mock path for DeepSeek if not available for this isolated test
#     deepseek_model_path_for_test = "mock_deepseek_path_for_divot5_test"
#
#     # Simulate a raw fingerprint that DivoT5 might receive
#     raw_fp_from_deepseek = {
#         "indent": 4, "quotes": "mixed?", "linelen": 70,
#         "snake_pct": 0.60, "camel_pct": 0.30, "screaming_pct": 0.10,
#         "docstyle": "google?" # Another ambiguous field
#     }
#     # Or call the actual get_deepseek_draft_fingerprint if testing integration
#     # raw_fp_from_deepseek = get_deepseek_draft_fingerprint(example_stats, deepseek_model_path_for_test)
#
#     example_code_snippet = "def myFunc(a_var):\n  return a_var # Example snippet"
#
#     # IMPORTANT: Replace with an ACTUAL path to a DivoT5 model directory.
#     # If no model is available, it will use the mock fallback.
#     from pathlib import Path # Ensure Path is imported if this __main__ is uncommented
#     divot5_model_path = "path/to/your/divot5_model_dir"
#
#     print("\n--- Testing DivoT5 Refined Output ---")
#     if T5ForConditionalGeneration is None:
#         print("Transformers/PyTorch not installed, test will use mock response.")
#     elif not Path(divot5_model_path).exists() and not divot5_model_path.startswith("path/to"):
#         print(f"DivoT5 model not found at '{divot5_model_path}', test will use mock response if loading fails.")
#
#     refined_key_values = get_divot5_refined_output(
#         example_code_snippet,
#         raw_fp_from_deepseek,
#         divot5_model_path
#     )
#     print("\nDivoT5 Refined Key/Value String Output:")
#     print(refined_key_values)
#
#     # Example of how this string might be used in the next step (DeepSeek Polish)
#     # It would be parsed or directly fed if the polish step can handle it.
#     # For now, just showing the string.


# Simple JSON grammar for LlamaCPP (GBNF format).
# This grammar expects a JSON object with specific string keys and allows string, number, or boolean values.
# It's a simplified grammar; a more robust one would handle nesting, arrays, etc.
# For this specific fingerprint, values are mostly strings or numbers.
JSON_FINGERPRINT_GRAMMAR_STR = r'''
root   ::= object
value  ::= object | array | string | number | boolean | "null"
object ::= "{}" | "{" members "}"
members ::= pair ("," pair)*
pair   ::= string ":" value
array  ::= "[]" | "[" elements "]"
elements ::= value ("," value)*
string ::= """ (([#x20-#x21] | [#x23-#x5B] | [#x5D-#xFFFF]) | #x5C ([#x22#x5C#x2F#x62#x66#x6E#x72#x74] | #x75[0-9a-fA-F]{4}))* """
number ::= ("-")? (("0") | ([1-9][0-9]*)) ("." [0-9]+)? (("e" | "E") (("-" | "+")?) [0-9]+)?
boolean ::= "true" | "false"
'''
# Note: The above grammar is a general JSON grammar. For the specific fingerprint,
# we might want a more constrained one if all keys are known and types are fixed.
# However, the prompt to DeepSeek already specifies the keys. This general JSON grammar
# ensures the *structure* is JSON, letting the LLM fill in values based on the prompt.


def get_deepseek_polished_json(
    cleaned_key_value_string: str,
    model_path: str,
    n_gpu_layers: int = -1,
    verbose: bool = False
) -> str:
    """
    Loads/uses a DeepSeek GGUF model to polish a string of key/value pairs into
    a valid JSON object string, potentially using a grammar constraint.

    Args:
        cleaned_key_value_string: The string of key/value pairs from DivoT5.
        model_path: Path to the DeepSeek GGUF model file.
        n_gpu_layers: Number of layers to offload to GPU.
        verbose: Whether LlamaCPP should be verbose.

    Returns:
        A string that should be a valid JSON representation of the fingerprint.
        Returns a mock/default JSON string if LlamaCPP is not available or model fails.
    """
    if Llama is None or LlamaGrammar is None:
        print("Error: llama-cpp-python is not installed. Cannot polish JSON with DeepSeek GGUF model.")
        # Return a mock response that looks like a valid JSON version of the input
        # Assuming cleaned_key_value_string is like: '"indent": 4, "quotes": "single", ...'
        # A simple mock is to wrap it in braces if it's not already.
        if cleaned_key_value_string.strip().startswith("{") and cleaned_key_value_string.strip().endswith("}"):
            return cleaned_key_value_string
        return f"{{{cleaned_key_value_string}}}"


    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048, # Context window, ensure it's enough for prompt + response
            verbose=verbose
        )
        grammar = LlamaGrammar.from_string(JSON_FINGERPRINT_GRAMMAR_STR)
    except Exception as e:
        print(f"Error loading DeepSeek GGUF model or grammar from {model_path}: {e}")
        print("Falling back to mock JSON polishing response.")
        if cleaned_key_value_string.strip().startswith("{") and cleaned_key_value_string.strip().endswith("}"):
            return cleaned_key_value_string
        return f"{{{cleaned_key_value_string}}}"

    # Prompt for this step: User playbook says "Send the cleaned key/value string back to DeepSeek
    # with a grammar mask that forces valid JSON. DeepSeek inserts any missing keys,
    # ensures proper quoting, and emits..."
    # The system prompt might need to guide it to complete/validate the JSON structure.

    system_prompt = "You are a helpful assistant that completes and validates JSON. Given the following key-value pairs, ensure they form a complete and valid JSON object representing a style fingerprint. The required keys are indent, quotes, linelen, snake_pct, camel_pct, screaming_pct, and docstyle. Ensure all string values are properly quoted."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": cleaned_key_value_string}
    ]

    polished_json_string = "{}" # Default to empty JSON string

    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.05, # Very low temperature for precise JSON formatting
            max_tokens=300,   # Max tokens for the JSON response
            grammar=grammar   # Apply the JSON grammar
        )
        if response and response['choices'] and response['choices'][0]['message']['content']:
            polished_json_string = response['choices'][0]['message']['content'].strip()
        else:
            print("Warning: DeepSeek (polish pass) model returned an empty or unexpected response.")
            # Fallback: try to make the input valid JSON if possible
            if cleaned_key_value_string.strip().startswith("{") and cleaned_key_value_string.strip().endswith("}"):
                 polished_json_string = cleaned_key_value_string
            else:
                 polished_json_string = f"{{{cleaned_key_value_string}}}"

    except Exception as e:
        print(f"Error during DeepSeek (polish pass) model inference: {e}")
        print("Falling back to basic JSON wrapping due to inference error.")
        if cleaned_key_value_string.strip().startswith("{") and cleaned_key_value_string.strip().endswith("}"):
             polished_json_string = cleaned_key_value_string
        else:
             polished_json_string = f"{{{cleaned_key_value_string}}}"

    return polished_json_string

# (Optional: Add to the existing __main__ block in llm_interfacer.py or create a new one for testing this)
# if __name__ == '__main__':
#     # This main block would be for testing the whole llm_interfacer.
#     # To test get_deepseek_polished_json specifically:
#
#     # Simulate a cleaned key/value string from DivoT5
#     # This string might already be close to JSON, or it might be just key-value pairs.
#     # User spec: DivoT5 snaps ambiguous fields to canonical values and clamps numbers:
#     # "indent": 4, "quotes": "single", "linelen": 110, ...
#     divot5_output_example = '"indent": 4, "quotes": "single", "linelen": 110, "snake_pct": 0.82, "camel_pct": 0.12, "screaming_pct": 0.06, "docstyle": "google"'
#
#     # IMPORTANT: Replace with an ACTUAL path to a DeepSeek GGUF model file.
#     from pathlib import Path # Ensure Path is imported if this __main__ is uncommented
#     deepseek_model_path_for_polish = "path/to/your/deepseek-coder-gguf-model.gguf"
#
#     print("\n--- Testing DeepSeek Polished JSON ---")
#     if Llama is None:
#         print("llama-cpp-python not installed, test will use mock response.")
#     elif not Path(deepseek_model_path_for_polish).exists() and not deepseek_model_path_for_polish.startswith("path/to"):
#         print(f"DeepSeek GGUF model not found at '{deepseek_model_path_for_polish}', test will use mock response if loading fails.")
#
#     polished_fp_json_str = get_deepseek_polished_json(
#         divot5_output_example,
#         deepseek_model_path_for_polish,
#         verbose=False # Set to True for detailed LlamaCPP logging
#     )
#     print("\nDeepSeek Polished JSON String Output:")
#     print(polished_fp_json_str)
#
#     # Try to parse it to see if it's valid
#     try:
#         import json
#         parsed_json = json.loads(polished_fp_json_str)
#         print("\nSuccessfully parsed the polished JSON:")
#         print(json.dumps(parsed_json, indent=2))
#     except json.JSONDecodeError as e:
#         print(f"\nFailed to parse polished JSON: {e}")
#     except Exception as e_gen:
#         print(f"\nAn error occurred with the polished JSON: {e_gen}")


# --- New function for LLM Code Fix Suggestion ---
def get_llm_code_fix_suggestion(
    model_path: str, # Path to GGUF model
    original_code_script: str, # The LibCST script that failed
    error_traceback: str,
    phase_description: str,
    target_file: Optional[str],
    additional_context: Optional[Dict[str, Any]], # e.g., style profile, snippets
    n_gpu_layers: int = -1,
    max_tokens: int = 1024, # Max tokens for the suggested script
    temperature: float = 0.4, # Temperature for generation
    verbose: bool = False
) -> Optional[str]:
    """
    Uses a GGUF language model to suggest a fix for a failing LibCST script.
    """
    if Llama is None:
        print("Error: llama-cpp-python not installed. Cannot get LLM code fix suggestion.")
        # Return a mock suggestion that includes parts of the error and original script info
        return f"# Mock fix for error: {error_traceback[:100]}...\n# Original script had {len(original_code_script)} chars.\npass # LLM disabled - Apply actual fix here"

    try:
        # Increased n_ctx for potentially longer prompts including code and traceback
        llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=4096, verbose=verbose)
    except Exception as e:
        print(f"Error loading LLM model from {model_path}: {e}")
        return f"# Mock fix due to model load error for: {error_traceback[:100]}...\npass"

    # Construct a detailed prompt
    prompt_parts = [
        "You are an expert Python programmer and code assistant, specialized in writing and debugging LibCST refactoring scripts.",
        "The following LibCST script was intended to perform a refactoring operation but failed validation or execution.",
        f"Refactoring Operation Description: {phase_description}",
        f"Target File for Refactoring: {target_file if target_file else 'N/A'}",
    ]
    if additional_context:
        if 'style_profile' in additional_context and additional_context['style_profile']:
            try:
                style_profile_str = json.dumps(additional_context['style_profile'], indent=2)
                prompt_parts.append(f"Project Style Profile (relevant parts for context):\n{style_profile_str}")
            except (TypeError, OverflowError) as json_e:
                 prompt_parts.append(f"Project Style Profile (relevant parts for context): Error serializing - {json_e}")

        if 'code_snippets' in additional_context and additional_context['code_snippets']: # Assuming code_snippets is Dict[str, str]
            for fname, snippet in additional_context['code_snippets'].items():
                prompt_parts.append(f"Relevant code snippet from '{fname}':\n```python\n{snippet}\n```")

    prompt_parts.extend([
        "Original LibCST script that failed:",
        "```python",
        original_code_script,
        "```",
        "Validation Error Traceback or Description:",
        "```text", # Marking traceback as text
        error_traceback,
        "```",
        "Please provide a revised version of the LibCST script that fixes the error. Output only the complete, revised Python script for the LibCST code. Do not add any explanations or markdown formatting before or after the script block."
    ])

    full_prompt = "\n\n".join(prompt_parts)

    if verbose: # Print more of the prompt if verbose is on
        print(f"LLM Code Fix Prompt (first 1000 chars):\n{full_prompt[:1000]}...")
    else:
        print(f"LLM Code Fix Prompt (first 200 chars):\n{full_prompt[:200]}...")


    try:
        response = llm.create_completion(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["```python\n", "\n```\n", "\n```"], # More robust stopping, though model should just output script
            echo=False # Don't echo the prompt in the completion
        )

        suggested_script = response['choices'][0]['text'].strip()

        # Basic cleaning: Sometimes models might still include markdown even if asked not to.
        if suggested_script.startswith("```python"):
            suggested_script = suggested_script[len("```python"):].strip()
        if suggested_script.startswith("```"): # Generic code block start
             suggested_script = suggested_script[len("```"):].strip()
        if suggested_script.endswith("```"):
            suggested_script = suggested_script[:-len("```")].strip()

        return suggested_script
    except Exception as e:
        print(f"Error during LLM code fix suggestion inference: {e}")
        return f"# Mock fix due to inference error for: {error_traceback[:100]}...\n# Original script length: {len(original_code_script)}\npass"
