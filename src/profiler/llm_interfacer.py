import random
from typing import Dict, Literal, Union, Optional, List, Any
from dataclasses import dataclass, field
import io
import tokenize
import re
import ast
from collections import Counter
import json
from pathlib import Path

# Guard LlamaCPP import
try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    Llama = None # type: ignore
    LlamaGrammar = None # type: ignore
    print("Warning: llama-cpp-python not found. GGUF model interaction will be disabled.")

# Guard Transformers/Torch import
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
except ImportError:
    torch = None # type: ignore
    T5ForConditionalGeneration = None # type: ignore
    T5TokenizerFast = None # type: ignore
    print("Warning: PyTorch or Hugging Face Transformers not found. DivoT5 SafeTensor model interaction will be disabled.")


@dataclass
class SingleSampleFingerprint:
    indent: int
    quotes: Literal["single", "double"]
    linelen: int
    camel_pct: float
    snake_pct: float
    docstyle: Literal["google", "numpy", "epytext", "restructuredtext", "plain", "other"]
    has_type_hints: Optional[bool] = None
    spacing_around_operators: Optional[bool] = None

    def to_dict(self) -> Dict[str, Union[int, str, float, bool, None]]:
        return {
            "indent": self.indent,
            "quotes": self.quotes,
            "linelen": self.linelen,
            "camel_pct": round(self.camel_pct, 2),
            "snake_pct": round(self.snake_pct, 2),
            "docstyle": self.docstyle,
            "has_type_hints": self.has_type_hints,
            "spacing_around_operators": self.spacing_around_operators,
        }

TRIPLE_SINGLE_QUOTE = chr(39) * 3
TRIPLE_DOUBLE_QUOTE = chr(34) * 3

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
    stats: Dict[str, Any] = {}
    stats['indent_modal'] = _get_modal_indent(code_snippet)
    single_quotes_count = 0; double_quotes_count = 0; f_strings_count = 0; identifiers = []
    try:
        code_bytes = code_snippet.encode('utf-8')
        token_stream = tokenize.tokenize(io.BytesIO(code_bytes).readline)
        for token_info in token_stream:
            if token_info.type == tokenize.STRING:
                token_string = token_info.string; is_fstring = False; prefix_len = 0
                lowered_token_string = token_string.lower()
                if lowered_token_string.startswith(("rf", "fr")): is_fstring = True; prefix_len = 2
                elif lowered_token_string.startswith(("r", "f", "u", "b")):
                    if lowered_token_string.startswith("f"): is_fstring = True
                    prefix_len = 1
                if is_fstring: f_strings_count += 1
                actual_string_part = token_string[prefix_len:]
                if actual_string_part.startswith(TRIPLE_SINGLE_QUOTE) and actual_string_part.endswith(TRIPLE_SINGLE_QUOTE): single_quotes_count += 1
                elif actual_string_part.startswith(TRIPLE_DOUBLE_QUOTE) and actual_string_part.endswith(TRIPLE_DOUBLE_QUOTE): double_quotes_count += 1
                elif actual_string_part.startswith("'") and actual_string_part.endswith("'"): single_quotes_count += 1
                elif actual_string_part.startswith('"') and actual_string_part.endswith('"'): double_quotes_count += 1
            elif token_info.type == tokenize.NAME: identifiers.append(token_info.string)
    except tokenize.TokenError: pass
    stats['quotes_single'] = single_quotes_count; stats['quotes_double'] = double_quotes_count; stats['f_strings'] = f_strings_count
    stats['line_len_95p'] = _get_line_length_percentile(code_snippet)
    snake_case_count = 0; camelCase_count = 0; UPPER_SCREAMING_count = 0
    valid_idents_for_case_analysis = [ident for ident in identifiers if len(ident) > 1 or ident == '_']
    for ident in valid_idents_for_case_analysis:
        if re.fullmatch(r'[a-z0-9_]+', ident) and not any(c.isupper() for c in ident): snake_case_count += 1
        elif re.fullmatch(r'[a-z]+[A-Z][a-zA-Z0-9_]*', ident): camelCase_count += 1
        elif re.fullmatch(r'[A-Z0-9_]+', ident) and ident.isupper(): UPPER_SCREAMING_count += 1
    total_countable_idents = len(valid_idents_for_case_analysis)
    stats['snake_pct'] = round(snake_case_count / total_countable_idents, 2) if total_countable_idents > 0 else 0.0
    stats['camel_pct'] = round(camelCase_count / total_countable_idents, 2) if total_countable_idents > 0 else 0.0
    stats['screaming_pct'] = round(UPPER_SCREAMING_count / total_countable_idents, 2) if total_countable_idents > 0 else 0.0
    docstring_markers = []; tree = None
    try: tree = ast.parse(code_snippet)
    except SyntaxError: pass
    if tree:
        for node in ast.walk(tree):
            docstring = ast.get_docstring(node, clean=False)
            if docstring:
                unique_markers_in_doc = set()
                if "Args:" in docstring or "Arguments:" in docstring: unique_markers_in_doc.add("Args:")
                if "Parameters:" in docstring: unique_markers_in_doc.add("Parameters:")
                if "Returns:" in docstring or "Return:" in docstring: unique_markers_in_doc.add("Returns:")
                if ":param" in docstring: unique_markers_in_doc.add(":param")
                docstring_markers.extend(list(unique_markers_in_doc))
        if docstring_markers: docstring_markers = sorted(list(set(docstring_markers)))
    stats['doc_tokens'] = ", ".join(docstring_markers) if docstring_markers else "none"
    stats_block_lines = ["STATS_START"] + [f"{k}: {v}" for k,v in stats.items()] + ["STATS_END"]
    return "\n".join(stats_block_lines)

def get_deepseek_draft_fingerprint(stats_block: str, model_path: str, n_gpu_layers: int = -1, verbose: bool = False) -> Dict[str, Any]:
    default_error_fingerprint: Dict[str, Any] = {"indent": None, "quotes": None, "linelen": None, "snake_pct": None, "camel_pct": None, "screaming_pct": None, "docstyle": None, "error": "Unknown error during fingerprint generation."}
    expected_fingerprint_keys = {"indent", "quotes", "linelen", "snake_pct", "camel_pct", "screaming_pct", "docstyle"}
    if Llama is None:
        error_msg = "llama-cpp-python not installed. Cannot get LLM style fingerprint."
        print(f"LLM_Interfacer Error: {error_msg}")
        return {**default_error_fingerprint, "error": error_msg}
    resolved_model_path_str = model_path; is_placeholder_path = False
    if not model_path or model_path == "path/to/your/deepseek-coder-gguf-model.gguf" or model_path.endswith("placeholder_deepseek.gguf"):
        resolved_model_path_str = "./models/placeholder_deepseek.gguf"; is_placeholder_path = True
        print(f"LLM_Interfacer Warning: Model path is a placeholder or not provided. Using standard placeholder: '{resolved_model_path_str}'.")
    resolved_model_path = Path(resolved_model_path_str)
    if not resolved_model_path.exists() or not resolved_model_path.is_file():
        error_msg = f"Model file does not exist at resolved path: {resolved_model_path}"
        if is_placeholder_path: error_msg += " (This is a placeholder path. Please provide a valid GGUF model path or place the model at './models/placeholder_deepseek.gguf' for it to be found by default.)"
        print(f"LLM_Interfacer Error: {error_msg}")
        return {**default_error_fingerprint, "error": error_msg}
    try:
        llm = Llama(model_path=str(resolved_model_path), n_gpu_layers=n_gpu_layers, n_ctx=2048, verbose=verbose)
    except Exception as e:
        error_msg = f"Error loading GGUF model from {resolved_model_path}: {e}"; print(f"LLM_Interfacer Error: {error_msg}")
        return {**default_error_fingerprint, "error": error_msg}
    system_prompt = ("You are a style analysis assistant. Based on the provided code statistics, generate a JSON object representing the style fingerprint. The JSON object must only contain the following keys, with appropriate values derived from the statistics: 'indent' (int), 'quotes' (string, e.g., 'single', 'double', 'mixed?'), 'linelen' (int), 'snake_pct' (float, 0.0-1.0), 'camel_pct' (float, 0.0-1.0), 'screaming_pct' (float, 0.0-1.0), 'docstyle' (string, e.g., 'google', 'numpy', 'plain', 'unknown?'). Ensure screaming_pct refers to UPPER_CASE_SNAKE_CASE.")
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": stats_block}]
    raw_json_output = "{}";
    try:
        response = llm.create_chat_completion(messages=messages, temperature=0.1, max_tokens=256)
        if response and response['choices'] and response['choices'][0]['message']['content']: raw_json_output = response['choices'][0]['message']['content'].strip()
        else: print("LLM_Interfacer Warning: GGUF model returned an empty or unexpected response for draft fingerprint.")
    except Exception as e:
        error_msg = f"Error during GGUF model inference for draft fingerprint: {e}"; print(f"LLM_Interfacer Error: {error_msg}")
        return {**default_error_fingerprint, "error": error_msg}
    try:
        json_match = re.search(r'\{.*\}', raw_json_output, re.DOTALL)
        if json_match: raw_json_output = json_match.group(0)
        parsed_fingerprint = json.loads(raw_json_output)
        if not isinstance(parsed_fingerprint, dict):
            error_msg = f"LLM output parsed to JSON but is not a dictionary: {type(parsed_fingerprint)}"; print(f"LLM_Interfacer Warning: {error_msg}. Raw output: {raw_json_output}")
            return {**default_error_fingerprint, "error": error_msg}
        current_keys = set(parsed_fingerprint.keys()); missing_keys = expected_fingerprint_keys - current_keys
        if missing_keys:
            error_msg = f"LLM output JSON is missing expected keys: {', '.join(missing_keys)}"; print(f"LLM_Interfacer Warning: {error_msg}. Parsed JSON: {parsed_fingerprint}")
            for key_to_add in missing_keys: parsed_fingerprint[key_to_add] = None
            final_fingerprint = {key: parsed_fingerprint.get(key) for key in expected_fingerprint_keys}; final_fingerprint["error"] = error_msg
            return final_fingerprint
        result_fingerprint = {key: parsed_fingerprint[key] for key in expected_fingerprint_keys}; result_fingerprint["error"] = None
        return result_fingerprint
    except json.JSONDecodeError as e:
        error_msg = f"GGUF model output was not valid JSON: '{raw_json_output}'. Error: {e}"; print(f"LLM_Interfacer Warning: {error_msg}")
        return {**default_error_fingerprint, "error": error_msg}
    except Exception as e_gen:
        error_msg = f"Unexpected error processing GGUF model output: {e_gen}. Raw output: '{raw_json_output}'"; print(f"LLM_Interfacer Warning: {error_msg}")
        return {**default_error_fingerprint, "error": error_msg}

def get_divot5_refined_output(code_snippet: str, raw_fingerprint_dict: Dict[str, Any], model_path: str, num_denoising_steps: int = 10, device: Optional[str] = None, verbose: bool = False) -> Optional[str]:
    if T5ForConditionalGeneration is None or T5TokenizerFast is None or torch is None:
        print("LLM_Interfacer Error: Transformers/PyTorch not installed. Cannot interact with DivoT5 model.")
        return None
    resolved_model_path_str = model_path; is_placeholder_path = False
    if not model_path or model_path == "path/to/your/divot5_model_dir":
        resolved_model_path_str = "./models/placeholder_divot5_refiner/"; is_placeholder_path = True
        print(f"LLM_Interfacer Warning: DivoT5 model path not provided or is a default placeholder. Using standard placeholder: '{resolved_model_path_str}'.")
    resolved_model_path = Path(resolved_model_path_str)
    if not resolved_model_path.exists() or not resolved_model_path.is_dir():
        error_msg = f"DivoT5 model directory does not exist at resolved path: {resolved_model_path}"
        if is_placeholder_path: error_msg += " (This is a placeholder path. Please provide a valid DivoT5 model directory or place the model at './models/placeholder_divot5_refiner/')"
        print(f"LLM_Interfacer Error: {error_msg}"); return None
    selected_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    if verbose: print(f"LLM_Interfacer: Attempting to load DivoT5 model from {resolved_model_path} onto device: {selected_device}")
    else: print(f"LLM_Interfacer: Loading DivoT5 model from {resolved_model_path}...")
    try:
        tokenizer = T5TokenizerFast.from_pretrained(str(resolved_model_path))
        model = T5ForConditionalGeneration.from_pretrained(str(resolved_model_path)).to(selected_device); model.eval()
    except Exception as e:
        print(f"LLM_Interfacer Error: Error loading DivoT5 model from {resolved_model_path}: {e}"); return None
    filtered_raw_fp_dict = {k: v for k, v in raw_fingerprint_dict.items() if k != "error" and v is not None}
    raw_fp_parts = []
    for key, value in filtered_raw_fp_dict.items():
        if isinstance(value, str): raw_fp_parts.append(f'"{key}": "{value}"')
        elif isinstance(value, (int, float)): raw_fp_parts.append(f'"{key}": {value}')
        elif isinstance(value, bool): raw_fp_parts.append(f'"{key}": {str(value).lower()}')
        else: raw_fp_parts.append(f'"{key}": "{str(value)}"')
    input_text = f"USER_CODE_START\n{code_snippet}\nUSER_CODE_END\nRAW_FINGERPRINT_START\n{', '.join(raw_fp_parts)}\nRAW_FINGERPRINT_END"
    try:
        if verbose: print(f"LLM_Interfacer: DivoT5 input text (first 500 chars):\n{input_text[:500]}...")
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(selected_device)
        output_max_length = max(200, min(num_denoising_steps * 40, 512))
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_length=output_max_length, num_beams=4, early_stopping=True, length_penalty=1.0)
        if outputs is not None and len(outputs) > 0:
            refined_output_string = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if verbose: print(f"LLM_Interfacer: DivoT5 raw output: '{refined_output_string}'")
            return refined_output_string
        else: print("LLM_Interfacer Warning: DivoT5 model returned an empty or unexpected response."); return None
    except Exception as e:
        print(f"LLM_Interfacer Error: Error during DivoT5 model inference: {e}"); return None

def get_divot5_code_infill(
    model_path: str,
    prompt: str,
    verbose: bool = False,
    device_str: Optional[str] = None,
    max_length_infill: int = 256,
    num_beams_infill: int = 3,
    temperature_infill: float = 0.5
) -> Optional[str]:
    """
    Performs code in-filling using a DivoT5 model.
    The prompt should be formatted according to the specific DivoT5 FIM model's requirements
    (e.g., using special tokens like <PREFIX>, <SUFFIX>, <MIDDLE>).
    """
    if T5ForConditionalGeneration is None or T5TokenizerFast is None or torch is None:
        print("LLM_Interfacer Error: Transformers/PyTorch not installed. Cannot use DivoT5 for code infill.")
        return None

    resolved_model_path_str = model_path
    is_placeholder_path = False
    default_placeholder = "./models/placeholder_divot5_infill_model/"

    if not model_path or model_path == "path/to/your/divot5_infill_model_dir" or model_path.endswith("placeholder_divot5_infill_model/"):
        resolved_model_path_str = default_placeholder
        is_placeholder_path = True
        print(f"LLM_Interfacer Warning: DivoT5 infill model path is a placeholder or not provided. Using standard placeholder: '{resolved_model_path_str}'.")

    resolved_model_path = Path(resolved_model_path_str)
    if not resolved_model_path.is_dir():
        error_msg = f"DivoT5 infill model directory does not exist at resolved path: {resolved_model_path}"
        if is_placeholder_path:
            error_msg += f" (This is a placeholder path. Ensure '{default_placeholder}' exists or provide a valid model path.)"
        print(f"LLM_Interfacer Error: {error_msg}")
        return None

    device = device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    log_prefix = "LLM_Interfacer (DivoT5 Infill):"

    if verbose: print(f"{log_prefix} Attempting to load model from {resolved_model_path} onto device: {device}")
    else: print(f"{log_prefix} Loading model from {resolved_model_path}...")

    try:
        tokenizer = T5TokenizerFast.from_pretrained(str(resolved_model_path))
        model = T5ForConditionalGeneration.from_pretrained(str(resolved_model_path)).to(device)
        model.eval()
    except Exception as e_load:
        print(f"{log_prefix} Error: Failed to load DivoT5 infill model from {resolved_model_path}: {e_load}")
        return None

    if verbose: print(f"{log_prefix} Prompt for infill (first 300 chars):\n{prompt[:300]}...")

    try:
        # Max_length for input tokenization; DivoT5 typically handles up to 1024 or 2048.
        # This should be long enough for prefix + suffix context.
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True, padding="longest").to(device)

        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length_infill, # Max length of the *generated* infill part
            num_beams=num_beams_infill,
            temperature=temperature_infill,
            early_stopping=True
        )

        if outputs is not None and len(outputs) > 0:
            infilled_code = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if verbose: print(f"{log_prefix} Raw infilled output: '{infilled_code}'")
            if not infilled_code: # Check if the decoded string is empty
                 print(f"{log_prefix} Warning: DivoT5 infill model returned an empty string after decoding.")
                 return None # Treat empty string as failure to infill
            return infilled_code
        else:
            print(f"{log_prefix} Warning: DivoT5 infill model returned no output or unexpected output structure.")
            return None

    except Exception as e_infer:
        print(f"{log_prefix} Error: Error during DivoT5 infill model inference: {e_infer}")
        return None

JSON_FINGERPRINT_GRAMMAR_STR = r'''
root   ::= object
value  ::= object | array | string | number | boolean | "null"
object ::= "{}" | "{" members "}"
members ::= pair ("," pair)*
pair   ::= string ":" value
array  ::= "[]" | "[" elements "]"
elements ::= value ("," value)*
string ::= "\"\"\" (([#x20-#x21] | [#x23-#x5B] | [#x5D-#xFFFF]) | #x5C ([#x22#x5C#x2F#x62#x66#x6E#x72#x74] | #x75[0-9a-fA-F]{4}))* "\"\"\""
number ::= ("-")? (("0") | ([1-9][0-9]*)) ("." [0-9]+)? (("e" | "E") (("-" | "+")?) [0-9]+)?
boolean ::= "true" | "false"
'''

def get_deepseek_polished_json(cleaned_key_value_string: str, model_path: str, n_gpu_layers: int = -1, verbose: bool = False) -> Optional[str]:
    if Llama is None or LlamaGrammar is None:
        error_msg = "llama-cpp-python or LlamaGrammar not available. Cannot polish JSON with GGUF model."
        print(f"LLM_Interfacer Error: {error_msg}")
        return None
    resolved_model_path_str = model_path; is_placeholder_path = False
    if not model_path or model_path == "path/to/your/deepseek-coder-gguf-model.gguf" or model_path.endswith("placeholder_deepseek.gguf"):
        resolved_model_path_str = "./models/placeholder_deepseek.gguf"; is_placeholder_path = True
        print(f"LLM_Interfacer Warning: Model path for JSON polishing is a placeholder or not provided. Using standard placeholder: '{resolved_model_path_str}'.")
    resolved_model_path = Path(resolved_model_path_str)
    if not resolved_model_path.exists() or not resolved_model_path.is_file():
        error_msg = f"GGUF model for JSON polishing does not exist at: {resolved_model_path}"
        if is_placeholder_path: error_msg += " (This is a placeholder path. Please provide a valid GGUF model or place it at './models/placeholder_deepseek.gguf')"
        print(f"LLM_Interfacer Error: {error_msg}"); return None
    try:
        llm = Llama(model_path=str(resolved_model_path), n_gpu_layers=n_gpu_layers, n_ctx=2048, verbose=verbose)
        grammar = LlamaGrammar.from_string(JSON_FINGERPRINT_GRAMMAR_STR)
    except Exception as e:
        error_msg = f"Error loading GGUF model or grammar from {resolved_model_path} for polish: {e}"; print(f"LLM_Interfacer Error: {error_msg}")
        return None
    system_prompt = ("You are a JSON formatting assistant. Convert the following string of key-value pairs into a single, valid JSON object. Ensure all keys and string values are double-quoted, and the overall structure is a correct JSON object. The required keys are 'indent', 'quotes', 'linelen', 'snake_pct', 'camel_pct', 'screaming_pct', and 'docstyle'. If a key is missing from the input, try to infer a sensible default or set its value to null if appropriate for the JSON structure, but ensure all listed keys are present in the final JSON.")
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": cleaned_key_value_string}]
    if verbose: print(f"LLM_Interfacer: Polishing JSON with prompt (user content snippet): {cleaned_key_value_string[:200]}...")
    try:
        response = llm.create_chat_completion(messages=messages, temperature=0.05, max_tokens=512, grammar=grammar) # Increased max_tokens
        if response and response['choices'] and response['choices'][0]['message']['content']:
            polished_json_string = response['choices'][0]['message']['content'].strip()
            if verbose: print(f"LLM_Interfacer: Raw polished JSON output from LLM: '{polished_json_string}'")
            try:
                json.loads(polished_json_string); return polished_json_string
            except json.JSONDecodeError as je:
                error_msg = f"LLM output for JSON polishing was not valid JSON despite grammar constraint: '{polished_json_string}'. Error: {je}"; print(f"LLM_Interfacer Warning: {error_msg}")
                return None
        else: print("LLM_Interfacer Warning: GGUF model (polish pass) returned an empty or unexpected response."); return None
    except Exception as e:
        error_msg = f"Error during GGUF model (polish pass) inference: {e}"; print(f"LLM_Interfacer Error: {error_msg}")
        return None

def get_llm_code_fix_suggestion(model_path: str, original_code_script: str, error_traceback: str, phase_description: str, target_file: Optional[str], additional_context: Optional[Dict[str, Any]], n_gpu_layers: int = -1, max_tokens: int = 2048, temperature: float = 0.4, verbose: bool = False) -> Optional[str]:
    if Llama is None:
        print("LLM_Interfacer Error: llama-cpp-python not installed. Cannot get LLM code fix suggestion.")
        return f"# Mock fix for error: {error_traceback[:100]}...\n# Original script had {len(original_code_script)} chars.\npass # LLM disabled - Apply actual fix here"
    resolved_model_path_str = model_path; is_placeholder_path = False
    if not model_path or model_path.endswith("placeholder_deepseek.gguf") or model_path.endswith("placeholder_repair_model.gguf"): # General placeholder check
        resolved_model_path_str = "./models/placeholder_repair_model.gguf"; is_placeholder_path = True # Standardize to a repair placeholder
        print(f"LLM_Interfacer Warning: LLM repair model path is a placeholder or not provided. Using standard placeholder: '{resolved_model_path_str}'.")
    resolved_model_path = Path(resolved_model_path_str)
    if not resolved_model_path.exists() or not resolved_model_path.is_file():
        error_msg = f"LLM repair model file does not exist at: {resolved_model_path}"
        if is_placeholder_path: error_msg += " (This is a placeholder. Provide a valid GGUF model path or place model here.)"
        print(f"LLM_Interfacer Error: {error_msg}")
        return f"# Mock fix due to model path error: {error_msg}\n# Error: {error_traceback[:100]}\npass"
    try:
        llm = Llama(model_path=str(resolved_model_path), n_gpu_layers=n_gpu_layers, n_ctx=4096, verbose=verbose)
    except Exception as e:
        print(f"LLM_Interfacer Error: Error loading LLM repair model from {resolved_model_path}: {e}")
        return f"# Mock fix due to model load error for: {error_traceback[:100]}...\npass"

    # Construct the prompt for the LLM
    # This structure should be clear for the LLM to understand distinct pieces of information.
    prompt = f"""You are an expert Python programmer and code assistant, specialized in writing and debugging LibCST refactoring scripts.
Your task is to revise the provided failing LibCST codemod script to address the specified error.

**Error Traceback:**
```text
{error_traceback}
```

**Original LibCST Script that Failed:**
```python
{original_code_script}
```

**Context for the Refactoring Task:**
- **Overall Goal:** {phase_description if phase_description else "Not specified."}
- **Target File:** {target_file if target_file else "Not specified."}
"""

    if additional_context:
        if 'style_profile' in additional_context and additional_context['style_profile']:
            try:
                style_profile_str = json.dumps(additional_context['style_profile'], indent=2)
                prompt += f"\n- **Project Style Profile (for context):**\n```json\n{style_profile_str}\n```\n"
            except (TypeError, OverflowError) as json_e:
                prompt += f"\n- **Project Style Profile (for context):** Error serializing - {json_e}\n"

        if 'code_snippets' in additional_context and additional_context['code_snippets']:
            snippets_str = ""
            for fname, snippet in additional_context['code_snippets'].items():
                snippets_str += f"  - Snippet from '{fname}':\n    ```python\n{snippet}\n    ```\n"
            if snippets_str:
                prompt += f"\n- **Relevant Code Snippets from Target File(s) (for context):**\n{snippets_str}"

    prompt += """
**Instructions:**
Please output a new, complete, and syntactically correct Python script for the revised LibCST codemod that addresses the error.
Focus on fixing the error indicated in the traceback in relation to the original LibCST script.
Output ONLY the Python code for the LibCST script. Do not include explanations, apologies, or markdown formatting before or after the script block.
Ensure the revised script is complete and runnable.

**Revised LibCST Script:**
```python
""" # End of prompt, LLM should start its response with the script content.

    if verbose: print(f"LLM Code Fix Prompt (first 1000 chars):\n{prompt[:1000]}...")
    else: print(f"LLM Code Fix Prompt (first 200 chars):\n{prompt[:200]}...")

    try:
        # Using create_completion as the prompt is now fully formed.
        # Stop sequences are kept to try and ensure only the script block is returned.
        response = llm.create_completion( # Changed from create_chat_completion back to create_completion
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["```python\n", "\n```\n", "\n```", "\n```text"], # Added \n```text as a potential stop
            echo=False
        )
        suggested_script = response['choices'][0]['text'].strip()

        # Clean up potential ```python prefix or ``` suffix if LLM includes them despite stop tokens.
        if suggested_script.startswith("```python"):
            suggested_script = suggested_script[len("```python"):].strip()
        if suggested_script.startswith("```"): suggested_script = suggested_script[len("```"):].strip()
        if suggested_script.endswith("```"): suggested_script = suggested_script[:-len("```")].strip()
        return suggested_script
    except Exception as e:
        print(f"LLM_Interfacer Error: Error during LLM code fix suggestion inference: {e}")
        return f"# Mock fix due to inference error for: {error_traceback[:100]}...\n# Original script length: {len(original_code_script)}\npass"


def get_llm_cst_scaffold(
    model_path: str,
    prompt: str, # Specifically crafted prompt for scaffold generation
    verbose: bool = False,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    max_tokens_for_scaffold: int = 2048, # Max tokens for the entire scaffold output
    temperature: float = 0.3,
    stop: Optional[List[str]] = None # e.g. ["</s>"] or specific output terminators
) -> Optional[str]: # Returns raw LLM string output (expected to be CST script + summary)
    """
    Generates a LibCST scaffold script and an edit summary using a GGUF model.
    """
    if Llama is None:
        print("LLM_Interfacer Error: LlamaCPP not installed, cannot perform CST scaffold generation.")
        return None # Or return a mock/placeholder string if that's more useful for callers

    resolved_model_path_str = model_path
    is_placeholder_path = False
    # Use a general agent placeholder if a specific scaffold model isn't implied by the path
    if not model_path or model_path.endswith((".placeholder_llm_agent.gguf", ".placeholder_deepseek.gguf")): # Check against common placeholders
        resolved_model_path_str = "./models/placeholder_llm_agent.gguf"
        is_placeholder_path = True
        print(f"LLM_Interfacer Warning: Using default model path '{resolved_model_path_str}' for CST scaffold generation (original path: '{model_path}').")

    resolved_model_path = Path(resolved_model_path_str)
    if not resolved_model_path.is_file():
        error_msg = f"LLM_Interfacer Error: Model file not found at '{resolved_model_path}' for CST scaffold generation."
        if is_placeholder_path:
            error_msg += f" (This was determined to be a placeholder path. Ensure '{resolved_model_path_str}' exists or provide a valid model path.)"
        print(error_msg)
        return None

    try:
        llm = Llama(
            model_path=str(resolved_model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose
        )
    except Exception as e_load:
        print(f"LLM_Interfacer Error: Error loading GGUF model from '{resolved_model_path}' for CST scaffold generation: {e_load}")
        return None

    if verbose:
        print(f"LLM_Interfacer: Generating CST scaffold with prompt (first 300 chars):\n{prompt[:300]}...")

    try:
        # Using create_completion for more direct control if chat format is not strictly needed by model
        # For models fine-tuned on chat, create_chat_completion might be better.
        # Assuming a model that can take direct instruction prompts.
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens_for_scaffold,
            temperature=temperature,
            stop=stop if stop else [], # Ensure stop is a list
            echo=False # Don't echo the prompt in the output
        )

        if response and response['choices'] and response['choices'][0]['text']:
            scaffold_output = response['choices'][0]['text'].strip()
            if verbose:
                print(f"LLM_Interfacer: Raw scaffold output from LLM (first 300 chars):\n{scaffold_output[:300]}...")
            return scaffold_output
        else:
            print("LLM_Interfacer Warning: LLM returned no content for CST scaffold generation.")
            return None

    except Exception as e_infer:
        print(f"LLM_Interfacer Error: Error during LLM CST scaffold generation inference: {e_infer}")
        return None


def get_llm_code_infill(
    model_path: str,
    prompt: str, # Specifically crafted prompt for in-filling
    verbose: bool = False,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    max_tokens_for_infill: int = 512, # Max tokens for the code snippet to fill a hole
    temperature: float = 0.4,
    stop: Optional[List[str]] = None
) -> Optional[str]: # Returns raw LLM string output for the infill
    """
    Fills in a code hole using a GGUF model based on the provided prompt.
    """
    if Llama is None:
        print("LLM_Interfacer Error: LlamaCPP not installed, cannot perform code infill.")
        return None

    resolved_model_path_str = model_path
    is_placeholder_path = False
    # Default to a general agent model if not specified or clearly a placeholder
    # The check model_path.endswith((".gguf", ".placeholder_llm_agent.gguf")) in the brief was a bit confusing.
    # Correct logic: if model_path is sensible, use it. If it's empty or looks like a known placeholder, use the default infill placeholder.
    if not model_path or model_path.endswith("placeholder_llm_agent.gguf") or model_path.endswith("placeholder_deepseek.gguf"):
        resolved_model_path_str = "./models/placeholder_llm_agent.gguf" # Default model for general agent tasks including infill
        is_placeholder_path = True
        print(f"LLM_Interfacer Warning: Using default model path '{resolved_model_path_str}' for code infill (original path: '{model_path}').")

    resolved_model_path = Path(resolved_model_path_str)
    if not resolved_model_path.is_file():
        error_msg = f"LLM_Interfacer Error: Model file not found at '{resolved_model_path}' for code infill."
        if is_placeholder_path:
            error_msg += f" (This was determined to be a placeholder. Ensure '{resolved_model_path_str}' exists or provide a valid model path.)"
        print(error_msg)
        return None

    try:
        llm = Llama(
            model_path=str(resolved_model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose
        )
    except Exception as e_load:
        print(f"LLM_Interfacer Error: Error loading GGUF model from '{resolved_model_path}' for code infill: {e_load}")
        return None

    if verbose:
        print(f"LLM_Interfacer: Performing code infill with prompt (first 300 chars):\n{prompt[:300]}...")

    try:
        # Using create_completion. For models specifically trained for infilling with FIM tokens (e.g., <PRE>, <SUF>, <MID>),
        # the prompt and model parameters (like `infill=True`) would need adjustment.
        # This implementation assumes a general instruction-following model.
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens_for_infill,
            temperature=temperature,
            stop=stop if stop else [], # Ensure stop is a list
            echo=False
        )

        if response and response['choices'] and response['choices'][0]['text']:
            infilled_code = response['choices'][0]['text'].strip()
            if verbose:
                print(f"LLM_Interfacer: Raw infill output from LLM:\n{infilled_code}")
            return infilled_code
        else:
            print("LLM_Interfacer Warning: LLM returned no content for code infill.")
            return None # Explicitly return None for empty content

    except Exception as e_infer:
        print(f"LLM_Interfacer Error: Error during LLM code infill inference: {e_infer}")
        return None


def get_llm_polished_cst_script(
    model_path: str,
    prompt: str, # Specifically crafted prompt for polishing a CST script
    verbose: bool = False,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    max_tokens_for_polished_script: int = 2048, # Max tokens for the full polished script
    temperature: float = 0.2 # Low temperature for precise edits
) -> Optional[str]: # Returns raw LLM string output for the polished script
    """
    Polishes a LibCST script using a GGUF model based on the provided prompt.
    """
    if Llama is None:
        print("LLM_Interfacer Error: LlamaCPP not installed, cannot perform CST script polishing.")
        return None

    resolved_model_path_str = model_path
    is_placeholder_path = False
    # Default to a general agent model if not specified or clearly a placeholder
    if not model_path or model_path.endswith("placeholder_llm_agent.gguf") or model_path.endswith("placeholder_deepseek.gguf"):
        resolved_model_path_str = "./models/placeholder_llm_agent.gguf" # Default model for general agent tasks
        is_placeholder_path = True
        print(f"LLM_Interfacer Warning: Using default model path '{resolved_model_path_str}' for CST script polishing (original path: '{model_path}').")

    resolved_model_path = Path(resolved_model_path_str)
    if not resolved_model_path.is_file():
        error_msg = f"LLM_Interfacer Error: Model file not found at '{resolved_model_path}' for CST script polishing."
        if is_placeholder_path:
            error_msg += f" (This was determined to be a placeholder. Ensure '{resolved_model_path_str}' exists or provide a valid model path.)"
        print(error_msg)
        return None

    try:
        llm = Llama(
            model_path=str(resolved_model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose
        )
    except Exception as e_load:
        print(f"LLM_Interfacer Error: Error loading GGUF model from '{resolved_model_path}' for CST script polishing: {e_load}")
        return None

    if verbose:
        print(f"LLM_Interfacer: Polishing CST script with prompt (first 300 chars):\n{prompt[:300]}...")

    try:
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens_for_polished_script,
            temperature=temperature,
            echo=False
        )

        if response and response['choices'] and response['choices'][0]['text']:
            polished_script_str = response['choices'][0]['text'].strip()
            if verbose:
                print(f"LLM_Interfacer: Raw polished script output from LLM (first 300 chars):\n{polished_script_str[:300]}...")
            return polished_script_str
        else:
            print("LLM_Interfacer Warning: LLM returned no content for CST script polishing.")
            return None

    except Exception as e_infer:
        print(f"LLM_Interfacer Error: Error during LLM CST script polishing inference: {e_infer}")
        return None

# Main block for testing (commented out as per original structure)
# if __name__ == '__main__':
#     # Test get_deepseek_draft_fingerprint
#     example_stats_for_main = """STATS_START
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
#     main_deepseek_model_path = "./models/placeholder_deepseek.gguf"
#     print("\n--- Testing DeepSeek Draft Fingerprint (from __main__) ---")
#     draft_fp_main = get_deepseek_draft_fingerprint(example_stats_for_main, model_path=main_deepseek_model_path, verbose=False)
#     print("\nDeepSeek Draft Fingerprint Output (from __main__):")
#     print(json.dumps(draft_fp_main, indent=2))

#     # Test get_divot5_refined_output
#     raw_fp_from_deepseek_for_divo_main = {
#         "indent": 4, "quotes": "mixed?", "linelen": 70,
#         "snake_pct": 0.60, "camel_pct": 0.30, "screaming_pct": 0.10,
#         "docstyle": "google?", "error": None
#     }
#     example_code_snippet_for_divo_main = "def myFunc(a_var):\n  return a_var # Example snippet for DivoT5"
#     main_divot5_model_path = "./models/placeholder_divot5_refiner/"
#     print("\n--- Testing DivoT5 Refined Output (from __main__) ---")
#     refined_key_values_main = get_divot5_refined_output(
#         code_snippet=example_code_snippet_for_divo_main,
#         raw_fingerprint_dict=raw_fp_from_deepseek_for_divo_main,
#         model_path=main_divot5_model_path,
#         verbose=False
#     )
#     print("\nDivoT5 Refined Key/Value String Output (from __main__):")
#     if refined_key_values_main is not None:
#         print(refined_key_values_main)
#     else:
#         print("DivoT5 refinement returned None (indicating an error).")

#     # Test get_deepseek_polished_json
#     divot5_output_example_main = '"indent": 4, "quotes": "single", "linelen": 110, "snake_pct": 0.82, "camel_pct": 0.12, "screaming_pct": 0.06, "docstyle": "google"'
#     # main_deepseek_model_path is already defined
#     print("\n--- Testing DeepSeek Polished JSON (from __main__) ---")
#     polished_fp_json_str_main = get_deepseek_polished_json(
#         cleaned_key_value_string=divot5_output_example_main,
#         model_path=main_deepseek_model_path, # Use the same GGUF model for this pass
#         verbose=False
#     )
#     print("\nDeepSeek Polished JSON String Output (from __main__):")
#     if polished_fp_json_str_main is not None:
#         print(polished_fp_json_str_main)
#         try:
#             # json import is at top of file
#             parsed_json_main = json.loads(polished_fp_json_str_main)
#             print("\nSuccessfully parsed the polished JSON (from __main__):")
#             print(json.dumps(parsed_json_main, indent=2))
#         except json.JSONDecodeError as e:
#             print(f"\nFailed to parse polished JSON (from __main__): {e}")
#     else:
#         print("DeepSeek JSON polishing returned None (indicating an error).")

#     # Test get_llm_code_fix_suggestion
#     main_repair_model_path = "./models/placeholder_repair_model.gguf" # Define a placeholder for repair model
#     print("\n--- Testing LLM Code Fix Suggestion (from __main__) ---")
#     mock_failed_script_main = "import libcst\ndef malformed_function((("
#     mock_traceback_main = "SyntaxError: unexpected EOF while parsing"
#     mock_phase_desc_main = "Add a new function to the module."
#     mock_target_file_main = "src/example/target.py"
#     mock_additional_context_main = {"style_profile": {"indent": 2, "quotes": "single"}}

#     suggested_fix_main = get_llm_code_fix_suggestion(
#         model_path=main_repair_model_path,
#         original_code_script=mock_failed_script_main,
#         error_traceback=mock_traceback_main,
#         phase_description=mock_phase_desc_main,
#         target_file=mock_target_file_main,
#         additional_context=mock_additional_context_main,
#         verbose=False
#     )
#     print(f"\nSuggested Code Fix (from __main__):\n{suggested_fix_main}")

#     # Test get_llm_score_for_text
#     main_scorer_model_path = "./models/placeholder_scorer.gguf"
#     print("\n--- Testing LLM Score for Text (from __main__) ---")
#     mock_scorer_prompt = "Evaluate the quality of this code snippet on a scale of 0.0 to 1.0: def foo(): pass"
#     score = get_llm_score_for_text(
#         model_path=main_scorer_model_path,
#         prompt=mock_scorer_prompt,
#         verbose=True
#     )
#     if score is not None:
#         print(f"LLM Score (from __main__): {score:.2f}")
#     else:
#         print("LLM Score (from __main__): Failed to get score.")

#     # Test get_llm_cst_scaffold
#     main_agent_model_path = "./models/placeholder_llm_agent.gguf" # Assume same model for scaffold and infill for now
#     print("\n--- Testing LLM CST Scaffold Generation (from __main__) ---")
#     mock_scaffold_prompt = "Generate a LibCST script to add a function 'my_new_test_func' with a placeholder body, and provide an edit summary."
#     scaffold_output = get_llm_cst_scaffold(
#         model_path=main_agent_model_path,
#         prompt=mock_scaffold_prompt,
#         verbose=True
#     )
#     if scaffold_output:
#         print(f"LLM Scaffold Output (from __main__):\n{scaffold_output}")
#     else:
#         print("LLM Scaffold Generation (from __main__): Failed to get output.")

#     # Test get_llm_code_infill
#     print("\n--- Testing LLM Code Infill (from __main__) ---")
#     mock_infill_prompt = "Complete the following Python code snippet inside the placeholder __HOLE_0__:\n```python\ndef my_function():\n    # Code before hole\n    __HOLE_0__\n    # Code after hole\n```"
#     infill_output = get_llm_code_infill(
#         model_path=main_agent_model_path, # Can use same model as scaffolding
#         prompt=mock_infill_prompt,
#         verbose=True
#     )
#     if infill_output:
#         print(f"LLM Infill Output (from __main__):\n{infill_output}")
#     else:
#         print("LLM Code Infill (from __main__): Failed to get output.")

#     # Test get_divot5_code_infill
#     main_divot5_infill_model_path = "./models/placeholder_divot5_infill_model/" # Example path
#     print("\n--- Testing DivoT5 Code Infill (from __main__) ---")
#     # Example FIM prompt - actual format depends on the DivoT5 FIM model training
#     # This is a common format: <PREFIX_FILE_PATH>path/to/file.py<PREFIX_BEFORE_CURSOR>def foo():\n    print("hello")\n <SUFFIX_AFTER_CURSOR>\n    print("world")<MIDDLE>
#     # Or a simpler natural language + context:
#     mock_divot5_infill_prompt = """Fill in the missing Python code.
# Context: We are trying to complete a function that prints two messages.
# File Path: example/test.py
# Code before missing part:
# ```python
# def my_incomplete_function():
#     print("Starting...")
# ```
# Code after missing part:
# ```python
#     print("Finished.")
# ```
# Fill in the middle part:"""

#     divot5_infill_output = get_divot5_code_infill(
#         model_path=main_divot5_infill_model_path,
#         prompt=mock_divot5_infill_prompt,
#         verbose=True
#     )
#     if divot5_infill_output:
#         print(f"DivoT5 Infill Output (from __main__):\n{divot5_infill_output}")
#     else:
#         print("DivoT5 Code Infill (from __main__): Failed to get output (this is expected if placeholder model/path is used).")
