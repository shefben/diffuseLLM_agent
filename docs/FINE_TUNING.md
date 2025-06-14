# Fine-Tuning AI Models for [Your Project Name]

This document provides guidance on how to use the data collected by this application to fine-tune the underlying AI models for improved performance and project-specific adaptation.

## Introduction

Fine-tuning allows the pre-trained Large Language Models (LLMs) and T5/DivoT5 models used in this project to better adapt to your specific codebase's style, common patterns, and the types of issues or features you typically request. This can lead to more accurate style profiles, better spec normalizations, and more relevant code generation/repair suggestions.

The primary source of data for fine-tuning code generation and repair tasks is the `success_memory.jsonl` file.

## Data Source: `success_memory.jsonl`

*   **Location:** This file is typically found in the data directory specified during `PhasePlanner` initialization (e.g., `project_root/.agent_data/success_memory.jsonl`).
*   **Format:** JSON Lines (JSONL) - each line is a separate JSON object.
*   **Schema (per line/entry):**
    *   `timestamp_utc` (str): ISO 8601 timestamp of when the entry was logged.
    *   `spec_issue_id` (Optional[str]): ID of the issue/spec.
    *   `spec_issue_description` (str): The raw issue description that led to the successful patch.
    *   `spec_target_files` (List[str]): Target files from the spec.
    *   `spec_operations_summary` (List[str]): Summary of operations from the spec.
    *   `successful_diff_summary` (str): A textual diff of the changes applied (first 1000 chars).
    *   `successful_script_hash` (str): A hash of the final validated LibCST script string that produced the diff. (The full script is not stored here to keep the log manageable, but could be cross-referenced if scripts are saved elsewhere, e.g., by CommitBuilder).
    *   `patch_source` (str): Information about which agent component or repair stage generated the successful script (e.g., "LLMCore_polish_after_repair_attempt_1").

**Note on Full Scripts for Fine-Tuning:** The `success_memory.jsonl` stores a *hash* of the successful LibCST script. To get the actual script content for fine-tuning, you would need to correlate these log entries with the patch sets saved by the `CommitBuilder` (e.g., in `.autopatches/<patch_set_name>/`), assuming the LibCST script itself is saved there or can be reconstructed from the final applied code. Future enhancements could log the full script if storage permits.

## Preparing Data for Fine-Tuning

The primary use of `success_memory.jsonl` is for fine-tuning the GGUF LLMs responsible for code generation, polishing, and repair tasks (Phases R5).

### For GGUF LLMs (Code Generation, Polishing, Repair)

You'll typically want to create prompt-completion pairs. The exact format depends on the fine-tuning library or scripts you use (e.g., Alpaca format, ShareGPT format).

*   **Repair Tasks (`LLMCore.propose_repair_diff`):**
    *   **Prompt:** Could be constructed from:
        *   The original failing LibCST script (would need to be logged or retrieved).
        *   The error traceback that occurred (would need to be logged).
        *   Contextual information (phase description, target file, style profile, code snippets related to the error).
        *   Instruction: "Revise the following failing LibCST script to fix the error..."
    *   **Completion:** The `successful_script` (the fixed LibCST script) that was ultimately validated.
    *   *Challenge:* `success_memory.jsonl` logs *successful* outcomes. To fine-tune repairs, you'd ideally need a log of *failure-attempt-success* cycles, including the failing script and traceback. The `patch_history` within `CollaborativeAgentGroup` (if saved per execution) would be a better source for this. This documentation should highlight that `success_memory.jsonl` is more for "positive examples" of final scripts.

*   **Scaffolding/Polishing Tasks (`LLMCore.generate_scaffold_patch`, `LLMCore.polish_patch`):**
    *   Fine-tuning these based *only* on the final `successful_script` from `success_memory.jsonl` is less direct.
    *   One approach for scaffolding: Use `spec_issue_description` as part of the prompt and the `successful_script` as the target completion, asking the LLM to generate the LibCST script that would achieve the spec.
    *   For polishing: This would require having "unpolished" and "polished" versions of scripts. `success_memory.jsonl` only has the final version.

**Recommendation:** For effective fine-tuning of code generation/repair, enhance logging to save:
1.  The input prompt to the LLM for each generative step.
2.  The raw output from the LLM.
3.  The final accepted/corrected version of the code/script after validation or human review.
The `patch_history` attribute of `CollaborativeAgentGroup` might contain some of this for specific runs if it were persisted.

### For T5/DivoT5 Models (Style Analysis, Spec Normalization)

Fine-tuning these models typically requires different kinds of input-output pairs not directly available in `success_memory.jsonl`:

*   **Spec Normalization (T5):**
    *   Input: Raw issue text + bag-of-symbols.
    *   Output: The structured YAML spec string.
    *   Data would come from pairs of user inputs and the corresponding YAML specs generated by `T5Client` (if these were logged).
*   **Style Profiling (DivoT5):**
    *   Input: Raw code snippets + (for refiner) draft fingerprints from GGUF.
    *   Output: Refined key-value fingerprint strings or final `UnifiedStyleProfile` JSON.
    *   Data would come from successful style profiling runs if intermediate data was logged.

## Fine-Tuning Procedures (General Guidance)

Always refer to the documentation of the specific models and fine-tuning libraries you intend to use.

### Hugging Face `transformers` Models (e.g., T5, DivoT5)

1.  **Prepare `Dataset`:** Load your prepared input-output pairs into Hugging Face `Dataset` objects.
2.  **Tokenizer:** Ensure you use the correct tokenizer for the model.
3.  **`Trainer` API:** Use the Hugging Face `Trainer` or a custom PyTorch training loop.
4.  **Resources:**
    *   Hugging Face Fine-tuning Tutorials: [https://huggingface.co/docs/transformers/training](https://huggingface.co/docs/transformers/training)

### GGUF Models (LLMs like DeepSeek Coder, Llama, Phi-2)

GGUF is a quantized format for inference. Fine-tuning is typically performed on the original model format (e.g., PyTorch/Safetensors from Hugging Face) and then converted to GGUF.

1.  **Choose a Base Model:** Start with a suitable pre-trained model (e.g., DeepSeek Coder, CodeLlama, Phi-2).
2.  **LoRA (Low-Rank Adaptation):** This is a popular and efficient technique for fine-tuning LLMs. Libraries like Hugging Face `PEFT` (Parameter-Efficient Fine-Tuning) provide tools for LoRA.
3.  **Fine-tuning Libraries/Scripts:**
    *   Hugging Face `transformers.Trainer` with `PEFT`.
    *   `trl` (Transformer Reinforcement Learning) for SFT (Supervised Fine-Tuning).
    *   Specialized fine-tuning scripts/repositories like `axolotl`, `unsloth` often simplify the process for popular models.
4.  **Data Format:** Prepare your data in the format required by your chosen fine-tuning script (e.g., JSONL with "prompt" and "completion" keys, or chat formats).
5.  **Conversion to GGUF:** After fine-tuning the PyTorch model (with or without LoRA merged), convert the fine-tuned model to GGUF format using the tools provided by `llama.cpp`.

## Model Configuration in This Application

Once you have fine-tuned your models and saved them locally (e.g., in the `./models/` directory):

*   Update the relevant model path configurations passed to `PhasePlanner` (and subsequently to `LLMCore`, `DiffusionCore`, `T5Client`, etc.) to point to your new fine-tuned model files or directories.
*   For example, update `llm_model_path`, `scorer_model_path`, `t5_spec_model_path`, `divot5_refiner_model_path`, `divot5_unifier_model_path`, `sentence_transformer_model_path` in your main application script or configuration files.

Regularly evaluating your fine-tuned models against a benchmark set of tasks relevant to your project will help track improvements.
