# Training the CorePredictor Model

The `CorePredictor` (`src/learning/core_predictor.py`) is designed to predict which agent core (`LLMCore` or `DiffusionCore`) might be more suitable for a given task, based on features derived from the phase context. This document outlines how to collect data and train an initial version of this classifier model.

Training is an **offline process**; the application uses a pre-trained model specified by `model_path` in the `CorePredictor`'s configuration (defaulting to `./models/core_predictor.joblib`).

## 1. Data Collection

The primary data sources for training the `CorePredictor` are:
*   `success_memory.jsonl`: Located in the agent's data directory (e.g., `project_root/.agent_data/success_memory.jsonl`). This file logs details of successfully applied patches. Each entry contains:
    *   `spec_issue_description`
    *   `spec_target_files`
    *   `spec_operations_summary` (list of operation names from the `Spec`)
    *   `patch_source` (e.g., "LLMCore_polish_initial", "LLMCore_repair_attempt_1", "DiffusionCore_expansion_v1"). This will serve as the **label** for our classifier.
*   `patch_meta.json` files: Located in each saved patch set directory (e.g., `project_root/.autopatches/<patch_set_name>/patch_meta.json`). This also contains `patch_source` and the original `spec` data.
*   **(Future Enhancement):** Ideally, logging failures and the `patch_history` from `CollaborativeAgentGroup` for each run (successful or not) would provide richer data, including which core was attempted even if it ultimately failed before another succeeded.

**Process:**
1.  **Gather Data:** Collect all `success_memory.jsonl` entries and/or `patch_meta.json` files from multiple agent runs and diverse tasks.
2.  **Label Extraction:** The `patch_source` field is the target variable. You'll need to map these string labels to a consistent format if necessary (e.g., map all "LLMCore_*" to "LLMCore", and "DiffusionCore_*" to "DiffusionCore" if you're predicting the initial successful core type rather than the exact step).
3.  **Feature Extraction (Conceptual - to match `_transform_features`):**
    For each data point (successful patch), you need to reconstruct or extract the features that `CorePredictor._transform_features` would expect. This is the most challenging part as `success_memory.jsonl` doesn't store the raw `phase_ctx` or `context_data` directly.
    *   **`operation_type`:** Can be derived from `spec_operations_summary[0]` (the first operation name) if the prediction is for the initial phase. If predicting for a specific step in `CollaborativeAgentGroup` (e.g. scaffold vs expand), then more detailed logging of the `phase_ctx` for *that specific step* would be needed.
    *   **`num_input_code_lines`:** This would require access to the original code snippets associated with the phase that `patch_source` refers to. Not directly in `success_memory.jsonl`.
    *   **`num_target_symbols`:** Can be estimated from `len(spec_target_files)` or by parsing parameters from `spec_operations_summary`.
    *   **`num_parameters_in_op`:** Can be derived by inspecting the detailed `parameters` field if the full `Spec` object corresponding to the entry is retrieved or logged.
    *   **Recommendation:** For robust training, the system should log the exact `features: Dict[str, Any]` dictionary that *would have been fed* to `CorePredictor.predict()` alongside the `patch_source` that ultimately succeeded for that phase. This logging could be added to `CollaborativeAgentGroup.run()` when a final patch is chosen.

## 2. Feature Engineering

The `CorePredictor._transform_features` method provides the blueprint for converting the raw feature dictionary into a numerical vector:
*   **One-Hot Encoding:** `operation_type` is one-hot encoded based on `CorePredictor.KNOWN_OPERATION_TYPES`.
*   **Numerical Features:** `num_input_code_lines`, `num_target_symbols`, `num_parameters_in_op` are scaled and capped.
Ensure your offline feature extraction process mirrors this transformation.

## 3. Model Training (Offline)

1.  **Prepare Dataset:** Create a dataset where each row consists of the transformed feature vector and the corresponding label (e.g., "LLMCore" or "DiffusionCore").
2.  **Choose a Classifier:** A simple logistic regression, decision tree, or random forest from `scikit-learn` is a good starting point.
    ```python
    # Example using scikit-learn
    # from sklearn.model_selection import train_test_split
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import classification_report
    # import joblib

    # Assume X is your list of feature vectors, y is your list of labels
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model = LogisticRegression()
    # model.fit(X_train, y_train)

    # print(classification_report(y_test, model.predict(X_test)))

    # Save the trained model
    # model_save_path = Path("./models/core_predictor.joblib")
    # model_save_path.parent.mkdir(parents=True, exist_ok=True)
    # joblib.dump(model, model_save_path)
    # print(f"Trained CorePredictor model saved to {model_save_path}")
    ```
3.  **Evaluate:** Evaluate the model on a hold-out test set.
4.  **Save Model:** If satisfied, save the trained model using `joblib.dump()` to the path configured in `CorePredictor` (default: `./models/core_predictor.joblib`).

## 4. Using the Trained Model

Once the `core_predictor.joblib` file is in the `./models/` directory (or the path configured for `CorePredictor`), the application will automatically load and use it for predictions.

## Future Enhancements
*   Log more detailed feature sets during agent runs for easier training data creation.
*   Implement more sophisticated feature engineering.
*   Experiment with different classification models.
*   Track prediction accuracy and retrain periodically.
