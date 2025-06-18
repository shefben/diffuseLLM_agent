from typing import Optional, Dict, Any, List
from pathlib import Path
import random  # For placeholder predict
import os  # For os.remove in __main__ if used

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore
    print(
        "Warning: joblib library not found. CorePredictor model persistence will be disabled."
    )

from src.utils.memory_logger import load_success_memory  # Added import


class CorePredictor:
    # Define at class level or in __init__. Let's do it here for clarity of definition.
    # These should represent the typical 'name' field from an OperationSpec in a Phase.
    TYPICAL_OPERATION_NAMES = sorted(
        [
            "add_function",
            "extract_method",
            "rename_variable",
            "add_decorator",
            "add_import",
            "remove_import",
            "change_function_signature",
            "modify_function_logic",
            "add_class",
            "modify_class_structure",
            "delete_file",
            "create_file",
            "add_comments",
            "remove_comments",
            "refactor_conditional",
            "optimize_loop",
            "replace_string_literal",
            "format_code_style",  # This might be too generic for core selection
            "unknown_operation",  # Fallback
        ]
    )

    def __init__(self, model_path: Optional[Path] = None, verbose: bool = False):
        self.verbose = verbose
        self.model_path = (
            model_path if model_path else Path("./models/core_predictor.joblib")
        )
        self.model: Any = None  # To store the loaded scikit-learn model
        self.label_encoder: Any = None  # To store the scikit-learn LabelEncoder
        self.is_ready: bool = False

        # Make KNOWN_OPERATION_TYPES an instance variable for easier access and potential modification per instance if ever needed.
        self.KNOWN_OPERATION_TYPES = self.TYPICAL_OPERATION_NAMES

        self._load_model()

    def _load_model(self) -> None:
        if joblib is None:
            if self.verbose:
                print(
                    "CorePredictor Info: joblib not available, model loading skipped."
                )
            self.is_ready = False
            return

        # Ensure parent directory exists before trying to load from it
        # Though for loading, we just check if the file itself exists.
        # self.model_path.parent.mkdir(parents=True, exist_ok=True) # More relevant for saving.

        if self.model_path.exists() and self.model_path.is_file():
            try:
                self.model = joblib.load(self.model_path)
                if self.verbose:
                    print(
                        f"CorePredictor Info: Model loaded successfully from {self.model_path}"
                    )

                # Attempt to load the label encoder
                label_encoder_path = self.model_path.parent / (
                    self.model_path.stem + "_label_encoder.joblib"
                )
                if label_encoder_path.exists():
                    self.label_encoder = joblib.load(label_encoder_path)
                    if self.verbose:
                        print(
                            f"CorePredictor Info: Label encoder loaded successfully from {label_encoder_path}"
                        )
                    self.is_ready = (
                        True  # Ready if model and label encoder (if present) are loaded
                    )
                elif self.model:  # Model loaded, but no label encoder found - might be an old model or error
                    print(
                        f"CorePredictor Warning: Model loaded from {self.model_path}, but corresponding label encoder not found at {label_encoder_path}. Predictions might be numerical if model outputs numbers."
                    )
                    # Consider it ready if model is there, but prediction needs to be careful.
                    # For robust operation, if label_encoder is expected, this could be self.is_ready = False
                    self.is_ready = (
                        True  # Or False, depending on strictness. Let's be optimistic.
                    )

            except Exception as e:
                print(
                    f"CorePredictor Error: Failed to load model or label encoder from {self.model_path} (or derived path): {e}"
                )
                self.model = None
                self.label_encoder = None
                self.is_ready = False
        else:
            if self.verbose:
                print(
                    f"CorePredictor Info: Model file not found at {self.model_path}. Predictor will use placeholder logic."
                )
            self.is_ready = False

    def predict(self, features: Dict[str, Any]) -> Optional[str]:
        if not self.is_ready or self.model is None:
            if self.verbose:
                print(
                    "CorePredictor Info: Model not ready or not loaded. Using placeholder prediction logic."
                )
            # Fall through to placeholder logic defined below
        else:  # Real model is loaded
            feature_vector = self._transform_features(features)
            if feature_vector is None:
                if self.verbose:
                    print(
                        "CorePredictor Error: Feature transformation failed. Cannot predict with real model, falling back to placeholder."
                    )
                # Fall through to placeholder logic
            else:
                try:
                    # Scikit-learn models expect a 2D array: [n_samples, n_features]
                    predicted_numerical_label_array = self.model.predict(
                        [feature_vector]
                    )
                    predicted_numerical_label = predicted_numerical_label_array[0]

                    if self.label_encoder:
                        predicted_label_str = self.label_encoder.inverse_transform(
                            [predicted_numerical_label]
                        )[0]
                        if self.verbose:
                            print(
                                f"CorePredictor: Model prediction (decoded): '{predicted_label_str}' from numerical '{predicted_numerical_label}' using features: {str(features)[:100]}..."
                            )
                        return predicted_label_str
                    else:
                        # If no label encoder, return the raw numerical prediction as string
                        predicted_label_raw_str = str(predicted_numerical_label)
                        if self.verbose:
                            print(
                                f"CorePredictor Warning: No label encoder loaded. Model prediction (raw numerical): '{predicted_label_raw_str}' from features: {str(features)[:100]}..."
                            )
                        return predicted_label_raw_str
                except Exception as e:
                    print(
                        f"CorePredictor Error: Real model prediction failed: {e}. Falling back to placeholder."
                    )
                    # Fall through to placeholder logic

        # Fallback / Placeholder prediction logic
        if self.verbose:
            print(
                f"CorePredictor: Using placeholder prediction logic. Features: {str(features)[:200]}..."
            )

        op_type = features.get("operation_type", "")

        if (
            "extract" in op_type.lower()
            or "refactor" in op_type.lower()
            or "rename" in op_type.lower()
        ):
            return "LLMCore"
        elif (
            "add" in op_type.lower()
            or "create" in op_type.lower()
            or "implement" in op_type.lower()
        ):
            return "LLMCore"

        return random.choice(["LLMCore", "DiffusionCore", "LLMCore"])

    def train(
        self, training_data_path: Path, model_output_path: Optional[Path] = None
    ) -> bool:
        if self.verbose:
            print(
                f"\nCorePredictor: train() called with data directory: {training_data_path}"
            )

        loaded_entries = load_success_memory(
            data_directory=training_data_path, verbose=self.verbose
        )

        if not loaded_entries:
            print(
                "CorePredictor Info: No training data loaded from success memory. Training cannot proceed."
            )
            return False

        X_features: List[List[float]] = []  # List of feature vectors
        y_labels: List[str] = []  # List of labels (e.g., "LLMCore", "DiffusionCore")

        for entry in loaded_entries:
            raw_features: Dict[str, Any] = {}

            label = entry.get("patch_source")
            if not label or not isinstance(label, str):
                if self.verbose:
                    print(
                        f"CorePredictor Warning: Skipping entry due to missing or invalid 'patch_source' (label): {entry.get('entry_id', 'Unknown ID')}"
                    )
                continue

            spec_operations_list = entry.get("spec_operations_details", [])
            raw_features["num_operations"] = len(spec_operations_list)

            op_type_counts = {op_name: 0 for op_name in self.KNOWN_OPERATION_TYPES}
            for op in spec_operations_list:
                if isinstance(op, dict):  # Ensure op is a dictionary
                    op_name = op.get("name", "unknown_operation")
                    if op_name in op_type_counts:
                        op_type_counts[op_name] += 1
                    else:
                        op_type_counts["unknown_operation"] += 1
                else:  # Handle case where an item in spec_operations_list is not a dict
                    op_type_counts["unknown_operation"] += 1
            raw_features.update(op_type_counts)

            # Old single operation_type feature is removed.
            # raw_features["operation_type"] = ...

            # Extract num_target_symbols (using num_target_files as proxy)
            target_files = entry.get("spec_target_files", [])
            raw_features["num_target_symbols"] = (
                len(target_files) if isinstance(target_files, list) else 0
            )

            # Placeholder comments for features not currently in success_memory.jsonl
            raw_features["num_input_code_lines"] = (
                0  # Placeholder: Not in current log. Would require parsing 'code_before' or similar from a richer log.
            )
            raw_features["num_parameters_in_op"] = (
                0  # Placeholder: Not in current log. Would require parsing 'spec_operations_summary' if it were structured dicts.
            )

            vectorized_features = self._transform_features(raw_features)
            if vectorized_features is not None:
                X_features.append(vectorized_features)
                y_labels.append(label)
            elif self.verbose:
                print(
                    f"CorePredictor Warning: Failed to transform features for entry: {entry.get('entry_id', 'Unknown ID')}. Raw features: {raw_features}"
                )

        if not X_features or not y_labels:
            print(
                "CorePredictor Info: No features extracted or labels available after processing loaded entries. Training cannot proceed."
            )
            return False

        print(f"  INFO: Extracted {len(X_features)} feature sets for training.")

        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier  # Or LogisticRegression
            from sklearn.metrics import classification_report
            from sklearn.preprocessing import LabelEncoder

            sklearn_available = True
        except ImportError:
            sklearn_available = False
            print(
                "CorePredictor Warning: scikit-learn not found. Actual model training will be skipped."
            )

        if not sklearn_available or not X_features or not y_labels:
            print(
                "CorePredictor Info: Skipping model training due to missing scikit-learn or no data."
            )
            return False

        # Label Encoding
        self.label_encoder = LabelEncoder()
        y_numerical = self.label_encoder.fit_transform(y_labels)
        if self.verbose:
            print(
                f"  INFO: Labels encoded. Classes: {list(self.label_encoder.classes_)}"
            )

        # Train/Test Split
        # Stratify if there's more than one unique label and enough samples per label
        stratify_option = (
            y_numerical
            if len(set(y_numerical)) > 1
            and len(y_numerical) > len(set(y_numerical)) * 2
            else None
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_features,
            y_numerical,
            test_size=0.2,
            random_state=42,
            stratify=stratify_option,
        )
        if self.verbose:
            print(
                f"  INFO: Data split. Train size: {len(X_train)}, Test size: {len(X_test)}"
            )

        # Model Instantiation and Training
        # Using simpler parameters for faster placeholder training
        classifier = RandomForestClassifier(
            random_state=42,
            n_estimators=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
        )
        try:
            classifier.fit(X_train, y_train)
            self.model = classifier
            self.is_ready = True  # Model is trained
            if self.verbose:
                print("  INFO: RandomForestClassifier model trained.")
        except Exception as e:
            print(f"CorePredictor Error: Model training failed: {e}")
            self.model = None
            self.is_ready = False
            return False

        # Evaluation
        if self.model and X_test:  # Ensure X_test is not empty
            try:
                y_pred = self.model.predict(X_test)
                # Ensure all predicted labels are known to the encoder before generating report
                # This can happen if a class is only in test set due to small data.
                # For simplicity, we'll assume fit_transform on all y_labels covers this for now.
                # For a robust solution, handle cases where y_pred might contain new labels not seen by encoder.
                # However, with stratify and proper data, this should be less of an issue.
                report_target_names = list(self.label_encoder.classes_)
                # If y_test or y_pred are empty, classification_report can fail.
                if len(y_test) > 0 and len(y_pred) > 0:
                    report = classification_report(
                        y_test,
                        y_pred,
                        target_names=report_target_names,
                        zero_division=0,
                    )
                    if self.verbose:
                        print(
                            "  INFO: Model evaluation complete. Classification report on test set:\n",
                            report,
                        )
                elif self.verbose:
                    print(
                        "  INFO: Skipping classification report due to empty y_test or y_pred."
                    )
            except Exception as e_report:
                if self.verbose:
                    print(
                        f"CorePredictor Warning: Could not generate classification report: {e_report}"
                    )
        elif self.verbose:
            print(
                "  INFO: Skipping model evaluation as model or test data is unavailable."
            )

        # Model Saving
        output_path = model_output_path if model_output_path else self.model_path
        if self.model and joblib:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.model, output_path)
                # Also save the label encoder
                if self.label_encoder:
                    label_encoder_path = output_path.parent / (
                        output_path.stem + "_label_encoder.joblib"
                    )
                    joblib.dump(self.label_encoder, label_encoder_path)
                    if self.verbose:
                        print(
                            f"  INFO: Trained model and label encoder saved to: {output_path} and {label_encoder_path}"
                        )
                else:  # Should not happen if training set self.label_encoder
                    if self.verbose:
                        print(
                            f"  INFO: Trained model saved to: {output_path}. Label encoder was not set."
                        )
                return True  # Successful training and saving
            except Exception as e:
                print(
                    f"CorePredictor Error: Failed to save trained model or label encoder to {output_path}: {e}"
                )
                self.is_ready = False  # Failed to save, so not fully ready
                return False
        elif not joblib:
            print(
                "CorePredictor Error: joblib not installed. Trained model cannot be saved."
            )
            self.is_ready = False  # Cannot persist model
            return False  # Indicate failure to save
        elif (
            not self.model
        ):  # Should not happen if training occurred and set self.model
            print("CorePredictor Info: No model was trained. Nothing to save.")
            # is_ready would be False from training failure
            return False  # Training effectively failed if no model produced

    # Placeholder for feature transformation logic needed before calling a real model's predict
    # def _transform_features(self, features: Dict[str, Any]) -> Any:
    #     # This would convert the dict of features into a numerical vector
    def _transform_features(self, features: Dict[str, Any]) -> Optional[List[float]]:
        if not isinstance(features, dict):
            if self.verbose:
                print("CorePredictor Error: Input features must be a dictionary.")
            return None

        feature_vector: List[float] = []

        # 1. Process Operation Type Counts
        # self.KNOWN_OPERATION_TYPES is sorted, ensuring consistent feature order.
        for known_op_name in self.KNOWN_OPERATION_TYPES:
            count = float(features.get(known_op_name, 0))
            feature_vector.append(
                min(count, 5.0)
            )  # Cap count at 5.0 as a simple scaling

        # 2. Process Numerical Features
        num_operations = float(features.get("num_operations", 0))
        feature_vector.append(min(num_operations / 10.0, 5.0))  # Scale and cap

        num_target_symbols = float(
            features.get("num_target_symbols", 0)
        )  # This is effectively num_target_files
        feature_vector.append(min(num_target_symbols / 10.0, 5.0))  # Scale and cap

        # Placeholders - these will be 0.0 for now as data isn't in logs
        num_input_code_lines = float(features.get("num_input_code_lines", 0))
        feature_vector.append(min(num_input_code_lines / 100.0, 5.0))

        num_parameters_in_op = float(features.get("num_parameters_in_op", 0))
        feature_vector.append(min(num_parameters_in_op / 5.0, 3.0))

        if self.verbose:
            print(
                f"CorePredictor: Transformed features to vector (length {len(feature_vector)}): {str(feature_vector)[:200]}..."
            )

        # Ensure the number of features matches what a trained model would expect.
        # This is a placeholder; a real model would have a fixed feature set size.
        # If self.model is trained, it might have n_features_in_ or similar.
        # For now, this is flexible.
        return feature_vector


if __name__ == "__main__":
    print("--- CorePredictor Conceptual Test ---")

    # Test without a pre-existing model file
    # Ensure the default models directory exists for this test's default model_path
    Path("./models").mkdir(parents=True, exist_ok=True)

    predictor_no_model = CorePredictor(
        model_path=Path("./models/non_existent_predictor.joblib"), verbose=True
    )

    # Updated sample features to reflect the new raw_features structure
    op_counts_1 = {op_name: 0 for op_name in predictor_no_model.KNOWN_OPERATION_TYPES}
    op_counts_1["add_function"] = 1
    sample_features_1 = {
        "num_operations": 1,
        "num_target_symbols": 1,
        "num_input_code_lines": 0,
        "num_parameters_in_op": 0,
    }
    sample_features_1.update(op_counts_1)
    prediction1 = predictor_no_model.predict(sample_features_1)
    print(f"Prediction for features_1 (add_function): {prediction1}")

    op_counts_2 = {op_name: 0 for op_name in predictor_no_model.KNOWN_OPERATION_TYPES}
    op_counts_2["extract_method"] = 2  # Example: two extract_method operations
    op_counts_2["add_import"] = 1
    sample_features_2 = {
        "num_operations": 3,  # 2 extract_method + 1 add_import
        "num_target_symbols": 2,
        "num_input_code_lines": 0,
        "num_parameters_in_op": 0,
    }
    sample_features_2.update(op_counts_2)
    prediction2 = predictor_no_model.predict(sample_features_2)
    print(f"Prediction for features_2 (2 extract_method, 1 add_import): {prediction2}")

    op_counts_3 = {op_name: 0 for op_name in predictor_no_model.KNOWN_OPERATION_TYPES}
    op_counts_3["unknown_operation"] = 1  # Test unknown operation
    sample_features_3 = {
        "num_operations": 1,
        "num_target_symbols": 1,
        "num_input_code_lines": 0,
        "num_parameters_in_op": 0,
    }
    sample_features_3.update(op_counts_3)
    prediction3 = predictor_no_model.predict(sample_features_3)
    print(f"Prediction for features_3 (unknown_operation): {prediction3}")

    # Setup for training test
    import tempfile
    import json  # For writing jsonl

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_data_dir = Path(temp_dir_name)
        dummy_training_data_path = temp_data_dir / "success_memory.jsonl"

        # Create dummy success_memory.jsonl data
        sample_log_entries = [
            {
                "timestamp_utc": "2023-01-01T10:00:00Z",
                "spec_issue_id": "ISSUE-001",
                "spec_issue_description": "Add a login function.",
                "spec_target_files": ["src/auth.py"],
                "spec_operations_details": [
                    {
                        "name": "add_function",
                        "target_file": "src/auth.py",
                        "parameters": {"function_name": "login"},
                    },
                    {
                        "name": "add_import",
                        "target_file": "src/auth.py",
                        "import_statement": "from flask import session",
                    },
                ],
                "successful_diff_summary": "+ def login():\n+  pass",
                "successful_script": "class AddLoginCommand(...): ...",
                "patch_source": "LLMCore",
            },
            {
                "timestamp_utc": "2023-01-02T11:00:00Z",
                "spec_issue_id": "ISSUE-002",
                "spec_issue_description": "Refactor user model for clarity.",
                "spec_target_files": ["src/models.py"],
                "spec_operations_details": [
                    {
                        "name": "extract_method",
                        "target_file": "src/models.py",
                        "parameters": {"method_name": "get_user_details"},
                    },
                    {
                        "name": "rename_variable",
                        "target_file": "src/models.py",
                        "parameters": {"old_name": "usr", "new_name": "user_record"},
                    },
                ],
                "successful_diff_summary": "- usr = ...\n+ user_record = ...",
                "successful_script": "class RefactorUserCommand(...): ...",
                "patch_source": "DiffusionCore",
            },
            {  # Entry to test "unknown_operation"
                "timestamp_utc": "2023-01-03T12:00:00Z",
                "spec_issue_id": "ISSUE-003",
                "spec_issue_description": "Implement a custom sorting algorithm.",
                "spec_target_files": ["src/utils.py"],
                "spec_operations_details": [
                    {
                        "name": "custom_sort_algo",
                        "target_file": "src/utils.py",
                        "parameters": {},
                    }
                ],
                "successful_diff_summary": "+ def custom_sort(): ...",
                "successful_script": "class CustomSortCommand(...): ...",
                "patch_source": "LLMCore",
            },
        ]
        with open(dummy_training_data_path, "w", encoding="utf-8") as f_log:
            for entry in sample_log_entries:
                f_log.write(json.dumps(entry) + "\n")

        print(
            f"\n--- Training CorePredictor with dummy data from {dummy_training_data_path} ---"
        )
        # Use a unique model path for this test training
        trained_model_path = temp_data_dir / "trained_test_core_predictor.joblib"
        predictor_for_training = CorePredictor(
            model_path=trained_model_path, verbose=True
        )
        training_success = predictor_for_training.train(
            training_data_path=temp_data_dir, model_output_path=trained_model_path
        )
        print(f"Training reported success: {training_success}")

        if training_success and predictor_for_training.is_ready:
            print("\n--- Predicting with newly trained model ---")
            # Use one of the sample_features defined earlier, or create new ones aligned with training data
            prediction_after_train = predictor_for_training.predict(
                sample_features_1
            )  # Re-use sample_features_1
            print(
                f"Prediction for features_1 (add_function) using trained model: {prediction_after_train}"
            )

            op_counts_eval = {
                op_name: 0 for op_name in predictor_for_training.KNOWN_OPERATION_TYPES
            }
            op_counts_eval["extract_method"] = 1
            op_counts_eval["rename_variable"] = 1
            sample_features_eval = {
                "num_operations": 2,
                "num_target_symbols": 1,
                "num_input_code_lines": 0,
                "num_parameters_in_op": 0,
            }
            sample_features_eval.update(op_counts_eval)
            prediction_eval = predictor_for_training.predict(sample_features_eval)
            print(
                f"Prediction for features representing (extract_method, rename_variable): {prediction_eval}"
            )

    # Original test with a pre-existing dummy model (if joblib and sklearn available)
    # This part is kept for basic model loading test, separate from training test.
    if joblib:
        # Use a more specific, test-only model path to avoid conflicts
        dummy_model_path = Path("./models/dummy_core_predictor_test.joblib")
        dummy_model_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure ./models exists

        # Attempt to import scikit-learn only if joblib is available
        try:
            from sklearn.linear_model import LogisticRegression

            # Create and save a dummy model
            dummy_clf = LogisticRegression()
            # Dummy fit: requires X (nsamples, nfeatures), y (nsamples)
            dummy_X = [[0, 0, 0], [1, 1, 1]]  # Assuming 3 features for this dummy model
            dummy_y = [0, 1]  # Binary classification: LLMCore vs DiffusionCore
            dummy_clf.fit(dummy_X, dummy_y)
            joblib.dump(dummy_clf, dummy_model_path)
            if predictor_no_model.verbose:
                print(f"\nDummy scikit-learn model saved to {dummy_model_path}")

            predictor_with_model = CorePredictor(
                model_path=dummy_model_path, verbose=True
            )
            if predictor_with_model.is_ready:
                # Note: predict() will still use placeholder logic unless _transform_features and model.predict are un-commented
                # and _transform_features is implemented to match the dummy_X structure.
                print(
                    f"Prediction (with dummy model loaded at {dummy_model_path}) for features_1: {predictor_with_model.predict(sample_features_1)}"
                )
            else:
                print(f"Failed to ready predictor_with_model using {dummy_model_path}.")

            if dummy_model_path.exists():
                os.remove(dummy_model_path)  # Clean up the dummy model
                if predictor_no_model.verbose:
                    print(f"Cleaned up dummy model: {dummy_model_path}")

        except ImportError:
            if predictor_no_model.verbose:
                print(
                    "\nsklearn.linear_model.LogisticRegression not found. Skipping dummy model save/load test with sklearn."
                )
        except Exception as e_main_test:
            if predictor_no_model.verbose:
                print(f"\nError in dummy model test part of main: {e_main_test}")
            # Clean up if model path was defined and file exists, even if error occurred
            if "dummy_model_path" in locals() and dummy_model_path.exists():
                try:
                    os.remove(dummy_model_path)
                except Exception:
                    pass
    else:
        if predictor_no_model.verbose:
            print("\njoblib not installed, skipping dummy model save/load test.")

    print("--- End CorePredictor Conceptual Test ---")
