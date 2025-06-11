# src/spec_normalizer/spec_fusion.py
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None # type: ignore
    print("Warning: PyYAML not installed. SpecFusion's YAML parsing will be disabled.")

try:
    from pydantic import ValidationError
except ImportError:
    ValidationError = None # type: ignore
    print("Warning: Pydantic not installed. SpecFusion's model validation will be disabled.")

# Imports for actual classes
from src.planner.spec_model import Spec # Assuming this is the correct path
from .t5_client import T5Client

if TYPE_CHECKING:
    from src.retriever.symbol_retriever import SymbolRetriever # For type hinting
else: # Runtime: try to import, or define a placeholder if it fails, for __main__ or basic runs
    try:
        from src.retriever.symbol_retriever import SymbolRetriever
    except ImportError:
        print("Warning: src.retriever.symbol_retriever.SymbolRetriever not found. Using placeholder for SpecFusion.")
        class SymbolRetriever: # type: ignore
            def __init__(self, digester: Any): self.digester = digester
            def get_all_fqns(self) -> List[str]: return ["placeholder.symbol1", "placeholder.symbol2"]


class SpecFusion:
    def __init__(
        self,
        t5_client: T5Client,
        symbol_retriever: 'SymbolRetriever',
        app_config: Dict[str, Any]
    ):
        """
        Initializes the SpecFusion component.
        Args:
            t5_client: An instance of T5Client.
            symbol_retriever: An instance of SymbolRetriever.
            app_config: Application configuration dictionary.
                        Expected keys:
                        - general.verbose (bool, optional)
        """
        self.t5_client = t5_client
        self.symbol_retriever = symbol_retriever
        self.app_config = app_config
        self.verbose = self.app_config.get("general", {}).get("verbose", False)
        if self.verbose:
            print(f"SpecFusion initialized. T5Client ready: {self.t5_client.is_ready}, SymbolRetriever type: {type(self.symbol_retriever)}")

    def normalise_request(self, raw_issue_text: str) -> Optional[Spec]:
        """
        Normalizes a raw issue text string into a structured Spec object.
        This involves retrieving context symbols, calling a T5 model to generate
        a YAML spec, parsing the YAML, and validating it into a Spec model.
        """
        if yaml is None:
            print("SpecFusion Error: PyYAML is not installed. Cannot parse YAML spec. Please install PyYAML.")
            return None
        if ValidationError is None and Spec is None : # Check if Pydantic related imports failed
             print("SpecFusion Error: Pydantic or Spec model not available. Cannot validate spec. Please ensure Pydantic is installed and Spec model is correctly defined.")
             return None


        if self.verbose:
            print(f"\nSpecFusion: Normalizing request: '{raw_issue_text[:100]}...'")

        # 1. Retrieve Enriched Context (FQNs and Success Examples)
        fqn_list: List[str] = []
        success_examples: List[Dict[str, str]] = []
        context_symbols_string = "No specific context symbols or examples retrieved." # Default

        if self.symbol_retriever:
            try:
                if self.verbose: print("SpecFusion: Retrieving enriched context (FQNs and examples)...")
                # Max_fqns and max_success_examples can be made configurable if needed.
                retrieved_context = self.symbol_retriever.get_enriched_context_for_spec_fusion(raw_issue_text)

                fqn_list = retrieved_context.get("fqns", [])
                success_examples = retrieved_context.get("success_examples", [])

                context_parts = []
                if fqn_list:
                    context_parts.append("Relevant Project Symbols FQNs: " + ", ".join(fqn_list))
                    if self.verbose: print(f"SpecFusion: Retrieved {len(fqn_list)} FQNs.")

                if success_examples:
                    example_strs = ["Previously successful related examples:"]
                    for i, ex in enumerate(success_examples, 1):
                        example_strs.append(f"Example {i}:\n  Issue: {ex.get('issue')}\n  Successful Script Preview: {ex.get('script_preview')}")
                    context_parts.append("\n".join(example_strs))
                    if self.verbose: print(f"SpecFusion: Retrieved {len(success_examples)} success examples.")

                if context_parts:
                    context_symbols_string = "\n\n".join(context_parts)

                if self.verbose and context_symbols_string != "No specific context symbols or examples retrieved.":
                     print(f"SpecFusion: Enriched context string preview (first 300 chars): '{context_symbols_string[:300]}...'")
                elif self.verbose:
                    print("SpecFusion: No FQNs or success examples retrieved to form context string.")

            except Exception as e_retrieval:
                print(f"SpecFusion Warning: Error during enriched context retrieval: {e_retrieval}. Proceeding with limited/no context.")
                context_symbols_string = f"Error retrieving enriched context: {str(e_retrieval)[:100]}"
        elif self.verbose:
            print("SpecFusion: SymbolRetriever not available. Proceeding without enriched context.")

        # 2. Call T5Client
        if self.verbose: print("SpecFusion: Requesting YAML spec from T5Client...")
        yaml_spec_str = self.t5_client.request_spec_from_text(raw_issue_text, context_symbols_string) # Pass the enriched context

        if yaml_spec_str is None:
            print("SpecFusion Error: T5Client failed to generate YAML spec string (returned None).")
            return None

        if self.verbose:
            print(f"SpecFusion: Received YAML spec string from T5Client (length: {len(yaml_spec_str)}):\n{yaml_spec_str[:500]}...")

        # 3. Parse YAML String
        parsed_dict: Optional[Dict[str, Any]] = None
        try:
            if self.verbose: print("SpecFusion: Parsing YAML spec string...")
            parsed_dict = yaml.safe_load(yaml_spec_str)
        except yaml.YAMLError as e:
            print(f"SpecFusion Error: Failed to parse YAML spec from T5. Error: {e}")
            if self.verbose: print(f"Problematic YAML string:\n{yaml_spec_str}")
            return None

        if not isinstance(parsed_dict, dict):
            print(f"SpecFusion Error: Parsed YAML is not a dictionary, but type {type(parsed_dict)}. Content: {str(parsed_dict)[:200]}")
            return None

        if self.verbose: print(f"SpecFusion: Successfully parsed YAML to dictionary: {list(parsed_dict.keys())}")

        # 4. Validate and Instantiate Spec Model
        spec_object: Optional[Spec] = None
        try:
            if self.verbose: print("SpecFusion: Validating dictionary and instantiating Spec model...")
            # Add raw inputs to the dict before Spec instantiation if desired
            parsed_dict["raw_issue_text"] = raw_issue_text
            parsed_dict["raw_yaml_spec"] = yaml_spec_str

            spec_object = Spec(**parsed_dict)
            if self.verbose: print("SpecFusion: Successfully normalized request to Spec object.")
            return spec_object
        except ValidationError as e_pydantic: # If Pydantic's ValidationError was imported
            print(f"SpecFusion Error: Failed to validate Spec model from parsed YAML. Errors:\n{e_pydantic}")
            return None
        except TypeError as e_type: # Catch potential TypeError if Spec(**parsed_dict) fails for other reasons
            print(f"SpecFusion Error: TypeError during Spec model instantiation. This might be due to unexpected fields or structure. Error: {e_type}")
            print(f"Parsed dictionary was: {parsed_dict}")
            return None


if __name__ == '__main__':
    from unittest.mock import MagicMock
    from src.utils.config_loader import load_app_config # For __main__
    # Ensure RepositoryDigester is available for the mock setup, even if just a placeholder
    try:
        from src.digester.repository_digester import RepositoryDigester
    except ImportError:
        print("Warning (__main__): src.digester.repository_digester.RepositoryDigester not found. Using placeholder.")
        class RepositoryDigester: # type: ignore
             def __init__(self, repo_path: Any): self.repo_path = repo_path; self.faiss_id_to_metadata = []
             # Add app_config to mock if real one takes it
            # def __init__(self, repo_path: Any, app_config: Dict[str, Any]):
            #     self.repo_path = repo_path; self.app_config = app_config; self.faiss_id_to_metadata = []

    print("--- SpecFusion __main__ Example ---")

    app_cfg_main = load_app_config()
    app_cfg_main["general"]["verbose"] = True
    if not app_cfg_main["general"].get("project_root"): # Needed by SymbolRetriever's data_dir logic
        app_cfg_main["general"]["project_root"] = str(Path.cwd())

    # Mock T5Client
    class MockT5Client:
        def __init__(self, app_config: Dict[str, Any]): # Updated mock signature
            self.app_config = app_config
            self.verbose = self.app_config.get("general",{}).get("verbose",False)
            self.is_ready = True # Assume ready for mock
            print(f"MockT5Client initialized, verbose: {self.verbose}")

        def request_spec_from_text(self, raw_issue_text: str, context_symbols_string: Optional[str] = None) -> Optional[str]:
            if self.verbose: print(f"MockT5Client.request_spec_from_text called. Issue: '{raw_issue_text[:50]}...', Symbols: '{str(context_symbols_string)[:50]}...'")
            # Simulate YAML output based on input
            if "add function" in raw_issue_text:
                return """
issue_description: "Add a new function `calculate_total` to `billing.py`"
target_files:
  - "src/billing.py"
operations:
  - name: "add_function"
    target_file: "src/billing.py"
    function_name: "calculate_total"
    parameters: {"items": "List[Item]", "discount": "Optional[float]"}
    return_type: "float"
    docstring: "Calculates the total price after applying an optional discount."
    body_scaffold: "total = sum(item.price for item in items)\nif discount:\n  total *= (1 - discount)\nreturn total"
acceptance_tests:
  - "Test with items and no discount."
  - "Test with items and a discount."
  - "Test with empty items list."
"""
            else:
                return f"issue_description: '{raw_issue_text}'\ntarget_files: ['unknown.py']\noperations: []\nacceptance_tests: ['basic_test']"

    # Mock SymbolRetriever (if the real one isn't available or for isolated testing)
    # If real SymbolRetriever was imported successfully, SymbolRetrieverToUse will be it.
    # Otherwise, MockSymbolRetrieverForFusionMain will be used.
    # This setup allows testing with the real SymbolRetriever if available and properly configured,
    # or falls back to a local mock for this __main__ block.

    # Try to use the real SymbolRetriever first
    try:
        from src.retriever.symbol_retriever import SymbolRetriever as RealSymbolRetriever
        # Check if the imported one is not the placeholder already defined at file level for TYPE_CHECKING
        if RealSymbolRetriever.__module__ == "src.retriever.symbol_retriever":
            SymbolRetrieverToUseInMain = RealSymbolRetriever
            print("Info (__main__): Using real SymbolRetriever.")
        else: # Fallback if import was tricky (e.g. relative import issues in __main__)
            raise ImportError("Imported SymbolRetriever is not the expected one.")
    except ImportError:
        print("Warning (__main__): Real SymbolRetriever not found or import issue. Using local MockSymbolRetrieverForFusionMain.")
        class MockSymbolRetrieverForFusionMain:
            def __init__(self, digester: Any, app_config: Dict[str, Any]): # Updated mock signature
                self.app_config = app_config
                self.verbose = self.app_config.get("general",{}).get("verbose",False)
                print(f"MockSymbolRetrieverForFusionMain initialized, verbose: {self.verbose}")
            def get_enriched_context_for_spec_fusion(self, raw_issue_text: str, max_fqns: int = 300, max_success_examples: int = 2) -> Dict[str, Any]: # Match real signature
                if self.verbose: print(f"MockSymbolRetrieverForFusionMain.get_enriched_context_for_spec_fusion called, issue: '{raw_issue_text[:20]}...'")
                symbols = ["example.utils.helper_a", "example.services.main_service.process_data", "another.mock.symbol"]
                return {"fqns": symbols[:max_fqns] if max_fqns is not None else symbols, "success_examples": []}
        SymbolRetrieverToUseInMain = MockSymbolRetrieverForFusionMain # type: ignore

    # Setup
    # Use a more complete MockDigester for SymbolRetriever, or the real one if testing integration
    class MockDigesterForFusionMain:
        def __init__(self, repo_path):
            self.repo_path = Path(repo_path)
            self.faiss_id_to_metadata = [ # Populate with some data for get_context_symbols_for_spec_fusion
                {"fqn": "example.utils.helper_a", "item_type": "function_code"},
                {"fqn": "example.services.main_service.process_data", "item_type": "method_code"},
                {"fqn": "example.models.User", "item_type": "class_code"},
                {"fqn": "example.module_overview", "item_type": "docstring_for_module"},
                {"fqn": "example.constants.MAX_USERS", "item_type": "global_variable"}, # This might be filtered out by SymbolRetriever
            ]
            self.embedding_model = MagicMock() # SymbolRetriever __init__ checks this
            self.faiss_index = MagicMock()     # And this
            self.np_module = MagicMock()       # And this (if np is None globally in retriever)
             # Add app_config if real RepositoryDigester takes it (it does now)
            # self.app_config = app_cfg_main # Assuming it's in scope
            print(f"MockDigesterForFusionMain initialized for {repo_path}")

    # If RepositoryDigester takes app_config, pass it to the mock.
    # mock_digester = RepositoryDigester(repo_path="dummy_repo_for_specfusion_main", app_config=app_cfg_main) # If using real
    mock_digester = MockDigesterForFusionMain(repo_path="dummy_repo_for_specfusion_main") # Basic mock

    mock_t5_client = MockT5Client(app_config=app_cfg_main)

    try:
        # If SymbolRetrieverToUseInMain is the real one, it needs a valid digester and app_config.
        mock_symbol_retriever = SymbolRetrieverToUseInMain(digester=mock_digester, app_config=app_cfg_main) # type: ignore
    except Exception as e_sr_init:
        print(f"Error initializing SymbolRetrieverToUseInMain: {e_sr_init}. Falling back to basic mock for __main__.")
        # Fallback to an even simpler mock if the chosen one fails with MockDigester
        class BasicMockSymbolRetriever:
            def __init__(self, digester:Any, app_config: Dict[str, Any]): pass
            def get_enriched_context_for_spec_fusion(self, raw_issue_text: str, max_fqns: int = 300, max_success_examples: int = 2) -> Dict[str, Any]:
                return {"fqns": ["fallback.symbol1", "fallback.symbol2"][:max_fqns] if max_fqns is not None else [], "success_examples": []}
        mock_symbol_retriever = BasicMockSymbolRetriever(digester=mock_digester, app_config=app_cfg_main)


    spec_fuser = SpecFusion(
        t5_client=mock_t5_client,
        symbol_retriever=mock_symbol_retriever,
        app_config=app_cfg_main
    )

    test_issue_1 = "User wants to add function `calculate_total` in `billing.py` with items and discount."
    print(f"\n--- Test Case 1: {test_issue_1} ---")
    spec_obj_1 = spec_fuser.normalise_request(test_issue_1)

    if spec_obj_1:
        print("\nSuccessfully generated Spec object 1:")
        # Assuming Spec has a .model_dump_json() method (Pydantic V2) or similar
        if hasattr(spec_obj_1, 'model_dump_json'):
            print(spec_obj_1.model_dump_json(indent=2))
        else: # Fallback for Pydantic V1 or non-Pydantic
            print(spec_obj_1)
    else:
        print("\nFailed to generate Spec object 1.")

    test_issue_2 = "A generic bug fix is needed."
    print(f"\n--- Test Case 2: {test_issue_2} ---")
    spec_obj_2 = spec_fuser.normalise_request(test_issue_2)
    if spec_obj_2:
        print("\nSuccessfully generated Spec object 2:")
        if hasattr(spec_obj_2, 'model_dump_json'):
            print(spec_obj_2.model_dump_json(indent=2))
        else:
            print(spec_obj_2)
    else:
        print("\nFailed to generate Spec object 2.")

    print("\n--- SpecFusion __main__ Example Done ---")
