import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import time # For __main__ example, not strictly tests
import tempfile
import shutil
from typing import Callable, Optional, List, Union, Any, Dict # Added Union, Any, Dict for type hints

# Adjust imports based on actual project structure
# Assume Spec, SymbolRetriever, normalise_request, RepositoryDigester are importable for mocking/typing
try:
    from src.specs.schemas import Spec
except ImportError: # Fallback for testing if main path not set up
    from pydantic import BaseModel, Field
    class Spec(BaseModel): # Fallback Spec definition
        task: str = Field(..., description="Free-text summary.")
        target_symbols: List[str] = Field(default_factory=list)
        operations: List[str] = Field(default_factory=list)
        acceptance: List[str] = Field(default_factory=list)
        raw_output: Optional[str] = None
        raw_yaml: Optional[str] = None
        parsed_dict: Optional[Dict[str, Any]] = None
        class Config: extra = 'allow'


from src.spec_normalizer.spec_fusion import SpecFusion
# These are imported to be patched where SpecFusion uses them:
# from src.retriever.symbol_retriever import SymbolRetriever
# from src.spec_normalizer.t5_client import normalise_request
from src.digester.repository_digester import RepositoryDigester # For type hint and mock

# For mocking np if SymbolRetriever's __init__ checks it via digester.np_module
try:
    import numpy as np
except ImportError:
    np = None


class TestSpecFusion(unittest.TestCase):

    def setUp(self):
        self.mock_digester_instance = MagicMock(spec=RepositoryDigester)
        self.mock_digester_instance.embedding_model = MagicMock()
        self.mock_digester_instance.faiss_index = MagicMock()
        self.mock_digester_instance.np_module = np # Provide np reference for SymbolRetriever init

        self.t5_model_path = "dummy/t5_model"
        self.raw_request_example = "Test raw request"
        self.retriever_top_k_example = 30
        self.retriever_sim_threshold_example = 0.3

    @patch('src.spec_normalizer.spec_fusion.normalise_request')
    @patch('src.spec_normalizer.spec_fusion.SymbolRetriever')
    def test_generate_spec_from_request_success_flow(
        self, mock_symbol_retriever_class, mock_normalise_request_func
    ):
        mock_retriever_instance = MagicMock() # spec=SymbolRetriever removed as SymbolRetriever itself might be a placeholder
        mock_retriever_instance.retrieve_symbol_bag.return_value = "symbol1|symbol2"
        mock_symbol_retriever_class.return_value = mock_retriever_instance

        expected_spec_obj = Spec(task="Processed: Test raw request", target_symbols=["symbol1", "symbol2"])
        mock_normalise_request_func.return_value = expected_spec_obj

        spec_fuser = SpecFusion(
            t5_model_path=self.t5_model_path,
            digester=self.mock_digester_instance,
            retriever_top_k=self.retriever_top_k_example,
            retriever_sim_threshold=self.retriever_sim_threshold_example
        )

        t5_device_arg = "cpu"
        t5_max_input_arg = 512
        t5_max_output_arg = 256

        actual_spec = spec_fuser.generate_spec_from_request(
            self.raw_request_example,
            t5_device=t5_device_arg,
            t5_max_input_length=t5_max_input_arg,
            t5_max_output_length=t5_max_output_arg
        )

        mock_symbol_retriever_class.assert_called_once_with(self.mock_digester_instance)
        mock_retriever_instance.retrieve_symbol_bag.assert_called_once_with(
            self.raw_request_example,
            top_k=self.retriever_top_k_example,
            similarity_threshold=self.retriever_sim_threshold_example
        )
        mock_normalise_request_func.assert_called_once_with(
            raw_request_text=self.raw_request_example,
            symbol_bag_string="symbol1|symbol2",
            model_path=self.t5_model_path,
            device=t5_device_arg,
            max_input_length=t5_max_input_arg,
            max_output_length=t5_max_output_arg
        )
        self.assertEqual(actual_spec, expected_spec_obj)

    @patch('src.spec_normalizer.spec_fusion.normalise_request')
    @patch('src.spec_normalizer.spec_fusion.SymbolRetriever')
    def test_generate_spec_symbol_retrieval_fails(
        self, mock_symbol_retriever_class, mock_normalise_request_func
    ):
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_symbol_bag.side_effect = Exception("FAISS exploded")
        mock_symbol_retriever_class.return_value = mock_retriever_instance

        expected_spec_obj = Spec(task="Error: normalise_request failed or was not called properly")
        mock_normalise_request_func.return_value = expected_spec_obj

        spec_fuser = SpecFusion(self.t5_model_path, self.mock_digester_instance)

        with patch('builtins.print') as mock_print:
            actual_spec = spec_fuser.generate_spec_from_request(self.raw_request_example)

        mock_retriever_instance.retrieve_symbol_bag.assert_called_once_with(
            self.raw_request_example, top_k=50, similarity_threshold=0.25
        )
        self.assertTrue(any("Error during symbol retrieval: FAISS exploded" in str(c.args) for c in mock_print.call_args_list))

        mock_normalise_request_func.assert_called_once_with(
            raw_request_text=self.raw_request_example,
            symbol_bag_string="",
            model_path=self.t5_model_path,
            device=None, max_input_length=1024, max_output_length=512
        )
        self.assertEqual(actual_spec, expected_spec_obj)


    @patch('src.spec_normalizer.spec_fusion.normalise_request')
    @patch('src.spec_normalizer.spec_fusion.SymbolRetriever')
    def test_generate_spec_normalise_request_fails(
        self, mock_symbol_retriever_class, mock_normalise_request_func
    ):
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve_symbol_bag.return_value = "symbol1"
        mock_symbol_retriever_class.return_value = mock_retriever_instance

        error_spec = Spec(task="Error: T5 inference failed") # Simpler error spec
        mock_normalise_request_func.return_value = error_spec

        spec_fuser = SpecFusion(self.t5_model_path, self.mock_digester_instance)
        actual_spec = spec_fuser.generate_spec_from_request(self.raw_request_example)

        mock_normalise_request_func.assert_called_once()
        self.assertEqual(actual_spec, error_spec)
        self.assertIn("Error: T5 inference failed", actual_spec.task)

    @patch('src.spec_normalizer.spec_fusion.normalise_request') # To mock the call from generate_spec
    def test_generate_spec_symbol_retriever_unavailable(self, mock_normalise_request_func_inner):
        # This test checks behavior when SpecFusion's self.symbol_retriever is None
        # This happens if the SymbolRetriever class itself is None (placeholder) when SpecFusion is initialized

        expected_spec_obj = Spec(task="Generated with no symbol bag")
        mock_normalise_request_func_inner.return_value = expected_spec_obj

        with patch('src.spec_normalizer.spec_fusion.SymbolRetriever', None): # Make SymbolRetriever appear as None during SpecFusion init
            with patch('builtins.print') as mock_print:
                 spec_fuser = SpecFusion(self.t5_model_path, self.mock_digester_instance)

        self.assertIsNone(spec_fuser.symbol_retriever)
        self.assertTrue(any("SpecFusion initialized without a valid SymbolRetriever" in str(c.args) for c in mock_print.call_args_list))

        actual_spec = spec_fuser.generate_spec_from_request(self.raw_request_example)

        mock_normalise_request_func_inner.assert_called_once_with(
            raw_request_text=self.raw_request_example,
            symbol_bag_string="",
            model_path=self.t5_model_path,
            device=None, max_input_length=1024, max_output_length=512
        )
        self.assertEqual(actual_spec, expected_spec_obj)


if __name__ == '__main__':
    unittest.main()
