import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import json # For Pydantic model_dump_json in asserts, and dicts
import yaml # For YAMLError and yaml.dump
from typing import Tuple, Optional, List, Dict, Any # For type hints

# Adjust imports based on actual project structure
try:
    from src.specs.schemas import Spec # The Pydantic model
except ImportError:
    from pydantic import BaseModel, Field
    class Spec(BaseModel): # Fallback Spec definition for testing
        task: str = Field(..., description="Free-text summary.")
        target_symbols: List[str] = Field(default_factory=list)
        operations: List[str] = Field(default_factory=list)
        acceptance: List[str] = Field(default_factory=list)
        raw_output: Optional[str] = None
        raw_yaml: Optional[str] = None
        parsed_dict: Optional[Dict[str, Any]] = None
        class Config: extra = 'allow'


from src.spec_normalizer.t5_client import normalise_request, _load_t5_model_and_tokenizer, _model_cache, _tokenizer_cache
# Import Transformers components for type checking mocks and patching globals
# These are patched in setUp via 'src.spec_normalizer.t5_client.COMPONENT_NAME'
# import src.spec_normalizer.t5_client

class TestT5Client(unittest.TestCase):

    def setUp(self):
        self.model_path_example = "dummy/t5-model-for-spec"
        self.raw_request_example = "add feature X"
        self.symbol_bag_example = "module.func_A|module.ClassB"

        self.expected_spec_dict = {
            "task": "add feature X with context",
            "target_symbols": ["module.func_A", "module.ClassB"],
            "operations": ["add_function", "modify_class"],
            "acceptance": ["test_feature_x_added"]
        }
        self.valid_yaml_output_from_model = f"<SPEC_YAML>\n{yaml.dump(self.expected_spec_dict)}</SPEC_YAML>"

        _model_cache.clear()
        _tokenizer_cache.clear()

        self.mock_tokenizer_instance = MagicMock()
        # Simulate tokenizer __call__ behavior
        self.mock_tokenizer_instance.return_value = MagicMock(
            input_ids="dummy_input_ids_tensor",
            attention_mask="dummy_attention_mask_tensor"
        )
        # Simulate tokenizer decode behavior
        self.mock_tokenizer_instance.decode.return_value = self.valid_yaml_output_from_model


        self.mock_model_instance = MagicMock()
        # Simulate model generate behavior
        self.mock_model_instance.generate.return_value = MagicMock(spec=list) # e.g., [torch.tensor([...])]
        # Simulate model to(device) behavior
        self.mock_model_instance.to.return_value = self.mock_model_instance
        self.mock_model_instance.eval = MagicMock()


        self.patcher_auto_tokenizer = patch('src.spec_normalizer.t5_client.AutoTokenizer.from_pretrained', return_value=self.mock_tokenizer_instance)
        self.patcher_auto_model = patch('src.spec_normalizer.t5_client.AutoModelForSeq2SeqLM.from_pretrained', return_value=self.mock_model_instance)

        self.MockAutoTokenizer = self.patcher_auto_tokenizer.start()
        self.MockAutoModel = self.patcher_auto_model.start()

        # Mock torch availability and cuda.is_available()
        self.mock_torch_cuda_is_available = MagicMock(return_value=False) # Default to CPU
        self.patcher_torch_module = patch('src.spec_normalizer.t5_client.torch',
                                          MagicMock(cuda=MagicMock(is_available=self.mock_torch_cuda_is_available),
                                                    no_grad=MagicMock, # For with torch.no_grad()
                                                    tensor=MagicMock() # If any tensor ops are done directly
                                                    ))
        self.MockTorch = self.patcher_torch_module.start()

        # Use real yaml for safe_load but patch it to ensure it's seen as available by t5_client
        self.patcher_yaml_module = patch('src.spec_normalizer.t5_client.yaml', yaml)
        self.MockYaml = self.patcher_yaml_module.start()


    def tearDown(self):
        self.patcher_auto_tokenizer.stop()
        self.patcher_auto_model.stop()
        self.patcher_torch_module.stop()
        self.patcher_yaml_module.stop()


    def test_load_t5_model_and_tokenizer_success_and_cache(self):
        model1, tokenizer1 = _load_t5_model_and_tokenizer(self.model_path_example, "cpu")
        self.assertIsNotNone(model1)
        self.assertIsNotNone(tokenizer1)
        self.MockAutoTokenizer.from_pretrained.assert_called_once_with(self.model_path_example)
        self.MockAutoModel.from_pretrained.assert_called_once_with(self.model_path_example)
        self.mock_model_instance.to.assert_called_with("cpu")
        self.mock_model_instance.eval.assert_called_once()


        model2, tokenizer2 = _load_t5_model_and_tokenizer(self.model_path_example, "cpu")
        self.assertIs(model1, model2)
        self.assertIs(tokenizer1, tokenizer2)
        self.MockAutoTokenizer.from_pretrained.assert_called_once()
        self.MockAutoModel.from_pretrained.assert_called_once()


    @patch('src.spec_normalizer.t5_client._load_t5_model_and_tokenizer')
    def test_normalise_request_success(self, mock_load_model_tok):
        mock_load_model_tok.return_value = (self.mock_model_instance, self.mock_tokenizer_instance)
        # Configure .generate to return something that decode can handle
        self.mock_model_instance.generate.return_value = ["dummy_output_sequence_tensor"]
        self.mock_tokenizer_instance.decode.return_value = self.valid_yaml_output_from_model

        spec = normalise_request(self.raw_request_example, self.symbol_bag_example, self.model_path_example)

        mock_load_model_tok.assert_called_once_with(self.model_path_example, "cpu") # Assuming CPU default

        expected_input_text = f"<RAW_REQUEST> {self.raw_request_example} </RAW_REQUEST> <SYMBOL_BAG> {self.symbol_bag_example} </SYMBOL_BAG>"
        self.mock_tokenizer_instance.assert_called_once_with(
            expected_input_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length"
        )
        self.mock_model_instance.generate.assert_called_once()
        call_args = self.mock_model_instance.generate.call_args
        self.assertEqual(call_args.kwargs.get('temperature'), 0.3)
        self.assertEqual(call_args.kwargs.get('top_p'), 0.9)
        self.assertEqual(call_args.kwargs.get('max_length'), 512)

        self.mock_tokenizer_instance.decode.assert_called_once_with("dummy_output_sequence_tensor", skip_special_tokens=True)

        self.assertIsInstance(spec, Spec)
        self.assertEqual(spec.task, self.expected_spec_dict["task"])
        self.assertEqual(spec.target_symbols, self.expected_spec_dict["target_symbols"])

    @patch('src.spec_normalizer.t5_client.torch', None)
    def test_normalise_request_dependencies_not_available(self, mock_torch_none):
        spec = normalise_request(self.raw_request_example, self.symbol_bag_example, self.model_path_example)
        self.assertTrue(spec.task.startswith("Error: T5 client dependencies"))

    @patch('src.spec_normalizer.t5_client._load_t5_model_and_tokenizer', return_value=(None, None))
    def test_normalise_request_model_load_fails(self, mock_load_model_tok):
        spec = normalise_request(self.raw_request_example, self.symbol_bag_example, self.model_path_example)
        self.assertEqual(spec.task, f"Error: Failed to load T5 model/tokenizer from {self.model_path_example}.")

    @patch('src.spec_normalizer.t5_client._load_t5_model_and_tokenizer')
    def test_normalise_request_generate_fails(self, mock_load_model_tok):
        mock_load_model_tok.return_value = (self.mock_model_instance, self.mock_tokenizer_instance)
        self.mock_model_instance.generate.side_effect = Exception("CUDA OOM")
        spec = normalise_request(self.raw_request_example, self.symbol_bag_example, self.model_path_example)
        self.assertEqual(spec.task, "Error: T5 inference failed: CUDA OOM")

    @patch('src.spec_normalizer.t5_client._load_t5_model_and_tokenizer')
    def test_normalise_request_output_missing_tags(self, mock_load_model_tok):
        mock_load_model_tok.return_value = (self.mock_model_instance, self.mock_tokenizer_instance)
        self.mock_tokenizer_instance.decode.return_value = yaml.dump(self.expected_spec_dict)
        spec = normalise_request(self.raw_request_example, self.symbol_bag_example, self.model_path_example)
        self.assertTrue(spec.task.startswith("Error: Malformed T5 output (missing SPEC_YAML tags)"))
        self.assertEqual(spec.raw_output, yaml.dump(self.expected_spec_dict))


    @patch('src.spec_normalizer.t5_client._load_t5_model_and_tokenizer')
    def test_normalise_request_invalid_yaml(self, mock_load_model_tok):
        mock_load_model_tok.return_value = (self.mock_model_instance, self.mock_tokenizer_instance)
        invalid_yaml_str = "<SPEC_YAML>\nkey: value\n  bad_indent\n</SPEC_YAML>"
        self.mock_tokenizer_instance.decode.return_value = invalid_yaml_str
        spec = normalise_request(self.raw_request_example, self.symbol_bag_example, self.model_path_example)
        self.assertTrue(spec.task.startswith("Error: Failed to parse YAML from T5"))
        self.assertEqual(spec.raw_yaml, "key: value\n  bad_indent")


    @patch('src.spec_normalizer.t5_client._load_t5_model_and_tokenizer')
    def test_normalise_request_pydantic_validation_error(self, mock_load_model_tok):
        mock_load_model_tok.return_value = (self.mock_model_instance, self.mock_tokenizer_instance)
        invalid_spec_data = {"task_wrong_key": "test"}
        valid_yaml_invalid_data = f"<SPEC_YAML>\n{yaml.dump(invalid_spec_data)}</SPEC_YAML>"
        self.mock_tokenizer_instance.decode.return_value = valid_yaml_invalid_data

        spec = normalise_request(self.raw_request_example, self.symbol_bag_example, self.model_path_example)
        self.assertTrue(spec.task.startswith("Error: Failed to validate Spec from T5"))
        self.assertEqual(spec.raw_yaml, yaml.dump(invalid_spec_data).strip())
        self.assertEqual(spec.parsed_dict, invalid_spec_data)


if __name__ == '__main__':
    unittest.main()
