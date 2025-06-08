import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np # For creating dummy embeddings and for SymbolRetriever's np check

# Adjust import path as necessary
from src.retriever.symbol_retriever import SymbolRetriever
from src.digester.repository_digester import RepositoryDigester # For type hint of mock_digester
from typing import Any # For NumpyNdarray fallback if numpy is not available in source

# It's useful to have a reference to the numpy module as it's used in SymbolRetriever
# This ensures that if SymbolRetriever does 'import numpy as np', our patch targets it correctly.
# The actual module 'src.retriever.symbol_retriever' is patched in setUp for 'np'.

class TestSymbolRetriever(unittest.TestCase):

    def setUp(self):
        self.mock_digester = MagicMock(spec=RepositoryDigester)

        self.mock_embedding_model = MagicMock()
        # SentenceTransformer.encode typically returns a 2D ndarray for a list of sentences
        self.dummy_query_embedding_2d = np.array([[0.1, 0.2, 0.3] * 128], dtype=np.float32) # Dim 384, shape (1, 384)
        self.mock_embedding_model.encode.return_value = self.dummy_query_embedding_2d

        self.mock_faiss_index = MagicMock()
        self.mock_faiss_index.ntotal = 0
        self.mock_faiss_index.search.return_value = (np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64))

        self.mock_digester.embedding_model = self.mock_embedding_model
        self.mock_digester.faiss_index = self.mock_faiss_index
        self.mock_digester.faiss_id_to_metadata = []

        # Patch 'np' in the context of the 'src.retriever.symbol_retriever' module
        self.patch_numpy_in_symbol_retriever = patch('src.retriever.symbol_retriever.np', np)
        self.mock_np_in_retriever = self.patch_numpy_in_symbol_retriever.start()
        self.mock_digester.np_module = self.mock_np_in_retriever # Ensure digester has the same np instance


    def tearDown(self):
        self.patch_numpy_in_symbol_retriever.stop()


    def test_init_success(self):
        retriever = SymbolRetriever(self.mock_digester)
        self.assertIsNotNone(retriever)
        self.assertEqual(retriever.np_module, np) # Check if it stored the numpy reference

    def test_init_embedding_model_missing(self):
        self.mock_digester.embedding_model = None
        with self.assertRaisesRegex(ValueError, "embedding_model is not initialized"):
            SymbolRetriever(self.mock_digester)

    def test_init_faiss_index_missing(self):
        self.mock_digester.faiss_index = None
        with self.assertRaisesRegex(ValueError, "faiss_index is not initialized"):
            SymbolRetriever(self.mock_digester)

    @patch('src.retriever.symbol_retriever.np', None)
    def test_init_numpy_missing(self, mock_np_none_in_module):
        # This test specifically tests the ImportError if 'np' is None globally within symbol_retriever.py
        # The digester's np_module attribute isn't directly checked by SymbolRetriever's __init__ for np presence,
        # but SymbolRetriever *uses* np directly.
        with self.assertRaisesRegex(ImportError, "Numpy is required for SymbolRetriever but not found/imported globally in symbol_retriever.py."):
            SymbolRetriever(self.mock_digester)


    def test_retrieve_symbol_bag_success(self):
        retriever = SymbolRetriever(self.mock_digester)
        self.mock_faiss_index.ntotal = 3

        # faiss.IndexFlatL2.search returns actual L2 distances.
        # Cosine Similarity = 1 - (L2_distance^2 / 2)
        # Target Sims: 0.9 (dist=0.4472), 0.5 (dist=1.0), 0.1 (dist=1.3416)
        l2_dist_sim_0_9 = np.sqrt(2 - 2 * 0.9)
        l2_dist_sim_0_5 = np.sqrt(2 - 2 * 0.5)
        l2_dist_sim_0_1 = np.sqrt(2 - 2 * 0.1)

        self.mock_faiss_index.search.return_value = (
            np.array([[l2_dist_sim_0_9, l2_dist_sim_0_5, l2_dist_sim_0_1]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64)
        )

        self.mock_digester.faiss_id_to_metadata = [
            {"fqn": "module.func_a", "item_type": "function_code"},
            {"fqn": "module.ClassB.method_c", "item_type": "method_code"},
            {"fqn": "module.another_func", "item_type": "function_code"}
        ]

        result = retriever.retrieve_symbol_bag("test request", top_k=3, similarity_threshold=0.25)
        self.assertEqual(result, "module.func_a|module.ClassB.method_c")
        self.mock_embedding_model.encode.assert_called_once_with(["test request"], show_progress_bar=False)

        args_search, kwargs_search = self.mock_faiss_index.search.call_args
        query_vector_passed = args_search[0]
        top_k_passed = args_search[1]

        self.assertTrue(isinstance(query_vector_passed, np.ndarray))
        self.assertEqual(query_vector_passed.shape, (1,384))
        self.assertEqual(query_vector_passed.dtype, np.float32)
        self.assertEqual(top_k_passed, 3)


    def test_retrieve_symbol_bag_uniqueness_and_order(self):
        retriever = SymbolRetriever(self.mock_digester)
        self.mock_faiss_index.ntotal = 2 # Only 2 unique items in metadata
        # Simulate high similarity for all, but FAISS returns ID 0 twice
        dist_high_sim = np.sqrt(2 - 2 * 0.9)
        self.mock_faiss_index.search.return_value = (
            np.array([[dist_high_sim, dist_high_sim - 0.01, dist_high_sim + 0.01]], dtype=np.float32),
            np.array([[0, 1, 0]], dtype=np.int64) # ID 0, then 1, then 0 again (but 1 is more similar than the second 0)
        )
        self.mock_digester.faiss_id_to_metadata = [
            {"fqn": "module.func_a"}, {"fqn": "module.func_b"}
        ]
        result = retriever.retrieve_symbol_bag("test", top_k=3, similarity_threshold=0.1)
        # Expected: func_b (from ID 1, highest sim) then func_a (from ID 0)
        # The sorting is by similarity, then FAISS ID order if sims are equal.
        # Here, dist for ID 1 is smallest (highest sim), then ID 0, then ID 0 again.
        # So order should be module.func_b, then module.func_a
        self.assertEqual(result, "module.func_b|module.func_a")


    def test_retrieve_symbol_bag_empty_request_or_index(self):
        retriever = SymbolRetriever(self.mock_digester)
        self.assertEqual(retriever.retrieve_symbol_bag("", top_k=5), "")

        self.mock_faiss_index.ntotal = 0
        self.assertEqual(retriever.retrieve_symbol_bag("test", top_k=5), "")


    def test_retrieve_symbol_bag_no_results_from_faiss(self):
        retriever = SymbolRetriever(self.mock_digester)
        self.mock_faiss_index.ntotal = 10
        self.mock_faiss_index.search.return_value = (np.array([[]],dtype=np.float32), np.array([[]],dtype=np.int64))
        self.assertEqual(retriever.retrieve_symbol_bag("test"), "")

    def test_retrieve_symbol_bag_faiss_id_out_of_bounds(self):
        retriever = SymbolRetriever(self.mock_digester)
        self.mock_faiss_index.ntotal = 1
        dist_high_sim = np.sqrt(2 - 2 * 0.9)
        self.mock_faiss_index.search.return_value = (
            np.array([[dist_high_sim]], dtype=np.float32),
            np.array([[100]], dtype=np.int64) # ID 100 is out of bounds
        )
        self.mock_digester.faiss_id_to_metadata = [{"fqn": "module.func_a"}]
        with patch('builtins.print') as mock_print:
            result = retriever.retrieve_symbol_bag("test")
            self.assertEqual(result, "")
            self.assertTrue(any("FAISS ID 100 out of bounds" in str(c.args) for c in mock_print.call_args_list))


    def test_retrieve_symbol_bag_no_symbols_meet_threshold(self):
        retriever = SymbolRetriever(self.mock_digester)
        self.mock_faiss_index.ntotal = 1
        dist_low_sim = np.sqrt(2 - 2 * 0.1) # Sim = 0.1
        self.mock_faiss_index.search.return_value = (
            np.array([[dist_low_sim]]),
            np.array([[0]])
        )
        self.mock_digester.faiss_id_to_metadata = [{"fqn": "module.func_a"}]
        result = retriever.retrieve_symbol_bag("test", similarity_threshold=0.25)
        self.assertEqual(result, "")

    def test_retrieve_symbol_bag_faiss_search_exception(self):
        retriever = SymbolRetriever(self.mock_digester)
        self.mock_faiss_index.search.side_effect = Exception("FAISS Search Crashed")
        with patch('builtins.print') as mock_print:
            result = retriever.retrieve_symbol_bag("test")
            self.assertEqual(result, "")
            self.assertTrue(any("Error during FAISS search: FAISS Search Crashed" in str(c.args) for c in mock_print.call_args_list))

    def test_retrieve_symbol_bag_embedding_encode_bad_format(self):
        retriever = SymbolRetriever(self.mock_digester)
        self.mock_embedding_model.encode.return_value = "not_an_array"
        with patch('builtins.print') as mock_print:
            result = retriever.retrieve_symbol_bag("test")
            self.assertEqual(result, "")
            self.assertTrue(any("Unexpected query embedding format" in str(c.args) for c in mock_print.call_args_list))


if __name__ == '__main__':
    unittest.main()
