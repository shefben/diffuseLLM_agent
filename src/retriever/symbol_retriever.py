# src/retriever/symbol_retriever.py
from typing import List, Optional, TYPE_CHECKING, Set, Any # Added Set and Any
import numpy as np

if TYPE_CHECKING:
    from src.digester.repository_digester import RepositoryDigester # To avoid circular import if used only for type hint

class SymbolRetriever:
    def __init__(self, digester: 'RepositoryDigester', verbose: bool = False):
        """
        Initializes the SymbolRetriever with a RepositoryDigester instance.

        Args:
            digester: An instance of RepositoryDigester that has already processed
                      a repository and contains the embedding model, FAISS index,
                      and metadata.
            verbose: If True, enables detailed logging.
        """
        if digester.embedding_model is None: # Check this first as it's a primary dependency for retrieval
            print("SymbolRetriever Warning: RepositoryDigester's embedding_model is not initialized. Retrieval will be limited.")
            # Depending on strictness, could raise ValueError here. For now, allow init but retrieval will be impaired.

        # FAISS index is also crucial for embedding-based retrieval
        if digester.faiss_index is None:
            print("SymbolRetriever Warning: RepositoryDigester's faiss_index is not initialized. Embedding-based retrieval will be impaired.")

        if np is None: # Global numpy check for SymbolRetriever's own operations
            # This might be redundant if all np ops are on digester.np_module, but good as a safeguard
            print("SymbolRetriever Warning: Numpy not found globally. Some operations might fail if not using digester.np_module.")

        self.digester = digester
        self.embedding_model = digester.embedding_model # Might be None
        self.faiss_index = digester.faiss_index       # Might be None
        self.faiss_id_to_metadata = digester.faiss_id_to_metadata # Might be empty
        self.np_module = np # Uses the global np imported in this file for its own operations
        self.verbose = verbose
        if self.verbose:
            print(f"SymbolRetriever initialized. Embedding model ready: {bool(self.embedding_model)}, FAISS index ready: {bool(self.faiss_index)}")


    def _l2_normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalizes a vector."""
        if self.np_module is None: # Should have been caught in __init__
            raise RuntimeError("Numpy is not available for L2 normalization.")
        norm = self.np_module.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def retrieve_symbol_bag(
        self,
        raw_request: str,
        top_k: int = 50,
        similarity_threshold: float = 0.25
    ) -> str:
        """
        Retrieves a 'bag-of-symbols' (FQN strings) relevant to the raw_request
        by performing a FAISS search on pre-computed embeddings.

        Args:
            raw_request: The raw text request from the user.
            top_k: The number of nearest neighbors to retrieve from FAISS.
            similarity_threshold: Minimum cosine similarity for a symbol to be included.

        Returns:
            A string containing unique FQNs joined by '|', or an empty string.
        """
        if not raw_request or not self.embedding_model or not self.faiss_index or self.faiss_index.ntotal == 0 or self.np_module is None:
            if self.np_module is None: print("SymbolRetriever: Numpy not available for FAISS search.")
            return ""

        print(f"SymbolRetriever: Encoding raw request: '{raw_request[:100]}...'")
        query_embedding_list = self.embedding_model.encode([raw_request], show_progress_bar=False)

        query_embedding: np.ndarray
        if not isinstance(query_embedding_list, self.np_module.ndarray) or query_embedding_list.ndim != 2 or query_embedding_list.shape[0] != 1:
            if isinstance(query_embedding_list, list) and len(query_embedding_list) == 1 and isinstance(query_embedding_list[0], self.np_module.ndarray):
                query_embedding = query_embedding_list[0].astype(self.np_module.float32).reshape(1, -1)
            elif isinstance(query_embedding_list, self.np_module.ndarray) and query_embedding_list.ndim == 1:
                query_embedding = query_embedding_list.astype(self.np_module.float32).reshape(1, -1)
            else:
                print("Warning: Unexpected query embedding format. Cannot perform FAISS search.")
                return ""
        else:
            query_embedding = query_embedding_list.astype(self.np_module.float32)

        # Assuming SentenceTransformer models output L2-normalized embeddings.
        # If not, uncomment:
        # query_embedding = self._l2_normalize_vector(query_embedding.flatten()).reshape(1, -1)

        print(f"SymbolRetriever: Querying FAISS with top_k={top_k}...")
        try:
            distances, faiss_ids = self.faiss_index.search(query_embedding, top_k)
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            return ""

        relevant_fqns_ordered: List[str] = []
        processed_fqns: Set[str] = set()

        if faiss_ids.size > 0:
            for i in range(faiss_ids.shape[1]):
                faiss_id = faiss_ids[0, i]
                if faiss_id == -1:  continue

                l2_dist = distances[0, i]
                cosine_similarity = 1 - (l2_dist**2 / 2)

                if cosine_similarity >= similarity_threshold:
                    if 0 <= faiss_id < len(self.faiss_id_to_metadata):
                        metadata = self.faiss_id_to_metadata[int(faiss_id)]
                        fqn = metadata.get("fqn")
                        if fqn and fqn not in processed_fqns:
                            relevant_fqns_ordered.append(fqn)
                            processed_fqns.add(fqn)
                    else:
                        print(f"Warning: FAISS ID {faiss_id} out of bounds for metadata list (len {len(self.faiss_id_to_metadata)}).")

        if not relevant_fqns_ordered:
            print("SymbolRetriever: No symbols found meeting similarity threshold.")
            return ""

        symbol_bag_string = "|".join(relevant_fqns_ordered)
        print(f"SymbolRetriever: Constructed symbol_bag: '{symbol_bag_string[:200]}...'")
        return symbol_bag_string

    def get_context_symbols_for_spec_fusion(self, max_symbols: Optional[int] = 500) -> List[str]:
        """
        Retrieves a list of FQNs for context, prioritizing relevant item types.
        This implementation primarily uses faiss_id_to_metadata.
        """
        if self.verbose: print("SymbolRetriever: Retrieving context symbols for spec fusion...")

        if not hasattr(self.digester, 'faiss_id_to_metadata') or not self.digester.faiss_id_to_metadata:
            if self.verbose: print("  SymbolRetriever Warning: faiss_id_to_metadata is empty or not available in digester. No symbols to retrieve.")
            return []

        collected_fqns: Set[str] = set()
        # Prioritize these item types for spec context
        relevant_item_types = {
            "function_code",
            "method_code",
            "class_code",
            "docstring_for_module" # Module FQNs can be useful context
        }

        for item in self.digester.faiss_id_to_metadata:
            try:
                fqn = item.get('fqn')
                item_type = item.get('item_type')
                if isinstance(fqn, str) and fqn and isinstance(item_type, str) and item_type in relevant_item_types:
                    collected_fqns.add(fqn)
            except Exception as e: # pylint: disable=broad-except
                if self.verbose: print(f"  SymbolRetriever Warning: Error processing metadata item {item}: {e}")

        if not collected_fqns:
            if self.verbose: print("  SymbolRetriever: No FQNs collected after filtering by relevant types.")
            return []

        sorted_fqns = sorted(list(collected_fqns))

        if max_symbols is not None and len(sorted_fqns) > max_symbols:
            if self.verbose: print(f"  SymbolRetriever: Truncating symbol list from {len(sorted_fqns)} to {max_symbols}.")
            return sorted_fqns[:max_symbols]

        if self.verbose: print(f"  SymbolRetriever: Returning {len(sorted_fqns)} context symbols.")
        return sorted_fqns

# Example Usage (conceptual, as it needs a populated RepositoryDigester)
if __name__ == '__main__':
    from unittest.mock import MagicMock # Import MagicMock for the faiss part if faiss is None
    print("SymbolRetriever example usage requires a populated RepositoryDigester.")

    if np: # Check if numpy is available for the mock example
        class MockEmbeddingModel:
            def encode(self, texts, show_progress_bar=False):
                return np.array([np.random.rand(384).astype(np.float32) for _ in texts])
            def get_sentence_embedding_dimension(self):
                return 384

        class MockFaissIndex:
            def __init__(self, dim): self.dim = dim; self.ntotal = 0
            def search(self, query_vec, k):
                num_results = min(k, 2)
                dists_corrected = np.array([
                    [np.sqrt(2 - 2 * 0.875), np.sqrt(2 - 2 * 0.5)] + [2.0]*(k-num_results)
                ], dtype=np.float32)
                ids = np.array([[0, 1] + ([-1]*(k-num_results) if k > num_results else [])], dtype=np.int64)
                return dists_corrected[:, :num_results], ids[:, :num_results]
            def add(self, vectors): self.ntotal += len(vectors)


        class MockDigester:
            def __init__(self, verbose_retriever=False):
                self.embedding_model = MockEmbeddingModel()
                self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()

                global faiss
                if faiss is None:
                    faiss_mock_for_example = MagicMock()
                    faiss_mock_for_example.IndexFlatL2 = MockFaissIndex
                    self.faiss_index = faiss_mock_for_example.IndexFlatL2(self.embedding_dimension)
                elif self.embedding_dimension:
                     self.faiss_index = MockFaissIndex(self.embedding_dimension) # Ensure mock for predictability
                else:
                    self.faiss_index = None

                self.faiss_id_to_metadata = [
                    {"fqn": "module.func_a", "item_type": "function_code"},
                    {"fqn": "module.ClassB.method_c", "item_type": "method_code"},
                ]
                self.np_module = np # Provide numpy reference

        mock_digester_instance = MockDigester()

        # Ensure the mock digester has a FAISS index for the test to proceed
        if mock_digester_instance.faiss_index is None and mock_digester_instance.embedding_dimension:
            print("Manually setting MockFaissIndex for example as global faiss might be real.")
            mock_digester_instance.faiss_index = MockFaissIndex(mock_digester_instance.embedding_dimension)


        if mock_digester_instance.embedding_model and mock_digester_instance.faiss_index:
            retriever = SymbolRetriever(mock_digester_instance) # type: ignore

            raw_request_example = "How to use function A?"
            # With L2 distances of sqrt(0.25)=0.5 and sqrt(1.0)=1.0
            # Cosine sims: 1 - (0.25/2) = 0.875,  1 - (1.0/2) = 0.5

            print(f"\nTesting with threshold 0.7 (should get func_a):")
            symbol_bag = retriever.retrieve_symbol_bag(raw_request_example, top_k=5, similarity_threshold=0.7)
            print(f"Example symbol_bag (thresh 0.7): '{symbol_bag}'") # Expected: "module.func_a"
            assert symbol_bag == "module.func_a"

            print(f"\nTesting with threshold 0.4 (should get both):")
            symbol_bag_both = retriever.retrieve_symbol_bag(raw_request_example, top_k=5, similarity_threshold=0.4)
            print(f"Example symbol_bag (thresh 0.4): '{symbol_bag_both}'") # Expected: "module.func_a|module.ClassB.method_c"
            assert symbol_bag_both == "module.func_a|module.ClassB.method_c"

            print(f"\nTesting with threshold 0.9 (should get none):")
            symbol_bag_none = retriever.retrieve_symbol_bag(raw_request_example, top_k=5, similarity_threshold=0.9)
            print(f"Example symbol_bag (thresh 0.9): '{symbol_bag_none}'") # Expected: ""
            assert symbol_bag_none == ""
        else:
            print("Mock digester components not fully initialized for example.")
    else:
        print("Numpy not available, cannot run SymbolRetriever example.")
