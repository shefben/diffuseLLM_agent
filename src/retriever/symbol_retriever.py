# src/retriever/symbol_retriever.py
from typing import List, Optional, TYPE_CHECKING, Set, Any # Added Set and Any
import numpy as np

if TYPE_CHECKING:
    from src.digester.repository_digester import RepositoryDigester # To avoid circular import if used only for type hint

class SymbolRetriever:
    def __init__(self, digester: 'RepositoryDigester'):
        """
        Initializes the SymbolRetriever with a RepositoryDigester instance.

        Args:
            digester: An instance of RepositoryDigester that has already processed
                      a repository and contains the embedding model, FAISS index,
                      and metadata.
        """
        if digester.embedding_model is None:
            raise ValueError("RepositoryDigester's embedding_model is not initialized.")
        if digester.faiss_index is None:
            raise ValueError("RepositoryDigester's faiss_index is not initialized.")

        # Check for numpy availability (np is imported globally in repository_digester)
        # This check ensures that if repository_digester.np is None, we raise an error.
        if not hasattr(digester, 'np') or digester.np is None:
             # Or, more directly, check the global np imported in this file if SymbolRetriever
             # itself uses np directly for its operations beyond what digester provides.
             # The current code uses np directly for query_embedding manipulation.
            if np is None:
                raise ImportError("Numpy is required for SymbolRetriever but not found/imported globally in symbol_retriever.py.")


        self.digester = digester
        self.embedding_model = digester.embedding_model
        self.faiss_index = digester.faiss_index
        self.faiss_id_to_metadata = digester.faiss_id_to_metadata
        self.np_module = np # Store a reference to the numpy module

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

# Example Usage (conceptual, as it needs a populated RepositoryDigester)
if __name__ == '__main__':
    print("SymbolRetriever example usage requires a populated RepositoryDigester.")

    if np: # Check if numpy is available for the mock example
        class MockEmbeddingModel:
            def encode(self, texts, show_progress_bar=False):
                return np.array([np.random.rand(384).astype(np.float32) for _ in texts])
            def get_sentence_embedding_dimension(self): # Needed by digester init
                return 384

        class MockFaissIndex:
            def __init__(self, dim): self.dim = dim; self.ntotal = 0
            def search(self, query_vec, k):
                num_results = min(k, 2) # Simulate finding 2 items
                # Simulate distances: 0.5 (high sim), 1.0 (medium sim)
                dists = np.array([[0.5**0.5 * np.sqrt(2), 1.0**0.5 * np.sqrt(2)] + [2.0]*(k-2)], dtype=np.float32) if k > 0 else np.array([[]], dtype=np.float32)
                # Corrected L2 for cosine: L2_dist = sqrt(2 - 2*cos_sim)
                # So, if target cos_sim = 0.875 -> L2_dist = sqrt(2 - 1.75) = sqrt(0.25) = 0.5
                # If target cos_sim = 0.5   -> L2_dist = sqrt(2 - 1.0) = sqrt(1.0) = 1.0
                dists_corrected = np.array([
                    [np.sqrt(2 - 2 * 0.875), np.sqrt(2 - 2 * 0.5)] + [2.0]*(k-2) # Sim: 0.875, 0.5
                ], dtype=np.float32)

                ids = np.array([[0, 1] + [-1]*(k-2)], dtype=np.int64) if k > 0 else np.array([[]], dtype=np.int64)
                return dists_corrected[:, :num_results], ids[:, :num_results]
            def add(self, vectors): self.ntotal += len(vectors)


        class MockDigester: # Mocking RepositoryDigester for example
            def __init__(self):
                self.embedding_model = MockEmbeddingModel()
                self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
                # Mock faiss module for this example's scope if not available globally
                global faiss
                if faiss is None: # If global faiss is None (not imported)
                    faiss_mock_for_example = MagicMock()
                    faiss_mock_for_example.IndexFlatL2 = MockFaissIndex
                    self.faiss_index = faiss_mock_for_example.IndexFlatL2(self.embedding_dimension)
                elif self.embedding_dimension: # global faiss is available
                     self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension) # Use real if available, or mock if needed
                     # For this example, let's ensure it's our mock for predictable search
                     self.faiss_index = MockFaissIndex(self.embedding_dimension)

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
