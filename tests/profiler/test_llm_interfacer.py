import unittest
import random
from typing import Dict, Union, Literal, Optional

from src.profiler.llm_interfacer import get_style_fingerprint_from_llm, SingleSampleFingerprint

class TestLLMInterfacer(unittest.TestCase):

    def test_get_style_fingerprint_from_llm_structure_and_types(self):
        """Test the structure and types of the returned fingerprint dictionary."""
        sample_code = "def foo(): pass"
        fingerprint = get_style_fingerprint_from_llm(sample_code)

        self.assertIsInstance(fingerprint, dict)

        # Check for expected keys based on SingleSampleFingerprint.to_dict()
        expected_keys = [
            "indent", "quotes", "linelen", "camel_pct", "snake_pct",
            "docstyle", "has_type_hints", "spacing_around_operators"
        ]
        for key in expected_keys:
            self.assertIn(key, fingerprint, f"Key '{key}' missing in fingerprint")

        # Check types (some are literals, so check inclusion)
        self.assertIsInstance(fingerprint["indent"], int)
        self.assertIn(fingerprint["quotes"], ["single", "double"])
        self.assertIsInstance(fingerprint["linelen"], int)
        self.assertIsInstance(fingerprint["camel_pct"], float)
        self.assertIsInstance(fingerprint["snake_pct"], float)
        self.assertIn(fingerprint["docstyle"], ["google", "numpy", "epytext", "restructuredtext", "plain", "other"])

        if fingerprint["has_type_hints"] is not None:
            self.assertIsInstance(fingerprint["has_type_hints"], bool)
        if fingerprint["spacing_around_operators"] is not None:
            self.assertIsInstance(fingerprint["spacing_around_operators"], bool)

    def test_get_style_fingerprint_from_llm_variability_with_randomness(self):
        """Test that subsequent calls can produce different results due to mock randomness."""
        sample_code = "class MyClass: def __init__(self): self.value = 10"

        # It's hard to guarantee difference with few calls if random range is small for some fields.
        # Instead, let's check if values are within expected ranges, implying randomness is active.
        # Or, set seed and check for determinism if seed were exposed (it's not directly by the func).
        # The current mock uses random.choice and random.uniform.

        fingerprints = [get_style_fingerprint_from_llm(sample_code) for _ in range(10)]

        # Check if some values vary across a few samples - not a strict test but indicative.
        # Example: check if 'indent' values are not all the same (unless random choice happens to be same)
        indents = {fp["indent"] for fp in fingerprints}
        quotes_types = {fp["quotes"] for fp in fingerprints}

        # This test is probabilistic. If it fails, it might be due to chance.
        # A better test for randomness would be to mock random.choice/uniform,
        # but for a mock function, just ensuring it runs and returns valid structure is often enough.
        self.assertTrue(len(indents) > 1 or len(fingerprints) <= 1 or len({2,4}) == 1,
                        f"Indents were all the same ({indents}), expected some variation over 10 runs or only one choice possible.")
        self.assertTrue(len(quotes_types) > 1 or len(fingerprints) <= 1 or len({"single", "double"}) == 1,
                        f"Quote types were all the same ({quotes_types}), expected some variation or only one choice possible.")

    def test_single_sample_fingerprint_dataclass(self):
        """Test the SingleSampleFingerprint dataclass itself for type validation (conceptual)."""
        # Dataclasses don't enforce types strictly at runtime without Pydantic or similar.
        # This test is more about ensuring the .to_dict() method works and includes all fields.
        fp_instance = SingleSampleFingerprint(
            indent=4, quotes="single", linelen=88, camel_pct=0.1, snake_pct=0.9,
            docstyle="google", has_type_hints=True, spacing_around_operators=True
        )
        fp_dict = fp_instance.to_dict()

        expected_keys = [
            "indent", "quotes", "linelen", "camel_pct", "snake_pct",
            "docstyle", "has_type_hints", "spacing_around_operators"
        ]
        for key in expected_keys:
            self.assertIn(key, fp_dict)

        self.assertEqual(fp_dict["camel_pct"], 0.10) # Check rounding
        self.assertEqual(fp_dict["snake_pct"], 0.90)

if __name__ == '__main__':
    unittest.main()
