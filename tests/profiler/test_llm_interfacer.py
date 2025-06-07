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

# Add new imports if needed for the new test class
from src.profiler.llm_interfacer import collect_deterministic_stats, _get_modal_indent, _get_line_length_percentile

class TestDeterministicStatsCollection(unittest.TestCase):

    def test_get_modal_indent(self):
        code_no_indent = "print('hello')"
        self.assertEqual(_get_modal_indent(code_no_indent), 4) # Default

        code_two_spaces = "def foo():\n  pass\n  x = 1"
        self.assertEqual(_get_modal_indent(code_two_spaces), 2)

        code_four_spaces = "def foo():\n    pass\n    x = 1"
        self.assertEqual(_get_modal_indent(code_four_spaces), 4)

        code_mixed_indent = "def foo():\n  two_a\n  two_b\n    four_a\n  two_c"
        self.assertEqual(_get_modal_indent(code_mixed_indent), 2) # 2 is more common

        code_only_blank_lines = "\n\n"
        self.assertEqual(_get_modal_indent(code_only_blank_lines), 4) # Default

        code_no_leading_space_lines_but_content = "a=1\nb=2"
        self.assertEqual(_get_modal_indent(code_no_leading_space_lines_but_content), 4) # Default as no *leading spaces*

    def test_get_line_length_percentile(self):
        code_short_lines = "a\nbb\nccc" # lengths 1, 2, 3
        # Sorted: [1, 2, 3]. Index for 0.95 percentile: int(3*0.95)-1 = int(2.85)-1 = 2-1 = 1. Value is 2.
        self.assertEqual(_get_line_length_percentile(code_short_lines, 0.95), 2)
        # For 0.5 percentile: int(3*0.5)-1 = int(1.5)-1 = 1-1 = 0. Value is 1.
        self.assertEqual(_get_line_length_percentile(code_short_lines, 0.50), 1)


        code_10_lines_len_10 = "\n".join(["a" * 10] * 10)
        self.assertEqual(_get_line_length_percentile(code_10_lines_len_10, 0.95), 10)

        code_empty = ""
        self.assertEqual(_get_line_length_percentile(code_empty), 88) # Default

    def test_collect_deterministic_stats_basic(self):
        code = """
def my_func_snake(param_one: int): # snake_case func, param
    # This is a docstring.
    # Args:
    #   param_one: an int.
    # Returns:
    #   None
    s = 'single_quoted_string'
    d = "double_quoted_string"
    f = f"f_string_with_{param_one}"
    CONSTANT_VAL = 100 # screaming
    anotherVarCamel = True # camelCase
    if param_one > 0:
        print(s) # 4 space indent
    else:
        print(d) # 4 space indent
"""
        # Expected modal indent: 4 (from print statements)
        # Quotes: single=1, double=1, f_string=1
        # Line lengths: Let's assume the longest line is reasonably short for this example.
        #   The f-string line is likely longest.
        # Identifiers: my_func_snake, param_one, s, d, f, CONSTANT_VAL, anotherVarCamel, print (x2)
        #   Countable (len>1 or _): my_func_snake, param_one, CONSTANT_VAL, anotherVarCamel (4)
        #   Snake: my_func_snake, param_one (2/4 = 0.50)
        #   Camel: anotherVarCamel (1/4 = 0.25)
        #   Screaming: CONSTANT_VAL (1/4 = 0.25)
        # Doc markers: Args:, Returns:

        stats_block = collect_deterministic_stats(code)

        self.assertIn("STATS_START", stats_block)
        self.assertIn("STATS_END", stats_block)

        expected_lines = {
            "indent_modal: 4",
            "quotes_single: 1",
            "quotes_double: 1",
            "f_strings: 1",
            # "line_len_95p: ?", # Hard to predict exactly without running it on the exact string above
            "snake_pct: 0.50", # my_func_snake, param_one (s,d,f are single letter, print is builtin-like)
                               # Identifiers: my_func_snake, param_one, CONSTANT_VAL, anotherVarCamel
                               # snake: my_func_snake, param_one (2)
                               # camel: anotherVarCamel (1)
                               # screaming: CONSTANT_VAL (1)
                               # total = 4. snake=0.50, camel=0.25, screaming=0.25
            "camel_pct: 0.25",
            "screaming_pct: 0.25",
            "doc_tokens: Args:, Returns:",
        }

        for line in expected_lines:
            self.assertIn(line, stats_block)

        # Check line_len_95p separately
        # Actual code string for stats:
        # line1: "def my_func_snake(param_one: int): # snake_case func, param" (len 60)
        # line2: "    # This is a docstring." (len 28)
        # line3: "    # Args:" (len 14)
        # line4: "    #   param_one: an int." (len 27)
        # line5: "    # Returns:" (len 16)
        # line6: "    #   None" (len 14)
        # line7: "    s = 'single_quoted_string'" (len 32)
        # line8: "    d = "double_quoted_string"" (len 33)
        # line9: "    f = f"f_string_with_{param_one}"" (len 38)
        # line10: "    CONSTANT_VAL = 100 # screaming" (len 37)
        # line11: "    anotherVarCamel = True # camelCase" (len 40)
        # line12: "    if param_one > 0:" (len 23)
        # line13: "        print(s) # 4 space indent" (len 34)
        # line14: "    else:" (len 10)
        # line15: "        print(d) # 4 space indent" (len 34)
        # Lengths: [60, 28, 14, 27, 16, 14, 32, 33, 38, 37, 40, 23, 34, 10, 34] (15 lines)
        # Sorted: [10, 14, 14, 16, 23, 27, 28, 32, 33, 34, 34, 37, 38, 40, 60] (15 lines)
        # 0.95 percentile index: int(15 * 0.95) - 1 = int(14.25) - 1 = 14 - 1 = 13.
        # Value at index 13 is 40.
        self.assertIn("line_len_95p: 40", stats_block)


    def test_collect_deterministic_stats_more_quotes_and_indents(self):
        code = """
# No initial docstring for module
class Test: # 0 indent
  def method(self): # 2 indent
      '''Single multi'''
      x = "double" # 6 indent
      y = "double" # 6 indent
      z = f'''{x}''' # 6 indent, f-string, single quote underlying
      a = 'single'   # 6 indent
      if True: # 6 indent
          pass   # 10 indent
"""
        # Expected: indent_modal: 6
        # Quotes: single=2 (''' and 'single'), double=2 ("double" x2), f_strings=1
        # Doc markers: none
        stats_block = collect_deterministic_stats(code)
        self.assertIn("indent_modal: 6", stats_block)
        self.assertIn("quotes_single: 2", stats_block)
        self.assertIn("quotes_double: 2", stats_block)
        self.assertIn("f_strings: 1", stats_block)
        self.assertIn("doc_tokens: none", stats_block)

    def test_collect_deterministic_stats_identifier_edge_cases(self):
        code = "data_point = 1; _private_var = 2; __dunder__ = 3; X = 4; YZ = 5;"
        # Identifiers: data_point, _private_var, __dunder__, X, YZ
        # Countable for case analysis (len>1 or _): data_point, _private_var, __dunder__, YZ (X is skipped)
        #   snake_case: data_point, _private_var, __dunder__ (3) -> 3/4 = 0.75
        #   camelCase: 0
        #   screaming: YZ (1) -> 1/4 = 0.25
        stats_block = collect_deterministic_stats(code)
        self.assertIn("snake_pct: 0.75", stats_block)
        self.assertIn("camel_pct: 0.00", stats_block)
        self.assertIn("screaming_pct: 0.25", stats_block) # YZ is all upper

    def test_collect_deterministic_stats_empty_snippet(self):
        stats_block = collect_deterministic_stats("")
        self.assertIn("indent_modal: 4", stats_block) # Default
        self.assertIn("quotes_single: 0", stats_block)
        self.assertIn("quotes_double: 0", stats_block)
        self.assertIn("f_strings: 0", stats_block)
        self.assertIn("line_len_95p: 88", stats_block) # Default
        self.assertIn("snake_pct: 0.00", stats_block)
        self.assertIn("camel_pct: 0.00", stats_block)
        self.assertIn("screaming_pct: 0.00", stats_block)
        self.assertIn("doc_tokens: none", stats_block)

    def test_collect_deterministic_stats_syntax_error_robustness(self):
        code_with_syntax_error = "def func( :"
        # Should not raise, should return best-effort stats
        stats_block = collect_deterministic_stats(code_with_syntax_error)
        self.assertIn("STATS_START", stats_block)
        # Tokenizer might get some stats, AST parsing for docstrings will fail.
        self.assertIn("doc_tokens: none", stats_block)
        # Other stats depend on how much tokenizer could process.
        # Example: indent_modal might be default or based on partial lines.
        self.assertIn("indent_modal:", stats_block)


if __name__ == '__main__':
    unittest.main()
