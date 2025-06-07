import random
from typing import Dict, Literal, Union, Optional
from dataclasses import dataclass, field # Using dataclass for simplicity

# Define the structure for the style fingerprint tuple from a single sample
@dataclass
class SingleSampleFingerprint:
    indent: int
    quotes: Literal["single", "double"]
    linelen: int # Max line length observed or preferred for this sample
    camel_pct: float # Percentage of camelCase identifiers in the sample
    snake_pct: float # Percentage of snake_case identifiers in the sample
    docstyle: Literal["google", "numpy", "epytext", "restructuredtext", "plain", "other"]
    # Add other potential fields based on common style aspects
    # For example, presence of type hints, spacing around operators, etc.
    # These would be derived by the hypothetical LLM from the sample.
    has_type_hints: Optional[bool] = None
    spacing_around_operators: Optional[bool] = None # True if consistent, False if not, None if N/A

    def to_dict(self) -> Dict[str, Union[int, str, float, bool, None]]:
        return {
            "indent": self.indent,
            "quotes": self.quotes,
            "linelen": self.linelen,
            "camel_pct": round(self.camel_pct, 2), # Ensure consistent formatting
            "snake_pct": round(self.snake_pct, 2),
            "docstyle": self.docstyle,
            "has_type_hints": self.has_type_hints,
            "spacing_around_operators": self.spacing_around_operators,
        }

# Placeholder/mock function for LLM interaction
def get_style_fingerprint_from_llm(sample_code_snippet: str) -> Dict[str, Union[int, str, float, bool, None]]:
    """
    Mock function to simulate getting a style fingerprint for a single code
    sample from an LLM-Coder core.

    In a real implementation, this would involve:
    1. Formatting the request to the LLM.
    2. Sending the sample_code_snippet to the LLM.
    3. Receiving and parsing the LLM's response (likely JSON).
    4. Converting the response into the SingleSampleFingerprint structure.

    Args:
        sample_code_snippet: A string containing the code sample.

    Returns:
        A dictionary representing the style fingerprint for the sample.
    """
    # Simulate LLM processing by generating random (but plausible) values
    # This is where the actual LLM call would be.
    # For now, we'll just generate some mock data.

    # To make it slightly more interesting, let's vary based on snippet length a bit
    # This is purely for making the mock more "dynamic" but not truly intelligent.
    is_long_snippet = len(sample_code_snippet) > 200

    mock_fingerprint = SingleSampleFingerprint(
        indent=random.choice([2, 4]),
        quotes=random.choice(["single", "double"]),
        linelen=random.choice([79, 88, 99, 120]),
        # Simulate some variation in naming conventions
        camel_pct=random.uniform(0.0, 0.4) if is_long_snippet else random.uniform(0.0, 0.2),
        snake_pct=random.uniform(0.6, 1.0) if is_long_snippet else random.uniform(0.8, 1.0),
        docstyle=random.choice(["google", "numpy", "plain", "other"]),
        has_type_hints=random.choice([True, False, None]),
        spacing_around_operators=random.choice([True, False, None]) if is_long_snippet else True
    )

    # Ensure percentages don't exceed 1.0 if summed (though they are independent here)
    if mock_fingerprint.camel_pct + mock_fingerprint.snake_pct > 1.0:
        if mock_fingerprint.camel_pct > mock_fingerprint.snake_pct:
            mock_fingerprint.camel_pct = 1.0 - mock_fingerprint.snake_pct
        else:
            mock_fingerprint.snake_pct = 1.0 - mock_fingerprint.camel_pct

    # Ensure snake_pct and camel_pct are non-negative after adjustment
    mock_fingerprint.camel_pct = max(0.0, mock_fingerprint.camel_pct)
    mock_fingerprint.snake_pct = max(0.0, mock_fingerprint.snake_pct)


    return mock_fingerprint.to_dict()

if __name__ == '__main__':
    example_sample_code_short = "def foo(): return 'bar'"
    example_sample_code_long = """
class MyClass:
    def __init__(self, value: int = 0):
        self.value: int = value

    def get_value(self) -> int:
        # A slightly longer method
        # with some comments
        return self.value + 10 * 2
"""

    print("--- Mock LLM Fingerprint for SHORT sample ---")
    fingerprint1 = get_style_fingerprint_from_llm(example_sample_code_short)
    for k, v in fingerprint1.items():
        print(f"  {k}: {v}")

    print("\n--- Mock LLM Fingerprint for LONG sample ---")
    fingerprint2 = get_style_fingerprint_from_llm(example_sample_code_long)
    for k, v in fingerprint2.items():
        print(f"  {k}: {v}")

    # Test multiple calls to see variation
    print("\n--- Multiple calls for short sample (expect variation due to random) ---")
    for i in range(3):
        fp = get_style_fingerprint_from_llm(example_sample_code_short)
        print(f"  Run {i+1}: indent={fp['indent']}, quotes='{fp['quotes']}', linelen={fp['linelen']}")
