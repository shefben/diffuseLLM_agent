from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Placeholder for more specific types if they are defined later
# e.g., for Black configuration, Ruff rules, etc.
BlackConfigType = Dict[str, Any]
RuffConfigType = Dict[str, Any] # Could be more structured, e.g., a list of rules, selections
DocstringPatternsType = List[Dict[str, Any]] # e.g., list of {'style': 'google', 'template': '...', 'prevalence': 0.8}
NamingConventionsType = Dict[str, str] # e.g., {'function': 'snake_case', 'class': 'PascalCase'}


@dataclass
class StyleFingerprint:
    """
    Represents the style fingerprint of a codebase, including formatting,
    linting rules, docstring conventions, and naming patterns.
    """
    black_config: Optional[BlackConfigType] = None
    ruff_config: Optional[RuffConfigType] = None

    # For docstrings, we might store identified patterns or a chosen template
    # Example: list of common docstring structures or a path to a template file
    docstring_styles: DocstringPatternsType = field(default_factory=list)
    docstring_template_file: Optional[str] = None # Path to a generated/chosen template

    # For naming conventions, store dominant patterns per identifier type
    # Example: {'function': 'snake_case', 'class': 'PascalCase', 'variable': 'snake_case'}
    naming_conventions: NamingConventionsType = field(default_factory=dict)

    # Could also include raw extracted data if needed for further processing
    # For example, raw counts of different naming styles before deciding the dominant one.
    raw_docstrings_analysis: Optional[Any] = None
    raw_identifiers_analysis: Optional[Any] = None

    # Potential future fields:
    # detected_python_version: Optional[str] = None
    # project_license: Optional[str] = None
    # average_line_length: Optional[float] = None
    # comment_style: Optional[str] = None # e.g., prevalence of inline vs. block comments

    def is_empty(self) -> bool:
        """Checks if the fingerprint contains any substantial information."""
        return not (
            self.black_config or
            self.ruff_config or
            self.docstring_styles or
            self.docstring_template_file or
            self.naming_conventions
        )

    def __str__(self) -> str:
        return (
            f"StyleFingerprint(\n"
            f"  Black Config: {self.black_config}\n"
            f"  Ruff Config: {self.ruff_config}\n"
            f"  Docstring Styles: {self.docstring_styles}\n"
            f"  Docstring Template File: {self.docstring_template_file}\n"
            f"  Naming Conventions: {self.naming_conventions}\n"
            f")"
        )

if __name__ == '__main__':
    # Example of creating and printing an empty fingerprint
    empty_fp = StyleFingerprint()
    print("--- Empty Style Fingerprint ---")
    print(empty_fp)
    print(f"Is empty: {empty_fp.is_empty()}")

    # Example of a partially filled fingerprint
    partial_fp = StyleFingerprint(
        black_config={"line_length": 88, "target_version": ["py38"]},
        ruff_config={"select": ["E", "F", "W"], "ignore": ["E501"]},
        docstring_styles=[{"style": "google", "prevalence": 0.9}],
        naming_conventions={"function": "snake_case", "class": "PascalCase"}
    )
    print("\n--- Partially Filled Style Fingerprint ---")
    print(partial_fp)
    print(f"Is empty: {partial_fp.is_empty()}")
