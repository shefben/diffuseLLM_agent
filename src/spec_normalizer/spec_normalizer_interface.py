# src/spec_normalizer/spec_normalizer_interface.py
from abc import ABC, abstractmethod
from typing import Optional

class SpecNormalizerModelInterface(ABC):
    @abstractmethod
    def generate_spec_yaml(
        self,
        raw_issue_text: str,
        context_symbols_string: Optional[str] = None,
        mcp_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generates a YAML specification string from raw issue text and context.
        Returns None if generation fails.
        """
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """
        Indicates if the model is loaded and ready to serve requests.
        """
        pass
