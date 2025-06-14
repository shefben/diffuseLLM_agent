# src/spec_normalizer/diffusion_spec_normalizer.py
from typing import Dict, Any, Optional
from pathlib import Path

from .spec_normalizer_interface import SpecNormalizerModelInterface
from .t5_client import T5Client # For fallback
from src.utils.config_loader import DEFAULT_APP_CONFIG # For default T5 path if needed by internal T5Client

class DiffusionSpecNormalizer(SpecNormalizerModelInterface):
    def __init__(self, app_config: Dict[str, Any]):
        self.app_config = app_config
        self.verbose = self.app_config.get("general", {}).get("verbose", False)

        self.diffusion_model_path = self.app_config.get("models", {}).get("spec_diffusion_model_path") # New config key
        self._is_diffusion_model_configured = False
        self.fallback_t5_client: Optional[T5Client] = None

        if self.diffusion_model_path:
            # Placeholder: In a real scenario, load/check the diffusion model
            # For now, just checking if path is provided is enough to simulate "configured"
            if Path(self.diffusion_model_path).exists(): # Basic check
                self._is_diffusion_model_configured = True
                if self.verbose:
                    print(f"DiffusionSpecNormalizer: Configured with diffusion model path: {self.diffusion_model_path}")
            else:
                if self.verbose:
                    print(f"DiffusionSpecNormalizer Warning: Diffusion model path {self.diffusion_model_path} provided but not found. Will use T5 fallback.")

        if not self._is_diffusion_model_configured:
            if self.verbose:
                print("DiffusionSpecNormalizer: Diffusion model not configured or path invalid. Initializing T5Client as fallback.")
            # Pass the main app_config to T5Client so it can find its own model paths
            self.fallback_t5_client = T5Client(app_config=self.app_config)

    @property
    def is_ready(self) -> bool:
        if self._is_diffusion_model_configured:
            # Placeholder: Real check for diffusion model readiness
            return True
        if self.fallback_t5_client:
            return self.fallback_t5_client.is_ready
        return False

    def generate_spec_yaml(self, raw_issue_text: str, context_symbols_string: Optional[str] = None) -> Optional[str]:
        if self._is_diffusion_model_configured:
            if self.verbose:
                print(f"DiffusionSpecNormalizer: Would use diffusion model for: {raw_issue_text[:50]}...")
            # Placeholder: Actual call to diffusion model
            context_info_for_mock = "No context symbols provided."
            if context_symbols_string:
                context_info_for_mock = context_symbols_string[:200] + "..." if len(context_symbols_string) > 200 else context_symbols_string

            mock_yaml = f'''
issue_description: "Processed by (placeholder) DiffusionSpecNormalizer: {raw_issue_text[:100]}"
target_files:
  - "src/mock/diffusion_target.py"
operations:
  - name: "diffusion_generated_op"
    description: "Operation suggested by diffusion spec normalizer."
context_used: |
  {context_info_for_mock}
acceptance_tests:
  - "Ensure diffusion normalization worked."
'''
            return mock_yaml.strip()
        elif self.fallback_t5_client and self.fallback_t5_client.is_ready:
            if self.verbose:
                print(f"DiffusionSpecNormalizer: Using T5Client fallback for: {raw_issue_text[:50]}...")
            return self.fallback_t5_client.generate_spec_yaml(raw_issue_text, context_symbols_string)
        else:
            if self.verbose:
                print("DiffusionSpecNormalizer Error: Neither diffusion model nor T5 fallback is ready.")
            return None
