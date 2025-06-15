# src/spec_normalizer/diffusion_spec_normalizer.py
from typing import Dict, Any, Optional
from pathlib import Path

from .spec_normalizer_interface import SpecNormalizerModelInterface
from .t5_client import T5Client  # Fallback implementation

try:
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
except ImportError:  # pragma: no cover - transformers is optional at runtime
    torch = None  # type: ignore
    T5ForConditionalGeneration = None  # type: ignore
    T5TokenizerFast = None  # type: ignore
    print(
        "Warning: PyTorch or Hugging Face Transformers not available. DiffusionSpecNormalizer will use the T5 fallback only."
    )


class DiffusionSpecNormalizer(SpecNormalizerModelInterface):
    def __init__(self, app_config: Dict[str, Any]):
        self.app_config = app_config
        self.verbose = self.app_config.get("general", {}).get("verbose", False)

        self.diffusion_model_path = self.app_config.get("models", {}).get(
            "spec_diffusion_model_path"
        )
        self._is_diffusion_model_configured = False
        self.fallback_t5_client: Optional[T5Client] = None
        self.model: Optional[T5ForConditionalGeneration] = None
        self.tokenizer: Optional[T5TokenizerFast] = None
        self.device: Optional[str] = None
        self._is_ready: bool = False

        if (
            self.diffusion_model_path
            and T5ForConditionalGeneration
            and T5TokenizerFast
            and torch
        ):
            model_path = Path(self.diffusion_model_path)
            if model_path.exists():
                try:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    if self.verbose:
                        print(
                            f"DiffusionSpecNormalizer: Loading diffusion model from {model_path} on {self.device}"
                        )
                    self.tokenizer = T5TokenizerFast.from_pretrained(str(model_path))
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        str(model_path)
                    ).to(self.device)  # type: ignore
                    self.model.eval()  # type: ignore
                    self._is_diffusion_model_configured = True
                    self._is_ready = True
                except Exception as e:
                    print(
                        f"DiffusionSpecNormalizer Error: Failed to load diffusion model from {model_path}: {e}"
                    )
                    self._is_diffusion_model_configured = False
            else:
                if self.verbose:
                    print(
                        f"DiffusionSpecNormalizer Warning: Diffusion model path {model_path} not found. Falling back to T5Client."
                    )

        if not self._is_diffusion_model_configured:
            if self.verbose:
                print(
                    "DiffusionSpecNormalizer: Diffusion model unavailable. Initializing T5Client as fallback."
                )
            self.fallback_t5_client = T5Client(app_config=self.app_config)

    @property
    def is_ready(self) -> bool:
        if self._is_diffusion_model_configured and self._is_ready:
            return True
        if self.fallback_t5_client:
            return self.fallback_t5_client.is_ready
        return False

    def generate_spec_yaml(
        self, raw_issue_text: str, context_symbols_string: Optional[str] = None
    ) -> Optional[str]:
        if self._is_diffusion_model_configured and self.model and self.tokenizer:
            if self.verbose:
                print(
                    f"DiffusionSpecNormalizer: Generating spec with diffusion model for: {raw_issue_text[:50]}..."
                )

            prompt = (
                "Translate the following issue description and context into a YAML specification.\n\n"
                f"Context symbols:\n{context_symbols_string if context_symbols_string else 'None'}\n\n"
                f"Issue:\n{raw_issue_text}\n\nGenerate YAML:"
            )
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                    padding="longest",
                ).to(self.device)
                gen_cfg = {"max_length": 512, "num_beams": 4, "early_stopping": True}
                outputs = self.model.generate(inputs.input_ids, **gen_cfg)  # type: ignore
                yaml_str = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                ).strip()
                if self.verbose:
                    print(f"DiffusionSpecNormalizer: Model output:\n{yaml_str}")
                return yaml_str
            except Exception as e:
                print(
                    f"DiffusionSpecNormalizer Error: Inference failed: {e}. Falling back to T5Client."
                )

        if self.fallback_t5_client and self.fallback_t5_client.is_ready:
            if self.verbose:
                print(
                    f"DiffusionSpecNormalizer: Using T5Client fallback for: {raw_issue_text[:50]}..."
                )
            return self.fallback_t5_client.generate_spec_yaml(
                raw_issue_text, context_symbols_string
            )
        else:
            if self.verbose:
                print(
                    "DiffusionSpecNormalizer Error: Neither diffusion model nor T5 fallback is ready."
                )
            return None
