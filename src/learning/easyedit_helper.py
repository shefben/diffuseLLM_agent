"""Wrapper around easyedit for targeted model edits."""
from pathlib import Path

try:
    from easyeditor import EditTrainer, ROMEHyperParams
except ImportError:  # pragma: no cover - optional dependency
    EditTrainer = None  # type: ignore
    ROMEHyperParams = None  # type: ignore


def apply_easyedit(model_dir: Path, instruction: str, new_response: str, verbose: bool = False) -> bool:
    """Apply a targeted weight edit to the model using EasyEdit if available."""
    if EditTrainer is None or ROMEHyperParams is None:
        if verbose:
            print("EasyEdit not installed. Skipping edit.")
        return False
    try:
        hparams = ROMEHyperParams(src=instruction, tgt=new_response)
        trainer = EditTrainer(model_dir=str(model_dir), hparams=hparams)
        trainer.edit()
        trainer.save_model()
        if verbose:
            print(f"EasyEdit applied to model at {model_dir}")
        return True
    except Exception as e:
        if verbose:
            print(f"EasyEdit failed: {e}")
        return False
