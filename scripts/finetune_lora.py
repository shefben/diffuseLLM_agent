#!/usr/bin/env python3
"""Fine-tune a HuggingFace model with LoRA adapters using success_memory.jsonl."""

import argparse
import json
from pathlib import Path

from src.utils.config_loader import load_app_config
from src.learning.ft_dataset import build_finetune_dataset

try:
    from transformers import (
        TrainingArguments,
        Trainer,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import LoraConfig, get_peft_model

    transformers_available = True
except ImportError:  # pragma: no cover - optional dependencies
    transformers_available = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune model using success memory")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_app_config(args.config)
    project_root = Path(cfg.get("general", {}).get("project_root", "."))
    if not project_root.is_absolute():
        project_root = Path.cwd() / project_root

    data_dir = args.data_dir or Path(
        cfg.get("general", {}).get("data_dir", ".agent_data")
    )
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    al_cfg = cfg.get("active_learning", {})
    model_name = args.model_name or al_cfg.get("model_name", "gpt2")
    output_dir = args.output_dir or Path(al_cfg.get("output_dir", "./lora_adapters"))
    dataset_path = args.dataset_path or data_dir / "finetune_dataset.jsonl"
    verbose = args.verbose or cfg.get("general", {}).get("verbose", False)

    if not dataset_path.exists():
        ok = build_finetune_dataset(data_dir, dataset_path, verbose=verbose)
        if not ok:
            print("Failed to prepare fine-tune dataset")
            return

    if not transformers_available:
        print("transformers/peft libraries not available. Cannot train.")
        return

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_config)

    # Load dataset
    texts = []
    with open(dataset_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            obj = json.loads(line)
            texts.append((obj["prompt"], obj["completion"]))
    inputs = tokenizer(
        [t[0] for t in texts], padding=True, truncation=True, return_tensors="pt"
    )
    labels = tokenizer(
        [t[1] for t in texts], padding=True, truncation=True, return_tensors="pt"
    )["input_ids"]
    dataset = [
        dict(input_ids=ids, labels=label_ids)
        for ids, label_ids in zip(inputs["input_ids"], labels)
    ]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
