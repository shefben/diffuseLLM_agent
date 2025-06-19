# Usage Guide

This document walks through the typical workflow for running **diffuseLLM_agent** on your own codebase.

## 1. Profile the Repository

Before the assistant can generate patches it must learn the project’s style and build the initial knowledge store.

```bash
python3 scripts/profile_style.py /path/to/your/project
```

This step scans the code, creates `style_fingerprint.json`, and generates configuration files such as `.black.toml` and `ruff.toml` in the project root.

## 2. Launch the Web Interface

After profiling, start the assistant and web dashboard:

```bash
python3 scripts/launch_assistant.py /path/to/your/project
```

The script digests the repository, initializes databases, and opens a Flask web interface (default `http://localhost:5001`).
Parameters like the port and active learning settings are loaded from `config.yaml`, so extra flags are usually unnecessary. Pass `--active-learning` to force training even if the config disables it.
Set `general.use_vllm` to `true` in the config if you installed the optional vLLM package for faster model inference.

## 3. Submitting Issues

From the web interface:

1. Enter a plain‑text feature request or bug description in the form.
2. Click **Apply Patch**.
3. The assistant normalizes the request, plans a sequence of operations, and generates a validated patch.
4. The proposed plan is shown in a text box so you can tweak phase order or parameters before hitting **Approve Plan & Run**.
5. Successful patches are logged in **Success Memory** for review and future fine‑tuning.

## 4. Command-Line Operation

You can also run the assistant without the web UI using a YAML spec:

```bash
python3 scripts/run_assistant.py /path/to/spec.yaml
```

## 5. Fine‑Tuning and Active Learning

`success_memory.jsonl` holds a log of accepted diffs. Use these entries to build a dataset for LoRA training:

```bash
python3 scripts/prepare_finetune_data.py /path/to/project
python3 scripts/finetune_lora.py ./lora_dataset --model-name gpt2
```

Once a fine‑tuned adapter is produced, point the assistant’s configuration to it and restart `launch_assistant.py`.

For automated training, run the standalone active learning loop:

```bash
python3 scripts/active_learning_loop.py /path/to/project gpt2 ./lora_output
```

## 6. Training the CorePredictor

When enough patches have accumulated, update the core selection model:

```bash
python3 scripts/train_core_predictor.py /path/to/project
```

The classifier will learn from logged patch metadata and help choose between the LLM and diffusion cores more effectively.

## 7. More Information

See [`docs/FINE_TUNING.md`](FINE_TUNING.md) and [`docs/CORE_PREDICTOR_TRAINING.md`](CORE_PREDICTOR_TRAINING.md) for advanced topics like dataset formats and model training tips.
