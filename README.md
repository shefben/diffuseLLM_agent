# diffuseLLM_agent

**diffuseLLM_agent** is a self‑hosted Python assistant that learns your project’s style and automatically generates validated patches. It couples a lightweight diffusion model with an ≤8 B parameter LLM to keep resource usage modest while still producing high‑quality code. A small scoring model guides planning so the most promising patch strategy is tried first.

At startup the assistant parses the entire repository, builds graphs of functions and types, infers naming conventions, and creates embeddings for fast retrieval. When you submit an issue through the web interface or a YAML spec on the command line, the system plans a sequence of refactor operations, generates a patch cooperatively between the LLM and diffusion models, validates it with Ruff, Pyright and tests, and finally logs the result for continuous learning.

## Models

The assistant relies on three cooperating models:

1. **LLMCore** – a general purpose code LLM (≤8 B parameters) that drafts and polishes patches.
2. **DiffusionCore** – a lightweight diffusion model that expands LLM scaffolds and re‑denoises failing regions.
3. **Scorer model** – a small LLM used during planning to rank candidate refactor sequences.

Paths to each model and various parameters are configured in `config.yaml`.
Command‑line arguments act as overrides when needed.

## Project Phases

1. **Style Profiling** – Sample files to learn indentation, quoting, identifier casing and docstring style. Generates `.black.toml`, `ruff.toml`, and `naming_conventions.db`.
2. **Repository Digestion** – Build call graphs and embeddings using Tree‑sitter, LibCST and MiniLM, stored in a FAISS index with a signature trie for duplicate detection.
   A lightweight knowledge graph is populated with call relations that can be queried during planning. The `/query_graph` web endpoint lets you explore these relations interactively.
3. **Spec Normalisation** – Clean free‑text issues into YAML specs using a diffusion pipeline backed by Tiny‑T5.
4. **Generative Planning** – The planner uses an LLM to propose refactor operations from scratch, infers parameters for alternatives, and scores candidate sequences using a small LLM guided by the style fingerprint.
5. **Agent Collaboration** – LLM and diffusion cores iteratively scaffold, expand and polish patches while sharing validation errors for auto‑repair.
6. **Validation** – Ruff, Pyright and scoped pytest runs confirm patches are safe before committing.
7. **Commit Builder** – Format and land the patch locally or via pull request with a changelog.
8. **Active Learning** – Log successful patches, train a classifier to choose the best core, and fine‑tune LoRA adapters on the collected diffs.

## Installation

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed setup instructions. In short:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some features depend on optional packages such as `transformers` and GPU‑enabled `torch`.

## Quick Start

1. **Profile the repository**
   ```bash
   python3 scripts/profile_style.py /path/to/project
   ```
2. **Launch the assistant with the web UI**
   ```bash
   python3 scripts/launch_assistant.py /path/to/project
   ```
   Visit <http://localhost:5001> to submit an issue, review the generated plan, and approve or edit it before execution.
   Model paths and training parameters are read from `config.yaml`, so command‑line flags are optional.

For more detailed workflows—including active learning and fine‑tuning—see the [Usage Guide](docs/HOW_TO_USE.md).

## Helper Scripts

- `profile_style.py` – build the style fingerprint and config files
- `launch_assistant.py` – initialize all phases and start the web interface
- `start_webui.py` – start only the Flask app when components are already initialized
- `run_assistant.py` – run the planner on a YAML spec without the web UI
- `active_learning_loop.py` – periodically fine‑tune models based on success memory
- `prepare_finetune_data.py` – convert logged patches into a fine‑tuning dataset
- `finetune_lora.py` – run LoRA training on the prepared dataset
- `train_core_predictor.py` – update the classifier that selects between the LLM and diffusion cores

## Further Reading

- [docs/FINE_TUNING.md](docs/FINE_TUNING.md) – Details on preparing training data and fine‑tuning models
- [docs/CORE_PREDICTOR_TRAINING.md](docs/CORE_PREDICTOR_TRAINING.md) – How to train the core selection classifier

The project aims to deliver an always‑on teammate that evolves your code safely and in line with your established conventions.
