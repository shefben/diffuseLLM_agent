# diffuseLLM_agent
The project delivers a self-contained Python assistant that becomes an expert on any Python codebase you point it at.

The assistant is a self-hosted Python toolchain whose only mission is to make a codebase evolve safely, quickly, and stylistically consistent—without asking a human to micromanage the details.
At startup it turns every file in the repo into a rich, queryable knowledge graph that captures functions, classes, inferred types, call relationships, data-flow edges, docstrings, style conventions, and embeddings.
From that moment on, the tool lives beside the code: when a developer phrases a feature request or bug report in plain English, the system translates the request into an exact spec, plans the modification, generates minimal patches that reuse existing helpers, validates them, auto-repairs small mistakes, and finally produces a ready-to-merge pull request.
It achieves this on commodity hardware by combining three pillars:

Static analysis and indexing that off-load structural reasoning away from neural models.

A tiny diffusion model that polishes human language into machine-friendly specs and performs global code denoising when big refactors are required.

A compact (< 8 B) LLM that handles local edits and high-level planning while obeying project style through grammar constraints and retrieval-augmented context.

The end goal is an always-on teammate that understands the project like a senior developer, writes code indistinguishable from hand-written style, never duplicates functionality, and guarantees each change compiles, type-checks, lint-checks, and passes tests before it reaches the main branch.

Phase 1 – Style and convention profiler

• Scan representative files, mine formatter and linter diffs, cluster docstrings and identifiers, then store a “style fingerprint” and commit Black/Ruff config files.

Phase 2 – Repository digestion and incremental knowledge store

• Parse every module with Tree-sitter and LibCST, build call and program-dependence graphs, infer types, embed symbols with MiniLM, add them to a two-tier FAISS index, create a signature trie, and set up a watchdog for live updates.

Phase 3 – Spec normalization with LoRA-compressed diffusion

• Feed raw user text plus nearby symbol names into a lightweight diffusion model that emits a canonical YAML spec defining targets, operations, and acceptance criteria.

Phase 4 – Planner layer driven by a symbolic grammar and small LLM scorer

• Enumerate legal action sequences with a task grammar, score them with a ≤ 4 B LLM that sees the style fingerprint and graph stats, and cache successful mappings for reuse.

Phase 5 – Twin-core Agent Groups for each phase

• Broadcast phase context to an LLM-Coder and a Code-Diffusion editor running in parallel; accept the first validated patch or merge their best hunks if both differ; block duplicated helpers via signature-similarity checks.

Phase 6 – Validator and auto-repair loop

• Run Ruff, Pyright, Black diff check, and only the PDG-impacted tests; allow up to three self-repairs per Agent Group, applying trivial lint fixes locally before reinvoking the models.

Phase 7 – Commit and pull-request builder

• Re-format the final diff, generate a changelog, open a pull request or local commit, and attach both the motivating YAML spec and validator transcript.

Phase 8 – Active learning and continuous refinement

• Log which core’s patch shipped, retrain a selector to pick winners earlier, embed every accepted diff into a “success memory,” and periodically fine-tune LoRA adapters on that memory to align even tighter with project idioms.

## Running the Assistant

Several helper scripts simplify common workflows:

- `scripts/profile_style.py`: generate a style fingerprint and config files for a repository.
- `scripts/run_assistant.py`: run the planner on a YAML spec against a project.
- `scripts/start_webui.py`: launch the Flask dashboard after profiling and digesting the codebase.
- `scripts/active_learning_loop.py`: periodically fine-tune LoRA adapters and retrain the `CorePredictor` using `success_memory.jsonl`.

The web UI now lets you generate patches directly and trigger dataset creation for fine-tuning. Use the "Apply Patch" button after entering an issue and check the success memory page for results.

The web UI exposes endpoints to submit issue text and inspect logged patches.
