# Installation Guide

This guide covers how to install **diffuseLLM_agent** and its dependencies.

## 1. Prerequisites

- **Python 3.10+** is recommended. The assistant relies on modern language features and packages that work best on recent Python versions.
- A POSIX environment (Linux or macOS) with access to a terminal.
 - Optional GPU acceleration for faster model inference. Install the
   [vLLM](https://github.com/vllm-project/vllm) package and set
   `general.use_vllm: true` in `config.yaml` to enable it.

## 2. Clone the Repository

```bash
git clone https://github.com/your-org/diffuseLLM_agent.git
cd diffuseLLM_agent
```

## 3. Set Up a Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install Dependencies

All required packages are listed in `requirements.txt`. Some features rely on optional libraries such as `transformers` and `torch`. Install everything with:

```bash
pip install -r requirements.txt
```

> **Note**: Depending on your platform you may need extra system packages (e.g. build tools, `faiss` libraries) for some dependencies to compile correctly.

## 5. Verify the Installation

Run the test suite to ensure the environment is ready:

```bash
pytest -q
```

Missing optional packages will cause some tests to be skipped or fail. Install any additional requirements as needed.

## 6. Running with Docker

If you prefer an isolated setup, build and run the assistant in a container:

```bash
docker build -t diffuse-agent .
docker run -p 5001:5001 -v /path/to/project:/workspace/project diffuse-agent
```

The application will start the web UI on port 5001 inside the container.

You are now ready to profile a repository and launch the assistant. Continue to the [Usage Guide](HOW_TO_USE.md) for detailed instructions.
