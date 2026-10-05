# 🦙 Local LLaMA Server & Vector Embeddings Project

A scaffolded Jupyter Notebook environment managed with [**uv**](https://docs.astral.sh/uv/) for interacting with local or network-hosted `llama-server` instances and experimenting with vector embeddings, cosine similarity, semantic search, and clustering via **NumPy** and **scikit-learn**.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) installed:
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Or via Homebrew:
  brew install uv
  ```

### 2. Environment Setup & Installation
Sync and lock all dependencies into the project `.venv`:
```bash
uv sync
```

### 3. Configure Server Endpoint
Edit [.env](file:///Users/manabutokunaga/development/llm/local-model/.env) to set the IP address or hostname of your `llama-server`:
```ini
LLAMA_SERVER_HOST=http://10.200.0.103:11434
LLAMA_API_BASE=http://10.200.0.103:11434/v1
LLAMA_API_KEY=no-key-required
DEFAULT_CHAT_MODEL=local-model
DEFAULT_EMBEDDING_MODEL=local-embedding
```

> [!IMPORTANT]
> When running `llama-server` on a remote or LAN machine (e.g. `10.200.0.103`), make sure to bind to `0.0.0.0` so it accepts network requests:
> ```bash
> llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 11434 --embedding
> ```

---

## 🧪 Running Diagnostics & Tests

Run the built-in diagnostic CLI tool:
```bash
uv run main.py
```

This verifies:
1. Active configuration from `.env`.
2. Connection status to the configured `llama-server`.
3. NumPy vector normalization and top-$k$ cosine similarity calculations.

---

## 📓 Launching JupyterLab & Notebooks

Start JupyterLab using `uv`:
```bash
uv run jupyter lab
```
Or start classic notebook interface:
```bash
uv run jupyter notebook
```

### Included Notebooks:

| Notebook | Description |
| :--- | :--- |
| [**`01_llama_server_quickstart.ipynb`**](file:///Users/manabutokunaga/development/llm/local-model/notebooks/01_llama_server_quickstart.ipynb) | Connecting to `llama-server`, checking `/health` and `/props`, chat completions, streaming responses, and structured JSON output via OpenAI Python SDK. |
| [**`02_vector_embeddings_numpy.ipynb`**](file:///Users/manabutokunaga/development/llm/local-model/notebooks/02_vector_embeddings_numpy.ipynb) | Generating text embeddings, implementing $L_2$ normalization, cosine similarity matrices, top-$k$ semantic search engine, and PCA 2D cluster visualization using pure NumPy & Matplotlib. |

---

## 🛠️ Project Structure

```text
.
├── .env                       # Active environment configuration
├── .env.example               # Template environment configuration
├── .gitignore                 # Git ignore rules for virtualenv, data, and notebooks
├── pyproject.toml             # Project manifest and dependencies managed by uv
├── README.md                  # Project documentation
├── main.py                    # Diagnostic CLI tool
├── src/
│   ├── __init__.py            # Library exports
│   └── llama_helper.py        # llama-server client, health checker, and NumPy vector math utilities
├── scripts/
│   └── generate_notebooks.py  # Generator script for starter notebooks
└── notebooks/
    ├── 01_llama_server_quickstart.ipynb
    └── 02_vector_embeddings_numpy.ipynb
```

---

## 📦 Installed Packages

- **Notebook & Kernels**: `jupyterlab`, `ipykernel`
- **LLM Clients**: `openai`, `httpx`, `requests`, `pydantic`
- **Data & Vector Math**: `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `pandas`
- **Utilities**: `python-dotenv`, `tqdm`, `rich`
