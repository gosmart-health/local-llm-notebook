"""Generate starter Jupyter notebooks for the project."""

import os
from pathlib import Path
import nbformat as nbf

notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 1. 01_llama_server_quickstart.ipynb
# -----------------------------------------------------------------------------
nb1 = nbf.v4.new_notebook()
nb1.metadata = {
    "language_info": {"name": "python", "version": "3.14.0"},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
}

nb1.cells = [
    nbf.v4.new_markdown_cell(
        """# 🦙 Local Model Server Quickstart & Diagnostics

This notebook guides you through connecting to your local or LAN-hosted model server (`Ollama` / `llama-server`) using Python and the OpenAI SDK.

### Features Covered:
1. **Server Health & Diagnostics**: Check root connectivity and discover available models via `client.models.list()`.
2. **OpenAI-Compatible Client**: Initialize the official `openai` Python SDK targeting your local endpoint.
3. **Chat Completions & Streaming**: Run prompt completions, chat interactions, and real-time token streaming.
4. **Structured Outputs**: Request JSON formatted output from the model.
"""
    ),
    nbf.v4.new_code_cell(
        """# Ensure project root is in sys.path and load environment variables
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")
print(f"Project root: {project_root}")
print(f"Server Host: {os.getenv('LLAMA_SERVER_HOST', 'http://localhost:11434')}")
print(f"API Base:    {os.getenv('LLAMA_API_BASE', 'http://localhost:11434/v1')}")
print(f"Chat Model:  {os.getenv('DEFAULT_CHAT_MODEL', 'qwen2.5-coder:7b')}")
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Check Server Status & Inspect Model Metadata

We test server connectivity and retrieve model metadata (context length, parameter size, quantization, etc.).
"""
    ),
    nbf.v4.new_code_cell(
        """from src.llama_helper import check_server_health, get_server_props
from rich.pretty import pprint

health = check_server_health()
print(f"Server Health Status: {health.get('status')}")
pprint(health)

props = get_server_props()
print("\\nServer Properties & Model Info:")
pprint(props)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. Initialize OpenAI Client

The server exposes a drop-in `/v1` OpenAI-compatible REST API. We configure the standard `OpenAI` client pointing to `LLAMA_API_BASE`.
"""
    ),
    nbf.v4.new_code_cell(
        """from src.llama_helper import get_openai_client, DEFAULT_CHAT_MODEL

client = get_openai_client()
print(f"OpenAI Client base_url: {client.base_url}")
print(f"Default model:          {DEFAULT_CHAT_MODEL}")

# Query models from the server
models = client.models.list()
print("\\nDiscovered models on server:")
for m in models.data:
    print(f" • {m.id}")
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. Basic Chat Completion

Send a standard chat completion request with system and user messages.
"""
    ),
    nbf.v4.new_code_cell(
        """try:
    response = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a concise AI assistant. Explain concepts clearly and briefly."},
            {"role": "user", "content": "What is the primary difference between dense vector embeddings and sparse representations?"}
        ],
        temperature=0.7,
        max_tokens=150
    )
    print("--- Response ---")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Request failed: {e}\\n(Ensure llama-server is running with: llama-server -m <model.gguf> --port 11434)")
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Streaming Responses (Real-Time Token Output)

Stream tokens progressively as `llama-server` generates them.
"""
    ),
    nbf.v4.new_code_cell(
        """prompt = "Write a 4-line poem about linear algebra, high dimensions, and vectors."

try:
    stream = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.7,
        max_tokens=120
    )
    print("--- Streaming Output ---")
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()
except Exception as e:
    print(f"Streaming failed: {e}")
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Structured JSON Output

Local llama models can return structured JSON outputs for extraction tasks.
"""
    ),
    nbf.v4.new_code_cell(
        """try:
    response = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a metadata extractor. Respond with a JSON object containing keys: 'term', 'definition', 'use_cases'."},
            {"role": "user", "content": "Define 'Cosine Similarity'."}
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=200
    )
    print("--- Structured JSON Response ---")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Structured output test note: {e}")
"""
    ),
]

with open(notebooks_dir / "01_llama_server_quickstart.ipynb", "w") as f:
    nbf.write(nb1, f)

# -----------------------------------------------------------------------------
# 2. 02_vector_embeddings_numpy.ipynb
# -----------------------------------------------------------------------------
nb2 = nbf.v4.new_notebook()
nb2.metadata = {
    "language_info": {"name": "python", "version": "3.14.0"},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
}

nb2.cells = [
    nbf.v4.new_markdown_cell(
        """# 📐 Vector Embeddings & Math with NumPy

This notebook explores vector embeddings generated from `llama-server` (or local fallback embeddings) and implements vector operations from scratch using **NumPy**.

### Topics Covered:
1. **Embedding Extraction**: Fetching vector embeddings via `/v1/embeddings`.
2. **Vector Math Fundamentals**:
   - $L_2$ Normalization: $\\hat{v} = \\frac{v}{\\|v\\|_2}$
   - Cosine Similarity: $\\cos(\\theta) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = \\hat{u} \\cdot \\hat{v}$
   - Pairwise Distance & Similarity Matrices.
3. **Semantic Search Engine**:
   - Building a document index with embeddings.
   - Vector indexing and Top-$K$ semantic similarity ranking using `np.argpartition`.
4. **Clustering & 2D Projection**:
   - Visualizing semantic clusters in 2D space with PCA and `matplotlib`.
"""
    ),
    nbf.v4.new_code_cell(
        """# Ensure project root is in sys.path
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.llama_helper import (
    get_embeddings,
    cosine_similarity,
    normalize_vectors,
    pairwise_similarity_matrix,
    top_k_similar,
    check_server_health
)

# Set random seed for reproducibility
np.random.seed(42)
print("NumPy version:", np.__version__)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Helper: Embedding Generator with Fallback

If `llama-server` is running with embedding enabled (`--embedding` flag or embedding model loaded), we fetch embeddings directly from the server. If the server is offline during development, a deterministic mock embedding is generated so all math/visualizations remain interactive.
"""
    ),
    nbf.v4.new_code_cell(
        """def generate_embeddings_safe(texts: list[str]) -> np.ndarray:
    \"\"\"Fetch embeddings from llama-server or fallback to deterministic mock vectors.\"\"\"
    try:
        health = check_server_health()
        if health.get("status") == "online":
            return get_embeddings(texts)
    except Exception:
        pass
    
    # Deterministic fallback for offline testing
    print("[Note] Using synthetic normalized embeddings (start llama-server with --embedding for live model vectors)")
    vectors = []
    for text in texts:
        # Create a repeatable pseudo-vector based on string hash
        hash_val = hash(text) % (2**32)
        rng = np.random.RandomState(hash_val)
        v = rng.randn(384).astype(np.float32)
        # Add category clustering bias based on keywords
        if any(w in text.lower() for w in ["cat", "dog", "pet", "animal"]):
            v[0:50] += 2.0
        elif any(w in text.lower() for w in ["python", "code", "programming", "algorithm", "software"]):
            v[50:100] += 2.0
        elif any(w in text.lower() for w in ["space", "galaxy", "star", "planet", "telescope"]):
            v[100:150] += 2.0
        vectors.append(v)
    return normalize_vectors(np.array(vectors))
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. Sample Dataset & Embedding Generation

Let's define a corpus of sentences spanning different semantic domains (Animals, Programming, Astronomy).
"""
    ),
    nbf.v4.new_code_cell(
        """corpus = [
    # Animals
    "The domestic cat is a small carnivorous mammal.",
    "Dogs are loyal pets that enjoy outdoor games and fetching balls.",
    "A golden retriever is a popular breed of domestic dog.",
    "Felines are known for their agility and predatory instincts.",
    
    # Programming / Computer Science
    "Python is an interpreted, high-level programming language.",
    "NumPy provides support for large multi-dimensional arrays and matrices.",
    "Writing clean modular code improves software maintainability.",
    "Algorithms and data structures form the backbone of computer science.",
    
    # Astronomy / Space
    "The James Webb Space Telescope captures deep infrared images of distant galaxies.",
    "Stars generate energy through nuclear fusion in their cores.",
    "Mars is the fourth planet from the Sun and is known as the Red Planet.",
    "Black holes have gravitational pulls so strong that not even light can escape."
]

categories = ["Animals"] * 4 + ["Tech/Code"] * 4 + ["Space"] * 4

embeddings = generate_embeddings_safe(corpus)
print(f"Generated embeddings matrix shape: {embeddings.shape}")
print(f"Dimension of each embedding vector: {embeddings.shape[1]}")
print(f"L2 Norm of first vector (should be ~1.0): {np.linalg.norm(embeddings[0]):.4f}")
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. Pairwise Cosine Similarity Matrix

With normalized vectors ($\\|v\\|_2 = 1$), the cosine similarity matrix is simply the matrix product $M = V V^T$.
"""
    ),
    nbf.v4.new_code_cell(
        """sim_matrix = pairwise_similarity_matrix(embeddings)

# Plotting the heatmap
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1)

ax.set_xticks(range(len(corpus)))
ax.set_yticks(range(len(corpus)))
short_labels = [f"[{categories[i][:3]}] {c[:25]}..." for i, c in enumerate(corpus)]
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(short_labels, fontsize=8)

plt.colorbar(im, label="Cosine Similarity")
plt.title("Pairwise Semantic Similarity Heatmap", fontsize=13, pad=15)
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Semantic Search Engine (Query vs Corpus)

Given a natural language query, we:
1. Embed the query $\\vec{q}$
2. Compute similarity $\\vec{s} = V \\cdot \\hat{q}$
3. Retrieve top-$K$ highest ranking documents using `top_k_similar`
"""
    ),
    nbf.v4.new_code_cell(
        """queries = [
    "How to write efficient Python software?",
    "Tell me about feline hunting and behavior",
    "Discoveries of planets and galaxies in outer space"
]

for query in queries:
    query_vec = generate_embeddings_safe([query])[0]
    top_indices, top_scores = top_k_similar(query_vec, embeddings, k=3)
    
    print("=" * 70)
    print(f"🔍 QUERY: '{query}'")
    print("=" * 70)
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
        print(f" Rank {rank} [Score: {score:.4f} | {categories[idx]}]: {corpus[idx]}")
    print()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. 2D Dimensionality Reduction & Cluster Visualization (PCA)

Using Principal Component Analysis (PCA) to project high-dimensional embeddings into 2D space for visual clustering.
"""
    ),
    nbf.v4.new_code_cell(
        """pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(10, 7))
colors = {"Animals": "#e74c3c", "Tech/Code": "#3498db", "Space": "#2ecc71"}

for cat in set(categories):
    mask = [c == cat for c in categories]
    ax.scatter(
        reduced_vectors[mask, 0],
        reduced_vectors[mask, 1],
        c=colors[cat],
        label=cat,
        s=120,
        alpha=0.85,
        edgecolors="k"
    )

for i, txt in enumerate(corpus):
    ax.annotate(
        f"{corpus[i][:30]}...",
        (reduced_vectors[i, 0] + 0.02, reduced_vectors[i, 1] + 0.02),
        fontsize=8,
        alpha=0.85
    )

ax.set_title("2D Projection of Vector Embeddings (PCA)", fontsize=13)
ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(title="Semantic Category")
plt.tight_layout()
plt.show()
"""
    ),
]

with open(notebooks_dir / "02_vector_embeddings_numpy.ipynb", "w") as f:
    nbf.write(nb2, f)

print("Notebooks created successfully!")
