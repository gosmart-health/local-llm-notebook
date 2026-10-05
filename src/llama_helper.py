"""Universal Local Model Server Client (LLaMA Server / Ollama) & Vector Embeddings."""

from __future__ import annotations

import os
from typing import Any, Sequence

import httpx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

DEFAULT_HOST = os.getenv("LLAMA_SERVER_HOST", "http://localhost:11434")
DEFAULT_API_BASE = os.getenv("LLAMA_API_BASE", f"{DEFAULT_HOST}/v1")
DEFAULT_API_KEY = os.getenv("LLAMA_API_KEY", "no-key-required")
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "qwen2.5-coder:7b")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "qwen2.5-coder:7b")
DEFAULT_TIMEOUT = float(os.getenv("LLAMA_REQUEST_TIMEOUT", "120"))


def get_openai_client(
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> OpenAI:
    """Return an OpenAI SDK client configured for the local server.

    Args:
        base_url: Optional base URL (defaults to LLAMA_API_BASE, e.g. http://localhost:11434/v1).
        api_key: Optional API key (defaults to LLAMA_API_KEY).
        timeout: Request timeout in seconds.
    """
    return OpenAI(
        base_url=base_url or DEFAULT_API_BASE,
        api_key=api_key or DEFAULT_API_KEY,
        timeout=timeout or DEFAULT_TIMEOUT,
    )


def check_server_health(host: str | None = None) -> dict[str, Any]:
    """Check the health status of the server (compatible with llama-server, Ollama, & v1 endpoints).

    Args:
        host: Server host URL (defaults to LLAMA_SERVER_HOST).
    """
    target_host = (host or DEFAULT_HOST).rstrip("/")

    # 1. Probe Ollama /api/version or root /
    try:
        r_ver = httpx.get(f"{target_host}/api/version", timeout=3.0)
        if r_ver.status_code == 200:
            version_info = r_ver.json()
            models_list = get_server_models(target_host)
            return {
                "status": "online",
                "host": target_host,
                "server_type": f"Ollama v{version_info.get('version', '')}".strip(),
                "models": models_list,
            }
    except Exception:
        pass

    # 2. Probe llama.cpp /health
    try:
        r_health = httpx.get(f"{target_host}/health", timeout=3.0)
        if r_health.status_code == 200:
            data = r_health.json() if r_health.headers.get("content-type", "").startswith("application/json") else {}
            return {
                "status": "online",
                "host": target_host,
                "server_type": "llama.cpp llama-server",
                "details": data,
                "models": get_server_models(target_host),
            }
    except Exception:
        pass

    # 3. Probe OpenAI /v1/models
    try:
        client = get_openai_client(base_url=f"{target_host}/v1")
        models_resp = client.models.list()
        return {
            "status": "online",
            "host": target_host,
            "server_type": "OpenAI-compatible server",
            "models": [m.id for m in models_resp.data],
        }
    except Exception as e:
        return {
            "status": "unreachable",
            "host": target_host,
            "error": str(e),
        }


def get_server_props(host: str | None = None) -> dict[str, Any]:
    """Retrieve server and model properties (supports both llama-server /props and Ollama /api/tags).

    Args:
        host: Server host URL (defaults to LLAMA_SERVER_HOST).
    """
    target_host = (host or DEFAULT_HOST).rstrip("/")

    # 1. Try llama-server /props
    try:
        r_props = httpx.get(f"{target_host}/props", timeout=3.0)
        if r_props.status_code == 200:
            return {
                "server_backend": "llama-server",
                "props": r_props.json(),
            }
    except Exception:
        pass

    # 2. Try Ollama /api/tags & /api/version
    try:
        r_tags = httpx.get(f"{target_host}/api/tags", timeout=3.0)
        if r_tags.status_code == 200:
            tags_data = r_tags.json()
            models_info = {}
            for m in tags_data.get("models", []):
                name = m.get("name")
                details = m.get("details", {})
                models_info[name] = {
                    "size_gb": round(m.get("size", 0) / (1024**3), 2),
                    "parameter_size": details.get("parameter_size", "unknown"),
                    "quantization": details.get("quantization_level", "unknown"),
                    "context_length": details.get("context_length", "unknown"),
                    "embedding_length": details.get("embedding_length", "unknown"),
                    "family": details.get("family", "unknown"),
                    "capabilities": m.get("capabilities", []),
                }
            
            ver = "unknown"
            try:
                ver = httpx.get(f"{target_host}/api/version", timeout=2.0).json().get("version", "unknown")
            except Exception:
                pass

            return {
                "server_backend": "Ollama",
                "version": ver,
                "model_count": len(models_info),
                "models": models_info,
            }
    except Exception as e:
        return {"error": f"Failed to retrieve properties: {e}"}

    return {"error": "Server does not expose /props or /api/tags"}


def get_server_models(host: str | None = None) -> list[str]:
    """Retrieve list of model names available on the server."""
    target_host = (host or DEFAULT_HOST).rstrip("/")

    # Try Ollama /api/tags
    try:
        r_tags = httpx.get(f"{target_host}/api/tags", timeout=3.0)
        if r_tags.status_code == 200:
            return [m["name"] for m in r_tags.json().get("models", [])]
    except Exception:
        pass

    # Try OpenAI /v1/models
    try:
        client = get_openai_client(base_url=f"{target_host}/v1")
        models_resp = client.models.list()
        return [m.id for m in models_resp.data]
    except Exception:
        return []


def get_embeddings(
    texts: str | Sequence[str],
    model: str | None = None,
    client: OpenAI | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Generate vector embeddings for input text(s) via OpenAI-compatible endpoint.

    Args:
        texts: A single text string or list of text strings.
        model: Model identifier (defaults to DEFAULT_EMBEDDING_MODEL).
        client: Optional OpenAI client instance.
        normalize: If True, L2-normalizes the vectors so cosine similarity = dot product.

    Returns:
        np.ndarray: 1D array of shape (dim,) if single string input,
                   or 2D array of shape (N, dim) if sequence input.
    """
    is_single = isinstance(texts, str)
    input_list = [texts] if is_single else list(texts)

    oa_client = client or get_openai_client()
    target_model = model or DEFAULT_EMBEDDING_MODEL

    response = oa_client.embeddings.create(
        model=target_model,
        input=input_list,
    )

    # Sort embeddings by index to preserve order
    sorted_data = sorted(response.data, key=lambda x: x.index)
    vectors = np.array([item.embedding for item in sorted_data], dtype=np.float32)

    if normalize:
        vectors = normalize_vectors(vectors)

    if is_single:
        return vectors[0]
    return vectors


# ==============================================================================
# Vector Math & Search Utilities (NumPy)
# ==============================================================================

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize 1D or 2D NumPy array.

    Args:
        vectors: Array of shape (dim,) or (N, dim).

    Returns:
        L2-normalized array of identical shape.
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm == 0:
            return vectors
        return vectors / norm
    elif vectors.ndim == 2:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms
    else:
        raise ValueError(f"Expected 1D or 2D array, got {vectors.ndim}D array.")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector(s) a and vector(s) b.

    Args:
        a: Shape (dim,) or (N, dim)
        b: Shape (dim,) or (M, dim)

    Returns:
        Float (if both 1D), 1D array of shape (N,) or (M,) (if one is 1D),
        or 2D matrix of shape (N, M) (if both are 2D).
    """
    a_norm = normalize_vectors(a)
    b_norm = normalize_vectors(b)

    if a_norm.ndim == 1 and b_norm.ndim == 1:
        return float(np.dot(a_norm, b_norm))
    elif a_norm.ndim == 1 and b_norm.ndim == 2:
        return np.dot(b_norm, a_norm)
    elif a_norm.ndim == 2 and b_norm.ndim == 1:
        return np.dot(a_norm, b_norm)
    elif a_norm.ndim == 2 and b_norm.ndim == 2:
        return np.dot(a_norm, b_norm.T)
    else:
        raise ValueError("Inputs must be 1D or 2D arrays")


def pairwise_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for a set of vectors.

    Args:
        vectors: 2D array of shape (N, dim).

    Returns:
        Square matrix of shape (N, N) where M[i, j] = cosine_sim(vec_i, vec_j).
    """
    normed = normalize_vectors(vectors)
    return np.dot(normed, normed.T)


def top_k_similar(
    query_vec: np.ndarray,
    corpus_vecs: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Find top-k most similar vectors in corpus to query vector.

    Args:
        query_vec: 1D query vector of shape (dim,).
        corpus_vecs: 2D corpus vectors of shape (N, dim).
        k: Number of top results to return.

    Returns:
        (top_indices, top_scores): Arrays of indices and cosine similarity scores sorted descending.
    """
    similarities = cosine_similarity(corpus_vecs, query_vec)
    k = min(k, len(similarities))
    # argpartition for efficiency on large corpus, then sort top k
    partitioned_indices = np.argpartition(similarities, -k)[-k:]
    sorted_top = partitioned_indices[np.argsort(-similarities[partitioned_indices])]
    return sorted_top, similarities[sorted_top]
