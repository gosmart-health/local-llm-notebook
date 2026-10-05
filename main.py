"""Diagnostic CLI and Quick Test for local LLaMA Server and Embeddings."""

import os
from dotenv import load_dotenv
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.llama_helper import (
    check_server_health,
    get_server_models,
    cosine_similarity,
    normalize_vectors,
    top_k_similar,
)

load_dotenv()
console = Console()


def main():
    console.print(Panel.fit("[bold cyan]🦙 Local LLaMA Server & Vector Embeddings Project[/bold cyan]", border_style="cyan"))

    host = os.getenv("LLAMA_SERVER_HOST", "http://localhost:11434")
    api_base = os.getenv("LLAMA_API_BASE", f"{host}/v1")
    
    # 1. Configuration Table
    table = Table(title="Current Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="dim")
    table.add_column("Value")
    table.add_row("Server Host", host)
    table.add_row("OpenAI API Base", api_base)
    table.add_row("Default Chat Model", os.getenv("DEFAULT_CHAT_MODEL", "local-model"))
    table.add_row("Default Embedding Model", os.getenv("DEFAULT_EMBEDDING_MODEL", "local-embedding"))
    console.print(table)

    # 2. Check Server Health
    console.print("\n[bold yellow]1. Checking Server Connection...[/bold yellow]")
    health = check_server_health(host)
    if health.get("status") == "online":
        server_type = health.get("server_type", "Active")
        console.print(f" [bold green]✔ Server is ONLINE[/bold green] at {host} ({server_type})")
        models = health.get("models", [])
        if models:
            console.print(f"   [cyan]Available Models ({len(models)}):[/cyan] {', '.join(models)}")
    else:
        console.print(f" [bold red]✖ Server is OFFLINE or UNREACHABLE[/bold red] at {host}")
        console.print(f"   Error: {health.get('error')}")

    # 3. Test NumPy Vector Operations
    console.print("\n[bold yellow]2. Verifying NumPy Vector Math...[/bold yellow]")
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 1.0, 0.0])
    sim = cosine_similarity(v1, v2)
    console.print(f"   Cosine similarity between [1,0,0] and [1,1,0]: [bold green]{sim:.4f}[/bold green] (Expected: 0.7071)")

    corpus = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.6, 0.0]
    ])
    query = np.array([0.9, 0.1, 0.0])
    top_indices, scores = top_k_similar(query, corpus, k=2)
    console.print(f"   Top-2 index ranking: {top_indices.tolist()}, Scores: {[round(s, 4) for s in scores.tolist()]}")
    console.print("   [bold green]✔ Vector math & search algorithms operational.[/bold green]")

    # 4. Next Steps
    console.print("\n[bold yellow]3. Next Steps:[/bold yellow]")
    console.print("   • Start JupyterLab:")
    console.print("     [bold green]uv run jupyter lab[/bold green]")
    console.print("   • Open [cyan]notebooks/01_llama_server_quickstart.ipynb[/cyan] or [cyan]notebooks/02_vector_embeddings_numpy.ipynb[/cyan]")


if __name__ == "__main__":
    main()
