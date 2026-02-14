"""DeepRecall CLI -- command-line interface for serving, querying, and ingesting."""

from __future__ import annotations

import os
import sys

import click
from dotenv import load_dotenv

load_dotenv()


@click.group()
@click.version_option(version="0.2.0", prog_name="deeprecall")
def main() -> None:
    """DeepRecall -- Recursive reasoning over your data."""


@main.command()
@click.option(
    "--vectorstore",
    type=click.Choice(["chroma", "milvus", "qdrant", "pinecone"]),
    default="chroma",
    help="Vector store backend.",
)
@click.option("--collection", default="deeprecall", help="Collection/index name.")
@click.option("--persist-dir", default=None, help="Persist directory (ChromaDB only).")
@click.option("--host", default="0.0.0.0", help="Server host.")
@click.option("--port", default=8000, type=int, help="Server port.")
@click.option("--backend", default="openai", help="LLM backend.")
@click.option("--model", default="gpt-4o-mini", help="LLM model name.")
@click.option("--api-keys", default=None, help="Comma-separated API keys for auth.")
@click.option("--rate-limit", default=None, type=int, help="Requests per minute per key.")
def serve(
    vectorstore: str,
    collection: str,
    persist_dir: str | None,
    host: str,
    port: int,
    backend: str,
    model: str,
    api_keys: str | None,
    rate_limit: int | None,
) -> None:
    """Start the OpenAI-compatible API server."""
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn required. Install with: pip install deeprecall[server]")
        sys.exit(1)

    engine = _build_engine(vectorstore, collection, persist_dir, backend, model)

    from deeprecall.adapters.openai_server import create_app

    keys_list = api_keys.split(",") if api_keys else None
    app = create_app(engine, api_keys=keys_list, requests_per_minute=rate_limit)

    click.echo(f"Starting DeepRecall server on {host}:{port}")
    click.echo(f"  Vector store: {vectorstore} ({collection})")
    click.echo(f"  Backend: {backend}/{model}")
    if keys_list:
        click.echo(f"  Auth: {len(keys_list)} API key(s) configured")
    if rate_limit:
        click.echo(f"  Rate limit: {rate_limit} req/min")
    click.echo(f"  Docs: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


@main.command()
@click.argument("query_text")
@click.option(
    "--vectorstore",
    type=click.Choice(["chroma", "milvus", "qdrant", "pinecone"]),
    default="chroma",
    help="Vector store backend.",
)
@click.option("--collection", default="deeprecall", help="Collection/index name.")
@click.option("--persist-dir", default=None, help="Persist directory (ChromaDB only).")
@click.option("--backend", default="openai", help="LLM backend.")
@click.option("--model", default="gpt-4o-mini", help="LLM model name.")
@click.option("--verbose", is_flag=True, help="Show detailed output.")
@click.option("--max-searches", default=None, type=int, help="Max search calls budget.")
@click.option("--max-tokens", default=None, type=int, help="Max token budget.")
@click.option("--max-time", default=None, type=float, help="Max time budget (seconds).")
def query(
    query_text: str,
    vectorstore: str,
    collection: str,
    persist_dir: str | None,
    backend: str,
    model: str,
    verbose: bool,
    max_searches: int | None,
    max_tokens: int | None,
    max_time: float | None,
) -> None:
    """Query documents with recursive reasoning."""
    from deeprecall.core.guardrails import QueryBudget

    budget = None
    if any(v is not None for v in [max_searches, max_tokens, max_time]):
        budget = QueryBudget(
            max_search_calls=max_searches,
            max_tokens=max_tokens,
            max_time_seconds=max_time,
        )

    engine = _build_engine(vectorstore, collection, persist_dir, backend, model, verbose=verbose)
    result = engine.query(query_text, budget=budget)

    click.echo(f"\nAnswer: {result.answer}")
    click.echo(f"\nSources: {len(result.sources)}")
    click.echo(f"Steps: {len(result.reasoning_trace)}")
    click.echo(f"Time: {result.execution_time:.2f}s")
    click.echo(f"LLM calls: {result.usage.total_calls}")

    if result.confidence is not None:
        click.echo(f"Confidence: {result.confidence:.2f}")
    if result.error:
        click.echo(f"Warning: {result.error}")
    if result.budget_status and result.budget_status.get("budget_exceeded"):
        click.echo(f"Budget: {result.budget_status.get('exceeded_reason')}")


@main.command()
@click.option("--path", required=True, type=click.Path(exists=True), help="Path to documents.")
@click.option(
    "--vectorstore",
    type=click.Choice(["chroma", "milvus", "qdrant", "pinecone"]),
    default="chroma",
    help="Vector store backend.",
)
@click.option("--collection", default="deeprecall", help="Collection/index name.")
@click.option("--persist-dir", default="./chroma_db", help="Persist directory (ChromaDB only).")
def ingest(
    path: str,
    vectorstore: str,
    collection: str,
    persist_dir: str,
) -> None:
    """Ingest documents from a file or directory into the vector store."""
    store = _build_vectorstore(vectorstore, collection, persist_dir)

    documents: list[str] = []
    metadatas: list[dict] = []

    if os.path.isfile(path):
        files = [path]
    else:
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith((".txt", ".md", ".py", ".json", ".csv"))
        ]

    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        documents.append(content)
        metadatas.append({"source": filepath, "filename": os.path.basename(filepath)})

    if not documents:
        click.echo("No documents found.")
        return

    ids = store.add_documents(documents=documents, metadatas=metadatas)
    click.echo(f"Ingested {len(ids)} documents into {vectorstore}/{collection}")
    click.echo(f"Total documents: {store.count()}")


@main.command()
@click.option(
    "--vectorstore",
    type=click.Choice(["chroma", "milvus", "qdrant", "pinecone"]),
    default="chroma",
    help="Vector store backend.",
)
@click.option("--collection", default="deeprecall", help="Collection/index name.")
@click.option("--persist-dir", default="./chroma_db", help="Persist directory (ChromaDB only).")
@click.argument("doc_ids", nargs=-1, required=True)
def delete(
    vectorstore: str,
    collection: str,
    persist_dir: str,
    doc_ids: tuple[str, ...],
) -> None:
    """Delete documents by ID from the vector store."""
    store = _build_vectorstore(vectorstore, collection, persist_dir)
    store.delete(ids=list(doc_ids))
    click.echo(f"Deleted {len(doc_ids)} document(s). Remaining: {store.count()}")


@main.command()
@click.option("--path", default="deeprecall.toml", help="Path for the config file.")
def init(path: str) -> None:
    """Generate a starter config file."""
    config_content = """\
# DeepRecall Configuration
# See: https://github.com/kothapavan1998/deeprecall

[engine]
backend = "openai"
model = "gpt-4o-mini"
max_iterations = 15
top_k = 5
verbose = false

[vectorstore]
type = "chroma"
collection = "deeprecall"
persist_dir = "./chroma_db"

[budget]
# Uncomment to enable budget limits
# max_search_calls = 20
# max_tokens = 50000
# max_time_seconds = 60.0

[server]
host = "0.0.0.0"
port = 8000
# api_keys = ["key1", "key2"]
# rate_limit = 60

[cache]
enabled = false
type = "memory"    # "memory" or "disk"
ttl = 3600
max_size = 1000
"""
    if os.path.exists(path):
        click.echo(f"Config file already exists: {path}")
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(config_content)
    click.echo(f"Created config file: {path}")


def _build_vectorstore(vectorstore: str, collection: str, persist_dir: str | None = None) -> object:
    """Build a vector store from CLI args."""
    if vectorstore == "chroma":
        from deeprecall.vectorstores.chroma import ChromaStore

        return ChromaStore(collection_name=collection, persist_directory=persist_dir)
    elif vectorstore == "milvus":
        from deeprecall.vectorstores.milvus import MilvusStore

        return MilvusStore(collection_name=collection)
    elif vectorstore == "qdrant":
        from deeprecall.vectorstores.qdrant import QdrantStore

        return QdrantStore(collection_name=collection)
    elif vectorstore == "pinecone":
        from deeprecall.vectorstores.pinecone import PineconeStore

        return PineconeStore(index_name=collection)
    else:
        raise ValueError(f"Unknown vectorstore: {vectorstore}")


def _build_engine(
    vectorstore: str,
    collection: str,
    persist_dir: str | None,
    backend: str = "openai",
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> object:
    from deeprecall.core.engine import DeepRecallEngine

    store = _build_vectorstore(vectorstore, collection, persist_dir)
    return DeepRecallEngine(
        vectorstore=store,
        backend=backend,
        backend_kwargs={
            "model_name": model,
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
