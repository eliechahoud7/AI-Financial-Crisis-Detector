"""Shared setup for the RAG scripts: configuration, embeddings, and index access (docs/decisions.md, D14).

The embedding model runs locally on CPU. It is loaded from the local Hugging
Face cache only; downloading it requires an explicit `allow_download=True`
(the `--allow-download` flag of scripts/build_rag_index.py).
"""

import platform
from pathlib import Path

import yaml

from fincrisis.rag import chunking as chk
from fincrisis.rag import documents as docs
from fincrisis.rag import index as idx

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_rag_settings(root: Path = PROJECT_ROOT) -> dict:
    return yaml.safe_load((Path(root) / "config" / "settings.yaml").read_text(encoding="utf-8"))["rag"]


def make_embeddings(model_name: str, allow_download: bool = False):
    from langchain_huggingface import HuggingFaceEmbeddings
    try:
        return HuggingFaceEmbeddings(model_name=model_name,
                                     model_kwargs={"device": "cpu", "local_files_only": not allow_download},
                                     encode_kwargs={"normalize_embeddings": True})
    except OSError as error:
        if allow_download:
            raise
        raise RuntimeError(f"Embedding model {model_name} is not in the local cache. Download it once with "
                           "scripts/build_rag_index.py --allow-download.") from error


def make_reranker(model_name: str, allow_download: bool = False):
    """A local cross-encoder as `reranker(question, passages) -> scores` (D31); loaded from the cache unless allowed."""
    from sentence_transformers import CrossEncoder
    try:
        model = CrossEncoder(model_name, device="cpu", local_files_only=not allow_download)
    except (OSError, ValueError) as error:
        if allow_download:
            raise
        raise RuntimeError(f"Reranker model {model_name} is not in the local cache. Download it once with "
                           "scripts/build_rag_index.py --allow-download.") from error

    def rerank(question: str, passages: list[str]) -> list[float]:
        return [float(s) for s in model.predict([(question, p) for p in passages], show_progress_bar=False)]
    return rerank


def retrieval_modes(settings: dict) -> tuple[str, ...]:
    """The retrieval modes to evaluate; rerank only when a reranker model is configured."""
    return ("dense", "lexical", "hybrid") + (("rerank",) if settings["retrieval"].get("rerank_model") else ())


def model_revision(model_name: str) -> str | None:
    """Commit hash of the cached model snapshot, if it can be determined."""
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_name, local_files_only=True)).name
    except Exception:
        return None


def chunking(settings: dict) -> dict:
    """The configured chunking (D30); settings without a `chunking` block use the original character splitter."""
    return {"method": "characters", **settings.get("chunking", {})}


def chunking_matches(manifest: dict, settings: dict) -> bool:
    """True if the index was chunked as the settings say (indexes built before D30 used characters)."""
    built, wanted = manifest.get("chunking", {"method": "characters"}), chunking(settings)
    if built.get("method") != wanted["method"]:
        return False
    return wanted["method"] == "characters" or built.get("min_tokens") == wanted.get("min_tokens")


def token_counter(embeddings):
    """(count, limit): the embedding model's own token count of a text, special tokens included, and the
    longest input it reads (`max_seq_length`); anything longer is cut off before embedding."""
    client = getattr(embeddings, "_client", None)
    tokenizer, limit = getattr(client, "tokenizer", None), getattr(client, "max_seq_length", None)
    if tokenizer is None or not limit:
        raise idx.RAGIndexError("structured chunking needs the embedding model's tokenizer and max_seq_length")
    return (lambda text: len(tokenizer(text, verbose=False)["input_ids"])), int(limit)


def split(settings: dict, loaded: list, embeddings) -> tuple[list, dict]:
    """Chunks for the index, and the chunking record for its manifest."""
    method = chunking(settings)
    if method["method"] == "characters":
        return docs.split_documents(loaded, settings["chunk_size"], settings["chunk_overlap"]), {"method": "characters"}
    if method["method"] != "structured":
        raise ValueError(f"unknown chunking method {method['method']!r}")
    count, limit = token_counter(embeddings)
    chunks = chk.split_structured(loaded, count, limit, method["min_tokens"],
                                  include_section=settings["retrieval"]["embed_section_headings"])
    return chunks, {"method": "structured", "max_tokens": limit, "min_tokens": method["min_tokens"]}


def source_list(settings: dict, root: Path = PROJECT_ROOT) -> list[str]:
    paths = docs.discover_sources(root, settings["sources"], settings.get("exclude", []))
    return [p.relative_to(Path(root).resolve()).as_posix() for p in paths]


def open_store(settings: dict, root: Path = PROJECT_ROOT, embeddings=None, allow_stale: bool = False):
    embeddings = embeddings or make_embeddings(settings["embedding_model"])
    store, manifest = idx.load_store(Path(root) / settings["index_dir"], embeddings, settings["embedding_model"],
                                     root, source_list(settings, root), allow_stale)
    wanted = settings["retrieval"]["embed_section_headings"]
    if manifest.get("embed_section_headings", False) != wanted:
        raise idx.RAGIndexError(f"index was built with embed_section_headings={manifest.get('embed_section_headings', False)}, "
                                f"config says {wanted}; rebuild with scripts/build_rag_index.py --force")
    if not chunking_matches(manifest, settings):
        raise idx.RAGIndexError(f"index was built with chunking {manifest.get('chunking', {'method': 'characters'})}, "
                                f"config says {chunking(settings)}; rebuild with scripts/build_rag_index.py --force")
    return store, manifest, embeddings


def make_retriever(store, settings: dict, mode: str | None = None):
    """The configured retriever for answering; None means the original dense path of `answer()` (D16).

    Pass `mode` explicitly (evaluation) to always get a `Retriever` object.
    """
    from fincrisis.rag.retrieval import Retriever
    r = settings["retrieval"]
    if mode is None and r["mode"] == "dense":
        return None
    mode = mode or r["mode"]
    if mode != "rerank":
        return Retriever(store, mode, r["rrf_k"])
    return Retriever(store, mode, r["rrf_k"], reranker=make_reranker(r["rerank_model"]),
                     candidates=r["rerank_candidates"])


def versions() -> dict:
    import faiss
    import langchain_core
    import langchain_text_splitters
    import sentence_transformers
    import torch
    import transformers
    return {"python": platform.python_version(), "langchain-core": langchain_core.__version__,
            "langchain-text-splitters": getattr(langchain_text_splitters, "__version__", None),
            "sentence-transformers": sentence_transformers.__version__, "transformers": transformers.__version__,
            "torch": torch.__version__, "faiss": faiss.__version__}
