"""Shared setup for the RAG scripts: configuration, embeddings, and index access (docs/decisions.md, D14).

The embedding model runs locally on CPU. It is loaded from the local Hugging
Face cache only; downloading it requires an explicit `allow_download=True`
(the `--allow-download` flag of scripts/build_rag_index.py).
"""

import platform
from pathlib import Path

import yaml

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


def model_revision(model_name: str) -> str | None:
    """Commit hash of the cached model snapshot, if it can be determined."""
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_name, local_files_only=True)).name
    except Exception:
        return None


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
    return store, manifest, embeddings


def make_retriever(store, settings: dict, mode: str | None = None):
    """The configured retriever for answering; None means the original dense path of `answer()` (D16).

    Pass `mode` explicitly (evaluation) to always get a `Retriever` object.
    """
    from fincrisis.rag.retrieval import Retriever
    r = settings["retrieval"]
    if mode is None and r["mode"] == "dense":
        return None
    return Retriever(store, mode or r["mode"], r["rrf_k"])


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
