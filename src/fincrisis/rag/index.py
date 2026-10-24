"""A FAISS vector store for LangChain, saved without pickle (docs/decisions.md, D14).

`FaissStore` implements LangChain's `VectorStore` interface (so `as_retriever()`
works) on top of a FAISS inner-product index over L2-normalized vectors, which
makes scores cosine similarities. It is saved as `index.faiss` (FAISS's own
format) plus `chunks.json`, so loading never unpickles anything. A manifest
records SHA-256 checksums of both files and of every source document, so an
altered or stale index is refused.
"""

import hashlib
import json
import shutil
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

INDEX_FILES = ("index.faiss", "chunks.json")


class RAGIndexError(Exception):
    """The index is missing, altered, stale, or built with different settings."""


def _normalized(vectors) -> np.ndarray:
    array = np.asarray(vectors, dtype="float32")
    if array.ndim != 2 or len(array) == 0:
        raise RAGIndexError("embeddings must be a non-empty 2-D array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


class FaissStore(VectorStore):
    def __init__(self, embedding, index, documents: list[Document]):
        if index.ntotal != len(documents):
            raise RAGIndexError("index and document count differ")
        self._embedding, self.index, self.documents = embedding, index, documents

    @property
    def embeddings(self):
        return self._embedding

    @classmethod
    def from_documents(cls, documents: list[Document], embedding, **kwargs) -> "FaissStore":
        if not documents:
            raise RAGIndexError("no chunks to index")
        vectors = _normalized(embedding.embed_documents([d.page_content for d in documents]))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(embedding, index, list(documents))

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs) -> "FaissStore":
        metadatas = metadatas or [{} for _ in texts]
        return cls.from_documents([Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)], embedding)

    def add_texts(self, texts, metadatas=None, **kwargs):
        raise NotImplementedError("rebuild the index with scripts/build_rag_index.py instead of adding to it")

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> list[tuple[Document, float]]:
        vector = _normalized([self._embedding.embed_query(query)])
        k = min(k, len(self.documents))
        scores, ids = self.index.search(vector, k)
        return [(self.documents[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

    def _select_relevance_score_fn(self):
        return lambda score: score  # cosine similarity already

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        chunks = [{"text": d.page_content, "metadata": d.metadata} for d in self.documents]
        (directory / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedding) -> "FaissStore":
        directory = Path(directory)
        index = faiss.read_index(str(directory / "index.faiss"))
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        return cls(embedding, index, [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_store(store: FaissStore, index_dir: Path, manifest: dict, force: bool) -> dict:
    """Write to a temporary directory, then move it into place; refuse to overwrite without force."""
    index_dir = Path(index_dir)
    if index_dir.exists() and not force:
        raise FileExistsError(f"{index_dir} already exists; use --force to rebuild")
    tmp = index_dir.with_name(index_dir.name + ".part")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        store.save(tmp)
        manifest = {**manifest, "index_sha256": {name: sha256_file(tmp / name) for name in INDEX_FILES}}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if index_dir.exists():
            shutil.rmtree(index_dir)
        tmp.replace(index_dir)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return manifest


def read_manifest(index_dir: Path) -> dict:
    path = Path(index_dir) / "manifest.json"
    if not path.exists():
        raise RAGIndexError(f"no index at {index_dir}; run scripts/build_rag_index.py")
    return json.loads(path.read_text(encoding="utf-8"))


def stale_sources(manifest: dict, root: Path, current_sources: list[str] | None = None) -> list[str]:
    """Sources changed or removed since the build, plus new sources not in the index."""
    changed = [s for s, digest in manifest["sources"].items()
               if not (Path(root) / s).exists() or sha256_file(Path(root) / s) != digest]
    if current_sources is not None:
        changed += sorted(set(current_sources) - set(manifest["sources"]))
    return changed


def load_store(index_dir: Path, embedding, embedding_model: str, root: Path,
               current_sources: list[str] | None = None, allow_stale: bool = False) -> tuple[FaissStore, dict]:
    index_dir = Path(index_dir)
    manifest = read_manifest(index_dir)
    for name in INDEX_FILES:
        if not (index_dir / name).exists() or sha256_file(index_dir / name) != manifest["index_sha256"][name]:
            raise RAGIndexError(f"{name} does not match manifest.json; rebuild the index")
    if manifest["embedding_model"] != embedding_model:
        raise RAGIndexError(f"index was built with {manifest['embedding_model']}, not {embedding_model}")
    stale = stale_sources(manifest, root, current_sources)
    if stale and not allow_stale:
        raise RAGIndexError(f"index is out of date for {stale}; rebuild with scripts/build_rag_index.py --force")
    store = FaissStore.load(index_dir, embedding)
    if len(store.documents) != manifest["chunks"]:
        raise RAGIndexError("chunk count differs from manifest.json")
    return store, manifest
