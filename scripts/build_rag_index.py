"""Build the local RAG index from the configured documents (docs/decisions.md, D14).

Usage:
    python scripts/build_rag_index.py                    # stop if an index exists
    python scripts/build_rag_index.py --force            # rebuild
    python scripts/build_rag_index.py --allow-download   # first run: download the embedding model once

Sources are the glob patterns in config/settings.yaml (`rag.sources`). Each
chunk keeps its source path and section heading for citations. The index is
written atomically to `rag.index_dir` with a manifest of checksums.
"""

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fincrisis.rag import documents as docs
from fincrisis.rag import index as idx
from fincrisis.rag import pipeline


def build(settings: dict, root: Path, embeddings, force: bool) -> dict:
    paths = docs.discover_sources(root, settings["sources"], settings.get("exclude", []))
    if not paths:
        raise idx.RAGIndexError(f"no documents match {settings['sources']}")
    loaded = docs.load_documents(paths, root)
    chunks = docs.split_documents(loaded, settings["chunk_size"], settings["chunk_overlap"])
    store = idx.FaissStore.from_documents(chunks, embeddings)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": settings["embedding_model"],
        "embedding_model_revision": pipeline.model_revision(settings["embedding_model"]),
        "dimension": int(store.index.d), "chunks": len(chunks),
        "chunk_size": settings["chunk_size"], "chunk_overlap": settings["chunk_overlap"],
        "sources": {d.metadata["source"]: d.metadata["sha256"] for d in loaded},
        "chunks_per_source": dict(Counter(c.metadata["source"] for c in chunks)),
    }
    return idx.save_store(store, Path(root) / settings["index_dir"], manifest, force)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild an existing index")
    parser.add_argument("--allow-download", action="store_true", help="download the embedding model if not cached")
    args = parser.parse_args()
    settings = pipeline.load_rag_settings()
    index_dir = pipeline.PROJECT_ROOT / settings["index_dir"]
    if index_dir.exists() and not args.force:
        print(f"{index_dir} already exists. Nothing was written. Use --force to rebuild.")
        return 1
    try:
        embeddings = pipeline.make_embeddings(settings["embedding_model"], args.allow_download)
        manifest = build(settings, pipeline.PROJECT_ROOT, embeddings, args.force)
    except (RuntimeError, idx.RAGIndexError, FileExistsError, ValueError) as error:
        print(f"Failed: {error}\nNothing was written.")
        return 1
    manifest["versions"] = pipeline.versions()
    print(f"Indexed {manifest['chunks']} chunks from {len(manifest['sources'])} documents into {index_dir}")
    for source, n in sorted(manifest["chunks_per_source"].items()):
        print(f"  {n:4} {source}")
    print(f"Embedding model: {manifest['embedding_model']} (revision {manifest['embedding_model_revision']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
