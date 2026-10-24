"""Find, load, and chunk documents for the RAG index (docs/decisions.md, D14).

Markdown files are split by headings first, so each chunk keeps the section it
came from, then into overlapping pieces of bounded size. Every chunk records
its source path (relative to the project root), section, and chunk id, which
are used for citations.
"""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = {".md", ".txt"}
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def discover_sources(root: Path, patterns: list[str], exclude: list[str] = ()) -> list[Path]:
    """Files matching the glob patterns under root, minus excluded ones; sorted, supported types only."""
    root = Path(root).resolve()
    found = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                resolved = path.resolve()
                if root not in resolved.parents:
                    raise ValueError(f"{path} is outside the project root")
                found.add(resolved)
    excluded = {p.resolve() for pattern in exclude for p in root.glob(pattern)}
    return sorted(found - excluded)


def load_documents(paths: list[Path], root: Path) -> list[Document]:
    root = Path(root).resolve()
    documents = []
    for path in paths:
        data = Path(path).read_bytes()
        text = data.decode("utf-8")
        if not text.strip():
            continue
        documents.append(Document(page_content=text, metadata={
            "source": Path(path).resolve().relative_to(root).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
        }))
    return documents


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    by_heading = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    by_size = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for document in documents:
        source = document.metadata["source"]
        if source.endswith(".md"):
            sections = by_heading.split_text(document.page_content)
        else:
            sections = [Document(page_content=document.page_content, metadata={})]
        for section in sections:
            heading = " > ".join(section.metadata[key] for _, key in HEADERS if key in section.metadata)
            for piece in by_size.split_text(section.page_content):
                if piece.strip():
                    chunks.append(Document(page_content=piece, metadata={
                        "source": source, "section": heading or "(no heading)",
                        "source_sha256": document.metadata["sha256"],
                    }))
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["chunk_id"] = f"{source}#{counters[source]}"
    return chunks
