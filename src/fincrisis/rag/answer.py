"""Grounded answers with citations (docs/decisions.md, D14).

Default (extractive): the answer consists only of sentences copied from the
retrieved chunks, each followed by a citation marker [n] that points to a
retrieved chunk. If the best retrieval score is below `min_relevance`, the
system returns "insufficient evidence" instead of an answer.

Optional (generative): a local Ollama model writes the answer from the same
evidence. Its citations are checked; if the answer cites nothing, or cites a
source that was not retrieved, the extractive answer is returned instead.
"""

import json
import re
import urllib.request

import numpy as np

INSUFFICIENT = "The indexed documents do not contain enough evidence to answer this question."
CITATION = re.compile(r"\[(\d+)\]")


def retrieve(store, question: str, k: int) -> list[dict]:
    hits = store.similarity_search_with_score(question, k=k)
    return [{"rank": i + 1, "score": float(score), "source": doc.metadata["source"], "section": doc.metadata["section"],
             "chunk_id": doc.metadata["chunk_id"], "text": doc.page_content} for i, (doc, score) in enumerate(hits)]


LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _units(text: str) -> list[str]:
    """Paragraphs with wrapped lines rejoined; list items, headings and table rows kept separate."""
    units, current = [], []

    def flush():
        if current:
            units.append(" ".join(current))
            current.clear()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or re.fullmatch(r"[-*_]{3,}", stripped):
            flush()
        elif stripped.startswith("|"):
            flush()
            if not TABLE_SEPARATOR.match(stripped):
                units.append(" | ".join(cell.strip() for cell in stripped.strip("|").split("|")))
        elif stripped.startswith("#"):
            flush()
            units.append(stripped.lstrip("# ").strip())
        elif LIST_ITEM.match(line) or stripped.startswith(">"):
            flush()
            current.append(LIST_ITEM.sub("", stripped).lstrip("#> ").strip())
        else:
            current.append(stripped)
    flush()
    return units


def sentences(text: str) -> list[str]:
    """Readable sentences from a Markdown chunk (markup removed), at least 25 characters long."""
    out = []
    for unit in _units(text):
        unit = re.sub(r"\*\*|__", "", unit)
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])", unit):
            part = part.strip()
            if len(part) >= 25 and any(c.isalpha() for c in part):
                out.append(part)
    return out


def extractive_answer(question: str, evidence: list[dict], embeddings, max_sentences: int) -> tuple[str, list[int]]:
    """Pick the evidence sentences most similar to the question; return text with [n] markers and cited ranks."""
    candidates = [(e["rank"], s) for e in evidence for s in sentences(e["text"])]
    if not candidates:
        return INSUFFICIENT, []
    q = np.asarray(embeddings.embed_query(question), dtype=float)
    vectors = np.asarray(embeddings.embed_documents([s for _, s in candidates]), dtype=float)
    q /= np.linalg.norm(q) or 1.0
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    order = np.argsort(-(vectors @ q), kind="stable")
    chosen, seen = [], set()
    for i in order:
        rank, sentence = candidates[i]
        if sentence not in seen:
            chosen.append((rank, sentence))
            seen.add(sentence)
        if len(chosen) == max_sentences:
            break
    return " ".join(f"{s} [{rank}]" for rank, s in chosen), sorted({rank for rank, _ in chosen})


def answer(question: str, store, embeddings, top_k: int, min_relevance: float, max_sentences: int,
           generator=None) -> dict:
    evidence = retrieve(store, question, top_k)
    usable = [e for e in evidence if e["score"] >= min_relevance]
    result = {"question": question, "top_score": evidence[0]["score"] if evidence else None,
              "min_relevance": min_relevance, "evidence": evidence, "warnings": []}
    if not usable:
        return {**result, "status": "insufficient_evidence", "method": "none", "answer": INSUFFICIENT, "citations": []}
    text, cited = extractive_answer(question, usable, embeddings, max_sentences)
    method = "extractive"
    if generator is not None:
        try:
            generated = generator(question, usable)
            valid, problem = check_citations(generated, {e["rank"] for e in usable})
            if valid:
                text, cited, method = generated, sorted({int(n) for n in CITATION.findall(generated)}), "generative"
            else:
                result["warnings"].append(f"generated answer rejected ({problem}); extractive answer returned")
        except Exception as error:  # an unavailable local model must not break answering
            result["warnings"].append(f"generator failed ({type(error).__name__}: {error}); extractive answer returned")
    citations = [{"marker": e["rank"], "source": e["source"], "section": e["section"], "chunk_id": e["chunk_id"],
                  "score": e["score"]} for e in usable if e["rank"] in cited]
    return {**result, "status": "answered", "method": method, "answer": text, "citations": citations}


def check_citations(text: str, allowed: set[int]) -> tuple[bool, str]:
    cited = {int(n) for n in CITATION.findall(text)}
    if not text.strip():
        return False, "empty answer"
    if not cited:
        return False, "no citations"
    if not cited <= allowed:
        return False, f"cites unretrieved evidence {sorted(cited - allowed)}"
    return True, ""


def ollama_generator(model: str, url: str = "http://localhost:11434/api/generate", timeout: float = 120.0, post=None):
    """A generator using a local Ollama server. `post(url, payload) -> dict` can be injected for tests."""
    def default_post(endpoint, payload):
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    send = post or default_post

    def generate(question: str, evidence: list[dict]) -> str:
        context = "\n\n".join(f"[{e['rank']}] ({e['source']} - {e['section']})\n{e['text']}" for e in evidence)
        prompt = ("Answer the question using only the numbered evidence below. Cite every claim with its "
                  "evidence number in square brackets, like [1]. If the evidence does not answer the question, "
                  f"reply exactly: {INSUFFICIENT}\n\nEvidence:\n{context}\n\nQuestion: {question}\nAnswer:")
        reply = send(url, {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}})
        return str(reply.get("response", "")).strip()
    return generate
