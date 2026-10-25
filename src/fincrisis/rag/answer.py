"""Grounded answers with citations (docs/decisions.md, D14).

Default (extractive): the answer consists only of sentences copied from the
retrieved chunks, each followed by a citation marker [n] that points to a
retrieved chunk. If the best retrieval score is below `min_relevance`, the
system returns "insufficient evidence" instead of an answer.

Generative (D15): a local Ollama model writes the answer from the same
evidence, with deterministic settings. The answer is accepted only if every
sentence cites retrieved evidence and its numbers and most of its words occur
in the cited passages; otherwise the extractive answer is returned with a
warning. If the model says the evidence is insufficient, so does the result.
"""

import json
import re
import urllib.request

import numpy as np

from fincrisis.rag.index import mask_digests

INSUFFICIENT = "The indexed documents do not contain enough evidence to answer this question."
CITATION = re.compile(r"\[(\d+)\]")


def in_scope(metadata: dict, scope: list[dict]) -> bool:
    """True if the chunk's source matches a scope entry and its section contains that entry's text."""
    return any(metadata["source"] == s["source"] and s["section"] in metadata["section"] for s in scope)


def retrieve_scoped(store, question: str, k: int, scope: list[dict]) -> list[dict]:
    """Dense retrieval restricted to chunks in `scope` (D17); scores are dense cosines."""
    scores = store.cosine_scores(question)
    allowed = [i for i, d in enumerate(store.documents) if in_scope(d.metadata, scope)]
    order = sorted(allowed, key=lambda i: -scores[i])[:k]
    return [{"rank": r, "score": float(scores[i]), "source": store.documents[i].metadata["source"],
             "section": store.documents[i].metadata["section"], "chunk_id": store.documents[i].metadata["chunk_id"],
             "text": store.documents[i].page_content} for r, i in enumerate(order, start=1)]


def retrieve(store, question: str, k: int) -> list[dict]:
    hits = store.similarity_search_with_score(question, k=k)
    return [{"rank": i + 1, "score": float(score), "source": doc.metadata["source"], "section": doc.metadata["section"],
             "chunk_id": doc.metadata["chunk_id"], "text": doc.page_content} for i, (doc, score) in enumerate(hits)]


LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _units(text: str) -> list[tuple[str, str]]:
    """(kind, text) units: paragraphs with wrapped lines rejoined, plus list items, headings, table rows and code.

    A code fence always ends the current unit, and lines between an opening
    and a closing fence form one "code" unit. A chunk boundary can cut a code
    block, leaving an odd number of fences; the chunk is then assumed to start
    inside the block if its first fence is in the first half of its lines
    (otherwise, to end inside it). This is a heuristic.
    """
    units, current = [], []
    lines = text.splitlines()
    fences = [i for i, line in enumerate(lines) if line.strip().startswith("```")]
    in_code = len(fences) % 2 == 1 and fences[0] < len(lines) / 2

    def flush(kind="text"):
        if current:
            units.append((kind, " ".join(current)))
            current.clear()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush("code" if in_code else "text")
            in_code = not in_code
        elif in_code:
            current.append(stripped)
        elif not stripped or re.fullmatch(r"[-*_]{3,}", stripped):
            flush()
        elif stripped.startswith("|"):
            flush()
            if not TABLE_SEPARATOR.match(stripped):
                units.append(("table", " | ".join(cell.strip() for cell in stripped.strip("|").split("|"))))
        elif stripped.startswith("#"):
            flush()
            units.append(("heading", stripped.lstrip("# ").strip()))
        elif stripped.startswith(">"):  # blockquote: wrapped lines continue the same paragraph
            quoted = stripped.lstrip("> ").strip()
            if quoted:
                current.append(quoted)
            else:
                flush()
        elif LIST_ITEM.match(line):
            flush()
            current.append(LIST_ITEM.sub("", stripped).lstrip("#> ").strip())
        else:
            current.append(stripped)
    flush("code" if in_code else "text")
    return units


ENDS_SENTENCE = re.compile(r"""[.!?:;)\]`"'*]$""")


def sentences(text: str) -> list[str]:
    """Readable evidence sentences from a Markdown chunk (markup removed), at least 25 characters long.

    Headings are not evidence and are skipped, as are list lead-ins ending in
    ":", and fenced code. A sentence cut by a chunk boundary or a code block is
    skipped: the first sentence of the chunk, or the first after a code block,
    if it starts lowercase, and the last sentence of the chunk if it stops
    mid-sentence. Other sentences in the same paragraph are kept (D17).
    """
    units = _units(text)
    out = []
    for i, (kind, unit) in enumerate(units):
        if kind in ("heading", "code"):
            continue
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])", re.sub(r"\*\*|__", "", unit))]
        continues = i == 0 or units[i - 1][0] == "code"  # chunk start, or text resuming after a code block
        if kind == "text" and continues and parts and parts[0][:1].islower():
            parts = parts[1:]
        if kind == "text" and i == len(units) - 1 and parts and not ENDS_SENTENCE.search(parts[-1]):
            parts = parts[:-1]
        for part in parts:
            if len(part) >= 25 and any(c.isalpha() for c in part) and not part.endswith(":"):
                out.append(part)
    return out


def extractive_answer(question: str, evidence: list[dict], embeddings, max_sentences: int,
                      scorer=None, relevant_continuation: bool = False, mask: bool = False) -> tuple[str, list[int]]:
    """Pick the best evidence sentences for the question; return text with [n] markers and cited ranks.

    By default a sentence's score is its embedding similarity to the question
    (D14), and the highest-scoring sentences are the answer. With a `scorer`
    (D32), the cross-encoder reranker scores each sentence, read after its
    passage's source and section, for how well it answers the question. The
    best one opens the answer, and the sentences that follow it in its passage
    come next: scored on their own, the steps and qualifications that complete
    a rule lose to sentences that merely repeat the question's words. Only then
    are the next-best sentences used.

    With `relevant_continuation`, a following sentence is used only if its
    cross-encoder score is above 0, the model's own boundary between relevant
    and not relevant (a probability of 0.5). Sentences at or below it are
    skipped, and the rest of the passage is still read. The best sentence
    always opens the answer. If no following sentence passes, the next-best
    sentences fill the answer as above.

    With `mask` (D33), the scorer reads long hex digests as a placeholder; the
    answer keeps the sentences as written.
    """
    candidates = [(e["rank"], s, e) for e in evidence for s in sentences(e["text"])]
    if not candidates:
        return INSUFFICIENT, []
    if scorer is not None:
        scores = np.asarray(scorer(question, [f"{e['source']} - {e['section']}\n{mask_digests(s) if mask else s}"
                                              for _, s, e in candidates]), dtype=float)
        best = int(np.argmax(scores))
        context = [i for i in range(best, len(candidates)) if candidates[i][2] is candidates[best][2]
                   and (i == best or not relevant_continuation or scores[i] > 0)]
        order = context + [int(i) for i in np.argsort(-scores, kind="stable") if int(i) not in context]
    else:
        q = np.asarray(embeddings.embed_query(question), dtype=float)
        vectors = np.asarray(embeddings.embed_documents([s for _, s, _ in candidates]), dtype=float)
        q /= np.linalg.norm(q) or 1.0
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        order = np.argsort(-(vectors @ q), kind="stable")
    chosen, seen = [], set()
    for i in order:
        rank, sentence, _ = candidates[i]
        if sentence not in seen:
            chosen.append((rank, sentence))
            seen.add(sentence)
        if len(chosen) == max_sentences:
            break
    return " ".join(f"{s} [{rank}]" for rank, s in chosen), sorted({rank for rank, _ in chosen})


def answer(question: str, store, embeddings, top_k: int, min_relevance: float, max_sentences: int,
           generator=None, min_support: float = 0.6, retriever=None, scope: list[dict] | None = None) -> dict:
    """Retrieve, then answer extractively or (if `generator` is given) generatively with grounding checks.

    With a `scope` (D17), dense retrieval is restricted to chunks in that scope.
    With a `retriever` (D16), evidence is its top-k ranking and the question is
    answered only if the best dense cosine over all chunks reaches
    `min_relevance`; if it has a `sentence_scorer` (D32), that scorer picks the
    extractive sentences. Otherwise dense retrieval is used. In both dense cases,
    passages below the threshold are dropped. A generated answer is used only if it passes
    `check_grounding`; otherwise the extractive answer is returned with a warning.
    If the model says the evidence is insufficient, or gives a non-answer, the
    result is "insufficient_evidence".
    """
    if scope is not None:
        evidence = retrieve_scoped(store, question, top_k, scope)
        top_score = evidence[0]["score"] if evidence else None
        usable = [e for e in evidence if e["score"] >= min_relevance]
    elif retriever is not None:
        evidence = retriever.search(question, top_k)
        top_score = retriever.max_dense_score(question)
        usable = evidence if top_score >= min_relevance else []
    else:
        evidence = retrieve(store, question, top_k)
        top_score = evidence[0]["score"] if evidence else None
        usable = [e for e in evidence if e["score"] >= min_relevance]
    result = {"question": question, "top_score": top_score,
              "min_relevance": min_relevance, "evidence": evidence, "warnings": [], "generation": None}
    if not usable:
        return {**result, "status": "insufficient_evidence", "method": "none", "answer": INSUFFICIENT, "citations": []}
    text, cited = extractive_answer(question, usable, embeddings, max_sentences,
                                    getattr(retriever, "sentence_scorer", None) if scope is None else None,
                                    getattr(retriever, "relevant_continuation", False), getattr(retriever, "mask", False))
    method = "extractive"
    if generator is not None:
        try:
            generated, info = generator(question, usable)
            result["generation"] = {**info, "raw_answer": generated}
            if is_abstention(generated):
                return {**result, "status": "insufficient_evidence", "method": "generative",
                        "answer": INSUFFICIENT, "citations": []}
            problems = check_grounding(generated, {e["rank"]: e["text"] for e in usable}, min_support, question)
            result["generation"]["grounding_problems"] = problems
            if problems:
                result["warnings"].append(f"generated answer rejected ({problems[0]}); extractive answer returned")
            else:
                text, method = generated, "generative"
                cited = sorted({int(n) for n in CITATION.findall(generated)})
        except Exception as error:  # an unavailable local model must not break answering
            result["warnings"].append(f"generator failed ({type(error).__name__}: {error}); extractive answer returned")
    citations = [{"marker": e["rank"], "source": e["source"], "section": e["section"], "chunk_id": e["chunk_id"],
                  "score": e["score"]} for e in usable if e["rank"] in cited]
    return {**result, "status": "answered", "method": method, "answer": text, "citations": citations}


STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being this that
these those it its into than then there their they them which who whom what when where why how not no nor
so such can could may might must shall should will would do does did done has have had having also only
very more most other some any each all both few many much same own just over under about above below
between after before during while because until again further once here our we you your he she his her""".split())
NUMBER = re.compile(r"\d+(?:[.,]\d+)*%?")


MIN_CLAIM_WORDS = 3  # below this many non-question content words, the word-share test is not meaningful


def _stem(word: str) -> str:
    """Very light suffix stripping, so "looks"/"look" and "reached"/"reach" match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9_-]+", text.lower()) if len(w) >= 3 and w not in STOPWORDS]


NON_ANSWER = re.compile(
    r"\b(?:is|are|was|were)\s+not\s+(?:directly\s+|explicitly\s+|clearly\s+)?"
    r"(?:stated|mentioned|specified|provided|given|available|reported|found)\b"
    r"|\b(?:does|do|did)\s+not\s+(?:state|mention|specify|provide|say|include)\b"
    r"|\bno\s+(?:information|mention|data)\s+(?:is|was|about|on|regarding)\b", re.IGNORECASE)


def is_abstention(text: str) -> bool:
    """The model's refusal sentence, or a hedged non-answer such as "X is not directly stated" (D16)."""
    plain = CITATION.sub("", text).strip().lower().rstrip(".")
    if plain.startswith(INSUFFICIENT.lower().rstrip(".")) or plain == "insufficient evidence":
        return True
    return bool(NON_ANSWER.search(plain))


def _cited_sentences(text: str) -> list[str]:
    """Sentences, with citation-only fragments such as "[2]." attached to the preceding sentence."""
    merged = []
    for part in (p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text)):
        if not part:
            continue
        if merged and not CITATION.sub("", part).strip(" .;,"):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return merged


def check_grounding(text: str, evidence: dict[int, str], min_support: float, question: str = "") -> list[str]:
    """Problems with a generated answer; empty if every sentence is cited and supported by its cited evidence.

    Checks, per sentence: it cites at least one retrieved passage; every number in
    it appears in the cited passages; and at least `min_support` of its content
    words appear in the cited passages. Words restated from the question (matched
    on a light stem, so "looks" matches "look") are not counted as claims, and the
    word-share test is skipped when fewer than MIN_CLAIM_WORDS other content words
    remain; the number check always applies (D16). This is a lexical heuristic, not
    proof of correctness: it cannot detect a wrong claim built from words that
    occur in the evidence.
    """
    question_words = {_stem(w) for w in _words(question)}
    if not text.strip():
        return ["empty answer"]
    all_cited = {int(n) for n in CITATION.findall(text)}
    if not all_cited:
        return ["no citations"]
    if not all_cited <= set(evidence):
        return [f"cites unretrieved evidence {sorted(all_cited - set(evidence))}"]
    problems = []
    for sentence in _cited_sentences(text):
        cited = {int(n) for n in CITATION.findall(sentence)}
        body = CITATION.sub("", sentence).strip()
        if not body.strip(" .;,"):
            continue
        if not cited:
            problems.append(f"uncited sentence: {body[:80]!r}")
            continue
        source = " ".join(evidence[n] for n in cited).lower()
        missing = [x for x in NUMBER.findall(body) if x.replace(",", "") not in source.replace(",", "")]
        if missing:
            problems.append(f"numbers {missing} not in cited evidence: {body[:80]!r}")
        words = [w for w in _words(body) if _stem(w) not in question_words]
        if len(words) >= MIN_CLAIM_WORDS:
            support = sum(_stem(w) in source for w in words) / len(words)
            if support < min_support:
                problems.append(f"only {support:.0%} of content words found in cited evidence: {body[:80]!r}")
    return problems


def ollama_status(url: str, model: str, timeout: float = 5.0, get=None) -> dict:
    """Whether the local Ollama server answers and has `model`; never raises."""
    def default_get(endpoint):
        with urllib.request.urlopen(endpoint, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    fetch = get or default_get
    try:
        version = fetch(f"{url}/api/version").get("version")
        models = {m.get("name") for m in fetch(f"{url}/api/tags").get("models", [])}
    except Exception as error:
        return {"available": False, "reason": f"no Ollama server at {url} ({type(error).__name__})"}
    if model not in models:
        return {"available": False, "reason": f"model {model} is not pulled (run: ollama pull {model})", "version": version}
    return {"available": True, "version": version, "model": model}


PROMPT = """You answer questions about a financial-analysis project using only numbered evidence passages.

Example of the required format (a different topic):
Evidence:
[1] (notes.md - Liquidity)
Banks hold cash buffers so they can meet withdrawals during stress.
[2] (notes.md - Leverage)
Leverage is debt divided by equity.
Question: Why do banks hold cash buffers?
Answer: Banks hold cash buffers so they can meet withdrawals during stress [1].

Evidence:
{context}

Question: {question}

Rules:
- Use only facts stated in the evidence above. Do not use outside knowledge.
- End every sentence with the number of the passage that supports it, like [1] or [2][3].
- Copy numbers, dates, and names exactly as written in the evidence.
- Answer in at most {max_sentences} short sentences.
- If the evidence does not answer the question, reply with exactly this sentence and nothing else:
{insufficient}

Answer:"""


def ollama_generator(model: str, url: str = "http://localhost:11434", timeout: float = 600.0,
                     options: dict | None = None, max_sentences: int = 3, post=None):
    """A generator using a local Ollama server; returns (answer, info). `post(url, payload)` can be injected for tests."""
    def default_post(endpoint, payload):
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    send = post or default_post
    settings = {"temperature": 0, "seed": 42, **(options or {})}

    def generate(question: str, evidence: list[dict]) -> tuple[str, dict]:
        context = "\n\n".join(f"[{e['rank']}] ({e['source']} - {e['section']})\n{e['text']}" for e in evidence)
        prompt = PROMPT.format(max_sentences=max_sentences, insufficient=INSUFFICIENT, context=context, question=question)
        reply = send(f"{url}/api/generate", {"model": model, "prompt": prompt, "stream": False, "options": settings})
        total = reply.get("total_duration")
        info = {"model": model, "options": settings, "prompt_tokens": reply.get("prompt_eval_count"),
                "output_tokens": reply.get("eval_count"), "seconds": round(total / 1e9, 2) if total else None}
        return str(reply.get("response", "")).strip(), info
    return generate
