"""Ask a question about the indexed documents (docs/decisions.md, D14 and D15).

Usage:
    python scripts/ask_rag.py "What is the market-stress target?"
    python scripts/ask_rag.py "..." --llm ollama    # a local Ollama model writes the answer
    python scripts/ask_rag.py "..." --json          # full result, including retrieved evidence

By default the answer is extractive: sentences from the retrieved passages, each
with a citation [n]. With --llm ollama, a local model writes the answer from the
same passages; it is used only if it passes the grounding checks, otherwise the
extractive answer is shown with a warning. If no passage reaches the relevance
threshold, or the model finds the evidence insufficient, the reply says so.
"""

import argparse
import json
import sys

from fincrisis.rag import answer as ans
from fincrisis.rag import index as idx
from fincrisis.rag import pipeline


def make_generator(settings: dict):
    """The Ollama generator, or (None, reason) if the server or model is unavailable."""
    o = settings["ollama"]
    status = ans.ollama_status(o["url"], o["model"])
    if not status["available"]:
        return None, status["reason"]
    return ans.ollama_generator(o["model"], o["url"], o["timeout_seconds"], o["options"],
                                settings["max_answer_sentences"]), None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--k", type=int, help="number of passages to retrieve (default: rag.top_k)")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--llm", choices=["ollama"], help="use the local Ollama model to write the answer")
    args = parser.parse_args()
    settings = pipeline.load_rag_settings()
    try:
        store, manifest, embeddings = pipeline.open_store(settings)
    except (RuntimeError, idx.RAGIndexError) as error:
        print(f"Failed: {error}")
        return 1
    generator, unavailable = make_generator(settings) if args.llm == "ollama" else (None, None)
    result = ans.answer(args.question, store, embeddings, args.k or settings["top_k"], settings["min_relevance"],
                        settings["max_answer_sentences"], generator, settings["ollama"]["min_support"],
                        pipeline.make_retriever(store, settings))
    if unavailable:
        result["warnings"].insert(0, f"local model unavailable ({unavailable}); extractive answer returned")
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    print(result["answer"])
    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f"  [{c['marker']}] {c['source']} - {c['section']} (score {c['score']:.3f})")
    elif result["method"] == "none":
        print(f"\n(best passage score {result['top_score']:.3f}, threshold {settings['min_relevance']})")
    print(f"\nMethod: {result['method']}" + (f" ({result['generation']['model']}, {result['generation']['seconds']} s)"
                                           if result.get("generation") else ""))
    for warning in result["warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
