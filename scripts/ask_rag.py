"""Ask a question about the indexed documents (docs/decisions.md, D14).

Usage:
    python scripts/ask_rag.py "What is the market-stress target?"
    python scripts/ask_rag.py "..." --json          # full result, including retrieved evidence
    python scripts/ask_rag.py "..." --llm ollama    # optional: a local Ollama model writes the answer

By default the answer is extractive: it consists only of sentences from the
retrieved passages, each with a citation [n]. If no passage reaches the
configured relevance threshold, it says the evidence is insufficient.
"""

import argparse
import json
import sys

from fincrisis.rag import answer as ans
from fincrisis.rag import index as idx
from fincrisis.rag import pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--k", type=int, help="number of passages to retrieve (default: rag.top_k)")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--llm", choices=["ollama"], help="use a local Ollama model to write the answer")
    args = parser.parse_args()
    settings = pipeline.load_rag_settings()
    try:
        store, manifest, embeddings = pipeline.open_store(settings)
    except (RuntimeError, idx.RAGIndexError) as error:
        print(f"Failed: {error}")
        return 1
    generator = ans.ollama_generator(settings["ollama_model"]) if args.llm == "ollama" else None
    result = ans.answer(args.question, store, embeddings, args.k or settings["top_k"], settings["min_relevance"],
                        settings["max_answer_sentences"], generator)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    print(result["answer"])
    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f"  [{c['marker']}] {c['source']} - {c['section']} (score {c['score']:.3f})")
    else:
        print(f"\n(best passage score {result['top_score']:.3f}, threshold {settings['min_relevance']})")
    for warning in result["warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
