"""Financial RAG Assistant: cited answers from the project documents (D14-D16)."""

import streamlit as st

import ui
from fincrisis.rag import answer as ans

ui.page_header("Financial RAG Assistant",
               "Ask questions about the project and get answers based on its documentation, with references to the "
               "sources.")

rag = ui.settings()["rag"]
status = ui.index_status()
if not status["ready"]:
    ui.show_index_problem(status)

use_llm = st.toggle("Use the local AI model to generate answers", key="rag_llm",
                    help=f"Uses {rag['ollama']['model']} on this computer. A generated answer is kept only if it "
                         "passes the grounding check; otherwise the extractive answer is shown.")
if use_llm:
    st.caption(f"{rag['ollama']['model']} is under the Qwen Research License: non-commercial use only. "
               "Generation can take a minute or more on a CPU.")

with st.form("rag_question"):
    question = st.text_input("Question", placeholder="Which VIX value is used for date t?", key="rag_text")
    asked = st.form_submit_button("Ask", disabled=not status["ready"], type="primary")
st.caption("Examples: How is the market-stress target defined? · Are the distress probabilities calibrated? · "
           "Why is VIX lagged by one session?")

if asked and question.strip():
    try:
        store, embeddings, retriever = ui.rag_store(ui.index_fingerprint(), str(ui.DATA_ROOT))
    except Exception as error:  # e.g. the embedding model is not in the local cache
        st.error(f"**The document index could not be opened:** {error}", icon=":material/error:")
        st.stop()
    generator, unavailable = ui.ollama_generator() if use_llm else (None, None)
    with st.spinner("Retrieving passages" + (" and generating with the local model..." if generator else "...")):
        result = ans.answer(question.strip(), store, embeddings, rag["top_k"], rag["min_relevance"],
                            rag["max_answer_sentences"], generator, rag["ollama"]["min_support"], retriever)
    if unavailable:
        result["warnings"].insert(0, f"local model unavailable ({unavailable}); extractive answer returned")
    st.session_state["rag_result"] = result

result = st.session_state.get("rag_result")
if result is not None:
    st.subheader("Answer")
    if result["status"] == "answered":  # citations and evidence stay in the result; the page shows plain text
        st.markdown(ui.hide_citations(result["answer"], result["citations"]))
    else:
        st.info(result["answer"], icon=":material/search_off:")
    method = result["method"]
    if result.get("generation"):
        g = result["generation"]
        method += f" ({g.get('model')}, {g.get('seconds')} s)"
    st.caption(f"Method: {method}")
    for warning in result["warnings"]:
        st.warning(warning, icon=":material/warning:")
