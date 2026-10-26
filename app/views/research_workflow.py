"""AI Research Workflow: the LangGraph workflow over the market signal, distress results and documents (D17)."""

import streamlit as st

import ui
from fincrisis.agents.graph import DISCLAIMER, REFUSAL, answer_method

ui.page_header("AI Research Workflow",
               "Ask a question about market stress, company financial distress, or project documentation. The system "
               "finds the most relevant information to answer you.")

status = ui.index_status()
if not status["ready"]:
    st.warning(f"The document index cannot be used ({status['problem']}). Market and distress answers still work, "
               f"without document context; document questions will report the problem. {status['remedy']}",
               icon=":material/warning:")

use_llm = st.toggle("Use the local AI model to generate document answers", key="wf_llm",
                    help="Applies to document questions only; market and distress context is always extractive.")

EXAMPLES = ["What is the current market stress level?", "How well do the bankruptcy models perform?",
            "Which VIX value is used for date t?", "Should I sell my stocks now?"]
st.pills("Examples", EXAMPLES, key="wf_example",
         on_change=lambda: st.session_state.update(wf_text=st.session_state.get("wf_example") or ""))
with st.form("wf_question"):
    question = st.text_input("Question", key="wf_text")
    asked = st.form_submit_button("Ask", type="primary")

if asked and question.strip():
    graph, notes = ui.workflow(use_llm, ui.index_fingerprint(), str(ui.DATA_ROOT))
    with st.spinner("Running the workflow..."):
        state = graph.invoke({"question": question.strip()})
    if state.get("route") == "documents":  # the model notes concern document questions only
        state["warnings"] = [*state.get("warnings", []), *notes]  # as scripts/run_crisis_agent.py does
    state["answer"] = state["answer"].removesuffix(f"\n\n{DISCLAIMER}")  # the page shows no disclaimer
    state["method"] = answer_method(state, use_llm)  # recorded with the run, so a later toggle does not relabel it
    st.session_state["wf_state"] = state

state = st.session_state.get("wf_state")
if state is not None:
    st.subheader("Answer")
    if state["answer"] == REFUSAL:
        st.info(state["answer"], icon=":material/block:")
    elif state.get("trace", [])[-1:] == ["report_error"]:
        st.error(state["answer"], icon=":material/error:")
    elif state.get("trace", [])[-1:] == ["abstain"]:
        st.info(state["answer"], icon=":material/search_off:")
    else:  # citations stay in the state; the page shows plain text
        st.markdown(ui.hide_citations(state["answer"], state.get("citations", [])))
    if state.get("method"):
        st.caption(f"Method: {state['method']['text']}")
    for warning in state.get("warnings", []):
        st.warning(warning, icon=":material/warning:")
