"""AI Research Workflow: the LangGraph workflow over the market signal, distress results and documents (D17)."""

import streamlit as st

import ui
from fincrisis.agents.graph import DISCLAIMER, REFUSAL

ui.page_header("AI Research Workflow",
               "A LangGraph workflow routes each question: the current market-stress signal, the company-distress "
               "results, or the project documents. Requests for investment or trading advice are declined.")

status = ui.index_status()
if not status["ready"]:
    st.warning(f"The document index cannot be used ({status['problem']}). Market and distress answers still work, "
               f"without document context; document questions will report the problem. {status['remedy']}",
               icon=":material/warning:")

use_llm = st.toggle("Let the local model write document answers (Ollama)", key="wf_llm",
                    help="Applies to document questions only; market and distress context is always extractive.")

EXAMPLES = ["What is the current market stress level?", "How well do the bankruptcy models perform?",
            "Which VIX value is used for date t?", "Should I sell my stocks now?"]
st.pills("Examples", EXAMPLES, key="wf_example",
         on_change=lambda: st.session_state.update(wf_text=st.session_state.get("wf_example") or ""))
with st.form("wf_question"):
    question = st.text_input("Question", key="wf_text")
    asked = st.form_submit_button("Run workflow", type="primary")

if asked and question.strip():
    graph, notes = ui.workflow(use_llm, ui.index_fingerprint(), str(ui.DATA_ROOT))
    with st.spinner("Running the workflow..."):
        state = graph.invoke({"question": question.strip()})
    state["warnings"] = [*state.get("warnings", []), *notes]  # as scripts/run_crisis_agent.py does
    state["answer"] = state["answer"].removesuffix(f"\n\n{DISCLAIMER}")  # the page shows no disclaimer
    st.session_state["wf_state"] = state

state = st.session_state.get("wf_state")
if state is not None:
    st.subheader("Answer")
    st.caption(f"Question: {state['question']}")
    if state["answer"] == REFUSAL:
        st.info(state["answer"], icon=":material/block:")
    elif state.get("trace", [])[-1:] == ["report_error"]:
        st.error(state["answer"], icon=":material/error:")
    elif state.get("trace", [])[-1:] == ["abstain"]:
        st.info(state["answer"], icon=":material/search_off:")
    else:
        st.markdown(state["answer"])
    ui.show_citations(state.get("citations", []))
    st.caption("Route: " + " → ".join(state.get("trace", [])))
    for warning in state.get("warnings", []):
        st.warning(warning, icon=":material/warning:")
    with st.expander("Full workflow state"):
        st.json(state, expanded=False)
