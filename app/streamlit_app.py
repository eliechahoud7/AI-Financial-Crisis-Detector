"""AI Financial Crisis Detector: local Streamlit dashboard (docs/decisions.md, D21).

Launch from the project root, so `.streamlit/config.toml` (localhost only, no
usage statistics) is used:

    .\\.venv\\Scripts\\python.exe -m streamlit run app\\streamlit_app.py

The dashboard only reads the frozen outputs; it never downloads data, trains
models, regenerates predictions or rebuilds the RAG index.
"""

import streamlit as st

import ui

st.set_page_config(page_title="AI Financial Crisis Detector", page_icon=":material/monitoring:", layout="wide")

PAGES = [
    st.Page("views/market_dashboard.py", title="Market Dashboard", icon=":material/show_chart:", default=True),
    st.Page("views/market_stress.py", title="Market Stress Analysis", icon=":material/warning:"),
    st.Page("views/company_distress.py", title="Company Distress", icon=":material/apartment:"),
    st.Page("views/rag_assistant.py", title="Financial RAG Assistant", icon=":material/menu_book:"),
    st.Page("views/research_workflow.py", title="AI Research Workflow", icon=":material/account_tree:"),
]

navigation = st.navigation(PAGES)
ui.sidebar()
navigation.run()
