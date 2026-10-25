"""Company Distress: frozen held-out results of the company-distress baselines (D4, D8, D9)."""

import pandas as pd
import streamlit as st

import ui
from fincrisis.dashboard import charts

ui.page_header("Company Distress",
               "This page presents the historical performance of machine learning models trained to identify "
               "financial distress using company data from Taiwan. Explore their evaluation results and compare "
               "their performance.")

results = ui.distress_results()
if results is not None:
    # A plainer wording of the frozen `label_caveat`, which stays unchanged in the results files.
    st.warning("The models are evaluated on their ability to identify the minority class in the dataset. The exact "
               "meaning of the target labels has not been independently verified, so the results should be "
               "interpreted as classification performance rather than confirmed bankruptcy predictions.",
               icon=":material/help:")
    families = results["test_metrics"]["results"]

    table = pd.DataFrame([{
        "family": charts.label(f), "configuration": r["config"], "average precision": r["average_precision"],
        "AP 95% interval": "{:.3f} to {:.3f}".format(*r["bootstrap_95ci_at_max_f1"]["average_precision"])
        if "bootstrap_95ci_at_max_f1" in r else "",
        "ROC-AUC": r["roc_auc"], "Brier": r["brier"], "frozen threshold": r["at_max_f1_threshold"]["threshold"],
        "precision at threshold": r["at_max_f1_threshold"]["precision"],
        "recall at threshold": r["at_max_f1_threshold"]["recall"]} for f, r in families.items()])
    ui.chart(charts.distress_model_metrics, results, table=table, key="metrics")

    left, right = st.columns(2)
    with left:
        ui.chart(charts.distress_precision_recall, results, key="pr")
    with right:
        family = st.selectbox("Confusion matrix for", [f for f in families if f != "dummy"],
                              format_func=charts.label, key="family")
        at = st.radio("Operating point", ["max_f1", "0.5"], horizontal=True, key="operating_point",
                      format_func=lambda v: {"max_f1": "Frozen max-F1 threshold", "0.5": "0.5"}[v])
        ui.chart(charts.distress_confusion, results, family=family, at=at, key="confusion")

    st.subheader("Cross-validation and feature groups")
    left, right = st.columns(2)
    with left:
        ui.chart(charts.distress_cv_folds, results, table=results["cv_results"], key="cv")
    with right:
        ui.chart(charts.distress_group_importance, results, table=results["permutation_importance"], key="importance",
                 caption="The pipeline saved only this grouped result; per-feature importance is not available.")

    with st.expander("Final report (held-out test)"):
        st.markdown(results["final_report"] or "Report not found.")
