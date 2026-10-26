"""Market Stress Analysis: the frozen market-stress models, their latest score and their evaluation (D2, D13)."""

import pandas as pd
import streamlit as st

import ui
from fincrisis.dashboard import charts

ui.page_header("Market Stress Analysis",
               "Assess market conditions using machine learning models and historical market data.")

results = ui.stress_results()
signal = ui.current_signal()

st.subheader("Latest snapshot day")
if signal is not None:
    p, s = signal["primary"], signal.get("secondary")
    state = ("flags **elevated market stress**: its latest score is at or above" if p["elevated"]
             else "assesses market conditions as **normal**: its latest score is below")
    cols = st.columns(3)
    cols[0].metric(f"{charts.label(p['name'])} score, {signal['as_of']}", f"{p['score']:.4f}",
                   f"threshold {p['threshold']:.4f}", delta_color="off")
    if "historical_percentile" in p:
        cols[1].metric("Historical percentile", f"{p['historical_percentile']:.0%}",
                       f"of {p['history_rows']:,} trading days", delta_color="off")
    if s:  # D35: a fitted model's frozen threshold is not transferable, so its score is shown without a verdict
        cols[2].metric(f"{charts.label(s['name'])} score (uncalibrated)", f"{s['score']:.3f}",
                       "for comparison only; no decision threshold", delta_color="off")
    st.markdown(f"The primary model currently {state} the level it uses to raise a stress signal. It is built to "
                "give early warning of market stress, and in historical testing it flagged "
                f"{p['test_recall_at_threshold']:.0%} of the trading days followed within about a month by a fall "
                "of at least 5% from the market's high.")
    for warning in signal["warnings"]:
        st.warning(warning, icon=":material/schedule:")

if results is not None:
    models = list(results["thresholds"])
    primary = results["primary_model"]

    st.subheader("Scores over time")
    model = st.selectbox("Model", models, index=models.index(primary), format_func=charts.label, key="timeline_model")
    ui.chart(charts.stress_score_timeline, results, model=model, current=signal, key="timeline",
             caption="Validation scores come from 15 walk-forward models (one per year, trained only on earlier "
                     "data); test scores from the final model trained on 1990-2014.")

    st.subheader("Evaluation")
    stage = st.radio("Results from", ["test", "validation"], horizontal=True, key="stage",
                     format_func=lambda v: {"test": "Held-out test (2015-2026)",
                                            "validation": "Walk-forward validation (2000-2014)"}[v])
    metrics = results["test_metrics" if stage == "test" else "cv_metrics"]
    intervals = metrics.get("bootstrap", {}).get("primary", {}).get("average_precision", {})
    table = pd.DataFrame([{
        "model": charts.label(name), "average precision": m["average_precision"],
        "95% interval": "{:.3f} to {:.3f}".format(*intervals[name]) if name in intervals else "",
        "ROC-AUC": m["roc_auc"], "frozen threshold": results["thresholds"].get(name),
        "precision at threshold": m["at_max_f1_threshold"]["precision"],
        "recall at threshold": m["at_max_f1_threshold"]["recall"]} for name, m in metrics["models"].items()])
    ui.chart(charts.stress_model_metrics, results, stage, table=table, key="metrics")

    left, right = st.columns(2)
    with left:
        per_year = pd.DataFrame([{"model": charts.label(name), **row} for name, m in metrics["models"].items()
                                 for row in (m.get("per_year") or [])])
        ui.chart(charts.stress_per_year, results, stage, table=per_year, key="per_year")
    with right:
        ui.chart(charts.stress_calibration, results, stage, key="calibration",
                 caption="Rule scores are not probabilities, so they have no calibration curve.")

    if stage == "test":
        left, right = st.columns(2)
        with left:
            ui.chart(charts.stress_precision_recall, results, key="pr")
        with right:
            chosen = st.selectbox("Confusion matrix for", models, index=models.index(primary),
                                  format_func=charts.label, key="confusion_model")
            ui.chart(charts.stress_confusion, results, model=chosen, key="confusion")
    else:
        st.caption("Precision-recall curves and confusion matrices are shown for the held-out test only.")

    with st.expander("Final report (held-out test)"):
        st.markdown(results["final_report"] or "Report not found.")
    with st.expander("Validation report"):
        st.markdown(results["cv_report"] or "Report not found.")
