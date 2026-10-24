"""Disclosure texts shown by the dashboard (docs/decisions.md, D18).

Numbers come from the frozen outputs passed in, not from constants, so the
texts cannot drift from the results they describe. The research-only notice
is the workflow's disclaimer (D17), so the CLI and the dashboard say the same.
"""

from dataclasses import dataclass
from datetime import date

from fincrisis.agents.graph import DISCLAIMER
from fincrisis.dashboard.loaders import ArtifactError, snapshot_age_days

RESEARCH_ONLY = DISCLAIMER


@dataclass(frozen=True)
class Notice:
    level: str  # "info", "warning" or "error"
    text: str


def snapshot_notice(snapshot: dict, stale_after_days: int, today: date | None = None) -> Notice:
    """As-of date of the frozen market snapshot; a warning once it is older than `stale_after_days`."""
    age = snapshot_age_days(snapshot["as_of"], today)
    text = (f"Market data: frozen snapshot as of {snapshot['as_of']} (last completed trading day; downloaded "
            f"{snapshot['downloaded_at_utc'][:10]}). Source: {snapshot['source']}. The dashboard never downloads data.")
    if age > stale_after_days:
        return Notice("warning", f"{text} The snapshot is {age} days old, so the market figures are not current.")
    return Notice("info", text)


def artifact_notice(error: ArtifactError) -> Notice:
    return Notice("error", f"{error} {error.remedy or ''}".strip())


def market_stress_limitations(results: dict) -> list[str]:
    """Limitations of the frozen market-stress models (D2, D13), from their selection and test results."""
    selection, test = results["selection"], results["test_metrics"]
    name = results["primary_model"]
    primary = test["models"][name]
    lo, hi = test["bootstrap"]["primary"]["average_precision"][name]
    at = primary["at_max_f1_threshold"]
    period = test["test"]
    alarms = ("most alerts are false alarms" if at["precision"] < 0.5 else "many alerts are false alarms")
    return [
        selection["caveat"],
        f"The primary model `{name}` was chosen on 2000-2014 walk-forward validation only, by the rule: "
        f"{selection['primary_model']['rule']}. Test results did not change that choice.",
        f"Held-out test {period['first_date']} to {period['last_date']}: average precision {primary['average_precision']:.3f} "
        f"(95% block-bootstrap interval {lo:.3f} to {hi:.3f}) against a base rate of {test['prevalence']:.3f}. At the "
        f"frozen threshold, precision is {at['precision']:.2f} and recall {at['recall']:.2f}, so {alarms}.",
        "Scores rank risk; none is the probability of a drawdown. The logistic-regression and XGBoost scores are not "
        "calibrated (see the calibration table).",
        "Label-1 days overlap in 20-session windows and cluster in a few episodes, so the effective sample is far "
        "smaller than the row count and the intervals are approximate.",
        "Yahoo Finance data is unofficial. Results describe 1990-2026 history and need not hold in the future.",
    ]


def distress_limitations(results: dict) -> list[str]:
    """Limitations of the company-distress baselines (D4, D9, docs/model_card.md), from their test results."""
    test = results["test_metrics"]
    lines = [test["label_caveat"]]
    for pair, d in test["paired_average_precision_differences"].items():
        lo, hi = d["interval"]
        if "dummy" not in pair and lo <= 0 <= hi:
            lines.append(f"The test set does not distinguish the models in `{pair}`: the 95% interval for their "
                         f"difference in average precision ({lo:+.3f} to {hi:+.3f}) includes zero.")
    lines += [
        "Probabilities are not calibrated: read the scores as a ranking, not as likelihoods (model card, section 9).",
        "The data covers Taiwanese companies from 1999-2009 only; applying the models elsewhere could systematically "
        "misjudge companies (model card, section 11).",
        "The dashboard does not score new companies: the models need all 91 financial-ratio inputs of the UCI "
        "dataset, preprocessed as in D8.",
    ]
    return lines


def rag_limitations(rag_settings: dict) -> list[str]:
    """Limitations of the document assistant (D14-D16)."""
    model = rag_settings["ollama"]["model"]
    lines = [
        f"Answers quote the indexed project documents only. If no passage reaches a relevance of "
        f"{rag_settings['min_relevance']:.2f}, the answer is 'insufficient evidence'.",
        "Extractive answers copy retrieved sentences; they can still quote a passage that does not answer the question.",
        "Generated answers are accepted only if they pass a word-overlap grounding check; the check is a heuristic and "
        "does not prove an answer correct.",
        "Retrieval was evaluated on small question sets (D14, D16).",
    ]
    if model.lower().startswith("qwen"):
        lines.append(f"The optional local model {model} is under the Qwen Research License: non-commercial use only.")
    return lines


def market_data_limitations(snapshot: dict) -> list[str]:
    """Limitations of the market history shown (D10, D11)."""
    lines = [
        f"Source: {snapshot['source']}. Prices are not checked against an official record.",
        f"The snapshot is frozen at {snapshot['as_of']} because the market-stress evaluation was run on it; a newer "
        "snapshot would be a separate, reviewed step.",
        "The VIX shown for a day is that day's close. The models use the previous session's close (D11), because the "
        "VIX closes after the S&P 500.",
        "These are historical observations, not forecasts.",
    ]
    return lines + [f"Snapshot check: {warning}" for warning in snapshot.get("warnings", [])]


def workflow_limitations() -> list[str]:
    """Limitations of the LangGraph workflow (D17)."""
    return [
        "Routing uses fixed keyword rules, not a language model, so paraphrased questions can take a different branch "
        "(for example, 'Is the market crashing?' goes to the documents).",
        "Market and distress answers report the frozen evaluated models and are explained with cited project "
        "documents; nothing is re-estimated.",
        "Requests for investment or trading advice are declined without running any tool.",
        "No question set measures the workflow's answers; their quality is that of the components it reuses.",
    ]
