"""Read-only tools the workflow calls (docs/decisions.md, D17).

Both tools reuse the existing pipelines and their frozen outputs; nothing is
refitted, re-selected, or written. Every file is verified against a recorded
SHA-256 before use, and joblib models are loaded only after that check
(loading a joblib file can run code).
"""

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fincrisis.features import market_features as mf


class ToolError(Exception):
    """A required frozen output is missing or does not match its recorded checksum."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    if not Path(path).exists():
        raise ToolError(f"missing {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_model(directory: Path, name: str) -> object:
    meta = _json(directory / "model_metadata.json")
    entry = meta["models"].get(name)
    if entry is None:
        raise ToolError(f"model {name} is not listed in {directory / 'model_metadata.json'}")
    path = directory / entry["file"]
    if not path.exists() or _sha256(path) != entry["sha256"]:
        raise ToolError(f"{path} is missing or does not match its recorded checksum; refusing to load it")
    return joblib.load(path)


def _close(path: Path) -> pd.Series:
    frame = pd.read_csv(path, float_precision="round_trip")
    return pd.Series(frame["Close"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(frame["Date"])))


def assess_market(root: Path, paths: dict, stale_after_days: int, today: date | None = None) -> dict:
    """Score the latest snapshot date with the frozen market-stress models (D10-D13)."""
    root = Path(root)
    raw_dir = root / paths["raw_data"] / "market"
    report_dir = root / paths["data_reports"] / "market_stress" / "models"
    model_dir = root / paths["models"] / "market_stress"
    dataset_path = root / paths["processed_data"] / "market_stress" / "dataset.csv"

    raw_meta = _json(raw_dir / "metadata.json")
    for series in raw_meta["series"].values():
        if _sha256(raw_dir / series["file"]) != series["sha256"]:
            raise ToolError(f"raw snapshot {series['file']} does not match its recorded checksum")
    selection_path = report_dir / "selection.json"
    selection, test = _json(selection_path), _json(report_dir / "test_metrics.json")
    if test.get("selection_sha256") != _sha256(selection_path):
        raise ToolError("selection.json differs from the one the frozen test evaluation used")

    features, _ = mf.build_features(_close(raw_dir / raw_meta["series"]["gspc"]["file"]),
                                    _close(raw_dir / raw_meta["series"]["vix"]["file"]))
    latest = features.iloc[[-1]]
    if latest.isna().any(axis=None):
        raise ToolError("latest snapshot row has missing features")
    as_of = latest.index[0].date()

    def score(name: str) -> float:
        kind = selection["models"][name]
        if kind == "rule":
            return float(latest[name.removeprefix("rule_")].iloc[0])
        return float(_load_model(model_dir, name).predict_proba(latest[mf.FEATURES])[:, 1][0])

    def describe(name: str) -> dict:
        value, threshold = score(name), selection["thresholds_max_f1"][name]
        m = test["models"][name]
        return {"name": name, "kind": selection["models"][name], "score": value, "threshold": threshold,
                "elevated": bool(value >= threshold),
                "test_average_precision": m["average_precision"], "test_roc_auc": m["roc_auc"],
                "test_precision_at_threshold": m["at_max_f1_threshold"]["precision"],
                "test_recall_at_threshold": m["at_max_f1_threshold"]["recall"]}

    primary = describe(selection["primary_model"]["name"])
    history = pd.read_csv(dataset_path, usecols=["date", "eligible", primary["name"].removeprefix("rule_")]) \
        if primary["kind"] == "rule" and dataset_path.exists() else None
    if history is not None:
        values = history.loc[history["eligible"], primary["name"].removeprefix("rule_")].to_numpy(dtype=float)
        primary["historical_percentile"] = float(np.mean(values <= primary["score"]))
        primary["history_rows"] = int(len(values))
    secondary = describe("xgboost") if "xgboost" in selection["models"] else None

    today = today or datetime.now(timezone.utc).date()
    age = (today - as_of).days
    warnings = []
    if age > stale_after_days:
        warnings.append(f"market snapshot is {age} days old (as of {as_of}); refresh it before relying on the signal")
    return {
        "as_of": str(as_of), "snapshot_age_days": age, "primary": primary, "secondary": secondary,
        "test_prevalence": test["prevalence"], "test_period": test["test"],
        "horizon_sessions": selection["horizon"], "warnings": warnings,
        "label_definition": "the S&P 500 closes at least 5% below its running high within the next 20 sessions (D2)",
    }


def context_answer(entries: list[dict], answer_fn) -> dict:
    """Cited explanation built from one best sentence per configured context entry (D17).

    `answer_fn(question, scope)` is the RAG `answer()` restricted to `scope` and
    asked for one sentence. Entries with no passage above the relevance threshold
    are skipped. Citation markers are renumbered 1..n across entries.
    """
    sentences, citations, warnings = [], [], []
    for entry in entries:
        result = answer_fn(entry["question"], [{"source": entry["source"], "section": entry["section"]}])
        warnings += result.get("warnings", [])
        if result["status"] != "answered" or not result["citations"]:
            continue
        mapping = {}
        for c in result["citations"]:
            mapping[c["marker"]] = len(citations) + 1
            citations.append({**c, "marker": mapping[c["marker"]]})
        sentences.append(re.sub(r"\[(\d+)\]", lambda m: f"[{mapping.get(int(m.group(1)), m.group(1))}]", result["answer"]))
    if not sentences:
        return {"status": "insufficient_evidence", "answer": "", "citations": [], "warnings": warnings}
    return {"status": "answered", "answer": " ".join(sentences), "citations": citations, "warnings": warnings}


def distress_summary(root: Path, paths: dict) -> dict:
    """Frozen held-out results of the company-distress baselines (D9); companies are not scored here."""
    root = Path(root)
    test = _json(root / paths["data_reports"] / "taiwan_bankruptcy" / "baselines" / "test_metrics.json")
    meta = _json(root / paths["models"] / "taiwan_bankruptcy" / "model_metadata.json")
    model_dir = root / paths["models"] / "taiwan_bankruptcy"
    for name, entry in meta["models"].items():
        if _sha256(model_dir / entry["file"]) != entry["sha256"]:
            raise ToolError(f"{entry['file']} does not match its recorded checksum")
    families = {}
    for family, r in test["results"].items():
        ci = r["bootstrap_95ci_at_max_f1"]
        families[family] = {"config": r["config"], "average_precision": r["average_precision"],
                            "average_precision_95ci": ci["average_precision"], "roc_auc": r["roc_auc"]}
    return {"test_rows": test["test_rows"], "test_label_1": test["test_label_1"], "families": families,
            "label_caveat": test["label_caveat"]}
