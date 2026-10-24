"""Read-only, integrity-checked loaders for the dashboard (docs/decisions.md, D18).

Every loader reads frozen outputs only; nothing is written, refitted or
downloaded. Each one checks what the pipelines recorded before returning
anything:

- the raw market snapshot against the checksums in its metadata.json (D10);
- the market-stress `selection.json` against the checksum that the frozen
  test evaluation and the saved models record, and the saved model files
  against `model_metadata.json` (D13);
- every frozen output the pipelines did not checksum themselves (prediction
  files, metrics, reports, model metadata, the dataset) against
  `config/frozen_artifacts.json`, a version-controlled manifest whose values
  come from hash records taken right after each frozen stage ran (D19);
- the prediction files, in addition, by recomputing the recorded average
  precision and label counts from them;
- the RAG index against its manifest, without loading the embedding model.

A missing or inconsistent artifact raises `ArtifactError`, whose `remedy`
says how to restore it, so a page can show the problem instead of crashing.
"""

from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from fincrisis.agents import tools
from fincrisis.data.market import MarketDataError, load_market_dataset
from fincrisis.features import market_features as mf
from fincrisis.rag import index as idx
from fincrisis.rag import pipeline

PROJECT_ROOT = pipeline.PROJECT_ROOT
FROZEN_MANIFEST = PROJECT_ROOT / "config" / "frozen_artifacts.json"  # read from the code checkout, never from `root`
TOLERANCE = 1e-9  # recomputed metrics must match the recorded ones to this precision

REMEDIES = {
    "settings": "Restore config/settings.yaml and config/frozen_artifacts.json from Git.",
    "snapshot": ("Restore the frozen snapshot in data/raw/market/. Running scripts/download_market_data.py would create "
                 "a new snapshot that no longer matches the frozen evaluation."),
    "dataset": "Run scripts/build_market_dataset.py; it rebuilds the dataset from the frozen snapshot without fitting.",
    "market_stress": ("Restore the frozen outputs in data/reports/market_stress/models/ and artifacts/models/market_stress/. "
                      "Re-running scripts/train_market_stress.py would create a new evaluation, not the frozen one."),
    "distress": ("Restore the frozen outputs in data/reports/taiwan_bankruptcy/baselines/ and "
                 "artifacts/models/taiwan_bankruptcy/ (see docs/model_card.md, section 12)."),
    "rag_index": "Run scripts/build_rag_index.py --force (add --allow-download the first time).",
}


class ArtifactError(Exception):
    """A required artifact is missing, altered, or inconsistent with what the pipeline recorded."""

    def __init__(self, message: str, remedy: str | None = None):
        super().__init__(message)
        self.remedy = remedy


@contextmanager
def _artifact(kind: str):
    """Turn any read or integrity failure into an ArtifactError with the remedy for `kind`."""
    try:
        yield
    except ArtifactError as error:
        error.remedy = error.remedy or REMEDIES[kind]
        raise
    except (tools.ToolError, MarketDataError, idx.RAGIndexError, OSError, ValueError, KeyError, TypeError) as error:
        raise ArtifactError(f"{type(error).__name__}: {error}", REMEDIES[kind]) from error


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _read_text(path: Path) -> str | None:
    return Path(path).read_text(encoding="utf-8") if Path(path).exists() else None


def _check_average_precision(frame: pd.DataFrame, recorded: dict, columns: dict, label: str, where: str) -> None:
    """Recompute average precision for each model from saved predictions and compare it with the recorded value."""
    for name, column in columns.items():
        _check(column in frame.columns, f"{where}: no prediction column {column!r}")
        value = float(average_precision_score(frame[label], frame[column]))
        _check(abs(value - recorded[name]) <= TOLERANCE,
               f"{where}: average precision of {name} is {value:.12f}, but {recorded[name]:.12f} was recorded")


def frozen_checksums() -> dict[str, str]:
    """Project-relative path -> recorded SHA-256 of the frozen outputs (D19)."""
    with _artifact("settings"):
        return tools.read_json(FROZEN_MANIFEST)["files"]


def _verify_frozen(root: Path, *directories: Path) -> None:
    """Check every manifest entry in `directories` (relative to `root`); each directory must have entries."""
    recorded = frozen_checksums()
    for directory in directories:
        prefix = Path(directory).as_posix().rstrip("/") + "/"
        entries = {path: digest for path, digest in recorded.items() if path.startswith(prefix) and "/" not in path[len(prefix):]}
        _check(bool(entries), f"config/frozen_artifacts.json records no checksums in {prefix}")
        for path, digest in entries.items():
            file = Path(root) / path
            _check(file.exists(), f"missing {path}")
            _check(tools.sha256_file(file) == digest,
                   f"{path} does not match its checksum in config/frozen_artifacts.json")


def verify_frozen_outputs(root: Path, paths: dict, component: str) -> None:
    """Check one component's frozen outputs against the manifest: "market_stress" or "distress" (D19)."""
    directories = {
        "market_stress": (Path(paths["data_reports"]) / "market_stress" / "models", Path(paths["models"]) / "market_stress",
                          Path(paths["processed_data"]) / "market_stress"),
        "distress": (Path(paths["data_reports"]) / "taiwan_bankruptcy" / "baselines",
                     Path(paths["models"]) / "taiwan_bankruptcy"),
    }[component]
    with _artifact(component):
        _verify_frozen(root, *directories)


def with_frozen_checks(agent_tools, root: Path, paths: dict):
    """The workflow's tools, with the manifest checked before the market and distress tools run (D19, D21).

    `tools.assess_market` and `tools.distress_summary` check only the checksums the pipelines recorded; this
    adds the manifest check the dashboard loaders make, so the workflow page reports altered outputs too.
    A failed check raises `ArtifactError`, which the workflow reports as an error (market) or as
    unavailable results (distress).
    """
    from dataclasses import replace

    def checked(component, tool):
        def run():
            verify_frozen_outputs(root, paths, component)
            return tool()
        return run
    return replace(agent_tools, assess_market=checked("market_stress", agent_tools.assess_market),
                   distress_summary=checked("distress", agent_tools.distress_summary))


def load_settings(root: Path = PROJECT_ROOT) -> dict:
    with _artifact("settings"):
        return yaml.safe_load((Path(root) / "config" / "settings.yaml").read_text(encoding="utf-8"))


def market_snapshot(root: Path, paths: dict) -> dict:
    """Closing prices of the frozen S&P 500 and VIX snapshot, after verifying their checksums (D10)."""
    raw_dir = Path(root) / paths["raw_data"] / "market"
    with _artifact("snapshot"):
        meta = tools.verify_raw_snapshot(raw_dir)
        spx = tools.read_close(raw_dir / meta["series"]["gspc"]["file"])
        vix = tools.read_close(raw_dir / meta["series"]["vix"]["file"])
        return {
            "spx": spx, "vix": vix,
            "as_of": date.fromisoformat(meta["last_completed_trading_date_in_all_series"]),
            "downloaded_at_utc": meta["downloaded_at_utc"], "source": meta["source"],
            "warnings": list(meta.get("warnings", [])),
            "series": {name: {k: s[k] for k in ("ticker", "rows", "first_date", "last_date")}
                       for name, s in meta["series"].items()},
        }


def market_features(snapshot: dict) -> tuple[pd.DataFrame, dict]:
    """The 16 backward-looking features for every snapshot date, as the pipeline computes them (D11, D12)."""
    with _artifact("snapshot"):
        return mf.build_features(snapshot["spx"], snapshot["vix"])


def market_dataset(root: Path, paths: dict) -> tuple[pd.DataFrame, dict]:
    """The labelled market-stress dataset, after its checksum check (D12). The last 20 sessions have no label."""
    directory = Path(paths["processed_data"]) / "market_stress"
    with _artifact("dataset"):
        _verify_frozen(root, directory)
        return load_market_dataset(Path(root) / directory)


def market_stress_results(root: Path, paths: dict) -> dict:
    """Frozen market-stress selection, validation and test results, with saved predictions (D13)."""
    report_dir = Path(root) / paths["data_reports"] / "market_stress" / "models"
    model_dir = Path(root) / paths["models"] / "market_stress"
    with _artifact("market_stress"):
        _verify_frozen(root, Path(paths["data_reports"]) / "market_stress" / "models", Path(paths["models"]) / "market_stress")
        selection_path = report_dir / "selection.json"
        selection, test = tools.read_json(selection_path), tools.read_json(report_dir / "test_metrics.json")
        cv, models = tools.read_json(report_dir / "cv_metrics.json"), tools.read_json(model_dir / "model_metadata.json")
        selection_sha = tools.sha256_file(selection_path)
        _check(test["selection_sha256"] == selection_sha == models["selection_sha256"],
               "selection.json differs from the one the frozen test evaluation and saved models used")
        _check(selection["dataset_sha256"] == test["dataset_sha256"] == models["dataset_sha256"],
               "selection, test results and saved models refer to different datasets")
        for name, entry in models["models"].items():
            path = model_dir / entry["file"]
            _check(path.exists() and tools.sha256_file(path) == entry["sha256"],
                   f"{path} is missing or does not match its recorded checksum")

        test_predictions = pd.read_csv(report_dir / "test_predictions.csv", parse_dates=["date"],
                                       float_precision="round_trip")
        _check(len(test_predictions) == test["rows"] and int(test_predictions["label"].sum()) == test["label_1"],
               "test_predictions.csv row or label counts differ from test_metrics.json")
        _check_average_precision(test_predictions, {n: m["average_precision"] for n, m in test["models"].items()},
                                 {n: n for n in test["models"]}, "label", "test_predictions.csv")
        oof_predictions = pd.read_csv(report_dir / "oof_predictions.csv", parse_dates=["date"],
                                      float_precision="round_trip")
        _check(len(oof_predictions) == cv["rows"] and int(oof_predictions["label"].sum()) == cv["label_1"],
               "oof_predictions.csv row or label counts differ from cv_metrics.json")
        _check_average_precision(oof_predictions, {n: m["average_precision"] for n, m in cv["models"].items()},
                                 {n: n for n in cv["models"]}, "label", "oof_predictions.csv")
        return {
            "selection": selection, "test_metrics": test, "cv_metrics": cv,
            "test_predictions": test_predictions, "oof_predictions": oof_predictions,
            "primary_model": selection["primary_model"]["name"],
            "thresholds": selection["thresholds_max_f1"],
            "final_report": _read_text(report_dir / "final_report.md"),
            "cv_report": _read_text(report_dir / "cv_report.md"),
        }


def current_market_signal(root: Path, paths: dict, stale_after_days: int, today: date | None = None) -> dict:
    """The frozen models' scores for the latest snapshot date (the workflow's `assess_market`, D17)."""
    verify_frozen_outputs(root, paths, "market_stress")
    with _artifact("market_stress"):
        return tools.assess_market(root, paths, stale_after_days, today)


def distress_results(root: Path, paths: dict) -> dict:
    """Frozen company-distress test results, cross-validation results and saved test predictions (D9)."""
    report_dir = Path(root) / paths["data_reports"] / "taiwan_bankruptcy" / "baselines"
    verify_frozen_outputs(root, paths, "distress")
    with _artifact("distress"):
        summary = tools.distress_summary(root, paths)  # verifies the saved model checksums
        test = tools.read_json(report_dir / "test_metrics.json")
        predictions = pd.read_csv(report_dir / "test_predictions.csv", float_precision="round_trip")
        _check(len(predictions) == test["test_rows"] and int(predictions["label"].sum()) == test["test_label_1"],
               "test_predictions.csv row or label counts differ from test_metrics.json")
        families = test["results"]
        _check_average_precision(predictions, {f: r["average_precision"] for f, r in families.items()},
                                 {f: r["config"] for f, r in families.items()}, "label", "test_predictions.csv")
        return {
            "summary": summary, "test_metrics": test,
            "selection": tools.read_json(report_dir / "selection.json"),
            "test_predictions": predictions,
            "cv_results": pd.read_csv(report_dir / "cv_results.csv"),
            "permutation_importance": pd.read_csv(report_dir / "permutation_importance.csv"),
            "label_caveat": test["label_caveat"],
            "final_report": _read_text(report_dir / "final_report.md"),
        }


def rag_index_status(rag_settings: dict, root: Path = PROJECT_ROOT) -> dict:
    """Whether the RAG index can be used, checked from its manifest without loading the embedding model (D14).

    Returns {"ready", "problem", "remedy", "built_at_utc", "chunks", "embedding_model"}.
    """
    index_dir = Path(root) / rag_settings["index_dir"]
    status = {"ready": False, "problem": None, "remedy": REMEDIES["rag_index"],
              "built_at_utc": None, "chunks": None, "embedding_model": rag_settings["embedding_model"]}
    try:
        manifest = idx.read_manifest(index_dir)
        status.update(built_at_utc=manifest.get("built_at_utc"), chunks=manifest["chunks"])
        for name in idx.INDEX_FILES:
            path = index_dir / name
            if not path.exists() or idx.sha256_file(path) != manifest["index_sha256"][name]:
                status["problem"] = f"{name} does not match the index manifest"
                return status
        if manifest["embedding_model"] != rag_settings["embedding_model"]:
            status["problem"] = f"the index was built with {manifest['embedding_model']}"
        elif manifest.get("embed_section_headings", False) != rag_settings["retrieval"]["embed_section_headings"]:
            status["problem"] = "the index was built with a different embed_section_headings setting"
        elif stale := idx.stale_sources(manifest, root, pipeline.source_list(rag_settings, root)):
            status["problem"] = f"the index is out of date for {', '.join(stale)}"
        else:
            status.update(ready=True, remedy=None)
    except (idx.RAGIndexError, OSError, ValueError, KeyError) as error:
        status["problem"] = str(error)
    return status


def snapshot_age_days(as_of: date, today: date | None = None) -> int:
    """Calendar days since the snapshot date, counted from today's UTC date as `assess_market` does."""
    return ((today or datetime.now(timezone.utc).date()) - as_of).days
