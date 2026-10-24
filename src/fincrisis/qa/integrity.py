"""Read-only integrity verification of every frozen input and output (docs/decisions.md, D22).

Each check compares files with checksums that were recorded independently of
the files themselves, so nothing vouches for itself:

- `config/frozen_artifacts.json` (D19): the frozen evaluation outputs, saved
  models, the market-stress dataset, and the RAG validation questions;
- the raw market snapshot against `raw_sha256` in the frozen `selection.json`
  (itself covered by the manifest), and against its own metadata;
- the Taiwan raw and processed data against the `input` checksums recorded in
  the frozen company-distress `test_metrics.json` (covered by the manifest);
- the RAG index against its own manifest, with the configured model and
  settings, and against the current documents (a stale index is reported).

Nothing is written, loaded as a model, or recomputed apart from file hashes.
"""

from dataclasses import dataclass
from pathlib import Path

from fincrisis.agents import tools
from fincrisis.dashboard import loaders


@dataclass(frozen=True)
class Check:
    group: str
    name: str
    status: str  # "ok", "mismatch" or "missing"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _compare(group: str, name: str, path: Path, expected: str | None) -> Check:
    if expected is None:
        return Check(group, name, "missing", "no recorded checksum")
    if not path.exists():
        return Check(group, name, "missing", f"{path} not found")
    actual = tools.sha256_file(path)
    if actual != expected:
        return Check(group, name, "mismatch", f"sha256 {actual[:12]}..., recorded {expected[:12]}...")
    return Check(group, name, "ok")


def _read(path: Path) -> dict | None:
    try:
        return tools.read_json(path)
    except (tools.ToolError, ValueError):
        return None


def manifest_checks(root: Path) -> list[Check]:
    """Every file listed in the version-controlled manifest (D19), checked under `root`."""
    try:
        recorded = loaders.frozen_checksums()
    except loaders.ArtifactError as error:
        return [Check("manifest", "config/frozen_artifacts.json", "missing", str(error))]
    return [_compare("manifest", path, Path(root) / path, digest) for path, digest in recorded.items()]


def market_snapshot_checks(root: Path, paths: dict) -> list[Check]:
    """Raw S&P 500 and VIX files against the frozen selection's record and the snapshot's own metadata (D10, D13)."""
    raw_dir = Path(root) / paths["raw_data"] / "market"
    selection_path = Path(root) / paths["data_reports"] / "market_stress" / "models" / "selection.json"
    selection, metadata = _read(selection_path), _read(raw_dir / "metadata.json")
    if selection is None:
        return [Check("market snapshot", "selection.json", "missing", f"{selection_path} not readable")]
    checks = []
    for name, digest in selection["raw_sha256"].items():
        checks.append(_compare("market snapshot", f"{name} (frozen selection record)", raw_dir / name, digest))
        own = None if metadata is None else next(
            (s["sha256"] for s in metadata["series"].values() if s["file"] == name), None)
        checks.append(_compare("market snapshot", f"{name} (snapshot metadata)", raw_dir / name, own))
    return checks


def distress_data_checks(root: Path, paths: dict) -> list[Check]:
    """Taiwan raw and processed data against the checksums in the frozen test results (D8, D9)."""
    metrics_path = Path(root) / paths["data_reports"] / "taiwan_bankruptcy" / "baselines" / "test_metrics.json"
    metrics = _read(metrics_path)
    if metrics is None:
        return [Check("distress data", "test_metrics.json", "missing", f"{metrics_path} not readable")]
    recorded = metrics["input"]
    raw = Path(root) / paths["raw_data"] / "taiwan_bankruptcy" / "data.csv"
    interim = Path(root) / paths["interim_data"] / "taiwan_bankruptcy"
    checks = [_compare("distress data", "raw data.csv", raw, recorded.get("raw_sha256"))]
    checks += [_compare("distress data", f"processed {name}", interim / name, digest)
               for name, digest in recorded["processed_sha256"].items()]
    return checks


def rag_index_checks(root: Path, rag_settings: dict) -> list[Check]:
    status = loaders.rag_index_status(rag_settings, root)
    if status["ready"]:
        return [Check("rag index", rag_settings["index_dir"], "ok", f"{status['chunks']} passages")]
    kind = "missing" if "no index" in (status["problem"] or "") else "mismatch"
    return [Check("rag index", rag_settings["index_dir"], kind, f"{status['problem']}. {status['remedy']}")]


def verify_all(root: Path = loaders.PROJECT_ROOT, settings: dict | None = None) -> list[Check]:
    settings = settings or loaders.load_settings()
    paths = settings["paths"]
    return (manifest_checks(root) + market_snapshot_checks(root, paths) + distress_data_checks(root, paths)
            + rag_index_checks(root, settings["rag"]))


def summary(checks: list[Check]) -> str:
    lines, groups = [], dict.fromkeys(c.group for c in checks)
    for group in groups:
        rows = [c for c in checks if c.group == group]
        bad = [c for c in rows if not c.ok]
        lines.append(f"{group}: {len(rows) - len(bad)}/{len(rows)} ok")
        lines += [f"  {c.status.upper()}: {c.name}" + (f" ({c.detail})" if c.detail else "") for c in bad]
    failed = sum(not c.ok for c in checks)
    lines.append("All integrity checks passed." if not failed else f"{failed} integrity check(s) failed.")
    return "\n".join(lines)
