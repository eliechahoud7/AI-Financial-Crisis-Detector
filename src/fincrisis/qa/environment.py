"""Installed packages against `requirements-lock.txt` (docs/decisions.md, D28).

The frozen results were produced with the locked versions; a different
scikit-learn, XGBoost or pandas could change scores (D23 would then catch the
effect). This check names the cause: every locked package must be installed at
exactly its locked version. Packages installed but not locked are reported
only.
"""

import re
from importlib import metadata
from pathlib import Path

from fincrisis.dashboard import loaders
from fincrisis.qa.integrity import Check

LOCK = loaders.PROJECT_ROOT / "requirements-lock.txt"


def _name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(path: Path = LOCK) -> dict[str, str]:
    versions = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line and not line.startswith("-"):
            name, version = line.split("==", 1)
            versions[_name(name)] = version.strip()
    return versions


def installed_versions() -> dict[str, str]:
    return {_name(d.metadata["Name"]): d.version for d in metadata.distributions() if d.metadata["Name"]}


def checks(lock: Path = LOCK, installed: dict[str, str] | None = None) -> list[Check]:
    try:
        locked = locked_versions(lock)
    except OSError as error:
        return [Check("environment", "requirements-lock.txt", "missing", str(error))]
    installed = installed_versions() if installed is None else installed
    out = []
    for name, version in sorted(locked.items()):
        have = installed.get(name)
        if have is None:
            out.append(Check("environment", name, "missing", f"locked {version}, not installed"))
        elif have != version:
            out.append(Check("environment", name, "mismatch", f"installed {have}, locked {version}"))
        else:
            out.append(Check("environment", name, "ok"))
    extra = sorted(set(installed) - set(locked) - {"fincrisis", "pip"})
    out.append(Check("environment", "unlocked packages", "ok", ", ".join(extra) or "none"))
    return out
