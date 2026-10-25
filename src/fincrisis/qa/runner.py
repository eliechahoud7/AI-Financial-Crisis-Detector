"""One QA run over the whole project (docs/decisions.md, D28).

Stages, each read-only:

1. environment: installed packages against the lock file;
2. integrity: frozen inputs and outputs against recorded checksums (D22);
3. reproducibility: today's code against the frozen results (D23);
4. rag regression: development-set retrieval and reference answers (D25);
5. tests: the unit-test suite, in a separate process.

A stage passes only if every check in it passes. The live tests against
Ollama and the dashboard server run only when asked (`live=True`).
"""

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

from fincrisis.dashboard import loaders
from fincrisis.qa.integrity import Check

LIVE_SWITCHES = ("FINCRISIS_LIVE_OLLAMA", "FINCRISIS_LIVE_SERVER")


@dataclass
class Stage:
    name: str
    checks: list[Check] = field(default_factory=list)
    seconds: float = 0.0
    note: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)


def parse_unittest(output: str, returncode: int) -> list[Check]:
    """Checks from unittest's summary lines ("Ran N tests", "OK (skipped=K)" or "FAILED (...)")."""
    ran = re.search(r"^Ran (\d+) tests?", output, re.MULTILINE)
    result = re.search(r"^(OK|FAILED)\b(?: \((.*)\))?\s*$", output, re.MULTILINE)
    if not ran or not result:
        return [Check("tests", "unittest", "mismatch", f"no test summary (exit code {returncode})")]
    detail = f"{ran.group(1)} tests" + (f" ({result.group(2)})" if result.group(2) else "")
    passed = result.group(1) == "OK" and returncode == 0
    failures = re.findall(r"^(?:FAIL|ERROR): (\S+) \((\S+)\)", output, re.MULTILINE)
    checks = [Check("tests", "unittest", "ok" if passed else "mismatch", detail)]
    # Python 3.12 prints the full path in parentheses; older versions print only the class.
    checks += [Check("tests", where if where.endswith("." + name) else f"{where}.{name}", "mismatch", "failed")
               for name, where in failures]
    return checks


def run_tests(live: bool = False) -> list[Check]:
    env = {k: v for k, v in os.environ.items() if k not in LIVE_SWITCHES}
    if live:
        env.update({switch: "1" for switch in LIVE_SWITCHES})
    process = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
                             cwd=loaders.PROJECT_ROOT, env=env, capture_output=True, text=True)
    return parse_unittest(process.stderr + process.stdout, process.returncode)


def default_stages(live: bool = False, tests: bool = True) -> list[tuple[str, callable]]:
    from fincrisis.qa import drift, environment, integrity, rag_regression
    stages = [("environment", environment.checks), ("integrity", integrity.verify_all),
              ("reproducibility", drift.verify_all), ("rag regression", rag_regression.run)]
    if tests:
        stages.append(("tests", lambda: run_tests(live)))
    return stages


def run(stages: list[tuple[str, callable]]) -> list[Stage]:
    results = []
    for name, check in stages:
        started = time.monotonic()
        try:
            checks = check()
        except Exception as error:  # a crashing stage fails the run; it must not hide the other stages
            checks = [Check(name, "stage", "mismatch", f"{type(error).__name__}: {error}")]
        results.append(Stage(name, checks, round(time.monotonic() - started, 1)))
    return results


def report(stages: list[Stage]) -> str:
    lines = []
    for stage in stages:
        bad = [c for c in stage.checks if not c.ok]
        mark = "PASS" if stage.ok else "FAIL"
        counts = f"{len(stage.checks) - len(bad)}/{len(stage.checks)} checks"
        if stage.name == "tests" and stage.checks:
            counts = stage.checks[0].detail
        lines.append(f"{mark}  {stage.name:<16} {counts:<28} {stage.seconds:>6.1f} s")
        lines += [f"        {c.status.upper()}: {c.group} / {c.name}" + (f" ({c.detail})" if c.detail else "")
                  for c in bad[:20]]
        if len(bad) > 20:
            lines.append(f"        ... and {len(bad) - 20} more")
    passed = all(s.ok for s in stages)
    lines.append("QA passed." if passed else "QA failed.")
    return "\n".join(lines)
