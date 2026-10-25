"""Guard for the RAG validation question set (docs/decisions.md, D16, D24).

The validation set was used once, for the final check of the RAG settings, and
must not be used to tune them. Until now that rule was only a convention: the
evaluation scripts would run on it whenever given its path. They now resolve
their question set here first:

- the development set (the default) is used as before;
- the validation set, recognised by its path or by its content (so a copy
  under another name is recognised too), is refused unless the caller passes
  `final_validation=True` (the scripts' `--final-validation`), and then only if
  it still matches the checksum in `config/frozen_artifacts.json` (D22);
- `final_validation=True` with any other file is refused, so the flag cannot
  be used by mistake.

The check runs before any index, model or Ollama server is touched.
"""

import hashlib
from pathlib import Path

from fincrisis.dashboard import loaders

VALIDATION_SET = "knowledge_base/eval/validation_questions.json"


class QuestionSetError(Exception):
    """The requested question set may not be used this way."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_validation_set(path: Path, recorded: str) -> bool:
    """True for the validation set itself or any copy of it (line endings ignored)."""
    data = Path(path).read_bytes()
    return recorded in (_digest(data), _digest(data.replace(b"\r\n", b"\n")))


def resolve_question_set(requested: str | None, development_set: str, root: Path,
                         final_validation: bool = False) -> tuple[Path, str]:
    """(question file, report-name suffix) for an evaluation run, or QuestionSetError.

    The suffix is "" for the development set and "_<file stem>" otherwise, as the
    scripts named their reports before this guard existed.
    """
    relative = requested or development_set
    path = Path(root) / relative
    if not path.exists():
        raise QuestionSetError(f"question set {relative} not found")
    try:
        recorded = loaders.frozen_checksums()[VALIDATION_SET]
    except (loaders.ArtifactError, KeyError) as error:
        raise QuestionSetError(f"the validation set's recorded checksum is unavailable ({error}); refusing to "
                               "evaluate until config/frozen_artifacts.json is restored") from error
    validation = is_validation_set(path, recorded) or Path(relative).as_posix() == VALIDATION_SET
    if validation and not final_validation:
        raise QuestionSetError(
            f"{relative} is the RAG validation set. It was used once for the final check (D16) and must not be "
            "used to tune settings. Use the development set (the default); pass --final-validation only to "
            "repeat the final check deliberately.")
    if validation and _digest(path.read_bytes()) != recorded:
        raise QuestionSetError(f"{relative} no longer matches its recorded checksum (D22); refusing to report on it")
    if final_validation and not validation:
        raise QuestionSetError(f"--final-validation applies only to the validation set, not to {relative}")
    suffix = "" if relative == development_set else "_" + Path(relative).stem
    return path, suffix
