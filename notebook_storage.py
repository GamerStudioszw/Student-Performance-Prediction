"""Explicit notebook file access within a mounted or locally synced Drive folder."""

import hashlib
from pathlib import Path
import shutil

from cohort import MAX_BYTES, read_cohort_csv


def prepare_drive_workspace(drive_root):
    root = Path(drive_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Drive folder is unavailable. Mount Google Drive or choose an existing synced folder.")
    workspace = root / "StudentInsight"
    for name in ("datasets", "reports"):
        folder = workspace / name
        if not folder.resolve().is_relative_to(root):
            raise ValueError("StudentInsight folders must stay inside the selected Drive folder.")
        folder.mkdir(parents=True, exist_ok=True)
    return workspace


def read_drive_cohort(workspace, filename, features):
    root = Path(workspace).resolve()
    directory = (root / "datasets").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Dataset folder must stay inside StudentInsight.")
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory) or path.suffix.lower() != ".csv":
        raise ValueError("Choose a CSV inside StudentInsight/datasets.")
    if not path.is_file():
        raise ValueError(f"Dataset not found: {path.name}. Add it to StudentInsight/datasets first.")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Use a CSV file smaller than 2 MB.")
    with path.open("rb") as stream:
        return read_cohort_csv(stream.read(MAX_BYTES + 1), features)


def save_drive_archive(archive, workspace):
    """Copy one explicitly selected ZIP, reject overwrites, and check copied bytes."""
    source = Path(archive)
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError("Choose an existing experiment ZIP to save.")
    root = Path(workspace).resolve()
    directory = (root / "reports").resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():
        raise ValueError("Drive reports folder is unavailable. Reconnect Drive first.")
    destination = directory / source.name
    # Exclusive creation ensures a repeat never replaces an existing report.
    with destination.open("xb") as target:
        try:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, target)
        except BaseException:
            target.close()
            destination.unlink(missing_ok=True)
            raise
    with source.open("rb") as stream:
        original_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with destination.open("rb") as stream:
        copied_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if original_hash != copied_hash:
        destination.unlink()
        raise OSError("Drive copy did not match the local ZIP; local results remain available.")
    return destination
