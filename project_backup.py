"""Create a portable, committed project snapshot for Drive or local handover."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import uuid
import zipfile


def save_project_snapshot(project_root, destination):
    """Save source ZIP, clean notebook and manifest; never copy ignored local data."""
    root = Path(project_root).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    top = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != root:
        raise ValueError("Choose the root of the StudentInsight Git repository.")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("Commit tracked project changes before saving a project snapshot.")
    revision = git("rev-parse", "HEAD").decode().strip()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    name = "StudentInsight-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name += "-" + revision[:12] + "-" + uuid.uuid4().hex[:8]
    final = destination / name

    # Build separately so an interrupted write is never presented as a completed backup.
    with tempfile.TemporaryDirectory(prefix=".studentinsight-", dir=destination) as temp:
        staging = Path(temp) / name
        staging.mkdir()
        archive = staging / "StudentInsight-Project.zip"
        subprocess.run(["git", "-C", str(root), "archive", "--format=zip",
                        "--prefix=StudentInsight/", "--output", str(archive), revision],
                       check=True)
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise OSError("Project ZIP integrity check failed.")
            notebook = json.loads(bundle.read("StudentInsight/notebooks/StudentInsight.ipynb"))
            files = [item.filename for item in bundle.infolist() if not item.is_dir()]
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        notebook_path = staging / "StudentInsight.ipynb"
        notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        hashes = {}
        for path in (archive, notebook_path):
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = {"git_revision": revision, "project_files": files, "sha256": hashes,
                    "scope": "Committed project files; excludes ignored/untracked files and Git history",
                    "cloud_sharing": "Not configured by this export"}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (staging / "START-HERE.txt").write_text(
            "StudentInsight project handover\n\n"
            "StudentInsight-Project.zip contains the committed application, datasets, tests, "
            "documentation, reports, slides and notebook. Extract it and follow README.md.\n"
            "Open StudentInsight.ipynb in Google Colab for training, charts and predictions. "
            "Colab setup fetches the current GitHub main branch; the ZIP preserves the revision "
            "listed in manifest.json. The Streamlit staff interface runs separately.\n"
            "Virtual environments, account databases, credentials, private uploads, ignored "
            "model binaries and untracked research downloads are excluded. Dependencies and "
            "model binaries can be recreated using the project instructions.\n"
            "Saving files does not grant anyone access. Share the containing Drive folder "
            "with the intended recipient as Viewer, then verify the sharing list.\n",
            encoding="utf-8")
        staging.rename(final)
    return final
