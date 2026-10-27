"""Portable handover snapshots exclude local secrets and preserve committed source."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from project_backup import save_project_snapshot


class ProjectBackupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Snapshot Test")
        self.git("config", "user.email", "test@example.invalid")
        (self.root / "notebooks").mkdir()
        (self.root / "notebooks/StudentInsight.ipynb").write_text(json.dumps({
            "cells": [{"cell_type": "code", "source": ["print('example')"],
                       "execution_count": 1, "outputs": [{"text": "example"}]}]}))
        (self.root / "app.py").write_text("# committed application\n")
        (self.root / ".gitignore").write_text(".local/\n")
        self.git("add", ".")
        self.git("commit", "-qm", "Fixture")
        (self.root / ".local").mkdir()
        (self.root / ".local/accounts.db").write_text("private fixture")
        (self.root / "untracked.csv").write_text("private fixture")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], stderr=subprocess.PIPE)

    def test_archive_matches_commit_excludes_private_files_and_has_valid_hashes(self):
        result = save_project_snapshot(self.root, self.base / "drive")
        manifest = json.loads((result / "manifest.json").read_text())
        with zipfile.ZipFile(result / "StudentInsight-Project.zip") as archive:
            self.assertEqual(archive.read("StudentInsight/app.py").decode().splitlines(),
                             ["# committed application"])
            self.assertFalse(any("accounts.db" in name or "untracked.csv" in name
                                 for name in archive.namelist()))
        for name, expected in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((result / name).read_bytes()).hexdigest(), expected)
        notebook = json.loads((result / "StudentInsight.ipynb").read_text())
        self.assertEqual(notebook["cells"][0]["outputs"], [])
        self.assertIsNone(notebook["cells"][0]["execution_count"])

    def test_dirty_tracked_files_are_not_silently_omitted(self):
        (self.root / "app.py").write_text("# unsaved revision\n")
        with self.assertRaisesRegex(ValueError, "Commit tracked"):
            save_project_snapshot(self.root, self.base / "drive")
        self.assertFalse((self.base / "drive").exists())

    def test_repeated_exports_preserve_previous_snapshot(self):
        first = save_project_snapshot(self.root, self.base / "drive")
        original = (first / "manifest.json").read_bytes()
        second = save_project_snapshot(self.root, self.base / "drive")
        self.assertNotEqual(first, second)
        self.assertEqual((first / "manifest.json").read_bytes(), original)

    def test_nested_project_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "root"):
            save_project_snapshot(self.root / "notebooks", self.base / "drive")
