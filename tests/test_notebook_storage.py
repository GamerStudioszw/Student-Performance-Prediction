"""Exercise Drive file operations using temporary folders, without cloud access."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from notebook_storage import prepare_drive_workspace, read_drive_cohort, save_drive_archive


class NotebookStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = prepare_drive_workspace(self.root)

    def test_missing_drive_is_not_silently_created(self):
        with self.assertRaises(ValueError):
            prepare_drive_workspace(self.root / "not-mounted")
        self.assertFalse((self.root / "not-mounted").exists())

    def test_reads_only_requested_csv_and_preserves_missing_g1(self):
        path = self.workspace / "datasets" / "class.csv"
        content = b"student_id,class_name,G2\nS1,A,12\n"
        path.write_bytes(content)
        frame = read_drive_cohort(self.workspace, "class.csv", ["G2"])
        self.assertEqual(frame.G2.iloc[0], 12)
        self.assertTrue(frame.G1.isna().all())
        self.assertEqual(path.read_bytes(), content)

    def test_outside_paths_and_missing_required_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            read_drive_cohort(self.workspace, "../outside.csv", ["G2"])
        with self.assertRaises(ValueError):
            read_drive_cohort(self.workspace, "missing.csv", ["G2"])
        (self.workspace / "datasets" / "bad.csv").write_text("student_id,class_name,G1\nS1,A,10\n")
        with self.assertRaisesRegex(ValueError, "G2"):
            read_drive_cohort(self.workspace, "bad.csv", ["G2"])

    def test_zip_copy_is_identical_and_does_not_overwrite(self):
        source = self.root / "experiment.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("report.csv", "predicted_grade\n12\n")
        saved = save_drive_archive(source, self.workspace)
        self.assertEqual(saved.read_bytes(), source.read_bytes())
        with self.assertRaises(FileExistsError):
            save_drive_archive(source, self.workspace)
        self.assertEqual(saved.read_bytes(), source.read_bytes())

    def test_failed_copy_keeps_source_and_removes_incomplete_destination(self):
        source = self.root / "experiment.zip"
        source.write_bytes(b"local report")
        with patch("notebook_storage.shutil.copyfileobj", side_effect=OSError("Disconnected")):
            with self.assertRaises(OSError):
                save_drive_archive(source, self.workspace)
        self.assertEqual(source.read_bytes(), b"local report")
        self.assertFalse((self.workspace / "reports" / source.name).exists())
