"""Download the original mathematics dataset from UCI, without extracting paths."""

import csv
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def download(destination: Path = ROOT / "data" / "student-mat.csv") -> Path:
    with urlopen(URL, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        if "student-mat.csv" in archive.namelist():
            csv_bytes = archive.read("student-mat.csv")
        else:
            with ZipFile(BytesIO(archive.read("student.zip"))) as nested:
                csv_bytes = nested.read("student-mat.csv")
    header = next(csv.reader(StringIO(csv_bytes.decode("utf-8-sig")), delimiter=";"), [])
    if not {"school", "G1", "G2", "G3", "studytime", "failures", "schoolsup", "famsup"}.issubset(header):
        raise ValueError("UCI returned an unexpected dataset format.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv_bytes)
    return destination


if __name__ == "__main__":
    print(f"Dataset saved to {download()}")
