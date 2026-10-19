"""Download original subject files from UCI, without extracting archive paths."""

import csv
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def download(destination: Path | None = None, *, subject="mat") -> Path:
    if subject not in {"mat", "por"}:
        raise ValueError("Subject must be mat or por.")
    filename = f"student-{subject}.csv"
    destination = Path(destination) if destination is not None else ROOT / "data" / filename
    with urlopen(URL, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        if filename in archive.namelist():
            csv_bytes = archive.read(filename)
        else:
            with ZipFile(BytesIO(archive.read("student.zip"))) as nested:
                csv_bytes = nested.read(filename)
    header = next(csv.reader(StringIO(csv_bytes.decode("utf-8-sig")), delimiter=";"), [])
    if not {"school", "G1", "G2", "G3", "studytime", "failures", "schoolsup", "famsup",
            "traveltime", "paid", "activities", "internet", "higher"}.issubset(header):
        raise ValueError("UCI returned an unexpected dataset format.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv_bytes)
    return destination


if __name__ == "__main__":
    for subject in ("mat", "por"):
        print(f"Dataset saved to {download(ROOT / 'data' / f'student-{subject}.csv', subject=subject)}")
