"""Download original subject files from UCI, without extracting archive paths."""

import csv
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024


def _read_member(archive, name):
    member = archive.getinfo(name)
    if member.file_size > MAX_DOWNLOAD_BYTES:
        raise ValueError("The dataset archive member exceeds the size limit.")
    with archive.open(member) as stream:
        content = stream.read(MAX_DOWNLOAD_BYTES + 1)
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError("The dataset archive member exceeds the size limit.")
    return content


def download(destination: Path | None = None, *, subject="mat") -> Path:
    if subject not in {"mat", "por"}:
        raise ValueError("Subject must be mat or por.")
    filename = f"student-{subject}.csv"
    destination = Path(destination) if destination is not None else ROOT / "data" / filename
    with urlopen(URL, timeout=60) as response:
        archive_bytes = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(archive_bytes) > MAX_DOWNLOAD_BYTES:
        raise ValueError("The dataset download exceeds the size limit.")
    with ZipFile(BytesIO(archive_bytes)) as archive:
        if filename in archive.namelist():
            csv_bytes = _read_member(archive, filename)
        else:
            with ZipFile(BytesIO(_read_member(archive, "student.zip"))) as nested:
                csv_bytes = _read_member(nested, filename)
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
