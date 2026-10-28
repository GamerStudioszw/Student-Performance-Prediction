"""Restore the pinned public Kaggle source without accepting arbitrary archive paths."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import json

ROOT=Path(__file__).resolve().parent
URL='https://www.kaggle.com/api/v1/datasets/download/aljarah/xAPI-Edu-Data?datasetVersionNumber=6'
LIMIT=2*1024*1024

def download(destination=None):
    destination=Path(destination) if destination is not None else ROOT/'data/kaggle/xAPI-Edu-Data.csv'
    expected=json.loads((ROOT/'data/kaggle/provenance.json').read_text())['sha256']
    with urlopen(URL,timeout=60) as response:
        body=response.read(LIMIT+1)
    if len(body)>LIMIT: raise ValueError('Kaggle archive exceeds the download limit.')
    with ZipFile(BytesIO(body)) as archive:
        item=archive.getinfo('xAPI-Edu-Data.csv')
        if item.file_size>LIMIT: raise ValueError('Kaggle CSV exceeds the size limit.')
        with archive.open(item) as stream: content=stream.read(LIMIT+1)
    if hashlib.sha256(content).hexdigest()!=expected:
        raise ValueError('Kaggle content does not match the pinned dataset. Existing data was not changed.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(content)
    return destination

if __name__=='__main__': print(download())
