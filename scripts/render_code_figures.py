"""Render exact source excerpts as numbered PNGs for the technical report."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'documents/milestones/figures'

def render(filename, source, first, last, output):
    lines = source.splitlines()[first-1:last]
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 25)
    title = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 28)
    width = max(1200, int(max(font.getlength(line) for line in lines)) + 150)
    image = Image.new('RGB', (width, 100 + len(lines)*36), '#f4f6f8')
    draw = ImageDraw.Draw(image)
    draw.text((26,22), filename, font=title, fill='#183844')
    draw.line((25,70,width-25,70), fill='#d9dfe3', width=2)
    for i,line in enumerate(lines):
        y = 86+i*36
        draw.text((24,y), str(first+i), font=font, fill='#687c89')
        draw.text((110,y), line, font=font, fill='#182c39')
    image.save(OUT/output)

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True, parents=True)
    render('auth.py | account validation | baseline 5af3632',
           (ROOT/'auth.py').read_text(encoding='utf-8'),103,115,'07-account-validation.png')
    render('kaggle_model.py | grouped validation and selection | baseline 5af3632',
           (ROOT/'kaggle_model.py').read_text(encoding='utf-8'),85,96,'08-kaggle-training.png')
    notebook=json.loads((ROOT/'notebooks/StudentInsight.ipynb').read_text(encoding='utf-8'))
    render('StudentInsight.ipynb | cell 6 | Drive connection',
           ''.join(notebook['cells'][6]['source']),3,13,'09-notebook-drive.png')
    print('Rendered three code images from actual source')
