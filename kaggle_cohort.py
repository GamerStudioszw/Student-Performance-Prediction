"""Strict class-file validation and exports for the Kaggle band predictor."""
import unicodedata
import pandas as pd
from cohort import MAX_ROWS, parse_csv, student_details
from kaggle_model import FEATURES, LABELS, BANDS, SOURCE, validate_features, predict

def read_class(content):
    frame=parse_csv(content)
    if len(frame)>MAX_ROWS:
        raise ValueError('Use no more than 5,000 records.')
    clean=validate_features(frame).reset_index(drop=True)
    if 'class_name' not in frame:
        raise ValueError('Missing column: class_name')
    for field in ['student_name','student_id','class_name']:
        values=frame[field].fillna('').astype(str) if field in frame else pd.Series('',index=frame.index)
        clean[field]=[student_details(**{field:value})[field] for value in values]
    if clean.class_name.eq('').any() or (clean.student_name.eq('') & clean.student_id.eq('')).any():
        raise ValueError('Every row needs a class_name and a student_name or student_id.')
    has_id=clean.student_id.ne('')
    if clean.loc[has_id,'student_id'].duplicated().any():
        raise ValueError('Student IDs must be unique within the file.')
    normalized=clean[['student_name','class_name']].map(lambda v:' '.join(unicodedata.normalize('NFKC',v).casefold().split()))
    if (normalized.duplicated(keep=False) & clean.student_name.ne('') & ~has_id).any():
        raise ValueError('Students sharing a name in a class need distinct student IDs.')
    if 'Class' in frame:
        values=frame.Class.fillna('').astype(str).str.strip()
        if not values.isin(['',*LABELS]).all():
            raise ValueError('Optional Class outcomes must be L, M, H or blank.')
        clean['recorded_band']=values.to_numpy()
    return clean

def analyse(model,frame,metrics):
    output=frame.copy()
    output['predicted_band']=predict(model,frame)
    output['predicted_performance']=output.predicted_band.map(BANDS)
    output['model']=metrics['selected_model']
    output['dataset_source']=SOURCE
    output['dataset_version']=6
    output['prediction_task']='Academic performance band, not an exact grade'
    return output

def template():
    return pd.DataFrame(columns=['student_name','student_id','class_name',*FEATURES]).to_csv(index=False).encode('utf-8')
