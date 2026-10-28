"""Kaggle xAPI academic-band classification, separate from UCI grade regression."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data/kaggle/xAPI-Edu-Data.csv'
SOURCE = 'https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data'
LABELS = ['L', 'M', 'H']
BANDS = {'L': 'Low', 'M': 'Middle', 'H': 'High'}
TOPICS = {'Arabic':'Arabic', 'Biology':'Biology', 'Chemistry':'Chemistry', 'English':'English',
          'French':'French', 'Geology':'Geology', 'History':'History', 'IT':'Information Technology',
          'Math':'Mathematics', 'Quran':'Quran', 'Science':'Science', 'Spanish':'Spanish'}
NUMERIC = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
CATEGORIES = {
    'StageID':['lowerlevel','MiddleSchool','HighSchool'],
    'GradeID':['G-02','G-04','G-05','G-06','G-07','G-08','G-09','G-10','G-11','G-12'],
    'Topic':list(TOPICS), 'Semester':['F','S'], 'StudentAbsenceDays':['Under-7','Above-7'],
}
FEATURES = list(CATEGORIES) + NUMERIC
FIELD_LABELS = {'StageID':'Education stage','GradeID':'Grade level','Topic':'Subject',
                'Semester':'Semester','StudentAbsenceDays':'Absence band',
                'raisedhands':'Raised hands count','VisITedResources':'Resource visits',
                'AnnouncementsView':'Announcement views','Discussion':'Discussion participation count'}
STAGE_GRADES = {'lowerlevel':['G-02','G-04','G-05','G-07'], 'MiddleSchool':['G-06','G-07','G-08'],
                'HighSchool':['G-09','G-10','G-11','G-12']}

def validate_features(frame, *, target=False):
    if not isinstance(frame,pd.DataFrame) or not frame.columns.is_unique or frame.empty:
        raise ValueError('Provide a non-empty table with unique columns.')
    required=FEATURES + (['Class'] if target else [])
    missing=set(required)-set(frame.columns)
    if missing:
        raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    clean=frame[required].copy()
    for field, options in {**CATEGORIES, **({'Class':LABELS} if target else {})}.items():
        if not clean[field].map(lambda x:isinstance(x,str)).all():
            raise ValueError(f'{field} requires a supported text category.')
        clean[field]=clean[field].str.strip()
        if not clean[field].isin(options).all():
            raise ValueError(f'{field} must be one of: '+', '.join(options))
    for field in NUMERIC:
        if clean[field].map(lambda x:isinstance(x,(bool,np.bool_,complex))).any():
            raise ValueError(f'{field} must be a whole-number count, not a boolean.')
        values=pd.to_numeric(clean[field],errors='coerce')
        if not (values.notna() & np.isfinite(values) & values.between(0,100) & (values%1==0)).all():
            raise ValueError(f'{field} must be a whole number from 0 to 100; missing values are not filled.')
        clean[field]=values.astype(int)
    if not all(grade in STAGE_GRADES[stage] for stage,grade in zip(clean.StageID,clean.GradeID)):
        raise ValueError('Grade level does not match education stage in this dataset.')
    return clean

def load_data(path=DATA_PATH):
    return validate_features(pd.read_csv(path),target=True)

def pipeline(estimator):
    transform=ColumnTransformer([
        ('counts',StandardScaler(),NUMERIC),
        ('categories',OneHotEncoder(categories=list(CATEGORIES.values()),handle_unknown='error',sparse_output=False),list(CATEGORIES))])
    return Pipeline([('preprocessing',transform),('classifier',estimator)])

def train_and_evaluate(frame=None):
    data=load_data() if frame is None else validate_features(frame,target=True)
    raw_count=len(data)
    data=data.drop_duplicates().copy()  # Identical selected inputs and outcomes, not inferred student identities.
    source_rows=data.index.to_numpy()
    groups=pd.util.hash_pandas_object(data[FEATURES],index=False).astype(str)
    train_idx,test_idx=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(data,groups=groups))
    train,test=data.iloc[train_idx],data.iloc[test_idx]
    train_groups=groups.iloc[train_idx]
    if min(train.Class.value_counts())<5 or set(test.Class)!=set(LABELS):
        raise ValueError('Insufficient class support for the fixed evaluation split.')
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(train,train.Class,groups=train_groups))
    specs=[('Most frequent baseline',DummyClassifier(strategy='most_frequent')),
           ('Logistic regression',LogisticRegression(max_iter=2000,C=1,random_state=42)),
           ('Random forest',RandomForestClassifier(n_estimators=250,max_depth=10,min_samples_leaf=3,
                                                   class_weight='balanced',random_state=42,n_jobs=1))]
    comparison=[]
    for name,estimator in specs:
        scores=cross_val_score(pipeline(estimator),train[FEATURES],train.Class,cv=folds,scoring='f1_macro',n_jobs=1)
        comparison.append({'model':name,'cv_macro_f1':float(scores.mean()),'cv_std':float(scores.std()),'fold_scores':scores.tolist()})
    best=max(range(len(specs)),key=lambda i:comparison[i]['cv_macro_f1'])
    fitted=pipeline(specs[best][1]).fit(train[FEATURES],train.Class)
    predicted=fitted.predict(test[FEATURES])
    metrics={'task':'Kaggle xAPI academic performance bands','source':SOURCE,'dataset_version':6,
             'sklearn_version':sklearn.__version__,
             'model_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'validated_data_sha256':hashlib.sha256(validate_features(frame,target=True).to_csv(index=False).encode()).hexdigest() if frame is not None else hashlib.sha256(load_data().to_csv(index=False).encode()).hexdigest(),
             'selected_model':specs[best][0],'features':FEATURES,'labels':LABELS,'raw_rows':raw_count,
             'unique_model_rows':len(data),'removed_duplicate_rows':raw_count-len(data),
             'train_rows':len(train),'test_rows':len(test),'comparison':comparison,
             'accuracy':float(accuracy_score(test.Class,predicted)),
             'balanced_accuracy':float(balanced_accuracy_score(test.Class,predicted)),
             'macro_f1':float(f1_score(test.Class,predicted,labels=LABELS,average='macro',zero_division=0)),
             'confusion_matrix':confusion_matrix(test.Class,predicted,labels=LABELS).tolist(),
             'classification_report':classification_report(test.Class,predicted,labels=LABELS,output_dict=True,zero_division=0),
             'split_rows':{'train':source_rows[train_idx].tolist(),'test':source_rows[test_idx].tolist()},
             'split_policy':'Seed 42, 20% of identical-input groups held out; grouped stratified 5-fold CV on remaining rows.',
             'selection_rule':'Highest mean validation macro F1; candidate order resolves exact ties. No test-based selection.',
             'limitations':'Records lack student identifiers and measurement cutoff dates. Identical predictor groups are separated, but repeated-student overlap cannot be ruled out. Results are retrospective, not verified early-warning or local-school performance.'}
    report=test.copy();report.insert(0,'source_row',source_rows[test_idx]);report['predicted_band']=predicted
    return fitted,metrics,report.reset_index(drop=True)

def predict(model,frame):
    return model.predict(validate_features(frame))

def save_evaluation():
    model,metrics,report=train_and_evaluate()
    metrics['data_sha256']=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    destination=ROOT/'artifacts/kaggle';destination.mkdir(parents=True,exist_ok=True)
    (destination/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    report.to_csv(destination/'test_predictions.csv',index=False)
    print(json.dumps({k:metrics[k] for k in ['selected_model','raw_rows','unique_model_rows','train_rows','test_rows','accuracy','macro_f1']},indent=2))
    return metrics

if __name__=='__main__':
    save_evaluation()
