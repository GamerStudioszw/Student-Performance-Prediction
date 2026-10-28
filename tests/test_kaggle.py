"""Check the distinct Kaggle task, leakage controls, uploads and real app route."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from auth import AuthStore
from cohort import export_csv
import kaggle_model as km
from kaggle_cohort import analyse, read_class

class KaggleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=km.load_data()
        cls.model,cls.metrics,cls.test=km.train_and_evaluate(cls.data)

    def test_source_provenance_and_only_recorded_subjects(self):
        metadata=json.loads((km.ROOT/'data/kaggle/provenance.json').read_text())
        self.assertEqual(hashlib.sha256(km.DATA_PATH.read_bytes()).hexdigest(),metadata['sha256'])
        self.assertEqual(len(self.data),480)
        self.assertEqual(set(self.data.Topic),set(km.TOPICS))
        self.assertFalse({'gender','NationalITy','Class','student_name','G2'} & set(km.FEATURES))

    def test_split_groups_and_selection_are_separate(self):
        train=self.data.loc[self.metrics['split_rows']['train']]
        test=self.data.loc[self.metrics['split_rows']['test']]
        def groups(frame): return set(pd.util.hash_pandas_object(frame[km.FEATURES],index=False))
        self.assertFalse(groups(train)&groups(test))
        best=max(self.metrics['comparison'],key=lambda row:row['cv_macro_f1'])
        self.assertEqual(self.metrics['selected_model'],best['model'])
        np.testing.assert_array_equal(self.test.predicted_band,km.predict(self.model,test))
        self.assertEqual(self.metrics['unique_model_rows'],474)

    def test_invalid_inputs_are_rejected(self):
        for field,value in [('raisedhands',True),('Discussion',np.inf),('VisITedResources',101),
                            ('AnnouncementsView',1.2),('Topic','Physics'),('GradeID','G-03'),('Semester','')]:
            frame=self.data.head(1).astype(object).copy();frame.loc[frame.index[0],field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError): km.validate_features(frame)
        with self.assertRaises(ValueError): km.validate_features(self.data.drop(columns='Topic'))

    def test_named_upload_and_export(self):
        frame=self.data.head(2).copy();frame['student_name']=['Learner One','Learner Two'];frame['class_name']='Example'
        clean=read_class(frame.to_csv(index=False).encode())
        output=analyse(self.model,clean,self.metrics)
        self.assertEqual(output.student_name.tolist(),frame.student_name.tolist())
        self.assertIn('recorded_band',output)
        np.testing.assert_array_equal(km.predict(self.model,frame),output.predicted_band)
        frame['student_name']='Same name'
        with self.assertRaises(ValueError):read_class(frame.to_csv(index=False).encode())
        frame['student_id']=['ID1','ID2']
        self.assertEqual(len(read_class(frame.to_csv(index=False).encode())),2)
        frame.loc[0,'student_name']='=SUM(1,2)'
        safe=export_csv(analyse(self.model,read_class(frame.to_csv(index=False).encode()),self.metrics))
        self.assertIn("'=SUM",safe.decode('utf-8-sig'))

    def test_kaggle_default_single_student_and_reference_switch(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'STUDENT_AUTH_DB':str(Path(directory)/'accounts.sqlite3')}):
            store=AuthStore();store.create_account('Demonstration Teacher','teacher','example-only passphrase for tests')
            app=AppTest.from_file(str(km.ROOT/'app.py'),default_timeout=60)
            app.session_state['auth_token']=store.login('teacher','example-only passphrase for tests')
            app.run();self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='analytics_dataset').value,'Kaggle academic performance')
            self.assertEqual(app.selectbox(key='kaggle_subject').options[0],'All subjects')
            app.text_input(key='kaggle_name').set_value('Example Student')
            next(b for b in app.button if b.label=='Predict performance band').click().run()
            self.assertFalse(app.exception);self.assertFalse(app.error)
            self.assertTrue(any(m.label=='Estimated performance band' for m in app.metric))
            app.selectbox(key='analytics_dataset').select('UCI grade reference').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='subject').options,['Mathematics','Portuguese language'])

if __name__=='__main__': unittest.main()
