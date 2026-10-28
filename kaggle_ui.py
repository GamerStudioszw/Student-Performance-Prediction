"""Educator interface for the Kaggle xAPI classifier."""
import hashlib
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st
from cohort import export_csv, student_details
from kaggle_cohort import analyse, read_class, template
from kaggle_model import (DATA_PATH, SOURCE, TOPICS, BANDS, LABELS, NUMERIC, CATEGORIES,
                          FEATURES, FIELD_LABELS, STAGE_GRADES, load_data, train_and_evaluate)

@st.cache_resource(show_spinner=False)
def load_classifier(data_hash,code_hash):
    return train_and_evaluate()

def render_kaggle_workspace():
    st.caption('STAFF WORKSPACE · KAGGLE EDUCATION DATA')
    st.title('Student Performance Analytics')
    st.write('Review academic performance bands, class engagement and students who may need support.')
    st.caption('Kaggle xAPI dataset · 480 historical records · 12 subjects · one pooled classifier. These are retrospective estimates, not verified early-warning predictions.')
    try:
        with st.spinner('Preparing the Kaggle performance model…'):
            model,metrics,historical=load_classifier(hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                hashlib.sha256(Path(__file__).with_name('kaggle_model.py').read_bytes()).hexdigest())
    except Exception:
        import logging
        logging.getLogger(__name__).exception('Kaggle model preparation failed')
        st.error('The Kaggle model could not be prepared. Check the dataset and installed dependencies.')
        return
    selected=st.sidebar.selectbox('Subject',['All subjects',*TOPICS],format_func=lambda x:TOPICS.get(x,x),key='kaggle_subject')
    tabs=st.tabs(['Class dashboard','Single-student prediction','Model quality','Data guide'])
    with tabs[0]:
        st.subheader('Class performance overview')
        mode=st.radio('Class data',['Historical Kaggle example','Upload class CSV'],horizontal=True,key='kaggle_source')
        if mode=='Historical Kaggle example':
            frame=historical.drop(columns='predicted_band').rename(columns={'Class':'recorded_band'}).copy()
            frame['student_id']='Source row '+frame.source_row.astype(str)
            frame['student_name']=''
            frame['class_name']=frame.StageID+' / '+frame.GradeID
            st.caption(f'{len(frame)} held-out records. Row labels identify dataset rows, not verified unique students. Groups are education stage and grade level.')
        else:
            st.download_button('Download Kaggle class template',template(),'kaggle_class_template.csv','text/csv')
            st.caption('Supply names or IDs, class_name, and all nine inputs. Use the Data guide for exact categories. Uploads are session-only and never retrain the model. Limit: 2 MiB / 5,000 rows.')
            upload=st.file_uploader('Upload Kaggle class CSV',type='csv',key='kaggle_upload')
            frame=None
            if upload is not None:
                try: frame=read_class(upload.getvalue())
                except ValueError as exc: st.error(str(exc))
            else: st.info('Upload a class file to review your students.')
        if frame is not None:
            report=analyse(model,frame,metrics)
            report['data_source']=mode
            if selected!='All subjects': report=report[report.Topic==selected]
            groups=st.multiselect('Classes / groups',sorted(report.class_name.unique()),default=sorted(report.class_name.unique()),key=f'kaggle_groups_{mode}_{selected}')
            report=report[report.class_name.isin(groups)]
            if report.empty: st.info('No records match the selected subject and groups.')
            else:
                cards=st.columns(4)
                cards[0].metric('Records in view',len(report))
                for card,band in zip(cards[1:],LABELS): card.metric(f'{BANDS[band]} performance estimates',int(report.predicted_band.eq(band).sum()))
                left,right=st.columns(2)
                with left:
                    st.markdown('#### Estimated performance distribution')
                    counts=report.predicted_band.value_counts().reindex(LABELS,fill_value=0).rename_axis('Band').reset_index(name='Records')
                    counts['Performance']=counts.Band.map(BANDS)
                    st.altair_chart(alt.Chart(counts).mark_bar().encode(x=alt.X('Performance:N',sort=list(BANDS.values())),y='Records:Q',color=alt.Color('Performance:N',legend=None),tooltip=['Performance','Records']).properties(height=250),width='stretch')
                with right:
                    st.markdown('#### Engagement averages')
                    averages=report[NUMERIC].mean().rename_axis('Input').reset_index(name='Average count')
                    averages.Input=averages.Input.map(FIELD_LABELS)
                    st.altair_chart(alt.Chart(averages).mark_bar(color='#087f8c').encode(y=alt.Y('Input:N',title=None),x='Average count:Q',tooltip=['Input','Average count']).properties(height=250),width='stretch')
                summary=pd.crosstab(report.class_name,report.predicted_band).reindex(columns=LABELS,fill_value=0).rename(columns=BANDS).reset_index()
                st.markdown('#### Class comparison')
                st.dataframe(summary,hide_index=True,width='stretch')
                query=st.text_input('Find student name or ID',key='kaggle_search')
                bands=st.multiselect('Performance bands',LABELS,default=LABELS,format_func=BANDS.get,key='kaggle_bands')
                visible=report[report.predicted_band.isin(bands)]
                if query.strip(): visible=visible[visible.student_name.str.contains(query.strip(),case=False,regex=False)|visible.student_id.str.contains(query.strip(),case=False,regex=False)]
                st.markdown('#### Student review table')
                columns=['student_name','student_id','class_name','Topic','predicted_performance',*NUMERIC,'StudentAbsenceDays']
                if 'recorded_band' in visible: columns.append('recorded_band')
                st.dataframe(visible[columns],hide_index=True,width='stretch',height=320)
                st.caption('Name and band filters affect this table and its export; group and subject filters also affect charts. Low means the dataset’s 0–69 band, not an institutional failure decision.')
                st.download_button('Download filtered student report',export_csv(visible),'kaggle_student_review.csv','text/csv')
                st.download_button('Download class summary',export_csv(summary),'kaggle_class_summary.csv','text/csv')
    with tabs[1]:
        st.subheader("Predict one student's performance band")
        st.caption('Use observed LMS counts from a comparable measurement period. The source provides no exact cutoff date; do not present this as a validated forecast before an exam.')
        with st.form('kaggle_prediction'):
            name=st.text_input('Student name (optional)',key='kaggle_name',max_chars=80)
            student_id=st.text_input('Student ID (optional)',key='kaggle_id',max_chars=80)
            group=st.text_input('Class / group (optional)',key='kaggle_class',max_chars=80)
            values={}
            left,right=st.columns(2)
            for i,(field,options) in enumerate(CATEGORIES.items()):
                with (left if i%2==0 else right):
                    default=options.index(selected) if field=='Topic' and selected in options else 0
                    values[field]=st.selectbox(FIELD_LABELS[field],options,index=default,key='input_kaggle_'+field,
                        format_func=(TOPICS.get if field=='Topic' else lambda x:x))
            for i,field in enumerate(NUMERIC):
                with (left if i%2==0 else right): values[field]=st.number_input(FIELD_LABELS[field],0,100,0,key='input_kaggle_'+field)
            st.caption('Counts range from 0 to 100. Zero means an observed zero, not missing. Education stage and grade must match the combinations in the Data guide.')
            submitted=st.form_submit_button('Predict performance band',type='primary',width='stretch')
        if submitted:
            st.session_state.pop('kaggle_result',None)
            try:
                details=student_details(name,student_id,group)
                result=analyse(model,pd.DataFrame([{**values,**details}]),metrics)
                st.session_state['kaggle_result']=result
            except ValueError as exc: st.error(str(exc))
        if 'kaggle_result' in st.session_state:
            result=st.session_state['kaggle_result']; row=result.iloc[0]
            st.write(f"Last submitted profile: {row.student_name or row.student_id or 'Unnamed student'} · {TOPICS[row.Topic]}")
            st.metric('Estimated performance band',row.predicted_performance)
            st.info('This is a model estimate of a dataset-defined category, not an exact mark or a probability of passing. Review the student’s circumstances before acting.')
            st.download_button('Download this prediction report',export_csv(result),'kaggle_single_student.csv','text/csv')
    with tabs[2]:
        st.subheader('Kaggle model quality')
        st.write(f"Selected model: **{metrics['selected_model']}** · selected using validation macro F1.")
        cards=st.columns(3)
        cards[0].metric('Test accuracy',f"{metrics['accuracy']:.1%}")
        cards[1].metric('Test macro F1',f"{metrics['macro_f1']:.3f}")
        cards[2].metric('Balanced accuracy',f"{metrics['balanced_accuracy']:.1%}")
        st.caption(f"Pooled test results across all subjects: {metrics['train_rows']} training records / {metrics['test_rows']} test records after removing {metrics['removed_duplicate_rows']} exact model-row duplicates. These metrics are not specific to the sidebar subject.")
        st.dataframe(pd.DataFrame(metrics['comparison']).drop(columns='fold_scores'),hide_index=True,width='stretch')
        st.markdown('#### Confusion matrix')
        st.caption('Rows are recorded bands; columns are predicted bands. L = Low, M = Middle, H = High.')
        st.dataframe(pd.DataFrame(metrics['confusion_matrix'],index=LABELS,columns=LABELS),width='stretch')
        st.write(metrics['split_policy']);st.write(metrics['limitations'])
        st.caption('Macro F1 gives equal weight to each performance band. A random retrospective holdout does not establish future performance at your institution. No calibrated probabilities are displayed.')
    with tabs[3]:
        st.subheader('Kaggle dataset and input guide')
        st.link_button("View the dataset on Kaggle",SOURCE)
        st.write('xAPI-Edu-Data, version 6, by Elaf Abu Amrieh, Thair Hamtini and Ibrahim Aljarah (University of Jordan). Collected from Kalboard 360. License: CC BY-SA 4.0. The original CSV is preserved.')
        st.write('Outcomes: Low (0–69), Middle (70–89), High (90–100), as defined by the source. Exact marks are absent. There are no Portuguese, Physics or Geography records; Geology and Science are separate recorded topics.')
        data=load_data()
        support=data.Topic.value_counts().rename_axis('Topic').reset_index(name='Source records')
        support.Topic=support.Topic.map(TOPICS)
        st.dataframe(support,hide_index=True,width='stretch')
        rows=[{'Column':f,'Meaning':FIELD_LABELS[f],'Allowed values':', '.join(CATEGORIES[f]) if f in CATEGORIES else 'Whole-number count 0–100'} for f in FEATURES]
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        st.write('Observed education-stage and grade combinations: '+ '; '.join(k+': '+', '.join(v) for k,v in STAGE_GRADES.items()))
        st.caption('The source records G-07 in more than one stage; those original categories are retained. Under-7 and Above-7 are source codes; exactly seven days is not defined by the source, so do not guess a band. Demographics, parent variables, names, IDs and the recorded outcome are excluded from model inputs. No missing input is invented.')
