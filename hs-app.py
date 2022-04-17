import streamlit as st
import pandas as pd
import numpy as np
from preprocess import preprocess
import pickle
import base64
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModel
# st.beta_set_page_config(page_title='IndoHateSpeech')
np.set_printoptions(suppress=True)


tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
model = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")


def load_models(deep=False):
    # LOAD MODELS
    if not deep:
        cat_model = pickle.load(open('rf_cat.sav', 'rb'))
        hs_model = pickle.load(open('rf_hs.sav', 'rb'))
        level_model = pickle.load(open('rf_level.sav', 'rb'))
        target_model = pickle.load(open('rf_target.sav', 'rb'))

    else: # deep == True
        cat_model = load_model('rcnn_category.hdf5')
        hs_model = load_model('rcnn_hs.hdf5')
        level_model = load_model('rcnn_level.hdf5')
        target_model = load_model('rcnn_target.hdf5')

    return cat_model, hs_model, level_model, target_model


def get_demo(cat_model, hs_model, level_model, target_model, deep=False):
    st.write("""

    ## Demo
    """)
    text = st.text_input("For a quick demo, type an example sentence and press ENTER or RETURN:", value='')

    if text != '':
        st.write('Input: ', text)
        
        if not deep:
            text = [text]
            prediction1 = cat_model.predict_proba(text)[0]
            prediction2 = hs_model.predict_proba(text)[0]
            prediction3 = level_model.predict_proba(text)[0]
            prediction4 = target_model.predict_proba(text)[0]

            individual_score = round(prediction1[0],2) * 100
            group_score = round(prediction1[1],2) * 100

            hs_score = round(prediction2[0],2) * 100
            abusive_score = round(prediction2[1],2) * 100

            weak_score = round(prediction3[0],2) * 100
            moderate_score = round(prediction3[1],2) * 100
            strong_score = round(prediction3[2],2) * 100

            religion_score = round(prediction4[0],2) * 100
            race_score = round(prediction4[1],2) * 100
            physical_score = round(prediction4[2],2) * 100
            gender_score = round(prediction4[3],2) * 100
            other_score = round(prediction4[4],2) * 100

        else:
            text = tokenizer(text)['input_ids']
            text = sequence.pad_sequences([text], maxlen=80)
            # print(text)
            prediction1 = cat_model.predict(text)[0]
            prediction2 = hs_model.predict(text)[0]
            prediction3 = level_model.predict(text)[0]
            prediction4 = target_model.predict(text)[0]

            individual_score = np.round(prediction1[0],4) * 100
            group_score = np.round(prediction1[1],4) * 100

            hs_score = np.round(prediction2[0],4) * 100
            abusive_score = np.round(prediction2[1],4) * 100

            weak_score = np.round(prediction3[0],4) * 100
            moderate_score = np.round(prediction3[1],4) * 100
            strong_score = np.round(prediction3[2],4) * 100

            religion_score = np.round(prediction4[0],4) * 100
            race_score = np.round(prediction4[1],4) * 100
            physical_score = np.round(prediction4[2],4) * 100
            gender_score = np.round(prediction4[3],4) * 100
            other_score = np.round(prediction4[4],4) * 100

        st.write("""
        ### Results

        **Category**
        - Individual: {:10.2f}%
        - Group: {:10.2f}%

        **Hatefulness and Abusiveness**
        - Hatefulness: {:10.2f}%
        - Abusiveness: {:10.2f}%

        **Level**
        - Weak: {:10.2f}%
        - Moderate: {:10.2f}%
        - Strong: {:10.2f}%

        **Target**
        - Religion: {:10.2f}%
        - Race: {:10.2f}%
        - Physical: {:10.2f}%
        - Gender: {:10.2f}%
        - Other: {:10.2f}%
        """.format(individual_score, group_score, hs_score, abusive_score, weak_score, moderate_score, strong_score,
                        religion_score, race_score, physical_score, gender_score, other_score))

def get_table_download_link_csv(df):
#csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f'You can download a sample dataset <a href="data:file/csv;base64,{b64}" download="numbermap.csv" target="_blank">here</a> or view it below.'
    return href

def audit_dataset(cat_model, hs_model, level_model, target_model, deep=False):
    # Upload CSV

    st.write("""

    ## Dataset audit

    Please upload a CSV file containing a single column of text. After uploading, you will be able to download an audited version of your dataset with the selected labels.
    **We will not be storing your dataset at all!** All the preprocessing and inference are done on-the-fly.
    """)


    example = pd.read_csv("example.csv")
    st.markdown(get_table_download_link_csv(example), unsafe_allow_html=True)
    st.write(example)

    st.write('Please upload your CSV file below:')
    uploaded_file = st.file_uploader("If your CSV file exceeds 200MB, please consider splitting your dataset into chunks!", type=["csv"])
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("""
        #### Dataset preview
        """)
        st.write(df.head())

    col = st.text_input("Please enter the column name containing your text and press ENTER or RETURN:", value='')
    if col != '':
        st.write('Column name: ', col)

    st.write('Please select the desired target label predictions:')
    hateful_abusive = st.checkbox('Hatefulness and Abusiveness')
    target = st.checkbox('Target')
    category = st.checkbox('Category')
    level = st.checkbox('Level')

    if st.button("Audit"):
        text_column = df[col]
        X = []

        if not deep:

            for t in text_column.values:
                X.append(preprocess(t))

            if hateful_abusive:
                hs_preds = hs_model.predict_proba(X)
                df["Hatefulness"] = hs_preds[:,0]
                df["Abusiveness"] = hs_preds[:,1]

            if category:
                cat_preds = cat_model.predict_proba(X)
                df["Category_Individual"] = cat_preds[:,0]
                df["Category_Group"] = cat_preds[:,1]

            if level:
                level_preds = level_model.predict_proba(X)
                df["Level_Weak"] = level_preds[:,0]
                df["Level_Moderate"] = level_preds[:,1]
                df["Level_Strong"] = level_preds[:,2]

            if target:
                target_preds = target_model.predict_proba(X)
                df["Target_Religion"] = target_preds[:,0]
                df["Target_Race"] = target_preds[:,1]
                df["Target_Physical"] = target_preds[:,2]
                df["Target_Gender"] = target_preds[:,3]
                df["Target_Other"] = target_preds[:,4]

        else:

            for text in text_column.values:
                t = tokenizer(str(text))['input_ids']
                X.append(t)
                
            X = sequence.pad_sequences(X, maxlen=80)

            if hateful_abusive:
                hs_preds = np.round(hs_model.predict(X), 6)
                df["Hatefulness"] = hs_preds[:,0]
                df["Abusiveness"] = hs_preds[:,1]

            if category:
                cat_preds = np.round(cat_model.predict(X), 6)
                df["Category_Individual"] = cat_preds[:,0]
                df["Category_Group"] = cat_preds[:,1]

            if level:
                level_preds = np.round(level_model.predict(X), 6)
                df["Level_Weak"] = level_preds[:,0]
                df["Level_Moderate"] = level_preds[:,1]
                df["Level_Strong"] = level_preds[:,2]

            if target:
                target_preds = np.round(target_model.predict(X), 6)
                df["Target_Religion"] = target_preds[:,0]
                df["Target_Race"] = target_preds[:,1]
                df["Target_Physical"] = target_preds[:,2]
                df["Target_Gender"] = target_preds[:,3]
                df["Target_Other"] = target_preds[:,4]
            
        st.write("""
        ### Audited dataset preview
        """)
        st.write(df.head())


        # csv = df.to_csv(index=False)
        # b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        # href = f'<a href="data:file/csv;base64,{b64}" download=audited_dataset.csv>Download as CSV</a>'

        # st.markdown(href, unsafe_allow_html=True)

        df_csv = df.to_csv()
        st.download_button("Download as CSV", df_csv, "audited_dataset.csv")

if __name__ == "__main__":

    DEEP = True
    cat_model, hs_model, level_model, target_model = load_models(deep=DEEP)

    st.write("""
    # IndoHateSpeech 🇮🇩🤬

    This web app predicts Indonesian hate speech. This app provides scores (%) for the following:
    - Hatefulness and Abusiveness
    - Target: Religion, Race, Physical, Gender, Other
    - Category: Individual or Group
    - Level: Weak, Moderate, Strong

    The model employs a Recurrent-CNN architecture trained on the [Multilabel Hate Speech and Abusive Language Detection Dataset](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection) with an average accuracy of 78.50 and an F1-score of 0.7035.
    """)

    get_demo(cat_model, hs_model, level_model, target_model, deep=DEEP)
    audit_dataset(cat_model, hs_model, level_model, target_model, deep=DEEP)