import streamlit as st
import pandas as pd
import numpy as np
from preprocess import preprocess
import pickle

PREDICT = False 

# st.beta_set_page_config(page_title='IndoHateSpeech')

# LOAD MODELS
cat_model = pickle.load(open('rf_cat.sav', 'rb'))
hs_model = pickle.load(open('rf_hs.sav', 'rb'))
level_model = pickle.load(open('rf_level.sav', 'rb'))
target_model = pickle.load(open('rf_target.sav', 'rb'))

st.write("""
# IndoHateSpeech 🇮🇩🤬

This web app predicts Indonesian hate speech. This app provides scores (%) for the following:
- Hatefulness and Abusiveness
- Target: Religion, Race, Physical, Gender, Other
- Category: Individual or Group
- Level: Weak, Moderate, Strong

The model employs a Recurrent-CNN architecture trained on the [Multilabel Hate Speech and Abusive Language Detection Dataset](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection) with an average accuracy of 78.50 and an F1-score of 0.7035.
""")


st.write("""

## Demo
""")


text = st.text_input("For a quick demo, type an example sentence and press ENTER or RETURN:", value='')

if text != '':
    st.write('Input: ', text)
    text = [text]
    prediction1 = cat_model.predict_proba(text)[0]
    prediction2 = hs_model.predict_proba(text)[0]
    prediction3 = level_model.predict_proba(text)[0]
    prediction4 = target_model.predict_proba(text)[0]

    individual_score = round(prediction1[0],2) * 100
    group_score = round(prediction1[1],2) * 100

    hs_score = prediction2[0] * 100
    abusive_score = round(prediction2[1],2) * 100

    weak_score = round(prediction3[0],2) * 100
    moderate_score = round(prediction3[1],2) * 100
    strong_score = round(prediction3[2],2) * 100

    religion_score = round(prediction4[0],2) * 100
    race_score = round(prediction4[1],2) * 100
    physical_score = round(prediction4[2],2) * 100
    gender_score = round(prediction4[3],2) * 100
    other_score = round(prediction4[4],2) * 100

    st.write("""
    ### Results

    **Category**
    - Individual: {}%
    - Group: {}%

    **Hatefulness and Abusiveness**
    - Hatefulness: {}%
    - Abusiveness: {}%

    **Level**
    - Weak: {}%
    - Moderate: {}%
    - Strong: {}%

    **Target**
    - Religion: {}%
    - Race: {}%
    - Physical: {}%
    - Gender: {}%
    - Other: {}%
    """.format(individual_score, group_score, hs_score, abusive_score, weak_score, moderate_score, strong_score,
                    religion_score, race_score, physical_score, gender_score, other_score))

# Upload CSV

st.write("""

## Dataset audit

Please upload a CSV file containing a single column of text. After uploading, you will be able to download an audited version of your dataset with the selected labels.
""")

st.write('Please upload your CSV file below:')

uploaded_file = st.file_uploader("If your CSV file exceeds 200MB, please consider splitting your dataset into chunks!", type=["csv"])
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

if st.button("Audit!"):
    text_column = df[col]
    X = []

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
        
    st.write("""
    ### Audited dataset preview
    """)
    st.write(df.head())
    df_csv = df.to_csv()
    st.download_button("Download as CSV", df_csv, "audited_dataset.csv")