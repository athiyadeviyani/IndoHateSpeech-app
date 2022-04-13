import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

app = Flask(__name__)
model1 = pickle.load(open('rf_cat.sav', 'rb'))
model2 = pickle.load(open('rf_hs.sav', 'rb'))
model3 = pickle.load(open('rf_level.sav', 'rb'))
model4 = pickle.load(open('rf_target.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    # feature_list = list(feature_list.values())
    # feature_list = list(map(int, feature_list))
    # final_features = np.array(feature_list).reshape(1, 12) 
    final_features = [list(feature_list.values())[0]]
    
    prediction1 = model1.predict_proba(final_features)[0]
    prediction2 = model2.predict_proba(final_features)[0]
    prediction3 = model3.predict_proba(final_features)[0]
    prediction4 = model4.predict_proba(final_features)[0]

    prediction_text=' CATEGORY:  Individual={} Group={} <br> \
    HATE:     Hate_Speech={} Abusive={} <br> \
    LEVEL:    Weak={} Moderate={} Strong={} <br> \
    TARGET:   Religion={} Race={} Physical={} Gender={} Other={} <br>'.format(
        str(prediction1[0]), str(prediction1[1]), 
        str(prediction2[0]), str(prediction2[1]), 
        str(prediction3[0]), str(prediction3[1]), str(prediction3[2]), 
        str(prediction4[0]), str(prediction4[1]), str(prediction4[2]), str(prediction4[3]), str(prediction4[4]))
    # print(prediction_text)

    cat_pred = 'CATEGORY:  Individual={} Group={}'.format(str(prediction1[0]), str(prediction1[1]))
    hs_pred = 'HATE:     Hate_Speech={} Abusive={}'.format(str(prediction2[0]), str(prediction2[1]))
    level_pred = 'LEVEL:    Weak={} Moderate={} Strong={}'.format(str(prediction3[0]), str(prediction3[1]), str(prediction3[2]))
    target_pred = 'Religion={} Race={} Physical={} Gender={} Other={}'.format(str(prediction4[0]), str(prediction4[1]), str(prediction4[2]), str(prediction4[3]), str(prediction4[4]))

    return render_template('index.html', input_text='input text="{}"'.format(final_features[0]), 
    cat_pred=cat_pred,
    hs_pred=hs_pred,
    level_pred=level_pred,
    target_pred=target_pred)


if __name__ == "__main__":
    app.run(debug=True)