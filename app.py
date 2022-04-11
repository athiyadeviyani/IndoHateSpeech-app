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
    final_features = list(feature_list.values())[0]
    
    prediction1 = model1.predict(final_features)
    prediction2 = model2.predict(final_features)
    prediction3 = model3.predict(final_features)
    prediction4 = model4.predict(final_features)
    # output = int(prediction[0])
    # if output == 1:
    #     text = ">50K"
    # else:
    #     text = "<=50K"

    return render_template('index.html', prediction_text='cat={} hs={} level={} target={}'.format(str(prediction1), str(prediction2), str(prediction3), str(prediction4)))


if __name__ == "__main__":
    app.run(debug=True)