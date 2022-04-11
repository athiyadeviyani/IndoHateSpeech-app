import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle

final_features = ['antek antek jokowi']

model1 = pickle.load(open('rf_cat.sav', 'rb'))
model2 = pickle.load(open('rf_hs.sav', 'rb'))
model3 = pickle.load(open('rf_level.sav', 'rb'))
model4 = pickle.load(open('rf_target.sav', 'rb'))

prediction1 = model1.predict_proba(final_features)[0]
prediction2 = model2.predict_proba(final_features)[0]
prediction3 = model3.predict_proba(final_features)[0]
prediction4 = model4.predict_proba(final_features)[0]

prediction_text=' CATEGORY:  Individual={} Group={} \n \
HATE:     Hate_Speech={} Abusive={} \n \
LEVEL:    Weak={} Moderate={} Strong={} \n \
TARGET:   Religion={} Race={} Physical={} Gender={} Other={}'.format(
    str(prediction1[0]), str(prediction1[1]), 
    str(prediction2[0]), str(prediction2[1]), 
    str(prediction3[0]), str(prediction3[1]), str(prediction3[2]), 
    str(prediction4[0]), str(prediction4[1]), str(prediction4[2]), str(prediction4[3]), str(prediction4[4]))
print(prediction_text)