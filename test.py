import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle

final_features = ['antek antek jokowi']

model1 = pickle.load(open('rf_cat.sav', 'rb'))
model2 = pickle.load(open('rf_hs.sav', 'rb'))
model3 = pickle.load(open('rf_level.sav', 'rb'))
model4 = pickle.load(open('rf_target.sav', 'rb'))

prediction1 = model1.predict(final_features)
prediction2 = model2.predict(final_features)
prediction3 = model3.predict(final_features)
prediction4 = model4.predict(final_features)

prediction_text='cat={} hs={} level={} target={}'.format(str(prediction1), str(prediction2), str(prediction3), str(prediction4))
print(prediction_text)