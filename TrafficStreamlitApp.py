# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:12:28 2021
@author: steve

This python when run opens up a web-app in your browser.
It reads in a 'pkl' file...which is a ML model.
In the web-app, you can modify 4 features to get a prediction.
"""
#%% Imports...
import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
#import shap
from datetime import datetime
from PIL import Image

pd.set_option('display.max_columns', 250)
pd.set_option('display.width', 250)

# Reads in saved classification model
#%% dataprep function
def datapreparation(pr):
    pr.drop(['Crash_year', 'monthday','speeddif','vMphGrp'], axis=1,
            inplace=True)
    if 'vehicle_number' in pr.columns:
        pr.pop('vehicle_number')  # Return item and drop from frame
    if 'InvolvedVeh' in pr.columns:
        pr.pop('InvolvedVeh')  # Return item and drop from frame
    todummylist = list(pr.select_dtypes(np.object).columns)
    pr.fillna(method='pad', inplace=True)
    if 'crash' in pr.columns:  # Allow me to run this frm multiple places in code
        y = pr.pop('crash')  # Return item and drop from frame
    X = pr.copy()
    # Function to dummy all the categorical variables used for modeling courtesy of April Chen
    def dummy_pr(pr, todummylist):
        for x in todummylist:
            dummies = pd.get_dummies(pr[x], prefix=x, dummy_na=False)
            pr = pr.drop(x, 1)
            pr = pd.concat([pr, dummies], axis=1)
        return pr
    X = dummy_pr(X, todummylist)
    return X

#%% Main code
col1, col2, col3 = st.columns((1,3,1))
image = Image.open('PASSicon.jpg')
col2.image(image, use_column_width=True)

image2 = Image.open('SQLinsightLogo.png')
#col2.image(image2, use_column_width=True)
#col2.write("[SQLinsight](https://SQLinsight.net)") 

st.sidebar.image(image2, use_column_width=True)
start = datetime.now()
xgb1 = pickle.load(open('FloridaTraffic2014xgb.pkl', 'rb'))
# xgb1 = pickle.load(open('https://github.com/steveSchneider2/PASS/FloridaTraffic2014xgb.pkl?raw=true', 'rb'))

url = 'https://github.com/steveSchneider2/data/blob/main/FloridaTraffic/traffic116k_88_76pc.csv?raw=true'

st.title('Car Crash Prediction App  \nStreamlit: {:.6s}  Conda environment: MLFlowProtobuf'.
         format(st.__version__))

# https://kthaisociety.medium.com/using-shap-to-explain-machine-learning-models-3f8f9c3b1f5e
# explainer = shap.TreeExplainer(xgb1)
# print('Need 19 minutes to do these shap calculations... figure shap_values')
# shap_values = explainer.shap_values(X)  # 20 sec!  @depth = 9
# print('\n now: 3 different plots')
# shap.summary_plot(shap_values, X, feature_names=X.columns)  # 12 sec
# shap.summary_plot(shap_values, X, feature_names=X.columns, plot_type='violin' )  # 12 sec This is just smoother than the onve above
# shap.summary_plot(shap_values, X, feature_names=X.columns, plot_type='bar' )  # 12 sec

st.sidebar.header('Change these values for a new prediction')
#st.sidebar.write("[SQLinsight](https://SQLinsight.net)") 
@st.cache
def readdata():
    pr = pd.read_csv(url)
    X = datapreparation(pr)
    return X
X = readdata()

end = datetime.now()
processTime = end - start
print('\nFunction "getdata()": Seconds to download SQL records is: ', processTime.total_seconds())
meanX = pd.DataFrame(X.mean().to_dict(),index=[X.index.values[-1]])

# Collects user input features into dataframe
def user_input_features():
#    sex = st.sidebar.selectbox('Sex',('male','female'))
    speed = st.sidebar.slider('Speed', 0,99,31)
    drvage = st.sidebar.slider('Driver Age', 12,99,44)
    vehage = st.sidebar.slider('Veh Age', 1,23,7)
    dayhour = st.sidebar.slider('Hour of Day', 0,23,7)
    data = {
            'vehmph': speed,
            'drvage': drvage,
            'vehage': vehage,
            'dayhour': dayhour}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

meanX[['vehmph','drvage','vehage','dayhour']] = input_df[['vehmph','drvage','vehage','dayhour']].values

# Print specified input parameters
st.header('Specified Input parameters')
st.write(meanX)
#st.write('---')

# Apply model to make predictions
prediction_proba = xgb1.predict_proba(meanX)
prediction = xgb1.predict(meanX)
#prediction_proba = 
xgb1.predict_proba(X[1:2])

st.header('Prediction of Crash Severity')

if prediction == 0:
    Crash = 'Fender Bender with confidence of: ' + '{:.1%}'.format(prediction_proba.item(0),2)
    st.markdown(f'<p style="text-align:center;background-image: linear-gradient(to right,#1aa3ff, #00ff00);color:#ffffff;font-size:24px;border-radius:2%;">{Crash}</p>', unsafe_allow_html=True)
else:
    Crash = 'BIG CRASH! with confidence of: ' + '{:.1%}'.format(prediction_proba.item(1),2)
    st.markdown(f'<p style="text-align:center;background-image: linear-gradient(to right,#ff1af3, #f3e124);color:#1313d8;font-size:24px;border-radius:2%;">{Crash}</p>', unsafe_allow_html=True)

st.write('---')

st.write("""This app predicts the likelyhood that a crash is 'major' or not!
\nThis DOES NOT predict crashes.  
If you DO have an accident, it predicts whether or not it is bad.
""")

col1, col2 = st.columns((3,1))
#image = Image.open('PASSicon.jpg')
col1.write("Many thanks to the [Data Professor](https://www.youtube.com/dataprofessor) for teaching me about streamlt!") 
image2 = Image.open('SQLinsightLogo.png')
col2.image(image2, use_column_width=True)
col2.write("[SQLinsight](https://SQLinsight.net)") 
