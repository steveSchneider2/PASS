# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:02:55 2021

@author: steve
"""
# %% Imports
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import (train_test_split)
from sklearn.metrics import (
                             confusion_matrix, accuracy_score,
                             roc_curve, auc)
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
# from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from datetime import datetime
import pyodbc
import os
import sys
import shap  #for  the Shap charts of course!
import random
import time
from subprocess import check_output
from numpy import genfromtxt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

# import pandas_profiling as pp
os.environ["PATH"] += os.pathsep + 'c:/Program Files (x86)/Graphviz/bin/'
# %% Intro definitions

def setstartingconditions():
    pd.set_option('display.max_columns', 120)
    pd.set_option('display.width', 300)
    #rcParams['figure.figsize'] = 12, 12

    # plot_tree(xgb._Booster)
    start = datetime.now()
    print(start.strftime('%a, %d %B, %Y', ))
    print(start.strftime('%c'))
    mdlsttime = start.strftime('%c')

    try:
        filename = os.path.basename(__file__)
    except NameError:
        filename = 'working'
    # else:
    #     filename = 'unk'

    mdl = xgb.XGBClassifier
    txtmdl = str(mdl).replace('<class \'', '')
    txtmdl = str(txtmdl).replace('\'>', '')
#    print(__doc__)

    pver = str(format(sys.version_info.major) + '.' +
               format(sys.version_info.minor) + '.' +
               format(sys.version_info.micro))
    print('Python version: {}'.format(pver))
    print('SciKitLearn:    {}'.format(sklearn.__version__))
    print('matplotlib:     {}'.format(matplotlib.__version__))
    print('XGBoost:        {}'.format(xgb.__version__))
    print('Conda environment: {:12s}'.format(os.environ['CONDA_DEFAULT_ENV']))
    # plt.style.use('ggplot') # this is used for ALL future plt plots...
    plt.style.use('classic')  # this is used for ALL future plt plots...

    csfont = {'fontname': 'Comic Sans MS'}
    font = {'family': 'monospace',
            'weight': 'normal',
            'size': 12,
            }
    return txtmdl, mdl, mdlsttime, filename, start

# %% getData

def getdata(records):
    start = datetime.now()
    conn = pyodbc.connect('DRIVER={SQL Server};'
                          'SERVER=(local);'
                          'DATABASE=Traffic;'
                          'Trusted_Connection=yes;')
    SqlParameters = f'@cnt={records},@yr=2014,@veh#=1,@day1=3,@day2=28'
#    SqlParameters = '@cnt=35000,@yr=2014,@veh#=2,@day1=11,@day2=19'
    SQL = 'usp_GetEqualNbrMajorMinorCrashesA '
#    SQL = 'select * from Crash110'
    SQLCommand = SQL
    SQLCommand = SQL + SqlParameters
    # SQLCommand = ("usp_GetEqualNbrMajorMinorCrashes  @cnt=100,@yr=2014,@veh#=1")
    # SQLCommand = 'select * from vw_CrashSubsetEngineeredR'  #' where vehmph > 0 '
    dt_string = start.strftime("%m/%d/%Y %H:%M:%S")
    dt_string = start.strftime("%m/%d/%Y %H:%M")
    dt_string
    url = 'https://github.com/steveSchneider2/data/blob/main/FloridaTraffic/traffic116k_88_76pc.csv?raw=true'
#    url = 'https://github.com/steveSchneider2/Machine-Learning/blob/main/traffic236k.csv?raw=true'
    pr = pd.read_csv(url)
#    pr = pd.read_csv('D:/ML/Python/data/traffic/traffic236k_95_16.csv', header=0)
#    pr = pd.read_csv('data/traffic72k_94_83.csv', header=0)
    #pr = pd.read_csv('D:/dDocuments/ML/Python/data/Traffic/traffic116k_88_76pc.csv')
#    pr = pd.read_sql(SQLCommand, conn)
    end = datetime.now()
    processTime = end - start
    print('\nFunction "getdata()": Seconds to download SQL records is: ', processTime.total_seconds())
#    pr.to_csv(r'traffic236k.csv', index=False)
    return pr, SQLCommand, SQL, SqlParameters, dt_string

# %% def datapreparation() Feature Engineering
def datapreparation():
    pr.drop(['Crash_year', 'monthday','speeddif','vMphGrp'], axis=1,
            inplace=True)
    if 'vehicle_number' in pr.columns:
        pr.pop('vehicle_number')  # Return item and drop from frame
    if 'InvolvedVeh' in pr.columns:
        pr.pop('InvolvedVeh')  # Return item and drop from frame
    todummylist = list(pr.select_dtypes(np.object).columns)
    class_names_gnb = (['minor', 'major'])
    pr.fillna(method='pad', inplace=True)
    j = pd.concat([pr['vehmph'],pr.crash], axis=1)
    major = j.vehmph[j.crash==1]
    minor = j.vehmph[j.crash==0]
    j = pd.concat([pr.rdSurfac,pr.crash], axis=1)
    major = j.rdSurfac[j.crash==1]
    minor = j.rdSurfac[j.crash==0]
    # if howfartoexecute != 'learningcurves':
#    plot_histograms_stacked(major, minor, 3,
#                            'Actual accidents vs Road Surface',
#                            labels)

    if 'crash' in pr.columns:  # Allow me to run this frm multiple places in code
        y = pr.pop('crash')  # Return item and drop from frame
    X = pr.copy()
    X.info(memory_usage='deep')

    # Function to dummy all the categorical variables used for modeling
    # courtesy of April Chen
    def dummy_pr(pr, todummylist):
        for x in todummylist:
            dummies = pd.get_dummies(pr[x], prefix=x, dummy_na=False)
            pr = pr.drop(x, 1)
            pr = pd.concat([pr, dummies], axis=1)
        return pr
    X = dummy_pr(X, todummylist)
#    X = standardize(X)
    trng_prct = .75
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=1 - trng_prct, random_state = 200)
    txt0 = f'Given ~{pr.shape[0]} records, {trng_prct*100}% are used for training.'
    txt1 = f'  That leaves {len(y_test)} for testing (evaluating the model).'
    txt2 = 'Those are the numbers reflected in this chart.'
    testsummary = ('{}').format(txt0) + '\n' + ('{}').format(txt1) +\
        '\n' + ('{}').format(txt2)
    categoricalVariablesList = pr.describe(
        include=['object', 'category']
        ).transpose().sort_values(by='unique', ascending=False)

    categoricalVariablesList = categoricalVariablesList.to_string(header=False)

    return X_train, X_test, y_train, y_test, pr, categoricalVariablesList,\
        testsummary, X, y, class_names_gnb, major, minor

# %% DefineCreateModel
def DefineCreateModel():
    # Choose all predictors except target & IDcols
    # https://stats.stackexchange.com/questions/204489/discussion-about-overfit-in-xgboost
    # https://www.capitalone.com/tech/machine-learning/how-to-control-your-xgboost-model/
    # https://towardsdatascience.com/xgboost-theory-and-practice-fb8912930ad6
    target = 'crash'
    predictors = [x for x in X_train.columns if x not in [target]]  # , IDcol]]
    # predictors = y_train.columns
    params = {
        "tree_method": "hist",  #
#        'tree_method': 'gpu_hist',  # src/learner.cc:180:
        "n_estimators": 1700,  # 1200,
        "learning_rate": .3,  # 0.02,
        # Regularization:
        # Below: 3 'frees tng to drop'; 30 'brings up trng significantly'
        'reg_alpha': 2,  # L1 0-1000 increase value to decrease overfitting
        "reg_lambda": 3,  # L2 0-1000 increase value to decrease overfitting
        'max_delta_step': 1,  # This seems ignored.. 3 to 13 had no affect
        # Pruning:  removes splits directly from the trees during or after the build process
        "max_depth": 5,  # increase..improves training, but can overfit
        "min_child_weight": 2,  #Pretty sensitive: raise to raise up training
        # Dropping gamma frees training to drop, no impact on test! (:
        "gamma": .5,  # <--Drop from 5 thru 8 below .8 dropped log loss!  TRAINING impact!!!
        # Sampling:
        "subsample": 0.95,  # (bagging of rows) (def=1); help w/overfitting
        "colsample_bytree": 0.95,  # avoid some columns to take too much credit
        'colsample_bylevel': 0.3,
#        'colsample_bynode': 0.6,
        "objective": 'binary:logistic',
        "nthread": -1,
        "scale_pos_weight": 1,
        # "eval_metric": "auc",  # vs "logloss"
        # "seed": 27,
        'random_state': 200,
        "verbosity": 1,
        #        'single_precision_histogram': True,
        "n_jobs": -1}

    xgb1 = XGBClassifier(**params)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    start = datetime.now()
    xgb1.fit(X_train, y_train, early_stopping_rounds=20,
#              eval_metric=["error", "logloss"], eval_set=eval_set, verbose=100)
              eval_metric=["error", "logloss"], eval_set=eval_set, verbose=100)
    end = datetime.now()
    getdataTime = end - start
    print('\nSeconds to train model is: ', getdataTime.total_seconds())

    predictions = xgb1.predict(X_test)
    actuals = y_test
    print('Confusion Matrix: ')
    print(confusion_matrix(actuals, predictions))
    accuracy = accuracy_score(y_test, predictions)
    print('Accuracy: %.2f%%' % (accuracy * 100))
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, predictions)

    auc_xgb = round(auc(fpr_xgb, tpr_xgb), 2)
    cm = confusion_matrix(y_test, predictions)

    # retrieve performance metrics
    results = xgb1.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    mdlTrainingSeconds = round(getdataTime.total_seconds(),2)
    if mdlTrainingSeconds > 60:
        mdlTrainingSeconds = round(mdlTrainingSeconds / 60,3)
        mdlTngTime = str(mdlTrainingSeconds) + ' Mins'
    else:
        mdlTngTime = str(mdlTrainingSeconds) + ' Sec'
    cm = confusion_matrix(y_test, predictions)
    return xgb1, predictions, actuals, accuracy, params, results, \
        x_axis, epochs, mdlTngTime, cm
# Note 'xgb1' is the resulting model!!!
#%% MAIN Code:
txtmdl, mdl, mdlsttime, filename, start = setstartingconditions()

pr, SQLCommand, SQL, SqlParameters, dt_string = getdata(118000)  #gets twice

X_train, X_test, y_train, y_test, pr, categoricalVariablesList,\
        testsummary, X, y, class_names_gnb, major, minor = \
        datapreparation()

print(testsummary)

# print(X_train.describe())
# print(y_train.head())
X_test.describe()

print('\n We are starting to model... \n')
xgb1, predictions, actuals, accuracy, params, results, x_axis, epochs,\
    mdlTngTime, cm = DefineCreateModel()
print('\n done modeling... \n')
# Saving the model
import pickle
pickle.dump(xgb1, open('FloridaTraffic2014xgb.pkl', 'wb'))

# https://kthaisociety.medium.com/using-shap-to-explain-machine-learning-models-3f8f9c3b1f5e
explainer = shap.TreeExplainer(xgb1)
print('Need 19 minutes to do these shap calculations... figure shap_values')
shap_values = explainer.shap_values(X)  # 19+ minutes!  @depth = 9
print('\n now: 3 different plots')
shap.summary_plot(shap_values, X, feature_names=X.columns)  # 12 sec
shap.summary_plot(shap_values, X, feature_names=X.columns, plot_type='violin' )  # 12 sec This is just smoother than the onve above
shap.summary_plot(shap_values, X, feature_names=X.columns, plot_type='bar' )  # 12 sec
#Put in chart: The function automatically includes another variable that your chosen variable interacts most with. 
shap.dependence_plot('vehmph', shap_values, X) #vehtype_car is chosen
shap.dependence_plot('roadtyp_County', shap_values, X) #urbanLoc_no is chosen
shap.dependence_plot('vehage', shap_values, X) #drSitu_unknown is chosen
shap.dependence_plot('dayhour', shap_values, X) #vehmph is chosen
#shap.plots.scatter(shap_values[:,1])
# https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/tree_explainer/Catboost%20tutorial.html
#do below in notebook...
#shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

# https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d
#shap_v, shap_abs, k2, colorlist = ABS_SHAP(shap_values,X)
# shap_v = ABS_SHAP(shap_values,X)
#shap_values = plot_SHAP_charts(k2, 18)  # <--THREE charts! ~30 sec for all 7

