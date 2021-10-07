"""
Created on Thu Sep 17 08:06:27 2020
Stopped work on Nov 30 2020 after achieving 95% accuracy! :) 
Re-ran on 1 Mar 2021 on Python 3.8

RUNTIME: 13 minutes (10 Sep 2021)
INPUT:  SQL SERVER TRAFFIC database; 240,000 records.
OUPUT:   26 graphs
NEEDS Python 3.7 and up. (for plot_confusion_matrix, at least.)
Notes:
     Anaconda install shap, graphViz
    conda install:
        python -m pip install xgboost
        python -m pip install shap==0.36.0


Spyder Editor
model = XGBoost
experimenting with
1. reducing the number of columns (factors) (now 10)
2. removing vehmph...because it might be 'overwhelming the other '
3. Neither of the above worked...still lousy response
4. I did find how to show parameter importance
5. Permutation Importances! :)
6. Consider scaling, or standardization of the mph. (no apparant impact!)

ENVIronment: MLFlow
Python version: 3.8.8
SciKitLearn:    0.24.2
matplotlib:     3.4.1
XGBoost:        1.4.0
July 26, 2021

a GREAT site on Learning curves...
https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/

"""
#%% Code setup Imports & starting conditions
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     learning_curve, ShuffleSplit)
from sklearn.metrics import (mean_squared_error, plot_confusion_matrix,
                             confusion_matrix, accuracy_score,
                             classification_report, roc_curve, auc,
                             recall_score, precision_score,
                             average_precision_score, plot_roc_curve,
                             brier_score_loss, f1_score)
from sklearn.metrics import matthews_corrcoef
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
# from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from datetime import datetime
import json
import pyodbc
import os
import sys
import shap  #for  the Shap charts of course!
import random
import time
from subprocess import check_output
from numpy import genfromtxt
import seaborn as sns
from sklearn import preprocessing
from operator import itemgetter

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

# import pandas_profiling as pp
os.environ["PATH"] += os.pathsep + 'c:/Program Files (x86)/Graphviz/bin/'
fignbr = 0
def setstartingconditions():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
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

    # plt.style.use('ggplot') # this is used for ALL future plt plots...
    plt.style.use('classic')  # this is used for ALL future plt plots...

    csfont = {'fontname': 'Comic Sans MS'}
    font = {'family': 'monospace',
            'weight': 'normal',
            'size': 12,
            }
    return txtmdl, mdl, mdlsttime, filename, start

#%% EDA --> Charts Exploratory Data Analysis
def plot_histograms_stacked(major, minor, bins=100,
                            title='Fig 1: Actual Accidents vs speed',
                            xlabel = 'Vehicle speed'):
    fig, ax = plt.subplots(2, 1, figsize=(16, 12))
    colors = ['blue', 'red']
    labels = f'{len(minor)} little crashes',f'{len(major)} big crashes'
    global fignbr; fignbr+=1
    title = f'Fig {fignbr}: Actual Accidents vs speed'
    ax[0].set_title(title, fontsize=24)
    # plt.xticks(np.arange(0, 100, 10))
    ax[0].set_xlabel(xlabel)
    ax[1].set_xlabel(labels)
    ax[0].hist([minor,major],bins=bins, stacked=True, color=colors)#,  width=.5)
    ax[1].hist([minor,major],bins=bins, stacked=True, log=True, color=colors, label=labels)#, width=.5)
    plt.legend(prop={'size': 16})
    plt.grid()
    #Next line minimizes 'padding' size outside of actual chart
    plt.subplots_adjust(left=.05, right=.95, top=.95, bottom=.05)
    plt.show()

def plot4histograms():
    # #1...
    j = pd.concat([X_test['vehmph'],y_test], axis=1)
    major = j.vehmph[j.crash==1]
    minor = j.vehmph[j.crash==0]
    global fignbr 
    plot_histograms_stacked(major, minor, 
                            title=f'Fig {fignbr}: TEST Data- what the validation data looks like...')
    plt.show()
    kwargs = dict(histtype='stepfilled', alpha=0.8, bins=90)
    labels = f'{pr.shape[0]} crashes'
    pr.hist(**kwargs)
    fignbr += 1
    plt.subplots_adjust(left=.1, right=.9)
    plt.suptitle(f'Fig {fignbr}: The Four integer factors')
    plt.xlabel(f'{labels}')
    # #2... the 'Drivers Situation' has 3 cases: Normal, unk, DrugAlcSleepSick...so you get 3 graphs
    pr.hist(['vehmph'],['drSitu'])  # 3 variables: 2 listed + count
    fignbr += 1
    plt.suptitle(f"Fig {fignbr}:Vehicle SPeed vs Driver Condition")
    # #3 of 3 histograms...
    pr.hist(['vehmph'],['hitrun'])
    fignbr += 1
    plt.suptitle(f'Fig {fignbr}: Vehicle speed vs Hit & Runs')
    plt.show()
    return

    # X_test.hist(['vehmph'], bins=90, width=.5)  # already shown via plot_histogram
    # X_test.hist(['vehmph'], **kwargs)  # already shown via plot_histogram
    # #4...
# https://towardsdatascience.com/powerful-eda-exploratory-data-analysis-in-just-two-lines-of-code-using-sweetviz-6c943d32f34
def mySweetviz():
    import sweetviz
    startSW = datetime.now()
    
    sweetTrain = X_train.copy()
    sweetY = pd.DataFrame(y_train)
    # Add the Crash column so SWEETVIZ can visualize it...
    sweetTrain['crash'] = sweetY
    sweetTrain.describe()
    sweetTrain.head()
    feature_config \
        = sweetviz.FeatureConfig(\
        skip=['hitrun_No','drvagegrp_older','rdSurfac_oil','rdSurfac_mud_dirt_gravel', \
              'rdSurfac_ice_frost','rdSurfac_Unk','rdSurfac_Water','rdSurfac_Sand',\
              'rdSurfac_Other','drvsex_Female','Vision_clear','drSitu_unknown',\
              'drDistract_No','license_valid','vhmtn_Working','vhmtn_Unknown',\
              'vhmtn_Parked','vhmvnt_U_turning','vhmvnt_StopInTraf','vhmvnt_Passing',\
              'vhmvnt_Parked','vhmvnt_ExitTrfc','vhmvnt_EnterTrfc','UrbanLoc_No',\
              'light_unk','light_dawn','weather_danger','vehmakegrp_Chrysler',\
              'vehmakegrp_Euro','vehmakegrp_Volvo','vehmakegrp_Korean',\
              'vehmakegrp_Volvo','wday_Monday','wday_Tuesday','wday_Wednesday',\
              'wday_Thursday','wday_Saturday','wday_Sunday','vehtype_other',\
              'vehtype_truck','vehtype_van'])
    my_report = sweetviz.compare([sweetTrain, "Train"], 
                                  [X_test, "Test"], 'crash',
                                  feature_config)
    my_report.show_html("Report.html")
    end = datetime.now()
    sweetDuration = end - startSW
    print(f'SweetViz duration: {sweetDuration}')
def myAutoviz():
    #Here, we are importing the Autoviz class
    from autoviz.AutoViz_Class import AutoViz_Class
    #Here, we instantiate the Autoviz class
    AV = AutoViz_Class()
    
    #Creates automated visualizations NEXT:  Can we do it with the df...
    df = AV.AutoViz('D:/ML/Python/data/traffic/traffic236k_95_16.csv')

def myPandaProfile(): #Run in notebook, not here in Spyder
    i=1 # test
# plot3histograms()
#%% EMP --> Charts Explore the modeling process and results
def plot_learningCurves_LogLoss_ClassError():
    explainLearnCurve = \
        ('A learning curve shows the validation and training score of an estimator for varying numbers of ' +
        '\ntraining samples. It is a tool to find out how much we benefit from adding more'+
    ' training data and whether the estimator suffers\n more from a variance error or a bias error.'+
    '\nLog Loss refers to how close the probability predictions approach either 0 or 1'+
    '\nHigh log loss would equate to lower confidence...because each prediction might be "easily" '+
    '\nswitched (when close to .5) As of 1 Sept, the below is a "good fit" ! :) ')
    # below is yet another way to print
    # print(format('How to visualise XGBoost model with learning curves?', '*^82'))
    # https://www.dezyre.com/recipes/evaluate-xgboost-model-with-learning-curves-example-2
    font = {'family': 'monospace',
            'weight': 'normal',
            'size': 12,
            }
    cm = confusion_matrix(y_test, predictions)
    mc = 100 * round(matthews_corrcoef(y_test, predictions),4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
    plt.rcParams.update({'figure.autolayout': True})
    global fignbr; fignbr+=1
    fig.suptitle(f'Fig #{fignbr}:   Accuracy: {round(accuracy,3)} LEARNING CURVES {mdlsttime}' +
                 f'  Last tree: {xgb1.get_booster().best_iteration}' +
                 f'\n{filename};  TrainingTime: {mdlTngTime}; Model: {txtmdl}',
                 fontsize=20)
    trans1 = transforms.blended_transform_factory(ax1.transAxes, ax1.transAxes)
    trans2 = transforms.blended_transform_factory(ax2.transAxes, ax2.transAxes)
    ax1.set_title('Log Loss')
    ax2.set_title('Classification Error')
    ax1.grid(True)
    ax2.grid(True)
    ax1.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax1.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax1.text(.1, .91, s = f'Model Params:\n{txtparams}', ha='left',
             wrap=True, va='top', fontsize=14, transform=trans1,
             bbox=dict(facecolor='aqua', alpha=0.5))  # , font=font)  #\
    ax1.text(.05, 0.25, transform=trans1,
             s='ax.plot(x_axis, results[\'validation_0\'][\'logloss\'],' +
             'label=\'Train\')\nTell .fit() to get metric data...'
             '\nxgb.fit(X_train, y_train, eval_metric=["error", "logloss"]'
             f'\n\n{SQL} \n{SqlParameters}', wrap=True,
             ha='left', va='bottom',
             fontsize=14, bbox=dict(facecolor='pink', alpha=0.5))
    ax1.text(0.95, .80, hitnmiss, ha='right', transform=trans1,
             bbox=dict(facecolor='pink', alpha=0.5), fontsize=20)  # ,
    ax1.text(0.95, .60, ha='right', transform=trans1,
             s=f'Confusion Matrix\n{cm[0,0]} {cm[0,1]}\n{cm[1,0]} {cm[1,1]}',
             fontsize=20, bbox=dict(facecolor='pink', alpha=0.5))  #

    ax2.text(.15, .95, '   Categorical Features:\n' + categoricalVariablesList,
             ha='left', transform=trans2, font = font, va='top',
             bbox=dict(facecolor='yellow', alpha=0.5))  #
    ax2.text(0.85, .45, s=f'MathCoef={mc}', ha='right', transform=trans2,
             bbox=dict(facecolor='pink', alpha=0.5), fontsize=20)  # ,
    ax2.text(.85, .02, xgbreport, ha='right', fontsize=16,
             bbox=dict(facecolor='blue', alpha=0.3), transform=trans2)
    ax2.plot(x_axis, results['validation_0']['error'], label='Train')
    ax2.plot(x_axis, results['validation_1']['error'], label='Test')
#    ax2.set_ylim(.1, .15)
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    fig.tight_layout()
    plt.show()


def ROC_rcvr_operating_curve(model):  # Chart #3
    font = {'family': 'monospace',
            'weight': 'normal',
            'size': 10,
            }
    cm = confusion_matrix(y_test, predictions)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr_xgb, tpr_xgb)
    fig = plt.figure(figsize=(16, 12))
    lw = 2
    pt2tr = 1
    parms = {"color": 'black'}
    plt.rcParams.update({'figure.autolayout': True})

    plot_roc_curve(model, X_test, y_test, **parms)
    # plt.plot(fpr_xgb, tpr_xgb, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.text(pt2tr, .9, horizontalalignment='right',
             s=f'Test Accuracy: {round(accuracy,3)}' +
             f' at: {dt_string}\nModel training time: {mdlTngTime}; {os.path.basename(__file__)}',
             font={'size': 14}, bbox=dict(facecolor='pink', alpha=0.5))  #
    plt.text(pt2tr, .85, f'Model Params:\n{txtparams}', ha='right',
             wrap=True, verticalalignment='top', fontsize=10,
             bbox=dict(facecolor='aqua', alpha=0.5))  # , font=font)  #
    plt.text(.6, .6, xgbreport, horizontalalignment='right', fontsize=10,
             bbox=dict(facecolor='blue', alpha=0.3))
    plt.text(.25, 1.0, hitnmiss, ha='right', va='top',
             bbox=dict(facecolor='pink', alpha=0.5), font=font)  # fontsize=14,
    plt.text(pt2tr, .15, horizontalalignment='right',
             s=f'Confusion Matrix\n{cm[0,0]} {cm[0,1]}\n{cm[1,0]} {cm[1,1]}',
             font=font, bbox=dict(facecolor='pink', alpha=0.5))  #
    plt.text(.05, .05, '   Categorical Features:\n' + categoricalVariablesList,
             horizontalalignment='left', font=font,  # transform=trans,
             bbox=dict(facecolor='yellow', alpha=0.5), va='bottom')  #
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    global fignbr; fignbr+=1
    plt.title(f'Fig {fignbr}: ROC curve {XGBClassifier.mro()[0]}, {dt_string}', fontsize=14)
    plt.legend(loc="lower right")
    fig.tight_layout()
    plt.show()


def plot_ConfusionMatrixChart(xgb1):  # Chart #4  Chart #5
    plt.rcParams.update({'figure.autolayout': False})
    cm = confusion_matrix(y_test, predictions)
    global fignbr; fignbr+=1
    ttltxt = f'Fig {fignbr}: XGBoost at: {mdlsttime}\nConfusion matrix, w/o normalization'
    fignbr+=1
    ttltxt2 = f'Fig {fignbr}: XGBoost at: {mdlsttime}\nConfusion matrix from {filename}'
    titles_options = [(ttltxt, None), (ttltxt2, 'true')]
    # plt.rcParams.update({'font.size': 22})  # increase/set internal fonts
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(xgb1, X_test, y_test,
                                     display_labels=class_names_gnb,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize,
                                     values_format='1.2f')  # Nbrs NOT in sci notat
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
        plt.text(.3, -.1, testsummary, fontsize=14,  # FancyBboxPatch
                 horizontalalignment='center', verticalalignment='bottom',
                 bbox=dict(facecolor='yellow', alpha=0.5))
        plt.text(0.45, .44, hitnmiss, horizontalalignment='right',
                  bbox=dict(facecolor='green', alpha=0.5))  # fontsize=14,
        plt.text(.39, 1.55, xgbreport, horizontalalignment='right', fontsize=9,
                 bbox=dict(facecolor='blue', alpha=0.3))
        plt.text(.55, .44,keyparams, horizontalalignment='left',
                 bbox=dict(facecolor='aqua', alpha=.5))
        plt.text(-.7, .64,SQLCommand, horizontalalignment='left', wrap=True,
                 bbox=dict(facecolor='aqua', alpha=.5), fontsize=8)
    plt.show()
    plt.rcParams.update({'figure.autolayout': True})


def plot_calibration_curve(est, name, fig_index):  # Chart 6 (plus 10,11)
    # MAKES 3..."LEARNING CURVE, SCALABILITY OF MODEL, PERFORMANCE OF MODEL
    # https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py
    """Plot calibration curve for est w/o and with calibration. """
    lcstart = datetime.now()

    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1.)

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [#(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)
    global fignbr; fignbr+=1
    end = datetime.now()
    duration = end-lcstart
    print(f'Calibration took {duration}')
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Fig {fignbr}: Calibration plots  (reliability curve)')

    ax2.set_xlabel(f"Mean predicted value.  This took {duration}"  )
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

def plot_2confidenceHistograms():  # Chart #7 chart #8
    lst = xgb1.predict_proba(X_test)[:, 1]
    lst.sort()
    plt.figure()
    ttl = len(lst)
    midl = int(ttl/2)
    midl
    fst = lst[:midl]
    Lst = lst[midl+1:ttl]
    tst = [fst, Lst]
    cmap = ['black', 'blue']
    global fignbr; fignbr+=1
    N, bins, patches = plt.hist(tst, bins=50,  # histtype='step',
                                color=cmap,
                                label='Least confidence\n is in the middle')
    plt.title(f'Fig {fignbr}: Confidence of: {txtmdl}')
    plt.legend()
    plt.ylabel('Count')
    plt.xlabel(' Probability')
    plt.show()

    plt.figure()
    cmap = ['black', 'red', 'blue']
    mid = np.searchsorted(lst, .5)
    fst = lst[:mid]
    doubtful = lst[mid+1:midl]
    Lst = lst[midl+1:14999]
    tst = [fst, doubtful, Lst]
    N, bins, patches = plt.hist(tst, bins=50,  # histtype='step',
                                color=cmap, rwidth=(1),
                                label='Least confidence\n RED in the middle')
    fignbr+=1
    plt.title(f'Fig {fignbr}: Confidence of: {txtmdl}')
    plt.legend()
    plt.axvline(x=.5, linestyle='--')
    plt.axvline(x=.6, linestyle='--')
    plt.ylabel('Count')
    plt.xlabel('   Probability')
    plt.show()

# below are chart # 9 10 11  (passing in the model...xgb1)
def plot_scalability_performance(estimator, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    # Took 28 minutes!!! on 3 March 2021 (To do pull processing out of charting)
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import ShuffleSplit
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    lcstart = datetime.now()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = estimator
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    title = r"Learning Curve (XGBoost37_min)"
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Chart #9     Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    end = datetime.now()
    duration = end-lcstart
    print(f'That took {duration}')

    # Plot learning curve
    axes[0].grid(True)
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel(f'Chart #10   Training examples from: {filename}')
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid(True)
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel(f'Chart #11   fit_times; CurveTngTime: {duration}')
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    plt.show()
    return plt, cv

#%% EXP --> Charts to Explain the model
# not using this one...(re-look this)
def treeplotter(gbm, num_boost_round, tree_nbr, txtparams):
    fig, ax = plt.subplots(figsize=(40, 10))
    plt.rcParams.update({'figure.autolayout': True})
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    xgb.plot_tree(gbm, num_trees=tree_nbr, ax=ax)
    plt.title(f'File: {filename};  {mdl} Tree# "first";  {mdlsttime}', fontsize=32)
    plt.text(.9, .9, horizontalalignment='right', transform=trans,
             s=f'Max # Boosters: {num_boost_round};\nLast "tree":' +
             ' {gbm.best_iteration}; ',
             fontsize=24, bbox=dict(facecolor='pink', alpha=0.5))
    plt.text(.1, .9, f'Model Params:\n{txtparams}', horizontalalignment='left',
             wrap=True, verticalalignment='top', fontsize=20, transform=trans,
             bbox=dict(facecolor='pink', alpha=0.5))  # , font=font
def plot_tree(tree2plot, graphdir):  # Chart #12
    """ We are using xgboost's plot tree.  sklearn has one also...
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html
    """
    global fignbr; fignbr+=1
    end = datetime.now()
    now = end.strftime('%c')
    plt.subplots_adjust(left=.05, right=.95, top=.1, bottom=.005)
    #tree2plot = tree2plot
    if graphdir == 'LR':
        fig, ax = plt.subplots(figsize=(50, 40))
    else:
        fig, ax = plt.subplots(figsize=(50, 50))
    plt.rcParams.update({'figure.autolayout': False})
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    xgb.plot_tree(xgb1, num_trees=tree2plot, rankdir=graphdir, ax=ax)
    if graphdir == 'LR':  # ie build tree Left to Right 
        plt.title(f'Fig: {fignbr} File: {filename};  Mdl: XGboost;  {now}',
                  fontsize=64)
        plt.text(.1, .5, f'Model Params:\n{txtparams}', ha='left',
                 wrap=True, va='bottom', fontsize=40, transform=trans,
                 bbox=dict(facecolor='pink', alpha=0.5))  # , font=font)  #
        plt.text(.01, .3, ha='left', transform=trans, va='top',
                 s=f'Max # Boosters: 3;\nTree #: {tree2plot}; ',
                 fontsize=48, bbox=dict(facecolor='pink', alpha=0.5))  #
        plt.xlabel('Chart #12')
    else:  #  ie 'wide'
        plt.title(f'Fig: {fignbr} File: {filename};  Mdl: XGboost Tree# "first";  {now}',
                  fontsize=64)
        plt.text(.95, .8, ha='right', transform=trans, va='top',
                 s=f'Max # Boosters: 3;\nLast "tree": {tree2plot}; ',
                 fontsize=24, bbox=dict(facecolor='pink', alpha=0.5))  #
        plt.text(.01, .9, f'Model Params:\n{txtparams}', ha='left',
                 wrap=True, va='top', fontsize=20, transform=trans,
                 bbox=dict(facecolor='pink', alpha=0.5))  # , font=font)  #
        plt.xlabel('Chart #12')
    fig.tight_layout()  # This reduces 1 inch side margins to near zero.
#    fig.set_size_inches(150, 100) # # to solve low resolution problem
    plt.show()
    plt.rcParams.update({'figure.autolayout': True})
#plot_tree(10, 'TD')  # top-down
#plot_tree(600, 'LR')
def plot_PermutationImportancechart(durationSeconds):
    # plt.rcdefaults()
    global fignbr; fignbr+=1
    now = datetime.now()
    plt.rcParams.update({'figure.autolayout': True})
    sorted_idx = result.importances_mean.argsort()  # yields ndarray
    sorted_idx = sorted_idx[-25:]
    # plt.rcParams.update({'figure.autolayout': True})
    fig, ax1 = plt.subplots()
    ax1.boxplot(result.importances[sorted_idx].T,
                vert=False, labels=X_test.columns[sorted_idx])
    # ax1.set_title(f"Permutation Importances (test set) Accuracy: {round(accuracy,3)}")
    ax1.set_title(f'Fig #{fignbr} Permutation Importances (test set) \n' +
                  f'  Accuracy is: {round(accuracy,3)} at: {now}', fontsize=14)
    plt.yticks(fontsize=9, rotation=15)
    plt.xticks(fontsize=10)

    plt.text(.05, 18, 'Notice the difference between \n"permutted importances"' +
             ' and "Feature importance!"\n learn the difference!', fontsize=12,
             horizontalalignment='left', verticalalignment='bottom',
             bbox=dict(facecolor='yellow', alpha=0.5))

    plt.text(.05, 1, testsummary, fontsize=10,  # FancyBboxPatch
             horizontalalignment='left', verticalalignment='bottom',
             bbox=dict(facecolor='yellow', alpha=0.5))

    # plt.text(0.05,4, xgb_variables, horizontalalignment='left',
    #          bbox=dict(facecolor='green', alpha=0.5))  # fontsize=14,
    plt.text(.05, 8, xgbreport, horizontalalignment='left', fontsize=11,
             bbox=dict(facecolor='blue', alpha=0.3))
    fig.tight_layout()
    if durationSeconds > 60:
        txtduration = str(round(durationSeconds / 60,2)) + ' Minutes'
    else: txtduration = str(round(durationSeconds,2)) + ' Seconds'
    plt.xlabel(f'Chart #13 Duration: {txtduration}')
    plt.show()
    return result, sorted_idx

# plot_PermutationImportancechart(45.0002) 
def plot_FeatureImportance(max_num_features, imptyp):  # Chart #14, 15
    # https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
    font = {'family': 'monospace',
            'weight': 'normal',
            'size': 10,
            }
    end = datetime.now()
    now = end.strftime('%c')
    plt.rcParams.update({'figure.autolayout': True})
    # trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    pt2tr = 1  # place text to the right  (right align in the graph...)
    fig, ax = plt.subplots(figsize=(12, 8))
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    plot_importance(booster=xgb1, max_num_features=max_num_features, ax=ax,
                    title=f'Feature importance {now}', importance_type=imptyp,
                    show_values=False)
    plt.yticks(fontsize=10, rotation=20)
    plt.xticks(fontsize=16)
    global fignbr; fignbr+=1
    plt.title(f'Fig: #{fignbr} xgboost Feature importance {now}; Importance_Type: {imptyp}',
              fontsize=16)
    # plt.text(pt2tr, .75, transform=trans,
    #           s=f'Test Accuracy: {round(accuracy_score(y_test, y_pred),3)}' +
    #           f' at: {dt_string}\n{os.path.basename(__file__)}',
    #           horizontalalignment='right',
    #           font={'size': 16}, bbox=dict(facecolor='pink', alpha=0.5))  #
    ax.text(pt2tr, .5, xgbreport, horizontalalignment='right', fontsize=11,
            bbox=dict(facecolor='blue', alpha=0.3), transform=trans)
    plt.text(pt2tr, .4, keyparams, ha='right', transform=trans,
             bbox=dict(facecolor='pink', alpha=0.5), font=font)
    plt.text(.8, .4, hitnmiss, ha='right', transform=trans,
             bbox=dict(facecolor='pink', alpha=0.5), font=font)
    plt.text(.12, .28, '      Categorical Features:\n' +
             categoricalVariablesList, ha='left', font=font, transform=trans,
             bbox=dict(facecolor='yellow', alpha=0.5))  #
    plt.text(pt2tr, .28,
             s=f'Confusion Matrix\n{cm[0,0]} {cm[0,1]}\n{cm[1,0]} {cm[1,1]}',
             ha='right', transform=trans,
             font=font, bbox=dict(facecolor='pink', alpha=0.5))  #
    plt.text(pt2tr, .2, SQLCommand, ha='right', transform=trans,
             bbox=dict(facecolor='pink', alpha=0.5), font=font)  # fontsize=14,
    plt.text(pt2tr, .1, 'Height set by scale of 0 to 1 (ax.Transform)',
             ha='right', bbox=dict(facecolor='pink', alpha=0.5),
             font=font, transform=trans)  #
    plt.text(pt2tr, 0.05, transform=trans, ha='right',
             s='plot_importance(xgb, max_num_features=max_ftrs, ax=ax,\'...',
             # Top {nmbrFtrs2List} Importance of Features')))',
             font=font, bbox=dict(facecolor='pink', alpha=0.5))  #
    plt.text(.33, .1, hitnmiss, ha='left', transform=trans,
             bbox=dict(facecolor='green', alpha=0.3), fontsize=11)  # ,
    plt.show()
def plot_sidebyside_feature_importanceVSpermutation():
    # Below link says possibly greater value examining permutation importance
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
    global fignbr; fignbr+=1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    fig.subplots_adjust(top=.75)
    plot_importance(xgb1, max_num_features=max_num_features, ax=ax1)
    plt.suptitle(f'Fig: {fignbr} Camparison of "Importances" ', y=.98)
    ax1.set_title('Feature importance',
                  fontdict={'fontsize': 14, 'color': 'r'})
    ax2.set_title('Permutation importance',
                  fontdict={'fontsize': 14, 'color': 'r'})
    ax1.tick_params(labelsize=11, labelcolor='black', labelrotation=45)
    ax2.tick_params(grid_color='r', labelsize=11, labelcolor='blue',
                    labelrotation=45)
    ax2.boxplot(result.importances[sorted_idx].T,
                vert=False, labels=X_test.columns[sorted_idx]) 
    ax2.set_xlabel('Chart #17')
    #fig.tight_layout()
    plt.show()
# plot_sidebyside_feature_importanceVSpermutation(17)
# https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20XGBoost.html
def plot_SHAP_charts(k2):  # Chart #17  #18
    plt.rcParams.update({'figure.autolayout': True})
    global fignbr; fignbr+=1
#    fig, ax = plt.subplots(figsize=(12, 12))
#    trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    plt.rcParams.update({'figure.autolayout': True})
    plt.title(f'Fig: {fignbr} XGBoost Shap values; {filename}; \n {txtmdl}  On: {dt_string}',
              fontsize=14)
    plt.yticks(fontsize=9, rotation=25)
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.show()

    fig, ax = plt.subplots()
    # density scatter plot of SHAP values for each feature to identify how much
    # impact each feature has on the model output for individuals in the
    # validation dataset. Features are sorted by the sum of the SHAP value
    # magnitudes across all samples
    # Note that when the scatter points don't fit on a line they pile up to
    # show density, and the color of each point represents the feature value
    # of that individual.
#    trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    plt.rcParams.update({'figure.autolayout': True})
    fignbr+=1
    plt.title(f'Fig: {fignbr} XGBoost Shap values; {filename}; \n {txtmdl}  On: {dt_string}',
              fontsize=14)
    plt.yticks(fontsize=10, rotation=35)
    shap.summary_plot(shap_values, X)
    plt.show()

    #Following: 'fig: 20' 
    # 'magically two charts in one, they share an axis, 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))
    fignbr+=1
    fig.suptitle(f'Fig: {fignbr} Shap barchart vs the "summary_plot"')
    k2.plot.barh(y='SHAP_abs', color = colorlist, #This on the left side
                 ax=ax1, legend=False, fontsize = 0)  #, **kwargs)
    shap.summary_plot(shap_values, X) # this is on the right side
    plt.show()

#https://slundberg.github.io/shap/notebooks/plots/dependence_plot.html
def plot_SHAP_charts2(k2):
    global fignbr; fignbr+=1
    shap.dependence_plot('vehmph', shap_values, X, display_features=X, show=False)
    plt.title(f'Fig: {fignbr} Speed vs driver situation"Normal"')
    plt.show()
    
    shap.dependence_plot('dayhour', shap_values, X, display_features=X, show=False)
    fignbr+=1
    plt.title(f'Fig: {fignbr} Hour of the day vs "other type of vehicle"')
    plt.show()

    shap.dependence_plot('vehage', shap_values, X, display_features=X, show=False)
    fignbr+=1
    plt.title(f'Fig: {fignbr} Age of Vehicle vs driver situation"Normal"')
    plt.show()

    shap.dependence_plot('drSitu_drgAlcSlpSic', shap_values, X, display_features=X, show=False)
    fignbr+=1
    plt.title(f'Fig: {fignbr} Speed vs "Drugs, alcohol, Sleep & Sickness"')
    plt.show()
    return shap_values

def ABS_SHAP(df_shap, df):
    # https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d
    # import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index', axis=1)

    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),
                         pd.Series(corr_list)], axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature,
    # and Column 2 is the correlation coefficient
    corr_df.columns = ['Variable', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')
    shap_abs = np.abs(shap_v)

    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable', 'SHAP_abs']
    k2 = k.merge(corr_df, left_on='Variable', right_on='Variable',
                 how='inner')
    k2 = k2.sort_values(by='SHAP_abs', ascending=True)[-20:]
    colorlist = k2['Sign']
    return shap_v, shap_abs, k2, colorlist
""" Plot it later
    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable', 'SHAP_abs']
    k2 = k.merge(corr_df, left_on='Variable', right_on='Variable',
                 how='inner')
    k2 = k2.sort_values(by='SHAP_abs', ascending=True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable', y='SHAP_abs', color=colorlist,
                      figsize=(5, 6), legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    ax.set_ylabel('SHAP_abs', fontsize=8)
    """

    # The following doesn't work...and won't until 2021 from Carlos Cordoba
    # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
    # plt.show()
# plot_SHAP_charts()
"""
shap.TreeExplainer
alias of shap.explainers._tree.Tree
shap.GradientExplainer
alias of shap.explainers._gradient.Gradient
shap.DeepExplainer
alias of shap.explainers._deep.Deep
shap.KernelExplainer
alias of shap.explainers._kernel.Kernel
shap.SamplingExplainer
alias of shap.explainers._sampling.Sampling
shap.PartitionExplainer
alias of shap.explainers._partition.Partition
"""

def plot_tree2():  # this is not working... not sure why.
    end = datetime.now()
    now = end.strftime('%c')
    tree2plot = 100
    fig, ax = plt.subplots(figsize=(40, 10))
    plt.rcParams.update({'figure.autolayout': False})
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    xgb.plot_tree(xgb, num_trees=tree2plot, ax=ax)
    plt.title(f'File: {filename};  {mdl} Tree# "first";  {now}', fontsize=32)
    plt.text(.9, .8, horizontalalignment='right', transform=trans,
             s=f'Max # Boosters: 3;\nLast "tree": {tree2plot}; ',
             fontsize=24, bbox=dict(facecolor='pink', alpha=0.5))  #
    plt.text(.1, .9, f'Model Params:\n{txtparams}', horizontalalignment='left',
             wrap=True, verticalalignment='top', fontsize=20, transform=trans,
             bbox=dict(facecolor='pink', alpha=0.5))  # , font=font)  #
    plt.show()

#%% Get & Prepare Data: SQL, CSV, Github
def getdata(records):
    start = datetime.now()
    # conn = pyodbc.connect('DRIVER={SQL Server};'
    #                       'SERVER=(local);'
    #                       'DATABASE=Traffic;'
    #                       'Trusted_Connection=yes;')
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
    url = 'https://github.com/steveSchneider2/data/tree/main/FloridaTraffic/traffic116k_88_76pc.csv?raw=true'
    #url = 'https://github.com/topics/floridatraffic.csv'
#    pr = pd.read_csv(url)
    pr = pd.read_csv('D:/ML/Python/data/traffic/traffic236k_95_16.csv', header=0)
#    pr = pd.read_csv('data/traffic72k_94_83.csv', header=0)
    #pr = pd.read_csv('D:/dDocuments/ML/Python/data/Traffic/traffic116k_88_76pc.csv')
#    pr = pd.read_sql(SQLCommand, conn)
    end = datetime.now()
    processTime = end - start
    print('Seconds to download SQL records is: ', processTime.total_seconds())
    #pr.to_csv(r'..\data\traffic140k.csv', index=False)
    return pr, SQLCommand, SQL, SqlParameters, dt_string


def standardize(df):
    # https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc
    names = df.columns
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    return scaled_df


def datapreparation():
    pr.drop(['Crash_year', 'monthday','speeddif','vMphGrp'], axis=1,
            inplace=True)
    # pr.drop(['license', 'drvsex', 'drvagegrp',
    #          'drDistract'], axis=1, inplace=True)
    if 'vehicle_number' in pr.columns:
        pr.pop('vehicle_number')  # Return item and drop from frame
    if 'InvolvedVeh' in pr.columns:
        pr.pop('InvolvedVeh')  # Return item and drop from frame
    type(pr.index)
    pr
    type(pr)
    pr.shape     # Like the R dim command
    pr.describe()
    pr.iloc[:9]
    pr.head()
    pr.corr()
    nbrrecords = pr.shape[0]
    pr.describe(include=['object', 'category']).transpose()
    pr.describe(
        include=['object', 'category']
        ).transpose().sort_values(by='unique', ascending=False)
#    pr.plot(pr['wday'], pr['vehmph'], kind='scatter')

    list(pr.select_dtypes(np.object).columns)  # Get just the non-numerics...

    todummylist = list(pr.select_dtypes(np.object).columns)

    class_names_gnb = (['minor', 'major'])
    pr.shape
    pr.fillna(method='pad', inplace=True)
    X = pr
    X.info(memory_usage='deep')

    j = pd.concat([pr['vehmph'],pr.crash], axis=1)
    major = j.vehmph[j.crash==1]
    minor = j.vehmph[j.crash==0]

    # j = pd.concat([pr.rdSurfac,pr.crash], axis=1)
    # labels = f'Drivers Situation in {pr.shape[0]} crashes'
    # major = j.rdSurfac[j.crash==1]
    # minor = j.rdSurfac[j.crash==0]
    # if howfartoexecute != 'learningcurves':
#    plot_histograms_stacked(major, minor, 3,
#                            'Actual accidents vs Road Surface',
#                            labels)

    if 'crash' in pr.columns:  # Allow me to run this frm multiple places in code
        y = pr.pop('crash')  # Return item and drop from frame

    # Function to dummy all the categorical variables used for modeling
    # courtesy of April Chen

    def dummy_pr(pr, todummylist):
        for x in todummylist:
            dummies = pd.get_dummies(pr[x], prefix=x, dummy_na=False)
            pr = pr.drop(x, 1)
            pr = pd.concat([pr, dummies], axis=1)
        return pr
    labels = f'{pr.shape[0]} crashes'
    X = dummy_pr(X, todummylist)
    X.head()
    #X = standardize(X)
    trng_prct = .75
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=1 - trng_prct, random_state = 200)
    type(X_train)
    X_train.shape
    X_test.shape

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

# %% MAIN Code--> Call to get & prep the data
txtmdl, mdl, mdlsttime, filename, start = \
    setstartingconditions()

#howfartoexecute = 'ALLlearningcurves'
howfartoexecute = 'learningcurves'

pr, SQLCommand, SQL, SqlParameters, dt_string = \
    getdata(118000)  #return 2*118,000...

X_train, X_test, y_train, y_test, pr, categoricalVariablesList,\
        testsummary, X, y, class_names_gnb, major, minor = \
        datapreparation()

# if howfartoexecute == 'learningcurves':
#     print('quit')
#     sys.exit()
# %%  5 Charts: data as it looks prior to modeling --> DRIVES FEATURE ENGINEERING
fignbr = 0
plot_histograms_stacked(major, minor,100)
plot4histograms()

X_train.describe()
y_train.head()
X_test.describe()

# %% Define & Fit the model

def strmodel():
    # Choose all predictors except target & IDcols
    # https://stats.stackexchange.com/questions/204489/discussion-about-overfit-in-xgboost
    # https://www.capitalone.com/tech/machine-learning/how-to-control-your-xgboost-model/
    # https://towardsdatascience.com/xgboost-theory-and-practice-fb8912930ad6
    target = 'crash'
    predictors = [x for x in X_train.columns if x not in [target]]  # , IDcol]]
    # predictors = y_train.columns
    params = {
        #Next line'use_label_encoder' added Oct 6th 2021, to remove this warning: C:\Users\steve\anaconda3\envs\MLFlowProtoBuf\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 
        'use_label_encoder': False, #besides fixing the error, this improved accuracy by 1% :)
        # 'tree_method': 'gpu_hist',  # src/learner.cc:180:
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
        #'colsample_bynode': 0.6,
        "objective": 'binary:logistic',
        "nthread": -1,
        "scale_pos_weight": 1,
        "eval_metric": 'error', #"auc",  # vs "logloss"
        # "seed": 27,
        'random_state': 200,
        "verbosity": 1,
        #        'single_precision_histogram': True,
        "n_jobs": -1}
    xgb1 = XGBClassifier(**params)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    start = datetime.now()
    xgb1.fit(X_train, y_train, early_stopping_rounds=20,
    #  eval_metric=["error", "logloss"], eval_set=eval_set, verbose=100)
       eval_metric=["error", "logloss"], eval_set=eval_set, verbose=100) # 06 Oct 2021 to fix warning
    end = datetime.now()
    getdataTime = end - start
    print('Seconds to train model is: ', getdataTime.total_seconds())
    predictions = xgb1.predict(X_test)
    # predictions = xgb1.predict_proba(X_test)
    actuals = y_test
    print(confusion_matrix(actuals, predictions))
    # accuracy_score(X_train, X_pred )
    accuracy = accuracy_score(y_test, predictions)
    # print(f'Accuracy: {accuracy} ')  # , or below, perhaps a nicer way...
    print('Accuracy: %.2f%%' % (accuracy * 100))
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, predictions)
    auc_xgb = round(auc(fpr_xgb, tpr_xgb), 2)
    cm = confusion_matrix(y_test, predictions)
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

print('\n We are starting to model... \n')
xgb1, predictions, actuals, accuracy, params, results, x_axis, epochs,\
    mdlTngTime, cm = strmodel()  # i need a better function name!
print('\n done modeling... \n')
# #############################################
# pr['vehmph'].value_counts(sort=True)

# %% Create variables for various charts
xgbreport = classification_report(y_test, predictions)
correctlables = (y_test == predictions).sum()
badlables = (y_test != predictions).sum()
hitnmiss = ("Correct: %d\nIncorrect: %d\ntotal: %d"
            % (correctlables, badlables, X_test.shape[0]))
lr = xgb1.get_params()['learning_rate']
md = xgb1.get_params()['max_depth']
ne = xgb1.get_params()['n_estimators']
bt = xgb1.get_booster().best_iteration
# mdlparams = f'Learn:{lr}, depth: {md}, trees: {ne} '
keyparams = str(f'Learn:{lr}   depth:   {md}\ntrees: {ne}  @stop: {bt}'+
            f'\nModel tng time: {mdlTngTime}')


# #######################  ROC CURVE ###############################
mdl = XGBClassifier.mro()[0]
txtparams = json.dumps(params)
txtparams = txtparams.replace(',', '\n')

# %% 11 Charts: Model's validity, accuracy,'calibration', confidence, lrning curves (3)
# Learning curve #1 & 2 (SIDE-BY-SIDE)
plot_learningCurves_LogLoss_ClassError() # THIS IS THE BEST!

# Following two options for optimum visibility of a tree:
plot_tree(0 , 'LR')
xgb.to_graphviz(xgb1, rankdir='LR') # this line goes to console...

ROC_rcvr_operating_curve(xgb1) # Fig 8

howfartoexecute =  'learningcurves' # 'alltheway'
# if howfartoexecute == 'learningcurves':
#     print('quit')
#     sys.exit()

plot_ConfusionMatrixChart(xgb1)  # makes 2...2nd is by percentage

startcls = datetime.now()
dt_string = startcls.strftime("%m/%d/%Y %H:%M:%S")
print(f'Start Calibration:  {dt_string}  \nExpect this to take 1.5 minutes...')

plot_calibration_curve(xgb1, "XGBoost", 1)  # expect 2 min
plot_2confidenceHistograms()
lcstart = datetime.now()
print(f'histograms done in {lcstart-startcls}')
print(f'Coming: LEARNING CURVE, SCALABILITY OF MODEL, PERFORMANCE OF MODEL...{lcstart}')
print('This takes more than several minutes!!!')
cv = 'none'
# 3 plots (in a row) LEARNING CURVE, SCALABILITY OF MODEL, PERFORMANCE OF MODEL
# This might take 15 minutes!!! Only enable, when you are patient!

#############
# plot_scalability_performance(xgb1, X, y,  ylim=(0.7, 1.01), n_jobs=4)  # 13 min OR LONGER!!
# Also generates this multi-page warning:
#  UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
#############

# %%  4 Charts: Model's explainability --> feature importance
plot_tree(1, 'LR')

#  following makes sklearn.utils.Bunch
from sklearn.inspection import permutation_importance

startcls = datetime.now()
dt_string = startcls.strftime("%m/%d/%Y %H:%M:%S")
print(f'Start Permuting importances:  {dt_string} expecting 4.5 minutes...')

with warnings.catch_warnings():
  warnings.simplefilter("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses  
# the below line creates the user warning that won't go away.  
result = permutation_importance(xgb1, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
end = datetime.now()
permutationDuration = end - startcls
durationSeconds = permutationDuration.total_seconds()
# The permutation calculation took 9.5 minutes!!!! Longer than modeling!
# ##################CHARTING ################################################

result, sorted_idx =\
    plot_PermutationImportancechart(durationSeconds) #Fig: 14?
max_num_features = 25
plot_FeatureImportance(max_num_features, 'weight')  #Fig: 15

# There are 3 methods to measure importance in XGBoost:
#     Weight. The number of times a feature is used to split the data across all trees.
#     Cover. The number of times a feature is used to split the data across all trees weighted by the number of training data points that go through those splits.
#     Gain. The average training loss reduction gained when using a feature for splitting.

plot_FeatureImportance(max_num_features, 'gain')  # could also be 'gain'
plot_sidebyside_feature_importanceVSpermutation()  # Plot 17

explainer = shap.TreeExplainer(xgb1)
print('Need 19 minutes to do these shap calculations...')
shap_values = explainer.shap_values(X)  # 19+ minutes!  @depth = 9
print('Shap_values calculated:  {} ...'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))

shap_v, shap_abs, k2, colorlist = ABS_SHAP(shap_values,X)
print('ABS_Shap calculated:  {} ...'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
# shap_v = ABS_SHAP(shap_values,X)
shap_values = plot_SHAP_charts(k2)  # <--THREE charts! ~30 sec for all 7
#shap_values = plot_SHAP_charts2(k2, 21)  # <--FOUR charts! ~30 sec for all 7
# plt.rcdefaults()
plot_tree(xgb1.get_booster().best_iteration, 'LR')


# %% We're done!
duration = datetime.now() - start
print(f'Total duration for {filename} is: {duration}')


