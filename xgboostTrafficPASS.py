"""

Overall comments.

Created on Thu Sep 17 08:06:27 2020.

Stopped work on Nov 30 2020 after achieving 95% accuracy! :)
Re-ran on 1 Mar 2021 on Python 3.8

RUNTIME: 13 minutes (10 Sep 2021)
INPUT:  SQL SERVER TRAFFIC database; 240,000 records.
OUPUT:   26 graphs
NEEDS Python 3.7 and up. (for plot_confusion_matrix, at least.)

Notes
-----
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
# pylint: disable=W0603,R0913,R0914

# %% Code setup Imports & starting conditions
from sklearn.inspection import permutation_importance
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib as mpl
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import (train_test_split,  # GridSearchCV,
                                     learning_curve, ShuffleSplit)
from sklearn.metrics import (plot_confusion_matrix,  # mean_squared_error,
                             confusion_matrix, accuracy_score,
                             classification_report, roc_curve, auc,
                             recall_score, precision_score,
                             plot_roc_curve,  # average_precision_score,
                             brier_score_loss, f1_score)
from sklearn.metrics import matthews_corrcoef
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
# from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
# from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from datetime import datetime
import json
# import pyodbc
import os
import sys
import shap  # for  the Shap charts of course!
# import random
# import time
# from subprocess import check_output
# from numpy import genfromtxt
# import seaborn as sns
from sklearn import preprocessing
# from operator import itemgetter
import sweetviz
#from autoviz.AutoViz_Class import AutoViz_Class

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

# import pandas_profiling as pp
os.environ["PATH"] += os.pathsep + 'c:/Program Files (x86)/Graphviz/bin/'
FIGNBR = 0


def setstartingconditions():
    """Print to console package versions plus more."""
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    # rcParams['figure.figsize'] = 12, 12
    # plot_tree(xgb._Booster)
    start1 = datetime.now()
    print(start1.strftime('%a, %d %B, %Y', ))
    print(start1.strftime('%c'))
    mdlsttime1 = start1.strftime('%c')

    try:
        thisfile = os.path.basename(__file__)
    except NameError:
        thisfile = 'working'
    # else:
    #     thisfile = 'unk'

    model = xgb.XGBClassifier
    txtmodel = str(model).replace('<class \'', '')
    txtmodel = str(txtmodel).replace('\'>', '')
#    print(__doc__)

    pver = str(format(sys.version_info.major) + '.' +
               format(sys.version_info.minor) + '.' +
               format(sys.version_info.micro))
    print('Python version: {}'.format(pver))
    print('SciKitLearn:    {}'.format(sklearn.__version__))
    print('matplotlib:     {}'.format(mpl.__version__))
    print('XGBoost:        {}'.format(xgb.__version__))

    # plt.style.use('ggplot') # this is used for ALL future plt plots...
    plt.style.use('classic')  # this is used for ALL future plt plots...

    # csfont = {'fontname': 'Comic Sans MS'}
    # font = {'family': 'monospace',
    #         'weight': 'normal',
    #         'size': 12,
    #         }
    return txtmodel, model, mdlsttime1, thisfile, start1

# %% EDA --> Charts Exploratory Data Analysis


def plot_histograms_stacked(bigcr, smlcr, bins=100,
                            title='Fig 1: Actual Accidents vs speed',
                            xlabel='Vehicle speed'):
    """
    Bigcr and smlcr: what you want plotted.

    Parameters
    ----------
    bigcr : integer
        Crash where veh damage>10000, or person is injured.
    smlcr : int
        Reverse of big crash.
    bins : int, optional
        How many 'bins' would you like. The default is 100.
    title : str, optional
        Example: The default is 'Fig 1: Actual Accidents vs speed'.
    xlabel : str, optional
        DESCRIPTION. The default is 'Vehicle speed'.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(2, 1, figsize=(16, 12))
    colors = ['blue', 'red']
    labels = f'{len(smlcr)} little crashes', f'{len(bigcr)} big crashes'
    global FIGNBR
    FIGNBR += 1
    title = f'Fig {FIGNBR}: Actual Accidents vs speed'
    ax[0].set_title(title, fontsize=24)
    # plt.xticks(np.arange(0, 100, 10))
    ax[0].set_xlabel(xlabel)
    ax[1].set_xlabel(labels)
    ax[0].hist([smlcr, bigcr], bins=bins, stacked=True,
               color=colors)  # ,  width=.5)
    ax[1].hist([smlcr, bigcr], bins=bins, stacked=True, log=True, color=colors,
               label=labels)  # , width=.5)
    plt.legend(prop={'size': 16})
    plt.grid()
    # Next line minimizes 'padding' size outside of actual chart
    plt.subplots_adjust(left=.05, right=.95, top=.95, bottom=.05)
    fig.tight_layout()
    plt.show()


def plot4histograms():
    """1st plot is 4 subplots: speed, driverage, veh age & hour of the day."""
    # #1...
    j = pd.concat([X_test['vehmph'], y_test], axis=1)
    bigcrsh = j.vehmph[j.crash == 1]
    ltlcrsh = j.vehmph[j.crash == 0]
    global FIGNBR
    plot_histograms_stacked(
        bigcrsh, ltlcrsh,
        title=f'Fig {FIGNBR}: TEST Data-what the validation data looks like...')
    plt.show()
    # #2
    kwargs = dict(histtype='stepfilled', alpha=0.8, bins=90)
    labels = f'{pr.shape[0]} crashes'
    pr.hist(**kwargs)
    FIGNBR += 1
    plt.subplots_adjust(left=.1, right=.9)
    plt.suptitle(f'Fig {FIGNBR}: The Four integer factors')
    plt.xlabel(f'{labels}')
    plt.show()
    # #3... the 'Drivers Situation' has 3 cases: Normal, unk,
    # DrugAlcSleepSick...so you get 3 graphs
    pr.hist(['vehmph'], ['drSitu'])  # 3 variables: 2 listed + count
    FIGNBR += 1
    plt.suptitle(f"Fig {FIGNBR}:Vehicle SPeed vs Driver Condition")
    plt.show()
    # #4 of 4 histograms...
    pr.hist(['vehmph'], ['hitrun'])
    FIGNBR += 1
    plt.suptitle(f'Fig {FIGNBR}: Vehicle speed vs Hit & Runs')
    plt.show()


# X_test.hist(['vehmph'], bins=90, width=.5)  # already shown via plot_histogram
# X_test.hist(['vehmph'], **kwargs)  # already shown via plot_histogram
# https://towardsdatascience.com/powerful-eda-exploratory-data-analysis-in-just-two-lines-of-code-using-sweetviz-6c943d32f34


def my_sweetviz():
    """Very nice library."""
    startsw = datetime.now()

    sweet_train = X_train.copy()
    sweet_y = pd.DataFrame(y_train)
    # Add the Crash column so SWEETVIZ can visualize it...
    sweet_train['crash'] = sweet_y
    sweet_train.describe()
    sweet_train.head()
    feature_config \
        = sweetviz.FeatureConfig(
            skip=['hitrun_No', 'drvagegrp_older', 'rdSurfac_oil', 'rdSurfac_mud_dirt_gravel',
                  'rdSurfac_ice_frost', 'rdSurfac_Unk', 'rdSurfac_Water', 'rdSurfac_Sand',
                  'rdSurfac_Other', 'drvsex_Female', 'Vision_clear', 'drSitu_unknown',
                  'drDistract_No', 'license_valid', 'vhmtn_Working', 'vhmtn_Unknown',
                  'vhmtn_Parked', 'vhmvnt_U_turning', 'vhmvnt_StopInTraf', 'vhmvnt_Passing',
                  'vhmvnt_Parked', 'vhmvnt_ExitTrfc', 'vhmvnt_EnterTrfc', 'UrbanLoc_No',
                  'light_unk', 'light_dawn', 'weather_danger', 'vehmakegrp_Chrysler',
                  'vehmakegrp_Euro', 'vehmakegrp_Volvo', 'vehmakegrp_Korean',
                  'vehmakegrp_Volvo', 'wday_Monday', 'wday_Tuesday', 'wday_Wednesday',
                  'wday_Thursday', 'wday_Saturday', 'wday_Sunday', 'vehtype_other',
                  'vehtype_truck', 'vehtype_van'])
    my_report = sweetviz.compare([sweet_train, "Train"],
                                 [X_test, "Test"], 'crash',
                                 feature_config)
    my_report.show_html("Report.html")
    endsv = datetime.now()
    sweetduration = endsv - startsw
    print(f'SweetViz duration: {sweetduration}')


def myautoviz():
    """Automate EDA.

    Returns
    -------
    None.

    """
    AV = AutoViz_Class()

    # Creates automated visualizations NEXT:  Can we do it with the df...
    AV.AutoViz('D:/ML/Python/data/traffic/traffic236k_95_16.csv')


def mypandaprofile():
    """Run in notebook, not here in Spyder."""
    # i = 1  # test
# plot3histograms()
# %% EMP --> Charts Explore the modeling process and results


def plot_learningcurves_logloss_classerror():
    """On the left is the log loss, on the right classification error."""
    explainlearncurve = \
    ('A learning curve shows the validation and training score of an estimator\n'
     ' for varying numbers of training samples. It is a tool to find out how \n'
     'much we benefit from adding more training data and whether the estimator\n'
     ' suffers more from a variance error or a bias error.  Log Loss refers to\n'
     ' how close the probability predictions approach either 0 or 1 High log \n'
     'loss would equate to lower confidence...because each prediction might be\n'
     ' "easily" switched (when close to .5) As of 1 Sept, the below is a \n'
     '"good fit" ! :) ')
    # below is yet another way to print
    # print(format('How to visualise XGBoost model with learning curves?', '*^82'))
    # https://www.dezyre.com/recipes/evaluate-xgboost-model-with-learning-curves-example-2
    font = {'family': 'monospace',
            'weight': 'normal',
            'size': 12,
            }
    conmtrx = confusion_matrix(y_test, predictions)
    mc = 100 * round(matthews_corrcoef(y_test, predictions), 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
    plt.rcParams.update({'figure.autolayout': True})
    global FIGNBR
    FIGNBR += 1
    fig.suptitle(f'Fig #{FIGNBR}:   Accuracy: {round(accuracy,3)} LEARNING CURVES {mdlsttime}' +
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
    ax1.text(.1, .91, s=f'Model Params:\n{txtparams}', ha='left',
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
             s=f'Confusion Matrix\n{conmtrx[0,0]} {conmtrx[0,1]}\n{conmtrx[1,0]} {conmtrx[1,1]}',
             fontsize=20, bbox=dict(facecolor='pink', alpha=0.5))  #

    ax2.text(.15, .95, '   Categorical Features:\n' + categoricalVariablesList,
             ha='left', transform=trans2, font=font, va='top',
             bbox=dict(facecolor='yellow', alpha=0.5))  #
    ax2.text(0.15, .55, s=f'MathewsCoef={round(mc,2)}', ha='left', transform=trans2,
             bbox=dict(facecolor='pink', alpha=0.5), fontsize=20)
    ax2.text(0.05, .5, s=explainlearncurve, ha='left', transform=trans2, va='top',
             bbox=dict(facecolor='mediumseagreen', alpha=0.5), fontsize=11)
    ax2.text(.85, .02, xgbreport, ha='right', fontsize=16,
             bbox=dict(facecolor='blue', alpha=0.3), transform=trans2)
    ax2.plot(x_axis, results['validation_0']['error'], label='Train')
    ax2.plot(x_axis, results['validation_1']['error'], label='Test')
#    ax2.set_ylim(.1, .15)
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    fig.tight_layout()
    plt.show()
#plot_learningcurves_logloss_classerror()


def ROC_rcvr_operating_curve(model):
    """Receiver Operating Characteristics...maximize the area under the curve."""
    font = {'family': 'monospace',
            'weight': 'normal',
            'size': 10,
            }
    conmtrx = confusion_matrix(y_test, predictions)
    # The next two commented lines are nice, but we don't use them.
    # fpr_xgb, tpr_xgb, _ = sklearn.metrics.roc_curve(y_test, predictions)
    # roc_auc = auc(fpr_xgb, tpr_xgb)
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
             s=(f'Test Accuracy: {round(accuracy,3)}'
                f' at: {dt_string}\nModel training time: {mdlTngTime};'
                f' {os.path.basename(__file__)}'),
             font={'size': 14}, bbox=dict(facecolor='pink', alpha=0.5))  #
    plt.text(pt2tr, .85, f'Model Params:\n{txtparams}', ha='right',
             wrap=True, verticalalignment='top', fontsize=10,
             bbox=dict(facecolor='aqua', alpha=0.5))  # , font=font)  #
    plt.text(.6, .6, xgbreport, horizontalalignment='right', fontsize=10,
             bbox=dict(facecolor='blue', alpha=0.3))
    plt.text(.25, 1.0, hitnmiss, ha='right', va='top',
             bbox=dict(facecolor='pink', alpha=0.5), font=font)  # fontsize=14,
    plt.text(pt2tr, .15, horizontalalignment='right',
             s=(f'Confusion Matrix\n{conmtrx[0,0]} {conmtrx[0,1]}\n'
                f'{conmtrx[1,0]} {conmtrx[1,1]}'),
             font=font, bbox=dict(facecolor='pink', alpha=0.5))  #
    plt.text(.05, .05, '   Categorical Features:\n' + categoricalVariablesList,
             horizontalalignment='left', font=font,  # transform=trans,
             bbox=dict(facecolor='yellow', alpha=0.5), va='bottom')  #
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    global FIGNBR
    FIGNBR += 1
    plt.title(
        f'Fig {FIGNBR}: ROC curve {XGBClassifier.mro()[0]}, {dt_string}', fontsize=14)
    plt.legend(loc="lower right")
    fig.tight_layout()
    plt.show()


def plot_ConfusionMatrixChart(MLmodel):
    """Create two images... 1 raw numbers, 1 by percent.

    Keyword Arguments:
    -----------------
    MLmodel -- Example: xgb1

    To Do:
    Add data, color, title parameters
    Also, note we depend on global variables for text boxes...not so good.

    """
    plt.rcParams.update({'figure.autolayout': False})
    # cm = confusion_matrix(y_test, predictions)
    global FIGNBR
    FIGNBR += 1
    ttltxt = f'Fig {FIGNBR}: XGBoost: {mdlsttime}\nConfusion matrix, w/o normalization'
    FIGNBR += 1
    ttltxt2 = f'Fig {FIGNBR}: XGBoost: {mdlsttime}\nConfusion matrix from {filename}'
    titles_options = [(ttltxt, None), (ttltxt2, 'true')]
    # plt.rcParams.update({'font.size': 22})  # increase/set internal fonts
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(MLmodel, X_test, y_test,
                                     display_labels=class_names_gnb,
                                     cmap=mpl.cm.Blues,
                                     normalize=normalize,
                                     values_format='1.2f')  # Nbrs NOT in sci notat
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
        plt.text(.3, -.1, testsummary, fontsize=12,  # FancyBboxPatch
                 horizontalalignment='center', verticalalignment='bottom',
                 bbox=dict(facecolor='yellow', alpha=0.5))
        plt.text(0.45, .44, hitnmiss, horizontalalignment='right',
                 bbox=dict(facecolor='green', alpha=0.5))  # fontsize=14,
        plt.text(.41, 1.55, xgbreport, horizontalalignment='right', fontsize=8,
                 bbox=dict(facecolor='blue', alpha=0.3))
        plt.text(.55, .44, keyparams, horizontalalignment='left',
                 bbox=dict(facecolor='aqua', alpha=.5))
        plt.text(-.6, .64, SQLCommand, horizontalalignment='left', wrap=True,
                 bbox=dict(facecolor='aqua', alpha=.5), fontsize=7)
    plt.show()
    plt.rcParams.update({'figure.autolayout': True})


def plot_calibration_curve(est, mdlname):
    # MAKES 3..."LEARNING CURVE, SCALABILITY OF MODEL, PERFORMANCE OF MODEL
    # https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py
    # https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/
    # https://newbedev.com/scikit_learn/auto_examples/calibration/plot_calibration_curve
    # http://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/
    """Plot calibration curve for est w/o and with calibration.

    Keyword Arguments:
    -----------------
        est  -- the model (variable name), example: xgb1
        name -- Name of the model example: 'XGBoost'
    Taken out for now:
        sigmoid takes a lot of time, and by docs, doesn't help in this case
    """
    calstart = datetime.now()

    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
#   sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline (Not today!)
    # lr = LogisticRegression(C=1.)
    interpretation = '''Below the diagonal:
        The model has over-forecast; the probabilities are too large.
Above the diagonal:
        The model has under-forecast; the probabilities are too small.

https://scikit-learn.org/stable/modules/calibration.html'''
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, mdlname in [  # (lr, 'Logistic'),
            (est, mdlname),
            (isotonic, mdlname + ' + Isotonic')]:  # ,
        #     (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % mdlname)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (mdlname, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=mdlname,
                 histtype="step", lw=2)
    global FIGNBR
    FIGNBR += 1
    calduration = datetime.now() - calstart
    print(f'Calibration took {calduration}')
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Fig {FIGNBR}: Calibration plots  (reliability curve)')
    ax1.text(0.03, .95, interpretation, ha='left',
             transform=transforms.blended_transform_factory(ax1.transAxes,
                                                            ax1.transAxes),
             va='top',
             bbox=dict(facecolor='pink', alpha=0.5), fontsize=9)  # ,

    ax2.set_xlabel(f"Mean predicted value.  This took {calduration}")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
# plot_calibration_curve(xgb1, 'sboot')


def plot_2confidenceHistograms():
    """Depends on xgb1 exisiting...ok, parameterize this ASAP."""
    lst = xgb1.predict_proba(X_test)[:, 1]
    lst.sort()
    plt.figure()
    ttl = len(lst)
    midl = int(ttl / 2)
    fst = lst[:midl]
    Lst = lst[midl + 1:ttl]
    tst = [fst, Lst]
    cmap = ['black', 'blue']
    global FIGNBR
    FIGNBR += 1
    # N, bins, patches =
    plt.hist(tst, bins=50,  # histtype='step',
             color=cmap,
             label='Least confidence\n is in the middle')
    plt.title(f'Fig {FIGNBR}: Confidence of: {txtmdl}')
    plt.legend()
    plt.ylabel('Count')
    plt.xlabel(' Probability')
    plt.show()

    plt.figure()
    cmap = ['black', 'red', 'blue']
    mid = np.searchsorted(lst, .5)
    fst = lst[:mid]
    doubtful = lst[mid + 1:midl]
    Lst = lst[midl + 1:14999]
    tst = [fst, doubtful, Lst]
    # N, bins, patches =
    plt.hist(tst, bins=50,  # histtype='step',
             color=cmap, rwidth=(1),
             label='Least confidence\n RED in the middle')
    FIGNBR += 1
    plt.title(f'Fig {FIGNBR}: Confidence of: {txtmdl}')
    plt.legend()
    plt.axvline(x=.5, linestyle='--')
    plt.axvline(x=.6, linestyle='--')
    plt.ylabel('Count')
    plt.xlabel('   Probability')
    plt.show()


def plot_scalability_performance(estimator, trndata, trntarget, axes=None,
                                 ylim=None, crossval=None, n_jobs=None,
                                 train_sizes=np.linspace(.1, 1.0, 5)):
    # Took 28 minutes!! 3 March 2021: (To do: pull processing out of charting)
    """
    Generate 3 plots.

    the test and training learning curve, the training
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

    trndata : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    trntarget : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to trndata for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    crossval : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for crossval are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``trntarget`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``trntarget`` is neither binary nor multiclass, :class:`KFold` is used.

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
    calstart = datetime.now()
    #  PROBLEM!!! crossval (the input, is being overwritten!)
    crossval = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    #    estimator = estimator
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    title = r"Learning Curve (XGBoost37_min)"
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Chart #9     Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, trndata, trntarget, cv=crossval,
                       n_jobs=n_jobs, train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    scaleduration = datetime.now() - calstart
    print(f'That took {scaleduration}')

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
    axes[2].set_xlabel(f'Chart #11   fit_times; CurveTngTime: {scaleduration}')
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    plt.show()

# %% EXP --> Charts to Explain the model


def treeplotter(gbm, num_boost_round, tree_nbr, txt_params):
    """Not using this one...(re-look this)."""
    fig, ax = plt.subplots(figsize=(40, 10))
    plt.rcParams.update({'figure.autolayout': True})
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    xgb.plot_tree(gbm, num_trees=tree_nbr, ax=ax)
    plt.title(
        f'File: {filename};  {mdl} Tree# "first";  {mdlsttime}', fontsize=32)
    plt.text(.9, .9, horizontalalignment='right', transform=trans,
             s=f'Max # Boosters: {num_boost_round};\nLast "tree":' +
             ' {gbm.best_iteration}; ',
             fontsize=24, bbox=dict(facecolor='pink', alpha=0.5))
    plt.text(.1, .9, f'Model Params:\n{txt_params}', horizontalalignment='left',
             wrap=True, verticalalignment='top', fontsize=20, transform=trans,
             bbox=dict(facecolor='pink', alpha=0.5))  # , font=font
    fig.tight_layout()


def plot_tree(tree2plot, graphdir):  # Chart #12
    """ We are using xgboost's plot tree.  sklearn has one also...
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html
    """
    global FIGNBR
    FIGNBR += 1
    tree_end = datetime.now()
    now = tree_end.strftime('%c')
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
        plt.title(f'Fig: {FIGNBR} File: {filename};  Mdl: XGboost;  {now}',
                  fontsize=64)
        plt.text(.1, .5, f'Model Params:\n{txtparams}', ha='left',
                 wrap=True, va='bottom', fontsize=40, transform=trans,
                 bbox=dict(facecolor='pink', alpha=0.5))  # , font=font)  #
        plt.text(.01, .3, ha='left', transform=trans, va='top',
                 s=f'Max # Boosters: 3;\nTree #: {tree2plot}; ',
                 fontsize=48, bbox=dict(facecolor='pink', alpha=0.5))  #
        plt.xlabel('Chart #12')
    else:  # ie 'wide'
        plt.title(f'Fig: {FIGNBR} File: {filename};  Mdl: XGboost Tree# "first";  {now}',
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
# plot_tree(10, 'TD')  # top-down
# plot_tree(600, 'LR')


def plot_permutationimportancechart(fn_durationseconds):
    # plt.rcdefaults()
    global FIGNBR
    FIGNBR += 1
    now = datetime.now()
    plt.rcParams.update({'figure.autolayout': True})
    sorted_idx2 = result.importances_mean.argsort()  # yields ndarray
    sorted_idx2 = sorted_idx2[-25:]
    # plt.rcParams.update({'figure.autolayout': True})
    fig, ax1 = plt.subplots()
    ax1.boxplot(result.importances[sorted_idx2].T,
                vert=False, labels=X_test.columns[sorted_idx2])
    # ax1.set_title(f"Permutation Importances (test set) Accuracy: {round(accuracy,3)}")
    ax1.set_title(f'Fig #{FIGNBR} Permutation Importances (test set) \n' +
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
    if fn_durationseconds > 60:
        txtduration = str(round(fn_durationseconds / 60, 2)) + ' Minutes'
    else:
        txtduration = str(round(fn_durationseconds, 2)) + ' Seconds'
    plt.xlabel(f'Chart #13 Duration: {txtduration}')
    plt.show()
    return result, sorted_idx2

# plot_PermutationImportancechart(45.0002)


def plot_feature_importance(num_features, imptyp):
    """Compare features.

    Parameters
    ----------
    num_features : int
        How many features you want in your chart.
    imptyp : str
        Gain, weight, cover.

    Returns
    -------
    None.
    https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27

    """
    font = {'family': 'monospace',
            'weight': 'normal',
            'size': 10,
            }
    fn_end = datetime.now()
    now = fn_end.strftime('%c')
    plt.rcParams.update({'figure.autolayout': True})
    if imptyp == 'weight':
        measurement = ('The number of times a feature is used to split the'
                       ' data across all trees.')
    elif imptyp == 'gain':
        measurement = ('The average training loss reduction gained when'
                       ' using a feature for splitting.')
    elif imptyp == 'cover':
        measurement = (
            'The number of times a feature is used to split the data across \n'
            ' all trees weighted by the number of training data points that \n'
            'go through those splits.')

    pt2tr = .98  # place text to the right  (right align in the graph...)
    fig, ax = plt.subplots(figsize=(12, 8))
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    plot_importance(booster=xgb1, max_num_features=num_features, ax=ax,
                    title=f'Feature importance {now}', importance_type=imptyp,
                    show_values=False)
    plt.yticks(fontsize=10, rotation=20)
    plt.xticks(fontsize=16)
    global FIGNBR
    FIGNBR += 1
    plt.title(f'Fig: #{FIGNBR} xgboost Feature importance {now};'
              f' Importance_Type: {imptyp}',
              fontsize=16)
    # plt.text(pt2tr, .75, transform=trans,
    #           s=f'Test Accuracy: {round(accuracy_score(y_test, y_pred),3)}' +
    #           f' at: {dt_string}\n{os.path.basename(__file__)}',
    #           horizontalalignment='right',
    #           font={'size': 16}, bbox=dict(facecolor='pink', alpha=0.5))  #
    ax.text(pt2tr, .8, measurement, ha='right', va='bottom', fontsize=11,
            bbox=dict(facecolor='mediumseagreen', alpha=0.3), transform=trans)
    ax.text(pt2tr, .5, xgbreport, horizontalalignment='right', fontsize=11,
            bbox=dict(facecolor='blue', alpha=0.3), transform=trans)
    plt.text(pt2tr, .4, keyparams, ha='right', transform=trans,
             bbox=dict(facecolor='pink', alpha=0.5), font=font)
    plt.text(.7, .4, hitnmiss, ha='right', transform=trans,
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
             s='plot_importance(xgb, num_features=max_ftrs, ax=ax,\'...',
             # Top {nmbrFtrs2List} Importance of Features')))',
             font=font, bbox=dict(facecolor='pink', alpha=0.5))  #
    plt.text(.33, .1, hitnmiss, ha='left', transform=trans,
             bbox=dict(facecolor='green', alpha=0.3), fontsize=11)  # ,
    fig.tight_layout()
    plt.show()


def plot_sidebyside_feature_importanceVSpermutation(num_features):
    # Below link says possibly greater value examining permutation importance
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
    global FIGNBR
    FIGNBR += 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    fig.subplots_adjust(top=.75)
    plot_importance(xgb1, max_num_features=num_features, ax=ax1)
    plt.suptitle(f'Fig: {FIGNBR} Camparison of "Importances" ', y=.98)
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
    # fig.tight_layout()
    plt.show()
# plot_sidebyside_feature_importanceVSpermutation(17)
# https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20XGBoost.html

# %% EXP2 --> Shaply charts


def plot_SHAP_charts(shapvalues_df):
    """Plot 3 shap charts.

    Parameters
    ----------
    shapvalues_df : dataframe  4 columns, 1 record for each feature
        Variable  SHAP_abs      Corr  Sign.  "Variable" is feature.

    Returns
    -------
    None.  But, does make the well known shap chart.
    """
    plt.rcParams.update({'figure.autolayout': True})
    global FIGNBR
    FIGNBR += 1
#    fig, ax = plt.subplots(figsize=(12, 12))
#    trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    plt.rcParams.update({'figure.autolayout': True})
    plt.title(f'Fig: {FIGNBR} XGBoost Shap values; {filename}; \n {txtmdl}  '
              f'On: {dt_string}', fontsize=14)
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
    FIGNBR += 1
    plt.title(f'Fig: {FIGNBR} XGBoost Shap values; {filename}; \n {txtmdl}'
              f'  On: {dt_string}', fontsize=14)
    plt.yticks(fontsize=10, rotation=35)
    shap.summary_plot(shap_values, X)
    plt.show()

    # Following: 'fig: 20'
    # 'magically two charts in one, they share an axis,
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))
    FIGNBR += 1
    fig.suptitle(f'Fig: {FIGNBR} Shap barchart vs the "summary_plot"')
    shap.summary_plot(shap_values, X)  # this is on the right side
    plt.show()

# https://slundberg.github.io/shap/notebooks/plots/dependence_plot.html


def plot_SHAP_charts2():
    """Plot 4 charts."""
    global FIGNBR
    FIGNBR += 1
    shap.dependence_plot('vehmph', shap_values, X,
                         display_features=X, show=False)
    plt.title(f'Fig: {FIGNBR} Speed vs driver situation"Normal"')
    plt.show()
    FIGNBR += 1
    shap.dependence_plot('rank(1)', shap_values, X,
                         display_features=X, show=False)
    plt.title(f'Fig: {FIGNBR} Speed vs driver situation"Normal"')
    plt.show()

    shap.dependence_plot('dayhour', shap_values, X,
                         display_features=X, show=False)
    FIGNBR += 1
    plt.title(f'Fig: {FIGNBR} Hour of the day vs "other type of vehicle"')
    plt.show()

    shap.dependence_plot('vehage', shap_values, X,
                         display_features=X, show=False)
    FIGNBR += 1
    plt.title(f'Fig: {FIGNBR} Age of Vehicle vs driver situation"Normal"')
    plt.show()

    shap.dependence_plot('drSitu_drgAlcSlpSic', shap_values,
                         X, display_features=X, show=False)
    FIGNBR += 1
    plt.title(f'Fig: {FIGNBR} Speed vs "Drugs, alcohol, Sleep & Sickness"')
    plt.show()
#    return shap_values


def ABS_SHAP(float_array, training_dataframe):
    """Create all the shap values for each record (big df).

    https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d
    Parameters
    ----------
    float_array : TYPE
        Passing in shap_values (as float_array) from explainer.shap_values(X).
    training_dataframe : TYPE
        DESCRIPTION.

    Returns
    -------
    df_shap : dataframe
        DESCRIPTION.
    shap_abs : dataframe
        This is just the same as df_shap, but all positive (absolute value).
    shapvalues_df : dataframe
        4 cols: the 22 'features' each with 'shap_abs' (strength), correlation
    shapcolorlist : series
        The 'color' of each feature (red leaning toward bad crash, blue safer.)

    shap_v, shap_abs, shapvaluesdf, colorlist
    """
    # import matplotlib as plt
    # Make a copy of the input data
    df_shap = pd.DataFrame(float_array)  # ~235,983 (ie each crash x 88 columns)
    feature_list = training_dataframe.columns
    df_shap.columns = feature_list
    df_v = training_dataframe.copy().reset_index().drop('index', axis=1)

    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(df_shap[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),
                         pd.Series(corr_list)], axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature,
    # and Column 2 is the correlation coefficient
    corr_df.columns = ['Variable', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')
    shap_abs = np.abs(df_shap)

    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable', 'SHAP_abs']
    shapvalues_df = k.merge(corr_df, left_on='Variable', right_on='Variable',
                            how='inner')
    shapvalues_df = shapvalues_df.sort_values(by='SHAP_abs',
                                              ascending=True)[-20:]
    shapcolorlist = shapvalues_df['Sign']
    return df_shap, shapvalues_df, shapcolorlist


    # Plot it
    # ax = shapvalues_df.plot.barh(x='Variable', y='SHAP_abs', color=colorlist,
    #                   figsize=(5, 6), legend=False)
    # ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    # ax.set_ylabel('SHAP_abs', fontsize=8)

# The following doesn't work...and won't until 2021 from Carlos Cordoba
# shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
# plt.show()
# shap.TreeExplainer
# alias of shap.explainers._tree.Tree
# shap.GradientExplainer
# alias of shap.explainers._gradient.Gradient
# shap.DeepExplainer
# alias of shap.explainers._deep.Deep
# shap.KernelExplainer
# alias of shap.explainers._kernel.Kernel
# shap.SamplingExplainer
# alias of shap.explainers._sampling.Sampling
# shap.PartitionExplainer
# alias of shap.explainers._partition.Partition


def plot_tree2():
    """Found that this is not working... not sure why. Fix later.

    Returns
    -------
    None.

    """
    ptend = datetime.now()
    now = ptend.strftime('%c')
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
    fig.tight_layout()
    plt.show()

# %% Get & Prepare Data: SQL, CSV, Github


def getdata(records):
    """Take the number passed in (x), gets x big crashes, x small crashes.

    Parameters
    ----------
    records : int
        One half number of records that you want to retrieve.

    Returns
    -------
    A dataframe of the desired records, the SQL used to generate the data.
    """
    sqlstart = datetime.now()
    # conn = pyodbc.connect('DRIVER={SQL Server};'
    #                       'SERVER=(local);'
    #                       'DATABASE=Traffic;'
    #                       'Trusted_Connection=yes;')
    sqlparams = f'@cnt={records},@yr=2014,@veh#=1,@day1=3,@day2=28'
#    sqlparams = '@cnt=35000,@yr=2014,@veh#=2,@day1=11,@day2=19'
    mysql = 'usp_GetEqualNbrMajorMinorCrashesA '
#    mysql = 'select * from Crash110'
    sqlcmd = mysql
    sqlcmd = mysql + sqlparams
    # sqlcmd = ("usp_GetEqualNbrMajorMinorCrashes  @cnt=100,@yr=2014,@veh#=1")
    # sqlcmd = 'select * from vw_CrashSubsetEngineeredR'  #' where vehmph > 0 '
    # url = \
    # 'https://github.com/steveSchneider2/data/tree/main/FloridaTraffic/
    #                                    traffic116k_88_76pc.csv?raw=true'
    # url = 'https://github.com/topics/floridatraffic.csv'
#    pr = pd.read_csv(url)
    traffic_df = pd.read_csv(
        'D:/ML/Python/data/traffic/traffic236k_95_16.csv', header=0)
#    pr = pd.read_csv('data/traffic72k_94_83.csv', header=0)
    # pr = pd.read_csv('D:/dDocuments/ML/Python/data/Traffic/traffic116k_88_76pc.csv')
#    pr = pd.read_sql(sqlcmd, conn)
    sqlend = datetime.now()
    processTime = sqlend - sqlstart
    print('Seconds to download SQL records is: ', processTime.total_seconds())
    # pr.to_csv(r'..\data\traffic140k.csv', index=False)
    return traffic_df, sqlcmd, mysql, sqlparams


def standardize(df):
    """Standardize the data.

    Parameters
    ----------
    df : dataframe
        The data we want to standardize.

    Returns
    -------
    scaled_df : dataframe
        Original data, but standardized.

    """
    # https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc
    names = df.columns
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    return scaled_df


def datapreparation():
    """Do certain things to the data so it is ready for the model.

    Returns
    -------
    None.

    """
    pr.drop(['Crash_year', 'monthday', 'speeddif', 'vMphGrp'], axis=1,
            inplace=True)
    # pr.drop(['license', 'drvsex', 'drvagegrp',
    #          'drDistract'], axis=1, inplace=True)
    if 'vehicle_number' in pr.columns:
        pr.pop('vehicle_number')  # Return item and drop from frame
    if 'InvolvedVeh' in pr.columns:
        pr.pop('InvolvedVeh')  # Return item and drop from frame
    # type(pr.index)
    # pr
    # type(pr)
    # pr.shape     # Like the R dim command
    # pr.describe()
    # pr.iloc[:9]
    # pr.head()
    # pr.corr()
    # nbrrecords = pr.shape[0]
    # pr.describe(include=['object', 'category']).transpose()
    # pr.describe(
    #     include=['object', 'category']
    # ).transpose().sort_values(by='unique', ascending=False)
    # pr.plot(pr['wday'], pr['vehmph'], kind='scatter')

    # list(pr.select_dtypes(np.object).columns)  # Get just the non-numerics...

    todummylist = list(pr.select_dtypes(np.object).columns)

    class_names = (['minor', 'major'])
    # pr.shape
    pr.fillna(method='pad', inplace=True)
    trn_data_df = pr  # .copy had strange results
    trn_data_df.info(memory_usage='deep')

    j = pd.concat([pr['vehmph'], pr.crash], axis=1)
    bigcr = j.vehmph[j.crash == 1]
    smlcr = j.vehmph[j.crash == 0]

    # j = pd.concat([pr.rdSurfac,pr.crash], axis=1)
    # labels = f'Drivers Situation in {pr.shape[0]} crashes'
    # bigcr = j.rdSurfac[j.crash==1]
    # smlcr = j.rdSurfac[j.crash==0]
    # if howfartoexecute != 'learningcurves':
#    plot_histograms_stacked(bigcr, smlcr, 3,
#                            'Actual accidents vs Road Surface',
#                            labels)

    if 'crash' in pr.columns:  # Allow me to run this frm multiple places in code
        target = pr.pop('crash')  # Return item and drop from frame

    # Function to dummy all the categorical variables used for modeling
    # courtesy of April Chen

    def dummy_pr(pr, todummylist):
        for x in todummylist:
            dummies = pd.get_dummies(pr[x], prefix=x, dummy_na=False)
            pr = pr.drop(x, 1)
            pr = pd.concat([pr, dummies], axis=1)
        return pr
    # labels = f'{pr.shape[0]} crashes'
    trn_data_df = dummy_pr(trn_data_df, todummylist)
    trn_data_df.head()
    # ### Note WE ARE NOT STANDARDIZING...WHEN THE BELOW IS COMMENTED.!!!
    # trn_data_df = standardize(trn_data_df)
    trng_prct = .75
    Xtrain, Xtest, ytrain, ytest =\
        train_test_split(trn_data_df, target, test_size=1 - trng_prct, random_state=200)
    # type(X_train)
    # X_train.shape
    # X_test.shape

    txt0 = f'Given ~{pr.shape[0]} records, {trng_prct*100}% are used for training.'
    txt1 = f'  That leaves {len(ytest)} for testing (evaluating the model).'
    txt2 = 'Those are the numbers reflected in this chart.'
    tstsummary = ('{}').format(txt0) + '\n' + ('{}').format(txt1) +\
        '\n' + ('{}').format(txt2)

    categoricalVariableList = pr.describe(
        include=['object', 'category']
    ).transpose().sort_values(by='unique', ascending=False)

    categoricalVariableList = categoricalVariableList.to_string(header=False)

    return Xtrain, Xtest, ytrain, ytest, pr, categoricalVariableList,\
        tstsummary, trn_data_df, target, class_names, bigcr, smlcr


# %% MAIN Code--> Call to get & prep the data
txtmdl, mdl, mdlsttime, filename, start = \
    setstartingconditions()

# howfartoexecute = 'ALLlearningcurves'
howfartoexecute = 'learningcurves'

pr, SQLCommand, SQL, SqlParameters = \
    getdata(118000)  # return 2*118,000...

X_train, X_test, y_train, y_test, pr, categoricalVariablesList,\
    testsummary, X, y, class_names_gnb, major, minor = \
    datapreparation()

# if howfartoexecute == 'learningcurves':
#     print('quit')
#     sys.exit()
# %%  5 Charts: data as it looks prior to modeling --> DRIVES FEATURE ENGINEERING
FIGNBR = 0
plot_histograms_stacked(major, minor, 100)
plot4histograms()

X_train.describe()
y_train.head()
X_test.describe()

# %% Define & Fit the model


def strmodel():
    """Create the Params, and then create and fit the model.

    Returns
    -------
    None.
    # Choose all predictors except target & IDcols
    # https://stats.stackexchange.com/questions/204489/discussion-about-overfit-in-xgboost
    # https://www.capitalone.com/tech/machine-learning/how-to-control-your-xgboost-model/
    # https://towardsdatascience.com/xgboost-theory-and-practice-fb8912930ad6

    """
    target = 'crash'
    predictors = [x for x in X_train.columns if x not in [target]]  # , IDcol]]
    # predictors = y_train.columns
    mdlparams = {
        # besides fixing the error, this next line improved accuracy by 1% :)
        'use_label_encoder': False,
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
        "min_child_weight": 2,  # Pretty sensitive: raise to raise up training
        # Dropping gamma frees training to drop, no impact on test! (:
        "gamma": .5,  # <--Drop from 5 thru 8 below .8 dropped log loss!  TRAINING impact!!!
        # Sampling:
        "subsample": 0.95,  # (bagging of rows) (def=1); help w/overfitting
        "colsample_bytree": 0.95,  # avoid some columns to take too much credit
        'colsample_bylevel': 0.3,
        # 'colsample_bynode': 0.6,
        "objective": 'binary:logistic',
        "nthread": -1,
        "scale_pos_weight": 1,
        "eval_metric": 'error',  # "auc",  # vs "logloss"
        # "seed": 27,
        'random_state': 200,
        "verbosity": 1,
        #        'single_precision_histogram': True,
        "n_jobs": -1}
    xgb1mdl = XGBClassifier(**mdlparams)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    mdlstart = datetime.now()
    xgb1mdl.fit(X_train, y_train, early_stopping_rounds=100,
                eval_metric=["error", "logloss"],  # 06 Oct 2021 fixes warning
                eval_set=eval_set, verbose=100)
    mdlend = datetime.now()
    getdataTime = mdlend - mdlstart
    print('Seconds to train model is: ', getdataTime.total_seconds())
    mdl_predictions = xgb1mdl.predict(X_test)
    # mdl_predictions = xgb1mdl.predict_proba(X_test)
    y_actuals = y_test
    print(confusion_matrix(y_actuals, mdl_predictions))
    # accuracy_score(X_train, X_pred )
    mdlaccuracy = accuracy_score(y_test, mdl_predictions)
    # print(f'Accuracy: {mdlaccuracy} ')  # , or below, perhaps a nicer way...
    print('Accuracy: %.2f%%' % (mdlaccuracy * 100))
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, mdl_predictions)
    auc_xgb = round(auc(fpr_xgb, tpr_xgb), 2)
    con_mtrx = confusion_matrix(y_test, mdl_predictions)
    mdlresults = xgb1mdl.evals_result()
    mdlepochs = len(mdlresults['validation_0']['error'])
    xaxis = range(0, mdlepochs)
    mdlTrainingSeconds = round(getdataTime.total_seconds(), 2)
    if mdlTrainingSeconds > 60:
        mdlTrainingSeconds = round(mdlTrainingSeconds / 60, 3)
        modelTngTime = str(mdlTrainingSeconds) + ' Mins'
    else:
        modelTngTime = str(mdlTrainingSeconds) + ' Sec'
    con_mtrx = confusion_matrix(y_test, mdl_predictions)
    return xgb1mdl, mdl_predictions, y_actuals, mdlaccuracy, mdlparams, mdlresults, \
        xaxis, mdlepochs, modelTngTime, con_mtrx
    # Note 'xgb1' is the resulting model!!!


print('\n We are starting to model... \n')
xgb1, predictions, actuals, accuracy, params, results, x_axis, epochs,\
    mdlTngTime, cm = strmodel()  # i need a better function name!
print('\n done modeling... \n')
# #############################################
# pr['vehmph'].value_counts(sort=True)

# %% Create variables for various charts
startcls = datetime.now()
dt_string = startcls.strftime("%m/%d/%Y %H:%M:%S")
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
keyparams = str(f'Learn:{lr}   depth:   {md}\ntrees: {ne}  HiPoint: {bt} '
                f'\nModel tng time: {mdlTngTime}')


# #######################  ROC CURVE ###############################
mdl = XGBClassifier.mro()[0]
txtparams = json.dumps(params)
txtparams = txtparams.replace(',', '\n')

# %% 11 Charts: Model's validity, accuracy,calibr, confidenc, lrning curves (3)
# Learning curve #1 & 2 (SIDE-BY-SIDE)
plot_learningcurves_logloss_classerror()  # THIS IS THE BEST!

# Following two options for optimum visibility of a tree:
plot_tree(0, 'LR')
xgb.to_graphviz(xgb1, rankdir='LR')  # this line goes to console...

# sys.exit()

ROC_rcvr_operating_curve(xgb1)  # Fig 8

howfartoexecute = 'learningcurves'  # 'alltheway'
# if howfartoexecute == 'learningcurves':
#     print('quit')
#     sys.exit()

plot_ConfusionMatrixChart(xgb1)  # makes 2...2nd is by percentage

print(f'Start Calibration:  {dt_string}  \nExpect this to take 5 minutes...')

plot_calibration_curve(xgb1, "XGBoost")  # expect 4.5 min
# sys.exit()
plot_2confidenceHistograms()
lcstart = datetime.now()
print(f'histograms done in {lcstart-startcls}')



# %% Scalability, performance curves (3 in 1 chart)
# too long to run for PASS... so comment out.
# print('Coming: LEARNING CURVE, SCALABILITY OF MODEL, PERFORMANCE OF'
#       f' MODEL...{lcstart}')
# print('This takes more than several minutes!!!')
# cv = 'none'
# 3 plots (in a row) LEARNING CURVE, SCALABILITY OF MODEL, PERFORMANCE OF MODEL
# This might take 15 minutes!!! Only enable, when you are patient!
#############
'''plot_scalability_performance(xgb1, X, y,  ylim=(0.7, 1.01), n_jobs=4)
 13 min OR LONGER!!
# Also generates this multi-page warning: (so... fix it, already!)
#  UserWarning: Use subset (sliced data) of np.ndarray is not recommended
because it will generate extra copies and increase memory consumption'''
#############
# %%  4 Charts: Model's explainability --> feature importance
plot_tree(1, 'LR')

#  following makes sklearn.utils.Bunch

startcls = datetime.now()
dt_string = startcls.strftime("%m/%d/%Y %H:%M:%S")
print(f'Start Permuting importances:  {dt_string} expecting 4.5 minutes...')

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
# the below line creates the user warning that won't go away.
result = permutation_importance(xgb1, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
end = datetime.now()
permutationDuration = end - startcls
durationSeconds = permutationDuration.total_seconds()
# The permutation calculation took 9.5 minutes!!!! Longer than modeling!
# ##################CHARTING ################################################

result, sorted_idx =\
    plot_permutationimportancechart(durationSeconds)  # Fig: 14?
max_num_features = 25
plot_feature_importance(max_num_features, 'weight')  # Fig: 15

# There are 3 methods to measure importance in XGBoost:
# Weight. The number of times a feature is used to split the data across all trees.
# Cover. The number of times a feature is used to split the data across all trees
# weighted by the number of training data points that go through those splits.
# Gain. The average training loss reduction gained when using a feature for splitting.

plot_feature_importance(max_num_features, 'gain')  # could also be 'gain'
plot_sidebyside_feature_importanceVSpermutation(max_num_features)  # Plot 17
# %% Shap charts coming...
explainer = shap.TreeExplainer(xgb1)
print('Need 16 minutes to do these shap calculations...')
startShap = datetime.now()
# notice, 1st for ALL records, 2nd just for training... (just saying)
shap_values = explainer.shap_values(X)  # 19+ minutes!  @depth = 9

# Generate A slicable set of parallel arrays representing a SHAP explanation.
waterfall_values = explainer(X_train)  # ~8 minutes!
durationShap = datetime.now() - startShap
print('Shap_values calculated:'
      '  {} ... 3 waterfalls coming:'.format(durationShap))
shap.plots.waterfall(waterfall_values[0])
shap.plots.waterfall(waterfall_values[1])
shap.plots.waterfall(waterfall_values[2])

# https://medium.com/dataman-in-ai/the-shap-with-more-elegant-charts-bc3e73fa1c0c
shap.plots.bar(waterfall_values.cohorts(2).abs.mean(0))

shap_v, shapvaluesdf, colorlist = ABS_SHAP(shap_values, X)
print('ABS_Shap calculated:  {} ...'.format(
    datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
# shap_v = ABS_SHAP(shap_values,X)
print('Printing 4 shap charts')
plot_SHAP_charts(shapvaluesdf)  # <--THREE charts! ~30 sec for all 7
print('Printed 4 focused shap charts: {} ...'.format(
    datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
plot_SHAP_charts2()  # <--Five charts! ~30 sec for all 7
# plt.rcdefaults()

# %% We're done!
plot_tree(xgb1.get_booster().best_iteration, 'LR')
xgb1.get_booster().num_boosted_rounds()
duration = datetime.now() - start
print(f'Total duration for {filename} is: {duration}')
# %% extra:
# https://h1ros.github.io/posts/interpretability-of-prediction-for-boston-housing-using-shap/
# this is creating: <shap.plots._force.AdditiveForceVisualizer at 0x21103d0f5b0>
# and Nothing is going to the plot plane... perhaps run in Jupyter?
# i = 0
# shap.force_plot(explainer.expected_value, shap_values[i, :], X.iloc[i, :])
