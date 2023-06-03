import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
import argparse
import random
import datetime
import os
import json
import joblib
import numpy as np
import time
import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from skorch.callbacks import LRScheduler, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config.config import Config, transform_input_filters, transform_input_filters2, transform_input_optim, \
    transform_input_thr
from config.config import two_args_str_int
from pathlib import Path



# Seed experiments
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





# Load FICO heloc data with special values converted to np.nan
from aix360.datasets.heloc_dataset import HELOCDataset, nan_preprocessing
data = HELOCDataset(custom_preprocessing=nan_preprocessing).data()

# Separate target variable
y = data.pop('RiskPerformance')

# Split data into training and test sets using fixed random seed
from sklearn.model_selection import train_test_split
aucs=[]
rules = []
conds = []
accs = []
for i in range(5):
    dfTrain, dfTest, yTrain, yTest = train_test_split(data, y, random_state=i, stratify=y)
    dfTrain.head().transpose()

    from aix360.algorithms.rbm import FeatureBinarizer
    fb = FeatureBinarizer(negations=True, returnOrd=True)
    dfTrain, dfTrainStd = fb.fit_transform(dfTrain)
    dfTest, dfTestStd = fb.transform(dfTest)
    dfTrain['ExternalRiskEstimate'].head()




    # from aix360.algorithms.rule_induction.ripper import RipperExplainer
    #
    # estimator = RipperExplainer()
    # estimator.fit(dfTrain, yTrain) # Run RIPPER rule induction
    # from sklearn.metrics import accuracy_score
    # print('Training accuracy:', accuracy_score(yTrain, estimator.predict(dfTrain)))
    # print('Test accuracy:', accuracy_score(yTest, estimator.predict(dfTest)))
    # #trxf_ruleset = estimator.explain()
    # #print(str(trxf_ruleset))
    #
    #
    #
    # from aix360.algorithms.rbm import BooleanRuleCG
    # br = BooleanRuleCG(lambda0=1e-3, lambda1=1e-3, CNF=True)
    #
    # # Train, print, and evaluate model
    # br.fit(dfTrain, yTrain)
    # from sklearn.metrics import accuracy_score
    # print('Training accuracy:', accuracy_score(yTrain, br.predict(dfTrain)))
    # print('Test accuracy:', accuracy_score(yTest, br.predict(dfTest)))
    # print('Predict Y=0 if ANY of the following rules are satisfied, otherwise Y=1:')
    # print(br.explain()['rules'])

    #linear_model = DecisionTreeRegressor(max_depth=5).fit(X_train, Y_train)

    from aix360.algorithms.rbm import LogisticRuleRegression
    lrr = LogisticRuleRegression(lambda0=0.005, lambda1=0.001, useOrd=True)
    #lrr = LogisticRegression()
    # Train, print, and evaluate model
    lrr.fit(dfTrain, yTrain, dfTrainStd)
    print('Training accuracy:', accuracy_score(yTrain, lrr.predict(dfTrain, dfTrainStd)))
    print('Test accuracy:', accuracy_score(yTest, lrr.predict(dfTest, dfTestStd)))
    print('Probability of Y=1 is predicted as logistic(z) = 1 / (1 + exp(-z))')
    print('where z is a linear combination of the following rules/numerical features:')
    #lrr.explain()
    from sklearn import metrics
    y_pred_proba = lrr.predict_proba(dfTest, dfTestStd)
    print(yTest,y_pred_proba)

    fpr, tpr, thresholds = metrics.roc_curve(yTest, y_pred_proba)
    auc = metrics.auc(fpr, tpr)
    print("AUC`", auc)
    aucs.append(auc)
    print(lrr.explain())
    print(lrr.explain()["rule/numerical feature"].to_numpy().tolist())
    Rule = len(lrr.explain()["rule/numerical feature"].to_numpy().tolist()) - 1
    condition = np.sum([x.count("AND") for x in lrr.explain()["rule/numerical feature"].to_numpy().tolist()]) + Rule
    print(Rule, condition)
    rules.append(Rule)
    conds.append(condition)
    accs.append(accuracy_score(yTest, lrr.predict(dfTest, dfTestStd)))



print(np.mean(aucs))
print(np.std(aucs))

print(np.mean(accs))
print(np.std(accs))

print(np.mean(conds))
print(np.std(conds))

print(np.mean(rules))
print(np.std(rules))


