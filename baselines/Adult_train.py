import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score
from aix360.algorithms.rule_induction.ripper import RipperExplainer
import time
from aix360.algorithms.rule_induction.rbm.boolean_rule_cg import BooleanRuleCG as BRCG
from aix360.algorithms.rbm import FeatureBinarizer


data_type = {'age': float,
             'workclass': str,
             'fnlwgt': float,
             'education': str,
             'education-num': float,
             'marital-status': str,
             'occupation': str,
             'relationship': str,
             'race': str,
             'sex': str,
             'capital-gain': float,
             'capital-loss': float,
             'native-country': str,
             'hours-per-week': float,
             'label': str}

col_names = ['age', 'workclass', 'fnlwgt', 'education',
             'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week',
             'native-country', 'label']

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                 header=None,
                 delimiter=', ',
                 engine='python',
                 names=col_names,
                 dtype=data_type)

df.columns = df.columns.str.replace('-', '_')
TARGET_COLUMN = 'label'
print(df.head())
POS_VALUE = '>50K' # Setting positive value of the label for which we train
values_dist = df[TARGET_COLUMN].value_counts()
print('Positive value {} occurs {} times.'.format(POS_VALUE,values_dist[POS_VALUE]))
print(values_dist)
# This is distribution of the two values of the target label

accuracy = {"LogisticRuleRegression":[],
            "RipperExplainer":[],
            "BRCG":[]
            }
aucs = []
rules = []
conds = []
accs = []

for i in range(5):
    train, test = train_test_split(df, test_size=0.2, random_state=i)
    # Split the data set into 80% training and 20% test set
    print('Training set:')
    print(train[TARGET_COLUMN].value_counts())
    print('Test set:')
    print(test[TARGET_COLUMN].value_counts())

    y_train = train[TARGET_COLUMN]
    x_train = train.drop(columns=[TARGET_COLUMN])

    y_test = test[TARGET_COLUMN]
    x_test = test.drop(columns=[TARGET_COLUMN])
    # Split data frames into features and label


    y_train2 = train[TARGET_COLUMN].apply(lambda x: 1 if x == POS_VALUE else 0)
    x_train2 = train.drop(columns=[TARGET_COLUMN])

    y_test2 = test[TARGET_COLUMN].apply(lambda x: 1 if x == POS_VALUE else 0)
    x_test2 = test.drop(columns=[TARGET_COLUMN])

    fb = FeatureBinarizer(negations=True)
    X_train_fb = fb.fit_transform(x_train2)
    x_test_fb = fb.transform(x_test2)
    # Split data frames into features and label

    fb = FeatureBinarizer(negations=True, returnOrd=True)
    dfTrain, dfTrainStd = fb.fit_transform(x_train2)
    dfTest, dfTestStd = fb.transform(x_test2)




    # LogisticRuleRegression
    from aix360.algorithms.rbm import LogisticRuleRegression
    lrr = LogisticRuleRegression(lambda0=0.005, lambda1=0.001, useOrd=True)
    # Train, print, and evaluate model
    lrr.fit(dfTrain, y_train2, dfTrainStd)
    y_pred_proba = lrr.predict_proba(dfTest, dfTestStd)
    print(dfTrain, dfTrainStd)
    print('Training accuracy:', accuracy_score(y_train2, lrr.predict(dfTrain, dfTrainStd)))
    print('Test accuracy:', accuracy_score(y_test2, lrr.predict(dfTest, dfTestStd)))
    accuracy["LogisticRuleRegression"].append(accuracy_score(y_test2, lrr.predict(dfTest, dfTestStd)))
    print('Probability of Y=1 is predicted as logistic(z) = 1 / (1 + exp(-z))')
    print('where z is a linear combination of the following rules/numerical features:')
    trxf_ruleset = lrr.explain()
    print(str(trxf_ruleset))
    from sklearn import metrics
    print(y_test2, y_pred_proba)
    #print(y_test2.numpy(), y_pred_proba.numpy())
    fpr, tpr, thresholds = metrics.roc_curve(y_test2, y_pred_proba)
    auc = metrics.auc(fpr, tpr)
    print("AUC", auc)

    print(lrr.explain()["rule/numerical feature"].to_numpy().tolist())
    Rule = len(lrr.explain()["rule/numerical feature"].to_numpy().tolist()) - 1
    condition = np.sum([x.count("AND") for x in lrr.explain()["rule/numerical feature"].to_numpy().tolist()]) + Rule
    print(Rule, condition)
    rules.append(Rule)
    conds.append(condition)
    aucs.append(auc)
    accs.append(accuracy_score(y_test2, lrr.predict(dfTest, dfTestStd)))
print(np.mean(aucs))
print(np.std(aucs))

print(np.mean(accs))
print(np.std(accs))

print(np.mean(conds))
print(np.std(conds))

print(np.mean(rules))
print(np.std(rules))

    #
    #
    # #RipperExplainer
    # estimator = RipperExplainer()
    # start_time = time.time()
    # estimator.fit(x_train, y_train, target_label=POS_VALUE) # Run RIPPER rule induction
    # end_time = time.time()
    # print('Training time (sec): ' + str(end_time - start_time))
    # # compute performance metrics on test set
    # y_pred = estimator.predict(x_test)
    # print('Accuracy:', accuracy_score(y_test, y_pred))
    # print('Balanced accuracy:', balanced_accuracy_score(y_test, y_pred))
    # print('Precision:', precision_score(y_test, y_pred, pos_label=POS_VALUE))
    # print('Recall:', recall_score(y_test, y_pred, pos_label=POS_VALUE))
    # trxf_ruleset = estimator.explain()
    # print(str(trxf_ruleset))
    #
    #
    #
    #
    # explainer = BRCG(silent=True)
    # start_time = time.time()
    # explainer.fit(X_train_fb, y_train2)
    # end_time = time.time()
    # print('Training time (sec): ' + str(end_time - start_time))
    # y_pred = explainer.predict(x_test_fb)
    # print('Accuracy:', accuracy_score(y_test2, y_pred))
    # print('Balanced accuracy:', balanced_accuracy_score(y_test2, y_pred))
    # print('Precision:', precision_score(y_test2, y_pred, pos_label=1))
    # print('Recall:', recall_score(y_test2, y_pred, pos_label=1))
    # trxf_ruleset = explainer.explain()
    # print(str(trxf_ruleset))




