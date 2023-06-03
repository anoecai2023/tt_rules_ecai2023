import warnings

from sklearn.tree import DecisionTreeRegressor

from sklearn import metrics

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
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from skorch import NeuralNetRegressor
from skorch.callbacks import EpochScoring
from skorch.callbacks import LRScheduler, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config.config import Config, transform_input_filters, transform_input_filters2, transform_input_optim, \
    transform_input_thr
from config.config import two_args_str_int
from utils import read_csv, DBEncoder, sample_generator

from pathlib import Path

# Load configuration


# Seed experiments
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
aucs = []
housing = fetch_california_housing()
rules = []
conds = []
accs = []
for seed in [0,1,6,7,8]:
    X_df, y_df, f_df, label_pos = read_csv("../dataset" + "/compas/compas" + ".data",
                                           "../dataset" + "/compas/compas" + ".info",
                                           shuffle=True)
    db_enc = DBEncoder(f_df)
    db_enc.fit(X_df, y_df)
    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
    y = np.argmax(y, axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test)



    #linear_model = DecisionTreeRegressor(max_depth=5).fit(X_train, Y_train)

    from aix360.algorithms.rbm import FeatureBinarizer
    fb = FeatureBinarizer(negations=True)
    X_train_fb = fb.fit_transform(pd.DataFrame(X_train))
    X_test_fb = fb.transform(pd.DataFrame(X_test))




    #
    # model = TTnet_general(features_size=config.general.features_size,
    #                       index=config.general.index_c_start,
    #                       oneforall=True,
    #                       T=0.0,
    #                       chanel_interest=1,
    #                       k=args.kernel_size[0],
    #                       device="cpu",
    #                       c_a_ajouter=0,
    #                       filter_size=args.filter_size[0],
    #                       p=args.padding[0],
    #                       s=args.stride[0],
    #                       embed_size=args.embed_size[0],
    #                       repeat=3,
    #                       block_LTT="LR",
    #                         regression = True
    #                       )
    #
    # auc = EpochScoring(scoring=args.epoch_scoring_scoring,
    #                    lower_is_better=args.epoch_scoring_lower_is_better)
    # lr_scheduler = LRScheduler(policy=ReduceLROnPlateau, monitor=args.lr_scheduler_monitor,
    #                            mode=args.lr_scheduler_mode, patience=args.lr_scheduler_patience,
    #                            factor=args.lr_scheduler_factor,
    #                            verbose=args.lr_scheduler_verbose)
    # early_stopping = EarlyStopping(monitor=args.early_stopping_monitor, patience=args.early_stopping_patience,
    #                                threshold=args.early_stopping_threshold,
    #                                threshold_mode=args.early_stopping_threshold_mode,
    #                                lower_is_better=args.early_stopping_lower_is_better)
    #
    # net = NeuralNetRegressor(model,
    #                             criterion=torch.nn.MSELoss,
    #                           device=args.device,
    #                           max_epochs=args.epochs_max,
    #                           callbacks=[lr_scheduler, early_stopping])
    #
    # optimizers = transform_input_optim(args.optimizers)
    # grid_params = {
    #     'lr': args.lrs,
    #     'optimizer': optimizers,  # [optim.AdamW, optim.Adam]
    #     'module__k': args.kernel_size,  # [i for i in range(3,10)],
    #     'module__s': args.stride,
    #     'module__p': args.padding,
    #     'module__repeat': args.repeat,
    #     'module__regression': [True],
    #     'module__filter_size': args.filter_size,  # [i for i in range(2,21)],
    #     'module__embed_size': args.embed_size,
    #     # 'batch_size': args.batch_size,
    #     'module__features_size': [config.general.features_size],
    #     # 'module__index': [config.general.index_c_start],
    #     # 'module__nclass': [1],
    #     # 'module__block_LTT':args.family_LTT,
    #     # 'module__dropoutclass':args.dropout_value_class,
    #     # 'module__dropoutLTT':args.dropout_value_cnn,
    #     # 'module__poly_flag':args.poly_flag,
    #     # 'optimizer__weight_decay': args.weight_decay
    # }
    #
    # scoring = {"MSE": make_scorer(mean_squared_error)}
    #
    # grid_net = GridSearchCV(net, grid_params, cv=3, scoring=scoring, verbose=1, refit="MSE")
    #
    # # Training
    # start_time = time.time()
    # print(X_train_fb.to_numpy(), type(Y_train[0]))
    # Y_train = Y_train.reshape(-1, 1)
    #
    # result = grid_net.fit(1.0*X_train_fb.to_numpy().astype(np.float32), Y_train.astype(np.float32))
    # end_time = time.time()
    # print("End Grid search: BEST parameters: ", grid_net.best_params_, "\n")
    # print("End Grid search: BEST score: ", grid_net.best_score_, "\n")
    # print("End Grid search: cv_results_ : ", grid_net.cv_results_, "\n")
    #
    #
    # # Testing
    # pred = grid_net.predict(X_test_fb.to_numpy().astype(np.float32))
    # #pred_proba = grid_net.predict_proba(X_test_fb.to_numpy().astype(np.float32))[::, 1]
    #
    # # save your model or results
    # joblib.dump(grid_net, path_save_model + 'model.pkl')
    # # load your model for further usage
    # load_best_gs = joblib.load(path_save_model + 'model.pkl')
    #
    #
    #
    # # Testing
    # pred = load_best_gs.predict(X_test_fb.to_numpy().astype(np.float32))
    # #pred_proba = load_best_gs.predict_proba(X_test_fb.to_numpy().astype(np.float32))[::, 1]
    #
    # print()
    # print(f'RMSE error = {mean_squared_error(Y_test, pred)}')
    #


    from aix360.algorithms.rbm import LogisticRuleRegression
    linear_model = LogisticRuleRegression(lambda0=0.01, lambda1=0.001)
    # print(X_train_fb.shape)X_test_fb
    linear_model.fit(X_train_fb, Y_train)
    Y_pred = linear_model.predict(X_test_fb)
    y_pred_proba = linear_model.predict_proba(X_test_fb)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_proba)
    auc = metrics.auc(fpr, tpr)
    print("AUC", auc)
    print(linear_model.explain()["rule"].to_numpy().tolist())
    Rule = len(linear_model.explain()["rule"].to_numpy().tolist())-1
    condition = np.sum([x.count("AND") for x in linear_model.explain()["rule"].to_numpy().tolist()]) + Rule
    print(Rule, condition)
    rules.append(Rule)
    conds.append(condition)
    aucs.append(auc)
    aucs.append(auc)
    accs.append(accuracy_score(Y_test, Y_pred))
print(np.mean(aucs))
print(np.std(aucs))

print(np.mean(accs))
print(np.std(accs))

print(np.mean(conds))
print(np.std(conds))

print(np.mean(rules))
print(np.std(rules))



