import warnings

from sklearn.tree import DecisionTreeRegressor

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
from ttnet.helper import read_csv, DBEncoder, sample_generator
from ttnet.model import TTnet_general
from pathlib import Path

# Load configuration
#config_general = Config()
config = Config(path="../config/house/")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="house")
parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--device", default=config.general.device, choices=["cpu", "gpu"])
parser.add_argument("--data_path", default=config.general.data_path)
parser.add_argument("--result_path", default=config.general.result_path)
parser.add_argument("--rf_path", default=config.general.rf_path)
parser.add_argument("--completeness", default=config.general.completeness)
parser.add_argument("--private_train_flag", default=config.general.private_train_flag, type=bool)
parser.add_argument("--features_size", default=config.general.features_size, type=int)
parser.add_argument("--index_c_start", default=config.general.index_c_start, type=int)

parser.add_argument("--flag", default=config.train_RF.flag, type=bool)
parser.add_argument("--rf_cv", default=config.train_RF.cv, type=int)

parser.add_argument("--n_estimators", default=config.rf_grid_param.n_estimators, type=transform_input_filters)
parser.add_argument("--criterion", default=config.rf_grid_param.criterion, type=transform_input_filters2)
parser.add_argument("--min_samples_split", default=config.rf_grid_param.min_samples_split, type=transform_input_filters)
parser.add_argument("--min_samples_leaf", default=config.rf_grid_param.min_samples_leaf, type=transform_input_filters)
parser.add_argument("--max_features", default=config.rf_grid_param.max_features, type=transform_input_filters2)

parser.add_argument("--size", default=config.gen.size, type=int)
parser.add_argument("--cont", default=config.gen.cont)
parser.add_argument("--clean", default=config.gen.clean, type=bool)
parser.add_argument("--proba_threshold", default=config.gen.proba_threshold, type=float)

parser.add_argument("--X_test_size", default=config.train.X_test_size, type=float)
parser.add_argument("--epochs_max", default=config.train.epochs_max, type=int)
parser.add_argument("--ttnet_cv", default=config.train.cv, type=int)

parser.add_argument("--epoch_scoring_scoring", default=config.train.epoch_scoring.scoring)
parser.add_argument("--epoch_scoring_lower_is_better", default=config.train.epoch_scoring.lower_is_better, type=bool)

parser.add_argument("--lr_scheduler_monitor", default=config.train.lr_scheduler.monitor)
parser.add_argument("--lr_scheduler_mode", default=config.train.lr_scheduler.mode)
parser.add_argument("--lr_scheduler_patience", default=config.train.lr_scheduler.patience, type=int)
parser.add_argument("--lr_scheduler_factor", default=config.train.lr_scheduler.factor, type=float)
parser.add_argument("--lr_scheduler_verbose", default=config.train.lr_scheduler.verbose, type=bool)

parser.add_argument("--early_stopping_monitor", default=config.train.early_stopping.monitor)
parser.add_argument("--early_stopping_patience", default=config.train.early_stopping.patience, type=int)
parser.add_argument("--early_stopping_threshold", default=config.train.early_stopping.threshold, type=float)
parser.add_argument("--early_stopping_threshold_mode", default=config.train.early_stopping.threshold_mode)
parser.add_argument("--early_stopping_lower_is_better", default=config.train.early_stopping.lower_is_better, type=bool)

parser.add_argument("--lrs", default=config.ttnet_grid_param.lrs, type=transform_input_thr)
parser.add_argument("--optimizers", default=config.ttnet_grid_param.optimizers)
parser.add_argument("--kernel_size", default=config.ttnet_grid_param.kernel_size, type=transform_input_filters)
parser.add_argument("--stride", default=config.ttnet_grid_param.stride, type=transform_input_filters)
parser.add_argument("--padding", default=config.ttnet_grid_param.padding, type=transform_input_filters)
parser.add_argument("--repeat", default=config.ttnet_grid_param.repeat, type=transform_input_filters)
parser.add_argument("--filter_size", default=config.ttnet_grid_param.filter_size, type=transform_input_filters)
parser.add_argument("--embed_size", default=config.ttnet_grid_param.embed_size, type=transform_input_filters)
parser.add_argument("--batch_size", default=config.ttnet_grid_param.batch_size, type=transform_input_filters)

args = parser.parse_args()

# Seed experiments
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# create res folder
date = str(datetime.datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
slsh = "/"
path_save_model = args.result_path + date + slsh
print()
print("Folder save: ", path_save_model)
if not os.path.exists(path_save_model):
    os.makedirs(path_save_model)
with open(path_save_model + 'commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
print("Use Hardware : ", args.device)
print()

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
aucs = []
housing = fetch_california_housing()
rules = []
conds = []
for seed in range(5):
    X_train, X_test, Y_train, Y_test = train_test_split(housing["data"], housing["target"], test_size = 0.2, random_state = seed)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    #linear_model = DecisionTreeRegressor(max_depth=5).fit(X_train, Y_train)

    from aix360.algorithms.rbm import FeatureBinarizer
    fb = FeatureBinarizer(negations=True)
    X_train_fb = fb.fit_transform(pd.DataFrame(X_train))
    X_test_fb = fb.transform(pd.DataFrame(X_test))





    model = TTnet_general(features_size=config.general.features_size,
                          index=config.general.index_c_start,
                          oneforall=True,
                          T=0.0,
                          chanel_interest=1,
                          k=args.kernel_size[0],
                          device="cpu",
                          c_a_ajouter=0,
                          filter_size=args.filter_size[0],
                          p=args.padding[0],
                          s=args.stride[0],
                          embed_size=args.embed_size[0],
                          repeat=3,
                          block_LTT="LR",
                            regression = True
                          )

    auc = EpochScoring(scoring=args.epoch_scoring_scoring,
                       lower_is_better=args.epoch_scoring_lower_is_better)
    lr_scheduler = LRScheduler(policy=ReduceLROnPlateau, monitor=args.lr_scheduler_monitor,
                               mode=args.lr_scheduler_mode, patience=args.lr_scheduler_patience,
                               factor=args.lr_scheduler_factor,
                               verbose=args.lr_scheduler_verbose)
    early_stopping = EarlyStopping(monitor=args.early_stopping_monitor, patience=args.early_stopping_patience,
                                   threshold=args.early_stopping_threshold,
                                   threshold_mode=args.early_stopping_threshold_mode,
                                   lower_is_better=args.early_stopping_lower_is_better)

    net = NeuralNetRegressor(model,
                                criterion=torch.nn.MSELoss,
                              device=args.device,
                              max_epochs=args.epochs_max,
                              callbacks=[lr_scheduler, early_stopping])

    optimizers = transform_input_optim(args.optimizers)
    grid_params = {
        'lr': args.lrs,
        'optimizer': optimizers,  # [optim.AdamW, optim.Adam]
        'module__k': args.kernel_size,  # [i for i in range(3,10)],
        'module__s': args.stride,
        'module__p': args.padding,
        'module__repeat': args.repeat,
        'module__regression': [True],
        'module__filter_size': args.filter_size,  # [i for i in range(2,21)],
        'module__embed_size': args.embed_size,
        # 'batch_size': args.batch_size,
        'module__features_size': [config.general.features_size],
        # 'module__index': [config.general.index_c_start],
        # 'module__nclass': [1],
        # 'module__block_LTT':args.family_LTT,
        # 'module__dropoutclass':args.dropout_value_class,
        # 'module__dropoutLTT':args.dropout_value_cnn,
        # 'module__poly_flag':args.poly_flag,
        # 'optimizer__weight_decay': args.weight_decay
    }

    scoring = {"MSE": make_scorer(mean_squared_error)}

    grid_net = GridSearchCV(net, grid_params, cv=3, scoring=scoring, verbose=1, refit="MSE")

    # Training
    start_time = time.time()
    print(X_train_fb.to_numpy(), type(Y_train[0]))
    Y_train = Y_train.reshape(-1, 1)

    result = grid_net.fit(1.0*X_train_fb.to_numpy().astype(np.float32), Y_train.astype(np.float32))
    end_time = time.time()
    print("End Grid search: BEST parameters: ", grid_net.best_params_, "\n")
    print("End Grid search: BEST score: ", grid_net.best_score_, "\n")
    print("End Grid search: cv_results_ : ", grid_net.cv_results_, "\n")


    # Testing
    pred = grid_net.predict(X_test_fb.to_numpy().astype(np.float32))
    #pred_proba = grid_net.predict_proba(X_test_fb.to_numpy().astype(np.float32))[::, 1]

    # save your model or results
    joblib.dump(grid_net, path_save_model + 'model.pkl')
    # load your model for further usage
    load_best_gs = joblib.load(path_save_model + 'model.pkl')



    # Testing
    pred = load_best_gs.predict(X_test_fb.to_numpy().astype(np.float32))
    #pred_proba = load_best_gs.predict_proba(X_test_fb.to_numpy().astype(np.float32))[::, 1]

    print()
    print(f'RMSE error = {mean_squared_error(Y_test, pred)}')



    from aix360.algorithms.rbm import GLRMExplainer, LinearRuleRegression
    linear_model = LinearRuleRegression()
    # print(X_train_fb.shape)X_test_fb
    linear_model.fit(X_train_fb, Y_train)
    Y_pred = linear_model.predict(X_test_fb)
    from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, max_error
    print(f'R2 Score = {r2_score(Y_test, Y_pred)}')
    print(f'Explained Variance = {explained_variance_score(Y_test, Y_pred)}')
    print(f'Mean abs. error = {mean_absolute_error(Y_test, Y_pred)}')
    print(Y_test, Y_pred)
    print(f'RMSE error = {mean_squared_error(Y_test, Y_pred)}')
    print(f'Max error = {max_error(Y_test, Y_pred)}')
    print(linear_model.explain()["rule"].to_numpy().tolist())
    Rule = len(linear_model.explain()["rule"].to_numpy().tolist())-1
    condition = np.sum([x.count("AND") for x in linear_model.explain()["rule"].to_numpy().tolist()]) + Rule
    print(Rule, condition)
    rules.append(Rule)
    conds.append(condition)
    aucs.append(mean_squared_error(Y_test, Y_pred))
print(np.mean(aucs))
print(np.std(aucs))

print(np.mean(conds))
print(np.std(conds))

print(np.mean(rules))
print(np.std(rules))