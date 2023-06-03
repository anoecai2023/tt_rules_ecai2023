import argparse
import ast
import pickle
import random
import os
import json

from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import torch

from torch.utils.data import TensorDataset

from config.config import Config, transform_input_filters, transform_input_filters2, transform_input_optim, \
    transform_input_thr, transform_input_str, transform_input_bool
from config.config import two_args_str_int
from utils import read_csv, DBEncoder

# Load configuration
config_general = Config()
nclass = 2
if config_general.dataset == "adult":
    config = Config(path="config/adult/")
elif config_general.dataset == "compas":
    config = Config(path="config/compas/")
elif config_general.dataset == "heloc":
    config = Config(path="config/heloc/")
elif config_general.dataset == "diabetes":
    config = Config(path="config/diabetes/")
    nclass = 3
elif config_general.dataset == "house":
    config = Config(path="config/house/")
    nclass = 1
else:
    raise "Dataset not recognized"

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=config_general.dataset)
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
parser.add_argument("--dropout_value_class", default=config.ttnet_grid_param.dropout_value_class,
                    type=transform_input_thr)
parser.add_argument("--dropout_value_cnn", default=config.ttnet_grid_param.dropout_value_cnn, type=transform_input_thr)
parser.add_argument("--family_LTT", default=config.ttnet_grid_param.family_LTT, type=transform_input_str)
parser.add_argument("--poly_flag", default=config.ttnet_grid_param.poly_flag, type=transform_input_bool)
parser.add_argument("--weight_decay", default=config.ttnet_grid_param.weight_decay, type=transform_input_thr)

parser.add_argument("--path_load_model", default=config.eval.path_load_model)

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


def pos_conv1D(shape, ksize=3, pad=0, stride=2):
    """Return the indices used in the 1D convolution"""

    arr = np.array(list(range(shape[0])))

    out = np.zeros((
        (arr.shape[0] - ksize + 2 * pad) // stride + 1,
        ksize)
    )
    shape = out.shape

    for i in range(0, shape[0]):
        sub = arr[i * stride:i * stride + ksize]
        v = sub.flatten()
        out[i] = v

    return out.astype(int)

def BitsToIntAFast(bits):
    if len(bits.shape) == 3:
        _, m, n = bits.shape  # number of columns is needed, not bits.size
    else:
        m, n = bits.shape
    a = 2 ** np.arange(n)[::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bits @ a  # this matmult is the key line of code


path_save_model = args.path_load_model
print("Loading model from: ", path_save_model)

with open(os.path.join(path_save_model, 'commandline_args.txt'), 'r') as jfile:
    cmd_args = json.load(jfile)
with open(os.path.join(path_save_model, 'thresholds_rules.json'), 'r') as jfile:
    thrs = json.load(jfile)
thrs["thresholds"] = ast.literal_eval(thrs["thresholds"])
thrs["repeat"] = int(thrs["repeat"])
thrs["continous_features"] = int(thrs["continous_features"])
thrs["poly_parameters"] = ast.literal_eval(thrs["poly_parameters"])

##################################################################################################################
# Load data & transform

if config_general.dataset == "heloc":
    from aix360.datasets.heloc_dataset import HELOCDataset, nan_preprocessing

    data = HELOCDataset(custom_preprocessing=nan_preprocessing).data()
    y = data.pop('RiskPerformance')
    # dfTrain, dfTest, yTrain, yTest = train_test_split(data, y,  random_state=seed, stratify=y)
    dfTrain = data
    yTrain = y
    dfTrain.head().transpose()
    from aix360.algorithms.rbm import FeatureBinarizer

    fb = FeatureBinarizer(negations=True, returnOrd=True)
    dfTrain, dfTrainStd = fb.fit_transform(dfTrain)
    # dfTest, dfTestStd = fb.transform(dfTest)
    dfTrain['ExternalRiskEstimate'].head()
    X_train, y_train = dfTrain.to_numpy(), yTrain.to_numpy()
    # X_test, y_test = dfTest.to_numpy(), yTest.to_numpy()
elif config_general.dataset == "house":
    from aix360.algorithms.rbm import FeatureBinarizer

    from sklearn.datasets import fetch_california_housing
    import pandas as pd
    import numpy as np

    housing = fetch_california_housing()

    X_train, y_train = housing["data"], housing["target"]

    fb = FeatureBinarizer(negations=True)
    X_train_fb = fb.fit_transform(pd.DataFrame(X_train)).to_numpy().astype(np.float32)
    X_train = X_train_fb
    y_train = y_train.reshape(-1, 1)
    y_train = y_train.astype(np.float32)


else:
    X_df, y_df, f_df, label_pos = read_csv(args.data_path + config_general.dataset + ".data",
                                           args.data_path + config_general.dataset + ".info",
                                           shuffle=True)
    db_enc = DBEncoder(f_df)
    db_enc.fit(X_df, y_df)
    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
    y = np.argmax(y, axis=1)
    if args.completeness == "complete":
        X = X
    elif args.completeness == "discrete":
        X = X[:, :db_enc.discrete_flen]
    elif args.completeness == "continuous":
        X = X[:, db_enc.discrete_flen:]
    else:
        raise 'Invalid completeness argument. Input "complete"/"discrete"/"continuous"'

    X_train = X
    y_train = y
    # print(y)
    # ok

gen_data = X_train
labels = y_train

##################################################################################################################
# load net


dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=False, num_workers=2)



truth_tables = np.load(os.path.join(path_save_model,"TruthTable.npy"))
print(truth_tables)

dnf_files = os.listdir(path_save_model)
dnf_files = [f for f in dnf_files if 'DNF' in f]


cpt_gates = 0
for f in dnf_files:
    with open(os.path.join(path_save_model, f), 'r') as file:
        dnf = file.read()
    cpt_gates += str(dnf).count("&")
    cpt_gates += str(dnf).count("|")
print('Number of gates for each filter: ', cpt_gates)


losstot = 0

cpt = 0
tot = 0
running_vacc = 0

lossmse = torch.nn.MSELoss()

y_label = None
pred_proba = None

# shape = (X_train[0,:].shape[0]+(thrs["repeat"]-1)*(cmd_args['features_size']+thrs["continous_features"]),)#X_train[0,:].shape
shape = (cmd_args['features_size']+thrs["continous_features"]+(thrs["repeat"])*(abs(thrs["continous_features"])),)
if cmd_args['features_size']+thrs["continous_features"] == 0:
    shape = X_train[0,:].shape



ksize, stride, pad = cmd_args['kernel_size'][0], cmd_args['stride'][0], cmd_args['padding'][0]
indexes = pos_conv1D(shape, ksize, pad, stride)

W = np.load(os.path.join(path_save_model, 'W.npy'))
B = np.load(os.path.join(path_save_model, 'B.npy'))

if thrs["poly_act"]:

    w1 = np.load(os.path.join(path_save_model, 'w1.npy'))
    b1 = np.load(os.path.join(path_save_model, 'b1.npy'))
    w2 = np.load(os.path.join(path_save_model, 'w2.npy'))
    b2 = np.load(os.path.join(path_save_model, 'b2.npy'))
    alpha, beta, gamma = thrs["poly_parameters"]

def softmax(x):
    # Subtracting the maximum value for numerical stability
    x -= np.max(x, axis=-1, keepdims=True)

    # Exponentiate the values
    exp_x = np.exp(x)

    # Compute the softmax probabilities
    softmax_probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    return softmax_probs


with torch.no_grad():
    for i, vdata in tqdm(enumerate(train_loader)):
        vinputs, vlabels = vdata
        if y_label is None:
            y_label = vlabels.numpy().astype(int)
        else:
            y_label = np.concatenate((y_label, vlabels.numpy().astype(int)), axis=0)

        inpt = vinputs.detach().cpu().numpy().copy()

        if cmd_args['features_size']+thrs["continous_features"] != 0:
            cont_features = inpt[:, thrs["continous_features"]:]
            inpt = inpt[:, :thrs["continous_features"]]

            for r in range(thrs["repeat"]):

                scale = np.load(os.path.join(path_save_model, f'preprocess_{r}_BN_scale.npy'))
                bias = np.load(os.path.join(path_save_model, f'preprocess_{r}_BN_bias.npy'))
                np_out = scale*cont_features+bias
                inpt = np.concatenate((inpt, (np_out > 0).astype(int)), axis=1)

        all_filters = []
        for filter_idx in range(truth_tables.shape[1]):
            block = truth_tables[:,filter_idx]
            filter_input = inpt.squeeze()[:,indexes]
            filter_input = BitsToIntAFast(filter_input).astype(int)
            all_filters.append(np.expand_dims(block[filter_input], axis=-1))
            # print(block[filter_input][:,:])
            # print(block[filter_input].shape)
            # ok

        in_classifier = np.concatenate(all_filters, axis=-1).transpose(0,2,1)
        out2 = in_classifier.reshape((in_classifier.shape[0], W.shape[1]))
        print(out2[:, 0])
        print(out2.shape)
        ok
        if thrs['poly_act']:
            out2 = out2@w1.transpose()+b1
            out2 = alpha + beta * out2 + gamma * out2 ** 2
            out2 = out2@w2.transpose() + b2
            out2 = softmax(out2)
        else:
            out2 = out2@W.transpose() + B


        if pred_proba is None:
            pred_proba = out2[:, -1]
        else:
            pred_proba = np.concatenate((pred_proba, out2[:, -1]), axis=0)

        if args.dataset in ['heloc', 'house']:
            losstot += lossmse(vlabels, torch.Tensor(out2)[:, -1])

        vpred = np.argmax(out2, axis=1)
        running_vacc += np.sum(vpred == vlabels.numpy().astype(int))

        cpt += 1
        tot += vlabels.shape[0]

print('ACC TRAIN {} valid {}'.format(0, running_vacc / tot))
print('RMSE: ', losstot)
print("ROC (%)", 100 * roc_auc_score(y_label, pred_proba))


