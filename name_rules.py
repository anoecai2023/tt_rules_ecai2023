import argparse
import ast
import copy
import itertools
import pickle
import random
import os
import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sympy import to_cnf, to_dnf, symbols, parse_expr
from sympy.logic.boolalg import truth_table
from tqdm import tqdm
import numpy as np
import torch

from torch.utils.data import TensorDataset

from config.config import Config, transform_input_filters, transform_input_filters2, transform_input_optim, \
    transform_input_thr, transform_input_str, transform_input_bool
from config.config import two_args_str_int
from utils import read_csv, DBEncoder, get_exp_with_y, get_expresion_methode1, compute_corr_rules

# Load configuration
config_general = Config()
nclass = 2
if config_general.dataset == "adult":
    config = Config(path="config/adult/")
    continuous_dict = {1: "age_ST_", 3: "poids_binaire_raciste_ST_",
                       5: "years_of_education_ST_", 11: "capital_gain_ST_",
                       12: "capital_loss_ST_", 13: "hours_per_week_ST_"}
elif config_general.dataset == "compas":
    config = Config(path="config/compas/")
    continuous_dict = {1: 'age_ST_', 4: 'diff_custody_ST_', 5: 'diff_jail_ST_', 6: 'priors_count_ST_'}
elif config_general.dataset == "heloc":
    config = Config(path="config/heloc/")
    continuous_dict = {}
elif config_general.dataset == "diabetes":
    config = Config(path="config/diabetes/")
    nclass = 3
    continuous_dict = {7: 'time_in_hospital_ST_', 9: 'num_lab_procedures_ST_', 10: 'num_procedures_ST_',
                       11: 'num_medications_ST_', 15: 'number_diagnoses_ST_'}
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
var_names = db_enc.X_fname
cont_feat_names = [continuous_dict[int(key)] for key in var_names[thrs["continous_features"]:]]
# print(y)
# ok

gen_data = X_train
labels = y_train

##################################################################################################################
# load net


dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=False, num_workers=2)

truth_tables = np.load(os.path.join(path_save_model, "TruthTable.npy"))
print(truth_tables)
print(truth_tables.shape)
n_filters = truth_tables.shape[-1]
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
shape = (cmd_args['features_size'] + thrs["continous_features"] + (thrs["repeat"]) * (abs(thrs["continous_features"])),)
if cmd_args['features_size'] + thrs["continous_features"] == 0:
    shape = X_train[0, :].shape

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



# cont_feat_names = [continuous_dict[int(key)] for key in var_names[thrs["continous_features"]:]]
var_names[thrs["continous_features"]:] = cont_feat_names


human_thresh = []
if cmd_args['features_size'] + thrs["continous_features"] != 0:
    mean, std = db_enc.mean, db_enc.std
    for r in range(thrs["repeat"]):
        scale = np.load(os.path.join(path_save_model, f'preprocess_{r}_BN_scale.npy'))
        bias = np.load(os.path.join(path_save_model, f'preprocess_{r}_BN_bias.npy'))
        thresholds = mean - std * bias / scale
        human_thresh.append(thresholds)
        cont_feat_names_thr = [cont_feat_names[i] + str(round(thresholds.values[i],2)) for i in range(len(cont_feat_names))]
        var_names = var_names + cont_feat_names_thr


# if cmd_args['features_size'] + thrs["continous_features"] != 0:
#     var_names = var_names + (thrs['repeat'] - 1) * cont_feat_names

fname_encode = {i: val for i, val in enumerate(var_names)}




vinputs = np.array(var_names)




def get_human_expr(dnf, patch):
    e = []

    for and_conds in dnf:
        cond = []
        for and_idx in and_conds:
            idx = int(and_idx.split('_')[-1])
            if and_idx.startswith('~'):
                sign = '~'
            else:
                sign = ''
            cond.append(sign + patch[idx])
        cond = ' & '.join(cond)
        e.append(cond)
    e = ' | '.join(e)
    return e


def get_human_expr_from_block(dnf, patches):
    expr = []

    for patch in patches:
        expr.append(get_human_expr(dnf, patch))
    return expr


all_express = []
for filter_idx in range(truth_tables.shape[1]):
    dnf_filter = [f for f in dnf_files if f'filter_{filter_idx}' in f][0]
    with open(os.path.join(path_save_model, dnf_filter), 'r') as f:
        dnf_filter = f.read()

    dnf_filter = dnf_filter.replace('(', '').replace(')', '').replace(' ', '').split('|')
    for i in range(len(dnf_filter)):
        dnf_filter[i] = dnf_filter[i].split('&')

    filter_input = vinputs[indexes]

    all_express.append(get_human_expr_from_block(dnf_filter, filter_input))

all_express = np.array(all_express)

count = 0
for r in all_express.flatten().tolist():
    count += r.count('|')
    count += r.count('&')

print("Number of gates: ", count)

rules = {}
for i, class_idx in enumerate(W):
    rules_idx = np.where(class_idx != 0)[0]
    rules[i] = []
    expres = all_express.flatten().tolist()
    for idx in rules_idx:
        rules[i].append((expres[idx], class_idx[idx]))


if not os.path.exists(os.path.join(path_save_model, 'human_expressions')):
    os.mkdir(os.path.join(path_save_model, 'human_expressions'))

with open(os.path.join(path_save_model, 'human_expressions', 'without_dontcare.txt'), 'w') as file:
    for cls, rules in rules.items():
        file.write(f'Class {cls}:\n\n')
        file.write('Weight\tRule\n')
        for r, w in rules:
            file.write(f'{w}\t{r}')
            file.write('\n')
        file.write('\n\n\n')


def generate_binary_strings(n):
    # Generate all numbers from 0 to 2^n - 1
    numbers = np.arange(2 ** n).astype(int)

    # Convert numbers to binary strings of length n
    binary_strings = [np.binary_repr(num, width=n) for num in numbers]

    return binary_strings


def human_donctcares_on_expr(save_path, block_occurence, fname_encode, input_var, filter_idx, W_LR, shapeici_out,
                             xy_pixel, all_expr):
    # print(input_var)
    variables = {i: fname_encode[input_var[i]].replace('?', 'NoInfo') for i in
                 range(len(input_var))}

    tt = np.load(os.path.join(save_path, 'truth_table.npy'))
    tt = tt[:, filter_idx]
    nbits = int(np.log2(tt.shape[0]))

    binary_numbers = generate_binary_strings(nbits)

    df = pd.DataFrame([list(string) for string in binary_numbers], columns=[f'{i}' for i in range(nbits)])
    df[f'Filter_{filter_idx}_Value_1'] = tt.astype(bool)
    df[f'Filter_{filter_idx}_dontcares_1'] = tt.astype(bool) & False
    tt = df.copy()
    path_tt_dc = os.path.join(save_path, 'human_expressions','Truth_Table_block' +
                              str(block_occurence) + '_filter_' + str(filter_idx) + "_" + str(xy_pixel) + '.csv')

    names_features = list(variables.values())
    num = []
    separate_f = []
    separate_f_var = []
    for ix, x in enumerate(names_features):
        umici = x.split("_")[0]
        if umici not in num:
            num.append(umici)
            separate_f.append([x])
            separate_f_var.append(["x_" + str(ix)])
        else:
            for indexpos in range(len(num)):
                if num[indexpos] == umici:
                    separate_f[indexpos].append(x)
                    separate_f_var[indexpos].append("x_" + str(ix))
            # else:
        # print(separate_f_var)
        # print(separate_f, num, umici, umici not in num)
    # we compute the expressions of expressions with less than  ksize clauses
    # print(separate_f, num, separate_f_var)
    if len(separate_f) < ksize:
        # print("ok1")
        var_sum = sum([len(f) for f in separate_f])
        assert var_sum == len(input_var)
        all_values = []
        for j in range(len(separate_f)):
            bin_values = [[0] * len(separate_f[j]) for _ in range(len(separate_f[j]) + 1)]
            # print("bin_values 0 ", bin_values)
            for i in range(len(separate_f[j])):
                bin_values[i + 1][i] = 1
            # print("bin_values 1 ",bin_values)
            all_values.append(bin_values)
        if len(all_values) == 1:
            table = all_values[0]
        else:
            table = list(itertools.product(all_values[0], all_values[1]))
            buff = []
            for ii, item in enumerate(table):
                buff.append(item[0] + item[1])
            del table
            table = copy.copy(buff)

        # print("all_values", all_values)
        # print("tbale 0", table)

        # print(ok)
        if len(all_values) > 2:

            for i in range(2, len(all_values)):
                # print("i, t, 0 ", i, table)
                table = list(itertools.product(table, all_values[i]))
                # print("i, t, 1 ", i, table)
                buff = []
                for ii, item in enumerate(table):
                    buff.append(item[0] + item[1])
                del table
                table = copy.copy(buff)
        # print("table 1", table)
        for tablevalue in table:
            # cpttabassert = 0
            # for tablevaluex in tablevalue:
            # cpttabassert+=len(tablevaluex)
            # print(cpttabassert,  args.kernel_size_per_block[block_occurence])
            assert len(tablevalue) == nbits
        # print("all_values", all_values)

        p = 1

        # print(table)
        # print(ok)

        for val in separate_f:
            p *= (len(val) + 1)

        assert len(table) == p

        small_tt = []
        small_tt2 = []
        for t in table:

            if type(t) is tuple:
                concat = np.concatenate(t)
            else:
                concat = t
            # print(concat)

            small_tt2.append(int("".join(str(i) for i in concat), 2))
            # if args.random_permut:
            #     separate_f_var2 = []
            #     for sfvar in separate_f_var:
            #         for sfvar2 in sfvar:
            #             separate_f_var2.append(sfvar2)
            #     concat_new = [0] * len(concat)
            #     for index_concat in range(len(concat)):
            #         valuecat = concat[index_concat]
            #         value_position = int(separate_f_var2[index_concat].replace("x_", ""))
            #         concat_new[value_position] = valuecat
            #     # print(concat_new)
            #     del concat
            #     concat = concat_new

            small_tt.append(int("".join(str(i) for i in concat), 2))

        # print(small_tt2)
        # print(small_tt)
        # print(ok)

        tt_dc = tt.copy()

        for idx in tt.index.values.tolist():
            # print(idx)
            # print(tt.iloc[idx])
            # print(tt.iloc[idx].drop(['Filter_0_Value_1',  'Filter_0_dontcares_1']).to_numpy())

            to_int = tt.iloc[idx].drop([f'Filter_{filter_idx}_Value_1',
                                        f'Filter_{filter_idx}_dontcares_1']).to_numpy()
            value = int("".join(str(i) for i in to_int), 2)
            # if args.dc_logic:
            if value not in small_tt:
                tt_dc.at[idx, f'Filter_{filter_idx}_dontcares_1'] = True
        # print(tt0)
        for df2 in [tt_dc]:
            # print(df2)
            answer = df2[f"Filter_{filter_idx}_Value_1"].to_numpy()
            dc = df2[f"Filter_{filter_idx}_dontcares_1"].to_numpy()
            condtion_filter = df2.index.values[answer].tolist()
            dc_filter = df2.index.values[dc].tolist()
            # condtion_filter_cnf = df2["index"].values[answer_cnf].tolist()
            # print(condtion_filter, dc_filter)
            df2.to_csv(path_tt_dc)
            exp_DNF, exp_CNF = get_expresion_methode1(nbits, condtion_filter,
                                                      dc_filter=dc_filter)
            # exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
            exp_CNF3 = get_exp_with_y(exp_DNF, exp_CNF)
            # print(exp_DNF, variables)

            # module.save_cnf_dnf(1.0, exp_CNF3, exp_DNF, exp_CNF, xypixel=xy_pixel)
            dnf = str(exp_DNF).replace(' ','').replace(')','').replace('(','').split('|')
            dnf = [d.split('&') for d in dnf]

            readable_dnf = get_human_expr(dnf, names_features)
            # readable_dnf = readable_expr(exp_DNF, variables, args)
            # print(readable_dnf)
            # print(f"Position : {xy_pixel}")
            # print()
            # print("Weight : ")
            list_W = []
            for idx in range(W_LR.shape[0]):
                # print(W_LR[idx][filteroccurence * shapeici_out + xy_pixel].item())
                # print(W_LR[idx_negative][filteroccurence * shapeici_out + xy_pixel].item())
                # print()
                list_W.append(W_LR[idx][filter_idx * shapeici_out + xy_pixel].item())
            # all_expr.append(
            #    (readable_dnf,
            #     list_W))

    else:
        tt.to_csv(path_tt_dc)
        # module.save_cnf_dnf(1.0, exp_CNF3, exp_DNF, exp_CNF, xypixel=xy_pixel)
        variables = {i: fname_encode[input_var[i]].replace('?', 'NoInfo') for i in range(len(input_var))}

        answer = df[f"Filter_{filter_idx}_Value_1"].to_numpy()
        dc = df[f"Filter_{filter_idx}_dontcares_1"].to_numpy()
        condtion_filter = df.index.values[answer].tolist()
        dc_filter = df.index.values[dc].tolist()

        exp_DNF, exp_CNF = get_expresion_methode1(nbits, condtion_filter,
                                                  dc_filter=dc_filter)
        # exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
        exp_CNF3 = get_exp_with_y(exp_DNF, exp_CNF)
        dnf = str(exp_DNF).replace(' ', '').replace(')', '').replace('(', '').split('|')
        dnf = [d.split('&') for d in dnf]

        readable_dnf = get_human_expr(dnf, names_features)

        # print(exp_DNF, variables)
        # readable_dnf = readable_expr(exp_DNF, variables, args)
        # print(readable_dnf)
        # print(f"Position : {xy_pixel}")
        # print()
        # print("Weight : ")
        list_W = []
        for idx in range(W_LR.shape[0]):
            # print(W_LR[idx][filteroccurence * shapeici_out + xy_pixel].item())
            # print(W_LR[idx_negative][filteroccurence * shapeici_out + xy_pixel].item())
            # print()
            list_W.append(W_LR[idx][filter_idx * shapeici_out + xy_pixel].item())
        # all_expr.append(
        #    (readable_dnf,
        #     list_W))
        # print()

    all_expr.append(
        (readable_dnf,
         list_W))

    return exp_CNF3, exp_DNF, exp_CNF, all_expr, readable_dnf

neurons_per_filter = W.shape[1]//(n_filters)


print('Injecting dont care terms ...')

reverse_dict = {value: key for key, value in fname_encode.items()}

all_expr = []
for filter_idx in tqdm(range(n_filters)):
    filter_input = vinputs[indexes]
    for xy_pixel in range(neurons_per_filter):
        patch = filter_input[xy_pixel]

        patch = [reverse_dict[val] for val in patch]
        cnf3, dnf, cnf, all_expr, human_dnf = human_donctcares_on_expr(path_save_model, 0, fname_encode, patch, filter_idx,
                                                           W, neurons_per_filter, xy_pixel, all_expr)


count = 0
for r,w in all_expr:
    count += r.count('|')
    count += r.count('&')

print("Number of gates: ", count)

rules = {}
for i, class_idx in enumerate(W):
    rules_idx = np.where(class_idx != 0)[0]
    rules[i] = []
    expres = all_expr.copy()
    for idx in rules_idx:
        rules[i].append((idx, expres[idx][0], class_idx[idx]))


with open(os.path.join(path_save_model, 'human_expressions', 'with_dontcare.txt'), 'w') as file:
    for cls, rules in rules.items():
        file.write(f'Class {cls}:\n\n')
        file.write('Position\tWeight\tRule\n')
        for i, r, w in rules:
            file.write(f'{i}\t{w}\t{r}')
            file.write('\n')
        file.write('\n\n\n')


corr_matrix = np.zeros((n_filters,n_filters))
for i in range(n_filters):
    for j in range(n_filters):
        tt1 = truth_tables[:, i]
        tt2 = truth_tables[:, j]
        corr = np.sum(tt1 == tt2) / tt1.shape[0]
        corr_matrix[i, j] = corr


ax = sns.heatmap(corr_matrix, annot=True)
ax.set(title=f"Correlation of filters",
       xlabel="Filter",
       ylabel="Filter", )
plt.savefig(os.path.join(path_save_model, f'correlation_filters.png'))


tt_files = os.listdir(os.path.join(path_save_model, 'human_expressions'))
tt_files = sorted([f for f in tt_files if 'Truth_Table' in f],
                  key=lambda x: (int(x.split('_')[4]), int(x.split('_')[5].replace('.csv', ''))))

tt_rules = {i:{} for i in range(n_filters)}
for f in tt_files:
    filter_idx = int(f.split('_')[4])
    position = int(f.split('_')[5].replace('.csv', ''))
    path_save_tt = os.path.join(path_save_model, 'human_expressions', f)
    tt = pd.read_csv(path_save_tt)
    tt = tt.drop(['Unnamed: 0'], axis=1)
    tt = tt[f'Filter_{filter_idx}_Value_1'].values.astype(int)
    tt_rules[filter_idx][position] = tt


if args.dataset in args.dataset in ["adult", "diabetes"]:
    thr_corr = 0.9
else:
    thr_corr = 0.8

pos_corr, neg_corr = compute_corr_rules(tt_rules, n_filters, neurons_per_filter, os.path.join(path_save_model,'human_expressions'), thr_corr)

print(pos_corr)
print(neg_corr)