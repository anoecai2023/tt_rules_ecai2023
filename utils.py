import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sympy import symbols, SOPform, POSform, to_cnf
from tqdm import tqdm
from scipy.stats import skewnorm


def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
        print(f_list)
    return f_list[:-1], int(f_list[-1][-1])


def read_csv(data_path, info_path, shuffle=False):
    D = pd.read_csv(data_path, header=None)
    if shuffle:
        D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    f_list, label_pos = read_info(info_path)
    f_df = pd.DataFrame(f_list)
    D.columns = f_df.iloc[:, 0]
    y_df = D.iloc[:, [label_pos]]
    X_df = D.drop(D.columns[label_pos], axis=1)
    f_df = f_df.drop(f_df.index[label_pos])
    return X_df, y_df, f_df, label_pos


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        self.f_df = f_df
        self.discrete = discrete
        self.y_one_hot = y_one_hot
        self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
        self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X_fname = None
        self.y_fname = None
        self.discrete_flen = None
        self.continuous_flen = None
        self.mean = None
        self.std = None

    def split_data(self, X_df):
        discrete_data = X_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(np.float64)
        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        self.label_enc.fit(y_df)
        self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns)) if self.y_one_hot else y_df.columns

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
        if not discrete_data.empty:
            # One-hot encoding
            self.feature_enc.fit(discrete_data)
            feature_names = discrete_data.columns
            self.X_fname = list(self.feature_enc.get_feature_names_out(feature_names))
            self.discrete_flen = len(self.X_fname)
            if not self.discrete:
                self.X_fname.extend(continuous_data.columns)
        else:
            self.X_fname = continuous_data.columns
            self.discrete_flen = 0
        self.continuous_flen = continuous_data.shape[1]

    def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        # Encode string value to int index.
        y = self.label_enc.transform(y_df.values.reshape(-1, 1))
        if self.y_one_hot:
            y = y.toarray()

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                           columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    self.mean = continuous_data.mean()
                    self.std = continuous_data.std()
                continuous_data = (continuous_data - self.mean) / self.std
        if not discrete_data.empty:
            # One-hot encoding
            discrete_data = self.feature_enc.transform(discrete_data)
            if not self.discrete:
                X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
            else:
                X_df = pd.DataFrame(discrete_data.toarray())
        else:
            X_df = continuous_data
        return X_df.values, y


def find_splits(db_enc):
    names = db_enc.X_fname[:db_enc.discrete_flen]
    temp = names[0][:2]
    idx = [0]
    for k in range(len(names)):
        if temp != names[k][:2]:
            idx.append(k)
            temp = names[k][:2]

    return idx


# sample_generator parameters:
# data = the actual dataset to be replicated
# db_enc = the DB Encoder fitted to the data
# size = number of data entried to be generated
# cont = "0" to generate continuous part from N(0,1), "1" to generate from Uniform(Quartile 1, Quartile 3), "2" to generate a fitted skewed normal distribution
# complete = "0" to generate both discrete and continuous parts, "1" to generate just the discrete part, "2" to generate just the continuous part
def sample_generator(data, db_enc, size, cont=0, complete=0):
    # Split the dataset
    data_dc, data_c = data[:, :db_enc.discrete_flen], data[:, db_enc.discrete_flen:]
    idx = find_splits(db_enc)
    gen_data = np.zeros((size, 1))

    # ===================Generating the discrete part=======================#
    if (complete == 0 or complete == 1):
        prob = np.sum(data_dc, axis=0) / len(data_dc)

        splitted_prob = []
        for i in range(len(idx) - 1):
            splitted_prob.append(prob[idx[i]: idx[i + 1]])
        splitted_prob.append(prob[idx[-1]:])

        for k in range(len(splitted_prob)):
            cum_prob = np.cumsum(splitted_prob[k])

            temp_data = np.zeros((size, len(cum_prob)))
            num = np.random.rand(size)
            for l in range(len(num)):
                for m in range(len(cum_prob)):
                    if num[l] < cum_prob[m]:
                        temp_data[l][m] = 1
                        break

            gen_data = np.append(gen_data, temp_data, axis=1)

    # ===================Generating the continuous part======================#
    if (complete == 0 or complete == 2):
        if cont == 0:
            cont_data = np.random.normal(0, 1, size=(size, data_c.shape[1]))
            gen_data = np.append(gen_data, cont_data, axis=1)

        elif cont == 1:
            q1 = np.quantile(data_c, 0.25, axis=0)
            q3 = np.quantile(data_c, 0.75, axis=0)

            for k in range(data_c.shape[1]):
                cont_data = np.random.uniform(q1[k], q3[k], size).reshape(size, 1)
                gen_data = np.append(gen_data, cont_data, axis=1)

        elif cont == 2:
            for k in range(db_enc.continuous_flen):
                a, loc, scale = skewnorm.fit(data_c[:, k])
                cont_data = skewnorm(a, loc, scale).rvs(size).reshape(size, 1)
                gen_data = np.append(gen_data, cont_data, axis=1)

    gen_data = np.delete(gen_data, 0, axis=1)
    return gen_data


def get_exp_with_y(exp_DNFstr, exp_CNFstr):
    exp_DNFstr, exp_CNFstr = str(exp_DNFstr).resace(" ", ""), str(exp_CNFstr).replace(" ", "")
    masks = exp_DNFstr.split("|")
    clausesnv = []
    for mask in masks:
        # print(mask)
        masknv = mask.replace("&", " | ")
        masknv = masknv.replace("x", "~x")
        masknv = masknv.replace("~~", "")
        masknv = masknv.replace(")", "").replace("(", "")
        masknv = "(" + masknv + ")"
        masknv = masknv.replace("(", "(y | ")
        clausesnv.append(masknv)
        # print(masknv)
    clauses = exp_CNFstr.split("&")
    for clause in clauses:
        # print(clause)
        clausenv = clause.replace("|", " | ")
        clausenv = clausenv.replace(")", "").replace("(", "")
        clausenv = "(" + clausenv + ")"
        clausenv = clausenv.replace(")", " | ~y)")
        clausesnv.append(clausenv)
    exp_CNF3 = " & ".join(clausesnv)

    return exp_CNF3


def get_exp_with_y(exp_DNFstr, exp_CNFstr):
    exp_DNFstr, exp_CNFstr = str(exp_DNFstr).replace(" ", ""), str(exp_CNFstr).replace(" ", "")
    masks = exp_DNFstr.split("|")
    clausesnv = []
    for mask in masks:
        # print(mask)
        masknv = mask.replace("&", " | ")
        masknv = masknv.replace("x", "~x")
        masknv = masknv.replace("~~", "")
        masknv = masknv.replace(")", "").replace("(", "")
        masknv = "(" + masknv + ")"
        masknv = masknv.replace("(", "(y | ")
        clausesnv.append(masknv)
        # print(masknv)
    clauses = exp_CNFstr.split("&")
    for clause in clauses:
        # print(clause)
        clausenv = clause.replace("|", " | ")
        clausenv = clausenv.replace(")", "").replace("(", "")
        clausenv = "(" + clausenv + ")"
        clausenv = clausenv.replace(")", " | ~y)")
        clausesnv.append(clausenv)
    exp_CNF3 = " & ".join(clausesnv)

    return exp_CNF3


def get_expresion_methode1(n, condtion_filter, dc_filter=None):
    if dc_filter is not None:
        dc_filtervf = dc_filter
        condtion_filter_vf = [x for x in condtion_filter if x not in dc_filtervf]
        # print(len(condtion_filter_vf), len(dc_filtervf), len(dc_filtervf) / 2 ** self.n)
    else:
        condtion_filter_vf = condtion_filter

    if n == 4:
        w1, x1, y1, v1 = symbols('x_0, x_1, x_2, x_3')
        exp_DNF = SOPform([w1, x1, y1, v1], minterms=condtion_filter_vf, dontcares=dc_filtervf)
        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    elif n == 8:
        w1, x1, y1, v1, w2, x2, y2, v2 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7')
        exp_DNF = SOPform([w1, x1, y1, v1, w2, x2, y2, v2], minterms=condtion_filter_vf, dontcares=dc_filtervf)
        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    elif n == 9:
        w1, x1, y1, v1, w2, x2, y2, v2, w3 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8')
        exp_DNF = SOPform([w1, x1, y1, v1, w2, x2, y2, v2, w3], minterms=condtion_filter_vf, dontcares=dc_filtervf)
        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    elif n == 10:
        w1, x1, y1, v1, w2, x2, y2, v2, w3, x3 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9')
        exp_DNF = SOPform([w1, x1, y1, v1, w2, x2, y2, v2, w3, x3], minterms=condtion_filter_vf,
                          dontcares=dc_filtervf)
        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    # elif self.n == 5:
    #    w1, x1, y1, v1, w2, x2, y2, v2, w3, x3 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9')
    #    exp_DNF = SOPform([w1, x1, y1, v1, w2], minterms=condtion_filter_vf, dontcares=dc_filtervf)
    #    if self.with_contradiction:
    #        exp_CNF = POSform([w1, x1, y1, v1, w2], minterms=condtion_filter_vf, dontcares=dc_filtervf)
    #    else:
    #        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    else:

        w1, x1, y1, v1, w2, x2, y2, v2, w10, x10, y10, v10, w20, x20, y20, v20 = symbols(
            'x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7,x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15')
        list_var = [w1, x1, y1, v1, w2, x2, y2, v2, w10, x10, y10, v10, w20, x20, y20, v20]
        exp_DNF = SOPform(list_var[:n],
                          minterms=condtion_filter_vf, dontcares=dc_filtervf)
        exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
    return exp_DNF, exp_CNF


def compute_corr_rules(tt_rules, n_filters, neurons_out, save_path, thr_corr):
    positive_correlation = []
    negative_correlation = []
    for xy_pixel in tqdm(range(neurons_out)):

        corr_matrix = np.zeros((n_filters, n_filters))
        for filteroccurence in range(n_filters):
            if xy_pixel in list(tt_rules[filteroccurence].keys()):
                values3 = tt_rules[filteroccurence][xy_pixel]
            else:
                values3 = None
            for filteroccurence2 in range(n_filters):
                if xy_pixel in list(tt_rules[filteroccurence2].keys()):
                    values3bis = tt_rules[filteroccurence2][xy_pixel]
                else:
                    values3bis = None
                if values3 is not None and values3bis is not None:
                    # print(values3, values3bis)
                    corr = np.sum(1.0 * np.array(values3) == 1.0 * np.array(values3bis)) / len(values3bis)
                    # print(abs(corr - 1),  abs(corr))
                    if abs(corr - 1) > abs(corr):
                        corr_vf = corr - 1
                    else:
                        corr_vf = corr

                    # if filteroccurence == filteroccurence2:
                    #    assert corr ==1
                    flag_print = True

                else:
                    corr_vf = 0
                corr_matrix[filteroccurence, filteroccurence2] = corr_vf

                if corr_vf > thr_corr and filteroccurence != filteroccurence2:
                    if (filteroccurence2, filteroccurence, xy_pixel) not in positive_correlation:
                        positive_correlation.append((filteroccurence, filteroccurence2, xy_pixel))
                if corr_vf < -1 * thr_corr and filteroccurence != filteroccurence2:
                    if (filteroccurence2, filteroccurence, xy_pixel) not in negative_correlation:
                        negative_correlation.append((filteroccurence, filteroccurence2, xy_pixel))

    with open(os.path.join(save_path, "positive_correlation.txt"), 'w') as f:
        f.write(str(positive_correlation))

    with open(os.path.join(save_path, "negative_correlation.txt"), 'w') as f:
        f.write(str(negative_correlation))

    return positive_correlation, negative_correlation
