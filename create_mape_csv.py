import time

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
from mapie.metrics import regression_coverage_score
import warnings
import itertools
import sys
from sklearn.metrics import mean_absolute_percentage_error



def get_folder_names(path):
    folder_names = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and item not in ['$RECYCLE.BIN', 'System Volume Information']:
            folder_names.append(item)

    return folder_names


def add_feature_interactions(df, feature_columns, interaction_level, clean=True):
    """
    Adds feature interactions to a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    feature_columns (list): A list of column names to create interactions from.
    interaction_level (int): The level of interaction (e.g., 2 for pairwise interactions).

    Returns:
    pd.DataFrame: The DataFrame with added interaction features.
    """
    # Make a copy of the DataFrame to avoid modifying the original one
    df_interacted = df.copy()

    # Initialize a list to keep track of feature names including interactions
    feature_list = feature_columns.copy()

    if clean:
        feature_list.remove('root')
        for feature in feature_list:
            non_constant_columns = [col for col in feature_list if df[col].nunique() > 1]
        feature_list = non_constant_columns

    # Generate all combinations of the specified interaction level
    interactions = list(itertools.combinations(feature_list, interaction_level))

    if clean:
        feature_names = ['root'] + feature_list

    for interaction in interactions:
        # Create a new column name based on the interacting features
        new_col_name = '_+_'.join(interaction)

        # Add the new column name to the feature names list
        feature_names.append(new_col_name)

        # Multiply the features to create the interaction term
        df_interacted[new_col_name] = df_interacted[interaction[0]]
        for feature in interaction[1:]:
            df_interacted[new_col_name] *= df_interacted[feature]

    # defragment

    return df_interacted.copy(), feature_names


def cond_cov(df, features, feature=None):
    if not feature:
        list_cov = []
        for f in features:
            list_cov.append(cond_cov(df, features, f))
        return min(list_cov)
    else:
        cond_df = df.loc[df[feature] == 1]
        warnings.filterwarnings("ignore")
        coverage = regression_coverage_score(
            cond_df["measured"].tolist(), cond_df["interval_min"].tolist(), cond_df["interval_max"].tolist()
        )
        return coverage


def mean_int(df, percentage=True):
    warnings.filterwarnings("ignore")
    min_list = df["interval_min"].tolist()
    max_list = df["interval_max"].tolist()
    value_list = df["predicted"].tolist()
    if percentage:
        intervals = [abs(max_list[i] - min_list[i]) / value_list[i] for i in range(len(min_list))]
    else:
        intervals = [abs(max_list[i] - min_list[i]) for i in range(len(min_list))]
    return np.mean(intervals)


if __name__ == "__main__":
    start = time.perf_counter_ns()
    args = sys.argv[1:]
    if len(args) > 0:
        exp_path = args[0]
    else:
        # folder_path = "/home/sjahns/Experimente/Experiment_x264"
        exp_path = "/mnt/e/Experiment_x264_energy"
    json_name = "config.json"
    json_path = os.path.join(exp_path, json_name)
    with open(json_path) as json_file:
        config = json.load(json_file)
    methods = config["METHODS"]
    alpha = config["alpha"]
    nfp = config["nfp"]
    sampling_strategies = get_folder_names(exp_path)
    if "old" in sampling_strategies:
        sampling_strategies.remove("old")
    col_method = []
    col_samp = []
    col_run = []
    col_cov = []
    col_mape = []
    col_inter = []
    for m in methods:
        for s in sampling_strategies:
            runs = get_folder_names(os.path.join(exp_path, s))
            for r in runs:
                df_pred = pd.read_csv(os.path.join(exp_path, s, r, f"{m[0]}_{m[1]}_{m[2]}_pred.csv"))
                not_features = ['predicted', 'interval_min', 'interval_max', 'measured']
                cov = regression_coverage_score(df_pred["measured"].tolist(), df_pred["interval_min"].tolist(), df_pred["interval_max"].tolist())
                inter = mean_int(df_pred)
                mape = mean_absolute_percentage_error(df_pred['measured'], df_pred['predicted'])
                col_method.append(f"{m[0]}_{m[1]}_{m[2]}")
                col_samp.append(s)
                col_run.append(r)
                col_cov.append(cov)
                col_inter.append(inter)
                col_mape.append(mape)

    cov_dict_inter = {'Methode': col_method, 'Strategie': col_samp, 'Run': col_run, "Mape": col_mape,
                      "Cov": col_cov, "Inter": col_inter}
    df_cov_inter = pd.DataFrame(cov_dict_inter)
    df_cov_inter = df_cov_inter.dropna()
    df_cov_inter.to_csv(os.path.join(exp_path, 'coverage_mape.csv'), sep=';', index=False)
    end = time.perf_counter_ns()
    print(f"Used Time: {(end - start) * 0.000000001}")
