import time
import pandas as pd
import os
import json
from mapie.metrics import regression_coverage_score
import warnings
import itertools
import sys


def get_folder_names(path):
    """
    Returns a list of folder names in the given path, excluding system folders.

    Parameters:
    path (str): The directory path.

    Returns:
    list: A list of folder names.
    """
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
    clean (bool): Whether to remove constant features and keep a root feature.

    Returns:
    pd.DataFrame, list: The DataFrame with added interaction features, and the list of feature names.
    """
    df_interacted = df.copy()  # Make a copy of the DataFrame
    feature_list = feature_columns.copy()  # Initialize a list of feature names

    # Remove 'root' feature and non-constant columns if 'clean' is True
    if clean:
        feature_list.remove('root')
        feature_list = [col for col in feature_list if df[col].nunique() > 1]

    # Generate all combinations of the specified interaction level
    interactions = list(itertools.combinations(feature_list, interaction_level))

    # Add interaction features to the DataFrame
    feature_names = ['root'] + feature_list if clean else feature_list.copy()
    for interaction in interactions:
        new_col_name = '_+_'.join(interaction)  # Create new column name
        feature_names.append(new_col_name)  # Add to feature names list
        df_interacted[new_col_name] = df_interacted[interaction[0]]  # Initialize interaction term
        for feature in interaction[1:]:
            df_interacted[new_col_name] *= df_interacted[feature]  # Multiply features to create interaction term

    return df_interacted.copy(), feature_names


def cond_cov(df, features, feature=None):
    """
    Calculates conditional coverage for a feature or a list of features.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    features (list): A list of feature names.
    feature (str): A specific feature to calculate coverage for (optional).

    Returns:
    float: The minimum coverage score.
    """
    if not feature:
        return min([cond_cov(df, features, f) for f in features])
    else:
        cond_df = df.loc[df[feature] == 1]
        warnings.filterwarnings("ignore")
        coverage = regression_coverage_score(
            cond_df["measured"].tolist(), cond_df["interval_min"].tolist(), cond_df["interval_max"].tolist()
        )
        return coverage


def cond_int(df, features, feature, percentage=False):
    """
    Calculates conditional intervals for a specific feature.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    features (list): A list of feature names.
    feature (str): The specific feature to calculate intervals for.
    percentage (bool): Whether to return intervals as a percentage of the predicted value.

    Returns:
    list: A list of interval values.
    """
    cond_df = df.loc[df[feature] == 1]
    warnings.filterwarnings("ignore")
    min_list = cond_df["interval_min"].tolist()
    max_list = cond_df["interval_max"].tolist()
    value_list = cond_df["predicted"].tolist()
    intervals = [
        abs(max_list[i] - min_list[i]) / value_list[i] if percentage else abs(max_list[i] - min_list[i])
        for i in range(len(min_list))
    ]
    return intervals


if __name__ == "__main__":
    # Start timer
    start = time.perf_counter_ns()

    # Read command line arguments or use default path
    args = sys.argv[1:]
    exp_path = args[0] if len(args) > 0 else "/mnt/e/Experiment_x264_energy"
    json_name = "config.json"
    json_path = os.path.join(exp_path, json_name)

    # Load configuration from JSON file
    with open(json_path) as json_file:
        config = json.load(json_file)
    methods = config["METHODS"]
    alpha = config["alpha"]
    nfp = config["nfp"]

    # Get list of sampling strategies
    sampling_strategies = get_folder_names(exp_path)
    if "old" in sampling_strategies:
        sampling_strategies.remove("old")

    # Initialize columns for results
    col_method, col_samp, col_run, col_cov, col_feature, col_inter = [], [], [], [], [], []

    # Iterate over methods, sampling strategies, and runs
    for method in methods:
        for strategy in sampling_strategies:
            runs = get_folder_names(os.path.join(exp_path, strategy))
            for run in runs:
                # Load validation and prediction data
                df_val = pd.read_csv(os.path.join(exp_path, strategy, run, "test.csv"))
                df_pred = pd.read_csv(os.path.join(exp_path, strategy, run, f"{method[0]}_{method[1]}_{method[2]}_pred.csv"))
                df_comb = pd.concat([df_val, df_pred], axis=1)
                df_comb.sort_values(by=['measured'], inplace=True)

                # Get feature columns excluding specific columns
                features = df_comb.columns.tolist()
                not_features = ['predicted', 'interval_min', 'interval_max', 'measured', nfp]
                features = [f for f in features if f not in not_features]

                # Add feature interactions
                df_comb, features = add_feature_interactions(df_comb, features, 2, clean=True)

                # Calculate conditional coverage and intervals for each feature
                for feature in features:
                    cov = cond_cov(df_comb, features, feature)
                    inter = cond_int(df_comb, features, feature, percentage=True)
                    col_method.append(f"{method[0]}_{method[1]}_{method[2]}")
                    col_samp.append(strategy)
                    col_run.append(run)
                    col_cov.append(cov)
                    col_feature.append(feature)
                    col_inter.append(inter)

    # Create DataFrame for conditional coverage and interactions
    cov_dict_inter = {
        'Methode': col_method, 'Strategie': col_samp, 'Run': col_run, "Feature": col_feature,
        "Cov": col_cov, "Inter": col_inter
    }
    df_cov_inter = pd.DataFrame(cov_dict_inter).dropna()
    df_cov_inter.to_csv(os.path.join(exp_path, 'cond_cov_interactions.csv'), sep=';', index=False)

    # Filter features for conditional coverage calculation
    features = df_cov_inter.Feature.unique().tolist()
    features_select = [x for x in features if "_+_" not in x]

    # Initialize columns for final conditional coverage results
    col_method, col_samp, col_run, col_cov = [], [], [], []

    # Calculate the minimum coverage for selected features
    for method in df_cov_inter.Methode.unique():
        for strategy in df_cov_inter.Strategie.unique():
            for run in df_cov_inter.Run.unique():
                df = df_cov_inter.loc[
                    (df_cov_inter['Methode'] == method) & (df_cov_inter['Strategie'] == strategy) &
                    (df_cov_inter['Run'] == run) & (df_cov_inter['Feature'].isin(features_select))
                ]
                col_cov.append(df.Cov.min())
                col_run.append(run)
                col_samp.append(strategy)
                col_method.append(method)

    # Create final DataFrame and save to CSV
    cov_dict = {'Methode': col_method, 'Strategie': col_samp, 'Run': col_run, "Cov": col_cov}
    df_cond_cov = pd.DataFrame(cov_dict)
    df_cond_cov.to_csv(os.path.join(exp_path, 'cond_cov.csv'), sep=';', index=False)

    # End timer and print elapsed time
    end = time.perf_counter_ns()
    print(f"Used Time: {(end - start) * 0.000000001} seconds")
