"""
Utilities for working with distributions and similar stuff
"""

import numpy as np
import pandas as pd
import os
import pymmwr
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm
from typing import Dict
from glob import glob
from sklearn.model_selection import KFold
from warnings import warn


def get_scores(results_dir, exp_name, target):
    """
    Return a dataframe of scores for given specifications
    """

    files = glob(os.path.join(results_dir, exp_name, target, "*.csv"))
    df = pd.DataFrame({
        "region": [],
        "model": [],
        "score": []
    })

    for f in files:
        scores = pd.read_csv(f)
        scores["model"] = os.path.basename(os.path.splitext(f)[0])
        df = pd.concat([df, scores], ignore_index=True)

    return df


def available_models(exp_dir):
    """
    Return name of models available as components in data_dir
    """

    return [
        model for model in
        os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, model))
    ]


def epiweek_to_season(epiweek: int) -> int:
    """
    Return first year of the season the epiweek belongs to
    """

    year, week = epiweek // 100, epiweek % 100

    if week <= 30:
        return year - 1
    else:
        return year


def epiweek_to_model_week(epiweek: int) -> int:
    """
    Convert epiweek to model week. Model week starts from 0 (epiweek 40)
    to 32 (epiweek 20). Additionally 33 means no onset.
    """

    if np.isnan(epiweek):
        return 33
    else:
        epiweek = int(epiweek)
        year, week = epiweek // 100, epiweek % 100
        if week in range(40, 54):
            return week - 40
        elif week in range(1, 21):
            season = epiweek_to_season(epiweek)
            return week + pymmwr.mmwr_weeks_in_year(season) - 40
        elif week in range(21, 40):
            warn("Model week is outside the planned range")
            season = epiweek_to_season(epiweek)
            return week + pymmwr.mmwr_weeks_in_year(season) - 40
        else:
            raise Exception(f"Unknown epiweek {epiweek} provided")


def encode_epiweeks_sin(epiweeks):
    """
    Sinusoidal encoding of epiweek
    """

    years, weeks = epiweeks // 100, epiweeks % 100
    ns = np.array([pymmwr.mmwr_weeks_in_year(y) for y in years])
    rads = 2 * np.pi * weeks / ns
    return np.array([np.sin(rads), np.cos(rads)]).T


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def cv_train_kfold(gen_model, train_model, X, y, k=10):
    """
    Train the model using KFold cross validation.

    Parameters
    ----------
    gen_model : function
        Function that returns a newly created model when called
    train_model : function
        Function taking in model, train_data, val_data and performing actual
        training. It returns keras training history for the run.
    X : np.ndarray (or a list of these)
        Numpy array representing the model input
    y : np.ndarray
        Numpy array representing the actual output
    k : int
        k value for the k-fold split
    """

    kf = KFold(n_splits=k, shuffle=True)

    histories = []
    for train_indices, val_indices in tqdm(kf.split(X)):
        model = gen_model()

        if type(X) == list:
            # Model with multiple input
            train_data = ([x_sub[train_indices] for x_sub in X], y[train_indices])
            val_data = ([x_sub[val_indices] for x_sub in X], y[val_indices])
        else:
            train_data = (X[train_indices], y[train_indices])
            val_data = (X[val_indices], y[val_indices])

        histories.append(train_model(model, train_data, val_data))

    return  [
        {
            "training_loss": history.history["loss"][-1],
            "validation_loss": history.history["val_loss"][-1],
            "history": history
        }
        for history in histories
    ]


def cv_train_loso(gen_model, train_model, X, y, yi):
    """
    Train the model using leave-one-season-out cross validation.

    Parameters
    ----------
    gen_model : function
        Function that returns a newly created model when called
    train_model : function
        Function taking in model, train_data, val_data and performing actual
        training. It returns keras training history for the run.
    X : np.ndarray (or list of np.ndarray)
        Numpy array (or list of those) representing the model input
    y : np.ndarray
        Numpy array representing the actual output
    yi : np.ndarray
        Numpy array of (epiweek, region) indices
    """

    seasons = [epiweek_to_season(i[0]) for i in yi]
    unique_seasons = list(set(seasons))

    histories = []
    for season in tqdm(unique_seasons):
        model = gen_model()
        train_indices = np.array([i != season for i in seasons])

        if type(X) == list or type(X) == tuple:
            # Model with multiple input
            train_data = ([x_sub[train_indices] for x_sub in X], y[train_indices])
            val_data = ([x_sub[~train_indices] for x_sub in X], y[~train_indices])
        else:
            train_data = (X[train_indices], y[train_indices])
            val_data = (X[~train_indices], y[~train_indices])

        histories.append(train_model(model, train_data, val_data))

    cv_metadata = [
        {
            "training_loss": history.history["loss"][-1],
            "validation_loss": history.history["val_loss"][-1],
            "history": history
        }
        for history in histories
    ]

    return cv_metadata


def cv_report(cv_metadata):
    """
    Report the results of cross validation
    """

    lens = [len(it["history"].history["loss"]) for it in cv_metadata]
    losses = [it["training_loss"] for it in cv_metadata]
    val_losses = [it["validation_loss"] for it in cv_metadata]

    return pd.DataFrame({
        "epochs": lens + [np.median(lens)],
        "train_loss": losses + [np.mean(losses)],
        "val_loss": val_losses + [np.mean(val_losses)]
    }, index=["it-" + str(i) for i in range(1, len(cv_metadata) + 1)] + ["mean (median for epochs)"])


def cv_plot(cv_metadata):
    """
    Plot the training histories for cross validation
    """

    n = len(cv_metadata)

    cols = 2
    rows = (n // cols) + ((n % 2) * 1)

    f, axes = plt.subplots(rows, cols, figsize=(10, 15))

    for plot_n in range(n):
        i = plot_n // 2
        j = plot_n % 2
        axes[i][j].plot(cv_metadata[plot_n]["history"].history["loss"])
        axes[i][j].plot(cv_metadata[plot_n]["history"].history["val_loss"])


def save_exp_summary(model, cv_rep: pd.DataFrame, final_metadata: Dict, output_file: str):
    """
    Save a summary text for current experiment
    """

    with open(output_file, "w") as fp:
        fp.write("Model summary\n")
        fp.write("-------------\n")
        model.summary(print_fn=lambda line: fp.write(line + "\n"))
        fp.write("\n\n")
        fp.write("Cross validation\n")
        fp.write("----------------\n")
        fp.write(tabulate(cv_rep))
        fp.write("\n\n")
        fp.write("Final training\n")
        fp.write("--------------\n")
        fp.write(f"Epochs: {final_metadata['epochs']}\n")
        fp.write(f"Loss: {final_metadata['loss']}")
