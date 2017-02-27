"""
Module for prediction using trained models from the notebook
"""

from typing import List

import numpy as np
from keras.models import load_model

from .submission import (MAP_REGION, MAP_TARGET, Submission, read_csv,
                         segment_from_X, sub_from_segments)


def generate_submission(models: List,
                        submissions: List[Submission],
                        out_csv: str,
                        smoothing=None):
    """
    Use trained models on CSVs to generating a submission file in out_csv
    Expected order of lists:
    - models : 1 to 4 wk ahead models
    - submissions : kcde, kde, sarima submissions
    """

    assert len(models) == 4
    assert len(submissions) == 3
    assert len(set([sub.time % 100 for sub in submissions])) == 1

    time = submissions[0].time
    week = time % 100

    targets = ["one_week", "two_weeks", "three_weeks", "four_weeks"]
    regions = list(MAP_REGION.keys())

    predictions = []

    for idx, target in enumerate(targets):
        X = np.stack(
            [
                np.vstack([sub.get_X(region, target) for region in regions])
                for sub in submissions
            ],
            axis=-1)
        X_week = np.repeat([week], X.shape[0])

        predictions.append(models[idx].predict([X, X_week]))

    segments = []
    for tid, target in enumerate(targets):
        for rid, region in enumerate(regions):
            segments.append(
                segment_from_X(predictions[tid][rid], 2, region, target, time))

    # Add dummy values for other targets
    dummy_week = np.zeros(33)
    dummy_percent = np.zeros(131)
    dummy_week[0] = 1
    dummy_percent[0] = 1
    for target in ["onset", "peak_week"]:
        for region in regions:
            segments.append(
                segment_from_X(dummy_week, 33, region, target, time))

    for region in regions:
        segments.append(
            segment_from_X(dummy_percent, 33, region, "peak", time))

    final_submission = sub_from_segments(segments)
    final_submission.to_csv(out_csv)
