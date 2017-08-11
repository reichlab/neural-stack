"""
Module to work with submission files
"""

from typing import Dict, List

import numpy as np
import pandas as pd

SUB_HEADER = [
    "Location",
    "Target",
    "Type",
    "Unit",
    "Bin_start_incl",
    "Bin_end_notincl",
    "Value"
]

# Map from targets used in models to that in submissions
MAP_TARGET = {
    1: "1 wk ahead",
    2: "2 wk ahead",
    3: "3 wk ahead",
    4: "4 wk ahead",
    "onset_wk": "Season onset",
    "peak_wk": "Season peak week",
    "peak": "Season peak percentage"
}

# Map from region used in models to that in submissions
MAP_REGION = {
    "nat": "US National",
    "hhs1": "HHS Region 1",
    "hhs2": "HHS Region 2",
    "hhs3": "HHS Region 3",
    "hhs4": "HHS Region 4",
    "hhs5": "HHS Region 5",
    "hhs6": "HHS Region 6",
    "hhs7": "HHS Region 7",
    "hhs8": "HHS Region 8",
    "hhs9": "HHS Region 9",
    "hhs10": "HHS Region 10"
}


def segment_from_X(X: np.ndarray, point_prediction, region: str, target: str) -> Dict:
    """
    Create a segment of rows going into submission
    """

    df = {key: [] for key in SUB_HEADER}

    def _append_row(row):
        df[SUB_HEADER[0]].append(MAP_REGION[row[0]])
        df[SUB_HEADER[1]].append(MAP_TARGET[row[1]])
        for i in range(2, 7):
            df[SUB_HEADER[i]].append(row[i])

    if target in ["onset_wk", "peak_wk"]:
        # It should have 33 rows
        # TODO Fix the number of bins if the season has different number of weeks
        if X.shape != (33,):
            raise ValueError(f"X.shape needed (33,), got {X.shape}")

        # Not outputting point predictions since that is autocalculated by
        # flusight
        # TODO Fix if needed
        _append_row([region, target, "Point", "week", None, None, point_prediction])

        bin_starts = list(range(40, 53)) + list(range(1, 21))
        bin_ends = [i + 1 for i in bin_starts]

        for idx, x in enumerate(X):
            _append_row([
                region, target, "Bin", "week",
                str(bin_starts[idx]), str(bin_ends[idx]), x
            ])

        # Add none bin with 0 probability
        if target == "onset":
            _append_row([region, target, "Bin", "week", "none", "none", 0.0])

    else:
        # Assume percentage bin of size 131
        if X.shape != (131,):
            raise ValueError(f"X.shape needed (131,), got {X.shape}")

        # Point prediction
        # TODO Fix if needed
        _append_row([region, target, "Point", "percent", None, None, point_prediction])

        bin_starts = np.linspace(0, 13, 131)
        bin_ends = [i + 0.1 for i in bin_starts]
        # Set last bin_end to 100 to conform with submission format
        bin_ends[-1] = 100

        for idx, x in enumerate(X):
            _append_row([
                region, target, "Bin", "percent",
                f"{bin_starts[idx]:.1f}", f"{bin_ends[idx]:.1f}", x
            ])

    return pd.DataFrame(df)[SUB_HEADER]


def sub_from_segments(segments: List):
    """
    Merge segments into a single Submission object
    Assume ordered segments in submission format
    """

    df = pd.concat([segment for segment in segments])

    return Submission(df)


class Submission:
    """
    Class for submission file in long format
    """

    def __init__(self, df) -> None:
        """
        Create submission object from df
        """

        self.df = df

    def to_csv(self, file_name) -> None:
        """
        Write submission file to given path
        """

        self.df.to_csv(file_name, na_rep="NA", index=False)

    def get_X(self, region: str, target: str) -> np.ndarray:
        """
        Return X for asked region and target
        """

        values = self.df[
            (self.df["Location"] == MAP_REGION[region]) & \
            (self.df["Target"] == MAP_TARGET[target])
        ]["Value"].as_matrix()[1:] # Skip point predictions

        return values
