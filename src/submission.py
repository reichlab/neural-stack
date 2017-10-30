"""
Module to work with submission files in CDC format
"""

from typing import Dict, List, Any, Union, Tuple

import numpy as np
import pandas as pd
from functools import cmp_to_key
from pathlib import Path

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

    df = {key: [] for key in SUB_HEADER}  # type: Dict[str, List[Any]]

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
        bin_ends = [i + 1 for i in bin_starts]  # type: List[Union[int, float]]

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

    return Submission(df=df)


class Submission:
    """
    Class for submission file in long format
    """

    def __init__(self, df: pd.DataFrame=None, csv: Union[Path, str]=None) -> None:
        """
        Create submission object from df. The df is something like this

        Location,Target,Type,Unit,Bin_start_incl,Bin_end_notincl,Value
        US National,1 wk ahead,Point,percent,NA,NA,1.7
        US National,2 wk ahead,Point,percent,NA,NA,2.2
        US National,3 wk ahead,Point,percent,NA,NA,2.5
        US National,4 wk ahead,Point,percent,NA,NA,2.8

        Ordering is not gauranteed in the df itself. While returning an asked
        subset (get_X), we sort the rows according to the ordering of
        Bin_start_incl
        """

        if csv:
            self.df = pd.read_csv(csv)
        elif df:
            self.df = df
        else:
            raise Exception("No argument provided to Submission")

    def to_csv(self, file_name) -> None:
        """
        Write submission file to given path
        """

        self.df.to_csv(file_name, na_rep="NA", index=False)

    def get_X(self, region: str, target: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return X for asked region and target with bin ordering.
        Ordering is simple for percent values, we go like [0.0, 0.1, ..., 13.0]

        For weeks, we need to go from 40 to 52/53 and then from 1
        to 20. Furthermore, we only return 33 week bins. This means we skip
        week 20 in a season with 53 weeks. In case of onset, we return 1 more
        bin (the last one) to account for the no onset case.
        """

        subset = self.df[
            (self.df["Location"] == MAP_REGION[region]) & \
            (self.df["Target"] == MAP_TARGET[target]) & \
            (self.df["Type"] == "Bin")
        ][["Bin_start_incl", "Value"]]

        if subset.shape[0] == 131:
            # These are percent bins
            mat = subset.apply(pd.to_numeric).sort_values(by="Bin_start_incl")[["Value", "Bin_start_incl"]].as_matrix()
            return (mat[:, 0], mat[:, 1])
        else:
            # These are week bins
            def _clear_bin(x):
                try:
                    return int(x)
                except ValueError:
                    return np.nan

            probs = [float(p) for p in subset["Value"].tolist()]
            bins = [_clear_bin(bs) for bs in subset["Bin_start_incl"].tolist()]

            year_end = max(bins)
            season_end = 20

            def _sort_key(it):
                season_weeks = list(range(40, year_end + 1)) + list(range(1, season_end + 1))

                # Return onset bin 'none' as last bin
                if np.isnan(it[1]):
                    return 34
                else:
                    return season_weeks.index(it[1])

            sorted_pairs = sorted(zip(probs, bins), key=_sort_key)

            # Skip last probability bin if we are in 53 week season
            if year_end == 53:
                if target == "onset_wk":
                    sorted_pairs.pop(len(sorted_pairs) - 2)
                else:
                    sorted_pairs.pop()

            return ([i[0] for i in sorted_pairs], [i[1] for i in sorted_pairs])
