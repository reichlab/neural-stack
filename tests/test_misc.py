"""
Misc test functions
"""

import src.utils.data as udata
import numpy as np

def test_model_week():
    for ew in [201434, 201354]:
        try:
            udata.epiweek_to_model_week(ew)
            assert False
        except:
            assert True

    assert udata.epiweek_to_model_week(200746) == 6
    assert udata.epiweek_to_model_week(200816) == 28
    assert udata.epiweek_to_model_week(201516) == 29

    assert udata.epiweek_to_model_week(np.nan) == 33
