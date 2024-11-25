import os, sys
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from conformal_utils import*


for run in range(40):
    data_name = "nyiso_nyc_bad_r"+str(run)+"_point_predictions"
    args = {
        "config": data_name,
        "T_burnin": 24*14,
        "alpha": 0.05,
        "seasonal_period": 24,
        "ahead": 6,
        "dataset": data_name,
        "start_date": '2022-12-15 00:00:00',
        "end_date": '2023-12-28 23:00:00',
        "methods": {
            "Quantile+Integrator (log)+Scorecaster": {
                "lr": 0.05,
                "Csat": 5,
                "KI": 10,
                "steps_ahead": 1
            }
        }
    }
    predict_confidence(args)