import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gurobipy as gp 
from gurobipy import GRB
from pyepo.model.grb import optGrbModel
import pickle as pkl
import cvxpy as cp
from scipy.stats import truncnorm

import multiprocessing
from tqdm import tqdm
import sys


def calculate_vars(outputs, soc_init, mode='aggressive'):
    select_function = max if mode == 'aggressive' else min
    chr = 0 if any(row[0].value[0] == 0 for row in outputs) else select_function(outputs, key=lambda x: x[0].value[0])
    if chr == 0:
        dis = 0 if any(row[1].value[0] == 0 for row in outputs) else select_function(outputs, key=lambda x: x[1].value[0])
        if dis == 0:
            return 0.0, 0.0, soc_init
        return dis[0].value[0], dis[1].value[0], dis[2].value[0]
    return chr[0].value[0], chr[1].value[0], chr[2].value[0]

def generate_sample(lower_bound_array, upper_bound_array):
    return lower_bound_array + (upper_bound_array - lower_bound_array) * np.random.rand(*lower_bound_array.shape)

def MPC_main_run(updated_forecasted_prices, updated_soc_init):
    forecasted_prices.value = updated_forecasted_prices
    soc_init.value = updated_soc_init
    original_stdout = sys.stdout  
    sys.stdout = open('/dev/null', 'w') 
    
    problem.solve()
    
    sys.stdout.close()
    sys.stdout = original_stdout
    return chr_var, dis_var, soc_var