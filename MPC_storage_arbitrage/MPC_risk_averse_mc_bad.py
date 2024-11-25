import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gurobipy as gp 
from gurobipy import GRB
from pyepo.model.grb import optGrbModel
import pickle as pkl
import cvxpy as cp

import multiprocessing
from tqdm import tqdm
import sys, os

from arbitrage_utils import*

for k in tqdm(range(40)):
    forecaster = 'bad_r'+str(k)

    data_name = "nyiso_nyc_"+forecaster+"_point_predictions"
    method = 'Quantile+Integrator (log)+Scorecaster'
    lr = 0.05
    T = 6
    results_filename = '../conformal_control/results/'+data_name+'_'+ method +'_lr_'+str(lr)+ '_T_6_0.05.pkl'
    with open(results_filename, 'rb') as handle:
        all_results = pkl.load(handle)
    D = int((24/T)*362)
    soc_max = 1  
    power_rating = 0.5
    eta = 0.9

    low_pred = np.array(all_results['lower_bounds'][24*16:-24])[:,:T]
    up_pred = np.array(all_results['upper_bounds'][24*16:-24])[:,:T]
    actual = np.array(all_results['y'][24*16:-24])[:,:T]
    pred = np.array(all_results['forecast'][24*16:-24])[:,:T]
    sampled_arrays = [generate_sample(low_pred, up_pred) for _ in range(100)]

    modes = ['forecasted','conservative','aggressive']
    modes_profits = {}
    for mode in modes:
        print(mode)
        plt.figure(figsize=(8, 6), dpi=300)
        chr_optimal = np.zeros([D, T])
        dis_optimal = np.zeros([D, T])
        revenue_list = np.zeros([D, T])
        soc_values = np.zeros([D, T])  
        soc_init_ = 0.5  

        actual_prices = actual

        soc_var = cp.Variable(T, nonneg=True)
        chr_var = cp.Variable(T, nonneg=True)
        dis_var = cp.Variable(T, nonneg=True)
        charge_decision = cp.Variable(T, boolean=True)
        discharge_decision = cp.Variable(T, boolean=True)
        discharge_active = cp.Variable(T, boolean=True)

        M = 1e6 

        forecasted_prices = cp.Parameter(T)
        soc_init = cp.Parameter()

        cons = [
            soc_var[0] - soc_init == chr_var[0] * eta - dis_var[0] / eta,
            *[soc_var[i] - soc_var[i-1] == chr_var[i] * eta - dis_var[i] / eta for i in range(1, T)],
            soc_var[T-1] >= soc_init,
            *[chr_var[i] <= power_rating for i in range(T)],
            *[soc_var[i] <= soc_max for i in range(T)],
            *[charge_decision[i] + discharge_decision[i] <= 1 for i in range(T)],
            *[dis_var[i] <= discharge_active[i] * power_rating for i in range(T)],
            *[dis_var[i] >= 0 for i in range(T)],
            *[forecasted_prices[i] >= -M * (1 - discharge_active[i]) for i in range(T)],
            *[forecasted_prices[i] <= M * discharge_active[i] for i in range(T)]
        ]

        revenue = cp.multiply(forecasted_prices, (dis_var - chr_var)) - 10 * dis_var
        problem = cp.Problem(cp.Maximize(cp.sum(revenue)), cons)

        for d in range(D):
            for t in range(T):
                if mode in ['conservative', 'aggressive']:
                    cases = [(sample[t+T*d, :], soc_init_) for i, sample in enumerate(sampled_arrays)]
                    pool = multiprocessing.Pool(processes=10)
                    outputs = pool.starmap(MPC_main_run, cases)
                    chr_optimal[d, t], dis_optimal[d, t], soc_init_ = calculate_vars(outputs, soc_init_, mode=mode)
                    revenue_list[d, t] = actual_prices[t+T*d, t]*(dis_optimal[d, t] - chr_optimal[d, t]) - 10*dis_optimal[d, t]
                    soc_values[d, t] = soc_init_
                elif mode == 'forecasted':
                    forecasted_prices_ = pred
                    chr_var_, dis_var_, soc_var_ = MPC_main_run(forecasted_prices_[t+T*d, :], soc_init_)
                else:
                    forecasted_prices_ = actual
                    chr_var_, dis_var_, soc_var_ = MPC_main_run(forecasted_prices_[t+T*d, :], soc_init_)

                if mode in ['actual', 'forecasted']:
                    chr_optimal[d, t] = chr_var_.value[0]
                    dis_optimal[d, t] = dis_var_.value[0]
                    revenue_list[d, t] = actual_prices[t+T*d, t]*(dis_var_.value[0] - chr_var_.value[0]) - 10*dis_var_.value[0]
                    soc_init_ = soc_var_.value[0]
                    soc_values[d, t] = soc_init_

        revenue_cum = (actual_prices[:T*D, 0] * (dis_optimal.flatten() - chr_optimal.flatten()) - 10 * dis_optimal.flatten()).cumsum()
        print('total profit for ' + mode + ' =', int(revenue_cum[-1]))

        modes_profits[mode] = revenue_cum

        solutions = {
            'RTP': actual_prices[:T*D, 0],
            'Discharge': dis_optimal.flatten(),
            'Charge': chr_optimal.flatten(),
            'SoC': soc_values.flatten(),
            'Revenue': revenue_list.flatten()
        }

        results_foldername = '.MC_bad/'+forecaster+'_arbitrage/'
        os.makedirs(results_foldername, exist_ok=True)

        pd.DataFrame(solutions).to_csv(f"{results_foldername}/solution_{data_name}_{method}_MPC_{mode}_{str(T)}_0.05.csv")

