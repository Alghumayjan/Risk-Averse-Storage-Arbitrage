import os, sys
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.forecasting.theta import ThetaModel
from tqdm import tqdm

def score_function(y, forecast):
    return np.array([forecast - y, y - forecast])
def set_function(forecast, q):
    return np.array([forecast - q[0], forecast + q[1]])
def m_shift(data):
    df = pd.DataFrame(np.array(data).T)
    for i in range(1, df.shape[1]):
        df.iloc[:, i] = df.iloc[:, i].shift(-i)
    return df

def aci(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in range(T_test):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            if alphat <= 1/(t_pred+1):
                qs[t] = np.infty
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.infty
    results = { "method": "ACI", "q" : qs, "alpha" : alphas}
    return results


def mytan(x):
    if x >= np.pi/2:
        return np.infty
    elif x <= -np.pi/2:
        return -np.infty
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI):
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)))
    out = KI * tan_out
    return  out

def saturation_fn_sqrt(x, t, Csat, KI):
    return KI * mytan((x * np.sqrt(t+1))/((Csat * (t+1))))


def quantile_integrator_log_scorecaster(
    scores,
    alpha,
    lr,
    data,
    T_burnin,
    Csat,
    KI,
    upper,
    ahead,
    integrate=True,
    proportional_lr=True,
    scorecast=True,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    scorecasts = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    seasonal_period = kwargs.get('seasonal_period')
    if seasonal_period is None:
        seasonal_period = 1
    for t in range(T_test):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue
        covereds[t] = qs[t] >= scores[t]
        grad = alpha if covereds[t_pred] else -(1-alpha)
        integrator_arg = (1-covereds)[:t_pred].sum() - (t_pred)*alpha
        integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI)
        if scorecast and t_pred > T_burnin and t+ahead < T_test:
            curr_scores = np.nan_to_num(scores[:t_pred])
            model = ThetaModel(
                    curr_scores.astype(float),
                    period=seasonal_period,
                    ).fit()
            if ahead ==1:
                scorecasts[t+ahead] = model.forecast(ahead)
            else:
                scorecasts[t+ahead] = model.forecast(ahead).iloc[-1]
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = integrator if integrate else 0
            qs[t+1] = qts[t+1] + integrators[t+1]
            if scorecast:
                qs[t+1] += scorecasts[t+1]
    results = {"method": "Quantile+Integrator (log)+Scorecaster", "q" : qs}
    return results


def predict_confidence(args):
    print(args['config'])
    config_name = args['config']
    ahead = args['ahead']
    start_date = args['start_date']
    end_date = args['end_date']

    for method in args['methods'].keys():
        print('training '+method)
        fn = None
        if method == "ACI":
            fn = aci
        elif method == "Quantile+Integrator (log)+Scorecaster":
            fn = quantile_integrator_log_scorecaster
        else:
            raise Exception(f"Method {method} not implemented")
        
        m_q0, m_q1, m_lower, m_upper, m_actual, m_forecast = [], [], [], [], [], []
        
        for h in tqdm(range(1,ahead+1)):
            df = pd.read_csv('../datasets/'+args['dataset']+'.csv')
            df = df.loc[:,['actual', 'forecast_'+str(h)]]
            df['timestamp'] = pd.date_range(start=start_date, end=end_date, freq='60min', inclusive='both')
            df.rename({'rtp': 'y'}, axis='columns', inplace=True)
            data = df.melt(id_vars=['timestamp'], value_name='target')
            data.rename({'variable': 'item_id'}, axis='columns', inplace=True)
            data.astype({'target': 'float64'})
            data = data.pivot(columns="item_id", index="timestamp", values="target")
            data['y'] = data['actual'].astype(float)
            data = data.interpolate()
            data.index = pd.to_datetime(data.index)

            data['scores'] = [ score_function(y, forecast) for y, forecast in zip(data['y'], data['forecast_'+str(h)]) ]

            lr = args['methods'][method]['lr']
            kwargs = args['methods'][method]
            kwargs["T_burnin"] = args["T_burnin"]
            kwargs["data"] = data
            kwargs["seasonal_period"] = args["seasonal_period"] if "seasonal_period" in args.keys() else None
            kwargs["config_name"] = config_name
            kwargs["ahead"] = h

            stacked_scores = np.stack(data['scores'].to_list())
            kwargs['upper'] = False
            q0 = fn(stacked_scores[:,0], args['alpha']/2, **kwargs)['q']
            kwargs['upper'] = True
            q1 = fn(stacked_scores[:,1], args['alpha']/2, **kwargs)['q']
            q = [ np.array([q0[i], q1[i]]) for i in range(len(q0)) ]

            sets = [ set_function(data['forecast_'+str(h)].interpolate().to_numpy()[i], q[i]) for i in range(len(q)) ]
            sets = [ np.array([np.minimum(sets[j][0], sets[j][1]), np.maximum(sets[j][1], sets[j][0])]) for j in range(len(sets)) ]

            lower = [np.minimum(set_function(data['forecast_'+str(h)].interpolate().to_numpy()[i], q[i])[0], 
                                    set_function(data['forecast_'+str(h)].interpolate().to_numpy()[i], q[i])[1]) 
                            for i in range(len(q))]

            upper = [np.maximum(set_function(data['forecast_'+str(h)].interpolate().to_numpy()[i], q[i])[0], 
                                    set_function(data['forecast_'+str(h)].interpolate().to_numpy()[i], q[i])[1]) 
                            for i in range(len(q))]
            
            m_q0.append(q0)
            m_q1.append(q1)
            m_lower.append(lower)
            m_upper.append(upper)
            m_actual.append(data['y'])
            m_forecast.append(data['forecast_'+str(h)])


        results = {'y': m_shift(m_actual), "forecast": m_shift(m_forecast),
                   "lower_bounds": m_shift(m_lower), "upper_bounds": m_shift(m_upper),
                   "q0": m_shift(m_q0), "q1": m_shift(m_q1)}
        results_foldername = './results/'
        os.makedirs(results_foldername, exist_ok=True)
        results_filename = results_foldername + config_name +'_'+ method +'_lr_'+str(lr)+ '_T_'+str(ahead)+"_"+str(args['alpha'])+".pkl"
        with open(results_filename, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
