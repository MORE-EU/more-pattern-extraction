import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from matplotlib import pyplot as plt

def enumerate2(start, end, step=1):
    i=0
    while start < pd.to_datetime(end):
        yield (i, start)
        start = pd.to_datetime(start) + pd.Timedelta(days=step)
        i += 1

def mape1(Yac, Ypre):
    mape1 = (np.mean(np.abs(Yac-Ypre)/np.mean(Yac)))
    return mape1

def mpe1(Yac, Ypre):
    mpe1 = (np.mean(Yac-Ypre)/np.mean(Yac))
    return mpe1

def predict(df_test, model, feats, target):
    df_x = df_test[feats]
    df_y = df_test[target]
    X = df_x.values
    y_true = df_y.values
    y_pred = model.predict(X)
    return y_pred

def score(y_true, y_pred):
    r_sq = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred) # MAE
    me = np.mean(y_true-y_pred) # Mean Erro y_true - y_pred -> if positive it means that y_true is more than the expected y_pred
    mape = mape1(y_true, y_pred)
    mpe = mpe1(y_true, y_pred)
    return r_sq, mae, me, mape, mpe

#comparing to the definition in myfunctions, it also returns mpe
def fit_linear_model(df, feats, target, a=1e-4, deg=3):
    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values

    polynomial_features = PolynomialFeatures(degree=deg)
    linear_regression = Ridge(alpha=a)

    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    r_sq, mae, me, mape, mpe = score(y, y_pred)
    return pipeline, y_pred, r_sq, mae, me, mape, mpe

def get_line_and_slope(values):
    ols = LinearRegression()
    X = np.arange(len(values)).reshape(-1,1)
    y = values.reshape(-1,1)
    ols.fit(X, y)
    line = ols.predict(X)
    slope = ols.coef_.item()
    intercept = ols.intercept_.item()
    return line, slope, intercept

def train_on_reference_points(df, w_train, ref_points, feats, target, random_state=0):
    df_train = pd.DataFrame([])
    df_val = pd.DataFrame([])
    for idx in range(ref_points.size):
        d_train_stop = pd.to_datetime(ref_points[idx]) + pd.Timedelta(days=w_train)
        df_tmp = df.loc[ref_points[idx]:str(d_train_stop)]
        df_tmp2 = df_tmp.sample(frac=1, random_state=random_state) # added random state for reproducibility during experiments
        size_train = int(len(df_tmp2) * 0.80)
        df_train = df_train.append(df_tmp2[:size_train])
        df_val = df_val.append(df_tmp2[size_train:])

    model, y_pred_train, r_sq_train, mae_train, me_train, mape_train = fit_linear_model(df_train, feats, target)
    y_pred_val = predict(df_val, model, feats, target)
    r_sq_val, mae_val, me_val, mape_val, mpe_val = score(df_val[target].values, y_pred_val)
    training_scores = np.array([r_sq_train, mae_train, me_train, mape_train])
    validation_scores = np.array([r_sq_val, mae_val, me_val, mape_val, mpe_val])

    print('Training Metrics:')
    print(f'MAE:{training_scores[1]:.3f} \nME(true-pred):{training_scores[2]:.3f} \nMAPE:{training_scores[3]:.3f} \nR2: {training_scores[0]:.3f}\n')
    print('Validation Metrics:')
    print(f'MAE:{validation_scores[1]:.3f} \nME(true-pred):{validation_scores[2]:.3f} \nMAPE:{validation_scores[3]:.3f} \nMPE:{validation_scores[4]:.3f} \nR2: {validation_scores[0]:.3f}\n')
    return model, training_scores, validation_scores

def predict_on_sliding_windows(df, win_size, step, model, feats, target):
    windows = []
    preds_test = []
    scores_list = []
    for i, time in enumerate2(min(df.index), max(df.index), step=step):
        window = pd.to_datetime(time) + pd.Timedelta(days=win_size)
        df_test = df.loc[time:window]
        if df_test.shape[0]>0:
            y_pred = predict(df_test, model, feats, target)
            r_sq, mae, me, mape, mpe = score(df_test[target].values, y_pred)
            scores_list.append([r_sq, mae, me, mape, mpe])
            preds_test.append(y_pred)
            windows.append((time, window))
    scores = np.array(scores_list)
    return scores, preds_test, windows

def get_top_deviations(scores, metric='mpe'):
    metrics = {'mpe': 4, 'me': 2}
    score_column = metrics[metric]
    indices = np.argsort(scores[:,score_column])[:5]
    return indices
def changepoint_scores(df, d1, d2, w_train, w_val, w_test):
    d_train_start = pd.to_datetime(d1) - pd.Timedelta(days=w_train) - pd.Timedelta(days=w_val)
    d_train_stop = pd.to_datetime(d1) - pd.Timedelta(days=w_val)
    d_test_stop = pd.to_datetime(d2) + pd.Timedelta(days=w_test)
    df_train = df.loc[str(d_train_start):str(d_train_stop)]
    df_val = df.loc[str(d_train_stop):str(d1)]
    df_test = df.loc[str(d2):str(d_test_stop)]
    if len(df_train) > 0 and len(df_test) > 0:
        model, y_pred_train, r_sq_train, mae_train, me_train, mape_train, mpe_train = fit_linear_model(df_train, ['irradiance', 'mod_temp'], 'power')
        y_pred_val = predict(df_val, model, ['irradiance', 'mod_temp'], 'power')
        y_pred_test = predict(df_test, model, ['irradiance', 'mod_temp'], 'power')
        
        r_sq_val, mae_val, me_val, mape_val, mpe_val = score(df_val['power'].values, y_pred_val)
        r_sq_test, mae_test, me_test, mape_test, mpe_test = score(df_test['power'].values, y_pred_test)
        score_train = np.array([-r_sq_train, mae_train, me_train, mape_train, mpe_train])
        score_val = np.array([-r_sq_val, mae_val, me_val, mape_val, mpe_val])
        score_test = np.array([-r_sq_test, mae_test, me_test, mape_test, mpe_test])
        return y_pred_train, score_train, y_pred_val, score_val, y_pred_test, score_test
    else:
        raise Exception("Either the training set is empty or the test set is empty")
