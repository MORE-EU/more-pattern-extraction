from sklearn.base import BaseEstimator 
from sklearn.base import ClusterMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import QuantileRegressor

import lightgbm as lgb
from estimator_util import *

scaler = MinMaxScaler()
class UnderperformanceEstimator(ClusterMixin, BaseEstimator):

    def __init__(self,  params = {'objective': 'quantile', 'metric': 'quantile', 'max_depth': 4, 'num_leaves': 30,'learning_rate': 0.1, 'n_estimators': 100, 'boosting_type': 'gbdt'},
                quantile = 0.9, segm_thsh = 0.3, chp_thsh = 0.999):
        self.params = params
        self.quantile = quantile
        self.segm_thsh = segm_thsh
        self.chp_thsh = chp_thsh
        self.scores = []
        
    def fit(self, df_train, y):
        df_train = df_train.copy()
        X = df_train.values
        y = y.copy()
        segm_thsh = self.segm_thsh
        chp_thsh = self.chp_thsh
        params = self.params
        quantile = self.quantile
        scaler.fit(X)
        X_t_scaled = scaler.transform(X)
        scaler.fit(y.reshape(-1, 1))
        y_true = scaler.transform(y.reshape(-1, 1))[:,0]
        qr = lgb.LGBMRegressor(alpha=quantile, **params)
        model = qr.fit(X_t_scaled,y_true)
        #prediction
        y_pred = model.predict(X_t_scaled)
        res = compute_residual(df_train.index, y_true, y_pred)
        chps = detect_changepoints(df_train.index, res, 500, chp_thsh)
        chps = chps.round("1D").drop_duplicates() 

        scores_index, scores_set = score_segments(df_train.index, res, chps)
        mask = (scores_index.values>np.quantile(scores_set, segm_thsh))[:,0]
        self.chps = chps
        self.res = res.values
        self.labels_ = np.array([0]*len(mask))

        self.labels_[mask] = 1
        self.scores = scores_index.values
        return self

    
