import pandas as pd
import numpy as np

from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

#### a Realized Covariances:####

import pandas as pd
import numpy as np


#### HMM ### ##

class MarketRegime ():
    def __init__(self, 
                 df: pd.DataFrame,
                 ):
        self.df = df
        pass
    
    #### HMM #####
    def _train_HMM (self, 
                    window_size: int):
        self.df['log_ret'] = np.log(self.df['close']).diff()
        self.df['vol'] = self.df['log_ret'].rolling(window_size).std()  # short-term vol
        features = self.df[['log_ret', 'vol']].dropna()
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(features)
        return model
    
    def _regime_HMM (self,
                     window_size: int,
                     df_predict: pd.DataFrame,
                     model):
        df_predict['log_ret'] = np.log(df_predict['close']).diff()
        df_predict['vol'] = self.df['log_ret'].rolling(window_size).std()  # short-term vol
        features = df_predict[['log_ret', 'vol']].dropna()
        hidden_states = model.predict(features)
        df_predict =  df_predict.loc[features.index].copy()
        df_predict['regime'] = hidden_states
        return df_predict
    
    ### Slope clustering ####

    def _train_SlopeCluster (self, 
                             window_size:int,
                             window_size_rolling:int,
                             ):
        
        self.df['ma34'] = self.df['close'].rolling(34).mean()
        self.df = self.df.dropna()
        timestamps = np.arange(window_size).reshape(-1, 1)
        slopes = []
        for i in range(window_size, len(self.df)):
            y = self.df['ma34'].values[i-window_size:i]
            X = timestamps
            reg = LinearRegression().fit(X, y)
            slopes.append(reg.coef_[0])
        self.df = self.df.iloc[window_size:].copy()
        self.df['slope'] = slopes
        # 4. Cluster slopes into regimes (3 clusters)
        num_clusters = 3
        X = self.df[['slope']].values
        slopecluster = KMeans(n_clusters=num_clusters, random_state=42).fit(X)
        return slopecluster 
    
    def _regime_SlopeCluster(self,
                             window_size:int,
                             window_size_rolling:int,
                             df_predict: pd.DataFrame,
                             cluster_model):
        df_predict['ma34'] = df_predict['close'].rolling(window_size_rolling).mean()
        df_predict = df_predict.dropna()
        timestamps = np.arange(window_size).reshape(-1, 1)
        slopes = []
        for i in range(window_size, len(df_predict)):
            y = df_predict['ma34'].values[i-window_size:i]
            X = timestamps
            reg = LinearRegression().fit(X, y)
            slopes.append(reg.coef_[0])
        df_predict = df_predict.iloc[window_size:].copy()
        df_predict['slope'] = slopes
        df_predict['regime'] = cluster_model.labels_
        return df_predict
    






        
    

