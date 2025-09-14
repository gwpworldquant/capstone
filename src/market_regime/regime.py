import pandas as pd
import numpy as np

from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans


class MarketRegime:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # -------------------------------
    # HMM training
    # -------------------------------
    def _train_HMM(self, window_size: int):
        self.df['log_ret'] = np.log(self.df['close']).diff()
        self.df['vol'] = self.df['log_ret'].rolling(window_size).std()
        features = self.df[['log_ret', 'vol']].dropna()

        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        model.fit(features)
        return model

    def _regime_HMM(self, window_size: int, df_predict: pd.DataFrame, model):
        df_predict = df_predict.copy()
        df_predict['log_ret'] = np.log(df_predict['close']).diff()
        df_predict['vol'] = df_predict['log_ret'].rolling(window_size).std()

        features = df_predict[['log_ret', 'vol']].dropna()
        hidden_states = model.predict(features)

        df_predict = df_predict.loc[features.index].copy()
        df_predict['regime'] = hidden_states
        return df_predict

    # -------------------------------
    # Slope clustering (t vs t-2 slope)
    # -------------------------------
    def _train_SlopeCluster(self, window_size: int, window_size_rolling: int):
        # 1. Compute MA34
        self.df['ma34'] = self.df['close'].rolling(window_size_rolling).mean()
        self.df = self.df.dropna().reset_index(drop=True)

        # 2. Compute slope (t vs t-2)
        slopes = [np.nan, np.nan]
        for i in range(2, len(self.df)):
            slope = (self.df['ma34'].iloc[i] - self.df['ma34'].iloc[i - 2]) / 2
            slopes.append(slope)

        self.df['slope'] = slopes

        # 3. Drop warm-up
        self.df = self.df.iloc[window_size:].copy().reset_index(drop=True)

        # 4. Fit KMeans
        num_clusters = 3
        X = self.df[['slope']].dropna().values
        slopecluster = KMeans(n_clusters=num_clusters, random_state=42).fit(X)

        # 5. Assign labels
        self.df.loc[self.df['slope'].notna(), 'regime'] = slopecluster.labels_

        return slopecluster

    def _regime_SlopeCluster(self,
                             window_size: int,
                             window_size_rolling: int,
                             df_predict: pd.DataFrame,
                             cluster_model):
        df_predict = df_predict.copy()

        # 1. Compute MA34
        df_predict['ma34'] = df_predict['close'].rolling(window_size_rolling).mean()
        df_predict = df_predict.dropna().reset_index(drop=True)

        # 2. Compute slope (t vs t-2)
        slopes = [np.nan, np.nan]
        for i in range(2, len(df_predict)):
            slope = (df_predict['ma34'].iloc[i] - df_predict['ma34'].iloc[i - 2]) / 2
            slopes.append(slope)

        df_predict['slope'] = slopes

        # 3. Drop warm-up
        df_predict = df_predict.iloc[window_size:].copy().reset_index(drop=True)

        # 4. Predict regime
        X_pred = df_predict[['slope']].dropna().values
        df_predict.loc[df_predict['slope'].notna(), 'regime'] = cluster_model.predict(X_pred)

        return df_predict
