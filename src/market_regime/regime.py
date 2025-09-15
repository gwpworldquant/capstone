import pandas as pd
import numpy as np

from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from collections import Counter

class MarketRegime:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def label_regime(self,df, price_col='close', regime_col='regime'):
        df = df.copy()
        results = []

        # Đánh dấu khi nào regime thay đổi
        df['regime_shift'] = df[regime_col].ne(df[regime_col].shift()).cumsum()

        for _, group in df.groupby('regime_shift'):
            regime = group[regime_col].iloc[0]
            start_idx = group.index[0]
            end_idx = group.index[-1]
            start_price = group[price_col].iloc[0]
            end_price = group[price_col].iloc[-1]

            results.append({
                'regime': regime,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_price': start_price,
                'end_price': end_price,
                'price_change': end_price - start_price
            })
            result = pd.DataFrame(results)
        median_stats = result.groupby('regime').min('price_change')
        sorted_states = median_stats.sort_values('price_change', ascending=False)
        print(sorted_states)
        n_states = len(sorted_states)
        if n_states == 1:
            return {sorted_states.index[0]:'no regime'}
        else:

            state_labels = {
                    sorted_states.index[0]: 'Bull- increase',       # s
                    sorted_states.index[1]: 'Sideways',   # middle gap
                    sorted_states.index[2]: 'Bear - decrease'        # largest gap
                }

            return  state_labels 

    
    def smooth_by_interval(self, 
                           data,
                           interval_size):
        
        intervals = [data[i:i+interval_size] for i in range(0, len(data), interval_size)]
        smoothed_data = []
        for interval in intervals:
            dominant = Counter(interval).most_common(1)[0][0]  # find dominant regime
            smoothed_data.extend([dominant] * len(interval))
        return smoothed_data


    # -------------------------------
    # HMM training
    # -------------------------------
    def _train_HMM(self, window_size_rolling: int):
        self.df['return'] = self.df['close'].pct_change()
        self.df['price_std'] = self.df['close'].rolling(window_size_rolling).std()
        self.df['vol_change'] = self.df['volume'].pct_change()
        features = self.df[['return', 'vol_change', 'price_std']].dropna()

        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        model.fit(features)
        ### Assign hidden states to df
        hidden_states = model.predict(features)
        self.df = self.df.loc[features.index].copy()
        self.df['regime'] = hidden_states
        self.df['regime'] = self.smooth_by_interval(self.df['regime'],interval_size = 8)
        state_labels = self.label_regime(self.df, price_col='close', regime_col='regime')

        return model, state_labels

    def _regime_HMM(self, 
                    window_size_rolling: int, 
                    df_predict: pd.DataFrame, 
                    model, 
                    state_labels):
        df_predict = df_predict.copy()
        df_predict['return'] = df_predict['close'].pct_change()
        df_predict['price_std'] = df_predict['close'].rolling(window_size_rolling).std()
        df_predict['vol_change'] = df_predict['volume'].pct_change()

        features = df_predict[['return', 'vol_change', 'price_std']].dropna()
        hidden_states = model.predict(features)
        df_predict = df_predict.loc[features.index].copy()
        df_predict['regime'] = hidden_states
        df_predict['regime'] = self.smooth_by_interval(df_predict['regime'],interval_size = 8)
        df_predict['regime_labels'] = df_predict['regime'].map(state_labels)
        return df_predict

    # -------------------------------
    # Slope clustering (Linear Regression slope)
    # -------------------------------
    def _train_SlopeCluster(self, slope_window: int, window_size_rolling: int):
        # 1. Compute MA
        self.df['ma'] = self.df['close'].rolling(window_size_rolling).mean()
        self.df = self.df.dropna().reset_index(drop=True)

        # 2. Compute slope with Linear Regression
        slopes = [np.nan] * (slope_window - 1)
        for i in range(slope_window - 1, len(self.df)):
            y = self.df['ma'].iloc[i - slope_window + 1:i + 1].values.reshape(-1, 1)
            x = np.arange(slope_window).reshape(-1, 1)  # time index
            reg = LinearRegression().fit(x, y)
            slopes.append(float(reg.coef_[0]))
        self.df['slope'] = slopes

        self.df['vol_change'] = self.df['volume'].pct_change()

        # 3. Drop warm-up
        self.df = self.df.iloc[slope_window:].copy().reset_index(drop=True)

        # 4. Fit KMeans with slope + volume change
        num_clusters = 3
        X = self.df[['slope', 'vol_change']].dropna().values

        slopecluster = KMeans(n_clusters=num_clusters, random_state=42).fit(X)

        # 5. Assign labels
        self.df.loc[self.df[['slope', 'vol_change']].notna().all(axis=1), 'regime'] = slopecluster.predict(X)
        self.df['regime'] = self.smooth_by_interval(self.df['regime'],interval_size = 8)
        state_labels = self.label_regime(self.df, price_col='close', regime_col='regime')

        return slopecluster, state_labels

    def _regime_SlopeCluster(self,
                             slope_window: int,
                             window_size_rolling: int,
                             df_predict: pd.DataFrame,
                             cluster_model, 
                             state_labels):
        df_predict = df_predict.copy()
       
        # 1. Compute MA
        df_predict['ma'] = df_predict['close'].rolling(window_size_rolling).mean()
        df_predict = df_predict.dropna().reset_index() 
       

        # 2. Compute slope with Linear Regression
        slopes = [np.nan] * (slope_window - 1)
        for i in range(slope_window - 1, len(df_predict)):
            y = df_predict['ma'].iloc[i - slope_window + 1:i + 1].values.reshape(-1, 1)
            x = np.arange(slope_window).reshape(-1, 1)
            reg = LinearRegression().fit(x, y)
            slopes.append(float(reg.coef_[0]))
        df_predict['slope'] = slopes
       
        df_predict['vol_change'] = df_predict['volume'].pct_change()
        


        # 3. Drop warm-up
        df_predict = df_predict.iloc[slope_window:].copy().reset_index(drop=True)

        # 4. Predict regime
        X_pred = df_predict[['slope', 'vol_change']].dropna().values
        df_predict.loc[df_predict[['slope', 'vol_change']].notna().all(axis=1), 'regime'] = \
            cluster_model.predict(X_pred)
    
        df_predict['regime'] = self.smooth_by_interval(df_predict['regime'],interval_size = 8)
        df_predict['regime_labels'] = df_predict['regime'].map(state_labels)
        

        return df_predict
