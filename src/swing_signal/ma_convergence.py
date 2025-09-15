import pandas as pd
import numpy as np


import pandas as pd

class CoiledSpringEMAConvergenceStrategy:
    def __init__(self, ema_periods=(34, 55, 84), thresholds=(0.01, 0.05)):
        """
        :param ema_periods: Tuple of EMA periods (default: (34, 55, 84))
        :param thresholds: Tuple of convergence thresholds for (15m, 30m)
        """
        self.ema_periods = ema_periods
        self.thresholds = thresholds

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA for a given series and period."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def is_converged(ema34, ema55, ema84, threshold=0.001) -> bool:
        """Check if EMA lines are converged within threshold."""
        max_val = max(ema34, ema55, ema84)
        min_val = min(ema34, ema55, ema84)
        return (max_val - min_val) / min_val < threshold

    def compute_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute EMAs for all configured periods."""
        for period in self.ema_periods:
            df[f'EMA{period}'] = self.ema(df['close'], period)
        return df

    def detect_convergence(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> tuple:
        """Add convergence columns for both 15m and 30m dataframes."""
        df_1h = self.compute_emas(df_4h)
        df_1h = self.compute_emas(df_4h)

        df_1h['converged'] = df_1h.apply(
            lambda row: self.is_converged(row['EMA34'], row['EMA55'], row['EMA84'], self.thresholds[0]),
            axis=1
        )
        df_4h['converged'] = df_4h.apply(
            lambda row: self.is_converged(row['EMA34'], row['EMA55'], row['EMA84'], self.thresholds[1]),
            axis=1
        )
        return df_1h, df_4h

    def generate_signals(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> list:
        """
        Generate buy/sell signals based on EMA convergence and regime filters.
        """
        signals = []

        for i in range(1, len(df_1h)):
            prev = df_1h.iloc[i - 1]
            curr = df_1h.iloc[i]

            # Match 30m time
            if len(df_4h) == 0:
                continue
            time_30m = df_4h.index[df_4h.index.get_indexer([curr.name], method='pad')[0]]
            if time_30m not in df_4h.index:
                continue

            # Regime filter: skip if 30m regime == 1 (bullish, for example)
            regime_4h = df_4h.loc[time_30m, 'regime'] if 'regime' in df_4h.columns else None
            if regime_4h == 1:
                continue

            converged_15m = curr['converged']
            converged_30m = df_4h.loc[time_30m, 'converged']

            if converged_15m and converged_30m:
                # BUY signal
                if (prev['EMA34'] < prev['EMA55']) and (curr['EMA34'] > curr['EMA55']) \
                   and (curr['EMA34'] > curr['EMA84']):
                    signals.append((curr.name, 'BUY', curr['close']))

                # # SELL signal
                # elif (prev['EMA34'] > prev['EMA55']) and (curr['EMA34'] < curr['EMA55']) \
                #      and (curr['EMA34'] < curr['EMA84']):
                #     signals.append((curr.name, 'SELL', curr['close']))

        return signals


class Dip_Tip_MA_Convergence_H4_to_H1:
    def __init__(self, ema_periods=(34, 55, 84), convergence_threshold=0.001, price_tolerance=0.005):
        self.ema_periods = ema_periods
        self.convergence_threshold = convergence_threshold
        self.price_tolerance = price_tolerance

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def compute_emas(self, df: pd.DataFrame) -> pd.DataFrame:
     
        for p in self.ema_periods:
            df[f'ema_{p}'] = self.ema(df['close'], p)
        df['ema_mid'] = df[[f'ema_{p}' for p in self.ema_periods]].mean(axis=1)
        df['max_ema_gap'] = df[[f'ema_{p}' for p in self.ema_periods]].max(axis=1) - \
                            df[[f'ema_{p}' for p in self.ema_periods]].min(axis=1)
        df['ema_converged'] = df['max_ema_gap'] / df['ema_mid'] < self.convergence_threshold
        return df

    def generate_signals(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
        
        df_4h = self.compute_emas(df_4h)

        # Danh sách mức kháng cự từ H4
        resistance_levels = df_4h.loc[df_4h['ema_converged'], 'ema_mid'].tolist()

        # Tạo cột tín hiệu H1
        df_1h['signal'] = False

        for i in range(len(df_1h)):
            price = df_1h['close'].iloc[i]
            # Nếu giá H1 rơi về gần bất kỳ mức kháng cự H4 trước đó
            for lvl in resistance_levels:
                if abs(price - lvl) / price <= self.price_tolerance:
                    df_1h.at[df_1h.index[i], 'signal'] = True
                    break

        return df_1h

        
        


        
            
        

        
