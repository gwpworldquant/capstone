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



class Dip_Tip_MA_Convergence():
    def __int__(self):
        pass

    def calculate_MA_fibo(self, 
                          df:pd.DataFrame):
        ema_periods = [34, 55, 84]
         # Calculate EMAs
        for p in ema_periods:
            df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
        return df
    
    def generate_signal(self, 
                        row,
                        price):
        if row['regime'] == 'bearish' and  row['ema_intersection'] and abs(row['close'] - row['min_ema'])<=price:
            return 'buy'
        elif row['bullish'] == 'bearish' and  row['ema_intersection'] and abs(row['close'] - row['min_ema'])<=price:
            return 'sell'
        else:
            return 0
 

    def signal_MA_convergence(self, 
                                 df:pd.DataFrame):
        
        # # Check if Close is above all EMAs
        df['above_all_ema'] = (df['close'] > df['ema_34']) & (df['close'] > df['ema_55']) & (df['close'] > df['ema_84'])

        # Check if Close is below all EMAs
        df['below_all_ema'] = (df['close'] < df['ema_34']) & (df['close'] < df['ema_55']) & (df['close'] < df['ema_84'])


        df['gap_ema_34'] = abs( df['close'] - df['ema_34'])
        df['gap_ema_55'] = abs(df['close'] - df['ema_55'])
        df['gap_ema_84'] = abs(df['close'] - df['ema_84'])
        df['min_gap_to_ema'] = df[['gap_ema_34', 'gap_ema_55', 'gap_ema_84']].min(axis=1)


        # Calculate absolute gaps between EMAs
        df['gap_34_55'] = abs(df['ema_34'] - df['ema_55'])
        df['gap_34_84'] = abs(df['ema_34'] - df['ema_84'])
        df['gap_55_84'] = abs(df['ema_55'] - df['ema_84'])

        # Find the maximum gap between any two EMAs
        df['max_ema_gap'] = df[['gap_34_55', 'gap_34_84', 'gap_55_84']].max(axis=1)

        # Define "intersection" where all EMAs are very close (gap <= 5)
        df['ema_intersection'] = df['max_ema_gap'] <1

        # find min price
        df['min_ema'] = df[['ema_34, ema_55, ema_84']].min(axis=1)


        # Create signal column
        df['signal'] = 0


        df['position'] = df.apply(self.generate_signal, axis=1)


        return df


        
        


        
            
        

        
