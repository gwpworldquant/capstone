import pandas as pd
import numpy as np


class Coiled_Spring_MA_Convergence:
    def __int__(self):
        pass

    def calculate_MA_fibo(self, 
                          df:pd.DataFrame):
        ema_periods = [34, 55, 84]
         # Calculate EMAs
        for p in ema_periods:
            df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
        return df
    
    def generate_signal(self, row):
        if row['above_all_ema'] and (1 <= row['min_gap_to_ema'] <= 5) and row['ema_intersection'] and  row['volume_spike']:
            return 'buy'
        elif row['below_all_ema'] and (1 <= row['min_gap_to_ema'] <= 5) and row['ema_intersection'] and row['volume_spike']:
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

        # volumne ratio:
        df['volume_spike'] = ((df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)) > 0.3

        # Create signal column
        df['signal'] = 0
        df['position'] = df.apply(self.generate_signal, axis=1)


        return df


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


        
        


        
            
        

        
