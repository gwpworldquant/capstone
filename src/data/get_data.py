
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime


# Get data from MT
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def fetch_mt5_data_between_dates(symbol: str = "EURUSD", 
                                 timeframe=mt5.TIMEFRAME_M1, 
                                 from_date: datetime = None,
                                 to_date: datetime = None) -> pd.DataFrame:
    """
    Fetch historical candlestick data from MetaTrader 5 between two dates.

    Parameters:
        symbol (str): Symbol to fetch (e.g., 'EURUSD')
        timeframe: MT5 timeframe (e.g., mt5.TIMEFRAME_M1)
        from_date (datetime): Start date
        to_date (datetime): End date

    Returns:
        pd.DataFrame: DataFrame containing OHLCV data with datetime index
    """
    from datetime import datetime, timedelta

    if from_date is None or to_date is None:
        print("Please provide both from_date and to_date.")
        return pd.DataFrame()

    print(f"Fetching {symbol} from {from_date} to {to_date}")

    # Initialize MT5
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return pd.DataFrame()

    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select symbol {symbol}, error:", mt5.last_error())
        mt5.shutdown()
        return pd.DataFrame()

    # Fetch data
    rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)

    # Shutdown MT5 connection
    mt5.shutdown()

    # Return DataFrame
    if rates is None or len(rates) == 0:
        print("No data received.")
        return pd.DataFrame()

    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.columns = data.columns.str.lower()

    return data
