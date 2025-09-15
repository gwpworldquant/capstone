import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Backtester:
    def __init__(self, df_1h_csp: pd.DataFrame, df_regime: pd.DataFrame, stoploss: float = 50):
        self.df_1h_csp = df_1h_csp.copy()
        self.df_regime = df_regime.copy()
        self.stoploss = stoploss
        self.df = None
        self.cumulative_return = None
        self.trades = []  # store trade details (entry, exit, pnl, duration)

    def merge_data(self):
        """Synchronize time between 1H data and 4H regime data"""
        self.df = pd.merge_asof(
            self.df_1h_csp.sort_values("time"),
            self.df_regime[["regime_labels"]].sort_index(),
            left_on="time", right_index=True, direction="backward"
        )
        return self.df

    def run_backtest(self):
        """Run the backtest based on entry/exit logic"""
        if self.df is None:
            self.merge_data()

        in_position = False
        entry_price = None
        entry_time = None
        positions, buy_signal, exit_signal = [], [], []

        for _, row in self.df.iterrows():
            if not in_position and row["signal"] == "BUY":
                in_position = True
                entry_price = row["close"]
                entry_time = row["time"]
                positions.append(1)
                buy_signal.append(True)
                exit_signal.append(False)

            elif in_position:
                exit_flag = False
                # Exit if Bear regime
                if row["regime_labels"] == "Bear - decrease":
                    exit_flag = True
                # Exit if stoploss is hit
                elif row["close"] <= entry_price - self.stoploss:
                    exit_flag = True

                if exit_flag:
                    in_position = False
                    exit_price = row["close"]
                    exit_time = row["time"]

                    # store trade
                    self.trades.append({
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": (exit_price - entry_price),
                        "return_pct": (exit_price - entry_price) / entry_price,
                        "duration": (exit_time - entry_time).total_seconds() / 3600  # in hours
                    })

                    entry_price, entry_time = None, None
                    positions.append(0)
                    buy_signal.append(False)
                    exit_signal.append(True)
                else:
                    positions.append(1)
                    buy_signal.append(False)
                    exit_signal.append(False)
            else:
                positions.append(0)
                buy_signal.append(False)
                exit_signal.append(False)

        self.df["position"] = positions
        self.df["buy_signal"] = buy_signal
        self.df["exit_signal"] = exit_signal

        # Strategy returns
        self.df["return"] = self.df["close"].pct_change()
        self.df["strategy_return"] = self.df["return"] * self.df["position"]
        self.cumulative_return = (1 + self.df["strategy_return"]).cumprod()
        return self.df

    def total_return(self):
        """Total return of the strategy"""
        if self.cumulative_return is None:
            raise ValueError("You need to run run_backtest() first.")
        return self.cumulative_return.iloc[-1]

    def sharpe_ratio(self, risk_free_rate: float = 0.0, annualize: bool = True):
        """Calculate Sharpe Ratio"""
        if self.df is None or "strategy_return" not in self.df:
            raise ValueError("You need to run run_backtest() first.")

        excess_return = self.df["strategy_return"] - risk_free_rate
        mean_return = excess_return.mean()
        std_return = excess_return.std()
        if std_return == 0:
            return np.nan

        sharpe = mean_return / std_return
        if annualize:
            sharpe *= np.sqrt(252 * 24)  # assuming 1H data = 6048 candles/year
        return sharpe

    def return_to_risk(self):
        """Return-to-Risk Ratio = Total Return / Max Drawdown"""
        if self.cumulative_return is None:
            raise ValueError("You need to run run_backtest() first.")

        running_max = self.cumulative_return.cummax()
        drawdown = (self.cumulative_return - running_max) / running_max
        max_dd = drawdown.min()
        if max_dd == 0:
            return np.nan
        return (self.cumulative_return.iloc[-1] - 1) / abs(max_dd)

    def win_rate(self):
        """Win Rate (%)"""
        if not self.trades:
            raise ValueError("No trades recorded, run run_backtest() first.")
        wins = [t for t in self.trades if t["pnl"] > 0]
        return len(wins) / len(self.trades) * 100

    def avg_trade_duration(self):
        """Average trade duration (in hours)"""
        if not self.trades:
            raise ValueError("No trades recorded, run run_backtest() first.")
        return np.mean([t["duration"] for t in self.trades])

    def plot_results(self):
        """Plot Price + Signals and Cumulative Return"""
        if self.df is None or self.cumulative_return is None:
            raise ValueError("You need to run run_backtest() first.")

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 10), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}
        )

        # 1️⃣ Price chart + EMA
        ax1.plot(self.df["time"], self.df["close"], label="Close", color="black", alpha=0.6)
        if "EMA34" in self.df:
            ax1.plot(self.df["time"], self.df["EMA34"], label="EMA34", color="blue", linewidth=1.2)
        if "EMA55" in self.df:
            ax1.plot(self.df["time"], self.df["EMA55"], label="EMA55", color="orange", linewidth=1.2)
        if "EMA84" in self.df:
            ax1.plot(self.df["time"], self.df["EMA84"], label="EMA84", color="purple", linewidth=1.2)

        # Buy signals
        ax1.scatter(self.df.loc[self.df["buy_signal"], "time"],
                    self.df.loc[self.df["buy_signal"], "close"],
                    marker="^", color="green", s=100, label="Buy")
        # Exit signals
        ax1.scatter(self.df.loc[self.df["exit_signal"], "time"],
                    self.df.loc[self.df["exit_signal"], "close"],
                    marker="v", color="red", s=100, label="Exit")

        ax1.set_title("Price + EMA + Buy/Exit Signals")
        ax1.legend()

        # 2️⃣ Strategy cumulative return
        ax2.plot(self.df["time"], self.cumulative_return, label="Strategy Return", color="green")
        ax2.axhline(1, color="gray", linestyle="--", linewidth=1)

        stats = (
            f"Sharpe: {self.sharpe_ratio():.2f} | "
            f"Return-to-Risk: {self.return_to_risk():.2f} | "
            f"Win Rate: {self.win_rate():.1f}% | "
            f"Avg Duration: {self.avg_trade_duration():.1f}h"
        )
        ax2.set_title(f"Cumulative Strategy Return | {stats}")
        ax2.legend()

        plt.tight_layout()
        plt.show()
