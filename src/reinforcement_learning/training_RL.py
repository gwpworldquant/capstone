import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# We get started by defining the environment
class TradingEnv(gym.Env):
    # To generate human readable outputs
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 X_seq,                 
                 next_returns,          
                 y_signal=None,         
                 prices=None,           
                 transaction_cost=0.0005,
                 include_signal_in_state=True,
                 max_episode_steps=None):
        super().__init__()

        assert X_seq.ndim == 3, "X_seq must be shape (N, window, n_features)"
        self.X_seq = X_seq.astype(np.float32)
        self.next_returns = next_returns.astype(np.float32)
        self.y_signal = y_signal
        self.prices = prices
        self.tc = float(transaction_cost)
        self.include_signal_in_state = include_signal_in_state

        self.N = self.X_seq.shape[0]
        self.window = self.X_seq.shape[1]
        self.n_features = self.X_seq.shape[2]
        # Observations will be flattened vectors
        obs_dim = self.window * self.n_features + (1 if include_signal_in_state else 0)
        # Large bounds for Box
        self.observation_space = spaces.Box(low=-np.finfo(np.float32).max,
                                            high=np.finfo(np.float32).max,
                                            shape=(obs_dim,),
                                            dtype=np.float32)
        # actions: 0 short (-1), 1 flat (0), 2 long (+1)
        self.action_space = spaces.Discrete(3)

        # internal pointers
        self.idx = 0
        self.prev_position = 0  # -1,0,1
        self.episode_step = 0
        self.max_episode_steps = max_episode_steps or (self.N - 1)
    
    def reset(self, *, seed=None, options=None):
        # Seeding
        super().reset(seed=seed)
    
        self.idx = 0
        self.prev_position = 0
        self.episode_step = 0
        obs = self._get_obs(self.idx)
        info = {}
        return obs, info

    def step(self, action):
        position = self._action_to_position(action)
        # reward uses the next_returns aligned at current index
        r = (position * self.next_returns[self.idx] + self.tc * abs(position - self.prev_position))
        self.prev_position = position
        self.idx += 1
        self.episode_step += 1
        done = (self.idx >= self.N - 1) or (self.episode_step >= self.max_episode_steps)

        obs = self._get_obs(self.idx) if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"index": int(self.idx), "position": position}
        truncated = False
        return obs, float(r), bool(done), truncated, info

    def _get_obs(self, idx):
        # returns flattened X_seq[idx]
        x = self.X_seq[idx].reshape(-1).astype(np.float32)
        if self.include_signal_in_state:
            sig_val = np.array([self.y_signal[idx]], dtype=np.float32) if self.y_signal is not None else np.array([0.0], dtype=np.float32)
            x = np.concatenate([x, sig_val], axis=0)
        return x

    def render(self, mode="human"):
        pass

    @staticmethod
    def _action_to_position(action):
        if int(action) == 0:
            return -1
        elif int(action) == 1:
            return 0
        elif int(action) == 2:
            return 1
        else:
            raise ValueError("Invalid action")

    def get_episode_returns(self, actions):
        # Map to positions
        positions = np.array([self._action_to_position(a) for a in actions], dtype=np.float32)
        prev_pos = 0
        rewards = []
        for i, pos in enumerate(positions):
            # align with next_returns[i]
            r = pos * self.next_returns[i] - self.tc * abs(pos - prev_pos)
            rewards.append(r)
            prev_pos = pos
        rewards = np.array(rewards, dtype=np.float32)
        cum_log_returns = np.cumsum(rewards)
        return rewards, positions, cum_log_returns


# Training and backtesting functions
def train_agent(train_npz_path,
                output_dir,
                total_timesteps=100_000,
                transaction_cost=0.0005,
                include_signal_in_state=True,
                policy_kwargs=None,
                seed=42):

    data = np.load(train_npz_path, allow_pickle=True)
    X = data["X"]
    next_return = data["next_return"]
    y_signal = data["y_signal"] if "y_signal" in data else None
    times = data["times"] if "times" in data else None
    prices = data["prices"] if "prices" in data else None

    # Append signal to the state
    env = TradingEnv(X_seq=X,
                     next_returns=next_return,
                     y_signal=y_signal,
                     prices=prices,
                     transaction_cost=transaction_cost,
                     include_signal_in_state=include_signal_in_state)
    # Wrap for SB3
    vec_env = DummyVecEnv([lambda: env])

    # Choose policy kwargs
    if policy_kwargs is None:
        policy_kwargs = dict(net_arch=[dict(pi=[256, 128], vf=[256, 128])])

    model = PPO("MlpPolicy",
                vec_env,
                verbose=1,
                seed=seed,
                policy_kwargs=policy_kwargs,
                batch_size=64,
                n_steps=2048,
                learning_rate=3e-4,
                gae_lambda=0.95)

    print("Starting training PPO for", total_timesteps, "timesteps...")
    model.learn(total_timesteps=total_timesteps)
    # Save model and env metadata
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"ppo_trading_{ts}.zip")
    model.save(model_path)
    print("Saved model to:", model_path)

    # Save metadata
    meta = {
        "model_path": model_path,
        "include_signal_in_state": include_signal_in_state,
        "transaction_cost": transaction_cost,
        "train_npz": train_npz_path
    }
    joblib.dump(meta, os.path.join(output_dir, f"meta_{ts}.joblib"))

    return model, env, meta

def backtest_agent(model, env, test_npz_path, output_dir, transaction_cost=0.0005):

    # Run the trained model on the test set and compare vs buy-and-hold
    data = np.load(test_npz_path, allow_pickle=True)
    X_test = data["X"]
    next_return_test = data["next_return"]
    y_signal_test = data["y_signal"] if "y_signal" in data else None
    times_test = data["times"] if "times" in data else None
    prices_test = data["prices"] if "prices" in data else None

    # Build test env consistent with training include_signal flag
    test_env = TradingEnv(X_seq=X_test,
                          next_returns=next_return_test,
                          y_signal=y_signal_test,
                          prices=prices_test,
                          transaction_cost=transaction_cost,
                          include_signal_in_state=env.include_signal_in_state)

    # Run the policy step by step
    obs, _ = test_env.reset()
    actions = []
    rewards = []
    positions = []
    done = False
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, truncated, info = test_env.step(action)
        # Break after collecting reward
        actions.append(int(action))
        rewards.append(float(r))
        positions.append(info.get("position", 0))
        if done:
            break

    actions = np.array(actions, dtype=int)
    rewards = np.array(rewards, dtype=float)
    positions = np.array(positions, dtype=float)
    cum_log_returns_agent = np.cumsum(rewards)
    wealth_agent = np.exp(cum_log_returns_agent)  # starting wealth = 1

    # Buy-and-hold: always long (position=1)
    # Align buy-and-hold returns with next_return_test[0:len(rewards)]
    br = next_return_test[:len(rewards)].astype(float)
    # No transaction costs for buy-and-hold after initial buy (we'll subtract an initial cost for first entry)
    initial_tc = transaction_cost * 1.0  # cost to enter long at start
    rewards_buy_and_hold = br.copy()
    rewards_buy_and_hold[0] = rewards_buy_and_hold[0] - initial_tc
    cum_log_returns_bh = np.cumsum(rewards_buy_and_hold)
    wealth_bh = np.exp(cum_log_returns_bh)

    # Also compute a 'signal benchmark' if y_signal available to trade according to dip tip signal
    wealth_signal = None
    cum_log_signal = None
    if y_signal_test is not None:
        # y_signal_test is aligned to last step of the sequence; we trade according to it
        sig = y_signal_test[:len(rewards)].astype(int)
        positions_sig = np.where(sig == 1, 1.0, 0.0)  # change if your signal also implies shorting
        prev_pos = 0.0
        rew_sig = []
        for i, pos in enumerate(positions_sig):
            r_sig = pos * next_return_test[i] - transaction_cost * abs(pos - prev_pos)
            rew_sig.append(r_sig)
            prev_pos = pos
        rew_sig = np.array(rew_sig, dtype=float)
        cum_log_signal = np.cumsum(rew_sig)
        wealth_signal = np.exp(cum_log_signal)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez_compressed(os.path.join(output_dir, f"backtest_{ts}.npz"),
                        actions=actions,
                        rewards=rewards,
                        positions=positions,
                        cum_log_returns_agent=cum_log_returns_agent,
                        wealth_agent=wealth_agent,
                        cum_log_returns_bh=cum_log_returns_bh,
                        wealth_bh=wealth_bh,
                        cum_log_signal=cum_log_signal,
                        wealth_signal=wealth_signal,
                        times=times_test[:len(rewards)] if times_test is not None else None,
                        prices=prices_test[:len(rewards)] if prices_test is not None else None)

    # Plot return curves
    wealth_signal = None
    plt.figure(figsize=(10, 6))
    plt.plot(wealth_agent, label="RL Agent")
    plt.plot(wealth_bh, label="Buy & Hold")
    if wealth_signal is not None:
        plt.plot(wealth_signal, label="Signal benchmark")
    plt.title("Backtest: Return")
    plt.xlabel("Step")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, f"backtest_plot_{ts}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Summaries
    def summary_from_cum_log(cum_log):
        final_log = float(cum_log[-1])
        final_return = np.exp(final_log) - 1.0
        # approximate annualization depends on data frequency
        return {"final_return": final_return, "final_wealth": float(np.exp(final_log)), "final_log_return": final_log}

    res = {
        "agent": summary_from_cum_log(cum_log_returns_agent) if len(rewards) > 0 else None,
        "buy_and_hold": summary_from_cum_log(cum_log_returns_bh) if len(rewards) > 0 else None,
        "signal": summary_from_cum_log(cum_log_signal) if (cum_log_signal is not None and len(cum_log_signal) > 0) else None,
        "plot": plot_path,
        "saved_npz": os.path.join(output_dir, f"backtest_{ts}.npz")
    }

    print("Backtest saved to:", res["saved_npz"])
    print("Wealth plot saved to:", plot_path)
    print("Agent final wealth:", res["agent"])
    print("Buy & Hold final wealth:", res["buy_and_hold"])
    if res["signal"] is not None:
        print("Signal benchmark final wealth:", res["signal"])

    return res

# Execution
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_npz", type=str, default="prepared_data/train_dip_tip.npz", help="Path to train npz produced by prepare_rl_data.py")
    p.add_argument("--test_npz", type=str, default="prepared_data/test_dip_tip.npz", help="Path to test npz produced by prepare_rl_data.py")
    p.add_argument("--output_dir", type=str, default="results", help="Where to save model and backtest outputs")
    p.add_argument("--total_timesteps", type=int, default=100_000, help="Total timesteps for RL training (increase for better fit)")
    p.add_argument("--transaction_cost", type=float, default=0.0005, help="Transaction cost per position change")
    p.add_argument("--include_signal_in_state", action="store_true", help="If set, include the signal value in the observation (useful if you want it)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()

    print("Loading and training...")
    model, env, meta = train_agent(train_npz_path=args.train_npz,
                                   output_dir=args.output_dir,
                                   total_timesteps=args.total_timesteps,
                                   transaction_cost=args.transaction_cost,
                                   include_signal_in_state=args.include_signal_in_state,
                                   seed=args.seed)

    print("Backtesting on test set...")
    res = backtest_agent(model, env, test_npz_path=args.test_npz, output_dir=args.output_dir, transaction_cost=args.transaction_cost)
    print("Results:", res)

if __name__ == "__main__":
    main()
