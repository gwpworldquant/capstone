from typing import Optional
import numpy as np
import gymnasium as gym
import duckdb
from datetime import timedelta
from collections import defaultdict
from tqdm import tqdm  # Progress bar
import pandas as pd

# THIS NEW VERSION SHOULD:
    # Input in the observation space is normalized with Z-score
    # Implement a simple trading system
        # -> Mkt order + ProfitT + StopL
        # -> Close last operation if it didn't profit in the time window
    # Implement rewards as 1,0, or -1
    # Implement DQN

# ----------------------------#
#         ENVIRONMENT         #
# ----------------------------#

class NaiveOtis(gym.Env):
    
    def __init__(self):
        super().__init__()
        
        # Observation space: 7 Z-score normalized values
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(low=-100, high=100, shape=(7,), dtype=np.float32)
        })
        
        # Now we define the action space
        self.action_space = gym.spaces.Discrete(3)
        
        # We setup the basic values
        self.state = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # We define some parameters of the trading strategy
        # We assume we enter with a market order and exit at PT or ST
        self.profit_point = 10
        self.stop_point = 8
        
        # Other parameters
        self.fees = 2 # Cents per stock, assuming trades of 100 stocks
        self.n_minutes = 30 # Minutes we check if the trade closed

    def _get_obs(self):
        """Transforms internal state to observation format
          
        Returns:
            array: Returns of last five 1min candles
        """
        return {"state": self.state.copy()}
    
    def _get_info(self):
        """Compute auxiliary information for debugging.
    
        Returns:
            dict: For testing, returns the observations again
        """
        return {"state": self.state.copy()}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode
        
        Args:
            options: Additional information (e.g., episode number)
            e.g., env.reset(options={"episode": 5})
            
        Returns:
            array: Observations
        """
        super().reset(seed=seed)
        
        # Connect to the DuckDB file
        self.conn = duckdb.connect("data_rl.duckdb")  # optionally move this outside for reuse
        
        # Define the current state based on which episode we are in
        self.episode = options.get("episode", 0) if options else 0
        
        # Load a single row from DuckDB
        query = "SELECT * FROM data LIMIT 1 OFFSET ?"
        row = self.conn.execute(query, [self.episode]).fetchone()

        # Convert to float32 NumPy array
        self.state = np.array(row, dtype=np.float32)
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        
        self.action = action
        reward = 0
        
        # 1. Apply the action
        # 2. Advance the environment
        # 3. Compute reward
        # 4. Check if episode is done
        # 5. Return all required outputs
        
        # This is a simple step method where the agent decides to buy, sell, or
        # do nothing. Based on the episode, we extract the time of the observation
        # and the price of the stock for the next n_minutes. Then, the setup is
        # to enter the trade with a market order and take profits profit_point
        # cents away from the current price. There is also a stop loss
        # set at stop_point cents away (-/+) from the current price.
        
        # We extract the time in which we are making the observation
        query = "SELECT ts_event FROM time LIMIT 1 OFFSET ?"
        tiempo = self.conn.execute(query, [self.episode]).fetchone()
        tiempo = tiempo[0]

        # We extract the traded prices for the security for the given time range
        start = tiempo + timedelta(minutes=1)
        end   = start + timedelta(minutes=self.n_minutes)

        query = f"""
        SELECT price
        FROM price_trades
        WHERE ts_event >= TIMESTAMPTZ '{start}'
          AND ts_event <  TIMESTAMPTZ '{end}'
        """
        price_list = self.conn.execute(query).fetchdf()

        # Current price at the time of observation
        # (first price right after the minute is over)
        # We assume that we entry at this point -> Mkt order
        current_price = price_list["price"].iloc[0]

        # Based on the action taken, we compute the price levels
        # 0 = no trade, 1 = go long 100 shares, 2 = go short 100 shares
        if action == 1:
            entry_price = round(current_price, 2) # Market order
            profit_price = round(current_price + self.profit_point/100, 2)
            stop_price = round(current_price - self.stop_point/100, 2)
        elif action == 2:
            entry_price = round(current_price, 2) # Market order
            profit_price = round(current_price - self.profit_point/100, 2)
            stop_price = round(current_price + self.stop_point/100, 2)
            
        # We compute the reward of the trade
        # We assume that the order got filled at the entry price (market order),
        # but if it did not reach the level to take profits, the exit point will
        # be either the StopLoss limit or the final price at the end of the time window

        # Boolean identifiers
        profited = False
        stopped = False

        if action == 0:
            reward = 0

        if action == 1:
            for idx, level_price in price_list["price"].items():
                if level_price >= profit_price:
                    profited = True
                    break
                elif level_price <= stop_price:
                    stopped = True
                    break
            # We compute rewards
            if profited:
                reward = 1
            elif stopped:
                reward = -1
            else:
                reward = -1
                print(f"Order got filled but didn't exit, episode: {self.episode}!")
                
        if action == 2:
            for idx, level_price in price_list["price"].items():
                    if level_price <= profit_price:
                        profited = True
                        break
                    elif level_price >= stop_price:
                        stopped = True
                        break
            # We compute rewards
            if profited:
                reward = 1
            elif stopped:
                reward = -1
            else:
                reward = -1
                print(f"Order got filled but didn't exit, episode: {self.episode}!")
                
        # We terminate the episode here, because we only account for onw step
        terminated = True
        
        # We just add truncated by default
        truncated =  False
        
        # We do not update the state because there is only one step
        observation = self._get_obs()
        
        # We return the information by default
        info = self._get_info()
        
        # We close the connection because it is a one step model
        self.conn.close()
        
        return observation, reward, terminated, truncated, info
        
# # We create the environment to test it
# env = NaiveOtis()
# from gymnasium.utils.env_checker import check_env

# # This will catch many common issues
# try:
#     check_env(env)
#     print("Environment passes all checks!")
# except Exception as e:
#     print(f"Environment has issues: {e}")    
    

# ----------------------#
#         AGENT         #
# ----------------------#  
    
class AgentOtis:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        
        """Initialize a Q-Learning agent.
    
        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env
        
        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        
        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []
        
    def get_action(self, obs: dict[str, np.ndarray]) -> int:
        """Choose an action using epsilon-greedy strategy.
    
        Returns:
            action: 0 = no trade, 1 = go long 100 shares, 2 = go short 100 shares
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
    
        # With probability (1-epsilon): exploit (best known action)
        else:
            obs_key = tuple(obs["state"])
            return int(np.argmax(self.q_values[obs_key]))

    def update(
        self,
        obs: dict[str, np.ndarray],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: dict[str, np.ndarray],
    ):
        """Q-learning update for single-step episode.
    
        Since there's no future step, we only update toward the immediate reward.
        """
    
        # How wrong was our current estimate?
        # We use the reward only, no bootstrapping -> one step environment
        obs_key = tuple(obs["state"])
        temporal_difference = reward - self.q_values[obs_key][action]
        
        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs_key][action] += self.lr * temporal_difference
        
        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        

# --------------------------#
#          TRAINING         #
# --------------------------# 

# We connect to the database
conn = duckdb.connect("data_rl.duckdb")
n_episodes = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
conn.close()
        
# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
#n_episodes = 100_000        # Number of hands to practice (setup below)
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration
     
# Create environment and agent
env = NaiveOtis()
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = AgentOtis(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)    

# Training loop
for episode in tqdm(range(n_episodes)):
    
    # Start a new episode
    obs, info = env.reset(options={"episode": episode})

    # Agent chooses action (initially random, gradually more intelligent)
    action = agent.get_action(obs)

    # Take action and observe result
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # Learn from this experience
    agent.update(obs, action, reward, terminated, next_obs)

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon() 
    
    
# # Convert to DataFrame
# q_table_df = pd.DataFrame([
#     {"state": state, "action_0": q[0], "action_1": q[1], "action_2": q[2]}
#     for state, q in agent.q_values.items()
# ])

# print(q_table_df.head())        
        
          

        