import gymnasium as gym
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

#########
# Data Preprocessing
#########
# def prepare_data(ticker="TRENT.NS", start="2015-01-01", end="2025-03-03"):
def prepare_data(ticker="ACI.NS", start="2015-01-01", end="2025-03-08"):
# def prepare_data(ticker="PRAJIND.NS", start="2015-01-01", end="2025-03-03"):
    """Fetch and process OHLCV data (add technical indicators later?)"""
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = df.columns.droplevel(1)

    df['Price_MA_5'] = df['Close'].rolling(window=5).mean()
    df['Price_MA_20'] = df['Close'].rolling(window=20).mean()
    df['Trend'] = np.where(df['Price_MA_5'] > df['Price_MA_20'], 1, -1)

    df.dropna(inplace=True)
    return df

def split_and_scale_data(data, train_split=0.8):
    """Split data in test/train; scale features"""
    split_idx = int(len(data) * train_split)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()

    train_close = train_df['Close'].values.reshape(-1, 1)
    test_close = test_df['Close'].values.reshape(-1, 1)

    #scaling features
    # feature_cols = [col for col in train_df.columns if col != 'Close']
    # print(train_df)
    # print(feature_cols)
    feature_cols = [col for col in train_df.columns if col not in ['Close', 'Trend']]
    # print(feature_cols)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.fit_transform(test_df[feature_cols])
    # print(train_scaled)

    #### Commenting adding of Trend
    train_features_scaled_df = pd.DataFrame(train_scaled, columns=feature_cols, index=train_df.index)
    test_features_scaled_df = pd.DataFrame(test_scaled, columns=feature_cols, index=test_df.index)

    # print(train_features_scaled_df)
    # print(train_df)           # have proper trend here

    train_features_scaled_df['Trend'] = train_df['Trend']
    test_features_scaled_df['Trend'] = test_df['Trend']

    train_scaled = train_features_scaled_df.values
    test_scaled = test_features_scaled_df.values

    # print(train_features_scaled_df)
    # print(train_scaled)

    return train_scaled, test_scaled, train_close, test_close, scaler


############
# Setting up Trading Environment
############
class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, scaled_data, actual_prices, window_size=30, initial_balance=1_00_000,
                    transaction_cost=0.001, slippage=0.002, max_steps=None):
        super().__init__()

        self.scaled_data = scaled_data
        self.actual_prices = actual_prices.flatten()
        self.window_size = window_size
        self.feature_dim = scaled_data.shape[1]
        self.max_steps = max_steps or len(scaled_data) - window_size

        self.initial_balance = float(initial_balance)
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.min_shares = 1

        self.action_space = gym.spaces.Discrete(3) #0=sell; 1=hold; 2=buy
        self.observation_space = gym.spaces.Box(
                # low = 0,
                low = -1,
                high = 1,
                shape = (window_size * self.feature_dim,),
                dtype=np.float32
        )

        ## lists to track Actions and Rewards
        self.debug_actions = []
        self.debug_rewards = []

        ##Differential Sharpe Ratio tracking
        self.prev_avg_return = 0
        self.prev_avg_sq_return = 0
        self.sharpe_decay = 0.95

        self.reset_vars()

    def reset_vars(self):
        self.current_step = self.window_size
        self.episode_steps = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance

        self.debug_actions = []
        self.debug_rewards = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_vars()
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.scaled_data[
            self.current_step - self.window_size:self.current_step
        ].flatten().astype(np.float32)
        return obs

    def _execute_trade(self, action):
        """Execute trade w slippage and transaction costs"""
        current_price = float(self.actual_prices[self.current_step])
        # slippage being both positive and negative, market can go either side at execution!?
        slippage_amount = current_price * self.slippage * np.random.normal(0, 0.5)

        # risk_per_trade = 0.8
        risk_per_trade = 0.10
        risk_per_trade_buy = 0.20
        available = self.balance * risk_per_trade_buy

        if action == 2:        # BUY
            executed_price = current_price + slippage_amount
            max_shares = int(available // executed_price)
            if max_shares >= self.min_shares:
                cost = max_shares * executed_price
                self.balance -= cost * (1 + self.transaction_cost)
                self.shares_held += max_shares

        elif action == 0 and self.shares_held > 0:      # SELL
            ##### Sells 10% of shares held, will need to make number of shares sold more dynamic?
            executed_price = current_price - slippage_amount
            sell_qty = int(self.shares_held * risk_per_trade)
            if (self.shares_held * risk_per_trade) > 0 and (self.shares_held * risk_per_trade) < 1:
                sell_qty = self.shares_held

            if sell_qty > 0:
                revenue = sell_qty * executed_price
                self.balance += revenue * (1 - self.transaction_cost)
                self.shares_held -= sell_qty

        self.portfolio_value = self.balance + self.shares_held * current_price

    def step(self, action):
        """Implementing Differential Sharpe Ratio as a reward fn"""
        self.debug_actions.append(action)

        #### Detter the agent from spamming sell initially
        reward = 0
        if action==0 and self.shares_held <=0:
            # print("here")
            # reward = -1
            # reward = -500
            reward = -750 * 1

        prev_value = self.portfolio_value
        prev_shares_held = self.shares_held
        self._execute_trade(action)

        self.current_step += 1
        self.episode_steps += 1

        current_price = float(self.actual_prices[self.current_step-1])
        current_price = float(self.actual_prices[self.current_step])        # current price is actually at current step!?

        return_pct = (self.portfolio_value - prev_value) / prev_value if prev_value > 0 else 0


        day_normalised_rfr = 0.05/252
        excess_return = return_pct - day_normalised_rfr

        new_avg_return = (self.sharpe_decay * self.prev_avg_return + (1- self.sharpe_decay) * excess_return)
        new_avg_sq_return = (self.sharpe_decay * self.prev_avg_sq_return + (1- self.sharpe_decay) * excess_return**2)

        # print(f"new_avg_return: {new_avg_return} | new_avg_sq_return: {new_avg_sq_return}")

        variance = max(new_avg_sq_return - new_avg_return**2, 1e-6)
        variance_prev = max(self.prev_avg_sq_return - self.prev_avg_return**2, 1e-6)
        std_dev_prev = np.sqrt(variance_prev)

        delta_avg_return = new_avg_return - self.prev_avg_return
        delta_avg_sq_return = new_avg_sq_return - self.prev_avg_sq_return


        ### DSR Calculation [final]
        term1 = (excess_return - self.prev_avg_return) / variance_prev
        term2_numerator = self.prev_avg_return * (excess_return**2 - self.prev_avg_sq_return)
        term2 = 0.5 * term2_numerator / (std_dev_prev**3 + 1e-6)
        sharpe_contribution = term1 - term2

        self.prev_avg_return = new_avg_return
        self.prev_avg_sq_return = new_avg_sq_return

        if reward == 0:
            reward = sharpe_contribution


        ### Added positional reward for encouraging hodling????
        if self.shares_held > 0:
            equity_ratio = (self.shares_held * current_price) / self.portfolio_value
            position_reward = equity_ratio * 10
            reward += position_reward

        if action == 2:
            reward += 7


        # print(f"Action: {action} | Reward: {reward}\n")

        # store reward
        self.debug_rewards.append(reward)

        # Termination Conditions
        terminated = self.portfolio_value <= self.initial_balance * 0.8
        truncated = (self.current_step >= len(self.scaled_data)-1) or \
                    (self.episode_steps >= self.max_steps)


        info = {
            "portfolio_value": self.portfolio_value,
            "return": return_pct,
            "action": action
        }

        return self._get_observation(), reward, terminated, truncated, info

##########
# Train and Eval
##########
if __name__ == "__main__":
    full_data = prepare_data()
    train_scaled, test_scaled, train_close, test_close, _ = split_and_scale_data(full_data)

    # Env setup
    max_episode_length = 252    # 1 trading year but explore more (2-3 year periods?)
    train_env = TradingEnv(
        train_scaled, train_close,
        window_size=30,
        max_steps=max_episode_length
    )
    test_env = TradingEnv(
        test_scaled, test_close,
        window_size=30,
        max_steps=len(test_scaled)-30
    )
    check_env(train_env)

    # DQN Model
    # model = DQN(
    #     "MlpPolicy",
    #     train_env,
    #     learning_rate=2e-5,
    #     buffer_size=200_000,
    #     learning_starts=15_000,
    #     batch_size=128,
    #     gamma=0.96,
    #     exploration_fraction=0.6,
    #     exploration_final_eps=0.02,
    #     policy_kwargs=dict(
    #         net_arch=[256, 128],
    #         normalize_images=False
    #     ),
    #     verbose=1
    # )

    # model = DQN(
    #     "MlpPolicy",
    #     train_env,
    #     learning_rate=1e-4,  # Slightly lower learning rate
    #     buffer_size=300_000,  # Increased buffer size
    #     learning_starts=20_000,  # Delayed learning start
    #     batch_size=256,  # Increased batch size
    #     gamma=0.99,  # Slightly higher gamma for long-term rewards
    #     exploration_fraction=0.8,  # Increased exploration
    #     exploration_initial_eps=1.0,  # Start with full exploration
    #     exploration_final_eps=0.05,  # Minimum exploration
    #     policy_kwargs=dict(
    #         net_arch=[512, 256],  # Deeper network
    #         normalize_images=False
    #     ),
    #     verbose=1
    # )

    # model = DQN(
    #     "MlpPolicy",
    #     train_env,
    #     learning_rate=3e-4,
    #     buffer_size=500_000,
    #     learning_starts=50_000,
    #     batch_size=512,
    #     tau=0.005,  # Soft update coefficient for target network
    #     gamma=0.95,  # Discount factor
    #     train_freq=4,  # Train every 4 steps
    #     gradient_steps=1,  # Gradient steps per training
    #     exploration_fraction=0.9,  # Longer exploration period
    #     exploration_initial_eps=1.0,  # Start with full randomness
    #     exploration_final_eps=0.05,  # Minimal random actions at end
    #     policy_kwargs=dict(
    #         net_arch=[512, 256, 128],  # Deeper network
    #         normalize_images=False
    #     ),
    #     verbose=1
    # )

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,          # Reduced to allow more stable learning
        buffer_size=1_000_000,       # Increased to remember more market states
        learning_starts=100_000,     # Learn after seeing more data
        batch_size=256,              # Keep moderate batch size
        tau=0.001,                   # Slower target network updates
        gamma=0.99,                  # Higher discount factor for long-term rewards
        train_freq=(4, "step"),      # Train every 4 steps
        gradient_steps=1,
        exploration_fraction=0.5,    # Shorter exploration period
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[512, 512, 256], # Deeper network
            normalize_images=False
        ),
        verbose=1
    )


    # Training
    print("Training...")
    # model.learn(total_timesteps=500_000, progress_bar=True)           #overfit like crazy
    model.learn(total_timesteps=300_000, progress_bar=True)
    # model.learn(total_timesteps=250_000, progress_bar=True)
    # model.learn(total_timesteps=150_000, progress_bar=True)
    # model.learn(total_timesteps=10, progress_bar=True)
    # model.save("trading_dqn_trent")
    # model.save("trading_dqn_aci")
    model.save("trading_dqn_praj2")


    print("\nAction Distribution Debugging:")
    print(f"Total Actions: {len(train_env.debug_actions)}")
    print(f"Buy (2): {train_env.debug_actions.count(2)} times")
    print(f"Hold (1): {train_env.debug_actions.count(1)} times")
    print(f"Sell (0): {train_env.debug_actions.count(0)} times")

    print("\nReward Stats:")
    print(f"Mean Reward: {np.mean(train_env.debug_rewards)}")
    print(f"Reward Range: {min(train_env.debug_rewards)} to {max(train_env.debug_rewards)}")

    # Eval
    print("\nEvaluating...")
    obs, _ = test_env.reset()
    portfolio_history = []
    action_history = []
    observation_history = []

    # Add a way to dump model's internal state
    def print_model_details(model):
        print("\nModel Details:")
        print("Policy Network Architecture:")
        print(model.policy.net_arch if hasattr(model.policy, 'net_arch') else "Not available")

        # Try to access some model internals
        try:
            q_network = model.q_net
            print("\nQ-Network Weights:")
            for name, param in q_network.named_parameters():
                print(f"{name}: {param.shape}")
        except Exception as e:
            print(f"Could not access Q-Network details: {e}")

    # Debug print added before evaluation loop
    print_model_details(model)

    max_eval_steps = len(test_scaled)-30
    current_step = 0

    while current_step < max_eval_steps:
        # print(f"\nStep: {current_step}")
        # print("Observation Shape: ", obs.shape)
        # print("Observation min/max: ", obs.min(), obs.max())

        action, _ = model.predict(obs, deterministic=False)
        # action, _ = model.predict(obs, deterministic=True)

        # print("Predicted Action: ", action)
        obs, reward, terminated, truncated, info = test_env.step(action)

        portfolio_history.append(info["portfolio_value"])
        action_history.append(info["action"])
        observation_history.append(obs)

        current_step += 1

        if terminated or truncated:
            break


    actions = np.array(action_history)
    total_steps = len(actions)
    print(f"Testing Action Distribution ({total_steps} steps):")
    unique, counts = np.unique(actions, return_counts=True)
    action_dist = dict(zip(unique, counts))
    for action, count in action_dist.items():
        print(f"Action: {action}: {count} times ({(count/total_steps)*100:.1f}%)")



    initial_value = portfolio_history[0]
    final_value = portfolio_history[-1]
    total_return = ((final_value - initial_value) / initial_value) * 100

    daily_returns = np.diff(portfolio_history) / portfolio_history[:-1]     # (final/initial)
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns)  * np.sqrt(252) if (len(daily_returns) > 0 and np.std(daily_returns) != 0) else 0         # considering rfr as 0


    # results
    print("Backtest Results: ")
    print(f"Initial Portfolio: Rs. {initial_value:.2f}")
    print(f"Final Portfolio: Rs. {final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    # Graphs
    import matplotlib.pyplot as plt

    # Portfolio Value graph
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(portfolio_history)
    plt.title("Portfolio Value During Testing")
    plt.xlabel("Trading Days")
    plt.ylabel("Value (Rs.)")
    plt.grid(True)

    # Action Taken graph
    plt.subplot(2,1,2)
    plt.hist(actions, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8)
    plt.title("Action Distribution")
    plt.xlabel("Actions (0:Sell, 1: Hold, 2: Buy)")
    plt.ylabel("Frequency")
    plt.xticks([0, 1, 2], ['Sell', 'Hold', 'Buy'])
    plt.tight_layout()

    plt.savefig("graph.png", bbox_inches='tight')
    plt.show()
