import numpy as np
from training import train_agent
from evaluation import plot_performance, calculate_performance
from agent import BitcoinTradingAgent

if __name__ == "__main__":
    state_size = (7, 4)
    action_size = 3  # Buy, Sell, Hold
    agent = BitcoinTradingAgent(state_size, action_size)
    results = train_agent(agent, num_episodes=1000)

    plot_performance(results)
    total_profit, success_rate = calculate_performance(results)
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
