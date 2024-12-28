import numpy as np
from agent import BitcoinTradingAgent
from data_generator import generate_test_data


def train_agent(agent, num_episodes=1000, batch_size=32):
    results = []
    test_data = generate_test_data(volatility=0.1)

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        balance = 1_000_000
        bitcoins = 0
        for data in test_data:
            state = np.array(data).reshape((1, 7, 4))
            action, confidence = agent.act(state)
            open_price = data[-1][0]

            if action == 0:  # Buy
                investment = balance * agent.max_investment
                bitcoins_bought = investment / open_price
                bitcoins += bitcoins_bought
                balance -= investment
            elif action == 1 and bitcoins > 0:  # Sell
                proceeds = bitcoins * open_price
                balance += proceeds
                bitcoins = 0

            current_investment_value = bitcoins * open_price
            total_balance = balance + current_investment_value
            agent.remember(state, action, total_balance - balance, state, False)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            results.append((episode, total_balance))

    return results
