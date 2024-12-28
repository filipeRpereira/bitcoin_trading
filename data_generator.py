import numpy as np


def generate_test_data(volatility=0.1, num_tests=10, num_days=7, base_price_range=(50000, 100000)):
    test_data = []
    for _ in range(num_tests):
        daily_prices = []
        last_price = np.random.uniform(*base_price_range)

        for _ in range(num_days):
            open_price = last_price * (1 + np.random.uniform(-volatility, volatility))
            close_price = open_price * (1 + np.random.uniform(-volatility, volatility))
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, volatility))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, volatility))

            daily_prices.append([open_price, close_price, high_price, low_price])
            last_price = close_price
        test_data.append(daily_prices)
    return test_data
