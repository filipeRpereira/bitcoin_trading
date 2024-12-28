import matplotlib.pyplot as plt


def plot_performance(results):
    balances, total_balances = zip(*results)
    plt.plot(total_balances, label='Total Balance')
    plt.xlabel('Days')
    plt.ylabel('Balance ($)')
    plt.title('Bitcoin Trading Performance')
    plt.legend()
    plt.show()


def calculate_performance(results):
    total_profit = results[-1][1] - results[0][1]
    success_rate = len([x for x in results if x[1] > 0]) / len(results)
    return total_profit, success_rate
