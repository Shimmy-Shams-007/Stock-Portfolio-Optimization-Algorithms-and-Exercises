# This document does not hold any main programs
# This is just a practise document for basic python exercises


import numpy as np
import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# on average there are 252 trading days in a year
NUM_TRADING_DAYS = 252

# we will generate random w (different portfolios)
NUM_PORTFOLIOS = 10000

# stocks we are going to handle
stocks = ['AAPL', 'AMD', 'TSLA', 'GOOG', 'AMZN', 'MSFT']

# historical data - define START and END dates
start_date = '2012-01-01'
end_date = dt.datetime.now()


def download_data():
    # name of the stock (key) - stock values (2012 - Present Date) as the values
    stock_data = {}

    for stock in stocks:
        # closing prices
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(stock_data)


# this is to visualize the stock data
def show_data(data):
    data.plot(figsize=(10, 5))
    plt.show()


# Calculating the return using the logarithmic function
def calculate_data(data):
    log_return = np.log(data / data.shift(1))
    return log_return[1:]  # Starting from the second row because the first row contains invalid values


# Showing the statistics and return values. Shows the mean and covariance of annual returns
def show_statistics(returns):
    print("This shows the annual expected return (mean) from the chosen stocks:\n",
          (returns.mean() * NUM_TRADING_DAYS), '\n')

    print("This shows the correlation (covariance) of the stocks between each other:\n",
          (returns.cov() * NUM_TRADING_DAYS), '\n')


# Shows the mean and variance of the stocks
def show_mean_variance(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))

    print("Expected portfolio mean (return): ", portfolio_return)
    print("Expected portfolio volatility (standard deviation): ", portfolio_volatility)


# Generating 10000 different random portfolios
def generate_portfolio(returns):
    portfolio_means = []
    portfolio_risk = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risk.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risk)


def sharpe_ratio(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))

    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])


# scipy optimize module can find the minimum of a given function
# the maximum of a f(x) is the minimum of -f(x)
def min_function_sharpe(weights, returns):
    return -sharpe_ratio(weights, returns)[2]


def portfolio_optimization(weights, returns):
    # Setting constraints to that the sum of the weights is equals to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)


def print_portfolio_optimization(optimum, returns):
    print("Optimal portfolio: ", optimum['x'].round(3))
    print("Expected return, volatility and Sharpe ratio: ",
          sharpe_ratio(optimum['x'].round(3), returns))


# Visualization of the random portfolios
def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label="Sharpe Ratio")
    plt.show()


# Visualization of the optimal portfolio
def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(sharpe_ratio(opt['x'], rets)[1], sharpe_ratio(opt['x'], rets)[0], 'g*', markersize=20.0)
    plt.show()


# Main program
if __name__ == '__main__':
    dataset = download_data()
    show_data(dataset)
    calculate_returns = calculate_data(dataset)
    show_statistics(calculate_returns)

    portfolio_weights, means, risk = generate_portfolio(calculate_returns)
    show_portfolios(means, risk)
    optimum = portfolio_optimization(portfolio_weights, calculate_returns)
    print_portfolio_optimization(optimum, calculate_returns)
    show_optimal_portfolio(optimum, calculate_returns, means, risk)