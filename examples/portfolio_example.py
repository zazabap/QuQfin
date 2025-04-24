from quqfin.core.portfolio.optimization import PortfolioOptimizer
import yfinance as yf
import numpy as np
import pandas as pd

def main():
    # Get data for popular tech stocks
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    data = pd.DataFrame()
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        data[ticker] = hist['Close']
    
    # Calculate returns and covariance
    returns = np.log(data/data.shift(1)).dropna()
    annual_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Portfolio optimization
    optimizer = PortfolioOptimizer(annual_returns.values, cov_matrix.values)
    markowitz_weights = optimizer.markowitz_optimization(target_return=0.15)
    risk_parity_weights = optimizer.risk_parity_optimization()
    
    print("Portfolio Optimization Results:")
    print("\nMarkowitz Portfolio:")
    for ticker, weight in zip(tickers, markowitz_weights):
        print(f"{ticker}: {weight:.2%}")
    
    print("\nRisk Parity Portfolio:")
    for ticker, weight in zip(tickers, risk_parity_weights):
        print(f"{ticker}: {weight:.2%}")

if __name__ == "__main__":
    main()