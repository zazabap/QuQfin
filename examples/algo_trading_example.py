from quqfin.core.algo_trading.strategies import TradingStrategies
import yfinance as yf
import pandas as pd
import numpy as np

def main():
    # Get Tesla data for trading signals
    tsla = yf.Ticker("TSLA")
    hist = tsla.history(period="1y")
    prices = hist['Close'].values
    
    trader = TradingStrategies(prices)
    
    # Generate trading signals
    ma_signals = trader.moving_average_crossover(short_window=20, long_window=50)
    momentum_signals = trader.momentum_strategy(lookback=20)
    mean_rev_signals = trader.mean_reversion(window=20)
    
    # Create trading summary
    signals_df = pd.DataFrame({
        'Price': prices[-5:],
        'MA Signal': ma_signals[-5:],
        'Momentum': momentum_signals[-5:],
        'Mean Reversion': mean_rev_signals[-5:]
    }, index=hist.index[-5:])
    
    print("TESLA Trading Signals (Last 5 Days)")
    print("-" * 50)
    print(signals_df)
    
    # Calculate strategy returns
    print("\nStrategy Performance:")
    for name, signals in [("MA", ma_signals), ("Momentum", momentum_signals), 
                         ("Mean Reversion", mean_rev_signals)]:
        returns = np.diff(np.log(prices)) * signals[:-1]
        total_return = np.sum(returns)
        print(f"{name} Strategy Return: {total_return:.2%}")

if __name__ == "__main__":
    main()